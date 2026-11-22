"""Read-only, bounded-memory image analysis for loose files and ZIP archives.

No resize, channel conversion, orientation correction or intensity scaling is
performed by the decoder. Display scaling is separate from measurement.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
import warnings
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from .eda import validate_yolo_row
from .eda_cache import measure_batches, signature
from .inspect_dataset import IMAGE_EXTENSIONS

LOGGER = logging.getLogger(__name__)
CACHE_SCHEMA = 2
INDEX_COLUMNS = ["image_key", "filestem", "source", "member", "extension",
                 "bytes", "mtime_ns", "crc"]


def index_images(roots: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Index supported image files and official image ZIPs without extracting.

    A root may be a directory, an explicit ZIP, or an individual image. Directory
    scanning considers only images_part*.zip, so annotation ZIPs are not decoded.
    Duplicate stems remain visible and are excluded later from ambiguous joins.
    """
    candidates: set[Path] = set()
    rows, issues = [], []
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            issues.append({"source": str(root), "issue": "source_missing"})
        elif root.is_file():
            candidates.add(root)
        else:
            candidates.update(p.resolve() for p in root.rglob("*") if p.is_file()
                              and (p.suffix.lower() in IMAGE_EXTENSIONS
                                   or (p.name.startswith("images_part")
                                       and p.suffix.lower() == ".zip")))
    for path in sorted(candidates):
        stat = path.stat()
        if path.suffix.lower() == ".zip":
            try:
                with ZipFile(path) as archive:
                    seen = set()
                    for info in archive.infolist():
                        member = PurePosixPath(info.filename)
                        if info.is_dir() or member.suffix.lower() not in IMAGE_EXTENSIONS:
                            continue
                        if info.filename in seen:
                            raise ValueError("Duplicate ZIP member names; archive excluded")
                        seen.add(info.filename)
                    for name in sorted(seen):
                        info = archive.getinfo(name)
                        member = PurePosixPath(name)
                        rows.append([f"{path}::{name}", member.stem, str(path), name,
                                     member.suffix.lower(), info.file_size,
                                     stat.st_mtime_ns, info.CRC])
            except (OSError, BadZipFile, ValueError) as error:
                issues.append({"source": str(path), "issue": str(error)})
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append([str(path), path.stem, str(path), "", path.suffix.lower(),
                         stat.st_size, stat.st_mtime_ns, None])
    return (pd.DataFrame(rows, columns=INDEX_COLUMNS),
            pd.DataFrame(issues, columns=["source", "issue"]))


class ImageStore:
    """Reuse a small number of archive handles, never an in-RAM image corpus."""

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._archives: dict[str, ZipFile] = {}

    def __enter__(self) -> ImageStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self._stack.close()

    @contextmanager
    def open(self, row: dict[str, Any]) -> Iterator[Any]:
        """Yield a seekable image stream; ZIP bytes are not extracted to disk."""
        source, member = str(row["source"]), str(row["member"])
        if member:
            if source not in self._archives:
                self._archives[source] = self._stack.enter_context(ZipFile(source))
            with self._archives[source].open(member) as stream:
                yield stream
        else:
            with Path(source).open("rb") as stream:
                yield stream

    def decode(self, row: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Fully decode the first frame without reducing its original precision.

        Unsupported color/palette arrays remain unconverted and are flagged.
        PNG bit depth comes from IHDR, not from NumPy dtype or PIL mode.
        """
        with self.open(row) as stream:
            header = stream.read(29)
            stream.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(stream) as image:
                    bits: int | str = "unknown"
                    unsigned_source = False
                    if image.format == "PNG" and header[:8] == b"\x89PNG\r\n\x1a\n":
                        bits = int(header[24])
                        unsigned_source = True
                    elif image.format == "TIFF":
                        value = image.tag_v2.get(258)
                        sample_format = image.tag_v2.get(339, (1,))
                        unsigned_source = (sample_format == 1 or sample_format == (1,))
                        if isinstance(value, int):
                            bits = value
                        elif value and len(set(value)) == 1:
                            bits = int(value[0])
                    elif image.format in {"JPEG", "BMP"} and image.mode in {"L", "RGB"}:
                        bits = 8
                        unsigned_source = True
                    image.load()
                    array = np.array(image)
                    info = {"width_px": image.width, "height_px": image.height,
                            "format": image.format, "mode": image.mode,
                            "stored_bits": bits, "dtype": str(array.dtype),
                            "unsigned_source": unsigned_source,
                            "channels": 1 if array.ndim == 2 else array.shape[-1],
                            "frames": getattr(image, "n_frames", 1)}
        return array, info

    def byte_hash(self, row: dict[str, Any]) -> str:
        """SHA-256 of original encoded bytes, streaming one file at a time."""
        digest = hashlib.sha256()
        with self.open(row) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()


def box_to_pixels(
    cx: float, cy: float, width: float, height: float,
    image_width: int, image_height: int,
) -> tuple[float, float, float, float]:
    """Convert YOLO center/extent to continuous x0,y0,x1,y1 pixel edges.

    No clipping or integer rounding: boundary violations must remain observable.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    issues = validate_yolo_row(["0", *(str(v) for v in (cx, cy, width, height))], 1)
    if issues:
        raise ValueError(f"Invalid normalized box: {issues}")
    return ((cx - width / 2) * image_width, (cy - height / 2) * image_height,
            (cx + width / 2) * image_width, (cy + height / 2) * image_height)


def pixel_features(
    array: np.ndarray, stored_bits: int | str, *, unsigned_source: bool | None = None,
    perceptual_hash: bool = True,
) -> dict[str, Any]:
    """Compute raw grayscale statistics with one image in memory.

    Near-endpoint fractions use 1% of the representable unsigned stored range,
    not 1% of the observed image range. Palette/color/float data are not silently
    treated as medical grayscale. Laplacian variance is a descriptive proxy only.
    """
    if array.ndim != 2 or array.dtype.kind not in "ui":
        raise ValueError("Pixel metrics require an unconverted integer grayscale array")
    if array.size == 0:
        raise ValueError("Empty pixel array")
    values = array.astype(np.float64)
    minimum, maximum = float(values.min()), float(values.max())
    p01, p05, median, p95, p99 = np.percentile(array, [1, 5, 50, 95, 99])
    result = {"pixel_min": minimum, "pixel_max": maximum,
              "pixel_mean": float(values.mean()), "pixel_median": float(median),
              "pixel_std": float(values.std()), "p01": float(p01),
              "p05": float(p05), "p95": float(p95), "p99": float(p99),
              "observed_range": maximum - minimum,
              "contrast_p99_p01": float(p99 - p01)}
    is_unsigned = array.dtype.kind == "u" if unsigned_source is None else unsigned_source
    unsigned_range = (is_unsigned and isinstance(stored_bits, int) and 0 < stored_bits <= 16
                      and minimum >= 0 and maximum <= 2 ** stored_bits - 1)
    if unsigned_range:
        ceiling = float(2 ** stored_bits - 1)
        normalized = values / ceiling
        result.update({"mean_fullscale": float(normalized.mean()),
                       "contrast_fullscale": float((p99 - p01) / ceiling),
                       "range_fullscale": (maximum - minimum) / ceiling,
                       "near_zero_fraction": float((values <= .01 * ceiling).mean()),
                       "near_ceiling_fraction": float((values >= .99 * ceiling).mean())})
        if min(values.shape) >= 3:
            laplacian = (normalized[:-2, 1:-1] + normalized[2:, 1:-1]
                         + normalized[1:-1, :-2] + normalized[1:-1, 2:]
                         - 4 * normalized[1:-1, 1:-1])
            result["laplacian_variance"] = float(laplacian.var())
    # A 64-bit average hash: coarse block means, no permanent image conversion.
    if perceptual_hash and min(values.shape) >= 8:
        block_means = np.array([block.mean() for strip in np.array_split(values, 8, axis=0)
                                for block in np.array_split(strip, 8, axis=1)])
        result["average_hash"] = np.packbits(block_means >= block_means.mean()).tobytes().hex()
    return result


def display_limits(array: np.ndarray) -> tuple[float, float]:
    """Return per-image P1/P99 display limits without changing source pixels."""
    lower, upper = np.percentile(array, [1, 99])
    if upper <= lower:
        lower, upper = float(array.min()), float(array.max())
    return float(lower), float(upper if upper > lower else lower + 1)


DECODE_ERRORS = (OSError, ValueError, BadZipFile, Image.DecompressionBombError,
                 Image.DecompressionBombWarning)
ISSUE_COLUMNS = ["image_key", "filestem", "issue"]
BASE_METRICS = ["image_key", "filestem", "width_px", "height_px", "format", "mode",
                "stored_bits", "dtype", "unsigned_source", "channels", "frames", "aspect_ratio"]


def measurement_state(ordered: pd.DataFrame, **settings: Any) -> dict[str, Any]:
    """Fingerprint the ordered inputs, decoder implementation and dependencies."""
    state = {"schema": CACHE_SCHEMA, "indexed": len(ordered),
             "index_sha256": hashlib.sha256(ordered[INDEX_COLUMNS].to_json(
                 orient="records").encode()).hexdigest(),
             "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "software": {"numpy": np.__version__, "pillow": Image.__version__,
                          "pandas": pd.__version__}, **settings}
    return {**state, "signature": signature(state)}


def _measure_image(store: ImageStore, row: dict[str, Any], hashes: bool) -> dict[str, Any]:
    identity = {"image_key": row["image_key"], "filestem": row["filestem"]}
    issues = []

    def add_issue(message):
        issues.append({**identity, "issue": message})

    result = None
    try:
        array, info = store.decode(row)
        if info["stored_bits"] == 16 and array.dtype.itemsize < 2:
            raise ValueError("16-bit source decoded with less than 16-bit capacity")
        result = {**identity, **info, "aspect_ratio": info["width_px"] / info["height_px"]}
        if info["channels"] == 1 and info["mode"] not in {"P", "1"}:
            try:
                result.update(pixel_features(array, info["stored_bits"],
                                             unsigned_source=info["unsigned_source"],
                                             perceptual_hash=hashes))
            except ValueError as error:
                add_issue(str(error))
        else:
            add_issue("Non-grayscale/palette: pixel metrics skipped")
        if info["frames"] > 1:
            add_issue("Multiple frames: first frame measured only")
    except DECODE_ERRORS as error:
        add_issue(f"Decode/measurement failure: {error}")
    if hashes and result is not None:
        try:
            result["file_sha256"] = store.byte_hash(row)
        except DECODE_ERRORS as error:
            add_issue(f"Encoded hash failure: {error}")
    return {"image_key": row["image_key"], "metrics": result, "issues": issues}


def measure_images(
    images: pd.DataFrame, *, limit: int | None = 256, seed: int = 42,
    cache_dir: Path | None = None, context: dict[str, Any] | None = None,
    hashes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Measure one image at a time and checkpoint complete batches."""
    if limit is not None and limit < 1:
        raise ValueError("Sample limit must be positive or None")
    ordered = images.sort_values("image_key").reset_index(drop=True)
    selected = (ordered.sample(n=min(limit, len(ordered)), random_state=seed)
                if limit is not None else ordered).sort_values("image_key")
    rows = {row["image_key"]: row for row in selected.to_dict("records")}
    if len(rows) != len(selected):
        raise ValueError("Image keys must be unique")
    state = measurement_state(ordered, seed=seed, limit=limit, hashes=hashes,
                              context=context or {}, selected=len(rows))

    def valid(outcome):
        record, issues = outcome["metrics"], outcome["issues"]
        key = outcome["image_key"]
        if not isinstance(issues, list) or not all(
            set(ISSUE_COLUMNS) <= issue.keys() and issue["image_key"] == key
            for issue in issues
        ):
            return False
        if record is None:
            return bool(issues)
        return (set(BASE_METRICS) <= record.keys() and record["image_key"] == key
                and record["filestem"] == rows[key]["filestem"])

    def progress(done, total):
        if done % 1024 == 0 or done == total:
            LOGGER.info("Images completed: %d/%d", done, total)

    with ImageStore() as store:
        outcomes = list(measure_batches(
            list(rows), lambda key: _measure_image(store, rows[key], hashes),
            directory=cache_dir, state=state, valid=valid, progress=progress))
    metrics = [row["metrics"] for row in outcomes if row["metrics"] is not None]
    issues = [issue for row in outcomes for issue in row["issues"]]
    return (pd.DataFrame(metrics) if metrics else pd.DataFrame(columns=BASE_METRICS),
            pd.DataFrame(issues, columns=ISSUE_COLUMNS), state)


def similar_hash_pairs(metrics: pd.DataFrame, max_distance: int = 4) -> pd.DataFrame:
    """Find average-hash candidates; cap at 1,000 to avoid quadratic full scans."""
    if len(metrics) > 1000:
        raise ValueError("Use a deterministic subset of at most 1,000 images")
    pairs = []
    if "average_hash" not in metrics:
        return pd.DataFrame(columns=["left", "right", "distance", "identical_bytes"])
    rows = metrics.dropna(subset=["average_hash"]).to_dict("records")
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            distance = (int(left["average_hash"], 16) ^ int(right["average_hash"], 16)).bit_count()
            if distance <= max_distance:
                pairs.append({"left": left["filestem"], "right": right["filestem"],
                              "distance": distance,
                              "identical_bytes": left["file_sha256"] == right["file_sha256"]})
    return pd.DataFrame(pairs, columns=["left", "right", "distance", "identical_bytes"])
