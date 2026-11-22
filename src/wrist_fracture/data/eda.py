"""Read-only metadata and ZIP annotation loading for reproducible EDA.

Unlike the image inspector, these tools require no decoded images, retain
normalized YOLO coordinates and report every invalid row without aborting an audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path, PurePosixPath
import re
from zipfile import BadZipFile, ZipFile

import pandas as pd
import yaml

EXPECTED_COLUMNS = (
    "filestem", "patient_id", "study_number", "timehash", "gender", "age",
    "laterality", "projection", "initial_exam", "ao_classification", "cast",
    "diagnosis_uncertain", "osteopenia", "fracture_visible", "metal",
    "pixel_spacing", "device_manufacturer",
)
BOX_COLUMNS = ["filestem", "member", "line", "class_id", "class_name",
               "cx", "cy", "width", "height", "area", "valid"]


def source_fingerprint(path: Path) -> dict[str, str | int]:
    """Identify an input by size and SHA-256 without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"file": path.name, "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def load_metadata(path: Path) -> pd.DataFrame:
    """Keep identifiers as strings (including leading zeros) and missing as NA."""
    frame = pd.read_csv(path, dtype={
        "filestem": "string", "patient_id": "string", "study_number": "string",
        "timehash": "string",
    })
    missing = set(EXPECTED_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")
    return frame


def missing_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Report all columns with a common row-count denominator."""
    counts = frame.isna().sum()
    return pd.DataFrame({"missing": counts, "percent": counts / len(frame) * 100
                         if len(frame) else float("nan")}).sort_values(
                             "missing", ascending=False)


def validate_yolo_row(fields: list[str], class_count: int) -> list[str]:
    """Return row issues; allow 1e-6 boundary tolerance for export rounding."""
    if len(fields) != 5:
        return ["field_count"]
    issues = []
    if not re.fullmatch(r"\d+", fields[0]) or not 0 <= int(fields[0]) < class_count:
        issues.append("class_id")
    try:
        cx, cy, width, height = map(float, fields[1:])
    except ValueError:
        return issues + ["nonnumeric_coordinates"]
    if not all(math.isfinite(v) for v in (cx, cy, width, height)):
        return issues + ["nonfinite_coordinates"]
    if any(v < 0 or v > 1 for v in (cx, cy, width, height)):
        issues.append("normalized_range")
    if width <= 0 or height <= 0:
        issues.append("nonpositive_dimensions")
    tolerance = 1e-6
    if (cx - width / 2 < -tolerance or cy - height / 2 < -tolerance
            or cx + width / 2 > 1 + tolerance or cy + height / 2 > 1 + tolerance):
        issues.append("outside_image")
    return issues


@dataclass
class AnnotationAudit:
    """Parsed rows, per-file completeness and issue-level diagnostics."""

    classes: dict[int, str]
    files: pd.DataFrame
    boxes: pd.DataFrame
    issues: pd.DataFrame


def read_yolo_zip(path: Path) -> AnnotationAudit:
    """Read the original YOLO export in place; never extract archive members."""
    files, boxes, issues = [], [], []
    with ZipFile(path) as archive:
        config = yaml.safe_load(archive.read("yolov5/meta.yaml"))
        names = config["names"]
        classes = (dict(enumerate(names)) if isinstance(names, list)
                   else {int(key): value for key, value in names.items()})
        if (set(classes) != set(range(len(classes)))
                or config.get("nc") != len(classes)
                or len(set(classes.values())) != len(classes)
                or "fracture" not in classes.values()):
            raise ValueError("Invalid or unsupported YOLO class configuration")
        members = sorted((info for info in archive.infolist()
                          if info.filename.startswith("yolov5/labels/")
                          and info.filename.endswith(".txt") and not info.is_dir()),
                         key=lambda info: info.filename)
        for info in members:
            member = info.filename
            stem = PurePosixPath(member).stem
            n_rows, n_invalid, n_valid, n_fracture = 0, 0, 0, 0
            try:
                text = archive.read(info).decode("utf-8-sig")
            except (UnicodeError, OSError, BadZipFile) as error:
                issues.append({"filestem": stem, "member": member,
                               "line": 0, "issue": "unreadable", "raw": str(error)})
                files.append({"filestem": stem, "member": member, "rows": 0,
                              "invalid_rows": 0, "valid_boxes": 0,
                              "fracture_boxes": 0, "parse_ok": False})
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                n_rows += 1
                fields = line.split()
                row_issues = validate_yolo_row(fields, len(classes))
                for issue in row_issues:
                    issues.append({"filestem": stem, "member": member,
                                   "line": line_number, "issue": issue, "raw": line})
                if row_issues:
                    n_invalid += 1
                if len(fields) != 5:
                    continue
                try:
                    class_id = int(fields[0])
                    cx, cy, width, height = map(float, fields[1:])
                except ValueError:
                    continue
                valid = not row_issues
                name = classes.get(class_id)
                n_valid += int(valid)
                n_fracture += int(valid and name == "fracture")
                boxes.append([stem, member, line_number, class_id, name,
                              cx, cy, width, height, width * height, valid])
            files.append({"filestem": stem, "member": member, "rows": n_rows,
                          "invalid_rows": n_invalid, "valid_boxes": n_valid,
                          "fracture_boxes": n_fracture, "parse_ok": n_invalid == 0})
    return AnnotationAudit(
        classes, pd.DataFrame(files, columns=["filestem", "member", "rows",
                                             "invalid_rows", "valid_boxes",
                                             "fracture_boxes", "parse_ok"]),
        pd.DataFrame(boxes, columns=BOX_COLUMNS),
        pd.DataFrame(issues, columns=["filestem", "member", "line", "issue", "raw"]),
    )


def link_annotations(metadata: pd.DataFrame, audit: AnnotationAudit) -> pd.DataFrame:
    """Retain every metadata row and link only unambiguous identifiers."""
    keys = metadata.filestem.astype("string")
    invalid = ~keys.str.fullmatch(r"[^\s/\\]+", na=False)
    duplicated = keys.duplicated(keep=False) & ~invalid
    ambiguous = audit.files.loc[audit.files.filestem.duplicated(keep=False), "filestem"]
    files = audit.files.loc[~audit.files.filestem.duplicated(keep=False)]
    linked = metadata.assign(filestem=keys.mask(invalid | duplicated)).merge(
        files, on="filestem", how="left", validate="many_to_one")
    linked["filestem"] = keys.to_numpy()
    linked["annotation_state"] = "missing"
    linked.loc[keys.isin(ambiguous).to_numpy(), "annotation_state"] = "ambiguous_annotation"
    linked.loc[duplicated.to_numpy(), "annotation_state"] = "ambiguous_metadata"
    linked.loc[invalid.to_numpy(), "annotation_state"] = "invalid_identifier"
    linked.loc[linked.parse_ok.eq(False), "annotation_state"] = "invalid_annotation"
    linked["annotation_status"] = "unknown"
    ok = linked.parse_ok.eq(True)
    linked.loc[ok & linked.rows.eq(0), "annotation_state"] = "empty"
    linked.loc[ok & linked.rows.gt(0), "annotation_state"] = "valid"
    linked.loc[ok & linked.fracture_boxes.gt(0), "annotation_status"] = "fracture_box"
    linked.loc[ok & linked.fracture_boxes.eq(0), "annotation_status"] = "no_fracture_box"
    return linked
