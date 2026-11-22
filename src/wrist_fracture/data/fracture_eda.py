"""Geometry and local intensity measurements on original fracture annotations."""

from hashlib import sha256
from math import ceil, floor
from pathlib import Path

import numpy as np
import pandas as pd

from .eda_cache import measure_batches
from .image_eda import DECODE_ERRORS, ImageStore, box_to_pixels, measurement_state

BOX_KEY = ["filestem", "member", "line"]
CONTRAST_COLUMNS = ["inside_n", "ring_n", "inside_p95_p05_dn", "ring_p95_p05_dn",
                    "abs_median_gap_dn", "contrast_status"]


def fracture_geometry(boxes):
    """Add normalized geometry for all valid fracture boxes."""
    selected = boxes.loc[boxes.valid & boxes.class_name.eq("fracture")].copy()
    selected["normalized_aspect"] = selected.width / selected.height
    selected["edge_distance"] = np.minimum.reduce([
        selected.cx - selected.width / 2, selected.cy - selected.height / 2,
        1 - selected.cx - selected.width / 2, 1 - selected.cy - selected.height / 2,
    ]).clip(min=0)
    return selected


def local_contrast(array, edges):
    """Measure P95–P5 in the box and a surrounding ring on original integer DN.

    Floor/ceil edges select intersected pixels. The outer rectangle adds 25% of
    the box extent on each side, at least one pixel, and is clipped to the image.
    """
    if array.ndim != 2 or array.dtype.kind not in "ui":
        raise ValueError("Očekivana je originalna integer grayscale slika")
    height, width = array.shape
    x0, y0, x1, y1 = edges
    left, top = max(0, floor(x0)), max(0, floor(y0))
    right, bottom = min(width, ceil(x1)), min(height, ceil(y1))
    inside = array[top:bottom, left:right]
    if not inside.size:
        return {"inside_n": 0, "ring_n": 0, "contrast_status": "prazan box"}
    dx, dy = max(1, ceil((x1 - x0) / 4)), max(1, ceil((y1 - y0) / 4))
    outer_l, outer_t = max(0, left - dx), max(0, top - dy)
    outer_r, outer_b = min(width, right + dx), min(height, bottom + dy)
    ring = np.concatenate([
        array[outer_t:top, outer_l:outer_r].ravel(),
        array[bottom:outer_b, outer_l:outer_r].ravel(),
        array[top:bottom, outer_l:left].ravel(),
        array[top:bottom, right:outer_r].ravel(),
    ])
    low, median, high = np.percentile(inside, [5, 50, 95])
    result = {"inside_n": inside.size, "ring_n": ring.size,
              "inside_p95_p05_dn": float(high - low), "contrast_status": "bez dostupnog prstena"}
    if ring.size:
        ring_low, ring_median, ring_high = np.percentile(ring, [5, 50, 95])
        result.update(ring_p95_p05_dn=float(ring_high - ring_low),
                      abs_median_gap_dn=float(abs(median - ring_median)), contrast_status="izmereno")
    return result


def measure_fractures(images, boxes, *, cache_dir=None):
    """Decode each linked image once and keep an outcome for each box."""
    ordered = images.loc[images.filestem.isin(boxes.filestem)].sort_values("image_key")
    groups = {stem: group.sort_values(BOX_KEY).to_dict("records")
              for stem, group in boxes.groupby("filestem")}
    rows = {row["image_key"]: row for row in ordered.to_dict("records")}
    state = measurement_state(
        ordered, stage="fracture_contrast",
        boxes_sha256=sha256(boxes[BOX_KEY + ["cx", "cy", "width", "height"]]
                           .sort_values(BOX_KEY).to_json(orient="records", double_precision=15)
                           .encode()).hexdigest(),
        contrast_code=sha256(Path(__file__).read_bytes()).hexdigest())

    def measure(key):
        row = rows[key]
        try:
            array, info = store.decode(row)
            if (info["channels"] != 1 or info["mode"] in {"P", "1"}
                    or (info["stored_bits"] == 16 and array.dtype.itemsize < 2)):
                raise ValueError("nepodržan grayscale format ili dubina")
        except DECODE_ERRORS as error:
            return {"image_key": key, "boxes": [], "issue": str(error)}
        results = []
        for box in groups[row["filestem"]]:
            identity = {name: box[name] for name in BOX_KEY}
            edges = box_to_pixels(box["cx"], box["cy"], box["width"], box["height"],
                                  info["width_px"], info["height_px"])
            try:
                result = local_contrast(array, edges)
            except ValueError as error:
                result = {"contrast_status": str(error)}
            results.append({**identity, **result})
        return {"image_key": key, "boxes": results, "issue": None}

    def valid(record):
        expected = groups[rows[record["image_key"]]["filestem"]]
        if record["issue"]:
            return not record["boxes"]
        return (len(record["boxes"]) == len(expected)
                and all("contrast_status" in row and all(row[key] == box[key] for key in BOX_KEY)
                        for row, box in zip(record["boxes"], expected)))

    with ImageStore() as store:
        outcomes = list(measure_batches(list(rows), measure, directory=cache_dir, state=state, valid=valid))
    measured = pd.DataFrame([box for row in outcomes for box in row["boxes"]],
                            columns=BOX_KEY + CONTRAST_COLUMNS)
    result = boxes[BOX_KEY].merge(measured, on=BOX_KEY, how="left", validate="one_to_one")
    failures = {rows[row["image_key"]]["filestem"]: row["issue"]
                for row in outcomes if row["issue"]}
    result["contrast_status"] = (result.contrast_status.fillna(result.filestem.map(failures))
                                 .fillna("slika nema jednoznačnu vezu"))
    return result, state
