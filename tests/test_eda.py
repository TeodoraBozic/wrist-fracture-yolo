"""Scientific invariants for read-only ZIP EDA helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import pandas as pd

from wrist_fracture.data.eda import (
    EXPECTED_COLUMNS, link_annotations, load_metadata, read_yolo_zip, validate_yolo_row,
)


class EdaTests(unittest.TestCase):
    def test_geometry_and_class_validation(self):
        self.assertEqual(validate_yolo_row("0 .5 .5 1 1".split(), 1), [])
        cases = {
            "1 .5 .5 .1 .1": "class_id",
            "0.0 .5 .5 .1 .1": "class_id",
            "0 nan .5 .1 .1": "nonfinite_coordinates",
            "0 .5 .5 0 .1": "nonpositive_dimensions",
            "0 .01 .5 .1 .1": "outside_image",
            "0 1.1 .5 .1 .1": "normalized_range",
            "0 .5": "field_count",
        }
        for row, expected in cases.items():
            with self.subTest(row=row):
                self.assertIn(expected, validate_yolo_row(row.split(), 1))

    def test_missing_invalid_empty_and_ambiguous_are_distinct(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotations.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("yolov5/meta.yaml", "nc: 1\nnames: [fracture]\n")
                archive.writestr("yolov5/labels/positive.txt", "0 .5 .5 .2 .3\n")
                archive.writestr("yolov5/labels/empty.txt", "")
                archive.writestr("yolov5/labels/bad.txt", "0 .5 .5 .2 .3\n0 nan 0 0 0")
                archive.writestr("yolov5/labels/a/duplicate.txt", "")
                archive.writestr("yolov5/labels/b/duplicate.txt", "")
            original = path.read_bytes()
            audit = read_yolo_zip(path)
            metadata = pd.DataFrame({"filestem": [
                "positive", "empty", "bad", "missing", "duplicate", "csvdup", "csvdup", None, "bad/key",
            ]})
            linked = link_annotations(metadata, audit).set_index("filestem")
            self.assertEqual(linked.loc["positive", "annotation_status"], "fracture_box")
            self.assertEqual(linked.loc["empty", "annotation_status"], "no_fracture_box")
            for name in ("bad", "missing", "duplicate"):
                self.assertEqual(linked.loc[name, "annotation_status"], "unknown")
            self.assertEqual(len(linked), len(metadata))
            self.assertEqual(linked.annotation_state.tolist(), [
                "valid", "empty", "invalid_annotation", "missing", "ambiguous_annotation",
                "ambiguous_metadata", "ambiguous_metadata", "invalid_identifier", "invalid_identifier",
            ])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_metadata_keeps_leading_zeroes_when_linked(self):
        with TemporaryDirectory() as temporary:
            csv = Path(temporary) / "dataset.csv"
            row = dict.fromkeys(EXPECTED_COLUMNS)
            row.update(filestem="0001", patient_id="0002", study_number="003", timehash="00004")
            pd.DataFrame([row]).to_csv(csv, index=False)
            archive_path = Path(temporary) / "labels.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("yolov5/meta.yaml", "nc: 2\nnames: [other, fracture]\n")
                archive.writestr("yolov5/labels/0001.txt", "0 .5 .5 .1 .2")
            linked = link_annotations(load_metadata(csv), read_yolo_zip(archive_path)).iloc[0]
            self.assertEqual(tuple(linked[["filestem", "patient_id", "study_number", "timehash"]]),
                             ("0001", "0002", "003", "00004"))
            self.assertEqual((linked.annotation_state, linked.annotation_status),
                             ("valid", "no_fracture_box"))


if __name__ == "__main__":
    unittest.main()
