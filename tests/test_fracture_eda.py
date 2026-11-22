"""Regression checks for sampling, resumable checkpoints and local contrast."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from wrist_fracture.data.eda_cache import measure_batches, signature
from wrist_fracture.data.eda_settings import stratified_examples
from wrist_fracture.data.fracture_eda import fracture_geometry, local_contrast, measure_fractures
from wrist_fracture.data.image_eda import index_images


class FractureEdaTests(unittest.TestCase):
    def test_contrast_uses_original_values_and_excludes_box_from_ring(self):
        array = np.full((4, 4), 1000, dtype=np.uint16)
        array[1:3, 1:3] = [[50000, 51000], [52000, 53000]]
        original = array.copy()
        result = local_contrast(array, (1, 1, 3, 3))
        self.assertEqual((result["inside_n"], result["ring_n"]), (4, 12))
        self.assertAlmostEqual(result["inside_p95_p05_dn"], 2700)
        self.assertEqual(result["ring_p95_p05_dn"], 0)
        self.assertEqual(result["abs_median_gap_dn"], 50500)
        self.assertEqual(result["contrast_status"], "izmereno")
        np.testing.assert_array_equal(array, original)
        full = local_contrast(array, (0, 0, 4, 4))
        self.assertEqual(full["ring_n"], 0)
        self.assertNotIn("ring_p95_p05_dn", full)
        edge = local_contrast(array, (0, 0, 1.5, 2))
        self.assertEqual((edge["inside_n"], edge["ring_n"]), (4, 5))

    def test_box_outcomes_include_decode_failures_and_missing_images(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            array = np.arange(16, dtype=np.uint16).reshape(4, 4)
            Image.fromarray(array).save(root / "good.png")
            (root / "broken.png").write_bytes(b"broken")
            images, _ = index_images([root])
            boxes = pd.DataFrame([
                {"filestem": stem, "member": f"{stem}.txt", "line": 1, "class_name": "fracture",
                 "valid": True, "cx": .5, "cy": .5, "width": .5, "height": .5, "area": .25}
                for stem in ["good", "broken", "not_available"]
            ])
            outcomes, state = measure_fractures(images, fracture_geometry(boxes), cache_dir=root / "cache")
            repeated, same = measure_fractures(images, fracture_geometry(boxes), cache_dir=root / "cache")
            pd.testing.assert_frame_equal(outcomes, repeated)
            self.assertEqual(state, same)
            result = outcomes.set_index("filestem")
            self.assertEqual(result.loc["good", "inside_n"], 4)
            self.assertIn("cannot identify image", result.loc["broken", "contrast_status"])
            self.assertEqual(result.loc["not_available", "contrast_status"], "slika nema jednoznačnu vezu")

    def test_sample_is_stable_and_keeps_rare_and_missing_strata(self):
        rows = pd.DataFrame({"image_key": [f"img-{i}" for i in range(8)],
                             "projection": ["1"] * 6 + ["3", None]})
        selected = stratified_examples(rows, ["projection"])
        pd.testing.assert_frame_equal(selected, stratified_examples(rows.iloc[::-1], ["projection"]))
        self.assertEqual(len(selected), 3)
        self.assertTrue({"img-6", "img-7"} <= set(selected.image_key))

    def test_checkpoints_resume_and_recompute_only_incomplete_batches(self):
        keys = list("abcd")
        state = {"signature": signature({"version": 1})}
        with TemporaryDirectory() as directory:
            calls = []

            def measure(key):
                calls.append(key)
                return {"image_key": key, "value": 1}

            def results():
                return measure_batches(keys, measure, directory=directory, state=state,
                                       valid=lambda row: row.get("value") == 1, batch_size=2)

            with self.assertRaises(RuntimeError):
                for row in results():
                    if row["image_key"] == "b":
                        raise RuntimeError("interrupted")
            self.assertEqual(calls, ["a", "b"])
            calls.clear()
            self.assertEqual([row["image_key"] for row in results()], keys)
            self.assertEqual(calls, ["c", "d"])
            path = Path(directory) / state["signature"] / "000002.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["records"].pop()
            payload["sha256"] = signature(payload["records"])
            path.write_text(json.dumps(payload), encoding="utf-8")
            calls.clear()
            self.assertEqual([row["image_key"] for row in results()], keys)
            self.assertEqual(calls, ["c", "d"])
