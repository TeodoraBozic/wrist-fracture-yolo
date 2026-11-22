"""Synthetic fixtures test mechanics only; never research results or galleries."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from wrist_fracture.data.image_eda import (
    ImageStore, box_to_pixels, display_limits, index_images, measure_images, pixel_features,
)


class ImageEdaTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.array = np.array([[0, 256, 4096], [65535, 1000, 2000]], dtype=np.uint16)
        self.path = self.root / "example.png"
        Image.fromarray(self.array).save(self.path)

    def test_original_16_bit_values_loose_and_zip(self):
        with ZipFile(self.root / "images_part1.zip", "w") as archive:
            archive.write(self.path, "images/example.png")
        index, issues = index_images([self.root])
        self.assertTrue(issues.empty)
        self.assertEqual(len(index), 2)
        with ImageStore() as store:
            for row in index.to_dict("records"):
                pixels, info = store.decode(row)
                np.testing.assert_array_equal(pixels, self.array)
                self.assertEqual(info["stored_bits"], 16)
                self.assertGreaterEqual(pixels.dtype.itemsize, 2)
                self.assertEqual(int(pixels.max()), 65535)

    def test_yolo_pixel_edges_and_no_clipping(self):
        np.testing.assert_allclose(box_to_pixels(.5, .5, .2, .4, 1000, 500),
                                   (400, 150, 600, 350))
        self.assertEqual(box_to_pixels(.5, .5, 1, 1, 1000, 500), (0, 0, 1000, 500))
        for bounds in [(0, .5, .2, .2), (.5, .5, 0, .2), (float('nan'), .5, .2, .2)]:
            with self.assertRaises(ValueError):
                box_to_pixels(*bounds, 1000, 500)

    def test_statistics_preserve_array_and_use_stored_range(self):
        original = self.array.copy()
        features = pixel_features(self.array, 16)
        self.assertEqual(features["pixel_max"], 65535)
        self.assertAlmostEqual(features["pixel_mean"], self.array.mean())
        self.assertAlmostEqual(features["near_ceiling_fraction"], 1 / 6)
        self.assertAlmostEqual(features["near_zero_fraction"], 2 / 6)
        low, high = display_limits(self.array)
        self.assertLess(low, high)
        np.testing.assert_array_equal(original, self.array)
        with self.assertRaises(ValueError):
            pixel_features(np.zeros((5, 5, 3), dtype=np.uint8), 8)
        signed = np.array([[1, 1000], [256, 32767]], dtype=np.int16)
        self.assertNotIn("mean_fullscale", pixel_features(signed, 16))

    def test_cache_state_changes_and_failures_remain_visible(self):
        (self.root / "broken.png").write_bytes(b"not an image")
        index, _ = index_images([self.root])
        before = self.path.read_bytes()
        cache = self.root / "cache"
        first, issues, state = measure_images(index, limit=None, cache_dir=cache,
                                               context={"release": 2})
        self.assertEqual(len(first), 1)
        self.assertEqual(len(issues), 1)
        second, _, same = measure_images(index, limit=None, cache_dir=cache,
                                         context={"release": 2})
        self.assertEqual(same, state)
        self.assertEqual(first.to_dict("records"), second.to_dict("records"))
        _, _, changed = measure_images(index, limit=None, cache_dir=cache,
                                       context={"release": 3})
        self.assertNotEqual(changed["signature"], state["signature"])
        Image.fromarray(self.array).save(self.root / "broken.png")
        updated_index, _ = index_images([self.root])
        updated, updated_issues, updated_state = measure_images(
            updated_index, limit=None, cache_dir=cache, context={"release": 2})
        self.assertNotEqual(updated_state["signature"], state["signature"])
        self.assertEqual(len(updated), 2)
        self.assertTrue(updated_issues.empty)
        self.assertTrue((cache / state["signature"] / "000000.json").is_file())
        self.assertEqual(before, self.path.read_bytes())

    def test_empty_inventory(self):
        index, issues = index_images([self.root / "absent"])
        self.assertTrue(index.empty)
        self.assertEqual(len(issues), 1)

    def test_16_bit_tiff_and_8_bit_png(self):
        Image.fromarray(self.array).save(self.root / "tiff.tiff")
        eight = np.array([[0, 255], [34, 129]], dtype=np.uint8)
        Image.fromarray(eight).save(self.root / "eight.png")
        index, _ = index_images([self.root])
        with ImageStore() as store:
            for name, expected, bits in [("tiff", self.array, 16), ("eight", eight, 8)]:
                row = index.loc[index.filestem.eq(name)].iloc[0].to_dict()
                pixels, info = store.decode(row)
                np.testing.assert_array_equal(pixels, expected)
                self.assertEqual(info["stored_bits"], bits)

    def test_overlay_artist_matches_pixel_edges(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from wrist_fracture.visualization.image_eda import image_gallery

        index, _ = index_images([self.root])
        boxes = pd.DataFrame([{
            "filestem": "example", "valid": True, "class_id": 3,
            "class_name": "fracture", "cx": .5, "cy": .5,
            "width": 1.0, "height": 1.0,
        }])
        fig = image_gallery(index, boxes, {3: "fracture"}, title="Test fixture",
                            overlay="fracture", columns=1)
        try:
            rectangle = fig.axes[0].patches[0]
            self.assertEqual(rectangle.get_xy(), (0, 0))
            self.assertEqual(rectangle.get_width(), 3)
            self.assertEqual(rectangle.get_height(), 2)
            self.assertEqual(fig.axes[0].images[0].get_extent(), [0, 3, 2, 0])
        finally:
            plt.close(fig)

    def test_gallery_pages_keep_every_selected_example(self):
        from wrist_fracture.visualization import image_eda as visualization

        selected = pd.DataFrame({"image_key": range(17)})
        with (patch.object(visualization, "image_gallery", return_value=None) as gallery,
              patch.object(visualization.plt, "show"),
              patch.object(visualization.plt, "close")):
            visualization.show_gallery_pages(selected, pd.DataFrame(), {}, title="Primeri")
        pages = [list(call.args[0].image_key) for call in gallery.call_args_list]
        self.assertEqual([key for page in pages for key in page], list(range(17)))


if __name__ == "__main__":
    unittest.main()
