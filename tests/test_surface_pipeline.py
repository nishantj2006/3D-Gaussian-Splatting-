"""Unit checks for code-driven support placement and speculative fill helpers."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from gsedit.reconstruction.background_preview import build, clone_surface
from gsedit.pipelines.surface_pipeline import color_mask, photo_region, project_mask, run, top_anchor


class SurfacePipelineTests(unittest.TestCase):
    def test_blue_region_and_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[20:70, 10:80] = (40, 60, 140)
            path = Path(folder) / "view.png"
            Image.fromarray(image).save(path)
            mask, fraction = photo_region(path, "blue", width=100)
            self.assertGreater(fraction, 0.30)
            self.assertTrue(mask[40, 40])
            camera = {"position": [0, 0, 0], "rotation": np.eye(3).tolist(),
                      "width": 100, "height": 100, "fx": 100, "fy": 100}
            points = np.array([[-0.1, -0.1, 1], [0.45, 0.45, 1], [0, 0, -1]])
            np.testing.assert_array_equal(project_mask(points, camera, mask),
                                          [True, False, False])

    def test_top_anchor_uses_upper_support_and_rejects_ambiguous_column(self):
        rng = np.random.default_rng(4)
        floor = np.column_stack((rng.normal(0, .2, 500), rng.normal(0, .2, 500),
                                 rng.normal(0, .01, 500)))
        top = np.column_stack((rng.normal(0, .2, 500), rng.normal(0, .2, 500),
                               rng.normal(1.0, .01, 500)))
        anchor, details = top_anchor(np.concatenate((floor, top)), (np.zeros(3), np.array([0, 0, 1])))
        self.assertAlmostEqual(anchor[2], 1, delta=.03)
        self.assertEqual(details["support_points"], 200)
        column = np.column_stack((np.zeros(500), np.zeros(500), np.linspace(0, 5, 500)))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            top_anchor(column, (np.zeros(3), np.array([0, 0, 1])))

    def test_fill_clones_preserve_schema_and_semantics(self):
        dtype = np.dtype([(name, "<f4") for name in
                          ("x", "y", "z", "scale_0", "scale_1", "scale_2",
                           "rot_0", "rot_1", "rot_2", "rot_3", "opacity",
                           "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0",
                           "semantic_0", "object_id")])
        donor = np.zeros(2, dtype=dtype)
        donor["semantic_0"] = [0.4, 0.8]
        donor["object_id"] = 9
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        a = clone_surface(donor, [1, 0], points, .04)
        b = clone_surface(donor, [1, 0], points, .04)
        self.assertEqual(a.dtype, donor.dtype)
        self.assertEqual(a.tobytes(), b.tobytes())
        np.testing.assert_allclose(a["semantic_0"], [.8, .4])
        np.testing.assert_allclose(a["object_id"], 0)
        np.testing.assert_allclose(a["z"], [3, 6])
        self.assertTrue(np.isfinite(np.column_stack([a[n] for n in a.dtype.names])).all())
    def test_unsafe_background_and_overwrite_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "allow-extrapolation"):
            build(SimpleNamespace(allow_extrapolation=False))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run(SimpleNamespace(output_dir=folder))
            with self.assertRaises(FileExistsError):
                build(SimpleNamespace(allow_extrapolation=True, output_dir=folder,
                                      floor_step=.04, wall_step=.08, render_width=540))



if __name__ == "__main__":
    unittest.main()
