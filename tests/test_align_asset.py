"""Geometry checks for automatic Gaussian asset placement."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from align_asset import (fit_transform, plane_frame, target_from_removal,
                         transform_gaussians)


class AlignAssetTests(unittest.TestCase):
    def test_tilted_plane_bbox_fit_rotates_centers_and_covariances(self):
        origin, frame = plane_frame([1, 2, 3], [0.1, -0.6, 0.8])
        coordinates = np.asarray([(x, y, z) for x in (-0.25, 0.25)
                                  for y in (-0.5, 0.5) for z in (0, 1)])
        rotation, scale, source, world, details = fit_transform(
            coordinates, origin, frame, [0.2, -0.4], [2, 1, 1.5],
            padding=0, clearance=0.005)
        self.assertEqual(details["yaw_degrees"], 90)
        self.assertAlmostEqual(scale, 1.5)
        dtype = np.dtype([(name, "<f4") for name in
                          ("x", "y", "z", "scale_0", "scale_1", "scale_2",
                           "rot_0", "rot_1", "rot_2", "rot_3",
                           "f_rest_0", "semantic_0")])
        asset = np.zeros(len(coordinates), dtype=dtype)
        for i, key in enumerate(("x", "y", "z")):
            asset[key] = coordinates[:, i]
        asset["rot_0"] = 1
        asset["semantic_0"] = 0.7
        transformed = transform_gaussians(asset, rotation, scale, source, world)
        points = np.column_stack([transformed[k] for k in ("x", "y", "z")])
        height = (points - origin) @ frame[:, 2]
        self.assertAlmostEqual(height.min(), 0.005, places=5)
        self.assertAlmostEqual(height.max(), 1.505, places=5)
        self.assertTrue(np.all(transformed["semantic_0"] == 0.7))
        self.assertTrue(np.allclose(transformed["scale_0"], np.log(1.5)))
        quaternion = np.column_stack([transformed[k] for k in
                                      ("rot_1", "rot_2", "rot_3", "rot_0")])
        up = Rotation.from_quat(quaternion).apply([0, 0, 1])
        np.testing.assert_allclose(up[0], frame[:, 2], atol=1e-6)

    def test_removed_object_sets_placement_box(self):
        rng = np.random.default_rng(4)
        dtype = np.dtype([(name, "<f4") for name in ("x", "y", "z")])
        original = np.zeros(300, dtype=dtype)
        original["x"][100:] = rng.uniform(-0.2, 0.2, 200)
        original["y"][100:] = rng.uniform(0.3, 0.7, 200)
        original["z"][100:] = rng.uniform(0.2, 1.0, 200)
        _, frame = plane_frame([0, 0, 0], [0, 0, 1])
        center, size, count = target_from_removal(
            original, original[:100], np.zeros(3), frame)
        self.assertEqual(count, 200)
        self.assertTrue(np.all(size > 0.01))
        self.assertAlmostEqual(center[1], 0.5, delta=0.05)

    def test_rotation_rejects_view_dependent_asset_without_sh_rotation(self):
        dtype = np.dtype([(name, "<f4") for name in
                          ("x", "y", "z", "scale_0", "scale_1", "scale_2",
                           "rot_0", "rot_1", "rot_2", "rot_3", "f_rest_0")])
        asset = np.zeros(4, dtype=dtype)
        asset["rot_0"] = 1
        asset["f_rest_0"] = 0.1
        with self.assertRaisesRegex(ValueError, "SH coefficients"):
            transform_gaussians(asset, Rotation.from_euler("z", 30, degrees=True).as_matrix(),
                                1, np.zeros(3), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
