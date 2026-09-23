import unittest

import numpy as np
import trimesh

from bridge import sample_mesh_gaussians


class MeshSamplingTests(unittest.TestCase):
    def test_repeatable_surface_samples_and_orientation(self):
        mesh = trimesh.creation.box(extents=(1, 2, 3))
        first = sample_mesh_gaussians(mesh, 1000, seed=13)
        second = sample_mesh_gaussians(mesh, 1000, seed=13)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        points, colors, scales, rotations = first
        self.assertEqual(points.shape, (1000, 3))
        self.assertEqual(colors.shape, (1000, 3))
        self.assertEqual(scales.shape, (1000, 3))
        self.assertEqual(rotations.shape, (1000, 4))
        self.assertTrue(np.all(scales[:, 2] < scales[:, 0]))
        np.testing.assert_allclose(np.linalg.norm(rotations, axis=1), 1, atol=1e-6)
        self.assertTrue(np.all(np.isfinite(points)))

    def test_rejects_bad_density(self):
        with self.assertRaises(ValueError):
            sample_mesh_gaussians(trimesh.creation.box(), 0)


if __name__ == "__main__":
    unittest.main()
