"""Deterministic checks for the CPU ablation's new lifting and growth steps."""

import unittest

import numpy as np

from gsedit.evaluation.benchmark_object_selection import ellipse_hits, footprint_cache, grow_graph


class BenchmarkObjectSelectionTests(unittest.TestCase):
    def test_ellipse_hit_requires_visible_mask_support(self):
        camera = {"position": [0, 0, 0], "rotation": np.eye(3).tolist(),
                  "width": 100, "height": 100, "fx": 100, "fy": 100}
        points = np.array([[0, 0, 1], [.2, 0, 1], [0, 0, 3]], dtype=np.float32)
        scales = np.log(np.full((3, 3), .02, dtype=np.float32))
        rotations = np.tile([1, 0, 0, 0], (3, 1)).astype(np.float32)
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True
        cache = footprint_cache(points, scales, rotations, np.ones(3), camera, mask.shape)
        np.testing.assert_array_equal(ellipse_hits(cache, mask, 3), [True, False, False])

    def test_graph_growth_respects_positive_support_and_appearance(self):
        points = np.array([[0, 0, 0], [.01, 0, 0], [.02, 0, 0],
                           [.03, 0, 0], [.04, 0, 0]], dtype=np.float32)
        rgb = np.array([[.2, .3, .7], [.2, .3, .7], [.2, .3, .7],
                        [.2, .3, .7], [.9, .8, .1]], dtype=np.float32)
        features = np.array([[1, 0]] * 5, dtype=np.float32)
        base = np.array([True, True, False, False, False])
        support = np.array([True, True, True, False, True])
        result, report = grow_graph(points, rgb, features, np.array([0, 1]),
                                    base, support, radius=.05)
        np.testing.assert_array_equal(result, [True, True, True, False, False])
        self.assertEqual(report["added"], 1)


if __name__ == "__main__":
    unittest.main()
