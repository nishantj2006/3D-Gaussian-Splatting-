"""Checks for independent parts and bounded projected-footprint selection."""

import unittest

import numpy as np

from gsedit.selection.whole_object_preview import associate_parts, projected_hits


class WholeObjectPreviewTests(unittest.TestCase):
    def test_footprint_reaches_mask_without_distant_leakage(self):
        camera = {"position": [0, 0, 0], "rotation": np.eye(3).tolist(),
                  "width": 100, "height": 100, "fx": 100, "fy": 100}
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True
        points = np.array([[.02, 0, 1], [.2, 0, 1], [0, 0, 3]])
        scales = np.log(np.array([.02, .02, .02]))
        np.testing.assert_array_equal(projected_hits(points, camera, mask, scales,
            depth_tolerance=.2, max_footprint_px=5), [True, False, False])

    def test_cross_view_association_preserves_separate_adjacent_part(self):
        points = np.array([[0, 0, 0], [.03, 0, 0], [.06, 0, 0],
                           [.08, 0, 0], [.11, 0, 0], [3, 0, 0]])
        proposals = [{"view": "a", "ids": np.array([0, 1, 2])},
                     {"view": "b", "ids": np.array([0, 1, 2])},
                     {"view": "a", "ids": np.array([3, 4])},
                     {"view": "b", "ids": np.array([3, 4])},
                     {"view": "a", "ids": np.array([5])}]
        parts, _ = associate_parts(proposals, np.array([0, 1]), points,
                                   overlap=.5, seed_distance=.1,
                                   min_seed_neighbors=1)
        self.assertEqual(len(parts), 2)  # Distant decoy is rejected.
        self.assertEqual(sum(p["connected"] for p in parts), 2)
        self.assertEqual([len(p["views"]) for p in parts[:2]], [2, 2])


if __name__ == "__main__":
    unittest.main()
