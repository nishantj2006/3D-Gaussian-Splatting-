"""Checks that multi-view contradictions prevent broad 3D deletion."""

import unittest

import numpy as np

from gsedit.selection.lift_grounded_masks import consensus, sample_ellipse


class LiftGroundedMasksTests(unittest.TestCase):
    def test_consensus_rejects_conflicting_or_distant_points(self):
        points = np.array([[0, 0, 0], [.1, 0, 0], [.2, 0, 0], [2, 0, 0]], dtype=np.float32)
        positive = np.array([3, 3, 2, 3], dtype=np.uint16)
        negative = np.array([0, 0, 2, 0], dtype=np.uint16)
        chosen, _ = consensus(positive, negative, np.array([0]), points,
                              min_views=2, min_agreement=.65, max_seed_distance=.5)
        np.testing.assert_array_equal(chosen, [True, True, False, False])

    def test_ellipse_samples_obey_mask(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10] = True
        cache = (np.array([0, 1]), np.array([10, 15]), np.array([10, 10]),
                 np.zeros((2, 4), dtype=np.float32), np.array([True, True]))
        ids, coverage, visible = sample_ellipse(cache, mask, 2)
        np.testing.assert_array_equal(ids, [0, 1])
        np.testing.assert_allclose(coverage, [1, 0], atol=1e-6)
        np.testing.assert_array_equal(visible, [True, True])


if __name__ == "__main__":
    unittest.main()
