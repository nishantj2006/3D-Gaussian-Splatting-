"""Fast deterministic checks for preview selection safety."""

import unittest

import numpy as np

from gsedit.selection.materialize_selection_preview import validate_indices
from gsedit.selection.refine_rendered_selection import candidate_pool


class RenderedSelectionTests(unittest.TestCase):
    def test_candidate_pool_keeps_baseline_and_nearby_splats(self):
        points = np.array([[0, 0, 0], [.2, 0, 0], [1, 0, 0],
                           [5, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(candidate_pool(points, np.array([0]),
                                                      np.array([2]), .3), [0, 1, 2])

    def test_invalid_or_broad_selection_rejected(self):
        for indices in (np.array([], dtype=int), np.array([0, 0]),
                        np.array([-1]), np.array([10]),
                        np.array([0, 1, 2])):
            with self.assertRaises(ValueError):
                validate_indices(indices, 10, .2)
        validate_indices(np.array([0, 1]), 10, .2)


if __name__ == "__main__":
    unittest.main()
