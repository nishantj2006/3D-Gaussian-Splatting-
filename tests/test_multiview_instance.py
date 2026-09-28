"""Geometry and safety checks for multi-view instance refinement."""

import unittest

import numpy as np

from gsedit.selection.multiview_instance import combine_evidence, evidence_for_view, project, seed_box


class MultiviewInstanceTests(unittest.TestCase):
    def setUp(self):
        self.camera = {"position": [0, 0, 0], "rotation": np.eye(3).tolist(),
                       "width": 100, "height": 100, "fx": 100, "fy": 100}

    def test_projection_and_box(self):
        points = np.array([[x / 100, y / 100, 1] for x in range(-20, 21, 4)
                           for y in range(-20, 21, 4)])
        x, y, depth, visible = project(points, self.camera, (100, 100))
        self.assertTrue(visible.all())
        self.assertTrue(np.all(depth == 1))
        box = seed_box(x, y, visible, (100, 100))
        self.assertLess(box[0], 35)
        self.assertGreater(box[2], 65)

    def test_depth_rejects_background_on_same_pixels(self):
        seeds = np.array([[0, 0, 1.0]])
        points = np.array([[0, 0, 1.0], [0, 0, 1.2], [0, 0, 3.0]])
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True
        np.testing.assert_array_equal(evidence_for_view(points, seeds, self.camera,
                                                        mask, 0.5), [True, True, False])

    def test_needs_cross_view_agreement_and_keeps_seed(self):
        points = np.array([[0, 0, 1], [.1, 0, 1], [.2, 0, 1], [3, 0, 1]], float)
        a = np.array([True, True, False, True])
        b = np.array([True, False, False, True])
        chosen, votes = combine_evidence(points, np.array([2]), [a, b], 0.5, 2)
        np.testing.assert_array_equal(chosen, [True, False, True, False])
        np.testing.assert_array_equal(votes, [2, 1, 0, 2])


if __name__ == "__main__":
    unittest.main()
