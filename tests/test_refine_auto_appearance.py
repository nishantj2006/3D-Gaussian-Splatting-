"""Learned-color growth stays local and preserves its seed selection."""

import unittest

import numpy as np

from gsedit.selection.refine_auto_appearance import refine


class RefineAutoAppearanceTests(unittest.TestCase):
    def test_similar_nearby_point_grows_but_far_and_different_do_not(self):
        rng = np.random.default_rng(42)
        seed_points = rng.normal(0, .01, (250, 3))
        seed_rgb = np.clip(rng.normal([.2, .3, .7], .01, (250, 3)), 0, 1)
        points = np.vstack([seed_points, [[.025, 0, 0], [4, 0, 0], [.02, 0, 0]]])
        colors = np.vstack([seed_rgb, [[.2, .3, .7], [.2, .3, .7], [.9, .1, .1]]])
        scores = np.full(len(points), .5)
        selected, details = refine(points, colors, scores, np.arange(250), .1)
        self.assertTrue(selected[:251].all())
        self.assertFalse(selected[251])
        self.assertFalse(selected[252])
        self.assertEqual(details["appearance_added"], 1)


if __name__ == "__main__":
    unittest.main()
