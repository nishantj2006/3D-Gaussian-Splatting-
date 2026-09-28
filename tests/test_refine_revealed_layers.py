"""Safety checks for the revealed-layer selector."""

import unittest

import numpy as np

from gsedit.selection.refine_revealed_layers import select_revealed


class RevealedLayerSelectionTests(unittest.TestCase):
    def test_requires_visibility_agreement_and_appearance(self):
        inside = np.array([4., 4., 4., .1, 4.])
        outside = np.array([.1, 3., .1, 0., .1])
        views = np.array([3, 3, 1, 3, 3])
        appearance = np.array([True, True, True, True, False])
        selected, agreement = select_revealed(
            inside, outside, views, appearance, min_inside=1.,
            min_views=2, min_agreement=.75)
        np.testing.assert_array_equal(selected, [True, False, False, False, False])
        self.assertGreater(agreement[0], agreement[1])


if __name__ == "__main__":
    unittest.main()
