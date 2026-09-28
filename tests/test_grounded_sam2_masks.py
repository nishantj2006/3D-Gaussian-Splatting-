"""Mask-ranking safety checks without downloading inference models."""

import unittest

import numpy as np

from gsedit.selection.grounded_sam2_masks import rank_mask


class GroundedSam2MaskTests(unittest.TestCase):
    def test_seed_consistent_mask_beats_background(self):
        target = np.zeros((100, 100), dtype=bool)
        target[30:60, 30:60] = True
        decoy = np.zeros((100, 100), dtype=bool)
        decoy[60:95, 60:95] = True
        x = np.array([35, 40, 45])
        y = np.array([35, 40, 45])
        valid = np.ones(3, dtype=bool)
        self.assertGreater(rank_mask(target, x, y, valid, .3)["quality"],
                           rank_mask(decoy, x, y, valid, .9)["quality"])

    def test_scene_sized_mask_is_rejected(self):
        mask = np.ones((20, 20), dtype=bool)
        self.assertIsNone(rank_mask(mask, np.array([3]), np.array([3]),
                                    np.array([True]), .9))


if __name__ == "__main__":
    unittest.main()
