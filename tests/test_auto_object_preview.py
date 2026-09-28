"""Safety and color-learning checks for automatic object previews."""

import unittest

import numpy as np

from gsedit.selection.auto_object_preview import coherent_component, learned_appearance, visible_mask_hits


class AutoObjectPreviewTests(unittest.TestCase):
    def test_visibility_rejects_hidden_surface(self):
        camera = {"position": [0, 0, 0], "rotation": np.eye(3).tolist(),
                  "width": 100, "height": 100, "fx": 100, "fy": 100}
        mask = np.zeros((100, 100), dtype=bool)
        mask[49:52, 49:52] = True
        points = np.array([[0, 0, 1], [0, 0, 1.2], [0, 0, 3]])
        np.testing.assert_array_equal(visible_mask_hits(points, camera, mask, .35),
                                      [True, True, False])

    def test_appearance_is_learned_not_named(self):
        rng = np.random.default_rng(5)
        seeds = np.clip(rng.normal([.15, .25, .75], .02, (300, 3)), 0, 1)
        samples = np.vstack([seeds[:5], [[.8, .2, .1]]])
        chosen, details = learned_appearance(seeds, samples)
        self.assertTrue(chosen[:5].all())
        self.assertFalse(chosen[-1])
        self.assertIn(details["components"], (1, 2, 3))

    def test_spatial_component_discards_distant_decoy(self):
        rng = np.random.default_rng(8)
        main = rng.normal([0, 0, 0], .05, (240, 3))
        decoy = rng.normal([5, 0, 0], .05, (30, 3))
        points = np.vstack([main, decoy])
        chosen = coherent_component(points, np.arange(len(points)), radius=.3, minimum=100)
        self.assertEqual(len(chosen), 240)
        self.assertTrue((chosen < 240).all())


if __name__ == "__main__":
    unittest.main()
