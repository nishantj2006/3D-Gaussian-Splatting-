"""Tests for the optional boundary color-matching preview."""

import unittest

import numpy as np

from refine_fill_color import adjust_added_colors, boundary_correction, fill_mask


class FillColorTests(unittest.TestCase):
    def test_fit_detects_low_frequency_brightness_mismatch(self):
        height = width = 160
        mask = np.zeros((height, width), dtype=bool)
        mask[40:120, 40:120] = True
        yy = np.arange(height)[:, None]
        image = np.full((height, width, 3), 0.55)
        image[mask] += np.broadcast_to(0.04 + 0.02 * (yy - 80) / 80,
                                       (height, width))[mask, None]
        image = np.uint8(np.clip(image * 255, 0, 255))
        coefficients, details = boundary_correction(image, mask)
        self.assertGreater(details["boundary_samples"], 100)
        self.assertLess(coefficients[0].mean(), -0.01)
        self.assertEqual(coefficients.shape, (3, 3))

    def test_only_added_rgb_changes(self):
        dtype = np.dtype([(name, "<f4") for name in
                          ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                           "semantic_0", "opacity")])
        vertices = np.zeros(3, dtype=dtype)
        vertices["z"] = 1
        vertices["semantic_0"] = [0.1, 0.8, 0.8]
        camera = {"position": [0, 0, 0], "rotation": np.eye(3).tolist(),
                  "width": 100, "height": 100, "fx": 100, "fy": 100}
        coefficients = np.array([[0.02, -0.01, 0.03], [0, 0, 0], [0, 0, 0]])
        adjusted = adjust_added_colors(vertices, 2, camera, 100, 100,
                                       coefficients, 1.5)
        self.assertEqual(vertices[0].tobytes(), adjusted[0].tobytes())
        np.testing.assert_array_equal(vertices["semantic_0"], adjusted["semantic_0"])
        np.testing.assert_allclose(
            (adjusted["f_dc_0"][-2:] - vertices["f_dc_0"][-2:]) * 0.2820947918,
            0.03, atol=1e-5)
        with self.assertRaisesRegex(ValueError, "strength"):
            adjust_added_colors(vertices, 2, camera, 100, 100,
                                coefficients, 0)


if __name__ == "__main__":
    unittest.main()
