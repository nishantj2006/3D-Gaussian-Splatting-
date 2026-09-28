import numpy as np

from gsedit.evaluation.evaluate_keyview_consistency import pair_agreement


def test_pair_agreement_requires_spatially_matching_surface_pixels():
    first = {"xyz": np.array([[0., 0., 0.], [1., 0., 0.]]),
             "rgb": np.array([[100, 100, 100], [200, 200, 200]], np.uint8)}
    second = {"xyz": np.array([[.01, 0., 0.], [4., 0., 0.]]),
              "rgb": np.array([[110, 100, 100], [0, 0, 0]], np.uint8)}
    report = pair_agreement(first, second, .1)
    assert report["matched"] == 1
    assert report["matched_fraction"] == .5
    np.testing.assert_allclose(report["color_l1"], 10/(3*255))


def test_pair_agreement_handles_disjoint_surfaces():
    first = {"xyz": np.array([[0., 0., 0.]]),
             "rgb": np.array([[10, 10, 10]], np.uint8)}
    second = {"xyz": np.array([[9., 0., 0.]]),
              "rgb": np.array([[10, 10, 10]], np.uint8)}
    assert pair_agreement(first, second, .1)["matched"] == 0
