"""Non-inference tests for autonomous 2D-to-3D background generation."""

import numpy as np

from gsedit.generation.generate_background_views import (choose_views, fuse_colors,
                                       inpaint_opencv, projected_membership)


def test_choose_views_covers_different_gaussians_deterministically():
    footprints = {"b": np.array([1, 1, 0, 0], bool),
                  "a": np.array([1, 1, 0, 0], bool),
                  "c": np.array([0, 0, 1, 1], bool)}
    assert choose_views(footprints, 2) == (["a", "c"], 1.0)


def test_fuse_generated_colors_preserves_unobserved_donors_and_checks_conflict():
    base = np.full((3, 3), .5)
    observations = [(np.array([0, 1]), np.array([[.1, .2, .3], [.1, .1, .1]])),
                    (np.array([0]), np.array([[.1, .2, .3]]))]
    result, counts, disagreement, consistent = fuse_colors(base, observations, .1)
    np.testing.assert_allclose(result[0], [.1, .2, .3])
    np.testing.assert_allclose(result[2], base[2])
    assert counts.tolist() == [2, 1, 0]
    assert disagreement < 1e-6 and consistent
    observations[1] = (np.array([0]), np.array([[.9, .9, .9]]))
    assert not fuse_colors(base, observations, .1)[3]


def test_opencv_fallback_never_changes_unmasked_pixels():
    photo = np.full((32, 32, 3), 120, dtype=np.uint8)
    photo[10:20, 10:20] = 5
    mask = np.zeros((32, 32), bool)
    mask[10:20, 10:20] = True
    output = inpaint_opencv(photo, mask, 3)
    np.testing.assert_array_equal(output[~mask], photo[~mask])


def test_projected_membership_rejects_points_outside_mask():
    camera = {"position": [0, 0, 0], "rotation": np.eye(3).tolist(),
              "width": 40, "height": 40, "fx": 20, "fy": 20}
    points = np.array([[0, 0, 2], [2, 0, 2], [0, 0, -1]])
    mask = np.zeros((40, 40), bool)
    mask[20, 20] = True
    _, _, hit = projected_membership(points, camera, mask)
    assert hit.tolist() == [True, False, False]
