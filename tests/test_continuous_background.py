"""Geometry and schema checks for the continuous wall/floor preview."""

import numpy as np
import pytest

from gsedit.reconstruction.build_continuous_background import (ray_intersections, surface_gaussians,
                                         wall_patch_grid)


def test_masked_rays_hit_wall_or_floor_not_removed_object_centers():
    # Camera looks along +Z; floor is Y=0, wall is Z=5.
    frame = np.column_stack((np.array([1., 0., 0.]),
                             np.array([0., 0., -1.]),
                             np.array([0., 1., 0.])))
    camera = {"position": [0., 1., 0.], "rotation": np.eye(3).tolist(),
              "width": 100, "height": 100, "fx": 50., "fy": 50.}
    wall, floor = ray_intersections(camera, np.array([[50., 50.], [50., 30.]]),
                                    (100, 100), np.zeros(3), frame,
                                    np.array([0., -1.]), 5.)
    assert len(wall) == 1 and len(floor) == 1
    np.testing.assert_allclose(wall[0], [0., 1., 5.], atol=1e-5)
    np.testing.assert_allclose(floor[0, 1], 0., atol=1e-5)


def test_wall_patch_is_continuous_rectangle_with_taper():
    ray_th = np.column_stack((np.linspace(-2, 2, 300),
                              np.linspace(.2, 3, 300)))
    bed_th = ray_th.copy()
    grid, taper, bounds = wall_patch_grid(ray_th, bed_th, .1, .5, 20000)
    assert len(grid) > 1000
    assert (grid[:, 1] >= 0).all()
    assert 0 < taper.min() < 1
    assert taper.max() == 1
    assert bounds["extended_high"][0] > bounds["core_high"][0]
    with pytest.raises(ValueError, match="Too few"):
        wall_patch_grid(ray_th[:5], bed_th, .1, .5, 20000)


def test_new_gaussians_keep_schema_and_semantics_from_surface_donors():
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("opacity", "<f4")] + [(f"scale_{i}", "<f4") for i in range(3)]
    fields += [(f"rot_{i}", "<f4") for i in range(4)]
    fields += [("f_rest_0", "<f4")]
    fields += [(f"semantic_{i}", "<f4") for i in range(128)]
    donors = np.zeros(2, dtype=fields)
    donors["semantic_0"] = [1, 2]
    donors["f_rest_0"] = 7
    result = surface_gaussians(donors, np.array([1, 0]),
        np.array([[3., 4., 5.], [6., 7., 8.]]), step=.05,
        basis=np.eye(3), opacity=np.array([.5, .8]))
    assert result.dtype == donors.dtype
    assert result["semantic_0"].tolist() == [2, 1]
    assert np.all(result["f_rest_0"] == 0)
    assert np.isfinite(result["opacity"]).all()
    np.testing.assert_array_equal(result["x"], [3., 6.])
    repeated = surface_gaussians(donors, np.array([1, 0]),
        np.array([[3., 4., 5.], [6., 7., 8.]]), step=.05,
        basis=np.eye(3), opacity=np.array([.5, .8]))
    np.testing.assert_array_equal(result, repeated)
