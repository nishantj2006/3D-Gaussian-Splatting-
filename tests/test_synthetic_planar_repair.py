"""Known hidden planar texture: geometry and shared-atlas regression test."""

import numpy as np

from gsedit.generation.generate_surface_atlas import backproject_plane, complete_atlas, sample_atlas


def test_smooth_hidden_planar_region_and_cross_view_consistency():
    yy, xx = np.mgrid[:64, :64]
    truth = np.stack((60 + xx * 2, 80 + yy * 2, 100 + xx + yy), axis=-1).astype(np.uint8)
    hole = (xx >= 25) & (xx < 39) & (yy >= 25) & (yy < 39)
    evidence = truth.copy()
    evidence[hole] = 0
    a = complete_atlas(evidence, ~hole, backend="opencv", pipe=None,
                       prompt="", steps=1, seed=7, radius=5)
    b = complete_atlas(evidence, ~hole, backend="opencv", pipe=None,
                       prompt="", steps=1, seed=7, radius=5)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[~hole], truth[~hole])
    assert np.abs(a[hole].astype(float)-truth[hole]).mean() < 16
    camera = {"position": [0, 0, 0], "rotation": np.eye(3).tolist(),
              "width": 40, "height": 40, "fx": 20, "fy": 20}
    shifted = dict(camera, position=[1, 0, 0])
    left, _ = backproject_plane(camera, [[20, 20]], (40, 40),
                                np.array([0, 0, 5]), np.array([0, 0, 1]))
    right, _ = backproject_plane(shifted, [[16, 20]], (40, 40),
                                 np.array([0, 0, 5]), np.array([0, 0, 1]))
    np.testing.assert_allclose(left, right, atol=1e-8)
    uv = np.tile(np.array([[32, 32]]), (2, 1))
    colors = sample_atlas(a, uv, np.array([0, 0]), np.array([63, 63]))
    np.testing.assert_allclose(colors[0], colors[1])
