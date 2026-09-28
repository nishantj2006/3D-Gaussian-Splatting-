import numpy as np
import pytest

from gsedit.reconstruction.fit_floor_from_masks import fit_plane


def test_floor_plane_fit_rejects_outliers_and_is_repeatable():
    rng = np.random.default_rng(4)
    floor = np.column_stack((rng.uniform(-2, 2, 300),
                             rng.uniform(-2, 2, 300),
                             rng.normal(0, .005, 300)))
    outliers = rng.uniform(-2, 2, (60, 3))
    points = np.concatenate((floor, outliers))
    a = fit_plane(points, threshold=.03, trials=200, seed=5)
    b = fit_plane(points, threshold=.03, trials=200, seed=5)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])
    assert a[2].sum() >= 295
    assert abs(a[1][2]) > .99
    assert a[3] < .02


def test_floor_plane_fit_rejects_sparse_points():
    with pytest.raises(ValueError, match="Too few"):
        fit_plane(np.zeros((3, 3)), threshold=.03, trials=20, seed=0)
