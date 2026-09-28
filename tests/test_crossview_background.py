import numpy as np
import pytest

from gsedit.reconstruction.crossview_background import backproject, calibrate_prior, project, warp_observed
from gsedit.reconstruction.recover_background_evidence import fuse_warps


def camera(x=0.):
    return {"position": [x, 0., 0.], "rotation": np.eye(3).tolist(),
            "fx": 50., "fy": 50., "width": 100, "height": 100}


def test_project_backproject_roundtrip():
    xy = np.array([[20., 30.], [60., 70.]])
    xyz = backproject(camera(), xy, np.array([2., 4.]), (100, 100))
    actual, z, valid = project(camera(), xyz, (100, 100))
    assert valid.all()
    assert np.allclose(actual, xy)
    assert np.allclose(z, [2., 4.])


def test_warp_excludes_unclean_pixels_and_targets_only_hole():
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[4, 4] = (30, 100, 150)
    rgb[2, 2] = (255, 0, 0)
    depth = np.full((8, 8), 2., np.float32)
    clean = np.zeros((8, 8), bool)
    clean[4, 4] = True
    hole = np.zeros((8, 8), bool)
    hole[4, 4] = hole[2, 2] = True
    src, dep, observed = warp_observed(camera(), camera(), rgb, depth, clean, hole)
    assert observed.sum() == 1
    assert observed[4, 4] and not observed[2, 2]
    assert np.array_equal(src[4, 4], rgb[4, 4])
    assert dep[4, 4] == pytest.approx(2.)


def test_fuse_rejects_conflicting_depths():
    rgb = np.ones((2, 2, 3), np.uint8) * 50
    hit = np.ones((2, 2), bool)
    a = (rgb, np.ones((2, 2), np.float32), hit)
    b = (rgb, np.ones((2, 2), np.float32) * 3, hit)
    _, depth, count, conflict = fuse_warps([a, b], (2, 2), depth_agreement=.1)
    assert not count.any()
    assert conflict.all()
    assert np.isnan(depth).all()


def test_calibrate_prior_is_robust_and_rejects_sparse_support():
    prior = np.arange(100., dtype=np.float32).reshape(10, 10) / 100 + 1
    actual = 2 * prior + 1
    actual[0, 0] = 100
    aligned, report = calibrate_prior(prior, actual, np.ones_like(prior, bool),
                                      min_samples=20)
    assert np.median(np.abs(aligned[1:] - actual[1:])) < .05
    assert report["observed_samples"] == 100
    with pytest.raises(ValueError, match="Too few"):
        calibrate_prior(prior, actual, np.zeros_like(prior, bool))
