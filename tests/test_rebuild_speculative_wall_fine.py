import numpy as np

from gsedit.reconstruction.rebuild_speculative_wall_fine import fine_grid


def test_fine_grid_only_subdivides_active_cells():
    active = np.array([[True, False], [False, True]])
    shape, ij, uv, coarse = fine_grid(active, np.array([2., 3.]), 1., .5)
    assert shape == (4, 4)
    assert len(uv) == 8
    assert active[coarse[:, 0], coarse[:, 1]].all()
    assert np.unique(ij, axis=0).shape[0] == len(ij)


def test_fine_grid_centers_respect_offset():
    active = np.ones((1, 1), bool)
    _, _, uv, _ = fine_grid(active, np.array([4., 7.]), 1., .25)
    assert np.allclose(uv.min(axis=0), [4.125, 7.125])
    assert np.allclose(uv.max(axis=0), [4.875, 7.875])
