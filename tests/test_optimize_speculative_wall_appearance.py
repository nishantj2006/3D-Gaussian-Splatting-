import numpy as np

from gsedit.reconstruction.optimize_speculative_wall_appearance import neighbor_pairs


def test_neighbor_pairs_stay_within_active_grid():
    active = np.array([[True, True, False], [False, True, True]])
    pairs = neighbor_pairs(active)
    assert {tuple(x) for x in pairs.tolist()} == {(0, 1), (1, 2), (2, 3)}


def test_neighbor_pairs_empty_for_islands():
    assert neighbor_pairs(np.eye(3, dtype=bool)).shape == (0, 2)
