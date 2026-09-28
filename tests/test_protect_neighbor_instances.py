import numpy as np

from gsedit.selection.protect_neighbor_instances import independent_groups


def test_neighbor_instances_remain_separate():
    proposals = [
        {"view": "a", "ids": np.array([1, 2, 3])},
        {"view": "b", "ids": np.array([1, 2, 4])},
        {"view": "a", "ids": np.array([20, 21, 22])},
        {"view": "b", "ids": np.array([20, 21, 23])},
    ]
    groups = independent_groups(proposals, minimum_jaccard=.2)
    assert sorted(map(sorted, groups)) == [[0, 1], [2, 3]]
