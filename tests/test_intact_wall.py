import numpy as np
import pytest

from gsedit.reconstruction.filter_intact_wall import clean_wall


def test_wall_donors_exclude_target_and_floor_boundary():
    wall = np.ones((12, 12), bool)
    target = np.zeros_like(wall)
    floor = np.zeros_like(wall)
    target[4, 4] = True
    floor[9:] = True
    result = clean_wall(wall, target, floor, margin=1)
    assert not result[4, 4]
    assert not result[4, 5]
    assert not result[8, 2]
    assert result[1, 1]


def test_wall_mask_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        clean_wall(np.ones((2, 2), bool), np.ones((3, 3), bool),
                   np.ones((2, 2), bool), 0)
