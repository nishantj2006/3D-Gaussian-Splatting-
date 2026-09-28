import numpy as np
import pytest

from gsedit.reconstruction.lock_shared_floor import FLOOR_LAYOUT, lock_floor


def test_floor_is_shared_and_exterior_unchanged():
    generated = np.full((4, 4, 3), 20, np.uint8)
    guide = np.full((4, 4, 3), 80, np.uint8)
    source = np.full((4, 4, 3), 140, np.uint8)
    layout = np.zeros((4, 4, 3), np.uint8)
    layout[1:3, 1:3] = FLOOR_LAYOUT
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    result, floor = lock_floor(generated, guide, layout, source, mask)
    assert floor.sum() == 4
    assert np.all(result[floor] == 80)
    assert np.all(result[~mask] == 140)


def test_floor_outside_edit_is_rejected():
    rgb = np.zeros((2, 2, 3), np.uint8)
    layout = rgb.copy()
    layout[0, 0] = FLOOR_LAYOUT
    with pytest.raises(ValueError, match="Invalid floor"):
        lock_floor(rgb, rgb, layout, rgb, np.zeros((2, 2), bool))
