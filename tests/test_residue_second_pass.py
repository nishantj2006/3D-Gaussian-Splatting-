"""Bed-residue masking must exclude independent nearby objects."""
import numpy as np
import pytest

from gsedit.selection.filter_residue_masks import safe_bed_mask


def test_frame_and_furniture_are_removed_from_residue_mask():
    bed = np.ones((7, 7), bool)
    frame = np.zeros_like(bed)
    furniture = np.zeros_like(bed)
    frame[1, 1] = True
    furniture[5, 5] = True
    result = safe_bed_mask(bed, frame, furniture, radius=1)
    assert not result[1, 1] and not result[5, 5]
    assert not result[2, 1] and not result[4, 5]
    assert result[3, 3]


def test_invalid_mask_shape_and_radius_fail():
    mask = np.zeros((4, 4), bool)
    with pytest.raises(ValueError):
        safe_bed_mask(mask, mask[:2], mask)
    with pytest.raises(ValueError):
        safe_bed_mask(mask, mask, mask, radius=33)
