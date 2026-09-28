import numpy as np

from gsedit.selection.complete_instance_masks import agree, frame_number
from gsedit.selection.segment_rendered_residue import rank_residual


def test_frame_numbers_and_conservative_agreement():
    assert frame_number("frame_0138") == 138
    a = np.zeros((12, 12), bool)
    b = np.zeros_like(a)
    a[2:8, 2:8] = True
    b[3:9, 2:8] = True
    merged, reason = agree([(a, 1.), (b, 1.)], min_iou=.5,
                           min_valid_fraction=.8)
    assert merged.sum() == (a & b).sum()
    assert reason.startswith("two_view_agreement")
    rejected, _ = agree([(a, .4)], min_iou=.5, min_valid_fraction=.8)
    assert rejected is None


def test_residue_must_match_its_independent_instance():
    bed = np.zeros((10, 10), bool)
    bed[1:5, 1:5] = True
    frame = np.zeros_like(bed)
    frame[5:9, 5:9] = True
    assert rank_residual(bed, bed, .8, min_area=.01, min_overlap=.5)
    assert rank_residual(frame, bed, .9, min_area=.01, min_overlap=.5) is None
