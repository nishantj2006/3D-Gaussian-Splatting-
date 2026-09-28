"""Regression tests for frame protection and edit validation helpers."""

import numpy as np

from gsedit.selection.collect_footprint_candidates import footprint_overlap
from gsedit.evaluation.evaluate_continuous_preview import compare_images
from gsedit.selection.protect_wooden_frame import lower_frame_strip


def test_frame_protection_keeps_only_lower_adjacent_candidate():
    bed = np.zeros((20, 20), dtype=bool)
    bed[5:14, 4:16] = True
    candidate = np.zeros_like(bed)
    candidate[1:3, 4:16] = True  # Spurious wall above bed.
    candidate[14:17, 4:16] = True  # Visible wooden base.
    strip = lower_frame_strip(candidate, bed, below_margin=4, above_margin=1)
    assert strip[14:17, 4:16].all()
    assert not strip[:5].any()


def test_broad_splat_footprint_can_overlap_without_center_in_mask():
    camera = {"position": [0., 0., 0.], "rotation": np.eye(3).tolist(),
              "width": 20, "height": 20, "fx": 20., "fy": 20.}
    points = np.array([[0., 0., 1.], [.2, 0., 1.]])
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 10] = True
    overlap = footprint_overlap(points, camera, mask, np.array([1, 5]))
    assert overlap.tolist() == [True, True]
    overlap_small = footprint_overlap(points, camera, mask, np.array([1, 1]))
    assert overlap_small.tolist() == [True, False]


def test_exterior_difference_is_measured_separately_from_edit():
    before = np.zeros((2, 2, 3), dtype=np.float32)
    after = before.copy()
    after[0, 0] = 1
    after[1, 1] = .5
    mask = np.array([[True, False], [False, False]])
    result = compare_images(before, after, mask)
    assert result["inside_l1_change"] == 1.
    np.testing.assert_allclose(result["outside_l1_change"], .5/3)
