import pytest

from gsedit.reconstruction.optimize_depth_guided_background import validate_inputs


def test_geometry_gate_refuses_3d_optimization():
    seed = {"original_kept": 4}
    depth = {"views": {"a": {"accepted": True}}}
    masks = {"views": {"a": {"accepted": True, "complete": True}}}
    with pytest.raises(ValueError, match="Geometry gate failed"):
        validate_inputs(seed, {"geometry_supported": False}, depth, masks,
                        set(), 6, 1)


def test_only_complete_non_holdout_depth_views_train():
    seed = {"original_kept": 4}
    geometry = {"geometry_supported": True, "supported_views": 3,
                "heldout_views": {k: {"supported": True} for k in ("a", "x", "y")}}
    depth = {"views": {"a": {"accepted": True}, "b": {"accepted": True},
                       "c": {"accepted": False}}}
    masks = {"views": {"a": {"accepted": True, "complete": True},
                       "b": {"accepted": True, "complete": False},
                       "c": {"accepted": True, "complete": True}}}
    first, train = validate_inputs(seed, geometry, depth, masks, {"x", "y"}, 6, 1)
    assert (first, train) == (4, ["a"])
    with pytest.raises(ValueError, match="Too few"):
        validate_inputs(seed, geometry, depth, masks, {"a", "x", "y"}, 6, 1)
