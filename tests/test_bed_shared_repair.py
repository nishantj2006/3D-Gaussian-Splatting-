import json

import numpy as np
from PIL import Image
import pytest

from gsedit.reconstruction.build_shared_floor_guides import build_atlas
from gsedit.selection.combine_instance_removal import combine_masks, validated_ids


def test_combined_instances_keep_separate_evidence(tmp_path):
    manifests = []
    for label, mask in (("body", np.array([[1, 0], [0, 0]], np.uint8)),
                        ("frame", np.array([[0, 1], [0, 0]], np.uint8))):
        path = tmp_path / f"{label}.png"
        Image.fromarray(mask * 255).save(path)
        manifest = tmp_path / f"{label}.json"
        manifest.write_text(json.dumps({"views": {
            "v1": {"accepted": True, "mask_path": str(path)}}}))
        manifests.append((label, manifest, tmp_path / f"{label}.npy"))
    views = combine_masks(manifests, tmp_path / "union", min_views=1)
    assert set(views["v1"]["instances"]) == {"body", "frame"}
    assert np.array_equal(np.asarray(Image.open(views["v1"]["mask_path"])) > 127,
                          [[True, True], [False, False]])


def test_ids_reject_bad_schema(tmp_path):
    path = tmp_path / "ids.npy"
    np.save(path, np.array([0, 2, 2]))
    assert validated_ids(path, 3).tolist() == [0, 2]
    np.save(path, np.array([0, 3]))
    with pytest.raises(ValueError, match="Out-of-range"):
        validated_ids(path, 3)


def test_shared_atlas_is_repeatable_and_rejects_sparse_donors():
    rng = np.random.default_rng(4)
    uv = rng.uniform(0, 1, (10000, 2))
    colors = np.tile(np.array([[130, 110, 90]], np.uint8), (len(uv), 1))
    samples = {"a": (uv[:5000], colors[:5000]),
               "b": (uv[5000:], colors[5000:])}
    target = [uv[:200]]
    params = dict(resolution=128, margin=.1, backend="opencv", model="",
                  steps=0, seed=0, min_observed_fraction=.005)
    first = build_atlas(samples, target, **params)
    second = build_atlas(samples, target, **params)
    assert np.array_equal(first[0], second[0])
    assert first[2].mean() >= .005
    with pytest.raises(ValueError, match="Sparse carpet donors"):
        build_atlas({}, target, **params)
