import json
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest

from gsedit.evaluation.prepare_instance_pilot import HOLDOUT, camera_transform, main
from gsedit.evaluation.run_split_splat_pilot import main as run_pilot


def test_pose_axes_roundtrip():
    camera = dict(rotation=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], position=[1, 2, 3])
    matrix = np.array(camera_transform(camera))
    matrix[:3, 1:3] *= -1
    assert np.array_equal(matrix[:3, :3], camera["rotation"])
    assert np.array_equal(matrix[:3, 3], camera["position"])


def test_prepare_excludes_holdout_and_protected_seeds(tmp_path, monkeypatch):
    pictures = tmp_path / "photos"
    masks = tmp_path / "input-masks"
    pictures.mkdir()
    masks.mkdir()
    names = ["train_a", "train_b", "train_c", *sorted(HOLDOUT)]
    cameras = []
    views = {}
    for name in names:
        Image.new("RGB", (8, 6), (20, 40, 60)).save(pictures / (name + ".png"))
        Image.new("L", (8, 6), 255).save(masks / (name + ".png"))
        cameras.append(dict(img_name=name, width=8, height=6, fx=4, fy=4,
                            position=[0, 0, 0], rotation=np.eye(3).tolist()))
        views[name] = dict(accepted=True, instances={"bed": {"mask_path": str(masks / (name + ".png"))}})
    vertices = np.zeros(8, dtype=[(k, "f4") for k in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]])
    vertices["x"] = np.arange(8)
    ply = tmp_path / "original.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(ply))
    original = ply.read_bytes()
    for key, value in [("cameras", cameras), ("manifest", {"views": views})]:
        (tmp_path / (key + ".json")).write_text(json.dumps(value))
    np.save(tmp_path / "seeds.npy", np.arange(8))
    np.save(tmp_path / "protected.npy", np.array([1]))
    output = tmp_path / "staged"
    argv = ["prepare", "--source-ply", str(ply), "--seed-ids", str(tmp_path / "seeds.npy"),
            "--protected-ids", str(tmp_path / "protected.npy"), "--cameras", str(tmp_path / "cameras.json"),
            "--images", str(pictures), "--manifest", str(tmp_path / "manifest.json"),
            "--output-dir", str(output), "--width", "8"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    report = json.loads((output / "preparation.json").read_text())
    assert not set(report["training_views"]) & HOLDOUT
    assert len(report["training_views"]) == 3
    assert 1 not in np.load(output / "initial-source-ids.npy")
    assert ply.read_bytes() == original
    assert json.loads((output / "transforms_test.json").read_text())["frames"] == []
    alpha = np.asarray(Image.open(output / "masks/train_a.png"))[:, :, 3]
    assert np.all(alpha == 255)
    with pytest.raises(FileExistsError):
        main()


def test_existing_training_run_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pilot", "--upstream-dir", str(tmp_path),
                                      "--dataset-dir", str(tmp_path), "--output-dir", str(tmp_path)])
    with pytest.raises(FileExistsError):
        run_pilot()
