"""Stage a mask-seeded instance training pilot without modifying source files."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


HOLDOUT = {"frame_0134", "frame_0141", "frame_0143"}


def camera_transform(camera):
    matrix = np.eye(4)
    matrix[:3, :3] = camera["rotation"]
    matrix[:3, 3] = camera["position"]
    matrix[:3, 1:3] *= -1  # COLMAP -> OpenGL; upstream reader reverses it.
    return matrix.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source-ply", "seed-ids", "protected-ids", "cameras", "images", "manifest", "output-dir"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--instance", default="bed")
    p.add_argument("--width", type=int, default=270)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(a.manifest.read_text())
    cameras = {c["img_name"]: c for c in json.loads(a.cameras.read_text())}
    sources = {f.stem: f for f in a.images.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    views = []
    for name, view in manifest["views"].items():
        mask_path = view.get("instances", {}).get(a.instance, {}).get("mask_path")
        if name not in HOLDOUT and view.get("accepted", False) and mask_path and name in cameras and name in sources:
            views.append((name, Path(mask_path)))
    if len(views) < 3:
        raise ValueError("Insufficient non-held-out instance masks")
    ply = PlyData.read(str(a.source_ply), mmap="r")
    vertices = ply["vertex"].data
    ids = np.setdiff1d(np.load(a.seed_ids), np.load(a.protected_ids))
    if len(ids) < 5:
        raise ValueError("Insufficient unprotected seed points")
    if ids.min() < 0 or ids.max() >= len(vertices):
        raise ValueError("Seed IDs outside source")
    out.mkdir(parents=True)
    (out / "images").mkdir()
    (out / "masks").mkdir()
    frames = []
    for name, path in sorted(views):
        camera = cameras[name]
        height = round(a.width * camera["height"] / camera["width"])
        image = Image.open(sources[name]).convert("RGB").resize((a.width, height), Image.Resampling.LANCZOS)
        image.save(out / "images" / (name + ".png"))
        mask = Image.open(path).convert("L").resize((a.width, height), Image.Resampling.NEAREST)
        mask = Image.fromarray((np.asarray(mask) > 127).astype(np.uint8) * 255)
        if not np.any(np.asarray(mask)):
            raise ValueError("Empty accepted mask: " + name)
        rgba = Image.new("RGBA", image.size, (255, 255, 255, 0))
        rgba.putalpha(mask)
        rgba.save(out / "masks" / (name + ".png"))
        factor = a.width / camera["width"]
        frames.append(dict(file_path="images/" + name, transform_matrix=camera_transform(camera),
                           fl_x=camera["fx"] * factor,
                           K=[[camera["fx"] * factor, 0, a.width/2],
                              [0, camera["fy"] * factor, height/2], [0, 0, 1]]))
    (out / "transforms_train.json").write_text(json.dumps({"frames": frames}, indent=2))
    (out / "transforms_test.json").write_text(json.dumps({"frames": []}))
    cloud = np.zeros(len(ids), dtype=[(k, "f4") for k in ("x", "y", "z", "nx", "ny", "nz")]
                     + [(k, "u1") for k in ("red", "green", "blue")])
    for key in ("x", "y", "z"):
        cloud[key] = vertices[key][ids]
    for i, key in enumerate(("red", "green", "blue")):
        cloud[key] = np.clip((.5 + .28209479177387814 * vertices[f"f_dc_{i}"][ids]) * 255, 0, 255)
    PlyData([PlyElement.describe(cloud, "vertex")]).write(str(out / "points3d.ply"))
    np.save(out / "initial-source-ids.npy", ids)
    report = dict(scope="mask_seeded_instance_timing_pilot_not_full_method", source=str(a.source_ply.resolve()),
                  training_views=[name for name, _ in sorted(views)], held_out=sorted(HOLDOUT),
                  initial_points=len(ids), width=a.width, instance=a.instance,
                  protected_seed_ids_excluded=True, no_background_reconstruction=True,
                  raw_native_output_not_128d_scene=True)
    (out / "preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
