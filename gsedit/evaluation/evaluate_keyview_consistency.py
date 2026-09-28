"""Diagnostic RGB agreement of image-first key views on hypothesized surfaces.

This is not a geometry validation: it only tests whether generated RGB values
agree at nearby 3D positions under the *same speculative* wall/floor layout.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from gsedit.reconstruction.build_imagefirst_guides import assign_surfaces
from gsedit.generation.generate_surface_atlas import backproject_plane
from gsedit.generation.inpaint_key_views import prepare_mask


def pair_agreement(first, second, radius):
    if len(first["xyz"]) == 0 or len(second["xyz"]) == 0:
        return {"matched": 0, "color_l1": None, "matched_fraction": 0.}
    tree = cKDTree(first["xyz"])
    distance, index = tree.query(second["xyz"], workers=-1)
    valid = distance <= radius
    if not valid.any():
        return {"matched": 0, "color_l1": None, "matched_fraction": 0.}
    delta = np.abs(first["rgb"][index[valid]].astype(float) -
                   second["rgb"][valid].astype(float))/255
    return {"matched": int(valid.sum()),
            "matched_fraction": float(valid.mean()),
            "color_l1": float(delta.mean()),
            "spatial_distance_q90": float(np.quantile(distance[valid], .9))}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    with open(args.guide_report, encoding="utf-8") as handle:
        guide = json.load(handle)
    with open(args.wall_fit, encoding="utf-8") as handle:
        wall = json.load(handle)
    wall["wall_offset"] = guide["guide_wall_offset"]
    with open(args.floor_fit, encoding="utf-8") as handle:
        floor = json.load(handle)
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    with open(args.target_manifest, encoding="utf-8") as handle:
        masks = json.load(handle)["views"]
    frame = np.asarray(wall["frame"], float)
    wall_normal = frame[:, :2] @ np.asarray(wall["wall_normal_floor_xy"], float)
    wall_origin = np.asarray(wall["origin"], float) + wall_normal*wall["wall_offset"]
    rng = np.random.default_rng(args.seed)
    views = {}
    for view in args.views:
        image = np.asarray(Image.open(Path(args.images)/f"{view}-inpainted.png")
                           .convert("RGB"))
        shape = image.shape[:2]
        mask = prepare_mask(masks[view]["mask_path"], (shape[1], shape[0]),
                            args.close_px, args.dilate_px)
        y, x = np.where(mask)
        pixels = np.column_stack((x, y))
        floor_xyz, floor_choice = assign_surfaces(cameras[view], pixels,
                                                  shape, floor, wall)
        if guide.get("roi_source_ids"):
            origin = np.asarray(floor["plane_origin"], float)
            basis = np.asarray(wall["frame"], float)[:, :2]
            uv = (floor_xyz-origin) @ basis
            low = np.asarray(guide["uv_low"], float)
            high = np.asarray(guide["uv_high"], float)
            floor_choice &= np.all((uv >= low) & (uv <= high), axis=1)
        wall_xyz, _ = backproject_plane(cameras[view], pixels, shape,
                                        wall_origin, wall_normal)
        rgb = image[y, x]
        per_surface = {}
        for kind, xyz, chosen in (("floor", floor_xyz, floor_choice),
                                  ("wall", wall_xyz, ~floor_choice)):
            xyz = xyz[chosen]
            colors = rgb[chosen]
            if len(xyz) > args.max_samples:
                ids = rng.choice(len(xyz), args.max_samples, replace=False)
                xyz, colors = xyz[ids], colors[ids]
            per_surface[kind] = {"xyz": xyz, "rgb": colors}
        views[view] = per_surface
    report = {"approved": False, "geometry_supported": False,
              "warning": "Agreement conditioned on a speculative wall plane",
              "wall_offset": wall["wall_offset"], "radius": args.radius,
              "pairs": {}}
    for i, a in enumerate(args.views):
        for b in args.views[i+1:]:
            report["pairs"][f"{a}/{b}"] = {
                kind: pair_agreement(views[a][kind], views[b][kind], args.radius)
                for kind in ("floor", "wall")}
    report["image_consistent"] = all(
        result["matched"] >= args.min_matches and
        result["color_l1"] is not None and
        result["color_l1"] <= args.max_color_l1
        for pair in report["pairs"].values() for result in pair.values())
    output.mkdir(parents=True)
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "guide-report", "target-manifest", "cameras",
                 "floor-fit", "wall-fit", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--close-px", type=int, default=5)
    p.add_argument("--dilate-px", type=int, default=2)
    p.add_argument("--radius", type=float, default=.15)
    p.add_argument("--max-samples", type=int, default=15000)
    p.add_argument("--min-matches", type=int, default=500)
    p.add_argument("--max-color-l1", type=float, default=.08)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
