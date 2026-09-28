"""Reproject genuinely observed wall/floor RGB-D into masked camera views.

This stage never invents geometry. A source pixel is eligible only if a clean
surface mask contains it and the independently tracked foreground excludes it.
Output is diagnostic evidence, not an edited PLY.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from gsedit.reconstruction.crossview_background import warp_observed
from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from scene.gaussian_model import GaussianModel


def mask_at(manifest, view, size):
    item = manifest["views"].get(view, {})
    if not item.get("accepted"):
        return None
    return np.asarray(Image.open(item["mask_path"]).convert("L").resize(
        size, Image.Resampling.NEAREST)) > 127


def photo_at(images, view, size):
    matches = [p for p in Path(images).glob(view + ".*") if
               p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(matches) != 1:
        raise ValueError(f"Expected one photo for {view}")
    return np.asarray(Image.open(matches[0]).convert("RGB").resize(
        size, Image.Resampling.LANCZOS))


def fuse_warps(warps, shape, *, depth_agreement):
    """Fuse observations only where at least two source views agree in depth."""
    h, w = shape
    rgb = np.zeros((h, w, 3), np.uint8)
    depth = np.full((h, w), np.nan, np.float32)
    count = np.zeros((h, w), np.uint8)
    conflict = np.zeros((h, w), bool)
    if not warps:
        return rgb, depth, count, conflict
    colors = np.stack([x[0] for x in warps])
    depths = np.stack([x[1] for x in warps])
    hits = np.stack([x[2] for x in warps])
    median = np.full((h, w), np.nan, np.float32)
    any_hit = hits.any(axis=0)
    if any_hit.any():
        median[any_hit] = np.nanmedian(np.where(hits[:, any_hit],
            depths[:, any_hit], np.nan), axis=0)
    consistent = hits & (np.abs(depths - median) <= depth_agreement)
    count = np.minimum(consistent.sum(axis=0), 255).astype(np.uint8)
    valid = count >= 2
    conflict = (hits.sum(axis=0) >= 2) & ~valid
    if valid.any():
        depth[valid] = (np.where(consistent, depths, 0).sum(axis=0) /
                        np.maximum(count, 1))[valid]
        rgb[valid] = np.rint((colors.astype(np.float32) * consistent[..., None]).sum(axis=0)[valid] /
                             count[valid, None]).astype(np.uint8)
    return rgb, depth, count, conflict


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    manifests = {}
    for label, path in (("target", args.target_manifest), ("wall", args.wall_manifest),
                        ("floor", args.floor_manifest)):
        with open(path, encoding="utf-8") as handle:
            manifests[label] = json.load(handle)
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {x["img_name"]: x for x in json.load(handle)}
    holdouts = set(args.holdout_views)
    training = sorted(v for v, item in manifests["target"]["views"].items() if
                      item.get("accepted") and item.get("complete", True) and
                      v in cameras and v not in holdouts)
    targets = args.views or sorted(v for v in manifests["target"]["views"] if v in cameras)
    if len(training) < 3 or not targets:
        raise ValueError("Insufficient complete training masks or target views")
    model = GaussianModel(3, 128)
    model.load_ply(args.scene)
    for key in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
                "_rotation", "_semantic_feature"):
        getattr(model, key).requires_grad_(False)
    source = {}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for view in training:
            camera = cameras[view]
            size = (args.width, round(camera["height"] * args.width / camera["width"]))
            fg = mask_at(manifests["target"], view, size)
            wall = mask_at(manifests["wall"], view, size)
            floor = mask_at(manifests["floor"], view, size)
            if fg is None or (wall is None and floor is None):
                continue
            clean = ((wall if wall is not None else np.zeros_like(fg)) |
                     (floor if floor is not None else np.zeros_like(fg))) & ~fg
            if clean.sum() < args.min_clean_pixels:
                continue
            rasterizer = GaussianRasterizer(camera_settings(camera, size[1], size[0])._replace(
                sh_degree=model.active_sh_degree))
            _, _, _, rendered_depth = rasterizer(
                means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
                shs=model.get_features, colors_precomp=None,
                semantic_feature=model.get_semantic_feature,
                opacities=model.get_opacity, scales=model.get_scaling,
                rotations=model.get_rotation, cov3D_precomp=None)
            source[view] = (photo_at(args.images, view, size),
                            rendered_depth[0].cpu().numpy(), clean)
    if len(source) < 2:
        raise ValueError("Too few clean RGB-D source views")
    output.mkdir(parents=True)
    results = {}
    for view in targets:
        camera = cameras[view]
        size = (args.width, round(camera["height"] * args.width / camera["width"]))
        hole = mask_at(manifests["target"], view, size)
        if hole is None:
            results[view] = {"supported": False, "reason": "missing_target_mask"}
            continue
        warps = [warp_observed(cameras[src], camera, *data, hole) for src, data in source.items()
                 if src != view]
        rgb, depth, count, conflict = fuse_warps(warps, hole.shape,
                                                 depth_agreement=args.depth_agreement)
        prefix = output / view
        Image.fromarray(rgb).save(str(prefix) + "-observed-rgb.png")
        Image.fromarray((count >= 2).astype(np.uint8) * 255).save(
            str(prefix) + "-observed-mask.png")
        Image.fromarray(conflict.astype(np.uint8) * 255).save(str(prefix) + "-conflict.png")
        np.save(str(prefix) + "-observed-depth.npy", depth)
        np.save(str(prefix) + "-support-count.npy", count)
        results[view] = {"supported": True, "hole_pixels": int(hole.sum()),
                         "observed_pixels": int(((count >= 2) & hole).sum()),
                         "observed_fraction": float(((count >= 2) & hole).sum() / max(hole.sum(), 1)),
                         "conflicting_pixels": int((conflict & hole).sum()),
                         "source_views": sorted(v for v in source if v != view)}
    report = {"source": str(Path(args.scene).resolve()), "training_views": sorted(source),
              "holdout_views": sorted(holdouts), "views": results,
              "elapsed_seconds": time.perf_counter() - started,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
              "approved": False, "note": "Observed RGB-D only; no generated color or depth."}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "cameras", "images", "target-manifest", "wall-manifest",
                 "floor-manifest", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--holdout-views", nargs="+", default=[])
    p.add_argument("--views", nargs="+")
    p.add_argument("--width", type=int, default=270)
    p.add_argument("--min-clean-pixels", type=int, default=500)
    p.add_argument("--depth-agreement", type=float, default=.15)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
