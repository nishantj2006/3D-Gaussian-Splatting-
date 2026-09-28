"""Fit a floor plane from intact, multi-view floor-mask Gaussian observations.

This is a scene-agnostic floor fit for the text-targeted edit workflow. It
excludes the requested object and rejects weak fits before any 3D fill is made.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

from gsedit.selection.multiview_instance import project
from utils.ply_semantic_utils import read_vertices


def fit_plane(points, *, threshold, trials, seed):
    xyz = np.asarray(points, dtype=np.float64)
    if len(xyz) < 100 or not np.isfinite(xyz).all():
        raise ValueError("Too few finite intact surface points")
    rng = np.random.default_rng(seed)
    best = np.zeros(len(xyz), dtype=bool)
    for _ in range(trials):
        triangle = xyz[rng.choice(len(xyz), 3, replace=False)]
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        inliers = np.abs((xyz-triangle[0]) @ normal) <= threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 100:
        raise ValueError("No supported floor plane")
    core = xyz[best]
    origin = np.median(core, axis=0)
    _, _, vt = np.linalg.svd(core-origin, full_matrices=False)
    normal = vt[-1]
    error = np.abs((xyz-origin) @ normal)
    inliers = error <= threshold
    return origin, normal, inliers, float(np.quantile(error[inliers], .9))


def visible_mask_votes(points, cameras, manifest, holdouts, *, depth_tolerance):
    votes = np.zeros(len(points), dtype=np.uint16)
    per_view = {}
    for view, info in sorted(manifest["views"].items()):
        if not info.get("accepted") or view in holdouts or view not in cameras:
            continue
        mask = np.asarray(Image.open(info["mask_path"]).convert("L")) > 127
        if not mask.any():
            continue
        x, y, depth, valid = project(points, cameras[view], mask.shape)
        ids = np.flatnonzero(valid)
        front = np.full(mask.shape, np.inf, dtype=np.float32)
        np.minimum.at(front, (y[ids], x[ids]), depth[ids])
        chosen = ids[mask[y[ids], x[ids]] &
                     (depth[ids] <= front[y[ids], x[ids]]+depth_tolerance)]
        votes[chosen] += 1
        per_view[view] = int(len(chosen))
    return votes, per_view


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    _, vertices = read_vertices(args.scene)
    xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float64)
    selected = np.unique(np.load(args.selected_indices, allow_pickle=False))
    if not len(selected) or selected.min() < 0 or selected.max() >= len(xyz):
        raise ValueError("Invalid target selection")
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    with open(args.floor_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    votes, per_view = visible_mask_votes(xyz, cameras, manifest,
                                         set(args.holdout_views),
                                         depth_tolerance=args.depth_tolerance)
    eligible = votes >= args.min_views
    eligible[selected] = False
    opacity = 1/(1+np.exp(-np.clip(vertices["opacity"], -40, 40)))
    eligible &= opacity >= args.min_opacity
    if sum(eligible) < args.min_points:
        raise ValueError(f"Too few independently visible floor donors: {eligible.sum()}")
    ids = np.flatnonzero(eligible)
    origin, normal, inliers, error = fit_plane(
        xyz[ids], threshold=args.threshold, trials=args.trials, seed=args.seed)
    object_direction = xyz[selected].mean(axis=0)-origin
    if object_direction @ normal < 0:
        normal = -normal
    ratio = float(inliers.mean())
    covered_views = sum(per_view[view] >= args.min_points_per_view for view in per_view)
    if (ratio < args.min_inlier_ratio or error > args.threshold or
            covered_views < args.min_supported_views):
        raise ValueError(f"Floor support is unreliable: inliers={ratio:.1%}, "
                         f"q90={error:.3f}, views={covered_views}")
    output.mkdir(parents=True)
    np.save(output / "donor-indices.npy", ids[inliers])
    report = {"plane_origin": origin.tolist(),
              "plane_normal_toward_removed_object": normal.tolist(),
              "plane_inlier_ratio": ratio, "plane_error_q90": error,
              "candidate_points": len(ids), "inlier_points": int(inliers.sum()),
              "supported_views": covered_views, "view_counts": per_view,
              "training_views": sorted(per_view), "holdout_views": args.holdout_views,
              "approved": False, "elapsed_seconds": time.perf_counter()-started,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    with open(output / "preview.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "selected-indices", "cameras", "floor-manifest", "output-dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--depth-tolerance", type=float, default=.08)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--min-opacity", type=float, default=.1)
    p.add_argument("--min-points", type=int, default=500)
    p.add_argument("--min-points-per-view", type=int, default=100)
    p.add_argument("--min-supported-views", type=int, default=3)
    p.add_argument("--min-inlier-ratio", type=float, default=.5)
    p.add_argument("--threshold", type=float, default=.07)
    p.add_argument("--trials", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
