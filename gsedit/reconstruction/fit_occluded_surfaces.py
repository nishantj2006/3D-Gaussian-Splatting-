"""Fit observed floor and vertical-wall supports around a masked object.

No scene-specific color, pixel position, or world coordinate is assumed.  The
wall is fitted only from original splats observed outside the object masks.
This module intentionally rejects unsupported planes rather than inventing a
replacement surface.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from gsedit.assets.align_asset import plane_frame
from gsedit.selection.multiview_instance import project
from utils.ply_semantic_utils import read_vertices


def line_ransac(points, *, threshold=.08, trials=600, seed=0):
    """Fit a vertical plane as a 2D line in floor coordinates."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 200 or not np.isfinite(points).all():
        raise ValueError("Too few finite exterior points for a wall plane")
    rng = np.random.default_rng(seed)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(trials):
        a, b = points[rng.choice(len(points), 2, replace=False)]
        tangent = b-a
        length = np.linalg.norm(tangent)
        if length < .2:
            continue
        normal = np.array([-tangent[1], tangent[0]])/length
        inliers = np.abs((points-a) @ normal) <= threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 100:
        raise ValueError("No supported vertical wall plane")
    core = points[best]
    _, _, vt = np.linalg.svd(core-core.mean(0), full_matrices=False)
    tangent = vt[0]
    normal = np.array([-tangent[1], tangent[0]])
    offset = float(np.median(core @ normal))
    errors = np.abs(points @ normal-offset)
    final = errors <= threshold
    return normal, offset, final, float(np.quantile(errors[final], .9))


def exterior_wall_candidates(points, selected, origin, frame, cameras, manifest,
                             *, height_min=.5, mask_margin_px=10):
    local = (points-origin) @ frame
    object_local = local[selected]
    lo = np.quantile(object_local[:, :2], .02, axis=0)-1.0
    hi = np.quantile(object_local[:, :2], .98, axis=0)+1.0
    nearby = np.all((local[:, :2] >= lo) & (local[:, :2] <= hi), axis=1)
    observed = np.zeros(len(points), dtype=np.uint8)
    selected_mask = np.zeros(len(points), dtype=bool)
    selected_mask[selected] = True
    for view, item in manifest["views"].items():
        if not item["accepted"] or view not in cameras:
            continue
        mask = np.asarray(Image.open(item["mask_path"]).convert("L")) > 127
        yy, xx = np.where(mask)
        if not len(yy):
            continue
        x, y, depth, visible = project(points, cameras[view], mask.shape)
        front = np.full(mask.shape, np.inf, dtype=np.float32)
        ids = np.flatnonzero(visible)
        np.minimum.at(front, (y[ids], x[ids]), depth[ids])
        visible_surface = np.zeros(len(points), dtype=bool)
        visible_surface[ids] = depth[ids] <= front[y[ids], x[ids]] + .25
        # Only observed exterior pixels, not occluded projections, support
        # a replacement wall plane.
        above = y < int(np.quantile(yy, .05)) - mask_margin_px
        within_columns = (x >= np.quantile(xx, .02)) & (x <= np.quantile(xx, .98))
        hit = visible_surface & above & within_columns & nearby & ~selected_mask & (
            local[:, 2] >= height_min)
        observed += hit.astype(np.uint8)
    return local, (observed >= 2), observed


def fit_support(args):
    _, vertices = read_vertices(args.scene)
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float64)
    selected = np.unique(np.load(args.selected_indices, allow_pickle=False))
    if not len(selected) or selected.min() < 0 or selected.max() >= len(points):
        raise ValueError("Invalid object selection")
    with open(args.floor_plane, encoding="utf-8") as handle:
        ground = json.load(handle)
    origin, frame = plane_frame(ground["plane_origin"],
                                ground["plane_normal_toward_removed_object"])
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    with open(args.mask_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    local, candidates, observed = exterior_wall_candidates(
        points, selected, origin, frame, cameras, manifest)
    normal, offset, inliers, error = line_ransac(
        local[candidates, :2], threshold=args.wall_threshold,
        trials=args.trials)
    candidate_ids = np.flatnonzero(candidates)
    wall_ids = candidate_ids[inliers]
    floor_ids = np.flatnonzero((np.abs(local[:, 2]) <= args.floor_threshold) &
                               ~np.isin(np.arange(len(points)), selected))
    floor_xyz = local[floor_ids]
    if len(floor_ids) < args.min_floor_points:
        raise ValueError("Too few intact floor Gaussians")
    floor_error = float(np.quantile(np.abs(floor_xyz[:, 2]), .9))
    wall_ratio = len(wall_ids)/max(len(candidate_ids), 1)
    if (len(wall_ids) < args.min_wall_points or wall_ratio < args.min_wall_ratio
            or error > args.wall_threshold or floor_error > args.floor_threshold):
        raise ValueError(f"Unreliable surfaces: wall={len(wall_ids)} ({wall_ratio:.2%}), "
                         f"wall q90={error:.3f}, floor q90={floor_error:.3f}")
    report = {"origin": origin.tolist(), "frame": frame.tolist(),
              "wall_normal_floor_xy": normal.tolist(), "wall_offset": offset,
              "wall_donors": len(wall_ids), "wall_candidate_points": len(candidate_ids),
              "wall_inlier_ratio": wall_ratio, "wall_error_q90": error,
              "floor_donors": len(floor_ids), "floor_error_q90": floor_error,
              "observed_exterior_points": int((observed > 0).sum()),
              "approved": False}
    return report, wall_ids, floor_ids


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--selected-indices", required=True)
    p.add_argument("--floor-plane", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--wall-threshold", type=float, default=.08)
    p.add_argument("--floor-threshold", type=float, default=.07)
    p.add_argument("--min-wall-points", type=int, default=300)
    p.add_argument("--min-wall-ratio", type=float, default=.15)
    p.add_argument("--min-floor-points", type=int, default=1000)
    p.add_argument("--trials", type=int, default=600)
    return p


if __name__ == "__main__":
    print(json.dumps(fit_support(parser().parse_args())[0], indent=2))
