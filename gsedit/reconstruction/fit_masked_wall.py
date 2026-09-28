"""Fit a vertical wall plane from independently segmented source-photo masks."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

from gsedit.assets.align_asset import plane_frame
from gsedit.reconstruction.fit_occluded_surfaces import line_ransac
from gsedit.selection.multiview_instance import project
from utils.ply_semantic_utils import read_vertices


def visible_wall_votes(points, cameras, manifest, views, *, depth_tolerance=.25):
    votes = np.zeros(len(points), dtype=np.uint16)
    by_view = {}
    for view in views:
        mask = np.asarray(Image.open(manifest["views"][view]["mask_path"])
                          .convert("L")) > 127
        x, y, depth, in_frame = project(points, cameras[view], mask.shape)
        ids = np.flatnonzero(in_frame)
        front = np.full(mask.shape, np.inf, dtype=np.float32)
        np.minimum.at(front, (y[ids], x[ids]), depth[ids])
        visible_ids = ids[depth[ids] <= front[y[ids], x[ids]] + depth_tolerance]
        hits = visible_ids[mask[y[visible_ids], x[visible_ids]]]
        votes[hits] += 1
        by_view[view] = int(len(hits))
    return votes, by_view


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse wall-fit folder: {output}")
    start = time.perf_counter()
    _, vertices = read_vertices(args.scene)
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float64)
    selected = np.unique(np.load(args.selected_indices, allow_pickle=False))
    with open(args.floor_plane, encoding="utf-8") as handle:
        ground = json.load(handle)
    origin, frame = plane_frame(ground["plane_origin"],
                                ground["plane_normal_toward_removed_object"])
    local = (points-origin) @ frame
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    with open(args.wall_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    training = sorted(set(view for view, item in manifest["views"].items()
                          if item["accepted"])-set(args.holdout_views))
    if len(training) < args.min_training_views:
        raise ValueError("Too few independent wall masks")
    votes, by_view = visible_wall_votes(points, cameras, manifest, training,
                                        depth_tolerance=args.depth_tolerance)
    candidate = (votes >= args.min_views) & (local[:, 2] >= args.min_height)
    candidate[selected] = False
    ids = np.flatnonzero(candidate)
    if len(ids) < args.min_wall_points:
        raise ValueError(f"Only {len(ids)} visible wall-mask splats")
    normal, offset, inliers, error = line_ransac(local[ids, :2],
        threshold=args.plane_threshold, trials=args.trials)
    wall_ids = ids[inliers]
    ratio = len(wall_ids)/len(ids)
    if len(wall_ids) < args.min_wall_points or ratio < args.min_inlier_ratio:
        raise ValueError(f"Wall plane insufficient: {len(wall_ids)} inliers, {ratio:.2%}")
    # Check that distinct wall-mask views support this plane, rather than one
    # repeated projection from a narrow camera cluster.
    supported_views = 0
    for view in training:
        mask = np.asarray(Image.open(manifest["views"][view]["mask_path"])
                          .convert("L")) > 127
        x, y, _, valid = project(points[wall_ids], cameras[view], mask.shape)
        ids_view = np.flatnonzero(valid)
        count = int(mask[y[ids_view], x[ids_view]].sum())
        supported_views += count >= args.min_view_splats
    if supported_views < args.min_training_views:
        raise ValueError(f"Wall plane supported by only {supported_views} distinct views")
    output.mkdir(parents=True)
    np.save(output / "wall-donor-indices.npy", wall_ids)
    np.save(output / "wall-votes.npy", votes)
    report = {"source": str(Path(args.scene).resolve()),
              "origin": origin.tolist(), "frame": frame.tolist(),
              "wall_normal_floor_xy": normal.tolist(), "wall_offset": offset,
              "wall_candidates": len(ids), "wall_donors": len(wall_ids),
              "inlier_ratio": ratio, "plane_error_q90": error,
              "training_views": training, "holdout_views": args.holdout_views,
              "supported_views": supported_views, "view_hits": by_view,
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "approved": False}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("wall_candidates", "wall_donors",
        "inlier_ratio", "plane_error_q90", "supported_views",
        "elapsed_seconds", "peak_rss_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--selected-indices", required=True)
    p.add_argument("--floor-plane", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--wall-manifest", required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--min-training-views", type=int, default=3)
    p.add_argument("--min-view-splats", type=int, default=100)
    p.add_argument("--min-wall-points", type=int, default=300)
    p.add_argument("--min-inlier-ratio", type=float, default=.15)
    p.add_argument("--min-height", type=float, default=.5)
    p.add_argument("--depth-tolerance", type=float, default=.25)
    p.add_argument("--plane-threshold", type=float, default=.08)
    p.add_argument("--trials", type=int, default=700)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
