"""Seed a reversible wall/carpet replacement from depth and photo evidence."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy.spatial import Delaunay, cKDTree

from gsedit.reconstruction.build_local_background import clone_donors, grid2, mask_votes
from gsedit.selection.multiview_instance import project
from utils.ply_semantic_utils import read_vertices, write_vertices


SH_C0 = .28209479177387814


def wall_semantic_donors(vertices, points, local, selected, cameras, wall_manifest,
                         views, color_reference, *, max_color_distance=.35):
    votes = mask_votes(points, cameras, wall_manifest, views)
    rgb = np.clip(np.column_stack([vertices[f"f_dc_{j}"] for j in range(3)])*
                  SH_C0+.5, 0, 1)
    near_color = np.linalg.norm(rgb-color_reference, axis=1) <= max_color_distance
    candidate = (votes >= 1) & (local[:, 2] >= .5) & near_color
    candidate[selected] = False
    ids = np.flatnonzero(candidate)
    if len(ids) < 50:
        raise ValueError("Too few wall-labeled Gaussians for semantic donors")
    return ids


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse depth-background folder: {output}")
    start = time.perf_counter()
    source_ply, vertices = read_vertices(args.scene)
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float64)
    selected = np.unique(np.load(args.selected_indices, allow_pickle=False))
    with open(args.wall_fit, encoding="utf-8") as handle:
        fit = json.load(handle)
    if fit["inlier_ratio"] < args.min_wall_inlier_ratio or fit["supported_views"] < 3:
        raise ValueError("Depth wall fit is not reliable enough for a preview")
    photo = np.load(args.wall_photo_samples, allow_pickle=False)
    origin = np.asarray(fit["origin"])
    frame = np.asarray(fit["frame"])
    normal = np.asarray(fit["wall_normal_floor_xy"])
    tangent = np.array([-normal[1], normal[0]])
    offset = fit["wall_offset"]
    local = (points-origin) @ frame
    bed = local[selected]
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {x["img_name"]: x for x in json.load(handle)}
    with open(args.bed_manifest, encoding="utf-8") as handle:
        bed_manifest = json.load(handle)
    with open(args.wall_manifest, encoding="utf-8") as handle:
        wall_manifest = json.load(handle)
    training = sorted(set(fit["training_views"])-set(args.holdout_views))
    if len(training) < 3:
        raise ValueError("Too few training views")
    floor_ids = np.flatnonzero((np.abs(local[:, 2]) <= args.floor_threshold))
    floor_ids = np.setdiff1d(floor_ids, selected, assume_unique=True)
    floor_hits = mask_votes(points[floor_ids], cameras, bed_manifest, training)
    floor_ids = floor_ids[floor_hits == 0]
    if len(floor_ids) < args.min_floor_donors:
        raise ValueError("Too few intact floor donors")
    lo = np.quantile(bed[:, :2], .03, axis=0)-args.floor_margin
    hi = np.quantile(bed[:, :2], .97, axis=0)+args.floor_margin
    floor_uv = grid2(lo, hi, args.floor_step, args.max_floor_cells)
    core = bed[np.all((bed[:, :2] >= lo) & (bed[:, :2] <= hi), axis=1), :2]
    floor_uv = floor_uv[Delaunay(core).find_simplex(floor_uv) >= 0]
    floor_xyz = origin + floor_uv @ frame[:, :2].T + args.floor_lift*frame[:, 2]
    floor_hits = mask_votes(floor_xyz, cameras, bed_manifest, training)
    floor_uv = floor_uv[floor_hits >= args.min_mask_views]
    floor_xyz = floor_xyz[floor_hits >= args.min_mask_views]
    floor_distance, floor_nearest = cKDTree(local[floor_ids, :2]).query(
        floor_uv, workers=-1)
    missing = floor_distance > args.existing_floor_distance
    floor_fill = clone_donors(vertices, floor_ids[floor_nearest[missing]],
                              floor_xyz[missing], args.floor_step)
    if len(floor_fill) == 0:
        raise ValueError("No floor gap supported by multi-view masks")
    bed_t = bed[:, :2] @ tangent
    t_low, t_high = np.quantile(bed_t, [.03, .97])
    height_max = np.quantile(bed[:, 2], .98)+args.wall_height_margin
    wall_th = grid2(np.array([t_low, 0.]), np.array([t_high, height_max]),
                    args.wall_step, args.max_wall_cells)
    wall_uv = wall_th[:, 0, None]*tangent + offset*normal
    wall_xyz = origin + wall_uv @ frame[:, :2].T + wall_th[:, 1, None]*frame[:, 2]
    wall_hits = mask_votes(wall_xyz, cameras, bed_manifest, training)
    wall_th = wall_th[wall_hits >= args.min_mask_views]
    wall_xyz = wall_xyz[wall_hits >= args.min_mask_views]
    if len(wall_xyz) == 0:
        raise ValueError("No wall patch supported by multi-view bed masks")
    reference_color = np.median(photo["rgb"], axis=0)
    donors = wall_semantic_donors(vertices, points, local, selected, cameras,
                                  wall_manifest, training, reference_color)
    donor_th = np.column_stack((local[donors, :2] @ tangent, local[donors, 2]))
    _, nearest = cKDTree(donor_th).query(wall_th, workers=-1)
    wall_fill = clone_donors(vertices, donors[nearest], wall_xyz, args.wall_step)
    photo_tree = cKDTree(photo["th"])
    dist, photo_index = photo_tree.query(wall_th, k=min(args.color_neighbors,
                                                      len(photo["th"])), workers=-1)
    if dist.ndim == 1:
        dist, photo_index = dist[:, None], photo_index[:, None]
    near_color = np.median(photo["rgb"][photo_index], axis=1)
    blend = np.exp(-dist[:, 0]/args.color_falloff)[:, None]
    wall_rgb = np.clip(reference_color + blend*(near_color-reference_color), .02, .98)
    for j in range(3):
        wall_fill[f"f_dc_{j}"] = (wall_rgb[:, j]-.5)/SH_C0
    # Every new wall splat gets the same pooled wall feature, keeping its
    # semantics independent from the bed and the preserved wooden frame.
    semantic_names = [k for k in vertices.dtype.names if k.startswith("semantic_")]
    wall_feature = np.median(np.column_stack([vertices[k][donors] for k in semantic_names]), axis=0)
    for i, key in enumerate(semantic_names):
        wall_fill[key] = wall_feature[i]
    keep = np.ones(len(vertices), dtype=bool)
    keep[selected] = False
    candidate = np.concatenate((vertices[keep].copy(), floor_fill, wall_fill))
    output.mkdir(parents=True)
    candidate_path = output / "surface-seeded-preview.ply"
    write_vertices(candidate_path, candidate, source_ply,
                   ["UNAPPROVED depth-guided wall/carpet preview; original preserved"])
    report = {"source": str(Path(args.scene).resolve()), "candidate": str(candidate_path),
              "wall_fit": str(Path(args.wall_fit).resolve()),
              "removed": len(selected), "floor_added": len(floor_fill),
              "wall_added": len(wall_fill), "floor_donors": len(floor_ids),
              "wall_semantic_donors": len(donors),
              "wall_photo_distance_q90": float(np.quantile(dist[:, 0], .9)),
              "wall_color_median": reference_color.tolist(),
              "semantic_dimensions": len(semantic_names),
              "training_views": training, "holdout_views": args.holdout_views,
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "approved": False,
              "warning": "Hidden wall and floor remain extrapolated, not observed."}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("removed", "floor_added",
        "wall_added", "wall_semantic_donors", "wall_photo_distance_q90",
        "elapsed_seconds", "peak_rss_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--selected-indices", required=True)
    p.add_argument("--wall-fit", required=True)
    p.add_argument("--wall-photo-samples", required=True)
    p.add_argument("--bed-manifest", required=True)
    p.add_argument("--wall-manifest", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-wall-inlier-ratio", type=float, default=.15)
    p.add_argument("--floor-threshold", type=float, default=.07)
    p.add_argument("--min-floor-donors", type=int, default=1000)
    p.add_argument("--floor-step", type=float, default=.05)
    p.add_argument("--wall-step", type=float, default=.08)
    p.add_argument("--floor-margin", type=float, default=.1)
    p.add_argument("--wall-height-margin", type=float, default=.25)
    p.add_argument("--floor-lift", type=float, default=.006)
    p.add_argument("--existing-floor-distance", type=float, default=.12)
    p.add_argument("--min-mask-views", type=int, default=2)
    p.add_argument("--max-floor-cells", type=int, default=200000)
    p.add_argument("--max-wall-cells", type=int, default=120000)
    p.add_argument("--color-neighbors", type=int, default=8)
    p.add_argument("--color-falloff", type=float, default=1.)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
