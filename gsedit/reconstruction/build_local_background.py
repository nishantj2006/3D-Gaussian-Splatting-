"""Make an unapproved, geometry-supported wall/floor fill behind a masked object.

The source scene is never changed.  Surface colors and 128D semantics are copied
from intact scene donors.  The filled footprint is limited by the supplied
multi-view object masks, not a hardcoded bed color or world-space box.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy.spatial import Delaunay, cKDTree

from gsedit.reconstruction.fit_occluded_surfaces import fit_support
from gsedit.selection.multiview_instance import project
from utils.ply_semantic_utils import read_vertices, write_vertices


def mask_votes(points, cameras, manifest, views):
    votes = np.zeros(len(points), dtype=np.uint16)
    for view in views:
        item = manifest["views"][view]
        if not item["accepted"]:
            continue
        mask = np.asarray(Image.open(item["mask_path"]).convert("L")) > 127
        x, y, _, valid = project(points, cameras[view], mask.shape)
        ids = np.flatnonzero(valid)
        votes[ids] += mask[y[ids], x[ids]].astype(np.uint16)
    return votes


def clone_donors(vertices, donor_indices, positions, spacing):
    if not len(donor_indices):
        return np.empty(0, dtype=vertices.dtype)
    clones = vertices[donor_indices].copy()
    for j, key in enumerate(("x", "y", "z")):
        clones[key] = positions[:, j]
        clones[f"scale_{j}"] = np.log(spacing)
    clones["rot_0"] = 1
    for key in ("rot_1", "rot_2", "rot_3"):
        clones[key] = 0
    clones["opacity"] = np.log(.65/.35)
    for key in clones.dtype.names:
        if key.startswith("f_rest_"):
            clones[key] = 0
    if "object_id" in clones.dtype.names:
        clones["object_id"] = 0
    return clones


def grid2(low, high, step, max_cells):
    count = np.ceil((high-low)/step).astype(int)
    if np.any(count <= 0) or int(np.prod(count)) > max_cells:
        raise ValueError("Surface footprint is empty or exceeds safety cell limit")
    mesh = np.stack(np.meshgrid(np.arange(count[0]), np.arange(count[1]),
                                indexing="ij"), axis=-1)
    return low + (mesh.reshape(-1, 2)+.5)*step


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    start = time.perf_counter()
    source_ply, vertices = read_vertices(args.scene)
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float64)
    selected = np.unique(np.load(args.selected_indices, allow_pickle=False))
    if not len(selected) or selected.min() < 0 or selected.max() >= len(vertices):
        raise ValueError("Invalid selected indices")
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    with open(args.mask_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    holdouts = set(args.holdout_views)
    training = sorted(set(manifest["views"])-holdouts)
    if len(training) < 3:
        raise ValueError("Insufficient training masks")
    support, wall_ids, floor_ids = fit_support(args)
    origin = np.asarray(support["origin"])
    frame = np.asarray(support["frame"])
    normal = np.asarray(support["wall_normal_floor_xy"])
    offset = support["wall_offset"]
    tangent = np.array([-normal[1], normal[0]])
    local = (points-origin) @ frame
    bed = local[selected]
    # Floor donors must be outside the masked object silhouette in at least
    # one observed training view; this avoids copying residual bed color.
    floor_points = points[floor_ids]
    floor_hits = mask_votes(floor_points, cameras, manifest, training)
    floor_ids = floor_ids[floor_hits == 0]
    if len(floor_ids) < args.min_floor_points:
        raise ValueError("Too few unmasked carpet donors")
    lo = np.quantile(bed[:, :2], .03, axis=0)-args.floor_margin
    hi = np.quantile(bed[:, :2], .97, axis=0)+args.floor_margin
    floor_uv = grid2(lo, hi, args.floor_step, args.max_floor_cells)
    core = bed[np.all((bed[:, :2] >= lo) & (bed[:, :2] <= hi), axis=1), :2]
    if len(core) < 100:
        raise ValueError("Insufficient object footprint for floor fill")
    hull = Delaunay(core)
    floor_uv = floor_uv[hull.find_simplex(floor_uv) >= 0]
    floor_xyz = origin + floor_uv @ frame[:, :2].T + args.floor_lift*frame[:, 2]
    floor_mask_hits = mask_votes(floor_xyz, cameras, manifest, training)
    floor_uv = floor_uv[floor_mask_hits >= args.min_mask_views]
    floor_xyz = floor_xyz[floor_mask_hits >= args.min_mask_views]
    floor_tree = cKDTree(local[floor_ids, :2])
    distance, nearest = floor_tree.query(floor_uv, workers=-1)
    missing = distance > args.existing_floor_distance
    floor_xyz = floor_xyz[missing]
    floor_donors = floor_ids[nearest[missing]]
    floor_fill = clone_donors(vertices, floor_donors, floor_xyz, args.floor_step)
    # Extend trusted above-object wall donors only along that fitted plane.
    trusted_wall_count = len(wall_ids)
    wall_geometry = np.abs(local[:, :2] @ normal-offset) <= args.wall_threshold
    wall_geometry &= (local[:, 2] >= .1) & (local[:, 2] <= args.max_wall_height)
    wall_geometry[selected] = False
    plausible_wall = np.flatnonzero(wall_geometry)
    plausible_wall = plausible_wall[
        mask_votes(points[plausible_wall], cameras, manifest, training) == 0]
    rgb = np.clip(np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]) *
                  .2820947918 + .5, 0, 1)
    appearance_tree = cKDTree(rgb[wall_ids])
    k = min(8, len(wall_ids)-1)
    trusted_distance = appearance_tree.query(rgb[wall_ids], k=k+1,
                                             workers=-1)[0][:, -1]
    appearance_limit = float(np.quantile(trusted_distance, .95))
    donor_distance = appearance_tree.query(rgb[plausible_wall], k=k,
                                           workers=-1)[0]
    if donor_distance.ndim > 1:
        donor_distance = donor_distance[:, -1]
    wall_ids = np.union1d(wall_ids, plausible_wall[donor_distance <= appearance_limit])
    # Candidate placement is clipped to pixels covered by multiple masks.
    wall_t = local[wall_ids, :2] @ tangent
    bed_t = bed[:, :2] @ tangent
    t_low, t_high = np.quantile(bed_t, [.03, .97])
    height_max = np.quantile(bed[:, 2], .98)+args.wall_height_margin
    wall_th = grid2(np.array([t_low, 0.]), np.array([t_high, height_max]),
                    args.wall_step, args.max_wall_cells)
    wall_uv = wall_th[:, 0, None]*tangent + offset*normal
    wall_xyz = origin + wall_uv @ frame[:, :2].T + wall_th[:, 1, None]*frame[:, 2]
    wall_hits = mask_votes(wall_xyz, cameras, manifest, training)
    wall_th = wall_th[wall_hits >= args.min_mask_views]
    wall_xyz = wall_xyz[wall_hits >= args.min_mask_views]
    wall_tree = cKDTree(np.column_stack((wall_t, local[wall_ids, 2])))
    wall_distance, wall_nearest = wall_tree.query(wall_th, workers=-1)
    if len(wall_fill := clone_donors(vertices, wall_ids[wall_nearest],
                                     wall_xyz, args.wall_step)) == 0:
        raise ValueError("No wall fill pixels supported by the source masks")
    if len(floor_fill) == 0:
        raise ValueError("No carpet gap supported by the source masks")
    wall_distance_q90 = float(np.quantile(wall_distance, .9))
    if wall_distance_q90 > args.max_donor_distance:
        raise ValueError(f"Wall donors are too far from the proposed fill: q90={wall_distance_q90:.3f}")
    keep = np.ones(len(vertices), dtype=bool)
    keep[selected] = False
    output.mkdir(parents=True)
    candidate = np.concatenate((vertices[keep].copy(), floor_fill, wall_fill))
    candidate_path = output / "surface-seeded-preview.ply"
    write_vertices(candidate_path, candidate, source_ply,
                   ["UNAPPROVED locally fitted wall/floor preview; original scene preserved"])
    report = {"source": str(Path(args.scene).resolve()),
              "selection": str(Path(args.selected_indices).resolve()),
              "candidate": str(candidate_path), "removed": len(selected),
              "floor_added": len(floor_fill), "wall_added": len(wall_fill),
              "floor_donors": len(floor_ids), "wall_donors": len(wall_ids),
              "trusted_wall_donors": trusted_wall_count,
              "wall_appearance_limit": appearance_limit,
              "wall_donor_distance_q90": wall_distance_q90,
              "surface_support": support, "training_views": training,
              "holdout_views": sorted(holdouts), "semantic_dimensions": sum(
                  k.startswith("semantic_") for k in vertices.dtype.names),
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "approved": False,
              "warning": "Hidden wall/floor texture is extrapolated from intact donors."}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("removed", "floor_added", "wall_added",
                        "wall_donor_distance_q90", "elapsed_seconds", "peak_rss_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--selected-indices", required=True)
    p.add_argument("--floor-plane", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--floor-step", type=float, default=.05)
    p.add_argument("--wall-step", type=float, default=.08)
    p.add_argument("--floor-margin", type=float, default=.1)
    p.add_argument("--wall-height-margin", type=float, default=.25)
    p.add_argument("--floor-lift", type=float, default=.006)
    p.add_argument("--existing-floor-distance", type=float, default=.12)
    p.add_argument("--max-donor-distance", type=float, default=1.0)
    p.add_argument("--max-wall-height", type=float, default=8.0)
    p.add_argument("--min-mask-views", type=int, default=2)
    p.add_argument("--min-floor-points", type=int, default=1000)
    p.add_argument("--max-floor-cells", type=int, default=200000)
    p.add_argument("--max-wall-cells", type=int, default=120000)
    p.add_argument("--wall-threshold", type=float, default=.08)
    p.add_argument("--floor-threshold", type=float, default=.07)
    p.add_argument("--min-wall-points", type=int, default=300)
    p.add_argument("--min-wall-ratio", type=float, default=.15)
    p.add_argument("--trials", type=int, default=600)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
