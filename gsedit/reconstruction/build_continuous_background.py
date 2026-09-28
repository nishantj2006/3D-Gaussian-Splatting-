"""Seed a continuous, reversible wall/carpet Gaussian replacement.

Masked camera rays locate the unseen surfaces; removed object centers are kept
only as provenance and a conservative spatial safety bound. The output is an
unapproved preview and never overwrites an existing PLY or preview directory.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from gsedit.reconstruction.build_local_background import grid2, mask_votes
from utils.ply_semantic_utils import read_vertices, write_vertices


SH_C0 = .28209479177387814


def ray_intersections(camera, pixels, shape, origin, frame, wall_normal, wall_offset,
                      max_depth=50.):
    """Intersect world-space camera rays with floor and vertical wall planes."""
    height, width = shape
    xy = np.asarray(pixels, dtype=np.float64)
    fx = camera["fx"]*width/camera["width"]
    fy = camera["fy"]*height/camera["height"]
    rays = np.column_stack(((xy[:, 0]-width/2)/fx,
                            (xy[:, 1]-height/2)/fy, np.ones(len(xy))))
    rays = rays @ np.asarray(camera["rotation"], dtype=np.float64).T
    camera_xyz = np.asarray(camera["position"], dtype=np.float64)
    floor_normal = frame[:, 2]
    wall_world_normal = frame[:, :2] @ wall_normal
    with np.errstate(divide="ignore", invalid="ignore"):
        floor_depth = ((origin-camera_xyz) @ floor_normal)/(rays @ floor_normal)
        wall_depth = (wall_offset-((camera_xyz-origin) @ frame)[:2] @ wall_normal)/(
            rays @ wall_world_normal)
    floor_valid = np.isfinite(floor_depth) & (floor_depth > .01) & (floor_depth < max_depth)
    wall_valid = np.isfinite(wall_depth) & (wall_depth > .01) & (wall_depth < max_depth)
    wall_xyz = camera_xyz + np.where(wall_valid, wall_depth, 0)[:, None]*rays
    wall_height = ((wall_xyz-origin) @ frame)[:, 2]
    wall_valid &= np.isfinite(wall_height) & (wall_height >= 0)
    choose_wall = wall_valid & ((wall_depth < floor_depth) | ~floor_valid)
    choose_floor = floor_valid & ((floor_depth < wall_depth) | ~wall_valid)
    floor_xyz = camera_xyz + np.where(floor_valid, floor_depth, 0)[:, None]*rays
    return wall_xyz[choose_wall], floor_xyz[choose_floor]


def gaussian_frame(frame, wall_normal):
    tangent = np.array([-wall_normal[1], wall_normal[0]])
    wall_matrix = np.column_stack((frame[:, :2] @ tangent, frame[:, 2],
                                   frame[:, :2] @ wall_normal))
    if np.linalg.det(wall_matrix) < .99:
        raise ValueError("Wall Gaussian frame is not right-handed")
    return tangent, wall_matrix


def surface_gaussians(vertices, donor_ids, positions, *, step, basis, opacity,
                      normal_scale=.012):
    if len(positions) != len(donor_ids):
        raise ValueError("Donors and positions must match")
    if not len(positions):
        raise ValueError("No supported surface Gaussians")
    if np.any((opacity <= 0) | (opacity >= 1)):
        raise ValueError("Surface opacity must be within (0,1)")
    result = vertices[donor_ids].copy()
    for axis, key in enumerate(("x", "y", "z")):
        result[key] = positions[:, axis]
    result["scale_0"] = np.log(step*.95)
    result["scale_1"] = np.log(step*.95)
    result["scale_2"] = np.log(normal_scale)
    q_xyzw = Rotation.from_matrix(basis).as_quat()
    for key, value in zip(("rot_0", "rot_1", "rot_2", "rot_3"),
                          (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])):
        result[key] = value
    result["opacity"] = np.log(opacity/(1-opacity))
    for key in result.dtype.names:
        if key.startswith("f_rest_"):
            result[key] = 0
    if "object_id" in result.dtype.names:
        result["object_id"] = 0
    return result


def wall_patch_grid(ray_th, bed_th, step, overlap, max_cells):
    """A rectangular surface with a tapered overlap, never a mask silhouette."""
    if len(ray_th) < 100:
        raise ValueError("Too few depth-supported wall rays")
    t_min, t_max = np.quantile(ray_th[:, 0], [.01, .99])
    t_min = max(t_min, np.quantile(bed_th[:, 0], .03)-1.5)
    t_max = min(t_max, np.quantile(bed_th[:, 0], .97)+1.5)
    h_max = min(np.quantile(ray_th[:, 1], .99)+.4,
                max(np.quantile(bed_th[:, 1], .97)+1.0, 1.0))
    if t_max-t_min < .5 or h_max < .5:
        raise ValueError("Unsupported wall footprint")
    core_low = np.array([t_min, 0.])
    core_high = np.array([t_max, h_max])
    lower = core_low-np.array([overlap, 0.])
    upper = core_high+overlap
    grid = grid2(lower, upper, step, max_cells)
    # Fade only at left/right/top. The wall must meet the floor at height zero.
    edge = np.minimum.reduce((grid[:, 0]-lower[0], upper[0]-grid[:, 0],
                              upper[1]-grid[:, 1]))
    alpha = np.clip(edge/max(overlap, 1e-6), .08, 1.)
    return grid, alpha, {"core_low": core_low.tolist(),
                         "core_high": core_high.tolist(),
                         "extended_low": lower.tolist(),
                         "extended_high": upper.tolist()}


def wall_colors(photo_th, photo_rgb, target_th, neighbors=16, falloff=2.):
    if len(photo_th) < neighbors:
        raise ValueError("Too few wall photo samples")
    distance, nearest = cKDTree(photo_th).query(target_th, k=neighbors, workers=-1)
    weights = 1/np.maximum(distance, .05)**2
    nearby = (photo_rgb[nearest]*weights[:, :, None]).sum(axis=1)/weights.sum(axis=1)[:, None]
    global_color = np.median(photo_rgb, axis=0)
    confidence = np.exp(-distance[:, 0]/falloff)[:, None]
    color = np.clip(confidence*nearby+(1-confidence)*global_color, .02, .98)
    return color, distance[:, 0]


def select_ray_support(cameras, bed_manifest, training, origin, frame, normal,
                       offset, stride, max_depth):
    wall, floor, counts = [], [], {}
    for view in training:
        mask = np.asarray(Image.open(bed_manifest["views"][view]["mask_path"])
                          .convert("L")) > 127
        yy, xx = np.where(mask[::stride, ::stride])
        pixels = np.column_stack((xx*stride+.5, yy*stride+.5))
        wxyz, fxyz = ray_intersections(cameras[view], pixels, mask.shape,
                                        origin, frame, normal, offset, max_depth)
        wall.append(wxyz)
        floor.append(fxyz)
        counts[view] = {"wall_rays": len(wxyz), "floor_rays": len(fxyz)}
    return np.concatenate(wall), np.concatenate(floor), counts


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    start = time.perf_counter()
    source_ply, source = read_vertices(args.source)
    _, base = read_vertices(args.base)
    selected = np.unique(np.load(args.selected_indices, allow_pickle=False))
    if not len(selected) or selected.min() < 0 or selected.max() >= len(source):
        raise ValueError("Invalid removal selection")
    original_count = len(source)-len(selected)
    if len(base) < original_count or base.dtype != source.dtype:
        raise ValueError("Base preview does not match source PLY")
    keep = np.ones(len(source), dtype=bool)
    keep[selected] = False
    for key in ("x", "y", "z"):
        if not np.array_equal(base[key][:original_count], source[key][keep]):
            raise ValueError("Base original splats do not align with source")
    with open(args.wall_fit, encoding="utf-8") as handle:
        fit = json.load(handle)
    with open(args.floor_fit, encoding="utf-8") as handle:
        floor_fit = json.load(handle)
    if (fit["inlier_ratio"] < args.min_wall_inlier_ratio or
            fit["plane_error_q90"] > args.max_wall_error or
            fit["supported_views"] < args.min_support_views or
            floor_fit["plane_inlier_ratio"] < args.min_floor_inlier_ratio):
        raise ValueError("Wall/floor plane support is too weak")
    origin = np.asarray(fit["origin"], dtype=np.float64)
    frame = np.asarray(fit["frame"], dtype=np.float64)
    normal = np.asarray(fit["wall_normal_floor_xy"], dtype=np.float64)
    tangent, wall_basis = gaussian_frame(frame, normal)
    offset = float(fit["wall_offset"])
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    with open(args.bed_manifest, encoding="utf-8") as handle:
        bed_manifest = json.load(handle)
    with open(args.wall_manifest, encoding="utf-8") as handle:
        wall_manifest = json.load(handle)
    holdouts = set(args.holdout_views)
    training = sorted(v for v, d in bed_manifest["views"].items()
                      if d.get("accepted") and v in cameras and v not in holdouts)
    if len(training) < args.min_support_views:
        raise ValueError("Too few accepted bed masks")
    wall_rays, floor_rays, ray_counts = select_ray_support(
        cameras, bed_manifest, training, origin, frame, normal, offset,
        args.ray_stride, args.max_ray_depth)
    source_xyz = np.column_stack([source[k] for k in ("x", "y", "z")]).astype(np.float64)
    source_local = (source_xyz-origin) @ frame
    bed_local = source_local[selected]
    bed_th = np.column_stack((bed_local[:, :2] @ tangent, bed_local[:, 2]))
    wall_local = (wall_rays-origin) @ frame
    wall_th = np.column_stack((wall_local[:, :2] @ tangent, wall_local[:, 2]))
    t_low, t_high = np.quantile(bed_th[:, 0], [.03, .97])
    wall_th = wall_th[(wall_th[:, 0] >= t_low-1.5) &
                      (wall_th[:, 0] <= t_high+1.5) &
                      (wall_th[:, 1] <= args.max_wall_height)]
    grid, taper, bounds = wall_patch_grid(wall_th, bed_th, args.wall_step,
                                           args.wall_overlap, args.max_wall_cells)
    wall_uv = grid[:, 0, None]*tangent + offset*normal
    wall_xyz = origin + wall_uv @ frame[:, :2].T + grid[:, 1, None]*frame[:, 2]
    clean_wall_views = [v for v in training if
                        wall_manifest["views"].get(v, {}).get("accepted")]
    bed_support = mask_votes(wall_xyz, cameras, bed_manifest, training)
    wall_support = mask_votes(wall_xyz, cameras, wall_manifest, clean_wall_views)
    supported = (bed_support >= args.min_surface_mask_views) | (
        wall_support >= args.min_surface_mask_views)
    if supported.sum() < 100:
        raise ValueError("Too few wall cells supported by bed or visible wall masks")
    grid = grid[supported]
    taper = taper[supported]
    wall_xyz = wall_xyz[supported]
    photo = np.load(args.wall_photo_samples, allow_pickle=False)
    wall_rgb, color_distance = wall_colors(photo["th"], photo["rgb"], grid,
                                           args.color_neighbors, args.color_falloff)
    wall_color_distance_q90 = float(np.quantile(color_distance, .9))
    if wall_color_distance_q90 > args.max_wall_color_distance:
        raise ValueError("Visible wall color samples are too far from the proposed "
                         f"patch: q90={wall_color_distance_q90:.3f}")
    wall_votes = mask_votes(source_xyz, cameras, wall_manifest, clean_wall_views)
    source_rgb = np.clip(np.column_stack([source[f"f_dc_{j}"] for j in range(3)])*
                         SH_C0+.5, 0, 1)
    source_wall_plane_dist = np.abs(source_local[:, :2] @ normal-offset)
    wall_donor_mask = (wall_votes >= 1) & (source_local[:, 2] >= .5) & (
        source_wall_plane_dist <= args.wall_donor_plane_distance) & (
        np.linalg.norm(source_rgb-np.median(photo["rgb"], axis=0), axis=1) <= .4)
    wall_donor_mask[selected] = False
    wall_ids = np.flatnonzero(wall_donor_mask)
    if len(wall_ids) < args.min_wall_donors:
        raise ValueError(f"Too few independent wall donors: {len(wall_ids)}")
    donor_th = np.column_stack((source_local[wall_ids, :2] @ tangent,
                                source_local[wall_ids, 2]))
    _, nearest = cKDTree(donor_th).query(grid, workers=-1)
    wall_fill = surface_gaussians(source, wall_ids[nearest], wall_xyz,
                                  step=args.wall_step, basis=wall_basis,
                                  opacity=np.clip(args.wall_opacity*taper, .05, .95))
    for j in range(3):
        wall_fill[f"f_dc_{j}"] = (wall_rgb[:, j]-.5)/SH_C0
    semantic_keys = [k for k in source.dtype.names if k.startswith("semantic_")]
    wall_semantic = np.median(np.column_stack([source[k][wall_ids] for k in semantic_keys]), axis=0)
    for i, key in enumerate(semantic_keys):
        wall_fill[key] = wall_semantic[i]
    # Floor support comes from mask-ray intersections plus the earlier,
    # depth-constrained floor seed. No deleted bed center is reused.
    floor_local = (floor_rays-origin) @ frame
    floor_xy = floor_local[:, :2]
    if args.old_floor_count < 0 or len(base) < original_count+args.old_floor_count:
        raise ValueError("Invalid old floor seed count")
    old_floor = (np.column_stack([base[k][original_count:original_count+args.old_floor_count]
                                  for k in ("x", "y", "z")])-origin) @ frame if args.old_floor_count else np.empty((0, 3))
    floor_support = np.concatenate((floor_xy, old_floor[:, :2]))
    floor_support = floor_support[np.all(np.isfinite(floor_support), axis=1)]
    bed_xy_low = np.quantile(bed_local[:, :2], .03, axis=0)-args.floor_bound_margin
    bed_xy_high = np.quantile(bed_local[:, :2], .97, axis=0)+args.floor_bound_margin
    floor_support = floor_support[np.all((floor_support >= bed_xy_low) &
                                         (floor_support <= bed_xy_high), axis=1)]
    if len(floor_support) < 20:
        raise ValueError("Too few floor-ray/seed intersections")
    floor_grid = grid2(floor_support.min(axis=0)-args.floor_overlap,
                       floor_support.max(axis=0)+args.floor_overlap,
                       args.floor_step, args.max_floor_cells)
    support_distance = cKDTree(floor_support).query(floor_grid, workers=-1)[0]
    floor_grid = floor_grid[support_distance <= args.floor_overlap]
    floor_alpha = np.clip((args.floor_overlap-support_distance[
        support_distance <= args.floor_overlap])/args.floor_overlap, .08, 1.)
    floor_xyz = origin + floor_grid @ frame[:, :2].T + args.floor_lift*frame[:, 2]
    floor_donor_mask = np.abs(source_local[:, 2]) <= args.floor_donor_height
    floor_donor_mask[selected] = False
    floor_ids = np.flatnonzero(floor_donor_mask)
    floor_ids = floor_ids[mask_votes(source_xyz[floor_ids], cameras,
                                     bed_manifest, training) == 0]
    if len(floor_ids) < args.min_floor_donors:
        raise ValueError("Too few intact carpet donors")
    floor_distance, floor_nearest = cKDTree(source_local[floor_ids, :2]).query(
        floor_grid, workers=-1)
    if np.quantile(floor_distance, .9) > args.max_floor_donor_distance:
        raise ValueError(f"Carpet donor region is too far away: "
                         f"q90={np.quantile(floor_distance, .9):.3f}")
    floor_fill = surface_gaussians(source, floor_ids[floor_nearest], floor_xyz,
                                   step=args.floor_step, basis=frame,
                                   opacity=np.clip(args.floor_opacity*floor_alpha, .05, .95))
    candidate = np.concatenate((base[:original_count].copy(), floor_fill, wall_fill))
    output.mkdir(parents=True)
    path = output / "candidate.ply"
    write_vertices(path, candidate, source_ply,
                   ["UNAPPROVED continuous wall/floor preview; originals unchanged"])
    # The provenance stores full removed PLY records and maps local opacity
    # edits back to original source IDs without confusing them with new fill.
    write_vertices(output / "removed-bed-gaussians.ply", source[selected].copy(),
                   source_ply, ["Removal provenance only; not replacement geometry"])
    with np.load(args.gate_evidence, allow_pickle=False) as evidence:
        changed = evidence["gate"] < 1
        changed_candidate = evidence["indices"][changed]
        changed_gate = evidence["gate"][changed]
    original_ids = np.flatnonzero(keep)
    np.savez_compressed(output / "deletion-provenance.npz",
                        removed_source_indices=selected,
                        attenuated_source_indices=original_ids[changed_candidate],
                        attenuation_gates=changed_gate,
                        removed_xyz=source_xyz[selected])
    with open(output / "view-footprints.json", "w", encoding="utf-8") as handle:
        json.dump({"source_bed_mask_manifest": str(Path(args.bed_manifest).resolve()),
                   "note": "Source masks are per-view visible footprints, not inferred hidden geometry.",
                   "training_views": training, "holdout_views": sorted(holdouts),
                   "ray_support": ray_counts}, handle, indent=2)
        handle.write("\n")
    report = {"source": str(Path(args.source).resolve()),
              "base": str(Path(args.base).resolve()), "candidate": str(path),
              "original_kept": original_count, "removed_source": len(selected),
              "local_opacity_edits": len(changed_candidate),
              "floor_added": len(floor_fill), "wall_added": len(wall_fill),
              "wall_fit_inlier_ratio": fit["inlier_ratio"],
              "wall_fit_q90_error": fit["plane_error_q90"],
              "floor_fit_inlier_ratio": floor_fit["plane_inlier_ratio"],
              "wall_color_distance_q90": wall_color_distance_q90,
              "floor_donor_distance_q90": float(np.quantile(floor_distance, .9)),
              "wall_donors": len(wall_ids), "floor_donors": len(floor_ids),
              "semantic_dimensions": len(semantic_keys), "surface_bounds": bounds,
              "training_views": training, "holdout_views": sorted(holdouts),
              "approved": False,
              "warning": "Hidden background appearance is extrapolated; visual review required.",
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("floor_added", "wall_added",
        "wall_color_distance_q90", "floor_donor_distance_q90",
        "elapsed_seconds", "peak_rss_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "base", "selected-indices", "gate-evidence", "wall-fit",
                 "floor-fit", "wall-photo-samples", "bed-manifest", "wall-manifest",
                 "cameras", "output-dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--old-floor-count", type=int, default=651)
    p.add_argument("--ray-stride", type=int, default=8)
    p.add_argument("--max-ray-depth", type=float, default=50.)
    p.add_argument("--min-wall-inlier-ratio", type=float, default=.3)
    p.add_argument("--max-wall-error", type=float, default=.3)
    p.add_argument("--min-floor-inlier-ratio", type=float, default=.8)
    p.add_argument("--min-support-views", type=int, default=5)
    p.add_argument("--wall-step", type=float, default=.08)
    p.add_argument("--floor-step", type=float, default=.05)
    p.add_argument("--wall-overlap", type=float, default=.65)
    p.add_argument("--floor-overlap", type=float, default=.3)
    p.add_argument("--min-surface-mask-views", type=int, default=2)
    p.add_argument("--floor-bound-margin", type=float, default=.5)
    p.add_argument("--floor-lift", type=float, default=.006)
    p.add_argument("--wall-opacity", type=float, default=.68)
    p.add_argument("--floor-opacity", type=float, default=.68)
    p.add_argument("--max-wall-height", type=float, default=10.)
    p.add_argument("--wall-donor-plane-distance", type=float, default=1.5)
    p.add_argument("--floor-donor-height", type=float, default=.07)
    p.add_argument("--min-wall-donors", type=int, default=50)
    p.add_argument("--min-floor-donors", type=int, default=1000)
    p.add_argument("--max-floor-donor-distance", type=float, default=1.5)
    p.add_argument("--color-neighbors", type=int, default=16)
    p.add_argument("--max-wall-color-distance", type=float, default=1.5)
    p.add_argument("--color-falloff", type=float, default=2.)
    p.add_argument("--max-floor-cells", type=int, default=100000)
    p.add_argument("--max-wall-cells", type=int, default=100000)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
