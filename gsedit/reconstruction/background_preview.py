"""Speculative planar wall/carpet preview behind an extracted bed.

Requires --allow-extrapolation because neither hidden wall nor hidden carpet
is observed. Creates a NEW PLY and diagnostics; never modifies inputs.
"""

from gsedit.runtime import PROJECT_ROOT, module_command

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import Delaunay, cKDTree

from gsedit.assets.align_asset import plane_frame
from gsedit.reconstruction.reconstruct_flat import removed_indices
from gsedit.pipelines.surface_pipeline import photo_region, project_mask, xyz_of
from utils.ply_semantic_utils import read_vertices, write_vertices

SH_C0 = 0.28209479177387814


def local_of(vertices, origin, frame):
    return (xyz_of(vertices) - origin) @ frame


def clone_surface(donors, indices, points, spacing, rgb=None):
    copies = donors[indices].copy()
    for j, axis in enumerate(("x", "y", "z")):
        copies[axis] = points[:, j]
        copies[f"scale_{j}"] = np.log(spacing * 1.15)
    copies["rot_0"] = 1
    for j in range(1, 4):
        copies[f"rot_{j}"] = 0
    copies["opacity"] = np.log(0.85 / 0.15)
    if rgb is not None:
        for j in range(3):
            copies[f"f_dc_{j}"] = (rgb[:, j] - 0.5) / SH_C0
    for name in copies.dtype.names:
        if name.startswith("f_rest_"):
            copies[name] = 0
    if "object_id" in copies.dtype.names:
        copies["object_id"] = 0
    return copies


def donors_for_wall(scene, cameras, images, views, origin, frame):
    xyz = xyz_of(scene)
    local = (xyz - origin) @ frame
    rgb = np.column_stack([scene[f"f_dc_{i}"] for i in range(3)]) * SH_C0 + 0.5
    hits = np.zeros(len(scene), dtype=bool)
    photo_colors = []
    for view in views:
        path = next(Path(images).glob(view + ".*"))
        bed_mask, _ = photo_region(path, "blue")
        yy = np.flatnonzero(bed_mask.any(axis=1))
        wall_mask = np.zeros_like(bed_mask)
        wall_mask[:max(1, int(yy.min()) - 15), 30:-30] = True
        hits |= project_mask(xyz, cameras[view], wall_mask)
        with Image.open(path) as source:
            photo = np.asarray(source.convert("RGB").resize(
                (bed_mask.shape[1], bed_mask.shape[0])), dtype=np.float32) / 255
        pixels = photo[wall_mask]
        # Beige/neutral background; exclude very dark corners and trim.
        pixels = pixels[(pixels.mean(axis=1) > 0.15) &
                        (pixels.max(axis=1) - pixels.min(axis=1) < 0.30)]
        if len(pixels):
            photo_colors.append(np.median(pixels, axis=0))
    tan = (rgb[:, 0] > rgb[:, 1] * 1.08) & (rgb[:, 1] > rgb[:, 2] * 1.03)
    donors = hits & tan & (local[:, 2] > 2) & (local[:, 2] < 8)
    if donors.sum() < 100 or not photo_colors:
        raise ValueError("Insufficient visible wall donors for even a speculative fill")
    return donors, np.median(photo_colors, axis=0)


def build(args):
    if not args.allow_extrapolation:
        raise ValueError("Hidden wall/floor is unobserved; pass --allow-extrapolation for preview")
    if not 0.01 <= args.floor_step <= 0.2 or not 0.02 <= args.wall_step <= 0.25:
        raise ValueError("Fill spacing is outside safe bounds")
    if args.render_width <= 0:
        raise ValueError("--render-width must be positive")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output folder: {output}")
    source_ply, original = read_vertices(args.scene)
    _, pruned = read_vertices(args.pruned)
    _, bed = read_vertices(args.bed_core)
    removed_indices(original, pruned)
    if bed.dtype != original.dtype:
        raise ValueError("Bed core PLY schema differs from scene")
    with open(args.ground_plane, encoding="utf-8") as handle:
        ground = json.load(handle)
    origin, frame = plane_frame(ground["plane_origin"],
                                ground["plane_normal_toward_removed_object"])
    cameras = {c["img_name"]: c for c in json.load(open(args.cameras, encoding="utf-8"))}
    bed_local = local_of(bed, origin, frame)
    uv = bed_local[:, :2]
    lo, hi = np.quantile(uv, [0.05, 0.95], axis=0)
    core = uv[np.all((uv >= lo) & (uv <= hi), axis=1)]
    _, _, vt = np.linalg.svd(core - np.median(core, axis=0), full_matrices=False)
    tangent = vt[0]
    wall_normal = np.array([-tangent[1], tangent[0]])
    camera_uv = np.array([(np.asarray(cameras[name]["position"]) - origin) @ frame[:, :2]
                          for name in args.views])
    if np.median((core.mean(axis=0) - camera_uv) @ wall_normal) < 0:
        wall_normal *= -1
    tangent = np.array([wall_normal[1], -wall_normal[0]])
    along = core @ tangent
    behind = core @ wall_normal
    wall_distance = float(np.quantile(behind, 0.95) + 0.25)
    t_min, t_max = np.quantile(along, [0.05, 0.95]) + [-0.3, 0.3]
    height_max = float(np.clip(np.quantile(bed_local[:, 2], 0.95) + 0.6, 4.5, 7))

    floor_local = local_of(pruned, origin, frame)
    floor_pool = np.abs(floor_local[:, 2]) < 0.07
    if floor_pool.sum() < 1000:
        raise ValueError("Insufficient observed floor donors")
    donor_uv = floor_local[floor_pool, :2]
    donor_tree = cKDTree(donor_uv)
    xy_lo, xy_hi = core.min(axis=0) - 0.15, core.max(axis=0) + 0.15
    floor_step = args.floor_step
    shape = np.ceil((xy_hi - xy_lo) / floor_step).astype(int)
    if np.prod(shape) > 200000:
        raise ValueError("Bed footprint too broad for bounded floor preview")
    grid = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij"), -1)
    target_uv = xy_lo + (grid.reshape(-1, 2) + 0.5) * floor_step
    hull = Delaunay(core)
    target_uv = target_uv[hull.find_simplex(target_uv) >= 0]
    distance, nearest = donor_tree.query(target_uv, workers=-1)
    missing = distance > 0.10
    floor_uv = target_uv[missing]
    floor_xyz = origin + floor_uv @ frame[:, :2].T + 0.006 * frame[:, 2]
    floor_donors = pruned[floor_pool]
    floor_fill = clone_surface(floor_donors, nearest[missing], floor_xyz, floor_step)

    wall_pool, wall_rgb = donors_for_wall(pruned, cameras, args.images, args.views,
                                         origin, frame)
    wall_donors = pruned[wall_pool]
    wall_local = floor_local[wall_pool]
    wall_error = np.abs(wall_local[:, :2] @ wall_normal - wall_distance)
    floor_donor_q90 = float(np.quantile(distance, 0.9))
    wall_plane_error_q90 = float(np.quantile(wall_error, 0.9))
    geometry_support_passed = floor_donor_q90 <= 0.25 and wall_plane_error_q90 <= 0.25
    wall_t = wall_local[:, :2] @ tangent
    wall_step = args.wall_step
    t_values = np.arange(t_min, t_max, wall_step)
    h_values = np.arange(0, height_max, wall_step)
    if len(t_values) * len(h_values) > 100000:
        raise ValueError("Wall patch too broad for bounded preview")
    tt, hh = np.meshgrid(t_values, h_values, indexing="ij")
    wall_t_h = np.column_stack((tt.ravel(), hh.ravel()))
    wall_uv = wall_t_h[:, 0, None] * tangent + wall_distance * wall_normal
    wall_xyz = origin + wall_uv @ frame[:, :2].T + wall_t_h[:, 1, None] * frame[:, 2]
    wall_nearest = cKDTree(np.column_stack((wall_t, wall_local[:, 2]))).query(
        wall_t_h, workers=-1)[1]
    donor_rgb = np.column_stack([wall_donors[f"f_dc_{i}"][wall_nearest] for i in range(3)]) * SH_C0 + 0.5
    color = np.clip(wall_rgb + np.clip(donor_rgb - np.median(donor_rgb, axis=0), -0.06, 0.06),
                    0.03, 0.97)
    wall_fill = clone_surface(wall_donors, wall_nearest, wall_xyz, wall_step, color)
    candidate = np.concatenate((pruned, floor_fill, wall_fill))
    output.mkdir(parents=True)
    write_vertices(output / "candidate.ply", candidate, source_ply,
                   ["UNAPPROVED speculative wall/carpet extrapolation behind bed"])
    report = {"approved": False, "speculative": True, "source": str(Path(args.scene).resolve()),
              "pruned": str(Path(args.pruned).resolve()),
              "bed_core": str(Path(args.bed_core).resolve()),
              "floor_added": int(len(floor_fill)), "wall_added": int(len(wall_fill)),
              "floor_missing_fraction": float(missing.mean()),
              "floor_donor_distance_quantiles": np.quantile(distance, [0.5, 0.9, 0.99]).tolist(),
              "wall_donors": int(wall_pool.sum()), "wall_rgb": wall_rgb.tolist(),
              "wall_normal_in_ground_frame": wall_normal.tolist(),
              "wall_plane_distance": wall_distance,
              "warning": "The bed occludes these surfaces in the source photos; geometry and texture are extrapolated."}
    report["floor_donor_distance_q90"] = floor_donor_q90
    report["wall_plane_error_q90"] = wall_plane_error_q90
    report["geometry_support_passed"] = bool(geometry_support_passed)
    report["quality"] = "unreviewed" if geometry_support_passed else "rejected"
    with open(output / "preview.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not args.skip_render:
        for name in args.views:
            subprocess.run([*module_command("render_ply_preview.py"),
                            "--ply", str(output / "candidate.ply"), "--cameras", args.cameras,
                            "--image-name", name, "--output", str(output / (name + ".png")),
                            "--width", str(args.render_width)], check=True)
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--pruned", required=True)
    p.add_argument("--bed-core", required=True, help="Conservative blue-bed selection PLY")
    p.add_argument("--ground-plane", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--floor-step", type=float, default=0.04)
    p.add_argument("--wall-step", type=float, default=0.08)
    p.add_argument("--render-width", type=int, default=540)
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--allow-extrapolation", action="store_true")
    return p


if __name__ == "__main__":
    build(parser().parse_args())
