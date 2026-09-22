"""Preview and approve a flat-surface Gaussian fill after object removal.

The input PLY must be an order-preserving subset of the original PLY, as
produced by remove.py. No original files are overwritten. This does not
recover unseen ground truth: it samples a nearby observed texture patch,
places small Gaussians on the fitted plane, and copies carpet semantics.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from utils.ply_semantic_utils import read_object_manifest, read_vertices, write_object_manifest, write_vertices


def xyz_of(vertices):
    return np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float64)


def removed_indices(original, pruned):
    """Require a byte-identical, order-preserving subset; return its complement."""
    if original.dtype != pruned.dtype or len(pruned) > len(original):
        raise ValueError("Original and pruned PLYs must have the same schema and ordered rows")
    # Every edit in remove.py preserves the entire vertex record and its order.
    source = original.view(np.dtype((np.void, original.dtype.itemsize))).ravel()
    kept = pruned.view(np.dtype((np.void, pruned.dtype.itemsize))).ravel()
    missing = np.ones(len(original), dtype=bool)
    j = 0
    for i, row in enumerate(source):
        if j < len(kept) and row == kept[j]:
            missing[i] = False
            j += 1
    if j != len(pruned) or not missing.any():
        raise ValueError("Pruned PLY is not an order-preserving subset of the original")
    return missing


def fit_plane(points, removed_points, seed=0, threshold=0.035, trials=350):
    low = np.quantile(removed_points, 0.01, axis=0) - 0.8
    high = np.quantile(removed_points, 0.99, axis=0) + 0.8
    nearby = points[np.all((points >= low) & (points <= high), axis=1)]
    if len(nearby) < 500:
        raise ValueError("Too few surrounding Gaussians to fit a surface")
    rng = np.random.default_rng(seed)
    sample = nearby[rng.choice(len(nearby), min(30000, len(nearby)), replace=False)]
    best = None
    best_count = 0
    for _ in range(trials):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        inliers = np.abs((sample - tri[0]) @ normal) < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best = sample[inliers]
    if best is None or best_count < max(500, len(sample) * 0.12):
        raise ValueError("No dominant flat surface near the removal; refusing to invent geometry")
    origin = np.mean(best, axis=0)
    _, _, vt = np.linalg.svd(best - origin, full_matrices=False)
    normal = vt[-1]
    residual = np.abs((sample - origin) @ normal)
    if np.median(residual[residual < threshold * 2]) > threshold:
        raise ValueError("Fitted surface is too uneven for a planar fill")
    object_dist = (removed_points - origin) @ normal
    far = object_dist[np.abs(object_dist) > 0.15]
    if len(far) >= 50 and np.median(far) < 0:
        normal = -normal
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    u = reference - np.dot(reference, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return origin, normal, u, v, best_count / len(sample)


def plane_coordinates(points, origin, normal, u, v):
    delta = points - origin
    return np.column_stack((delta @ u, delta @ v)), delta @ normal


def make_target(removed_uv, removed_distance, cell=0.025, repair_margin=0.05):
    near = np.abs(removed_distance) < 0.08
    if near.sum() >= 100:
        uv = removed_uv[near]
        basis = "removed_surface_splats"
    else:
        plausible = np.abs(removed_distance) < 3.0
        if plausible.sum() < 100:
            raise ValueError("Removal has no usable surface footprint")
        uv = removed_uv[plausible]
        basis = "object_projection"
    padding = max(0.1, repair_margin + 0.05)
    lo = np.quantile(uv, 0.005, axis=0) - padding
    hi = np.quantile(uv, 0.995, axis=0) + padding
    shape = np.maximum(1, np.ceil((hi - lo) / cell).astype(int))
    if np.prod(shape) > 250000:
        raise ValueError("Removal footprint is too large for this local flat-surface workflow")
    ij = np.floor((uv - lo) / cell).astype(int)
    good = np.all((ij >= 0) & (ij < shape), axis=1)
    occupied = np.zeros(tuple(shape), dtype=bool)
    occupied[ij[good, 0], ij[good, 1]] = True
    occupied = ndimage.binary_closing(occupied, iterations=2)
    occupied = ndimage.binary_fill_holes(occupied)
    if repair_margin < 0 or repair_margin > 0.5:
        raise ValueError("--repair-margin must be between 0 and 0.5")
    dilation_steps = round(repair_margin / cell)
    if dilation_steps:
        occupied = ndimage.binary_dilation(occupied, iterations=dilation_steps)
    labels, count = ndimage.label(occupied)
    if count == 0:
        raise ValueError("No coherent surface footprint found")
    sizes = np.bincount(labels.ravel())[1:]
    largest = np.argmax(sizes) + 1
    if sizes.max() / occupied.sum() < 0.65:
        raise ValueError("Removal contains several disconnected surface regions")
    return labels == largest, lo, cell, basis


def mask_lookup(uv, mask, low, cell):
    ij = np.floor((uv - low) / cell).astype(int)
    inside = np.all((ij >= 0) & (ij < np.array(mask.shape)), axis=1)
    result = np.zeros(len(uv), dtype=bool)
    valid = ij[inside]
    result[inside] = mask[valid[:, 0], valid[:, 1]]
    return result


def extend_target_from_view_mask(target, low, cell, mask_path, view_name,
                                 cameras_path, origin, normal, u, v, margin=0.05):
    """Back-project a reviewed 2D floor/shadow mask onto the fitted plane."""
    with open(cameras_path, encoding="utf-8") as handle:
        camera = next((item for item in json.load(handle) if item["img_name"] == view_name), None)
    if camera is None:
        raise ValueError(f"Extra repair view {view_name!r} is absent from cameras.json")
    with Image.open(mask_path) as image:
        marked = np.asarray(image.convert("L")) > 127
    if not marked.any() or marked.mean() > 0.25:
        raise ValueError("Extra repair mask must mark a nonempty, localized region")
    if margin < 0 or margin > 0.5:
        raise ValueError("--extra-mask-margin must be between 0 and 0.5")
    yy, xx = np.nonzero(marked)
    if len(xx) > 100000:
        take = np.linspace(0, len(xx) - 1, 100000, dtype=int)
        yy, xx = yy[take], xx[take]
    scale_x = camera["width"] / marked.shape[1]
    scale_y = camera["height"] / marked.shape[0]
    rays = np.column_stack(((xx * scale_x - camera["width"] / 2) / camera["fx"],
                            (yy * scale_y - camera["height"] / 2) / camera["fy"],
                            np.ones(len(xx)))) @ np.asarray(camera["rotation"]).T
    position = np.asarray(camera["position"])
    denominator = rays @ normal
    valid = np.abs(denominator) > 1e-5
    t = np.zeros(len(xx))
    t[valid] = np.dot(origin - position, normal) / denominator[valid]
    valid &= (t > 0) & (t < 100)
    points = position + rays[valid] * t[valid, None]
    extra_uv, _ = plane_coordinates(points, origin, normal, u, v)
    center = low + np.array(target.shape) * cell / 2
    extra_uv = extra_uv[np.linalg.norm(extra_uv - center, axis=1) < 2.0]
    if len(extra_uv) < 50:
        raise ValueError("Extra repair mask does not intersect the local surface")
    new_low = np.minimum(low, extra_uv.min(axis=0) - margin - cell)
    new_low = low - np.ceil((low - new_low) / cell) * cell
    old_high = low + np.array(target.shape) * cell
    new_high = np.maximum(old_high, extra_uv.max(axis=0) + margin + cell)
    shape = np.ceil((new_high - new_low) / cell).astype(int)
    if np.prod(shape) > 250000:
        raise ValueError("Extra repair mask makes the target region too large")
    expanded = np.zeros(tuple(shape), dtype=bool)
    offset = np.rint((low - new_low) / cell).astype(int)
    expanded[offset[0]:offset[0] + target.shape[0],
             offset[1]:offset[1] + target.shape[1]] = target
    marks = np.zeros_like(expanded)
    ij = np.floor((extra_uv - new_low) / cell).astype(int)
    inside = np.all((ij >= 0) & (ij < shape), axis=1)
    marks[ij[inside, 0], ij[inside, 1]] = True
    marks = ndimage.binary_closing(marks, iterations=2)
    marks = ndimage.binary_fill_holes(marks)
    if margin:
        marks = ndimage.binary_dilation(marks, iterations=round(margin / cell))
    expanded |= marks
    return expanded, new_low, int(marks.sum())


def refine_object_footprint(original, pruned, coarse_missing, seed=0, cell=0.025):
    """Replace a broad 3D box cut with a compact plane-projected object cut."""
    xyz = xyz_of(original)
    origin, normal, u, v, _ = fit_plane(xyz_of(pruned), xyz[coarse_missing], seed=seed)
    uv, distance = plane_coordinates(xyz, origin, normal, u, v)
    tall = coarse_missing & (distance > 0.25)
    if tall.sum() < 100:
        raise ValueError("Cannot refine removal: too few object splats above the surface")
    projected = uv[tall]
    low_quantile = np.quantile(projected, 0.01, axis=0)
    high_quantile = np.quantile(projected, 0.99, axis=0)
    core = projected[np.all((projected >= low_quantile) & (projected <= high_quantile), axis=1)]
    if len(core) < 100:
        raise ValueError("Projected object footprint is too sparse")
    from scipy.spatial import ConvexHull, Delaunay
    hull = ConvexHull(core)
    polygon = Delaunay(core[hull.vertices])
    low = core.min(axis=0) - 0.1
    high = core.max(axis=0) + 0.1
    shape = np.ceil((high - low) / cell).astype(int)
    if np.prod(shape) > 100000:
        raise ValueError("Object footprint is too broad for automatic refinement")
    grid = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij"), -1)
    centers = low + (grid.reshape(-1, 2) + 0.5) * cell
    footprint = (polygon.find_simplex(centers) >= 0).reshape(tuple(shape))
    footprint = ndimage.binary_dilation(footprint, iterations=6)
    within = mask_lookup(uv, footprint, low, cell)
    max_height = float(np.quantile(distance[tall], 0.995) + 0.2)
    refined_missing = (within & (distance > -0.12) & (distance < max_height))
    refined_missing |= coarse_missing & (distance > 0.15)
    details = {"refined_footprint_cells": int(footprint.sum()),
               "restored_overbroad_cut_points": int((coarse_missing & ~refined_missing).sum()),
               "new_footprint_cut_points": int((~coarse_missing & refined_missing).sum()),
               "refined_removed_points": int(refined_missing.sum())}
    return original[~refined_missing].copy(), refined_missing, details


def choose_donor(uv, distance, vertices, target, low, cell):
    # Search a bounded neighborhood. Distant parts of a room may have a
    # different material, lighting, or even a different physical plane.
    center = low + np.array(target.shape) * cell / 2
    local = np.all(np.abs(uv - center) < 4.0, axis=1)
    planar = np.abs(distance) < 0.055
    donor = local & planar & ~mask_lookup(uv, target, low, cell)
    rgb = np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]) * 0.2820947918 + 0.5
    luminance = rgb.mean(axis=1)
    # Exclude the darkest shadow/object-contaminated planar splats, but keep
    # genuine dark fibers of a textured donor patch.
    clean = donor & (luminance > 0.16) & (luminance < 1.15)
    scale_fields = [f"scale_{i}" for i in range(3)]
    if all(field in vertices.dtype.names for field in scale_fields):
        max_scale = np.exp(np.column_stack([vertices[field] for field in scale_fields])).max(axis=1)
        clean &= max_scale < 0.035
    if clean.sum() < 500:
        raise ValueError("Not enough clean nearby donor Gaussians on the fitted plane")
    donor_uv = uv[clean]
    donor_indices = np.flatnonzero(clean)
    donor_lum = luminance[clean]
    grid_low = low - 3.3
    grid_shape = np.ceil((np.array(target.shape) * cell + 6.6) / cell).astype(int)
    ij = np.floor((donor_uv - grid_low) / cell).astype(int)
    good = np.all((ij >= 0) & (ij < grid_shape), axis=1)
    ij = ij[good]
    counts = np.zeros(tuple(grid_shape), dtype=np.int32)
    brightness = np.zeros(tuple(grid_shape), dtype=np.float32)
    np.add.at(counts, (ij[:, 0], ij[:, 1]), 1)
    np.add.at(brightness, (ij[:, 0], ij[:, 1]), donor_lum[good])
    target_ij = np.argwhere(target) + np.rint((low - grid_low) / cell).astype(int)
    covered = ndimage.maximum_filter(counts > 0, size=3)
    extent = np.array(target.shape) * cell
    target_lum = float(np.median(donor_lum))
    best = None
    for step_u in range(-120, 121, 4):
        for step_v in range(-120, 121, 4):
            shift = np.array([step_u, step_v]) * cell
            if np.all(np.abs(shift) < extent + 0.12):
                continue  # Entire donor patch must be outside the hole.
            source = target_ij - np.array([step_u, step_v])
            if np.any(source < 0) or np.any(source >= grid_shape):
                continue
            local_counts = counts[source[:, 0], source[:, 1]]
            coverage = float(np.mean(covered[source[:, 0], source[:, 1]]))
            n = int(local_counts.sum())
            if n < 500:
                continue
            mean_lum = float(brightness[source[:, 0], source[:, 1]].sum() / n)
            score = coverage - 0.13 * abs(mean_lum - target_lum) - 0.005 * np.linalg.norm(shift)
            candidate = (score, coverage, n, shift, mean_lum)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None or best[1] < 0.55:
        raise ValueError("No nearby donor patch adequately covers the missing surface")
    _, coverage, _, shift, mean_lum = best
    take = mask_lookup(donor_uv + shift, target, low, cell)
    selected = donor_indices[take]
    if len(selected) < 500:
        raise ValueError("Best donor patch has too few actual Gaussians")
    return selected, shift, coverage, mean_lum


def image_file(images_dir, name):
    return next((images_dir / f"{name}{ext}" for ext in (".jpg", ".jpeg", ".png")
                if (images_dir / f"{name}{ext}").is_file()), None)


def sample_camera_colors(points, camera, image_path):
    relative = points - np.asarray(camera["position"])
    local = relative @ np.asarray(camera["rotation"])
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    x = camera["fx"] * local[:, 0] / np.maximum(local[:, 2], 1e-8) + camera["width"] / 2
    y = camera["fy"] * local[:, 1] / np.maximum(local[:, 2], 1e-8) + camera["height"] / 2
    valid = ((local[:, 2] > 0) & (x >= 2) & (y >= 2) &
             (x < rgb.shape[1] - 2) & (y < rgb.shape[0] - 2))
    colors = np.zeros((len(points), 3), dtype=np.float32)
    if valid.any():
        for channel in range(3):
            colors[valid, channel] = ndimage.map_coordinates(
                rgb[:, :, channel], [y[valid], x[valid]], order=1, mode="nearest")
    return colors, valid


def select_texture_view(points, donor_vertices, cameras_path, images_dir, preferred=None):
    with open(cameras_path, encoding="utf-8") as handle:
        cameras = json.load(handle)
    if preferred:
        cameras = [camera for camera in cameras if camera["img_name"] == preferred]
        if not cameras:
            raise ValueError(f"Donor view {preferred!r} is absent from cameras.json")
    sample_indices = np.linspace(0, len(points) - 1, min(1500, len(points)), dtype=int)
    expected = np.column_stack([donor_vertices[f"f_dc_{i}"][sample_indices]
                                for i in range(3)]) * 0.2820947918 + 0.5
    best = None
    for camera in cameras:
        path = image_file(images_dir, camera["img_name"])
        if path is None:
            continue
        colors, valid = sample_camera_colors(points[sample_indices], camera, path)
        fraction = float(valid.mean())
        if fraction < 0.90:
            continue
        # An occluding object gives image colors unlike the floor Gaussians.
        # This is also a deterministic view-selection score for an automatic run.
        discrepancy = float(np.median(np.abs(colors[valid] - expected[valid])))
        score = discrepancy + 0.5 * (1.0 - fraction)
        if best is None or score < best[0]:
            best = (score, camera, path)
    if best is None:
        raise ValueError("No source image sees enough of the donor patch")
    _, camera, path = best
    colors, valid = sample_camera_colors(points, camera, path)
    if valid.mean() < 0.9:
        raise ValueError("Selected source image has insufficient donor coverage")
    return camera, path, best[0]


def build_candidate(original, pruned, missing, cameras_path, images_dir, *, seed=0,
                    cell=0.025, grid_step=0.006, repair_margin=0.05,
                    cleanup_height=0.18, donor_view=None,
                    extra_repair_mask=None, extra_repair_view=None,
                    extra_mask_margin=0.05):
    xyz = xyz_of(original)
    kept_xyz = xyz_of(pruned)
    origin, normal, u, v, plane_ratio = fit_plane(kept_xyz, xyz[missing], seed=seed)
    orig_uv, orig_dist = plane_coordinates(xyz, origin, normal, u, v)
    kept_uv, kept_dist = plane_coordinates(kept_xyz, origin, normal, u, v)
    target, low, cell, basis = make_target(orig_uv[missing], orig_dist[missing],
                                          cell, repair_margin)
    extra_cells = 0
    if extra_repair_mask:
        target, low, extra_cells = extend_target_from_view_mask(
            target, low, cell, extra_repair_mask, extra_repair_view,
            cameras_path, origin, normal, u, v, extra_mask_margin)
    selected, shift, coverage, donor_lum = choose_donor(kept_uv, kept_dist, pruned, target, low, cell)

    # Remove existing shadow/base splats inside the repair region. Background
    # farther from the plane is deliberately left alone.
    affected = mask_lookup(kept_uv, target, low, cell)
    if cleanup_height <= 0 or cleanup_height > 1.0:
        raise ValueError("--cleanup-height must be between 0 and 1.0")
    replace = affected & (kept_dist > -0.18) & (kept_dist < cleanup_height)
    camera, image_path, view_score = select_texture_view(
        kept_xyz[selected], pruned[selected], cameras_path, images_dir, donor_view)
    texture_view = camera["img_name"]
    if grid_step <= 0 or grid_step > cell:
        raise ValueError("--grid-step must be positive and no larger than --cell")
    n_fine = np.ceil(np.array(target.shape) * cell / grid_step).astype(int)
    if np.prod(n_fine) > 1000000:
        raise ValueError("Texture grid is too large; increase --grid-step")
    mesh = np.stack(np.meshgrid(np.arange(n_fine[0]), np.arange(n_fine[1]), indexing="ij"), -1)
    fill_uv = low + (mesh.reshape(-1, 2) + 0.5) * grid_step
    fill_uv = fill_uv[mask_lookup(fill_uv, target, low, cell)]
    source_uv = fill_uv - shift
    donor_xyz = origin + source_uv[:, 0, None] * u + source_uv[:, 1, None] * v
    sampled, valid = sample_camera_colors(donor_xyz, camera, image_path)
    if valid.mean() < 0.95:
        raise ValueError("Donor texture image does not cover enough of the synthesized patch")
    fill_uv = fill_uv[valid]
    sampled = sampled[valid]
    padded = np.pad(target, 20)
    ring_mask = ndimage.binary_dilation(padded, iterations=16) & ~ndimage.binary_dilation(padded, iterations=4)
    ring_points = mask_lookup(kept_uv, ring_mask, low - 20 * cell, cell) & (np.abs(kept_dist) < 0.055)
    ring_indices = np.flatnonzero(ring_points)
    if len(ring_indices) >= 100:
        ring_indices = ring_indices[np.linspace(0, len(ring_indices) - 1,
                                                min(2500, len(ring_indices)), dtype=int)]
        ring_colors, ring_valid = sample_camera_colors(kept_xyz[ring_indices], camera, image_path)
        if ring_valid.sum() >= 100:
            color_delta = np.median(ring_colors[ring_valid], axis=0) - np.median(sampled, axis=0)
            color_delta = np.clip(color_delta, -0.15, 0.15)
            sampled = np.clip(sampled + color_delta, 0.0, 1.0)
        else:
            color_delta = np.zeros(3)
    else:
        color_delta = np.zeros(3)
    from scipy.spatial import cKDTree
    nearest = cKDTree(kept_uv[selected]).query(fill_uv - shift, k=1)[1]
    clones = pruned[selected[nearest]].copy()
    translated = origin + fill_uv[:, 0, None] * u + fill_uv[:, 1, None] * v + 0.006 * normal
    for i, axis in enumerate(("x", "y", "z")):
        clones[axis] = translated[:, i]
        clones[f"f_dc_{i}"] = (sampled[:, i] - 0.5) / 0.2820947918
    for field in clones.dtype.names:
        if field.startswith("f_rest_"):
            clones[field] = 0.0
    # The copied semantic descriptors remain carpet. Only geometry and RGB
    # are synthesized on the fitted plane.
    for i in range(3):
        clones[f"scale_{i}"] = np.log(grid_step * 1.10)
    clones["rot_0"] = 1.0
    for i in range(1, 4):
        clones[f"rot_{i}"] = 0.0
    cell_distance = ndimage.distance_transform_edt(target) * cell
    clone_ij = np.floor((fill_uv - low) / cell).astype(int)
    fade = np.clip(cell_distance[clone_ij[:, 0], clone_ij[:, 1]] / 0.07, 0.12, 1.0)
    opacity = np.clip(0.82 * fade, 0.03, 0.98)
    clones["opacity"] = np.log(opacity / (1 - opacity)).astype(np.float32)
    if "object_id" in clones.dtype.names:
        clones["object_id"] = 0
    candidate = np.concatenate((pruned[~replace], clones))
    details = {
        "plane_origin": origin.tolist(), "plane_normal_toward_removed_object": normal.tolist(),
        "plane_inlier_ratio": plane_ratio, "footprint_basis": basis,
        "target_cells": int(target.sum()), "target_cell_size": cell,
        "donor_shift_uv": shift.tolist(), "donor_coverage": coverage,
        "donor_mean_luminance": donor_lum, "removed_residual_surface_points": int(replace.sum()),
        "added_donor_gaussians": int(len(clones)), "candidate_points": int(len(candidate)),
        "donor_texture_view": texture_view, "donor_view_score": view_score,
        "texture_grid_step": grid_step,
        "repair_margin": repair_margin,
        "cleanup_height": cleanup_height,
        "extra_repair_cells": extra_cells,
        "extra_mask_margin": extra_mask_margin,
        "texture_color_delta": color_delta.tolist(),
    }
    return candidate, details


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preview(args):
    if not 0 < args.cell <= 0.2:
        raise ValueError("--cell must be between 0 and 0.2")
    if args.render_width <= 0:
        raise ValueError("--render-width must be positive")
    original_path = Path(args.original).resolve()
    pruned_path = Path(args.pruned).resolve()
    output_dir = Path(args.output_dir).resolve()
    cameras_path = Path(args.cameras).resolve()
    images_dir = Path(args.images).resolve()
    if not cameras_path.is_file() or not images_dir.is_dir():
        raise ValueError("Existing cameras.json and source image directory are required")
    with open(cameras_path, encoding="utf-8") as handle:
        cameras = {camera["img_name"] for camera in json.load(handle)}
    if any(name not in cameras for name in args.views):
        raise ValueError("Every --views entry must exist in cameras.json")
    if bool(args.extra_repair_mask) != bool(args.extra_repair_view):
        raise ValueError("--extra-repair-mask and --extra-repair-view must be supplied together")
    if args.extra_repair_mask and not Path(args.extra_repair_mask).is_file():
        raise ValueError("Extra repair mask file does not exist")
    for name in args.views:
        if image_file(images_dir, name) is None:
            raise ValueError(f"Source image missing for {name}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("--output-dir must be absent or empty; refusing to overwrite a preview")
    original_ply, original = read_vertices(original_path)
    _, pruned = read_vertices(pruned_path)
    missing = removed_indices(original, pruned)
    refinement = None
    if args.refine_mask:
        pruned, missing, refinement = refine_object_footprint(
            original, pruned, missing, seed=args.seed, cell=args.cell)
    candidate, details = build_candidate(original, pruned, missing, cameras_path, images_dir,
                                         seed=args.seed, cell=args.cell, grid_step=args.grid_step,
                                         repair_margin=args.repair_margin,
                                         cleanup_height=args.cleanup_height,
                                         donor_view=args.donor_view,
                                         extra_repair_mask=args.extra_repair_mask,
                                         extra_repair_view=args.extra_repair_view,
                                         extra_mask_margin=args.extra_mask_margin)
    if refinement:
        details.update(refinement)
    if args.dry_run:
        print(json.dumps(details, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.extra_repair_mask:
        shutil.copy2(args.extra_repair_mask, output_dir / "extra-repair-mask.png")
        details["extra_repair_mask"] = "extra-repair-mask.png"
        details["extra_repair_view"] = args.extra_repair_view
    candidate_path = output_dir / "candidate.ply"
    write_vertices(candidate_path, candidate, original_ply,
                   ["unapproved flat-surface reconstruction preview"])
    if refinement:
        write_vertices(output_dir / "refined-pruned.ply", pruned, original_ply,
                       ["unapproved plane-aware object removal preview"])
    manifest = read_object_manifest(pruned_path)
    write_object_manifest(candidate_path, manifest)
    details.update({"original": str(original_path), "pruned": str(pruned_path),
                    "cameras": str(cameras_path), "images": str(images_dir),
                    "views": args.views, "candidate_sha256": sha256(candidate_path),
                    "removed_original_points": int(missing.sum()), "approved": False})
    with open(output_dir / "preview.json", "w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)
        handle.write("\n")
    if not args.skip_render:
        metrics = {}
        for name in args.views:
            render_path = output_dir / f"{name}.png"
            subprocess.run([sys.executable, str(Path(__file__).with_name("render_ply_preview.py")),
                            "--ply", str(candidate_path), "--cameras", str(cameras_path),
                            "--image-name", name, "--output", str(render_path),
                            "--width", str(args.render_width)], check=True)
            if refinement:
                subprocess.run([sys.executable, str(Path(__file__).with_name("render_ply_preview.py")),
                                "--ply", str(output_dir / "refined-pruned.ply"),
                                "--cameras", str(cameras_path), "--image-name", name,
                                "--output", str(output_dir / f"{name}-pruned.png"),
                                "--width", str(args.render_width)], check=True)
            source_path = image_file(images_dir, name)
            with Image.open(source_path) as source, Image.open(render_path) as rendered:
                reference = source.convert("RGB").resize(rendered.size)
                comparison = Image.new("RGB", (rendered.width * 2, rendered.height))
                comparison.paste(reference, (0, 0))
                comparison.paste(rendered, (rendered.width, 0))
                comparison.save(output_dir / f"{name}-compare.png")
            if refinement:
                with Image.open(render_path) as rendered, Image.open(output_dir / f"{name}-pruned.png") as before:
                    after_rgb = np.asarray(rendered.convert("RGB"), dtype=np.float32) / 255
                    before_rgb = np.asarray(before.convert("RGB"), dtype=np.float32) / 255
                changed = ndimage.binary_dilation(
                    np.mean(np.abs(after_rgb - before_rgb), axis=2) > 0.08, iterations=8)
                metrics[name] = {
                    "changed_pixels": int(changed.sum()),
                    "very_dark_before": int(((before_rgb.mean(axis=2) < 0.15) & changed).sum()),
                    "very_dark_after": int(((after_rgb.mean(axis=2) < 0.15) & changed).sum()),
                }
        if metrics:
            details["view_metrics"] = metrics
            with open(output_dir / "preview.json", "w", encoding="utf-8") as handle:
                json.dump(details, handle, indent=2)
                handle.write("\n")
    print(json.dumps(details, indent=2))
    print(f"Preview (not approved): {candidate_path}")


def commit(args):
    directory = Path(args.preview_dir).resolve()
    with open(directory / "preview.json", encoding="utf-8") as handle:
        details = json.load(handle)
    candidate = directory / "candidate.ply"
    output = Path(args.output).resolve()
    if not args.approve:
        raise ValueError("Commit requires --approve after visual review of the preview renders")
    if not details.get("views") or not all((directory / f"{name}.png").is_file()
                                           for name in details["views"]):
        raise ValueError("Preview renders are missing; review all requested views before commit")
    if sha256(candidate) != details["candidate_sha256"]:
        raise ValueError("Candidate PLY changed after preview; regenerate the preview")
    if (output.exists() or output.is_symlink() or output.with_suffix(".objects.json").exists()
            or output in (candidate.resolve(), Path(details["original"]), Path(details["pruned"]))):
        raise ValueError("Refusing to overwrite an existing or source PLY")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, output)
    sidecar = candidate.with_suffix(".objects.json")
    if sidecar.is_file():
        shutil.copy2(sidecar, output.with_suffix(".objects.json"))
    print(f"Approved scene: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("preview", help="Generate an unapproved PLY and multiview renders")
    prepare.add_argument("--original", required=True)
    prepare.add_argument("--pruned", required=True)
    prepare.add_argument("--cameras", required=True)
    prepare.add_argument("--images", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--views", nargs="+", required=True)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--cell", type=float, default=0.025)
    prepare.add_argument("--grid-step", type=float, default=0.006)
    prepare.add_argument("--repair-margin", type=float, default=0.05,
                         help="Plane distance beyond removed floor splats to repair; increase for baked shadows")
    prepare.add_argument("--cleanup-height", type=float, default=0.18,
                         help="Remove residual splats this far above the repair plane")
    prepare.add_argument("--extra-repair-mask", help="Reviewed grayscale PNG marking additional floor/shadow pixels")
    prepare.add_argument("--extra-repair-view", help="Camera name matching --extra-repair-mask")
    prepare.add_argument("--extra-mask-margin", type=float, default=0.05,
                         help="Plane-space dilation around the extra repair mask")
    prepare.add_argument("--donor-view", help="Optional source camera for donor texture; auto-select by default")
    prepare.add_argument("--refine-mask", action="store_true",
                         help="Tighten an overbroad box cut to the object's plane-projected footprint")
    prepare.add_argument("--render-width", type=int, default=540)
    prepare.add_argument("--skip-render", action="store_true", help="For tests only; cannot commit")
    prepare.add_argument("--dry-run", action="store_true", help="Report geometry without writing")
    finalize = commands.add_parser("commit", help="Copy a reviewed preview to a new final PLY")
    finalize.add_argument("--preview-dir", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    if args.command == "preview":
        preview(args)
    else:
        commit(args)


if __name__ == "__main__":
    main()
