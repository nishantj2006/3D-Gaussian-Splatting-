"""Make explicit wall/floor image guides for masked key views.

The wall may be an uncertain *2D-only* hypothesis. These guides are never
accepted as 3D geometry or training data without separate validation.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

from gsedit.generation.generate_surface_atlas import backproject_plane
from gsedit.generation.inpaint_key_views import prepare_mask, resized_size
from utils.ply_semantic_utils import read_vertices


SH_C0 = .28209479177387814


def assign_surfaces(camera, pixels, shape, floor, wall):
    floor_origin = np.asarray(floor["plane_origin"], float)
    floor_normal = np.asarray(floor["plane_normal_toward_removed_object"], float)
    frame = np.asarray(wall["frame"], float)
    origin = np.asarray(wall["origin"], float)
    xy_normal = np.asarray(wall["wall_normal_floor_xy"], float)
    wall_normal = frame[:, :2] @ xy_normal
    wall_origin = origin + wall_normal*float(wall["wall_offset"])
    floor_xyz, floor_depth = backproject_plane(camera, pixels, shape,
                                               floor_origin, floor_normal)
    wall_xyz, wall_depth = backproject_plane(camera, pixels, shape,
                                             wall_origin, wall_normal)
    wall_height = (wall_xyz-origin) @ frame[:, 2]
    floor_ok = np.isfinite(floor_depth) & (floor_depth > 0) & (floor_depth < 50)
    wall_ok = (np.isfinite(wall_depth) & (wall_depth > 0) &
               (wall_depth < 50) & (wall_height >= -.05))
    use_floor = floor_ok & (~wall_ok | (floor_depth <= wall_depth))
    if not np.any(use_floor) or not np.any(~use_floor):
        raise ValueError(f"Wall/floor hypothesis gives no boundary: "
                         f"floor={use_floor.mean():.1%}, wall-valid={wall_ok.mean():.1%}, "
                         f"floor-valid={floor_ok.mean():.1%}")
    return floor_xyz, use_floor


def donor_colors(vertices, indices):
    xyz = np.column_stack([vertices[k][indices] for k in ("x", "y", "z")])
    colors = np.clip(np.column_stack([vertices[f"f_dc_{j}"][indices]
                                        for j in range(3)])*SH_C0+.5, 0, 1)
    return xyz, colors


def observed_floor_bound(camera, target_mask, floor_mask, floor, wall, *,
                         context_px, min_samples):
    near = distance_transform_edt(~target_mask) <= context_px
    visible = floor_mask & ~target_mask & near
    y, x = np.where(visible)
    if len(x) < min_samples:
        raise ValueError("Too few observed floor pixels beside target")
    xyz, depth = backproject_plane(camera, np.column_stack((x, y)),
                                    target_mask.shape,
                                    np.asarray(floor["plane_origin"], float),
                                    np.asarray(floor["plane_normal_toward_removed_object"], float))
    frame = np.asarray(wall["frame"], float)
    normal = np.asarray(wall["wall_normal_floor_xy"], float)
    q = ((xyz-np.asarray(wall["origin"], float)) @ frame)[:, :2] @ normal
    valid = np.isfinite(depth) & (depth > 0) & (depth < 50)
    if valid.sum() < min_samples:
        raise ValueError("Too few depth-valid floor pixels beside target")
    return float(np.quantile(q[valid], .99)), int(valid.sum())


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    with open(args.floor_fit, encoding="utf-8") as handle:
        floor = json.load(handle)
    with open(args.wall_fit, encoding="utf-8") as handle:
        wall = json.load(handle)
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    with open(args.target_manifest, encoding="utf-8") as handle:
        target = json.load(handle)["views"]
    with open(args.wall_manifest, encoding="utf-8") as handle:
        wall_masks = json.load(handle)["views"]
    with open(args.floor_manifest, encoding="utf-8") as handle:
        floor_masks = json.load(handle)["views"]
    _, vertices = read_vertices(args.scene)
    ids = np.load(args.floor_donors, allow_pickle=False)
    if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= len(vertices):
        raise ValueError("Invalid floor donor indices")
    xyz, colors = donor_colors(vertices, ids)
    tree = cKDTree(xyz)
    bounds = {}
    for view in args.views:
        floor_entry = floor_masks.get(view, {})
        if not floor_entry.get("accepted") or view not in cameras:
            raise ValueError(f"No accepted floor mask/camera for {view}")
        with Image.open(next(p for p in Path(args.images).glob(view + ".*")
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))) as source:
            size = resized_size(source, args.width)
        target_mask = prepare_mask(target[view]["mask_path"], size,
                                   args.close_px, args.dilate_px)
        floor_mask = np.asarray(Image.open(floor_entry["mask_path"]).convert("L")
                                .resize(size, Image.Resampling.NEAREST)) > 127
        bounds[view] = observed_floor_bound(cameras[view], target_mask, floor_mask,
                                            floor, wall, context_px=args.floor_context_px,
                                            min_samples=args.min_floor_pixels)
    old_offset = wall["wall_offset"]
    wall["wall_offset"] = max(old_offset, max(value[0] for value in bounds.values()) +
                              args.wall_clearance)
    output.mkdir(parents=True)
    report = {"approved": False, "geometry_supported": False,
              "warning": "Speculative wall plane; 2D diagnostic only",
              "wall_inlier_ratio": wall["inlier_ratio"],
              "original_wall_offset": old_offset,
              "guide_wall_offset": wall["wall_offset"],
              "observed_floor_bounds": bounds, "views": {}}
    for view in args.views:
        if not target.get(view, {}).get("accepted") or view not in cameras:
            raise ValueError(f"No accepted target mask/camera for {view}")
        paths = [p for p in Path(args.images).glob(view + ".*") if
                 p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if len(paths) != 1:
            raise ValueError(f"Expected one photo for {view}")
        original = Image.open(paths[0]).convert("RGB")
        size = resized_size(original, args.width)
        photo = np.asarray(original.resize(size, Image.Resampling.LANCZOS))
        mask = prepare_mask(target[view]["mask_path"], size,
                            args.close_px, args.dilate_px)
        wall_entry = wall_masks.get(view, {})
        if not wall_entry.get("accepted"):
            raise ValueError(f"No clean observed wall mask for {view}")
        observed_wall = np.asarray(Image.open(wall_entry["mask_path"])
                                   .convert("L").resize(size, Image.Resampling.NEAREST)) > 127
        observed_wall &= ~mask
        if observed_wall.sum() < args.min_wall_pixels:
            raise ValueError(f"Too few observed wall pixels in {view}")
        wall_rgb = np.median(photo[observed_wall], axis=0)
        y, x = np.where(mask)
        points = np.column_stack((x, y))
        floor_xyz, floor_choice = assign_surfaces(cameras[view], points,
                                                  (size[1], size[0]), floor, wall)
        guide = photo.copy()
        layout = np.zeros((*mask.shape, 3), np.uint8)
        layout[mask] = (150, 80, 30)
        layout[y[floor_choice], x[floor_choice]] = (50, 170, 50)
        guide[y[~floor_choice], x[~floor_choice]] = wall_rgb.astype(np.uint8)
        floor_queries = floor_xyz[floor_choice]
        distances, nearest = tree.query(floor_queries, k=args.neighbors, workers=-1)
        if args.neighbors == 1:
            distances, nearest = distances[:, None], nearest[:, None]
        weights = 1/np.maximum(distances, .05)**2
        floor_rgb = np.clip((colors[nearest]*weights[:, :, None]).sum(axis=1) /
                            weights.sum(axis=1)[:, None], 0, 1)
        distance_q90 = float(np.quantile(distances[:, 0], .9))
        floor_source = "gaussian_donors"
        if distance_q90 > args.max_3d_donor_distance:
            floor_entry = floor_masks[view]
            visible_floor = np.asarray(Image.open(floor_entry["mask_path"])
                                       .convert("L").resize(size, Image.Resampling.NEAREST)) > 127
            visible_floor &= ~mask
            brightness = photo.astype(np.float32).mean(axis=2)
            median_brightness = float(np.median(brightness[visible_floor]))
            visible_floor &= brightness >= .65 * median_brightness
            if visible_floor.sum() < args.min_floor_pixels:
                raise ValueError(f"Too little image floor texture for {view}")
            _, nearest_pixel = distance_transform_edt(~visible_floor,
                                                       return_indices=True)
            fy, fx = y[floor_choice], x[floor_choice]
            guide[fy, fx] = photo[nearest_pixel[0, fy, fx], nearest_pixel[1, fy, fx]]
            floor_source = "nearest_visible_photo_carpet"
        else:
            guide[y[floor_choice], x[floor_choice]] = (floor_rgb*255).astype(np.uint8)
        Image.fromarray(guide).save(output / f"{view}-guide.png")
        Image.fromarray(layout).save(output / f"{view}-layout.png")
        report["views"][view] = {"floor_fraction_of_mask": float(floor_choice.mean()),
                                  "floor_donor_distance_q90": distance_q90,
                                  "floor_guide_source": floor_source,
                                  "observed_wall_pixels": int(observed_wall.sum()),
                                  "guide": str(output / f"{view}-guide.png")}
    report["elapsed_seconds"] = time.perf_counter()-started
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "images", "cameras", "target-manifest", "wall-manifest",
                 "floor-manifest", "floor-fit", "floor-donors", "wall-fit", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--close-px", type=int, default=5)
    p.add_argument("--dilate-px", type=int, default=2)
    p.add_argument("--neighbors", type=int, default=8)
    p.add_argument("--min-wall-pixels", type=int, default=100)
    p.add_argument("--min-floor-pixels", type=int, default=100)
    p.add_argument("--floor-context-px", type=float, default=30.)
    p.add_argument("--wall-clearance", type=float, default=.3)
    p.add_argument("--max-3d-donor-distance", type=float, default=.5)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
