"""Generate image-first guides using one cross-view carpet atlas.

This is an unapproved 2D diagnostic.  The wall is an explicit hypothesis, not
validated 3D geometry.  Held-out views may be rendered but never feed the atlas.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from PIL import Image

from gsedit.reconstruction.build_imagefirst_guides import assign_surfaces
from gsedit.reconstruction.carpet_texture import transfer_texture
from gsedit.generation.generate_surface_atlas import backproject_plane
from gsedit.generation.inpaint_key_views import prepare_mask, resized_size
from gsedit.reconstruction.wall_color_field import predict_wall_colors
from utils.ply_semantic_utils import read_vertices


def target_roi(scene, indices, origin, basis, margin):
    _, vertices = read_vertices(scene)
    ids = np.load(indices, allow_pickle=False)
    if ids.ndim != 1 or len(ids) < 100 or ids.min() < 0 or ids.max() >= len(vertices):
        raise ValueError("Invalid target-region Gaussian IDs")
    xyz = np.column_stack([vertices[k][ids] for k in ("x", "y", "z")])
    uv = (xyz-origin) @ basis
    low = np.quantile(uv, .02, axis=0)-margin
    high = np.quantile(uv, .98, axis=0)+margin
    if np.max(high-low) > 100 or np.any(high <= low):
        raise ValueError("Target region is too broad for a floor atlas")
    return low, high


def image_for(images, view, size):
    paths = [p for p in Path(images).glob(view + ".*")
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(paths) != 1:
        raise ValueError(f"Expected one photo for {view}")
    return np.asarray(Image.open(paths[0]).convert("RGB").resize(
        size, Image.Resampling.LANCZOS))


def mask_for(entries, view, size):
    item = entries.get(view, {})
    if not item.get("accepted"):
        return None
    return np.asarray(Image.open(item["mask_path"]).convert("L").resize(
        size, Image.Resampling.NEAREST)) > 127


def atlas_coordinates(xyz, origin, basis, low, high, shape):
    uv = (xyz-origin) @ basis
    x = (uv[:, 0]-low[0])/(high[0]-low[0])*(shape[1]-1)
    y = (uv[:, 1]-low[1])/(high[1]-low[1])*(shape[0]-1)
    return x, y


def fill_atlas(rgb, known, backend, model, steps, seed):
    if backend == "texture":
        return transfer_texture(rgb, known)
    missing = (~known).astype(np.uint8)
    if backend == "opencv":
        return cv2.inpaint(rgb, missing, 7, cv2.INPAINT_TELEA)
    from gsedit.generation.generate_background_views import inpaint_diffusion, load_diffusion
    pipe = load_diffusion(model)
    generated = inpaint_diffusion(pipe, rgb, missing.astype(bool),
        "Seamless carpet texture matching the visible carpet, no furniture, "
        "no objects, consistent lighting", "bed, wooden frame, seams, holes",
        steps, seed)
    generated[known] = rgb[known]
    return generated


def build_atlas(samples, targets, *, resolution, margin, backend, model,
                steps, seed, min_observed_fraction, roi=None):
    if sum(len(points) for points in targets) < 100:
        raise ValueError("Too few target floor rays")
    target_uv = np.concatenate(targets)
    if roi is None:
        low = np.quantile(target_uv, .01, axis=0)-margin
        high = np.quantile(target_uv, .99, axis=0)+margin
    else:
        low, high = roi
    if np.any(high <= low) or np.max(high-low) > 100:
        raise ValueError("Unsupported floor atlas bounds")
    span = high-low
    shape = (max(64, round(resolution*span[1]/span.max()/8)*8),
             max(64, round(resolution*span[0]/span.max()/8)*8))
    sums = np.zeros((*shape, 3), np.float64)
    counts = np.zeros(shape, np.int32)
    support = {}
    for view, (uv, colors) in samples.items():
        x = np.rint((uv[:, 0]-low[0])/span[0]*(shape[1]-1)).astype(int)
        y = np.rint((uv[:, 1]-low[1])/span[1]*(shape[0]-1)).astype(int)
        valid = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
        np.add.at(sums, (y[valid], x[valid]), colors[valid])
        np.add.at(counts, (y[valid], x[valid]), 1)
        support[view] = int(valid.sum())
    known = counts > 0
    if known.mean() < min_observed_fraction or sum(v >= 100 for v in support.values()) < 2:
        raise ValueError(f"Sparse carpet donors: {known.mean():.1%} observed")
    observed = np.zeros((*shape, 3), np.uint8)
    observed[known] = np.rint(sums[known]/counts[known, None]).astype(np.uint8)
    filled = fill_atlas(observed, known, backend, model, steps, seed)
    return filled, observed, known, low, high, support


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    with open(args.target_manifest, encoding="utf-8") as handle:
        target = json.load(handle)["views"]
    with open(args.floor_manifest, encoding="utf-8") as handle:
        floor_masks = json.load(handle)["views"]
    with open(args.wall_manifest, encoding="utf-8") as handle:
        wall_masks = json.load(handle)["views"]
    with open(args.floor_fit, encoding="utf-8") as handle:
        floor = json.load(handle)
    with open(args.wall_fit, encoding="utf-8") as handle:
        wall = json.load(handle)
    true_offset = wall["wall_offset"]
    if args.guide_report:
        with open(args.guide_report, encoding="utf-8") as handle:
            wall["wall_offset"] = json.load(handle)["guide_wall_offset"]
    origin = np.asarray(floor["plane_origin"], float)
    normal = np.asarray(floor["plane_normal_toward_removed_object"], float)
    basis = np.asarray(wall["frame"], float)[:, :2]
    if bool(args.scene) != bool(args.roi_indices):
        raise ValueError("--scene and --roi-indices must be given together")
    roi = (target_roi(args.scene, args.roi_indices, origin, basis, args.roi_margin)
           if args.scene else None)
    train = sorted(set(args.training_views)-set(args.holdout_views))
    if len(train) < 2 or any(v not in cameras for v in args.views + train):
        raise ValueError("Too few training cameras")
    samples, targets = {}, []
    for view in train:
        camera = cameras[view]
        with Image.open(next(p for p in Path(args.images).glob(view + ".*")
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))) as img:
            size = resized_size(img, args.width)
        image = image_for(args.images, view, size)
        target_mask = mask_for(target, view, size)
        floor_mask = mask_for(floor_masks, view, size)
        if target_mask is None or floor_mask is None:
            continue
        yy, xx = np.where(target_mask)
        floor_xyz, floor_choice = assign_surfaces(camera, np.column_stack((xx, yy)),
                                                  image.shape[:2], floor, wall)
        if roi is not None:
            uv = (floor_xyz-origin) @ basis
            floor_choice &= np.all((uv >= roi[0]) & (uv <= roi[1]), axis=1)
        if floor_choice.any():
            targets.append((floor_xyz[floor_choice]-origin) @ basis)
        visible = floor_mask & ~target_mask
        yy, xx = np.where(visible)
        if len(xx) > args.max_pixels_per_view:
            chosen = np.linspace(0, len(xx)-1, args.max_pixels_per_view).astype(int)
            yy, xx = yy[chosen], xx[chosen]
        points, depth = backproject_plane(camera, np.column_stack((xx, yy)),
                                          image.shape[:2], origin, normal)
        colors = image[yy, xx]
        bright = colors.mean(axis=1)
        valid = np.isfinite(depth) & (depth > 0) & (depth < 50)
        if valid.any():
            valid &= bright >= args.min_brightness_fraction*np.median(bright[valid])
        samples[view] = ((points[valid]-origin) @ basis, colors[valid])
    atlas, observed, known, low, high, support = build_atlas(
        samples, targets, resolution=args.resolution, margin=args.margin,
        backend=args.backend, model=args.model, steps=args.steps, seed=args.seed,
        min_observed_fraction=args.min_observed_fraction, roi=roi)
    output.mkdir(parents=True)
    Image.fromarray(atlas).save(output / "floor-atlas.png")
    Image.fromarray(observed).save(output / "floor-observed.png")
    Image.fromarray((known*255).astype(np.uint8)).save(output / "floor-evidence.png")
    view_report = {}
    for view in args.views:
        camera = cameras[view]
        with Image.open(next(p for p in Path(args.images).glob(view + ".*")
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))) as img:
            size = resized_size(img, args.width)
        photo = image_for(args.images, view, size)
        mask = mask_for(target, view, size)
        observed_wall = mask_for(wall_masks, view, size)
        if mask is None or observed_wall is None:
            raise ValueError(f"Missing accepted target/wall mask in {view}")
        mask = prepare_mask(target[view]["mask_path"], size,
                            args.close_px, args.dilate_px)
        observed_wall &= ~mask
        if observed_wall.sum() < args.min_wall_pixels:
            raise ValueError(f"Too little visible wall in {view}")
        yy, xx = np.where(mask)
        xyz, is_floor = assign_surfaces(camera, np.column_stack((xx, yy)),
                                       photo.shape[:2], floor, wall)
        if roi is not None:
            uv = (xyz-origin) @ basis
            is_floor &= np.all((uv >= roi[0]) & (uv <= roi[1]), axis=1)
        guide = photo.copy()
        guide[yy[~is_floor], xx[~is_floor]] = predict_wall_colors(
            photo, observed_wall, yy[~is_floor], xx[~is_floor])
        qx, qy = atlas_coordinates(xyz[is_floor], origin, basis, low, high,
                                   atlas.shape[:2])
        inside = (qx >= 0) & (qx <= atlas.shape[1]-1) & (qy >= 0) & (qy <= atlas.shape[0]-1)
        if inside.mean() < args.min_target_coverage:
            raise ValueError(f"Atlas misses carpet target in {view}: {inside.mean():.1%}")
        rgb = cv2.remap(atlas, qx.astype(np.float32).reshape(1, -1),
                        qy.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)[0]
        fy, fx = yy[is_floor], xx[is_floor]
        guide[fy, fx] = rgb
        Image.fromarray(guide).save(output / f"{view}-guide.png")
        layout = np.zeros_like(photo)
        layout[mask] = [150, 80, 30]
        layout[fy, fx] = [50, 170, 50]
        Image.fromarray(layout).save(output / f"{view}-layout.png")
        view_report[view] = {"floor_fraction": float(is_floor.mean()),
                             "floor_atlas_coverage": float(inside.mean()),
                             "guide": str(output / f"{view}-guide.png")}
    report = {"approved": False, "geometry_supported": False,
              "warning": "2D guides use an unvalidated wall hypothesis; never train 3D from them",
              "original_wall_offset": true_offset,
              "guide_wall_offset": wall["wall_offset"],
              "wall_inlier_ratio": wall["inlier_ratio"],
              "training_views": train, "holdout_views": args.holdout_views,
              "roi_source_ids": str(Path(args.roi_indices).resolve()) if roi else None,
              "floor_evidence_fraction": float(known.mean()),
              "floor_photo_support": support, "uv_low": low.tolist(),
              "uv_high": high.tolist(), "views": view_report,
              "elapsed_seconds": time.perf_counter()-started,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "cameras", "target-manifest", "floor-manifest",
                 "wall-manifest", "floor-fit", "wall-fit", "output-dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--guide-report")
    p.add_argument("--scene")
    p.add_argument("--roi-indices")
    p.add_argument("--roi-margin", type=float, default=1.5)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--training-views", nargs="+", required=True)
    p.add_argument("--holdout-views", nargs="+", default=[])
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--resolution", type=int, default=768)
    p.add_argument("--margin", type=float, default=1.5)
    p.add_argument("--backend", choices=("opencv", "diffusion", "texture"), default="texture")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pixels-per-view", type=int, default=30000)
    p.add_argument("--min-observed-fraction", type=float, default=.005)
    p.add_argument("--min-target-coverage", type=float, default=.95)
    p.add_argument("--min-wall-pixels", type=int, default=100)
    p.add_argument("--min-brightness-fraction", type=float, default=.65)
    p.add_argument("--close-px", type=int, default=5)
    p.add_argument("--dilate-px", type=int, default=2)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
