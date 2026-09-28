"""Synthesize one shared wall/floor texture atlas and apply it to new Gaussians.

Original photos contribute only pixels whose observed depth agrees with the
surface and lie outside the target mask. Inpainting fills the remaining atlas
texels once; every view then uses the same 3D appearance. Preview only.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from PIL import Image
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from gsedit.generation.generate_background_views import SH_C0, inpaint_diffusion, load_diffusion
from scene.gaussian_model import GaussianModel
from utils.ply_semantic_utils import read_vertices, write_vertices


def surface_basis(fit, kind):
    origin = np.asarray(fit["origin"], dtype=np.float64)
    frame = np.asarray(fit["frame"], dtype=np.float64)
    if kind == "floor":
        return origin, frame[:, :2], frame[:, 2]
    normal = np.asarray(fit["wall_normal_floor_xy"], dtype=np.float64)
    tangent = np.array([-normal[1], normal[0]])
    basis = np.column_stack((frame[:, :2] @ tangent, frame[:, 2]))
    wall_origin = origin + (frame[:, :2] @ normal)*fit["wall_offset"]
    return wall_origin, basis, frame[:, :2] @ normal


def atlas_layout(xyz, origin, basis, *, resolution, margin):
    uv = (xyz-origin) @ basis
    low = uv.min(axis=0)-margin
    high = uv.max(axis=0)+margin
    span = high-low
    if np.any(span <= 0) or span.max() > 100:
        raise ValueError("Unsupported surface atlas bounds")
    shape = np.maximum(64, np.ceil(span/span.max()*resolution/8).astype(int)*8)
    width, height = int(shape[0]), int(shape[1])
    return low, high, (height, width)


def uv_pixels(uv, low, high, shape):
    height, width = shape
    u = (uv[:, 0]-low[0])/(high[0]-low[0])*(width-1)
    v = (uv[:, 1]-low[1])/(high[1]-low[1])*(height-1)
    return u, v


def backproject_plane(camera, pixels, shape, origin, normal):
    height, width = shape
    xy = np.asarray(pixels, dtype=np.float64)
    fx = camera["fx"]*width/camera["width"]
    fy = camera["fy"]*height/camera["height"]
    rays = np.column_stack(((xy[:, 0]-width/2)/fx,
                            (xy[:, 1]-height/2)/fy, np.ones(len(xy))))
    rays = rays @ np.asarray(camera["rotation"], dtype=np.float64).T
    center = np.asarray(camera["position"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = ((origin-center) @ normal)/(rays @ normal)
    xyz = center + np.where(np.isfinite(depth), depth, 0)[:, None]*rays
    return xyz, depth


def accumulate_atlas(sums, counts, uv, colors, low, high):
    u, v = uv_pixels(uv, low, high, counts.shape)
    x, y = np.rint(u).astype(int), np.rint(v).astype(int)
    inside = (x >= 0) & (x < counts.shape[1]) & (y >= 0) & (y < counts.shape[0])
    np.add.at(sums, (y[inside], x[inside]), colors[inside])
    np.add.at(counts, (y[inside], x[inside]), 1)
    return int(inside.sum())


def complete_atlas(observed_rgb, observed_mask, *, backend, pipe, prompt,
                   steps, seed, radius):
    if backend == "diffusion":
        generated = inpaint_diffusion(pipe, observed_rgb, ~observed_mask,
                                      prompt, "furniture, objects, text, seams, holes",
                                      steps, seed)
    else:
        generated = cv2.inpaint(observed_rgb, (~observed_mask).astype(np.uint8),
                                radius, cv2.INPAINT_TELEA)
    generated[observed_mask] = observed_rgb[observed_mask]
    return generated


def sample_atlas(atlas, uv, low, high):
    u, v = uv_pixels(uv, low, high, atlas.shape[:2])
    return cv2.remap(atlas, u.astype(np.float32).reshape(1, -1),
                     v.astype(np.float32).reshape(1, -1),
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)[0].astype(np.float32)/255


def load_masks(manifest_path, views):
    with open(manifest_path, encoding="utf-8") as handle:
        entries = json.load(handle)["views"]
    return {v: entries[v]["mask_path"] for v in views
            if v in entries and entries[v].get("accepted")}


def surface_label(manifest_path, views, fallback):
    with open(manifest_path, encoding="utf-8") as handle:
        entries = json.load(handle)["views"]
    labels = []
    for view in views:
        entry = entries.get(view, {})
        if not entry.get("accepted"):
            continue
        index = entry.get("selected_candidate", 0)-1
        candidates = entry.get("candidates", [])
        if 0 <= index < len(candidates):
            label = candidates[index].get("label", "").strip(" .")
            if label:
                labels.append(label)
    return Counter(labels).most_common(1)[0][0] if labels else fallback


def one_photo(images, view, size):
    paths = [p for p in Path(images).glob(view+".*") if
             p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(paths) != 1:
        raise ValueError(f"Expected one photo for {view}")
    return np.asarray(Image.open(paths[0]).convert("RGB").resize(size))


def render_depth(model, camera, shape):
    height, width = shape
    rasterizer = GaussianRasterizer(camera_settings(camera, height, width)._replace(
        sh_degree=model.active_sh_degree))
    with torch.no_grad():
        _, _, _, depth = rasterizer(
            means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
            shs=model.get_features, colors_precomp=None,
            semantic_feature=model.get_semantic_feature,
            opacities=model.get_opacity, scales=model.get_scaling,
            rotations=model.get_rotation, cov3D_precomp=None)
    return depth[0].cpu().numpy()


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    template, seed = read_vertices(args.seed)
    with open(args.seed_report, encoding="utf-8") as handle:
        seed_report = json.load(handle)
    original_count = int(seed_report["original_kept"])
    floor_count = int(seed_report["floor_added"])
    wall_count = int(seed_report["wall_added"])
    if len(seed) != original_count+floor_count+wall_count:
        raise ValueError("Seed report does not match PLY")
    with open(args.wall_fit, encoding="utf-8") as handle:
        wall_fit = json.load(handle)
    with open(args.floor_fit, encoding="utf-8") as handle:
        floor_fit = json.load(handle)
    if (wall_fit["inlier_ratio"] < args.min_wall_inlier_ratio or
            floor_fit["plane_inlier_ratio"] < args.min_floor_inlier_ratio or
            wall_fit["plane_error_q90"] > args.max_wall_error):
        raise ValueError("Wall/floor geometry is unsupported; refusing 3D atlas generation")
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    with open(args.target_manifest, encoding="utf-8") as handle:
        target_entries = json.load(handle)["views"]
    training = sorted(v for v, item in target_entries.items() if item.get("accepted")
                      and v not in args.holdout_views and v in cameras)
    if len(training) < args.min_views:
        raise ValueError("Too few non-held-out target views")
    surface_masks = {"floor": load_masks(args.floor_manifest, training),
                     "wall": load_masks(args.wall_manifest, training)}
    if any(len(paths) < args.min_views for paths in surface_masks.values()):
        raise ValueError("Too few clean surface masks")
    with open(args.source_scene, encoding="rb") as handle:
        if not handle.read(3) == b"ply":
            raise ValueError("Source scene is not PLY")
    model = GaussianModel(3, 128)
    model.load_ply(args.source_scene)
    xyz = np.column_stack([seed[k] for k in ("x", "y", "z")]).astype(np.float64)
    floor_xyz = xyz[original_count:original_count+floor_count]
    wall_xyz = xyz[original_count+floor_count:]
    layouts = {}
    for kind, points in (("floor", floor_xyz), ("wall", wall_xyz)):
        origin, basis, normal = surface_basis(wall_fit, kind)
        low, high, shape = atlas_layout(points, origin, basis,
                                        resolution=args.resolution,
                                        margin=args.atlas_margin)
        layouts[kind] = dict(origin=origin, basis=basis, normal=normal,
                             low=low, high=high, shape=shape,
                             sums=np.zeros((*shape, 3), dtype=np.float64),
                             counts=np.zeros(shape, dtype=np.int32))
    torch.cuda.reset_peak_memory_stats()
    view_support = {"wall": {}, "floor": {}}
    for view in training:
        camera = cameras[view]
        width = args.render_width
        height = round(camera["height"]*width/camera["width"])
        shape = (height, width)
        photo = one_photo(args.images, view, (width, height))
        target_mask = np.asarray(Image.open(target_entries[view]["mask_path"])
                                 .convert("L").resize((width, height),
                                    Image.Resampling.NEAREST)) > 127
        depth = render_depth(model, camera, shape)
        for kind, layout in layouts.items():
            if view not in surface_masks[kind]:
                continue
            mask = np.asarray(Image.open(surface_masks[kind][view]).convert("L")
                              .resize((width, height), Image.Resampling.NEAREST)) > 127
            yy, xx = np.where(mask & ~target_mask)
            if not len(xx):
                continue
            points, ray_depth = backproject_plane(camera,
                np.column_stack((xx, yy)), shape, layout["origin"], layout["normal"])
            support = np.isfinite(ray_depth) & (ray_depth > 0) & (
                np.abs(depth[yy, xx]-ray_depth) <= args.depth_tolerance)
            uv = (points[support]-layout["origin"]) @ layout["basis"]
            contributed = accumulate_atlas(layout["sums"], layout["counts"], uv,
                photo[yy[support], xx[support]].astype(np.float64),
                layout["low"], layout["high"])
            view_support[kind][view] = contributed
    for kind, layout in layouts.items():
        if sum(count >= args.min_photo_samples for count in
               view_support[kind].values()) < args.min_views:
            raise ValueError(f"Too little intact {kind} texture in distinct views")
        observed = layout["counts"] > 0
        if observed.mean() < args.min_observed_fraction:
            raise ValueError(f"Too little observed {kind} atlas texture: {observed.mean():.1%}")
        rgb = np.zeros((*layout["shape"], 3), dtype=np.uint8)
        rgb[observed] = np.rint(layout["sums"][observed] /
                                 layout["counts"][observed, None]).astype(np.uint8)
        layout["observed"] = observed
        layout["rgb"] = rgb
    output.mkdir(parents=True)
    pipe = load_diffusion(args.model) if args.backend == "diffusion" else None
    candidate = seed.copy()
    surfaces = {}
    for number, (kind, points, begin, end) in enumerate((
        ("floor", floor_xyz, original_count, original_count+floor_count),
        ("wall", wall_xyz, original_count+floor_count, len(seed)))):
        layout = layouts[kind]
        observed = layout["observed"]
        description = surface_label(args.floor_manifest if kind == "floor" else
                                    args.wall_manifest, training, kind)
        prompt = (f"Seamless, realistic {description} surface texture matching the visible "
                  "unmasked area, same material and lighting, no objects")
        filled = complete_atlas(layout["rgb"], observed, backend=args.backend,
                                pipe=pipe, prompt=prompt, steps=args.steps,
                                seed=args.random_seed+number, radius=args.opencv_radius)
        Image.fromarray(layout["rgb"]).save(output / f"{kind}-observed.png")
        Image.fromarray((observed*255).astype(np.uint8)).save(
            output / f"{kind}-evidence.png")
        Image.fromarray(filled).save(output / f"{kind}-atlas.png")
        colors = sample_atlas(filled, (points-layout["origin"]) @ layout["basis"],
                              layout["low"], layout["high"])
        for channel in range(3):
            candidate[f"f_dc_{channel}"][begin:end] = (colors[:, channel]-.5)/SH_C0
        for field in seed.dtype.names:
            if field.startswith("f_rest_"):
                candidate[field][begin:end] = 0
        surfaces[kind] = {"shape": layout["shape"],
                          "surface_description": description, "prompt": prompt,
                          "observed_fraction": float(observed.mean()),
                          "observed_texels": int(observed.sum()),
                          "photo_support": view_support[kind],
                          "uv_low": layout["low"].tolist(),
                          "uv_high": layout["high"].tolist()}
    write_vertices(output / "candidate.ply", candidate, template,
                   ["UNAPPROVED single-atlas wall/floor background generation"])
    report = {"seed": str(Path(args.seed).resolve()),
              "candidate": str(output / "candidate.ply"),
              "backend": args.backend, "model": args.model if pipe else None,
              "training_views": training, "holdout_views": args.holdout_views,
              "surfaces": surfaces, "semantic_dimensions": len([
                  name for name in seed.dtype.names if name.startswith("semantic_")]),
              "approved": False, "requires_visual_review": True,
              "elapsed_seconds": time.perf_counter()-started,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated()/1024**2}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-scene", "seed", "seed-report", "wall-fit", "floor-fit",
                 "cameras", "images", "target-manifest", "wall-manifest",
                 "floor-manifest", "output-dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--backend", choices=("diffusion", "opencv"), default="diffusion")
    p.add_argument("--model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--render-width", type=int, default=540)
    p.add_argument("--atlas-margin", type=float, default=.4)
    p.add_argument("--depth-tolerance", type=float, default=.12)
    p.add_argument("--min-photo-samples", type=int, default=100)
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--min-observed-fraction", type=float, default=.03)
    p.add_argument("--min-wall-inlier-ratio", type=float, default=.3)
    p.add_argument("--min-floor-inlier-ratio", type=float, default=.8)
    p.add_argument("--max-wall-error", type=float, default=.3)
    p.add_argument("--opencv-radius", type=int, default=7)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
