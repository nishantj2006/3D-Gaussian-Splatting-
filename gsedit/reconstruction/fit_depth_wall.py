"""Fit an observed wall plane from rendered depth inside clean photo masks."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from gsedit.assets.align_asset import plane_frame
from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from gsedit.reconstruction.fit_occluded_surfaces import line_ransac
from scene.gaussian_model import GaussianModel


def backproject(camera, x, y, depth, shape):
    height, width = shape
    fx = camera["fx"]*width/camera["width"]
    fy = camera["fy"]*height/camera["height"]
    rays = np.column_stack(((x-width/2)/fx, (y-height/2)/fy,
                            np.ones(len(x))))
    return np.asarray(camera["position"]) + (rays*depth[:, None]) @ np.asarray(
        camera["rotation"]).T


def sample_view(model, camera, mask, image, max_samples, rng):
    height, width = mask.shape
    rasterizer = GaussianRasterizer(camera_settings(camera, height, width)._replace(
        sh_degree=model.active_sh_degree))
    with torch.no_grad():
        _, _, _, depth = rasterizer(
            means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
            shs=model.get_features, colors_precomp=None,
            semantic_feature=model.get_semantic_feature,
            opacities=model.get_opacity, scales=model.get_scaling,
            rotations=model.get_rotation, cov3D_precomp=None)
    depth = depth[0].detach().cpu().numpy()
    yy, xx = np.where(mask)
    if len(xx) > max_samples:
        chosen = rng.choice(len(xx), max_samples, replace=False)
        xx, yy = xx[chosen], yy[chosen]
    dd = depth[yy, xx]
    good = np.isfinite(dd) & (dd > 0) & (dd < 100)
    xx, yy, dd = xx[good], yy[good], dd[good]
    return backproject(camera, xx, yy, dd, mask.shape), image[yy, xx], len(dd)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse depth-wall folder: {output}")
    start = time.perf_counter()
    with open(args.floor_plane, encoding="utf-8") as handle:
        ground = json.load(handle)
    origin, frame = plane_frame(ground["plane_origin"],
                                ground["plane_normal_toward_removed_object"])
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    with open(args.wall_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    training = sorted(set(view for view, item in manifest["views"].items()
                          if item["accepted"])-set(args.holdout_views))
    if len(training) < args.min_support_views:
        raise ValueError("Too few clean wall views")
    model = GaussianModel(args.sh_degree, args.semantic_dimensions)
    model.load_ply(args.scene)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                 "_scaling", "_rotation", "_semantic_feature"):
        getattr(model, name).requires_grad_(False)
    rng = np.random.default_rng(args.seed)
    samples, colors, origin_view = [], [], []
    view_counts = {}
    torch.cuda.reset_peak_memory_stats()
    for index, view in enumerate(training):
        mask = np.asarray(Image.open(manifest["views"][view]["mask_path"])
                          .convert("L")) > 127
        matches = list(Path(args.images).glob(view+".*"))
        if len(matches) != 1:
            raise ValueError(f"Expected one source image for {view}")
        image = np.asarray(Image.open(matches[0]).convert("RGB").resize(
            (mask.shape[1], mask.shape[0])), dtype=np.float32)/255
        xyz, rgb, count = sample_view(model, cameras[view], mask, image,
                                       args.samples_per_view, rng)
        local = (xyz-origin) @ frame
        valid = (local[:, 2] >= args.min_height) & (local[:, 2] <= args.max_height)
        samples.append(local[valid])
        colors.append(rgb[valid])
        origin_view.append(np.full(valid.sum(), index, dtype=np.uint8))
        view_counts[view] = {"mask_depth_samples": count,
                             "height_filtered_samples": int(valid.sum())}
        print(json.dumps({"view": view, **view_counts[view]}), flush=True)
    local = np.concatenate(samples)
    rgb = np.concatenate(colors)
    view_index = np.concatenate(origin_view)
    if len(local) < args.min_samples:
        raise ValueError("Insufficient depth-supported wall pixels")
    normal, offset, inliers, error = line_ransac(local[:, :2],
        threshold=args.plane_threshold, trials=args.trials, seed=args.seed)
    ratio = float(inliers.mean())
    supported = {view: int(((view_index == i) & inliers).sum())
                 for i, view in enumerate(training)}
    supported_views = sum(n >= args.min_view_inliers for n in supported.values())
    if (ratio < args.min_inlier_ratio or supported_views < args.min_support_views
            or error > args.plane_threshold):
        diagnostic = {"geometry_supported": False, "inlier_ratio": ratio,
                      "plane_error_q90": error, "supported_views": supported_views,
                      "sample_count": len(local), "training_views": training,
                      "holdout_views": args.holdout_views, "view_counts": view_counts,
                      "elapsed_seconds": time.perf_counter()-start,
                      "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated()/1024**2,
                      "approved": False}
        output.mkdir(parents=True)
        with open(output / "report.json", "w", encoding="utf-8") as handle:
            json.dump(diagnostic, handle, indent=2)
            handle.write("\n")
        raise ValueError(f"Depth wall unreliable: inliers {ratio:.2%}, "
                         f"views {supported_views}, q90 error {error:.3f}")
    output.mkdir(parents=True)
    tangent = np.array([-normal[1], normal[0]])
    th = np.column_stack((local[inliers, :2] @ tangent, local[inliers, 2]))
    np.savez_compressed(output / "wall-photo-samples.npz", th=th.astype(np.float32),
                        rgb=rgb[inliers].astype(np.float32), view=view_index[inliers])
    report = {"source": str(Path(args.scene).resolve()),
              "origin": origin.tolist(), "frame": frame.tolist(),
              "wall_normal_floor_xy": normal.tolist(), "wall_offset": offset,
              "sample_count": len(local), "inlier_samples": int(inliers.sum()),
              "inlier_ratio": ratio, "plane_error_q90": error,
              "training_views": training, "holdout_views": args.holdout_views,
              "supported_views": supported_views, "view_counts": view_counts,
              "view_inliers": supported,
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated()/1024**2,
              "approved": False}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("sample_count", "inlier_samples",
        "inlier_ratio", "plane_error_q90", "supported_views",
        "elapsed_seconds", "peak_rss_mb", "peak_gpu_allocated_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--floor-plane", required=True)
    p.add_argument("--wall-manifest", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples-per-view", type=int, default=8000)
    p.add_argument("--min-samples", type=int, default=10000)
    p.add_argument("--min-height", type=float, default=.5)
    p.add_argument("--max-height", type=float, default=10.)
    p.add_argument("--plane-threshold", type=float, default=.3)
    p.add_argument("--min-inlier-ratio", type=float, default=.15)
    p.add_argument("--min-view-inliers", type=int, default=200)
    p.add_argument("--min-support-views", type=int, default=3)
    p.add_argument("--trials", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--semantic-dimensions", type=int, default=128)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
