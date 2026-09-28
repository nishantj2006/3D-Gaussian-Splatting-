"""Estimate *prior* depth on generated reference images, never ground truth.

The monocular prediction is robustly aligned to rendered clean-background
depth outside the target mask. Weak alignment rejects the view. Even a good
alignment is not sufficient to approve missing geometry across camera views.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizer
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from gsedit.reconstruction.crossview_background import calibrate_prior
from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from gsedit.reconstruction.recover_background_evidence import mask_at
from scene.gaussian_model import GaussianModel


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {x["img_name"]: x for x in json.load(handle)}
    manifests = {}
    for label, path in (("target", args.target_manifest), ("wall", args.wall_manifest),
                        ("floor", args.floor_manifest)):
        with open(path, encoding="utf-8") as handle:
            manifests[label] = json.load(handle)
    views = sorted(set(args.views) - set(args.holdout_views))
    if len(views) < 2:
        raise ValueError("At least two non-held-out reference views required")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model)
    depth_model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()
    model = GaussianModel(3, 128)
    model.load_ply(args.scene)
    for key in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
                "_rotation", "_semantic_feature"):
        getattr(model, key).requires_grad_(False)
    output.mkdir(parents=True)
    reports = {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for view in views:
        if view not in cameras:
            raise ValueError(f"Unknown camera: {view}")
        image_path = Path(args.guides) / f"{view}{args.guide_suffix}"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        size = image.size
        complete = manifests["target"]["views"].get(view, {}).get("complete", True)
        target = mask_at(manifests["target"], view, size)
        wall = mask_at(manifests["wall"], view, size)
        floor = mask_at(manifests["floor"], view, size)
        if not complete or target is None or (wall is None and floor is None):
            reports[view] = {"accepted": False, "reason": "missing_complete_surface_masks"}
            continue
        clean = ((wall if wall is not None else np.zeros_like(target)) |
                 (floor if floor is not None else np.zeros_like(target))) & ~target
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output_depth = depth_model(**inputs).predicted_depth
            prior = F.interpolate(output_depth[:, None], size=(size[1], size[0]),
                                  mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
            rasterizer = GaussianRasterizer(camera_settings(cameras[view], size[1], size[0])._replace(
                sh_degree=model.active_sh_degree))
            _, _, _, rendered = rasterizer(
                means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
                shs=model.get_features, colors_precomp=None,
                semantic_feature=model.get_semantic_feature,
                opacities=model.get_opacity, scales=model.get_scaling,
                rotations=model.get_rotation, cov3D_precomp=None)
        reference = rendered[0].cpu().numpy()
        try:
            aligned, fit = calibrate_prior(prior, reference, clean,
                                           min_samples=args.min_samples)
            accepted = fit["median_observed_error"] <= args.max_observed_error
            if not accepted:
                fit["rejection"] = "poor_observed_depth_alignment"
        except ValueError as exc:
            aligned = np.full_like(prior, np.nan)
            fit = {"rejection": str(exc)}
            accepted = False
        np.save(output / f"{view}-prior-depth.npy", aligned.astype(np.float32))
        np.save(output / f"{view}-observed-depth.npy",
                np.where(clean, reference, np.nan).astype(np.float32))
        Image.fromarray(clean.astype(np.uint8) * 255).save(output / f"{view}-clean-mask.png")
        reports[view] = {"accepted": bool(accepted), "image": str(image_path),
                         "clean_samples": int(clean.sum()), "target_pixels": int(target.sum()),
                         **fit, "note": "In-mask depth is generated prior, not observed evidence"}
    report = {"source": str(Path(args.scene).resolve()), "model": args.model,
              "views": reports, "holdout_views": sorted(args.holdout_views),
              "elapsed_seconds": time.perf_counter() - started,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2
              if device == "cuda" else 0., "geometry_supported": False,
              "approved": False}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "cameras", "guides", "target-manifest", "wall-manifest",
                 "floor-manifest", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--holdout-views", nargs="+", default=[])
    p.add_argument("--guide-suffix", default="-guide.png")
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    p.add_argument("--min-samples", type=int, default=1000)
    p.add_argument("--max-observed-error", type=float, default=.15)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
