"""Save exact rasterized per-view influence of removed/attenuated source splats."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from scene.gaussian_model import GaussianModel


def render_influence(model, rasterizer, indices, strength):
    colors = torch.zeros((len(model.get_xyz), 3), device="cuda")
    ids = torch.as_tensor(indices, device="cuda", dtype=torch.long)
    weight = torch.as_tensor(strength, device="cuda", dtype=torch.float32)
    colors[ids] = weight[:, None].expand(-1, 3)
    image, _, _, _ = rasterizer(
        means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
        shs=None, colors_precomp=colors,
        semantic_feature=model.get_semantic_feature,
        opacities=model.get_opacity, scales=model.get_scaling,
        rotations=model.get_rotation, cov3D_precomp=None)
    return image[0].clamp(0, 1)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    with np.load(args.provenance, allow_pickle=False) as data:
        removed = data["removed_source_indices"]
        attenuated = data["attenuated_source_indices"]
        gates = data["attenuation_gates"]
    if len(removed) == 0 or len(attenuated) != len(gates):
        raise ValueError("Invalid provenance")
    ids = np.concatenate((removed, attenuated))
    strength = np.concatenate((np.ones(len(removed)), 1-gates))
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Removed and attenuated source IDs overlap")
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    with open(args.mask_manifest, encoding="utf-8") as handle:
        masks = json.load(handle)
    views = sorted(v for v, d in masks["views"].items()
                   if d.get("accepted") and v in cameras)
    if len(views) < 3:
        raise ValueError("Too few camera views")
    model = GaussianModel(3, 128)
    model.load_ply(args.source)
    if ids.min() < 0 or ids.max() >= len(model.get_xyz):
        raise ValueError("Provenance IDs outside source PLY")
    output.mkdir(parents=True)
    details = {}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for view in views:
            mask = np.asarray(Image.open(masks["views"][view]["mask_path"])
                              .convert("L").resize((args.width,
                              round(cameras[view]["height"]*args.width/
                                    cameras[view]["width"])),
                              Image.Resampling.NEAREST)) > 127
            height, width = mask.shape
            rasterizer = GaussianRasterizer(camera_settings(cameras[view], height, width))
            influence = render_influence(model, rasterizer, ids, strength).cpu().numpy()
            Image.fromarray((influence*255).astype(np.uint8)).save(
                output / f"{view}-influence.png")
            inside = float(influence[mask].mean()) if mask.any() else 0.
            outside = float(influence[~mask].mean()) if (~mask).any() else 0.
            details[view] = {"influence_path": str(output/f"{view}-influence.png"),
                             "inside_mean": inside, "outside_mean": outside,
                             "visible_pixels_above_0.1": int((influence >= .1).sum()),
                             "target_mask_path": masks["views"][view]["mask_path"]}
    report = {"source": str(Path(args.source).resolve()),
              "provenance": str(Path(args.provenance).resolve()),
              "removed_source_gaussians": len(removed),
              "attenuated_source_gaussians": len(attenuated),
              "views": details,
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated()/1024**2}
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("removed_source_gaussians",
        "attenuated_source_gaussians", "elapsed_seconds", "peak_rss_mb",
        "peak_gpu_allocated_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--width", type=int, default=270)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
