"""Find remaining Gaussians whose projected splat radii overlap bed masks.

Unlike center-only voting, this admits broad Gaussians whose centers lie outside
the object mask. Exact rasterizer attribution should still decide deletion.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from gsedit.selection.multiview_instance import project
from scene.gaussian_model import GaussianModel


def footprint_overlap(points, camera, mask, radii):
    height, width = mask.shape
    x, y, _, valid = project(points, camera, mask.shape)
    distance = distance_transform_edt(~mask)
    ids = np.flatnonzero(valid & (radii > 0))
    supported = np.zeros(len(points), dtype=bool)
    supported[ids] = distance[y[ids], x[ids]] <= radii[ids]
    return supported


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    with open(args.seed_report, encoding="utf-8") as handle:
        report = json.load(handle)
    original_count = report["original_kept"]
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    with open(args.mask_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    training = sorted(v for v, d in manifest["views"].items() if d.get("accepted")
                      and v in cameras and v not in args.holdout_views)
    if len(training) < args.min_views:
        raise ValueError("Too few accepted training masks")
    model = GaussianModel(3, 128)
    model.load_ply(args.seed)
    if not 0 < original_count < len(model.get_xyz):
        raise ValueError("Invalid original-count boundary")
    xyz = model.get_xyz[:original_count].detach().cpu().numpy()
    votes = np.zeros(original_count, dtype=np.uint16)
    positive_radius = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for view in training:
            image = Image.open(manifest["views"][view]["mask_path"]).convert("L")
            height = round(cameras[view]["height"]*args.width/cameras[view]["width"])
            mask = np.asarray(image.resize((args.width, height),
                                            Image.Resampling.NEAREST)) > 127
            rasterizer = GaussianRasterizer(camera_settings(
                cameras[view], height, args.width)._replace(sh_degree=3))
            _, _, radii, _ = rasterizer(
                means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
                shs=model.get_features, colors_precomp=None,
                semantic_feature=model.get_semantic_feature,
                opacities=model.get_opacity, scales=model.get_scaling,
                rotations=model.get_rotation, cov3D_precomp=None)
            radius = radii[:original_count].cpu().numpy()
            overlap = footprint_overlap(xyz, cameras[view], mask, radius)
            votes += overlap.astype(np.uint16)
            positive_radius.append(radius[radius > 0])
            print(json.dumps({"view": view, "footprint_candidates": int(overlap.sum())}),
                  flush=True)
    selected = np.flatnonzero(votes >= args.min_views)
    if not len(selected) or len(selected) > args.max_candidates:
        raise ValueError(f"Unsafe candidate count: {len(selected)}")
    output.mkdir(parents=True)
    np.save(output / "candidate-indices.npy", selected)
    np.save(output / "supporting-views.npy", votes)
    radius_values = np.concatenate(positive_radius)
    result = {"seed": str(Path(args.seed).resolve()),
              "training_views": training, "holdout_views": args.holdout_views,
              "original_count": original_count, "candidate_count": len(selected),
              "min_views": args.min_views,
              "radius_q99_pixels": float(np.quantile(radius_values, .99)),
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated()/1024**2}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", required=True)
    p.add_argument("--seed-report", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--width", type=int, default=270)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--max-candidates", type=int, default=100000)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
