"""Fit a reversible per-Gaussian object selection to multi-view image masks.

Only selection probabilities are optimized. Scene geometry, appearance, semantic
features, and the input PLY remain unchanged. Outputs are an unapproved preview.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizer
from gsedit.evaluation.evaluate_rendered_masks import camera_settings, render_selection, scores
from scene.gaussian_model import GaussianModel


def candidate_pool(points, seed, baseline, radius):
    distance = cKDTree(points[seed]).query(points, workers=-1)[0]
    pool = distance <= radius
    pool[baseline] = True
    return np.flatnonzero(pool)


def render_probabilities(model, rasterizer, pool, logits):
    colors = torch.zeros((len(model.get_xyz), 3), device="cuda")
    colors = colors.index_copy(0, pool, torch.sigmoid(logits)[:, None].expand(-1, 3))
    image, _, _, _ = rasterizer(
        means3D=model.get_xyz,
        means2D=torch.zeros_like(model.get_xyz),
        shs=None, colors_precomp=colors,
        semantic_feature=model.get_semantic_feature,
        opacities=model.get_opacity, scales=model.get_scaling,
        rotations=model.get_rotation, cov3D_precomp=None)
    return image[0].clamp(0, 1)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    start = time.perf_counter()
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    with open(args.mask_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    available = sorted(view for view, detail in manifest["views"].items()
                       if detail["accepted"] and view in cameras)
    holdouts = set(args.holdout_views)
    if not holdouts.issubset(available):
        raise ValueError("A holdout view has no accepted mask")
    training = sorted(set(available) - holdouts)
    if len(training) < args.min_training_views:
        raise ValueError("Insufficient training views")
    model = GaussianModel(args.sh_degree, args.semantic_dimensions)
    model.load_ply(args.scene)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                 "_scaling", "_rotation", "_semantic_feature"):
        getattr(model, name).requires_grad_(False)
    points = model.get_xyz.detach().cpu().numpy()
    baseline = np.unique(np.load(args.baseline_indices, allow_pickle=False))
    seed = np.unique(np.load(args.seed_indices, allow_pickle=False))
    n = len(points)
    for label, ids in (("baseline", baseline), ("seed", seed)):
        if ids.ndim != 1 or len(ids) == 0 or ids.min() < 0 or ids.max() >= n:
            raise ValueError(f"Invalid {label} selection")
    pool_np = candidate_pool(points, seed, baseline, args.max_seed_distance)
    pool = torch.as_tensor(pool_np, device="cuda", dtype=torch.long)
    baseline_flag = np.isin(pool_np, baseline)
    seed_flag = np.isin(pool_np, seed)
    initial = np.where(baseline_flag, args.initial_logit, -args.initial_logit)
    logits = torch.nn.Parameter(torch.tensor(initial, device="cuda", dtype=torch.float32))
    initial_prob = torch.sigmoid(logits.detach())
    seed_pool = torch.as_tensor(seed_flag, device="cuda")
    optimizer = torch.optim.Adam([logits], lr=args.learning_rate)
    rasters = {}
    targets = {}
    for view in available:
        target = np.asarray(Image.open(manifest["views"][view]["mask_path"]).convert("L")) > 127
        height, width = target.shape
        rasters[view] = GaussianRasterizer(camera_settings(cameras[view], height, width))
        targets[view] = torch.from_numpy(target.astype(np.float32)).cuda()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for epoch in range(args.epochs):
        for view in training:
            optimizer.zero_grad(set_to_none=True)
            prediction = render_probabilities(model, rasters[view], pool, logits)
            target = targets[view]
            data_loss = F.binary_cross_entropy(prediction.clamp(1e-5, 1-1e-5), target)
            agreement = (torch.sigmoid(logits) - initial_prob).abs().mean()
            seed_penalty = (1 - torch.sigmoid(logits[seed_pool])).mean()
            loss = data_loss + args.change_penalty * agreement + args.seed_penalty * seed_penalty
            loss.backward()
            optimizer.step()
            losses.append(float(data_loss.detach()))
        print(json.dumps({"epoch": epoch + 1, "mean_bce": float(np.mean(losses[-len(training):]))}), flush=True)
    probability = torch.sigmoid(logits).detach().cpu().numpy()
    selected = np.zeros(n, dtype=bool)
    selected[pool_np[probability >= args.selection_threshold]] = True
    selected[seed] = True
    baseline_mask = np.zeros(n, dtype=bool)
    baseline_mask[baseline] = True
    if selected.mean() > args.max_scene_fraction:
        raise ValueError("Selection exceeds scene safety limit")
    output.mkdir(parents=True)
    np.save(output / "selected-indices.npy", np.flatnonzero(selected))
    np.save(output / "pool-indices.npy", pool_np)
    np.save(output / "pool-probabilities.npy", probability)
    report = {"source": str(Path(args.scene).resolve()),
              "training_views": training, "holdout_views": sorted(holdouts),
              "baseline_splats": len(baseline), "seed_splats": len(seed),
              "candidate_pool_splats": len(pool_np), "selected_splats": int(selected.sum()),
              "added_vs_baseline": int((selected & ~baseline_mask).sum()),
              "removed_from_baseline": int((baseline_mask & ~selected).sum()),
              "metrics": {}, "approved": False,
              "metric_warning": "Automatic SAM 2 masks are pseudo-labels, not ground truth."}
    with torch.no_grad():
        for view in sorted(holdouts):
            target = targets[view] > .5
            report["metrics"][view] = {}
            for label, indices in (("baseline", baseline),
                                   ("refined", np.flatnonzero(selected))):
                image, _ = render_selection(model, rasters[view], indices)
                report["metrics"][view][label] = scores(image >= args.render_threshold,
                                                         target, .5)
                Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8)).save(
                    output / f"{view}-{label}.png")
    report["mean_training_bce_first_epoch"] = float(np.mean(losses[:len(training)]))
    report["mean_training_bce_last_epoch"] = float(np.mean(losses[-len(training):]))
    report["elapsed_seconds"] = time.perf_counter()-start
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    report["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated()/1024**2
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--baseline-indices", required=True)
    p.add_argument("--seed-indices", required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--min-training-views", type=int, default=4)
    p.add_argument("--max-seed-distance", type=float, default=.5)
    p.add_argument("--initial-logit", type=float, default=2.)
    p.add_argument("--learning-rate", type=float, default=.15)
    p.add_argument("--change-penalty", type=float, default=.005)
    p.add_argument("--seed-penalty", type=float, default=.01)
    p.add_argument("--selection-threshold", type=float, default=.5)
    p.add_argument("--render-threshold", type=float, default=.1)
    p.add_argument("--max-scene-fraction", type=float, default=.2)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--semantic-dimensions", type=int, default=128)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
