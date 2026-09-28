"""Conservatively select newly revealed object splats after a first removal.

The scene and first-pass selection are read-only. Exact rasterizer gradients
measure how much each *revealed* candidate contributes inside versus outside
the original photo masks. A color/semantic prior learned from the confirmed
first-pass splats rejects nearby wall/floor candidates. Output is preview-only.
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
from diff_gaussian_rasterization import GaussianRasterizer

from gsedit.evaluation.evaluate_rendered_masks import camera_settings, render_selection, scores
from gsedit.selection.refine_rendered_selection import candidate_pool
from scene.gaussian_model import GaussianModel


def appearance_gate(model, confirmed, candidates, color_quantile=.95,
                    semantic_quantile=.20, color_neighbors=8):
    """Calibrate appearance thresholds from confirmed splats, not named colors."""
    dc = model._features_dc[:, 0, :].detach().cpu().numpy()
    rgb = np.clip(dc * .2820947918 + .5, 0, 1)
    reference = rgb[confirmed]
    neighbors = min(color_neighbors + 1, len(reference))
    tree = cKDTree(reference)
    self_distance = tree.query(reference, k=neighbors, workers=-1)[0][:, -1]
    color_limit = float(np.quantile(self_distance, color_quantile))
    color_distance = tree.query(rgb[candidates], k=min(color_neighbors, len(reference)),
                                workers=-1)[0]
    if color_distance.ndim > 1:
        color_distance = color_distance[:, -1]
    semantic = model.get_semantic_feature[:, 0, :].detach().cpu().numpy()
    semantic /= np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-8)
    centroid = semantic[confirmed].mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-8)
    confirmed_cosine = semantic[confirmed] @ centroid
    semantic_limit = float(np.quantile(confirmed_cosine, semantic_quantile))
    candidate_cosine = semantic[candidates] @ centroid
    accepted = (color_distance <= color_limit) & (candidate_cosine >= semantic_limit)
    return accepted, color_distance, candidate_cosine, {
        "color_knn_distance_limit": color_limit,
        "semantic_cosine_limit": semantic_limit,
        "appearance_accepted": int(accepted.sum())}


def select_revealed(inside, outside, supporting_views, appearance_ok, *,
                    min_inside, min_views, min_agreement):
    total = inside + outside
    agreement = inside / np.maximum(total, 1e-8)
    selected = (inside >= min_inside) & (supporting_views >= min_views) & (
        agreement >= min_agreement) & appearance_ok
    return selected, agreement


def view_contributions(model, rasterizer, opacity, pool, target):
    """Exact per-Gaussian color influence with first-pass splats occlusion-free."""
    weights = torch.ones(len(pool), device="cuda", requires_grad=True)
    colors = torch.zeros((len(model.get_xyz), 3), device="cuda")
    colors = colors.index_copy(0, pool, weights[:, None].expand(-1, 3))
    image, _, _, _ = rasterizer(
        means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
        shs=None, colors_precomp=colors,
        semantic_feature=model.get_semantic_feature,
        opacities=opacity, scales=model.get_scaling,
        rotations=model.get_rotation, cov3D_precomp=None)
    contribution = image[0]
    inside = torch.autograd.grad((contribution * target).sum(), weights,
                                 retain_graph=True)[0]
    outside = torch.autograd.grad((contribution * (1-target)).sum(), weights)[0]
    return inside.detach().cpu().numpy(), outside.detach().cpu().numpy()


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
    if len(training) < args.min_views:
        raise ValueError("Too few training views")
    model = GaussianModel(args.sh_degree, args.semantic_dimensions)
    model.load_ply(args.scene)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                 "_scaling", "_rotation", "_semantic_feature"):
        getattr(model, name).requires_grad_(False)
    n = len(model.get_xyz)
    first = np.unique(np.load(args.first_indices, allow_pickle=False))
    seed = np.unique(np.load(args.seed_indices, allow_pickle=False))
    for label, ids in (("first", first), ("seed", seed)):
        if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= n:
            raise ValueError(f"Invalid {label} selection")
    points = model.get_xyz.detach().cpu().numpy()
    pool_all = candidate_pool(points, seed, first, args.max_seed_distance)
    pool_np = np.setdiff1d(pool_all, first, assume_unique=True)
    if not len(pool_np):
        raise ValueError("No newly revealed candidates in the spatial pool")
    appearance_ok, color_distance, cosine, appearance_report = appearance_gate(
        model, first, pool_np, color_quantile=args.color_quantile,
        semantic_quantile=args.semantic_quantile,
        color_neighbors=args.color_neighbors)
    pool = torch.as_tensor(pool_np, device="cuda", dtype=torch.long)
    opacity = model.get_opacity.clone()
    opacity[torch.as_tensor(first, device="cuda", dtype=torch.long)] = 0
    masks = {}
    rasters = {}
    for view in available:
        mask = np.asarray(Image.open(manifest["views"][view]["mask_path"]).convert("L")) > 127
        height, width = mask.shape
        masks[view] = torch.from_numpy(mask.astype(np.float32)).cuda()
        rasters[view] = GaussianRasterizer(camera_settings(cameras[view], height, width))
    total_inside = np.zeros(len(pool_np), dtype=np.float64)
    total_outside = np.zeros(len(pool_np), dtype=np.float64)
    supporting = np.zeros(len(pool_np), dtype=np.uint16)
    torch.cuda.reset_peak_memory_stats()
    for view in training:
        inside, outside = view_contributions(model, rasters[view], opacity, pool,
                                             masks[view])
        total_inside += inside
        total_outside += outside
        supporting += (inside >= args.min_view_contribution).astype(np.uint16)
        print(json.dumps({"view": view, "revealed_splats": int((inside > 0).sum())}),
              flush=True)
    add, agreement = select_revealed(total_inside, total_outside, supporting,
        appearance_ok, min_inside=args.min_total_contribution,
        min_views=args.min_views, min_agreement=args.min_agreement)
    selected = np.unique(np.concatenate([first, pool_np[add]]))
    if len(selected)/n > args.max_scene_fraction:
        raise ValueError("Result exceeds scene safety fraction")
    output.mkdir(parents=True)
    np.save(output / "selected-indices.npy", selected)
    np.save(output / "added-indices.npy", pool_np[add])
    np.savez_compressed(output / "candidate-evidence.npz", indices=pool_np,
                        inside=total_inside.astype(np.float32),
                        outside=total_outside.astype(np.float32),
                        supporting_views=supporting, color_distance=color_distance,
                        semantic_cosine=cosine, appearance_ok=appearance_ok,
                        selected=add)
    report = {"source": str(Path(args.scene).resolve()),
              "first_indices": str(Path(args.first_indices).resolve()),
              "training_views": training, "holdout_views": sorted(holdouts),
              "first_splats": len(first), "candidate_splats": len(pool_np),
              "added_splats": int(add.sum()), "selected_splats": len(selected),
              "appearance": appearance_report,
              "min_views": args.min_views,
              "min_agreement": args.min_agreement,
              "min_total_contribution": args.min_total_contribution,
              "heldout_mask_metrics": {}, "approved": False,
              "warning": "Automatic masks are pseudo-labels. RGB after-removal renders must be reviewed for residuals and wall/floor damage."}
    with torch.no_grad():
        for view in sorted(holdouts):
            target = masks[view] > .5
            report["heldout_mask_metrics"][view] = {}
            for label, ids in (("first", first), ("revealed", selected)):
                rendered, _ = render_selection(model, rasters[view], ids)
                report["heldout_mask_metrics"][view][label] = scores(
                    rendered >= args.render_threshold, target, .5)
                Image.fromarray((rendered.cpu().numpy()*255).astype(np.uint8)).save(
                    output / f"{view}-{label}-mask.png")
    report["elapsed_seconds"] = time.perf_counter()-start
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    report["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated()/1024**2
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("candidate_splats", "added_splats",
        "selected_splats", "appearance", "elapsed_seconds", "peak_rss_mb",
        "peak_gpu_allocated_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--first-indices", required=True)
    p.add_argument("--seed-indices", required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seed-distance", type=float, default=1.5)
    p.add_argument("--color-quantile", type=float, default=.95)
    p.add_argument("--semantic-quantile", type=float, default=.20)
    p.add_argument("--color-neighbors", type=int, default=8)
    p.add_argument("--min-view-contribution", type=float, default=.05)
    p.add_argument("--min-total-contribution", type=float, default=1.)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--min-agreement", type=float, default=.75)
    p.add_argument("--render-threshold", type=float, default=.1)
    p.add_argument("--max-scene-fraction", type=float, default=.2)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--semantic-dimensions", type=int, default=128)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
