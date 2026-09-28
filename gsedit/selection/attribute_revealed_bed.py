"""Preview-only cross-view peeling of bed splats revealed after first deletion.

Contributions are measured with the synthetic wall/floor hidden, so background
replacement cannot conceal the residual being identified. Boundary splats are
attenuated rather than deleted when they contribute outside bed masks.
"""

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
from gsedit.reconstruction.refine_local_background import candidate_pool
from gsedit.selection.refine_revealed_layers import view_contributions
from scene.gaussian_model import GaussianModel
from utils.ply_semantic_utils import read_vertices, write_vertices


def classify(inside, outside, views, *, min_inside, min_views, min_agreement,
             boundary_min_agreement):
    ratio = inside / np.maximum(inside+outside, 1e-8)
    confident = (inside >= min_inside) & (views >= min_views) & (ratio >= min_agreement)
    boundary = (inside >= min_inside) & (views >= min_views) & (
        ratio >= boundary_min_agreement) & ~confident
    return confident, boundary, ratio


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    start = time.perf_counter()
    source_ply, source = read_vertices(args.source)
    seed_ply, vertices = read_vertices(args.seed)
    if source.dtype.names != vertices.dtype.names:
        raise ValueError("PLY schemas differ")
    selected = np.unique(np.load(args.selected_indices, allow_pickle=False))
    original_count = len(source)-len(selected)
    source_xyz = np.column_stack([source[k] for k in ("x","y","z")])
    seed_xyz = np.column_stack([vertices[k] for k in ("x","y","z")])
    if args.all_original and args.candidate_indices:
        raise ValueError("Choose all-original or candidate-indices, not both")
    if args.all_original:
        pool_np = np.arange(original_count, dtype=np.int64)
    elif args.candidate_indices:
        pool_np = np.unique(np.load(args.candidate_indices, allow_pickle=False))
        if (pool_np.ndim != 1 or not len(pool_np) or pool_np.min() < 0 or
                pool_np.max() >= original_count):
            raise ValueError("Invalid footprint candidate indices")
    else:
        pool_np = candidate_pool(source_xyz, selected, seed_xyz, original_count,
                                 args.max_seed_distance)
    protected_count = 0
    if args.protected_source_indices:
        protected = np.unique(np.load(args.protected_source_indices, allow_pickle=False))
        if not len(protected) or protected.min() < 0 or protected.max() >= len(source):
            raise ValueError("Invalid protected source indices")
        keep = np.ones(len(source), dtype=bool)
        keep[selected] = False
        source_to_candidate = np.full(len(source), -1, dtype=np.int64)
        source_to_candidate[np.flatnonzero(keep)] = np.arange(original_count)
        protected_candidate = source_to_candidate[protected]
        before = len(pool_np)
        pool_np = np.setdiff1d(pool_np, protected_candidate[protected_candidate >= 0])
        protected_count = before-len(pool_np)
    if not len(pool_np) or len(pool_np) > args.max_pool_splats:
        raise ValueError(f"Unsafe pool size {len(pool_np)}")
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    with open(args.mask_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    training = sorted(v for v, d in manifest["views"].items() if d.get("accepted")
                      and v in cameras and v not in args.holdout_views)
    if len(training) < args.min_training_views:
        raise ValueError("Too few accepted training masks")
    model = GaussianModel(3, 128)
    model.load_ply(str(args.seed))
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                 "_scaling", "_rotation", "_semantic_feature"):
        getattr(model, name).requires_grad_(False)
    pool = torch.as_tensor(pool_np, device="cuda", dtype=torch.long)
    opacity = model.get_opacity.detach().clone()
    opacity[original_count:] = 0  # Expose source residual; fill must not hide it.
    rasters, masks = {}, {}
    for view in training:
        mask = Image.open(manifest["views"][view]["mask_path"]).convert("L")
        h = round(cameras[view]["height"]*args.width/cameras[view]["width"])
        mask = np.asarray(mask.resize((args.width, h), Image.Resampling.NEAREST)) > 127
        masks[view] = torch.from_numpy(mask.astype(np.float32)).cuda()
        rasters[view] = GaussianRasterizer(camera_settings(cameras[view], h, args.width))
    torch.cuda.reset_peak_memory_stats()
    rounds = []
    hard = np.zeros(len(pool_np), dtype=bool)
    boundary_gate = np.ones(len(pool_np), dtype=np.float32)
    for layer in range(args.layers):
        inside = np.zeros(len(pool_np), dtype=np.float64)
        outside = np.zeros(len(pool_np), dtype=np.float64)
        support = np.zeros(len(pool_np), dtype=np.uint16)
        for view in training:
            contribution, exterior = view_contributions(
                model, rasters[view], opacity, pool, masks[view])
            inside += np.maximum(contribution, 0)
            outside += np.maximum(exterior, 0)
            support += (contribution >= args.min_view_contribution).astype(np.uint16)
        confident, boundary, ratio = classify(
            inside, outside, support, min_inside=args.min_total_contribution,
            min_views=args.min_supporting_views, min_agreement=args.min_agreement,
            boundary_min_agreement=args.boundary_min_agreement)
        new_hard = confident & ~hard
        hard |= confident
        # Partial alpha only for candidates that contribute to both foreground
        # and exterior, preserving broad splats' non-bed footprint.
        boundary_gate[boundary] = np.minimum(boundary_gate[boundary],
            np.clip((args.min_agreement-ratio[boundary]) /
                    (args.min_agreement-args.boundary_min_agreement), .2, 1.))
        new_gate = np.ones(len(pool_np), dtype=np.float32)
        new_gate[hard] = 0
        new_gate[boundary & ~hard] = boundary_gate[boundary & ~hard]
        opacity[pool] = model.get_opacity.detach()[pool] * torch.from_numpy(new_gate).cuda()[:, None]
        rounds.append({"layer": layer+1, "new_confirmed": int(new_hard.sum()),
                       "total_confirmed": int(hard.sum()),
                       "boundary_attenuated": int((new_gate > 0).sum()-(new_gate == 1).sum())})
        print(json.dumps(rounds[-1]), flush=True)
        if not new_hard.any():
            break
    if hard.sum() / original_count > args.max_scene_fraction:
        raise ValueError("Refinement exceeds scene safety fraction")
    final_gate = np.ones(len(pool_np), dtype=np.float32)
    final_gate[hard] = args.minimum_gate
    final_gate[~hard] = np.maximum(boundary_gate[~hard], args.minimum_gate)
    result = vertices.copy()
    changed_indices = pool_np[final_gate < 1]
    changed_gates = final_gate[final_gate < 1]
    raw = np.asarray(result["opacity"][changed_indices], dtype=np.float64)
    initial = 1/(1+np.exp(-raw))
    adjusted = np.clip(initial*changed_gates, 1e-6, 1-1e-6)
    result["opacity"][changed_indices] = np.log(adjusted/(1-adjusted))
    output.mkdir(parents=True)
    candidate = output / "candidate.ply"
    write_vertices(candidate, result, seed_ply,
                   ["UNAPPROVED revealed-layer bed removal; original preserved"])
    np.savez_compressed(output / "evidence.npz", indices=pool_np, hard=hard,
                        gate=final_gate, inside=inside.astype(np.float32),
                        outside=outside.astype(np.float32), support=support)
    report = {"source": str(Path(args.source).resolve()), "seed": str(Path(args.seed).resolve()),
              "candidate": str(candidate), "training_views": training,
              "holdout_views": args.holdout_views, "pool_splats": len(pool_np),
              "protected_pool_exclusions": protected_count,
              "rounds": rounds, "confirmed_removed": int(hard.sum()),
              "boundary_attenuated": int(((final_gate < 1)&~hard).sum()),
              "preserved_original_splats": original_count-int(hard.sum()),
              "semantic_dimensions": 128, "approved": False,
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated()/1024**2}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("confirmed_removed", "boundary_attenuated",
        "elapsed_seconds", "peak_rss_mb", "peak_gpu_allocated_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--selected-indices", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--width", type=int, default=270)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--max-seed-distance", type=float, default=2.)
    p.add_argument("--candidate-indices")
    p.add_argument("--all-original", action="store_true")
    p.add_argument("--protected-source-indices")
    p.add_argument("--max-pool-splats", type=int, default=60000)
    p.add_argument("--min-training-views", type=int, default=4)
    p.add_argument("--min_supporting_views", type=int, default=2)
    p.add_argument("--min-view-contribution", type=float, default=.02)
    p.add_argument("--min-total-contribution", type=float, default=.5)
    p.add_argument("--min-agreement", type=float, default=.85)
    p.add_argument("--boundary-min-agreement", type=float, default=.55)
    p.add_argument("--minimum-gate", type=float, default=.001)
    p.add_argument("--max-scene-fraction", type=float, default=.1)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
