"""Locally tune a continuous replacement without retraining the source scene.

Training photos supervise visible exterior pixels. Inside bed masks, a frozen
cross-view Gaussian replacement render is the target; original photos still
show the bed and are deliberately not used there. All unrelated scene splats
and all semantic features remain frozen. Output is preview-only.
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
from gsedit.reconstruction.refine_local_background import load_view
from scene.gaussian_model import GaussianModel
from utils.ply_semantic_utils import read_vertices, write_vertices


def render(model, rasterizer, shs, opacity):
    image, _, _, _ = rasterizer(
        means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
        shs=shs, colors_precomp=None,
        semantic_feature=model.get_semantic_feature,
        opacities=opacity, scales=model.get_scaling,
        rotations=model.get_rotation, cov3D_precomp=None)
    return image.clamp(0, 1)


def mean_masked_l1(prediction, target, mask):
    return ((prediction-target).abs()*mask).sum()/(mask.sum()*3+1)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse optimization folder: {output}")
    start = time.perf_counter()
    source_ply, vertices = read_vertices(args.seed)
    with open(args.seed_report, encoding="utf-8") as handle:
        seed_report = json.load(handle)
    original_count = seed_report["original_kept"]
    if not 0 < original_count < len(vertices):
        raise ValueError("Invalid original/new Gaussian boundary")
    with np.load(args.gate_evidence, allow_pickle=False) as evidence:
        ratio = evidence["inside"]/(evidence["inside"]+evidence["outside"]+1e-8)
        boundary_np = evidence["indices"][(evidence["gate"] > args.hard_gate) &
                                           (evidence["gate"] < 1) &
                                           (ratio >= args.boundary_min_agreement)]
    if len(boundary_np) > args.max_boundary_splats or boundary_np.max(initial=0) >= original_count:
        raise ValueError("Unsafe boundary optimization pool")
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    with open(args.bed_manifest, encoding="utf-8") as handle:
        bed_manifest = json.load(handle)
    with open(args.wall_manifest, encoding="utf-8") as handle:
        wall_manifest = json.load(handle)
    training = sorted(v for v, d in bed_manifest["views"].items() if
                      d.get("accepted") and v in cameras and v not in args.holdout_views)
    if len(training) < args.min_training_views:
        raise ValueError("Too few accepted training cameras")
    model = GaussianModel(3, 128)
    model.load_ply(str(args.seed))
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                 "_scaling", "_rotation", "_semantic_feature"):
        getattr(model, name).requires_grad_(False)
    added = torch.arange(original_count, len(vertices), device="cuda")
    boundary = torch.as_tensor(boundary_np, device="cuda", dtype=torch.long)
    base_dc = model._features_dc.detach()
    base_rest = model._features_rest.detach()
    base_raw_alpha = model._opacity.detach()
    base_alpha = model.get_opacity.detach()
    dc_delta = torch.nn.Parameter(torch.zeros((len(added), 3), device="cuda"))
    alpha_delta = torch.nn.Parameter(torch.zeros((len(added), 1), device="cuda"))
    boundary_delta = torch.nn.Parameter(torch.zeros((len(boundary), 1), device="cuda"))
    optimizer = torch.optim.Adam([dc_delta, alpha_delta, boundary_delta], lr=args.learning_rate)
    rasters, masks, photos, wall_masks, targets = {}, {}, {}, {}, {}
    for view in training:
        mask, photo = load_view(view, cameras[view],
                                bed_manifest["views"][view]["mask_path"],
                                Path(args.images), args.width)
        h = mask.shape[1]
        config = camera_settings(cameras[view], h, args.width)._replace(sh_degree=3)
        rasterizer = GaussianRasterizer(config)
        wall_entry = wall_manifest["views"].get(view, {})
        if wall_entry.get("accepted"):
            wall = Image.open(wall_entry["mask_path"]).convert("L").resize(
                (args.width, h), Image.Resampling.NEAREST)
            wall = torch.from_numpy((np.asarray(wall)>127).astype(np.float32)).cuda()[None]
        else:
            wall = torch.zeros_like(mask)
        rasters[view], masks[view], photos[view], wall_masks[view] = (
            rasterizer, mask, photo, wall)
        with torch.no_grad():
            # The target is synthesized from this one shared 3D replacement,
            # not independently inpainted 2D frames.
            synthetic_alpha = base_alpha.clone()
            synthetic_alpha[boundary] = 0
            targets[view] = render(model, rasterizer, model.get_features,
                                   synthetic_alpha).detach()
    torch.cuda.reset_peak_memory_stats()
    history = []
    for epoch in range(args.epochs):
        terms = []
        for view in training:
            optimizer.zero_grad(set_to_none=True)
            dc = base_dc.index_copy(0, added,
                base_dc[added]+dc_delta.clamp(-args.max_dc_shift, args.max_dc_shift)[:, None, :])
            shs = torch.cat((dc, base_rest), dim=1)
            alpha = base_alpha.index_copy(0, added,
                torch.sigmoid(base_raw_alpha[added]+alpha_delta.clamp(
                    -args.max_alpha_shift, args.max_alpha_shift)))
            if len(boundary):
                alpha = alpha.index_copy(0, boundary,
                    torch.sigmoid(base_raw_alpha[boundary]+boundary_delta.clamp(
                        -args.max_boundary_shift, args.max_boundary_shift)))
            prediction = render(model, rasters[view], shs, alpha)
            mask = masks[view]
            outside = mean_masked_l1(prediction, photos[view], 1-mask)
            inside = mean_masked_l1(prediction, targets[view], mask)
            observed_wall = mean_masked_l1(prediction, photos[view], wall_masks[view])
            regularizer = (args.color_penalty*dc_delta.square().mean() +
                           args.alpha_penalty*alpha_delta.square().mean() +
                           args.boundary_penalty*boundary_delta.square().mean())
            loss = (args.outside_weight*outside + inside +
                    args.wall_weight*observed_wall + regularizer)
            loss.backward()
            optimizer.step()
            terms.append((float(inside.detach()), float(outside.detach()),
                          float(observed_wall.detach())))
        row = {"epoch": epoch+1, "inside_l1": float(np.mean([t[0] for t in terms])),
               "outside_l1": float(np.mean([t[1] for t in terms])),
               "observed_wall_l1": float(np.mean([t[2] for t in terms]))}
        history.append(row)
        print(json.dumps(row), flush=True)
    result = vertices.copy()
    delta_color = dc_delta.detach().clamp(-args.max_dc_shift, args.max_dc_shift).cpu().numpy()
    for j in range(3):
        result[f"f_dc_{j}"][original_count:] += delta_color[:, j]
    new_alpha = torch.sigmoid(base_raw_alpha[added]+alpha_delta.detach().clamp(
        -args.max_alpha_shift, args.max_alpha_shift)).cpu().numpy()[:, 0]
    result["opacity"][original_count:] = np.log(new_alpha/(1-new_alpha))
    if len(boundary_np):
        bd_alpha = torch.sigmoid(base_raw_alpha[boundary]+boundary_delta.detach().clamp(
            -args.max_boundary_shift, args.max_boundary_shift)).cpu().numpy()[:, 0]
        result["opacity"][boundary_np] = np.log(bd_alpha/(1-bd_alpha))
    output.mkdir(parents=True)
    candidate = output / "candidate.ply"
    write_vertices(candidate, result, source_ply,
                   ["UNAPPROVED local optimization of new fill and mixed boundary splats"])
    np.savez_compressed(output / "optimization-deltas.npz", added_start=original_count,
                        dc_delta=delta_color, boundary_indices=boundary_np,
                        added_alpha_delta=alpha_delta.detach().cpu().numpy(),
                        boundary_alpha_delta=boundary_delta.detach().cpu().numpy())
    report = {"seed": str(Path(args.seed).resolve()), "candidate": str(candidate),
              "training_views": training, "holdout_views": args.holdout_views,
              "added_optimized": len(added), "boundary_optimized": len(boundary),
              "mean_abs_dc_shift": float(np.abs(delta_color).mean()),
              "history": history, "semantic_dimensions": 128,
              "approved": False, "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated()/1024**2}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("added_optimized", "boundary_optimized",
        "elapsed_seconds", "peak_rss_mb", "peak_gpu_allocated_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("seed", "seed-report", "gate-evidence", "cameras", "bed-manifest",
                 "wall-manifest", "images", "output-dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--width", type=int, default=270)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=.05)
    p.add_argument("--hard-gate", type=float, default=.01)
    p.add_argument("--boundary-min-agreement", type=float, default=.55)
    p.add_argument("--max-boundary-splats", type=int, default=3000)
    p.add_argument("--min-training-views", type=int, default=4)
    p.add_argument("--outside-weight", type=float, default=3.)
    p.add_argument("--wall-weight", type=float, default=2.)
    p.add_argument("--color-penalty", type=float, default=.02)
    p.add_argument("--alpha-penalty", type=float, default=.002)
    p.add_argument("--boundary-penalty", type=float, default=.05)
    p.add_argument("--max-dc-shift", type=float, default=.2)
    p.add_argument("--max-alpha-shift", type=float, default=2.)
    p.add_argument("--max-boundary-shift", type=float, default=.5)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
