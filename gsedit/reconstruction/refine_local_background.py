"""Locally attenuate revealed splats against a cross-view background seed.

Only opacity of original splats close to a reviewed removal is optimized. The
seed PLY is never changed; synthetic targets inside the bed masks are rendered
with that local pool hidden, while original photos supervise the exterior.
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

from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from scene.gaussian_model import GaussianModel
from utils.ply_semantic_utils import read_vertices, write_vertices


def candidate_pool(source_points, selected, seed_points, original_count, radius):
    if radius <= 0 or original_count > len(seed_points):
        raise ValueError("Invalid local pool geometry")
    kept = np.ones(len(source_points), dtype=bool)
    kept[selected] = False
    if kept.sum() != original_count:
        raise ValueError("Seed does not match source minus selection")
    distance = cKDTree(source_points[selected]).query(seed_points[:original_count], workers=-1)[0]
    return np.flatnonzero(distance <= radius)


def render_rgb(model, rasterizer, opacity):
    image, _, _, _ = rasterizer(
        means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
        shs=model.get_features, colors_precomp=None,
        semantic_feature=model.get_semantic_feature,
        opacities=opacity, scales=model.get_scaling,
        rotations=model.get_rotation, cov3D_precomp=None)
    return image.clamp(0, 1)


def load_view(view, entry, mask_path, images, width):
    mask = Image.open(mask_path).convert("L")
    height = round(entry["height"] * width / entry["width"])
    mask = np.asarray(mask.resize((width, height), Image.Resampling.NEAREST)) > 127
    photo_path = next((images / (view + ext) for ext in (".jpg", ".png", ".jpeg")
                       if (images / (view + ext)).exists()), None)
    if photo_path is None:
        raise FileNotFoundError(f"No source photo for {view}")
    photo = np.asarray(Image.open(photo_path).convert("RGB").resize(
        (width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
    return (torch.from_numpy(mask.astype(np.float32)).cuda()[None],
            torch.from_numpy(photo).cuda().permute(2, 0, 1))


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    started = time.perf_counter()
    _, source = read_vertices(args.source)
    seed_ply, vertices = read_vertices(args.seed)
    if source.dtype.names != vertices.dtype.names:
        raise ValueError("Source and seed PLY schemas differ")
    selected = np.unique(np.load(args.selected_indices, allow_pickle=False))
    if len(selected) == 0 or selected.min() < 0 or selected.max() >= len(source):
        raise ValueError("Invalid source selection")
    original_count = len(source) - len(selected)
    source_xyz = np.column_stack([source[k] for k in ("x", "y", "z")])
    seed_xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")])
    pool_np = candidate_pool(source_xyz, selected, seed_xyz, original_count,
                             args.max_seed_distance)
    if not len(pool_np) or len(pool_np) > args.max_pool_splats:
        raise ValueError(f"Unsafe pool size: {len(pool_np)}")
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    with open(args.mask_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    holdouts = set(args.holdout_views)
    for view in holdouts:
        if view not in cameras or not manifest["views"].get(view, {}).get("accepted"):
            raise ValueError(f"Missing accepted holdout: {view}")
    training = sorted(v for v, d in manifest["views"].items()
                      if d.get("accepted") and v in cameras and v not in holdouts)
    if len(training) < args.min_training_views:
        raise ValueError("Insufficient training views")
    model = GaussianModel(3, 128)
    model.load_ply(str(args.seed))
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                 "_scaling", "_rotation", "_semantic_feature"):
        getattr(model, name).requires_grad_(False)
    pool = torch.as_tensor(pool_np, device="cuda", dtype=torch.long)
    base_opacity = model.get_opacity.detach()
    hidden_opacity = base_opacity.index_fill(0, pool, 0.)
    settings = {}
    masks = {}
    photos = {}
    targets = {}
    for view in training + sorted(holdouts):
        mask, photo = load_view(view, cameras[view],
                                manifest["views"][view]["mask_path"],
                                Path(args.images), args.width)
        height = round(cameras[view]["height"] * args.width / cameras[view]["width"])
        config = camera_settings(cameras[view], height, args.width)
        # The native renderer expects all 16 SH coefficients from a 5k model.
        config = config._replace(sh_degree=3)
        settings[view] = GaussianRasterizer(config)
        masks[view], photos[view] = mask, photo
        with torch.no_grad():
            targets[view] = render_rgb(model, settings[view], hidden_opacity).detach()
    logits = torch.nn.Parameter(torch.full((len(pool_np), 1), args.initial_logit,
                                           device="cuda"))
    optimizer = torch.optim.Adam([logits], lr=args.learning_rate)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for epoch in range(args.epochs):
        epoch_losses = []
        for view in training:
            optimizer.zero_grad(set_to_none=True)
            gate = torch.sigmoid(logits)
            opacity = base_opacity.index_copy(0, pool, base_opacity[pool] * gate)
            image = render_rgb(model, settings[view], opacity)
            mask = masks[view]
            inside = ((image-targets[view]).abs()*mask).sum() / (mask.sum()*3+1)
            outside = ((image-photos[view]).abs()*(1-mask)).sum() / (
                (1-mask).sum()*3+1)
            loss = inside + args.outside_weight*outside + args.gate_penalty*(1-gate).mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append((float(inside.detach()), float(outside.detach())))
        losses.append({"epoch": epoch+1, "inside_l1": float(np.mean([x[0] for x in epoch_losses])),
                       "outside_l1": float(np.mean([x[1] for x in epoch_losses]))})
        print(json.dumps(losses[-1]), flush=True)
    gate = torch.sigmoid(logits).detach().flatten().cpu().numpy()
    new_vertices = vertices.copy()
    changed_indices = pool_np[gate < 1]
    changed_gates = gate[gate < 1]
    base = 1/(1+np.exp(-np.asarray(vertices["opacity"][changed_indices], dtype=np.float64)))
    adjusted = np.clip(base*changed_gates, 1e-5, 1-1e-5)
    new_vertices["opacity"][changed_indices] = np.log(adjusted/(1-adjusted))
    output.mkdir(parents=True)
    candidate = output / "candidate.ply"
    write_vertices(candidate, new_vertices, seed_ply,
                   ["UNAPPROVED local opacity-refinement preview; source preserved"])
    np.savez_compressed(output / "local-gates.npz", indices=pool_np, gate=gate)
    report = {"source": str(Path(args.source).resolve()), "seed": str(Path(args.seed).resolve()),
              "candidate": str(candidate), "training_views": training,
              "holdout_views": sorted(holdouts), "pool_splats": len(pool_np),
              "attenuated_below_half": int((gate < .5).sum()),
              "gate_quantiles": np.quantile(gate, [0,.1,.5,.9,1]).tolist(),
              "losses": losses, "approved": False,
              "elapsed_seconds": time.perf_counter()-started,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated()/1024**2,
              "warning": "Synthetic hidden-surface target is plausible, not ground truth."}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("pool_splats", "attenuated_below_half",
        "elapsed_seconds", "peak_rss_mb", "peak_gpu_allocated_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--selected-indices", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--width", type=int, default=270)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=.25)
    p.add_argument("--initial-logit", type=float, default=3.)
    p.add_argument("--outside-weight", type=float, default=3.)
    p.add_argument("--gate-penalty", type=float, default=.01)
    p.add_argument("--max-seed-distance", type=float, default=1.)
    p.add_argument("--max-pool-splats", type=int, default=20000)
    p.add_argument("--min-training-views", type=int, default=4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
