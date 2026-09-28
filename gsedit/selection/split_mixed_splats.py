"""Preview-only geometric split of broad mixed foreground/background splats.

Each source splat is retained in provenance. Four smaller daughters cover its
two broadest covariance axes, and rasterized per-view attribution decides which
daughters can be suppressed. Ambiguous daughters are reported, not approved.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from gsedit.selection.refine_revealed_layers import view_contributions
from scene.gaussian_model import GaussianModel
from utils.ply_semantic_utils import read_vertices, write_vertices


def split_record(record, *, shift=.45, scale_factor=.6):
    scales = np.exp(np.array([record[f"scale_{j}"] for j in range(3)], np.float64))
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Invalid Gaussian scale")
    quat = np.array([record[f"rot_{j}"] for j in range(4)], np.float64)
    quat /= max(np.linalg.norm(quat), 1e-10)
    axes = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    broad = np.argsort(scales)[-2:]
    xyz = np.array([record[key] for key in ("x", "y", "z")], np.float64)
    alpha = 1/(1+np.exp(-float(record["opacity"])))
    daughter_alpha = 1-(1-alpha)**.25
    daughters = np.repeat(np.asarray(record)[None], 4).copy()
    for i, (a, b) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        offset = (a*shift*scales[broad[0]]*axes[:, broad[0]] +
                  b*shift*scales[broad[1]]*axes[:, broad[1]])
        for j, key in enumerate(("x", "y", "z")):
            daughters[key][i] = xyz[j] + offset[j]
        for axis in broad:
            daughters[f"scale_{axis}"][i] = np.log(scales[axis]*scale_factor)
        daughters["opacity"][i] = np.log(daughter_alpha/(1-daughter_alpha))
    return daughters


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    header, vertices = read_vertices(args.seed)
    with np.load(args.evidence, allow_pickle=False) as data:
        ids = data["indices"]
        gate = data["gate"]
    if len(ids) != len(gate):
        raise ValueError("Evidence shape mismatch")
    chosen = np.unique(ids[(gate > args.hard_gate) & (gate < 1)])
    if not len(chosen) or len(chosen) > args.max_splits or chosen.min() < 0 or (
            chosen.max() >= len(vertices)):
        raise ValueError(f"Unsafe mixed-splat count: {len(chosen)}")
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {x["img_name"]: x for x in json.load(handle)}
    with open(args.mask_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    training = sorted(v for v in manifest["views"] if
                      manifest["views"][v].get("accepted") and v in cameras and
                      v not in args.holdout_views)
    if len(training) < args.min_training_views:
        raise ValueError("Too few training masks")
    daughter = np.concatenate([split_record(vertices[i], shift=args.shift,
                            scale_factor=args.scale_factor) for i in chosen])
    initial = vertices.copy()
    original_alpha = np.clip(1/(1+np.exp(-initial["opacity"][chosen])), 1e-8, 1-1e-8)
    initial["opacity"][chosen] = np.log(args.minimum_opacity/(1-args.minimum_opacity))
    split = np.concatenate((initial, daughter))
    output.mkdir(parents=True)
    initial_path = output / "split-initial.ply"
    write_vertices(initial_path, split, header,
                   ["UNAPPROVED split of mixed boundary splats; original is unchanged"])
    model = GaussianModel(3, 128)
    model.load_ply(str(initial_path))
    for key in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling",
                "_rotation", "_semantic_feature"):
        getattr(model, key).requires_grad_(False)
    pool_np = np.arange(len(vertices), len(split), dtype=np.int64)
    pool = torch.as_tensor(pool_np, dtype=torch.long, device="cuda")
    inside = np.zeros(len(pool_np), np.float64)
    outside = np.zeros(len(pool_np), np.float64)
    support = np.zeros(len(pool_np), np.uint8)
    torch.cuda.reset_peak_memory_stats()
    for view in training:
        camera = cameras[view]
        height = round(camera["height"]*args.width/camera["width"])
        mask = np.asarray(Image.open(manifest["views"][view]["mask_path"])
                          .convert("L").resize((args.width, height),
                          Image.Resampling.NEAREST)) > 127
        raster = GaussianRasterizer(camera_settings(camera, height, args.width)._replace(
            sh_degree=3))
        contribution, exterior = view_contributions(model, raster,
            model.get_opacity.detach(), pool,
            torch.from_numpy(mask.astype(np.float32)).cuda())
        inside += np.maximum(contribution, 0)
        outside += np.maximum(exterior, 0)
        support += (contribution >= args.min_view_contribution).astype(np.uint8)
    ratio = inside/np.maximum(inside+outside, 1e-8)
    suppress = ((inside >= args.min_total_contribution) &
                (support >= args.min_support_views) & (ratio >= args.min_agreement))
    result = split.copy()
    result["opacity"][pool_np[suppress]] = np.log(args.minimum_opacity/(1-args.minimum_opacity))
    candidate = output / "candidate.ply"
    write_vertices(candidate, result, header,
                   ["UNAPPROVED split/attributed mixed-splat preview; source preserved"])
    np.savez_compressed(output / "evidence.npz", source_indices=chosen,
                        source_opacity=original_alpha, daughter_indices=pool_np,
                        inside=inside.astype(np.float32), outside=outside.astype(np.float32),
                        support=support, suppressed=suppress)
    report = {"source": str(Path(args.seed).resolve()), "candidate": str(candidate),
              "split_sources": len(chosen), "daughters": len(daughter),
              "suppressed_daughters": int(suppress.sum()),
              "unresolved_daughters": int(((inside + outside) < args.min_total_contribution).sum()),
              "training_views": training, "holdout_views": args.holdout_views,
              "semantic_dimensions": 128, "approved": False,
              "elapsed_seconds": time.perf_counter()-started,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated()/1024**2}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("seed", "evidence", "cameras", "mask-manifest", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--holdout-views", nargs="+", default=[])
    p.add_argument("--width", type=int, default=270)
    p.add_argument("--max-splits", type=int, default=3000)
    p.add_argument("--hard-gate", type=float, default=.01)
    p.add_argument("--shift", type=float, default=.45)
    p.add_argument("--scale-factor", type=float, default=.6)
    p.add_argument("--minimum-opacity", type=float, default=1e-5)
    p.add_argument("--min-training-views", type=int, default=3)
    p.add_argument("--min-total-contribution", type=float, default=.15)
    p.add_argument("--min-view-contribution", type=float, default=.01)
    p.add_argument("--min-support-views", type=int, default=1)
    p.add_argument("--min-agreement", type=float, default=.7)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
