"""Fit a wall from exact Gaussian contributions to independent wall masks."""

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


def contribution(model, rasterizer, mask):
    weights = torch.ones(len(model.get_xyz), device="cuda", requires_grad=True)
    colors = weights[:, None].expand(-1, 3)
    image, _, _, _ = rasterizer(
        means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
        shs=None, colors_precomp=colors,
        semantic_feature=model.get_semantic_feature,
        opacities=model.get_opacity, scales=model.get_scaling,
        rotations=model.get_rotation, cov3D_precomp=None)
    gradient = torch.autograd.grad((image[0]*mask).sum(), weights)[0]
    return gradient.detach().cpu().numpy()


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse wall-fit folder: {output}")
    start = time.perf_counter()
    model = GaussianModel(args.sh_degree, args.semantic_dimensions)
    model.load_ply(args.scene)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                 "_scaling", "_rotation", "_semantic_feature"):
        getattr(model, name).requires_grad_(False)
    points = model.get_xyz.detach().cpu().numpy().astype(np.float64)
    with open(args.floor_plane, encoding="utf-8") as handle:
        ground = json.load(handle)
    origin, frame = plane_frame(ground["plane_origin"],
                                ground["plane_normal_toward_removed_object"])
    local = (points-origin) @ frame
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {x["img_name"]: x for x in json.load(handle)}
    with open(args.wall_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    training = sorted(set(view for view, item in manifest["views"].items()
                          if item["accepted"])-set(args.holdout_views))
    if len(training) < args.min_training_views:
        raise ValueError("Too few clean wall masks")
    selected = np.unique(np.load(args.selected_indices, allow_pickle=False))
    total = np.zeros(len(points), dtype=np.float64)
    votes = np.zeros(len(points), dtype=np.uint16)
    view_support = {}
    torch.cuda.reset_peak_memory_stats()
    for view in training:
        mask = np.asarray(Image.open(manifest["views"][view]["mask_path"])
                          .convert("L")) > 127
        rasterizer = GaussianRasterizer(camera_settings(cameras[view], *mask.shape))
        target = torch.from_numpy(mask.astype(np.float32)).cuda()
        gain = contribution(model, rasterizer, target)
        total += gain
        votes += (gain >= args.min_view_contribution).astype(np.uint16)
        view_support[view] = int((gain >= args.min_view_contribution).sum())
        print(json.dumps({"view": view, "support": view_support[view]}), flush=True)
    candidate = (votes >= args.min_views) & (local[:, 2] >= args.min_height)
    candidate[selected] = False
    ids = np.flatnonzero(candidate)
    if len(ids) < args.min_wall_points:
        raise ValueError(f"Only {len(ids)} wall-supported Gaussians")
    normal, offset, inliers, error = line_ransac(local[ids, :2],
        threshold=args.plane_threshold, trials=args.trials)
    wall_ids = ids[inliers]
    ratio = len(wall_ids)/len(ids)
    if len(wall_ids) < args.min_wall_points or ratio < args.min_inlier_ratio:
        raise ValueError(f"Wall plane insufficient: {len(wall_ids)} inliers, {ratio:.2%}")
    output.mkdir(parents=True)
    np.save(output / "wall-donor-indices.npy", wall_ids)
    np.savez_compressed(output / "wall-evidence.npz", votes=votes,
                        contribution=total.astype(np.float32))
    report = {"source": str(Path(args.scene).resolve()),
              "origin": origin.tolist(), "frame": frame.tolist(),
              "wall_normal_floor_xy": normal.tolist(), "wall_offset": offset,
              "wall_candidates": len(ids), "wall_donors": len(wall_ids),
              "inlier_ratio": ratio, "plane_error_q90": error,
              "training_views": training, "holdout_views": args.holdout_views,
              "view_support": view_support,
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated()/1024**2,
              "approved": False}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("wall_candidates", "wall_donors",
        "inlier_ratio", "plane_error_q90", "elapsed_seconds",
        "peak_rss_mb", "peak_gpu_allocated_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--selected-indices", required=True)
    p.add_argument("--floor-plane", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--wall-manifest", required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--min-wall-points", type=int, default=100)
    p.add_argument("--min-training-views", type=int, default=3)
    p.add_argument("--min-inlier-ratio", type=float, default=.15)
    p.add_argument("--min-height", type=float, default=.5)
    p.add_argument("--min-view-contribution", type=float, default=.1)
    p.add_argument("--plane-threshold", type=float, default=.08)
    p.add_argument("--trials", type=int, default=700)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--semantic-dimensions", type=int, default=128)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
