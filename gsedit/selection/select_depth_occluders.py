"""Select foreground splats in front of a fitted masked wall/floor background.

Candidate splats must overlap the bed mask by their rasterized radius, lie a
safe depth margin in front of the background plane, and agree across views.
Independently protected frame IDs are never selected. Preview only.
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
from utils.ply_semantic_utils import read_vertices, write_vertices


def background_depth(camera, pixels, shape, origin, frame, wall_normal, wall_offset,
                     max_depth=100.):
    height, width = shape
    xy = np.asarray(pixels, dtype=np.float64)
    fx = camera["fx"]*width/camera["width"]
    fy = camera["fy"]*height/camera["height"]
    rays = np.column_stack(((xy[:, 0]-width/2)/fx,
                            (xy[:, 1]-height/2)/fy, np.ones(len(xy))))
    rays = rays @ np.asarray(camera["rotation"], dtype=np.float64).T
    camera_xyz = np.asarray(camera["position"], dtype=np.float64)
    floor_normal = frame[:, 2]
    wall_world_normal = frame[:, :2] @ wall_normal
    with np.errstate(divide="ignore", invalid="ignore"):
        floor_t = ((origin-camera_xyz) @ floor_normal)/(rays @ floor_normal)
        wall_t = (wall_offset-((camera_xyz-origin) @ frame)[:2] @ wall_normal)/(
            rays @ wall_world_normal)
    floor_t[~np.isfinite(floor_t) | (floor_t <= .01) | (floor_t >= max_depth)] = np.inf
    wall_t[~np.isfinite(wall_t) | (wall_t <= .01) | (wall_t >= max_depth)] = np.inf
    wall_xyz = camera_xyz + np.where(np.isfinite(wall_t), wall_t, 0)[:, None]*rays
    wall_h = ((wall_xyz-origin) @ frame)[:, 2]
    wall_t[(wall_h < 0) | (wall_h > 10.)] = np.inf
    return np.minimum(floor_t, wall_t)


def source_to_candidate_ids(source_count, selected_source, source_ids):
    keep = np.ones(source_count, dtype=bool)
    keep[selected_source] = False
    inverse = np.full(source_count, -1, dtype=np.int64)
    inverse[np.flatnonzero(keep)] = np.arange(keep.sum())
    return inverse[source_ids]


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    source_ply, source = read_vertices(args.source)
    seed_ply, seed = read_vertices(args.seed)
    selected_source = np.unique(np.load(args.selected_indices, allow_pickle=False))
    original_count = len(source)-len(selected_source)
    if seed.dtype != source.dtype or len(seed) <= original_count:
        raise ValueError("Source and seed PLY do not align")
    pool = np.unique(np.load(args.candidate_indices, allow_pickle=False))
    if not len(pool) or pool.min() < 0 or pool.max() >= original_count:
        raise ValueError("Invalid footprint pool")
    protected_source = np.unique(np.load(args.protected_source_indices, allow_pickle=False))
    protected_candidate = source_to_candidate_ids(len(source), selected_source, protected_source)
    protected_candidate = protected_candidate[protected_candidate >= 0]
    pool = np.setdiff1d(pool, protected_candidate)
    with open(args.wall_fit, encoding="utf-8") as handle:
        fit = json.load(handle)
    if fit["inlier_ratio"] < args.min_wall_inlier_ratio or fit["supported_views"] < 5:
        raise ValueError("Wall fit is unreliable")
    origin = np.asarray(fit["origin"])
    frame = np.asarray(fit["frame"])
    wall_normal = np.asarray(fit["wall_normal_floor_xy"])
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    with open(args.mask_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    views = sorted(v for v, d in manifest["views"].items() if d.get("accepted")
                   and v in cameras and v not in args.holdout_views)
    if len(views) < args.min_views:
        raise ValueError("Too few views")
    model = GaussianModel(3, 128)
    model.load_ply(args.seed)
    xyz = model.get_xyz[pool].detach().cpu().numpy()
    votes = np.zeros(len(pool), dtype=np.uint16)
    gap_sum = np.zeros(len(pool), dtype=np.float64)
    support = {}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for view in views:
            mask_image = Image.open(manifest["views"][view]["mask_path"]).convert("L")
            height = round(cameras[view]["height"]*args.width/cameras[view]["width"])
            mask = np.asarray(mask_image.resize((args.width, height),
                                                 Image.Resampling.NEAREST)) > 127
            rasterizer = GaussianRasterizer(camera_settings(cameras[view], height,
                args.width)._replace(sh_degree=3))
            _, _, radii, _ = rasterizer(
                means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
                shs=model.get_features, colors_precomp=None,
                semantic_feature=model.get_semantic_feature,
                opacities=model.get_opacity, scales=model.get_scaling,
                rotations=model.get_rotation, cov3D_precomp=None)
            radius = radii[torch.as_tensor(pool, device="cuda")].cpu().numpy()
            x, y, depth, valid = project(xyz, cameras[view], mask.shape)
            eligible = valid & (radius > 0)
            candidate_local = np.flatnonzero(eligible)
            distance, nearest = distance_transform_edt(~mask, return_indices=True)
            close = candidate_local[distance[y[candidate_local], x[candidate_local]] <=
                                    radius[candidate_local]]
            if len(close):
                pixel_x = nearest[1, y[close], x[close]]
                pixel_y = nearest[0, y[close], x[close]]
                backdrop = background_depth(cameras[view], np.column_stack((pixel_x, pixel_y)),
                    mask.shape, origin, frame, wall_normal, fit["wall_offset"])
                gap = backdrop-depth[close]
                passes = np.isfinite(gap) & (gap >= args.min_depth_gap)
                ids = close[passes]
                votes[ids] += 1
                gap_sum[ids] += gap[passes]
                support[view] = int(len(ids))
            else:
                support[view] = 0
            print(json.dumps({"view": view, "depth_supported": support[view]}), flush=True)
    chosen = (votes >= args.min_views) & (
        gap_sum/np.maximum(votes, 1) >= args.min_mean_depth_gap)
    opacity_before = 1/(1+np.exp(-np.asarray(seed["opacity"][pool], dtype=np.float64)))
    chosen &= opacity_before >= args.min_existing_opacity
    if chosen.sum() == 0:
        raise ValueError("No safely supported foreground occluders")
    if chosen.sum()/original_count > args.max_scene_fraction:
        raise ValueError("Depth selection exceeds scene safety fraction")
    result = seed.copy()
    result["opacity"][pool[chosen]] = np.log(args.minimum_opacity/(1-args.minimum_opacity))
    output.mkdir(parents=True)
    candidate = output / "candidate.ply"
    write_vertices(candidate, result, seed_ply,
                   ["UNAPPROVED depth-guided foreground suppression; frame IDs protected"])
    np.savez_compressed(output / "depth-evidence.npz", candidate_indices=pool,
                        votes=votes, mean_depth_gap=gap_sum/np.maximum(votes, 1),
                        selected=chosen, protected_candidate_indices=protected_candidate)
    report = {"source": str(Path(args.source).resolve()),
              "seed": str(Path(args.seed).resolve()), "candidate": str(candidate),
              "pool_splats": len(pool), "protected_excluded": len(protected_candidate),
              "newly_suppressed": int(chosen.sum()),
              "selection_fraction": float(chosen.sum()/original_count),
              "min_views": args.min_views, "min_depth_gap": args.min_depth_gap,
              "mean_selected_depth_gap": float((gap_sum[chosen]/votes[chosen]).mean()),
              "view_support": support, "approved": False,
              "elapsed_seconds": time.perf_counter()-started,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated()/1024**2}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("pool_splats", "protected_excluded",
        "newly_suppressed", "elapsed_seconds", "peak_rss_mb",
        "peak_gpu_allocated_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "seed", "selected-indices", "candidate-indices",
                 "protected-source-indices", "wall-fit", "cameras",
                 "mask-manifest", "output-dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--width", type=int, default=270)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--min-depth-gap", type=float, default=.4)
    p.add_argument("--min-mean-depth-gap", type=float, default=.6)
    p.add_argument("--min-existing-opacity", type=float, default=.01)
    p.add_argument("--minimum-opacity", type=float, default=1e-4)
    p.add_argument("--min-wall-inlier-ratio", type=float, default=.3)
    p.add_argument("--max-scene-fraction", type=float, default=.1)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
