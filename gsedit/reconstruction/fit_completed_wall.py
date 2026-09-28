"""Test whether aligned inpainted depth can complete one wall plane.

Generated depths propose a wall; real wall pixels in training photos and
untouched held-out views decide whether that proposal is geometrically sound.
No replacement Gaussians are produced by this diagnostic.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

from gsedit.assets.align_asset import plane_frame
from gsedit.reconstruction.crossview_background import backproject
from gsedit.reconstruction.fit_depth_wall import sample_view
from gsedit.reconstruction.fit_occluded_surfaces import line_ransac
from scene.gaussian_model import GaussianModel


def plane_support(points, origin, frame, normal, offset, threshold):
    local = (points-origin) @ frame
    error = np.abs(local[:, :2] @ normal-offset)
    return float((error <= threshold).mean()), float(np.quantile(error, .9))


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    with open(args.depth_report, encoding="utf-8") as handle:
        prior_report = json.load(handle)
    with open(args.floor_fit, encoding="utf-8") as handle:
        floor = json.load(handle)
    with open(args.wall_manifest, encoding="utf-8") as handle:
        wall = json.load(handle)["views"]
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {x["img_name"]: x for x in json.load(handle)}
    origin, frame = plane_frame(floor["plane_origin"],
                                floor["plane_normal_toward_removed_object"])
    rng = np.random.default_rng(args.seed)
    prior_points, prior_views = [], []
    for view, entry in sorted(prior_report["views"].items()):
        if view in args.holdout_views or not entry.get("accepted"):
            continue
        depth = np.load(Path(args.depth_report).parent/f"{view}-prior-depth.npy",
                        allow_pickle=False)
        layout = np.asarray(Image.open(Path(args.layouts)/f"{view}-layout.png").convert("RGB"))
        if depth.shape != layout.shape[:2]:
            raise ValueError(f"Layout and prior dimensions differ in {view}")
        wall_prior = np.all(layout == np.array([150, 80, 30]), axis=2)
        yy, xx = np.where(wall_prior & np.isfinite(depth) & (depth > 0))
        if len(xx) > args.max_prior_pixels_per_view:
            chosen = rng.choice(len(xx), args.max_prior_pixels_per_view, replace=False)
            xx, yy = xx[chosen], yy[chosen]
        if len(xx) < args.min_prior_pixels_per_view:
            continue
        xyz = backproject(cameras[view], np.column_stack((xx, yy)),
                          depth[yy, xx], depth.shape)
        local = (xyz-origin) @ frame
        good = (local[:, 2] >= args.min_height) & (local[:, 2] <= args.max_height)
        if good.sum() >= args.min_prior_pixels_per_view:
            prior_points.append(xyz[good])
            prior_views.append(view)
    report = {"prior_views": prior_views, "holdout_views": {},
              "geometry_supported": False, "approved": False,
              "source": str(Path(args.scene).resolve()),
              "note": "Generated depth is a prior, not observed hidden geometry."}
    if len(prior_views) < args.min_prior_views:
        report["reason"] = "too_few_aligned_prior_views"
    else:
        prior_xyz = np.concatenate(prior_points)
        local = (prior_xyz-origin) @ frame
        normal, offset, inliers, q90 = line_ransac(local[:, :2],
            threshold=args.plane_threshold, trials=args.trials, seed=args.seed)
        report.update(origin=origin.tolist(), frame=frame.tolist(),
                      wall_normal_floor_xy=normal.tolist(), wall_offset=float(offset),
                      prior_points=len(prior_xyz), prior_inlier_ratio=float(inliers.mean()),
                      prior_error_q90=float(q90))
        # The proposal has to explain real, visible wall geometry, not only its
        # own generated images. Use heldouts solely after fitting the plane.
        model = GaussianModel(3, 128)
        model.load_ply(args.scene)
        real = {}
        for view, entry in sorted(wall.items()):
            if not entry.get("accepted") or view not in cameras:
                continue
            mask = np.asarray(Image.open(entry["mask_path"]).convert("L")) > 127
            xyz, _, _ = sample_view(model, cameras[view], mask,
                np.zeros((*mask.shape, 3), np.float32), args.real_samples_per_view, rng)
            local_xyz = (xyz-origin) @ frame
            xyz = xyz[(local_xyz[:, 2] >= args.min_height) &
                      (local_xyz[:, 2] <= args.max_height)]
            if len(xyz) < args.min_real_samples_per_view:
                real[view] = {"supported": False, "samples": len(xyz),
                              "reason": "sparse_real_wall"}
                continue
            ratio, error = plane_support(xyz, origin, frame, normal, offset,
                                         args.plane_threshold)
            real[view] = {"supported": ratio >= args.min_real_inlier_ratio and
                          error <= args.max_real_q90, "samples": len(xyz),
                          "inlier_ratio": ratio, "error_q90": error}
        train = {v: d for v, d in real.items() if v not in args.holdout_views}
        held = {v: real.get(v, {"supported": False, "reason": "no_real_wall_mask"})
                for v in args.holdout_views}
        report["training_real_views"] = train
        report["heldout_views"] = held
        report["supported_views"] = sum(d["supported"] for d in held.values())
        report["training_real_supported_views"] = sum(d["supported"] for d in train.values())
        report["geometry_supported"] = bool(
            float(inliers.mean()) >= args.min_prior_inlier_ratio and
            report["training_real_supported_views"] >= args.min_real_training_views and
            report["supported_views"] >= args.min_heldout_views)
        if not report["geometry_supported"]:
            report["reason"] = "generated_wall_conflicts_with_real_depth"
    output.mkdir(parents=True)
    report["elapsed_seconds"] = time.perf_counter()-started
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    with open(output/"report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "depth-report", "layouts", "floor-fit", "wall-manifest",
                 "cameras", "output-dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--min-prior-views", type=int, default=3)
    p.add_argument("--min-prior-pixels-per-view", type=int, default=200)
    p.add_argument("--max-prior-pixels-per-view", type=int, default=3000)
    p.add_argument("--real-samples-per-view", type=int, default=4000)
    p.add_argument("--min-real-samples-per-view", type=int, default=500)
    p.add_argument("--min-height", type=float, default=.5)
    p.add_argument("--max-height", type=float, default=10.)
    p.add_argument("--plane-threshold", type=float, default=.3)
    p.add_argument("--trials", type=int, default=800)
    p.add_argument("--min-prior-inlier-ratio", type=float, default=.3)
    p.add_argument("--min-real-inlier-ratio", type=float, default=.3)
    p.add_argument("--max-real-q90", type=float, default=.6)
    p.add_argument("--min-real-training-views", type=int, default=3)
    p.add_argument("--min-heldout-views", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
