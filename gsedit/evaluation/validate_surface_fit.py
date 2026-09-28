"""Validate a fitted vertical wall against untouched camera views.

Training masks are never reused for this check. Held-out rendered source depth
inside independently detected wall masks must agree with the proposed plane.
The report is written even for an unsupported fit; the caller must stop.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

from gsedit.reconstruction.fit_depth_wall import sample_view
from scene.gaussian_model import GaussianModel


def plane_errors(xyz, fit):
    origin = np.asarray(fit["origin"], dtype=np.float64)
    frame = np.asarray(fit["frame"], dtype=np.float64)
    normal = np.asarray(fit["wall_normal_floor_xy"], dtype=np.float64)
    local = (xyz-origin) @ frame
    return np.abs(local[:, :2] @ normal-fit["wall_offset"])


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    with open(args.wall_fit, encoding="utf-8") as handle:
        fit = json.load(handle)
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    with open(args.wall_manifest, encoding="utf-8") as handle:
        masks = json.load(handle)["views"]
    model = GaussianModel(3, 128)
    model.load_ply(args.scene)
    rng = np.random.default_rng(args.seed)
    views = {}
    for view in args.holdout_views:
        entry = masks.get(view, {})
        if not entry.get("accepted") or view not in cameras:
            views[view] = {"supported": False, "reason": "No held-out wall mask"}
            continue
        mask = np.asarray(Image.open(entry["mask_path"]).convert("L")) > 127
        sample, _, count = sample_view(model, cameras[view], mask,
            np.zeros((*mask.shape, 3), dtype=np.float32), args.samples_per_view, rng)
        if len(sample) < args.min_samples_per_view:
            views[view] = {"supported": False, "samples": len(sample),
                           "reason": "Too few depth-supported wall pixels"}
            continue
        errors = plane_errors(sample, fit)
        ratio = float((errors <= args.plane_threshold).mean())
        q90 = float(np.quantile(errors, .9))
        views[view] = {"supported": ratio >= args.min_inlier_ratio and
                      q90 <= args.max_q90_error, "samples": count,
                      "inlier_ratio": ratio, "error_q90": q90}
    good = sum(value.get("supported", False) for value in views.values())
    supported = good >= args.min_supported_views
    output.mkdir(parents=True)
    report = {"wall_fit": str(Path(args.wall_fit).resolve()),
              "heldout_views": views, "supported_views": good,
              "geometry_supported": supported, "approved": False,
              "elapsed_seconds": time.perf_counter()-started,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    with open(output/"report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "wall-fit", "wall-manifest", "cameras", "output-dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--samples-per-view", type=int, default=5000)
    p.add_argument("--min-samples-per-view", type=int, default=500)
    p.add_argument("--plane-threshold", type=float, default=.3)
    p.add_argument("--min-inlier-ratio", type=float, default=.3)
    p.add_argument("--max-q90-error", type=float, default=.6)
    p.add_argument("--min-supported-views", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
