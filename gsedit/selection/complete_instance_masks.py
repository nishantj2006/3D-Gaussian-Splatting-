"""Conservatively fill missing *separate* instance masks by adjacent-view flow.

Flow is a proposal, not ground truth. Inconsistent or distant views stay
incomplete and are excluded from training; originals are never overwritten.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from PIL import Image


def frame_number(name):
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"View name lacks numeric frame index: {name}") from exc


def photo(images, view, shape):
    matches = [p for p in Path(images).glob(view + ".*") if
               p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(matches) != 1:
        raise ValueError(f"Expected one source photo for {view}")
    bgr = cv2.imread(str(matches[0]))
    return cv2.cvtColor(cv2.resize(bgr, (shape[1], shape[0])), cv2.COLOR_BGR2GRAY)


def flow_proposal(source_gray, target_gray, source_mask, *, fb_tolerance):
    """Pull source mask onto target, rejecting flow-disagreement pixels."""
    t_to_s = cv2.calcOpticalFlowFarneback(target_gray, source_gray, None,
        .5, 4, 21, 5, 7, 1.5, 0)
    s_to_t = cv2.calcOpticalFlowFarneback(source_gray, target_gray, None,
        .5, 4, 21, 5, 7, 1.5, 0)
    yy, xx = np.indices(source_mask.shape, dtype=np.float32)
    sx, sy = xx + t_to_s[..., 0], yy + t_to_s[..., 1]
    mapped = cv2.remap(source_mask.astype(np.uint8), sx, sy,
                       cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    reverse = cv2.remap(s_to_t, sx, sy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
    error = np.linalg.norm(t_to_s + reverse, axis=2)
    in_bounds = (sx >= 0) & (sx < source_mask.shape[1]) & (sy >= 0) & (
        sy < source_mask.shape[0])
    valid = in_bounds & (error <= fb_tolerance)
    return mapped & valid, float(valid[mapped].mean()) if mapped.any() else 0.


def agree(proposals, *, min_iou, min_valid_fraction):
    if not proposals or any(fraction < min_valid_fraction for _, fraction in proposals):
        return None, "flow_inconsistent"
    if len(proposals) == 1:
        return proposals[0][0], "single_adjacent_view"
    a, b = proposals[0][0], proposals[1][0]
    iou = (a & b).sum() / max((a | b).sum(), 1)
    if iou < min_iou:
        return None, f"proposal_disagreement_{iou:.3f}"
    return a & b, f"two_view_agreement_{iou:.3f}"


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    with open(args.manifest, encoding="utf-8") as handle:
        old = json.load(handle)
    labels = sorted(old["instances"])
    holdouts = set(args.holdout_views)
    views = old["views"]
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    new = {"source": old["source"], "instances": old["instances"], "views": {},
           "holdout_views": sorted(holdouts), "approved": False}
    for view, original in sorted(views.items()):
        entry = dict(original)
        entry["instances"] = dict(original.get("instances", {}))
        if view in holdouts or not original.get("accepted"):
            new["views"][view] = entry
            continue
        for label in labels:
            if label in entry["instances"]:
                continue
            sources = sorted((v for v, item in views.items() if v != view and
                item.get("accepted") and label in item.get("instances", {}) and
                v not in holdouts and abs(frame_number(v)-frame_number(view)) <= args.max_gap),
                key=lambda v: (abs(frame_number(v)-frame_number(view)), v))[:2]
            if not sources:
                entry.setdefault("completion_rejections", {})[label] = "no_adjacent_source"
                continue
            shape = np.asarray(Image.open(original["mask_path"]).convert("L")).shape
            target_gray = photo(args.images, view, shape)
            proposed = []
            for source in sources:
                source_gray = photo(args.images, source, shape)
                mask = np.asarray(Image.open(views[source]["instances"][label]["mask_path"])
                                  .convert("L").resize((shape[1], shape[0]),
                                  Image.Resampling.NEAREST)) > 127
                proposed.append(flow_proposal(source_gray, target_gray, mask,
                                               fb_tolerance=args.fb_tolerance))
            accepted, reason = agree(proposed, min_iou=args.min_iou,
                                     min_valid_fraction=args.min_valid_fraction)
            if accepted is None or accepted.mean() < args.min_area or (
                    accepted.mean() > args.max_area):
                entry.setdefault("completion_rejections", {})[label] = reason
                continue
            path = output / "masks" / f"{view}-{label}-flow.png"
            Image.fromarray(accepted.astype(np.uint8)*255).save(path)
            entry["instances"][label] = {"mask_path": str(path),
                                          "fraction": float(accepted.mean()),
                                          "source_views": sources, "method": "optical_flow",
                                          "confidence": reason}
        entry["complete"] = all(label in entry["instances"] for label in labels)
        if entry["complete"]:
            masks = [np.asarray(Image.open(entry["instances"][label]["mask_path"])
                                .convert("L")) > 127 for label in labels]
            if any(mask.shape != masks[0].shape for mask in masks):
                raise ValueError(f"Incompatible instance masks in {view}")
            # Ambiguous overlap is reported, not silently assigned to either instance.
            overlap = np.logical_and.reduce(masks)
            entry["instance_overlap_pixels"] = int(overlap.sum())
            union = np.logical_or.reduce(masks)
            path = output / "masks" / f"{view}-union.png"
            Image.fromarray(union.astype(np.uint8)*255).save(path)
            entry["mask_path"] = str(path)
            entry["mask_fraction"] = float(union.mean())
        new["views"][view] = entry
    new["incomplete_views"] = sorted(v for v, d in new["views"].items() if
                                     d.get("accepted") and not d.get("complete"))
    new["completed_views"] = sorted(v for v, d in new["views"].items() if
                                    d.get("complete") and not views[v].get("complete"))
    new["elapsed_seconds"] = time.perf_counter()-started
    new["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(new, handle, indent=2)
        handle.write("\n")
    return new


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--holdout-views", nargs="+", default=[])
    p.add_argument("--max-gap", type=int, default=2)
    p.add_argument("--fb-tolerance", type=float, default=2.)
    p.add_argument("--min-iou", type=float, default=.5)
    p.add_argument("--min-valid-fraction", type=float, default=.8)
    p.add_argument("--min-area", type=float, default=.003)
    p.add_argument("--max-area", type=float, default=.6)
    return p


if __name__ == "__main__":
    report = run(parser().parse_args())
    print(json.dumps({k: report[k] for k in ("completed_views", "incomplete_views",
                                             "elapsed_seconds", "peak_rss_mb")}, indent=2))
