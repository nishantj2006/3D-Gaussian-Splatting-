"""Build an independent, conservative wooden-frame protection from SAM masks.

The selected headboard mask is kept. Additional 'wooden bed frame' candidates
contribute only an outside-bed strip adjacent to the lower bed silhouette;
this excludes the spurious upper-wall pixels seen in broad detector masks.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

from gsedit.reconstruction.build_local_background import mask_votes
from utils.ply_semantic_utils import read_vertices


def lower_frame_strip(candidate, bed_mask, below_margin=60, above_margin=8):
    if candidate.shape != bed_mask.shape:
        raise ValueError("Frame and bed mask shapes differ")
    height, width = candidate.shape
    bottom = np.max(np.where(bed_mask, np.arange(height)[:, None], -1), axis=0)
    yy = np.arange(height)[:, None]
    near_lower_edge = ((bottom[None, :] >= 0) &
                       (yy >= bottom[None, :]-above_margin) &
                       (yy <= bottom[None, :]+below_margin))
    return candidate & ~bed_mask & near_lower_edge


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    with open(args.frame_manifest, encoding="utf-8") as handle:
        frames = json.load(handle)
    with open(args.bed_manifest, encoding="utf-8") as handle:
        beds = json.load(handle)
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    _, vertices = read_vertices(args.source)
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")])
    selected = np.unique(np.load(args.selected_indices, allow_pickle=False))
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    (output / "overlays").mkdir()
    report = {"source": str(Path(args.source).resolve()),
              "frame_manifest": str(Path(args.frame_manifest).resolve()),
              "bed_manifest": str(Path(args.bed_manifest).resolve()),
              "views": {}, "approved": False}
    for view, detail in frames["views"].items():
        if not detail.get("accepted") or view not in cameras or not beds["views"].get(view, {}).get("accepted"):
            report["views"][view] = {"accepted": False}
            continue
        bed = np.asarray(Image.open(beds["views"][view]["mask_path"]).convert("L")) > 127
        headboard = np.asarray(Image.open(detail["mask_path"]).convert("L")) > 127
        if bed.shape != headboard.shape:
            raise ValueError(f"Mask dimensions differ for {view}")
        protected = headboard.copy()
        strip_parts = []
        for candidate in detail["candidates"]:
            if ("wooden bed frame" not in candidate["label"].lower() or
                    not candidate.get("mask_path")):
                continue
            candidate_mask = np.asarray(Image.open(candidate["mask_path"]).convert("L")) > 127
            strip = lower_frame_strip(candidate_mask, bed, args.below_margin,
                                      args.above_margin)
            strip_parts.append(strip)
            protected |= strip
        if protected.mean() > args.max_protected_fraction:
            report["views"][view] = {"accepted": False,
                                      "rejection": "mask too broad"}
            continue
        path = output / "masks" / f"{view}-frame.png"
        Image.fromarray((protected*255).astype(np.uint8)).save(path)
        matches = list(Path(args.images).glob(view+".*"))
        if len(matches) == 1:
            image = np.asarray(Image.open(matches[0]).convert("RGB").resize(
                (bed.shape[1], bed.shape[0]))).copy()
            image[protected] = np.rint(.55*image[protected]+.45*np.array([255, 25, 25])).astype(np.uint8)
            Image.fromarray(image).save(output / "overlays" / f"{view}.png")
        report["views"][view] = {"accepted": True, "mask_path": str(path),
                                   "headboard_fraction": float(headboard.mean()),
                                   "lower_strip_fraction": float(np.logical_or.reduce(strip_parts).mean())
                                   if strip_parts else 0.,
                                   "protected_fraction": float(protected.mean())}
    valid_views = [v for v, d in report["views"].items() if d["accepted"]]
    if len(valid_views) < args.min_views:
        raise ValueError("Insufficient independent frame masks")
    votes = mask_votes(points, cameras, report, valid_views)
    protected_ids = np.flatnonzero(votes >= args.min_votes_per_splat)
    overlap = np.intersect1d(protected_ids, selected)
    np.save(output / "protected-source-indices.npy", protected_ids)
    np.save(output / "removal-conflicts.npy", overlap)
    report.update(accepted_views=len(valid_views),
                  protected_source_splats=len(protected_ids),
                  removed_source_conflicts=len(overlap),
                  warning="Center-vote 3D lift is conservative and not an instance ground truth.",
                  elapsed_seconds=time.perf_counter()-start,
                  peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: report[k] for k in ("accepted_views", "protected_source_splats",
        "removed_source_conflicts", "elapsed_seconds", "peak_rss_mb")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frame-manifest", "bed-manifest", "cameras", "source",
                 "selected-indices", "images", "output-dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--below-margin", type=int, default=60)
    p.add_argument("--above-margin", type=int, default=8)
    p.add_argument("--max-protected-fraction", type=float, default=.3)
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--min-votes-per-splat", type=int, default=2)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
