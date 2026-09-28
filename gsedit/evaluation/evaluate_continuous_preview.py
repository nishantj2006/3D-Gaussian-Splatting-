"""Measure local edit integrity without treating image metrics as approval."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

from utils.ply_semantic_utils import read_vertices


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)/255.


def compare_images(before, after, mask):
    difference = np.abs(after-before).mean(axis=2)
    return {"inside_l1_change": float(difference[mask].mean()),
            "outside_l1_change": float(difference[~mask].mean()),
            "outside_fraction_above_0.1": float((difference[~mask]>.1).mean()),
            "inside_near_black_fraction": float((after[mask].max(axis=1)<.02).mean())}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    _, source = read_vertices(args.source)
    _, baseline = read_vertices(args.baseline)
    _, candidate = read_vertices(args.candidate)
    selected = np.unique(np.load(args.selected_indices, allow_pickle=False))
    keep = np.ones(len(source), dtype=bool)
    keep[selected] = False
    original_count = keep.sum()
    if baseline.dtype != source.dtype or candidate.dtype != source.dtype:
        raise ValueError("PLY schema changed")
    if len(baseline) < original_count or len(candidate) < original_count:
        raise ValueError("Candidate lost original splats")
    semantic_keys = [k for k in source.dtype.names if k.startswith("semantic_")]
    geometry_keys = [k for k in source.dtype.names if k in ("x", "y", "z")]
    preserve = all(np.array_equal(candidate[k][:original_count], baseline[k][:original_count])
                   for k in source.dtype.names if k not in
                   ("opacity", "f_dc_0", "f_dc_1", "f_dc_2"))
    initial_positions_unchanged = all(np.array_equal(candidate[k][:original_count],
                                                       source[k][keep])
                                      for k in geometry_keys)
    with open(args.wall_fit, encoding="utf-8") as handle:
        fit = json.load(handle)
    origin = np.asarray(fit["origin"])
    frame = np.asarray(fit["frame"])
    normal = np.asarray(fit["wall_normal_floor_xy"])
    with open(args.candidate_report, encoding="utf-8") as handle:
        candidate_report = json.load(handle)
    floor_count = candidate_report["floor_added"]
    added_xyz = np.column_stack([candidate[k][original_count:] for k in geometry_keys])
    local = (added_xyz-origin) @ frame
    floor_error = np.abs(local[:floor_count, 2])
    wall_error = np.abs(local[floor_count:, :2] @ normal-fit["wall_offset"])
    with open(args.bed_manifest, encoding="utf-8") as handle:
        bed_manifest = json.load(handle)
    frame_manifest = None
    if args.frame_manifest:
        with open(args.frame_manifest, encoding="utf-8") as handle:
            frame_manifest = json.load(handle)
    views = {}
    for view in args.views:
        before = load_rgb(Path(args.baseline_renders)/f"{view}.png")
        after = load_rgb(Path(args.candidate_renders)/f"{view}.png")
        if before.shape != after.shape:
            raise ValueError("RGB render shapes differ")
        mask = np.asarray(Image.open(bed_manifest["views"][view]["mask_path"])
                          .convert("L").resize((after.shape[1], after.shape[0]),
                                                Image.Resampling.NEAREST)) > 127
        row = compare_images(before, after, mask)
        if frame_manifest and frame_manifest["views"].get(view, {}).get("accepted"):
            frame_mask = np.asarray(Image.open(frame_manifest["views"][view]["mask_path"])
                                    .convert("L").resize((after.shape[1], after.shape[0]),
                                                          Image.Resampling.NEAREST)) > 127
            if frame_mask.any():
                row["protected_frame_l1_change"] = float(np.abs(
                    after[frame_mask]-before[frame_mask]).mean())
                row["protected_frame_mask_fraction"] = float(frame_mask.mean())
        views[view] = row
    report = {"source": str(Path(args.source).resolve()),
              "baseline": str(Path(args.baseline).resolve()),
              "candidate": str(Path(args.candidate).resolve()),
              "original_kept": int(original_count),
              "candidate_gaussians": len(candidate),
              "property_count": len(candidate.dtype.names),
              "semantic_dimensions": len(semantic_keys),
              "original_positions_unchanged": initial_positions_unchanged,
              "nonappearance_original_properties_preserved": preserve,
              "floor_plane_error_q99": float(np.quantile(floor_error, .99)),
              "wall_plane_error_q99": float(np.quantile(wall_error, .99)),
              "views": views, "approved": False,
              "warning": "Numerical checks cannot establish that the bed is gone or hidden wall texture is plausible.",
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    output.mkdir(parents=True)
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "baseline", "candidate", "selected-indices",
                 "wall-fit", "candidate-report", "bed-manifest",
                 "baseline-renders", "candidate-renders", "output-dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--frame-manifest")
    p.add_argument("--views", nargs="+", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
