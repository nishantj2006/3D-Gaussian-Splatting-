"""Conservatively lift text-grounded image masks onto existing Gaussians.

The CPU lift samples projected anisotropic Gaussian footprints with a local
depth check. It accumulates both positive and negative visible-view evidence,
then optionally grows connected, mask-supported points. This is a preview,
not a full alpha-composited renderer or an approved deletion.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from gsedit.evaluation.benchmark_object_selection import footprint_cache, grow_graph
from gsedit.selection.multiview_instance import project
from utils.ply_semantic_utils import numbered_fields, read_vertices, write_vertices


def sample_ellipse(cache, mask, count):
    """Five weighted screen-space samples of each depth-visible Gaussian."""
    ids, x, y, axes, visible = cache
    height, width = mask.shape
    first_x, first_y, second_x, second_y = axes.T
    coverage = np.zeros(len(ids), dtype=np.float32)
    for dx, dy, weight in ((0, 0, .4), (first_x, first_y, .15),
                           (-first_x, -first_y, .15), (second_x, second_y, .15),
                           (-second_x, -second_y, .15)):
        xx = np.rint(x + dx).astype(np.int32)
        yy = np.rint(y + dy).astype(np.int32)
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        coverage[inside] += weight * mask[yy[inside], xx[inside]]
    return ids, coverage, visible


def consensus(positive, negative, seed, points, *, min_views=2,
              min_agreement=.65, max_seed_distance=.5):
    """Require repeated positive observations and reject contradictory views."""
    total = positive.astype(np.float32) + negative
    ratio = positive / np.maximum(total, 1)
    nearby = cKDTree(points[seed]).query(points, workers=-1)[0] <= max_seed_distance
    chosen = (positive >= min_views) & (ratio >= min_agreement) & nearby
    chosen[seed] = True
    return chosen, ratio


def evaluate_view(selected, points, camera, mask, *, radius=5):
    """Automatic-mask agreement proxies; these are not ground-truth accuracy."""
    x, y, depth, valid = project(points, camera, mask.shape)
    ids = np.flatnonzero(valid)
    selected_ids = ids[selected[ids]]
    if not len(selected_ids):
        return {"visible_selected": 0, "inside_mask_fraction": 0.0,
                "mask_pixel_coverage": 0.0}
    inside = float(mask[y[selected_ids], x[selected_ids]].mean())
    centers = np.zeros(mask.shape, dtype=bool)
    centers[y[selected_ids], x[selected_ids]] = True
    covered = distance_transform_edt(~centers) <= radius
    pixel_coverage = float(covered[mask].mean()) if mask.any() else 0.0
    return {"visible_selected": int(len(selected_ids)),
            "inside_mask_fraction": inside, "mask_pixel_coverage": pixel_coverage}


def overlay(path, image, points, camera, seed, selected):
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    seed_mask = np.zeros(len(points), dtype=bool)
    seed_mask[seed] = True
    for ids, color, radius in ((seed, (0, 255, 255), 1),
                               (np.flatnonzero(selected & ~seed_mask), (255, 25, 25), 2)):
        x, y, _, valid = project(points[ids], camera, (canvas.height, canvas.width))
        for xx, yy in zip(x[valid], y[valid]):
            draw.ellipse((int(xx)-radius, int(yy)-radius,
                          int(xx)+radius, int(yy)+radius), fill=color)
    canvas.save(path)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    start = time.perf_counter()
    source, vertices = read_vertices(args.scene)
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float32)
    scales = np.column_stack([vertices[f"scale_{i}"] for i in range(3)]).astype(np.float32)
    rotations = np.column_stack([vertices[f"rot_{i}"] for i in range(4)]).astype(np.float32)
    opacity = vertices["opacity"].astype(np.float32)
    seed = np.unique(np.load(args.seed_indices, allow_pickle=False))
    baseline = np.unique(np.load(args.baseline_indices, allow_pickle=False))
    if not len(seed) or seed.min() < 0 or seed.max() >= len(points):
        raise ValueError("Invalid seed selection")
    if not len(baseline) or baseline.min() < 0 or baseline.max() >= len(points):
        raise ValueError("Invalid baseline selection")
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    with open(args.mask_manifest, encoding="utf-8") as handle:
        mask_report = json.load(handle)
    available = {view: detail for view, detail in mask_report["views"].items()
                 if detail["accepted"] and view in cameras}
    holdouts = set(args.holdout_views)
    if not holdouts.issubset(available):
        raise ValueError(f"Holdout masks are missing: {sorted(holdouts-set(available))}")
    training = sorted(set(available) - holdouts)
    if len(training) < args.min_views:
        raise ValueError("Too few training masks")
    positive = np.zeros(len(points), dtype=np.uint16)
    negative = np.zeros(len(points), dtype=np.uint16)
    view_details = {}
    for view in training:
        mask = np.asarray(Image.open(available[view]["mask_path"]).convert("L")) > 127
        cache = footprint_cache(points, scales, rotations, opacity,
                                cameras[view], mask.shape,
                                max_radius=args.max_footprint_px)
        ids, coverage, visible = sample_ellipse(cache, mask, len(points))
        positive[ids[visible & (coverage >= args.positive_coverage)]] += 1
        negative[ids[visible & (coverage <= args.negative_coverage)]] += 1
        view_details[view] = {"positive_splats": int((visible & (coverage >= args.positive_coverage)).sum()),
                              "negative_splats": int((visible & (coverage <= args.negative_coverage)).sum())}
    selected, agreement = consensus(positive, negative, seed, points,
        min_views=args.min_views, min_agreement=args.min_agreement,
        max_seed_distance=args.max_seed_distance)
    baseline_mask = np.zeros(len(points), dtype=bool)
    baseline_mask[baseline] = True
    selected |= baseline_mask
    if selected.mean() > args.max_scene_fraction:
        raise ValueError("Consensus selection exceeds scene safety limit")
    support = positive > 0
    rgb = (np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]) *
           .2820947918 + .5).astype(np.float32)
    fields = numbered_fields(vertices.dtype.names, "semantic_")
    features = np.column_stack([vertices[name] for name in fields]).astype(np.float32)
    grown, growth = grow_graph(points, rgb, features, seed, selected, support,
                               radius=args.graph_radius,
                               max_seed_distance=args.max_seed_distance,
                               rounds=args.graph_rounds)
    if grown.mean() > args.max_scene_fraction:
        raise ValueError("Graph selection exceeds scene safety limit")
    output.mkdir(parents=True)
    np.save(output / "consensus-indices.npy", np.flatnonzero(selected))
    np.save(output / "graph-indices.npy", np.flatnonzero(grown))
    np.save(output / "positive-votes.npy", positive)
    np.save(output / "negative-votes.npy", negative)
    metrics = {}
    for view in args.holdout_views:
        mask = np.asarray(Image.open(available[view]["mask_path"]).convert("L")) > 127
        camera = cameras[view]
        metrics[view] = {}
        for name, selection in (("baseline", baseline_mask), ("consensus", selected),
                                ("graph", grown)):
            metrics[view][name] = evaluate_view(selection, points, camera, mask)
        matches = [p for p in Path(args.images).glob(view + ".*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if len(matches) != 1:
            raise ValueError(f"Expected one source image for {view}")
        with Image.open(matches[0]) as source_image:
            image = source_image.convert("RGB")
            if image.width > mask.shape[1]:
                image = image.resize((mask.shape[1], mask.shape[0]))
            image = np.asarray(image)
        overlay(output / f"{view}-consensus.png", image, points, camera, seed, selected)
        overlay(output / f"{view}-graph.png", image, points, camera, seed, grown)
    report = {"source": str(Path(args.scene).resolve()),
              "mask_manifest": str(Path(args.mask_manifest).resolve()),
              "training_views": training, "holdout_views": args.holdout_views,
              "seed_points": int(len(seed)), "baseline_points": int(len(baseline)),
              "consensus_points": int(selected.sum()), "graph_points": int(grown.sum()),
              "consensus_added_vs_baseline": int((selected & ~baseline_mask).sum()),
              "graph_added_vs_baseline": int((grown & ~baseline_mask).sum()),
              "growth": growth, "view_details": view_details,
              "heldout_mask_agreement": metrics,
              "semantic_dimensions": len(fields), "approved": False,
              "warning": "Heldout SAM 2 masks are automatic pseudo-labels, not manual ground truth. No PLY written until visual review.",
              "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    with open(output / "preview.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if args.write_ply:
        write_vertices(output / "consensus-pruned-preview.ply", vertices[~selected].copy(),
                       source, ["unapproved grounded-mask consensus preview"])
        write_vertices(output / "graph-pruned-preview.ply", vertices[~grown].copy(),
                       source, ["unapproved grounded-mask graph preview"])
    print(json.dumps({k: report[k] for k in ("seed_points", "baseline_points", "consensus_points",
                      "graph_points", "consensus_added_vs_baseline", "graph_added_vs_baseline",
                      "elapsed_seconds", "peak_rss_mb", "approved")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--seed-indices", required=True)
    p.add_argument("--baseline-indices", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--holdout-views", nargs="+", required=True)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--min-agreement", type=float, default=.65)
    p.add_argument("--positive-coverage", type=float, default=.55)
    p.add_argument("--negative-coverage", type=float, default=.15)
    p.add_argument("--max-footprint-px", type=float, default=10)
    p.add_argument("--max-seed-distance", type=float, default=.5)
    p.add_argument("--graph-radius", type=float, default=.08)
    p.add_argument("--graph-rounds", type=int, default=4)
    p.add_argument("--max-scene-fraction", type=float, default=.2)
    p.add_argument("--write-ply", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
