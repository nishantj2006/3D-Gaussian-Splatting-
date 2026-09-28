"""Refine a seeded 3D object using MobileSAM masks from several source views.

This is a preview-only, post-training pass. It never modifies the source PLY.
Seed indices can come from semantic selection or another object proposal; SAM
supplies image boundaries, while geometry and cross-view agreement constrain
which Gaussians receive the instance ID.
"""

from gsedit.runtime import PROJECT_ROOT, module_command

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from utils.ply_semantic_utils import add_float_property, read_vertices, write_vertices


def project(points, camera, shape):
    """Return pixel coordinates, camera depth, and in-frame visibility."""
    height, width = shape
    local = (points - np.asarray(camera["position"])) @ np.asarray(camera["rotation"])
    depth = local[:, 2]
    safe_depth = np.maximum(depth, 1e-6)
    x = np.rint(local[:, 0] / safe_depth * camera["fx"] * width /
                 camera["width"] + width / 2).astype(np.int32)
    y = np.rint(local[:, 1] / safe_depth * camera["fy"] * height /
                 camera["height"] + height / 2).astype(np.int32)
    valid = (depth > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x, y, depth, valid


def seed_component(points, indices, eps=0.5):
    """Discard stray seed clusters before constructing image prompts."""
    labels = DBSCAN(eps=eps, min_samples=5, n_jobs=-1).fit_predict(points[indices])
    counts = np.bincount(labels[labels >= 0])
    if not len(counts) or counts.max() < 100:
        raise ValueError("Seed indices have no coherent 3D component")
    return indices[labels == counts.argmax()]


def seed_box(x, y, valid, shape, padding=0.06):
    if valid.sum() < 30:
        raise ValueError("Too few seed splats are visible in a requested view")
    height, width = shape
    x0, x1 = np.quantile(x[valid], [0.01, 0.99])
    y0, y1 = np.quantile(y[valid], [0.01, 0.99])
    dx, dy = max((x1 - x0) * padding, 8), max((y1 - y0) * padding, 8)
    box = [max(0, int(x0 - dx)), max(0, int(y0 - dy)),
           min(width - 1, int(x1 + dx)), min(height - 1, int(y1 + dy))]
    if box[2] - box[0] < 10 or box[3] - box[1] < 10:
        raise ValueError("Seed projection is too small for an instance mask")
    return box


def evidence_for_view(points, seeds, camera, mask, max_depth_delta):
    """Mask membership plus nearby seed depth prevents selecting hidden wall/floor."""
    shape = mask.shape
    sx, sy, sd, sv = project(seeds, camera, shape)
    x, y, depth, valid = project(points, camera, shape)
    hit = np.zeros(len(points), dtype=bool)
    visible_ids = np.flatnonzero(valid)
    visible_ids = visible_ids[mask[y[visible_ids], x[visible_ids]]]
    if not len(visible_ids) or not sv.any():
        return hit
    seed_pixels = np.column_stack((sx[sv], sy[sv]))
    pixel_distance, neighbor = cKDTree(seed_pixels).query(
        np.column_stack((x[visible_ids], y[visible_ids])), workers=-1)
    seed_depth = sd[sv][neighbor]
    hit[visible_ids] = ((pixel_distance <= 45) &
                        (np.abs(depth[visible_ids] - seed_depth) <= max_depth_delta))
    return hit


def combine_evidence(points, seed_indices, view_evidence, max_distance, min_views):
    """Retain trusted seeds; admit nearby splats only with multi-view support."""
    if len(view_evidence) < min_views:
        raise ValueError("Not enough camera views for requested agreement")
    votes = np.sum(np.stack(view_evidence), axis=0)
    distance = cKDTree(points[seed_indices]).query(points, workers=-1)[0]
    selected = (votes >= min_views) & (distance <= max_distance)
    selected[seed_indices] = True
    return selected, votes


def load_photo(path, max_width):
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.width > max_width:
            image = image.resize((max_width, round(image.height * max_width / image.width)))
        return np.asarray(image)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    if len(set(args.views)) != len(args.views) or len(args.views) < 2:
        raise ValueError("Provide at least two distinct camera views")
    if args.min_views < 2 or args.min_views > len(args.views):
        raise ValueError("--min-views must be between 2 and the number of views")
    if args.max_distance <= 0 or args.max_depth_delta <= 0:
        raise ValueError("Distance tolerances must be positive")
    ply, vertices = read_vertices(args.scene)
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float64)
    indices = np.load(args.seed_indices, allow_pickle=False)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Seed indices must be a one-dimensional integer array")
    if len(indices) < 100 or (indices < 0).any() or (indices >= len(points)).any():
        raise ValueError("Seed indices are empty, too sparse, or outside the source scene")
    indices = np.unique(indices)
    core = seed_component(points, indices)
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    missing = set(args.views) - set(cameras)
    if missing:
        raise ValueError(f"Unknown camera views: {sorted(missing)}")
    photos = {}
    for view in args.views:
        matches = list(Path(args.images).glob(view + ".*"))
        if len(matches) != 1:
            raise ValueError(f"Expected one source image for {view}")
        photos[view] = load_photo(matches[0], args.max_width)

    from ultralytics import SAM  # Delay model import until inputs have been validated.
    sam = SAM(args.model)
    masks = {}
    evidence = []
    details = {}
    for view in args.views:
        image = photos[view]
        shape = image.shape[:2]
        sx, sy, _, sv = project(points[core], cameras[view], shape)
        box = seed_box(sx, sy, sv, shape)
        result = sam.predict(source=image, bboxes=box, imgsz=args.model_size,
                             device=args.device, verbose=False)[0]
        if result.masks is None or len(result.masks.data) != 1:
            raise ValueError(f"SAM did not produce one object mask for {view}")
        mask = result.masks.data[0].cpu().numpy().astype(bool)
        if mask.shape != shape:
            mask = np.asarray(Image.fromarray(mask).resize(
                (shape[1], shape[0]), Image.Resampling.NEAREST))
        seed_recall = float(mask[sy[sv], sx[sv]].mean())
        area_fraction = float(mask.mean())
        if seed_recall < args.min_seed_recall or area_fraction > args.max_mask_fraction:
            raise ValueError(f"Unreliable mask in {view}: seed recall={seed_recall:.2f}, "
                             f"image fraction={area_fraction:.2f}; no preview written")
        masks[view] = mask
        hit = evidence_for_view(points, points[core], cameras[view], mask,
                                args.max_depth_delta)
        evidence.append(hit)
        details[view] = {"sam_box": box, "mask_fraction": area_fraction,
                         "seed_recall": seed_recall, "supported_splats": int(hit.sum())}

    selected, votes = combine_evidence(points, indices, evidence,
                                       args.max_distance, args.min_views)
    if selected.mean() > args.max_scene_fraction:
        raise ValueError("Instance selection is too broad; no preview written")
    output.mkdir(parents=True)
    for view, mask in masks.items():
        overlay = photos[view].copy()
        overlay[mask] = np.rint(0.55 * overlay[mask] +
                                 0.45 * np.array([255, 45, 45])).astype(np.uint8)
        Image.fromarray(overlay).save(output / f"{view}-mask.png")
    np.save(output / "selected-indices.npy", np.flatnonzero(selected))
    labeled = add_float_property(vertices, "object_id")
    labeled["object_id"] = 0
    labeled["object_id"][selected] = args.instance_id
    write_vertices(output / "instance-labeled-preview.ply", labeled, ply,
                   ["unreviewed multi-view instance label preview"])
    write_vertices(output / "pruned-preview.ply", vertices[~selected].copy(), ply,
                   ["unreviewed multi-view instance removal preview"])
    report = {"source": str(Path(args.scene).resolve()), "seed_indices": str(Path(args.seed_indices).resolve()),
              "model": args.model, "instance_id": args.instance_id,
              "initial_seed_points": int(len(indices)), "coherent_seed_points": int(len(core)),
              "selected_points": int(selected.sum()), "added_to_seed": int((selected & ~np.isin(
                  np.arange(len(points)), indices)).sum()),
              "selected_fraction": float(selected.mean()), "vote_counts": np.bincount(votes).tolist(),
              "views": details, "approved": False,
              "warning": "Masks and removed scene require visual review; hidden background is not reconstructed."}
    with open(output / "preview.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not args.skip_render:
        renderer = module_command("render_ply_preview.py")
        for view in args.views:
            subprocess.run([*renderer, "--ply", str(output / "pruned-preview.ply"),
                            "--cameras", args.cameras, "--image-name", view,
                            "--output", str(output / f"{view}-pruned.png"),
                            "--width", str(args.render_width)], check=True)
    print(json.dumps({"output": str(output), "selected_points": report["selected_points"],
                      "added_to_seed": report["added_to_seed"], "approved": False}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--seed-indices", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--model", default="mobile_sam.pt")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--model-size", type=int, default=640)
    p.add_argument("--max-width", type=int, default=540)
    p.add_argument("--instance-id", type=int, default=44)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--max-distance", type=float, default=0.65)
    p.add_argument("--max-depth-delta", type=float, default=0.85)
    p.add_argument("--min-seed-recall", type=float, default=0.35)
    p.add_argument("--max-mask-fraction", type=float, default=0.65)
    p.add_argument("--max-scene-fraction", type=float, default=0.20)
    p.add_argument("--render-width", type=int, default=540)
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
