"""Conservative, multi-part whole-object removal preview from an existing seed.

Each SAM proposal remains a separate candidate. Cross-view overlap associates
proposals; proximity to the seed connects object parts only for the requested
whole-object operation. The source PLY is never modified.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, minimum_filter
from scipy.spatial import cKDTree

from gsedit.selection.auto_object_preview import choose_views, score_masks
from gsedit.selection.multiview_instance import project
from utils.ply_semantic_utils import add_float_property, read_vertices, write_vertices
from utils.semantic_utils import load_clip


def projected_hits(points, camera, mask, scales, *, depth_tolerance=0.25,
                   footprint_sigma=1.5, max_footprint_px=10):
    """Visible splats whose center or bounded projected footprint intersects a mask."""
    x, y, depth, valid = project(points, camera, mask.shape)
    ids = np.flatnonzero(valid)
    hits = np.zeros(len(points), dtype=bool)
    if not len(ids):
        return hits
    width = mask.shape[1]
    pixel = y[ids] * width + x[ids]
    front = np.full(mask.size, np.inf, dtype=np.float32)
    np.minimum.at(front, pixel, depth[ids])
    distance = distance_transform_edt(~mask)
    focal = float(camera["fx"]) * width / float(camera["width"])
    # Compare to nearby foreground centers as well: a hidden center may sit
    # on an otherwise empty pixel beside a broad foreground Gaussian.
    local_front = minimum_filter(front.reshape(mask.shape),
                                 size=2 * int(np.ceil(max_footprint_px)) + 1)
    footprint = np.minimum(max_footprint_px, footprint_sigma *
                           np.exp(scales[ids]) * focal / np.maximum(depth[ids], 1e-6))
    # A bounded footprint is an approximation, not a full anisotropic splat render.
    hits[ids] = ((distance[y[ids], x[ids]] <= footprint) &
                 (depth[ids] <= local_front[y[ids], x[ids]] + depth_tolerance))
    return hits


def associate_parts(proposals, seed_ids, points, *, overlap=0.12,
                    seed_distance=0.32, min_seed_neighbors=30):
    """Associate recurring masks, retaining separate connected part IDs."""
    if not proposals:
        return [], []
    parent = list(range(len(proposals)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, first in enumerate(proposals):
        for j in range(i + 1, len(proposals)):
            second = proposals[j]
            if first["view"] == second["view"]:
                continue
            shared = np.intersect1d(first["ids"], second["ids"], assume_unique=True).size
            union = len(first["ids"]) + len(second["ids"]) - shared
            size_ratio = max(len(first["ids"]), len(second["ids"])) / min(len(first["ids"]), len(second["ids"]))
            if shared / union >= overlap and size_ratio <= 4:
                parent[root(j)] = root(i)
    groups = {}
    for i in range(len(proposals)):
        groups.setdefault(root(i), []).append(i)
    tree = cKDTree(points[seed_ids])
    parts = []
    for members in groups.values():
        raw_ids = np.unique(np.concatenate([proposals[i]["ids"] for i in members]))
        views = sorted({proposals[i]["view"] for i in members})
        votes = np.zeros(len(points), dtype=np.uint8)
        for view in views:
            view_ids = np.unique(np.concatenate(
                [proposals[i]["ids"] for i in members if proposals[i]["view"] == view]))
            votes[view_ids] += 1
        ids = raw_ids[votes[raw_ids] >= min(2, len(views))]
        raw_count = len(raw_ids)
        near = tree.query(points[ids], workers=-1)[0] <= seed_distance
        ids = ids[near]
        if not len(ids):
            continue
        seed_overlap = np.intersect1d(ids, seed_ids, assume_unique=True).size
        parts.append({"ids": ids, "members": members, "views": views,
                      "raw_points": raw_count,
                      "seed_overlap": int(seed_overlap), "near_seed": int(near.sum()),
                      "connected": bool(seed_overlap or near.sum() >= min_seed_neighbors)})
    parts.sort(key=lambda part: (-part["seed_overlap"], -part["near_seed"], -len(part["ids"])))
    return parts, [root(i) for i in range(len(proposals))]


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    source, vertices = read_vertices(args.scene)
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float64)
    seed_ids = np.unique(np.load(args.seed_indices, allow_pickle=False))
    if seed_ids.ndim != 1 or not len(seed_ids) or seed_ids.min() < 0 or seed_ids.max() >= len(points):
        raise ValueError("Invalid seed indices")
    scales = np.max(np.column_stack([vertices[f"scale_{i}"] for i in range(3)]), axis=1)
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    model, processor = load_clip("cpu")
    views, scores = choose_views(args.images, cameras, args.text, model, processor,
                                 count=args.view_count, min_baseline=args.min_baseline)
    from ultralytics import SAM
    sam = SAM(args.sam_model)
    proposals, images, masks_by_view = [], {}, {}
    for view in views:
        matches = [p for p in Path(args.images).glob(view + ".*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one source image for {view}")
        with Image.open(matches[0]) as photo:
            image = photo.convert("RGB")
            if image.width > args.image_width:
                image = image.resize((args.image_width,
                                      round(image.height * args.image_width / image.width)))
            image = np.asarray(image)
        images[view] = image
        result = sam.predict(source=image, imgsz=args.sam_size,
                             device=args.device, verbose=False)[0]
        if result.masks is None:
            masks_by_view[view] = []
            continue
        masks = result.masks.data.cpu().numpy().astype(bool)
        try:
            _, ranking = score_masks(image, masks, args.text, model, processor)
        except ValueError:
            masks_by_view[view] = []
            continue
        best = ranking[0]["score"]
        masks_by_view[view] = []
        for item in ranking:
            if item["score"] < best - args.score_drop:
                continue
            mask = masks[item["mask_index"]]
            if mask.shape != image.shape[:2]:
                mask = np.asarray(Image.fromarray(mask).resize(
                    (image.shape[1], image.shape[0]), Image.Resampling.NEAREST))
            ids = np.flatnonzero(projected_hits(points, cameras[view], mask, scales,
                depth_tolerance=args.depth_tolerance,
                max_footprint_px=args.max_footprint_px))
            if len(ids) < args.min_proposal_points:
                continue
            proposal = {"view": view, "mask_index": item["mask_index"],
                        "score": item["score"], "area_fraction": item["area_fraction"],
                        "ids": ids}
            masks_by_view[view].append((len(proposals), mask))
            proposals.append(proposal)
    parts, _ = associate_parts(proposals, seed_ids, points,
        overlap=args.part_overlap, seed_distance=args.seed_distance,
        min_seed_neighbors=args.min_seed_neighbors)
    selected = np.zeros(len(points), dtype=bool)
    selected[seed_ids] = True
    labels = np.zeros(len(points), dtype=np.float32)
    labels[seed_ids] = 1
    part_report = []
    for number, part in enumerate(parts, 2):
        independent = len(part["views"]) >= args.min_part_views
        near_fraction = part["near_seed"] / part["raw_points"]
        safe_size = len(part["ids"]) <= args.max_part_fraction * len(points)
        include = (part["connected"] and independent and safe_size and
                   near_fraction >= args.min_near_fraction)
        if include:
            selected[part["ids"]] = True
            free_ids = part["ids"][labels[part["ids"]] == 0]
            labels[free_ids] = number
        part_report.append({"part_id": number, "views": part["views"],
                            "proposal_count": len(part["members"]),
                            "points": int(len(part["ids"])),
                            "seed_overlap": part["seed_overlap"],
                            "raw_points": int(part["raw_points"]),
                            "near_seed": part["near_seed"],
                            "near_fraction": float(near_fraction),
                            "safe_size": bool(safe_size),
                            "included": include,
                            "proposals": [{"view": proposals[i]["view"],
                                           "mask_index": int(proposals[i]["mask_index"]),
                                           "score": float(proposals[i]["score"])}
                                          for i in part["members"]]})
    if selected.mean() > args.max_scene_fraction:
        raise ValueError("Selection exceeds scene safety limit; no preview written")
    output.mkdir(parents=True)
    for view in views:
        overlay = images[view].copy()
        for index, mask in masks_by_view.get(view, []):
            number = next((j for j, part in enumerate(parts, 2)
                           if index in part["members"]), None)
            if number is None:
                continue
            color = np.array([(37 * number + 210) % 256,
                              (97 * number + 40) % 256, (151 * number + 70) % 256])
            overlay[mask] = np.rint(.55 * overlay[mask] + .45 * color).astype(np.uint8)
        Image.fromarray(overlay).save(output / f"{view}-parts.png")
    np.save(output / "whole-object-indices.npy", np.flatnonzero(selected))
    labeled = add_float_property(vertices, "part_id")
    labeled["part_id"] = labels
    write_vertices(output / "part-labeled-preview.ply", labeled, source,
                   ["unreviewed independent part IDs"])
    write_vertices(output / "pruned-preview.ply", vertices[~selected].copy(), source,
                   ["unreviewed multi-part whole-object removal preview"])
    report = {"source": str(Path(args.scene).resolve()), "text": args.text,
              "seed_points": int(len(seed_ids)), "selected_points": int(selected.sum()),
              "added_points": int(selected.sum() - len(seed_ids)),
              "selected_fraction": float(selected.mean()),
              "views": [{"name": view, "clip_score": scores[view]} for view in views],
              "parts": part_report, "approved": False,
              "warning": "Approximate footprint and instance associations require visual review; hidden background is unfilled."}
    with open(output / "preview.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), "seed_points": len(seed_ids),
                      "selected_points": int(selected.sum()),
                      "included_parts": sum(p["included"] for p in part_report),
                      "candidate_parts": len(parts), "approved": False}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--seed-indices", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--sam-model", default="mobile_sam.pt")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--view-count", type=int, default=5)
    p.add_argument("--min-baseline", type=float, default=0.35)
    p.add_argument("--image-width", type=int, default=540)
    p.add_argument("--sam-size", type=int, default=640)
    p.add_argument("--score-drop", type=float, default=0.06)
    p.add_argument("--depth-tolerance", type=float, default=0.25)
    p.add_argument("--max-footprint-px", type=float, default=10)
    p.add_argument("--min-proposal-points", type=int, default=75)
    p.add_argument("--part-overlap", type=float, default=0.12)
    p.add_argument("--seed-distance", type=float, default=0.32)
    p.add_argument("--min-seed-neighbors", type=int, default=30)
    p.add_argument("--min-part-views", type=int, default=2)
    p.add_argument("--min-near-fraction", type=float, default=0.2)
    p.add_argument("--max-part-fraction", type=float, default=0.04)
    p.add_argument("--max-scene-fraction", type=float, default=0.20)
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
