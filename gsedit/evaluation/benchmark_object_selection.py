"""Preview-only ablation of view diversity, splat/mask lifting, and 3D growth.

No scene is modified. This benchmark uses analytic screen-space Gaussian
footprints as a CPU proxy; it does not claim to perform full alpha compositing.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree

from gsedit.selection.auto_object_preview import choose_views, score_masks
from gsedit.selection.multiview_instance import project
from utils.ply_semantic_utils import numbered_fields, read_vertices
from utils.semantic_utils import load_clip
from gsedit.selection.whole_object_preview import associate_parts, projected_hits


def quaternion_matrix(q):
    """Vectorized scalar-first unit quaternions to local-to-world matrices."""
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    matrix = np.empty((len(q), 3, 3), dtype=np.float32)
    matrix[:, 0, 0] = 1 - 2 * (y*y + z*z)
    matrix[:, 0, 1] = 2 * (x*y - z*w)
    matrix[:, 0, 2] = 2 * (x*z + y*w)
    matrix[:, 1, 0] = 2 * (x*y + z*w)
    matrix[:, 1, 1] = 1 - 2 * (x*x + z*z)
    matrix[:, 1, 2] = 2 * (y*z - x*w)
    matrix[:, 2, 0] = 2 * (x*z - y*w)
    matrix[:, 2, 1] = 2 * (y*z + x*w)
    matrix[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return matrix


def footprint_cache(points, scales, rotations, opacity, camera, shape, max_radius=10):
    """Project anisotropic covariance to two screen-space axes, with depth gate."""
    height, width = shape
    x, y, depth, valid = project(points, camera, shape)
    ids = np.flatnonzero(valid)
    front = np.full(height * width, np.inf, dtype=np.float32)
    pixel = y[ids] * width + x[ids]
    np.minimum.at(front, pixel, depth[ids])
    front = minimum_filter(front.reshape(shape), size=2*int(np.ceil(max_radius))+1)
    camera_rot = np.asarray(camera["rotation"], dtype=np.float32)
    local = (points[ids] - np.asarray(camera["position"])) @ camera_rot
    f_x = float(camera["fx"]) * width / float(camera["width"])
    f_y = float(camera["fy"]) * height / float(camera["height"])
    z = np.maximum(local[:, 2], 1e-6)
    jac = np.zeros((len(ids), 2, 3), dtype=np.float32)
    jac[:, 0, 0] = f_x / z
    jac[:, 0, 2] = -f_x * local[:, 0] / (z*z)
    jac[:, 1, 1] = f_y / z
    jac[:, 1, 2] = -f_y * local[:, 1] / (z*z)
    world_axes = quaternion_matrix(rotations[ids]) * np.exp(scales[ids])[:, None, :]
    camera_axes = np.einsum("nij,jk->nik", world_axes.transpose(0, 2, 1), camera_rot)
    screen_axes = np.einsum("nij,njk->nik", jac, camera_axes.transpose(0, 2, 1))
    a = np.sum(screen_axes[:, 0, :]**2, axis=1)
    b = np.sum(screen_axes[:, 0, :] * screen_axes[:, 1, :], axis=1)
    c = np.sum(screen_axes[:, 1, :]**2, axis=1)
    delta = np.sqrt(np.maximum((a-c)**2 + 4*b*b, 0))
    major = np.sqrt(np.maximum((a+c+delta)/2, 0))
    minor = np.sqrt(np.maximum((a+c-delta)/2, 0))
    theta = .5 * np.arctan2(2*b, a-c)
    major = np.minimum(major, max_radius)
    minor = np.minimum(minor, max_radius)
    ux, uy = np.cos(theta), np.sin(theta)
    axes = np.column_stack((major*ux, major*uy, -minor*uy, minor*ux))
    alpha = 1 / (1 + np.exp(-np.clip(opacity[ids], -30, 30)))
    visible = (depth[ids] <= front[y[ids], x[ids]] + .25) & (alpha >= .04)
    return ids, x[ids], y[ids], axes, visible


def ellipse_hits(cache, mask, total_points, min_weight=.27):
    """Five samples of a projected Gaussian ellipse; weighted mask contribution."""
    ids, x, y, axes, visible = cache
    h, w = mask.shape
    dx1, dy1, dx2, dy2 = axes.T
    weights = np.zeros(len(ids), dtype=np.float32)
    for dx, dy, weight in ((0, 0, .4), (dx1, dy1, .15), (-dx1, -dy1, .15),
                           (dx2, dy2, .15), (-dx2, -dy2, .15)):
        xx = np.rint(x + dx).astype(np.int32)
        yy = np.rint(y + dy).astype(np.int32)
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        weights[inside] += weight * mask[yy[inside], xx[inside]]
    hits = np.zeros(total_points, dtype=bool)
    hits[ids] = visible & (weights >= min_weight)
    return hits


def select(proposals, seed, points, total, *, seed_distance=.32,
           min_near_fraction=.2, min_part_views=2):
    parts, _ = associate_parts(proposals, seed, points, seed_distance=seed_distance)
    selected = np.zeros(total, dtype=bool)
    selected[seed] = True
    included = []
    for index, part in enumerate(parts, 1):
        ratio = part["near_seed"] / part["raw_points"]
        if (len(part["views"]) >= min_part_views and part["connected"] and
            ratio >= min_near_fraction and len(part["ids"]) <= .04*total):
            selected[part["ids"]] = True
            included.append(index)
    if selected.mean() > .2:
        raise ValueError("Ablation exceeded the 20% scene safety limit")
    return selected, parts, included


def grow_graph(points, rgb, features, seed, base, positive_support,
               *, radius=.045, max_seed_distance=.42, rounds=3):
    """Conservative local growth with adaptive color/semantic thresholds."""
    tree = cKDTree(points[seed])
    distance, nearest = tree.query(points, workers=-1)
    candidate = np.flatnonzero((distance <= max_seed_distance) & positive_support & ~base)
    if not len(candidate):
        return base.copy(), {"candidates": 0, "added": 0}
    semantic = features[seed]
    semantic /= np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-8)
    seed_neighbors = tree.query(points[seed], k=2, workers=-1)[1][:, 1]
    color_limit = max(.05, float(np.quantile(np.linalg.norm(rgb[seed]-rgb[seed[seed_neighbors]], axis=1), .90)))
    semantic_limit = float(np.quantile(np.sum(semantic * semantic[seed_neighbors], axis=1), .10))
    candidate_sem = features[candidate]
    candidate_sem /= np.maximum(np.linalg.norm(candidate_sem, axis=1, keepdims=True), 1e-8)
    sem_agree = np.sum(candidate_sem * semantic[nearest[candidate]], axis=1) >= semantic_limit
    color_agree = np.linalg.norm(rgb[candidate] - rgb[seed[nearest[candidate]]], axis=1) <= color_limit
    eligible = candidate[sem_agree & color_agree]
    selected = base.copy()
    frontier = selected.copy()
    for _ in range(rounds):
        pending = eligible[~selected[eligible]]
        if not len(pending):
            break
        d = cKDTree(points[np.flatnonzero(frontier)]).query(points[pending], workers=-1)[0]
        new = pending[d <= radius]
        if not len(new):
            break
        selected[new] = True
        frontier[:] = False
        frontier[new] = True
    return selected, {"candidates": int(len(candidate)), "eligible": int(len(eligible)),
                      "added": int(np.count_nonzero(selected & ~base)),
                      "color_limit": color_limit, "semantic_limit": semantic_limit}


def overlay(path, image, points, camera, shape, seed, selected):
    canvas = Image.fromarray(image.copy())
    pen = ImageDraw.Draw(canvas)
    for ids, color, radius in ((seed, (0, 255, 255), 1),
                               (np.flatnonzero(selected & ~np.isin(np.arange(len(selected)), seed)),
                                (255, 30, 30), 2)):
        x, y, _, valid = project(points[ids], camera, shape)
        for xx, yy in zip(x[valid], y[valid]):
            pen.ellipse((int(xx)-radius, int(yy)-radius,
                         int(xx)+radius, int(yy)+radius), fill=color)
    canvas.save(path)


def run(args):
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to reuse benchmark folder: {out}")
    t0 = time.perf_counter()
    _, vertices = read_vertices(args.scene)
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float32)
    scales = np.column_stack([vertices[f"scale_{i}"] for i in range(3)]).astype(np.float32)
    quats = np.column_stack([vertices[f"rot_{i}"] for i in range(4)]).astype(np.float32)
    opacity = vertices["opacity"].astype(np.float32)
    seed = np.unique(np.load(args.seed_indices, allow_pickle=False))
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    model, processor = load_clip("cpu")
    base_views, _ = choose_views(args.images, cameras, args.text, model, processor,
                                 count=3, min_baseline=.35)
    visible_counts = {name: int(project(points[seed], camera,
                      (int(camera["height"]), int(camera["width"])))[3].sum())
                      for name, camera in cameras.items()}
    min_visible = max(args.min_seed_visible_points,
                      int(np.ceil(args.min_seed_visible_fraction * len(seed))))
    eligible_cameras = {name: camera for name, camera in cameras.items()
                        if visible_counts[name] >= min_visible}
    diverse_views, _ = choose_views(args.images, eligible_cameras, args.text, model, processor,
                                    count=args.diverse_views, min_baseline=args.min_baseline)
    all_views = list(dict.fromkeys(base_views + diverse_views))
    t_views = time.perf_counter()
    from ultralytics import SAM
    sam = SAM(args.sam_model)
    records, images = {}, {}
    for view in all_views:
        matches = [p for p in Path(args.images).glob(view + ".*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if len(matches) != 1:
            raise ValueError(f"Expected one image for {view}")
        with Image.open(matches[0]) as photo:
            image = photo.convert("RGB")
            if image.width > args.image_width:
                image = image.resize((args.image_width, round(image.height*args.image_width/image.width)))
            image = np.asarray(image)
        images[view] = image
        result = sam.predict(source=image, imgsz=args.sam_size,
                             device=args.device, verbose=False)[0]
        if result.masks is None:
            records[view] = []
            continue
        masks = result.masks.data.cpu().numpy().astype(bool)
        try:
            _, ranking = score_masks(image, masks, args.text, model, processor)
        except ValueError:
            records[view] = []
            continue
        best = ranking[0]["score"]
        records[view] = []
        for item in ranking:
            if item["score"] < best - args.score_drop:
                continue
            mask = masks[item["mask_index"]]
            if mask.shape != image.shape[:2]:
                mask = np.asarray(Image.fromarray(mask).resize(
                    (image.shape[1], image.shape[0]), Image.Resampling.NEAREST))
            records[view].append((item, mask))
    t_sam = time.perf_counter()
    max_scale = scales.max(axis=1)
    approx = {}
    for view in all_views:
        approx[view] = []
        for item, mask in records[view]:
            ids = np.flatnonzero(projected_hits(points, cameras[view], mask,
                max_scale, max_footprint_px=args.max_footprint_px))
            if len(ids) >= args.min_proposal_points:
                approx[view].append({"view": view, "mask_index": item["mask_index"],
                                     "score": item["score"], "ids": ids})
    t_approx = time.perf_counter()
    baseline, base_parts, base_included = select(
        [p for view in base_views for p in approx[view]], seed, points, len(points))
    diverse, diverse_parts, diverse_included = select(
        [p for view in diverse_views for p in approx[view]], seed, points, len(points))
    t_diverse = time.perf_counter()
    ellipse = {}
    for view in diverse_views:
        ellipse[view] = []
        cache = footprint_cache(points, scales, quats, opacity, cameras[view],
                                images[view].shape[:2], max_radius=args.max_footprint_px)
        for item, mask in records[view]:
            ids = np.flatnonzero(ellipse_hits(cache, mask, len(points)))
            if len(ids) >= args.min_proposal_points:
                ellipse[view].append({"view": view, "mask_index": item["mask_index"],
                                      "score": item["score"], "ids": ids})
    ell_selected, ell_parts, ell_included = select(
        [p for view in diverse_views for p in ellipse[view]], seed, points, len(points))
    t_ellipse = time.perf_counter()
    support = np.zeros(len(points), dtype=bool)
    for view in diverse_views:
        for proposal in ellipse[view]:
            support[proposal["ids"]] = True
    rgb = (np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]) *
           .2820947918 + .5).astype(np.float32)
    fields = numbered_fields(vertices.dtype.names, "semantic_")
    features = np.column_stack([vertices[name] for name in fields]).astype(np.float32)
    grown, growth = grow_graph(points, rgb, features, seed, ell_selected, support)
    t_graph = time.perf_counter()
    variants = {"baseline_3view": baseline, "diverse_views": diverse,
                "elliptical_footprint": ell_selected, "graph_growth": grown}
    out.mkdir(parents=True)
    seed_mask = np.zeros(len(points), dtype=bool)
    seed_mask[seed] = True
    report = {"source": str(Path(args.scene).resolve()), "seed_points": len(seed),
              "baseline_views": base_views, "diverse_view_names": diverse_views,
              "min_seed_visible": min_visible,
              "seed_visible_counts": {view: visible_counts[view] for view in all_views},
              "view_positions": {view: cameras[view]["position"] for view in all_views},
              "semantic_dimensions": len(fields), "growth": growth,
              "timings_seconds": {"view_ranking": t_views-t0, "sam_and_clip_masks": t_sam-t_views,
                                  "approximate_projections": t_approx-t_sam,
                                  "part_association": t_diverse-t_approx,
                                  "elliptical_lifting": t_ellipse-t_diverse,
                                  "graph_growth": t_graph-t_ellipse,
                                  "total": t_graph-t0},
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "approved": False,
              "warning": "No manual ground truth; overlap and geometric support are proxies, not accuracy scores. Ellipse sampling is not full alpha compositing."}
    part_meta = ((base_parts, base_included), (diverse_parts, diverse_included),
                 (ell_parts, ell_included), (ell_parts, ell_included))
    for (name, selected), (parts, included) in zip(variants.items(), part_meta):
        np.save(out / f"{name}-indices.npy", np.flatnonzero(selected))
        new = selected & ~seed_mask
        report[name] = {"selected_points": int(selected.sum()), "new_vs_seed": int(new.sum()),
                        "new_vs_baseline": int((selected & ~baseline).sum()),
                        "lost_vs_baseline": int((baseline & ~selected).sum()),
                        "included_parts": len(included), "candidate_parts": len(parts),
                        "outside_seed_42cm": int((new &
                            (cKDTree(points[seed]).query(points, workers=-1)[0] > .42)).sum())}
        for view in base_views:
            overlay(out / f"{name}-{view}.png", images[view], points,
                    cameras[view], images[view].shape[:2], seed, selected)
    with open(out / "benchmark.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k in variants or k in
                      ("timings_seconds", "peak_rss_mb", "baseline_views", "diverse_view_names")}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--seed-indices", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--sam-model", default="mobile_sam.pt")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--diverse-views", type=int, default=6)
    p.add_argument("--min-seed-visible-points", type=int, default=300)
    p.add_argument("--min-seed-visible-fraction", type=float, default=.10)
    p.add_argument("--min-baseline", type=float, default=1.0)
    p.add_argument("--image-width", type=int, default=540)
    p.add_argument("--sam-size", type=int, default=640)
    p.add_argument("--score-drop", type=float, default=.06)
    p.add_argument("--max-footprint-px", type=float, default=10)
    p.add_argument("--min-proposal-points", type=int, default=75)
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
