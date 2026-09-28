"""Preview text-driven object removal without scene-specific color rules.

CLIP chooses source views and scores MobileSAM's independent image masks.
Consistent visible 3D points seed a local Gaussian-mixture appearance model;
that learned appearance, text similarity and spatial proximity expand the
selection. The script never overwrites its source or approves removal.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import torch

from gsedit.selection.multiview_instance import project
from utils.ply_semantic_utils import numbered_fields, read_vertices, write_vertices
from utils.semantic_utils import encode_text, load_clip, load_pca


def choose_views(images, cameras, label, model, processor, *, count=2,
                 min_baseline=0.5, exclude_views=()):
    paths = [p for p in sorted(Path(images).iterdir()) if p.stem in cameras and
             p.stem not in exclude_views and
             p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(paths) < count:
        raise ValueError("Not enough source images with camera poses")
    with torch.inference_mode():
        text = processor(text=[label], return_tensors="pt")
        query = torch.nn.functional.normalize(model.get_text_features(**text), dim=-1)
        ranked = []
        for start in range(0, len(paths), 8):
            batch = paths[start:start + 8]
            photos = [Image.open(p).convert("RGB") for p in batch]
            inputs = processor(images=photos, return_tensors="pt", padding=True)
            vectors = torch.nn.functional.normalize(model.get_image_features(**inputs), dim=-1)
            ranked.extend((float(score), path.stem) for score, path in
                          zip((vectors @ query.T).flatten(), batch))
    selected = []
    for score, name in sorted(ranked, reverse=True):
        position = np.asarray(cameras[name]["position"])
        if all(np.linalg.norm(position - np.asarray(cameras[old]["position"])) >= min_baseline
               for _, old in selected):
            selected.append((score, name))
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError("Could not find enough distinct views of the requested object")
    return [name for _, name in selected], {name: score for score, name in selected}


def mask_crop(image, mask):
    yy, xx = np.nonzero(mask)
    if len(xx) < 500:
        return None
    x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    pixels = image[y0:y1, x0:x1].copy()
    pixels[~mask[y0:y1, x0:x1]] = 128
    return Image.fromarray(pixels)


def score_masks(image, masks, label, model, processor):
    crops, ids = [], []
    for i, mask in enumerate(masks):
        crop = mask_crop(image, mask)
        if crop is not None and 0.002 <= mask.mean() <= 0.60:
            ids.append(i)
            crops.append(crop)
    if not crops:
        raise ValueError("No usable image-object proposals")
    with torch.inference_mode():
        text = processor(text=[label], return_tensors="pt")
        query = torch.nn.functional.normalize(model.get_text_features(**text), dim=-1)
        scores = []
        for start in range(0, len(crops), 8):
            inputs = processor(images=crops[start:start + 8], return_tensors="pt", padding=True)
            features = torch.nn.functional.normalize(model.get_image_features(**inputs), dim=-1)
            scores.extend((features @ query.T).flatten().tolist())
    ranking = sorted(zip(scores, ids), reverse=True)
    return masks[ranking[0][1]], [{"score": float(s), "mask_index": int(i),
                                   "area_fraction": float(masks[i].mean())}
                                  for s, i in ranking]


def visible_mask_hits(points, camera, mask, depth_tolerance=0.35):
    height, width = mask.shape
    x, y, depth, valid = project(points, camera, mask.shape)
    hit = np.zeros(len(points), dtype=bool)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return hit
    pixel = y[ids] * width + x[ids]
    front = np.full(height * width, np.inf)
    np.minimum.at(front, pixel, depth[ids])
    hit[ids] = mask[y[ids], x[ids]] & (depth[ids] <= front[pixel] + depth_tolerance)
    return hit


def coherent_component(points, candidate_ids, *, radius=0.5, minimum=200):
    if len(candidate_ids) < minimum:
        raise ValueError("Too few cross-view 3D object seeds")
    labels = DBSCAN(eps=radius, min_samples=5, n_jobs=-1).fit_predict(points[candidate_ids])
    counts = np.bincount(labels[labels >= 0])
    if not len(counts) or counts.max() < minimum:
        raise ValueError("No coherent cross-view 3D component")
    return candidate_ids[labels == counts.argmax()]


def learned_appearance(seed_rgb, all_rgb, *, components=(1, 2, 3)):
    """Return membership learned from seed colors, without named color classes."""
    data = np.clip(seed_rgb, 0, 1)
    candidates = [GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-4,
                                  random_state=0).fit(data) for k in components]
    model = min(candidates, key=lambda candidate: candidate.bic(data))
    cutoff = float(np.quantile(model.score_samples(data), 0.10))
    return model.score_samples(np.clip(all_rgb, 0, 1)) >= cutoff, {
        "components": int(model.n_components), "rgb_means": model.means_.round(4).tolist(),
        "log_likelihood_cutoff": cutoff}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output folder: {output}")
    if args.scene and not Path(args.scene).is_file():
        raise FileNotFoundError(args.scene)
    source, vertices = read_vertices(args.scene)
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float64)
    rgb = np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]) * 0.2820947918 + 0.5
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    model, processor = load_clip("cpu")
    pca = load_pca(args.pca_path)
    fields = numbered_fields(vertices.dtype.names, "semantic_")
    if len(fields) != pca.n_components_:
        raise ValueError("Scene semantic dimensions do not match PCA")
    query = encode_text([args.text], pca, device="cpu", model=model,
                        processor=processor).cpu().numpy()[0]
    features = np.column_stack([vertices[name] for name in fields]).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
    similarity = features @ query
    vector_only = similarity > args.vector_threshold
    views, view_scores = choose_views(args.images, cameras, args.text, model, processor,
                                      count=args.view_count, min_baseline=args.min_baseline,
                                      exclude_views=args.exclude_views)
    from ultralytics import SAM
    sam = SAM(args.sam_model)
    masks, photos, details, evidence = {}, {}, {}, []
    for view in views:
        matches = list(Path(args.images).glob(view + ".*"))
        if len(matches) != 1:
            raise ValueError(f"Expected one photo for {view}")
        with Image.open(matches[0]) as photo:
            image = photo.convert("RGB")
            if image.width > args.image_width:
                image = image.resize((args.image_width,
                                      round(image.height * args.image_width / image.width)))
            image = np.asarray(image)
        result = sam.predict(source=image, imgsz=args.sam_size,
                             device=args.device, verbose=False)[0]
        if result.masks is None:
            raise ValueError(f"No image masks for {view}")
        proposals = result.masks.data.cpu().numpy().astype(bool)
        mask, ranking = score_masks(image, proposals, args.text, model, processor)
        if mask.shape != image.shape[:2]:
            mask = np.asarray(Image.fromarray(mask).resize(
                (image.shape[1], image.shape[0]), Image.Resampling.NEAREST))
        masks[view], photos[view] = mask, image
        hit = visible_mask_hits(points, cameras[view], mask,
                                depth_tolerance=args.depth_tolerance)
        evidence.append(hit)
        details[view] = {"global_score": view_scores[view], "proposal_count": len(proposals),
                         "top_masks": ranking[:3], "visible_hits": int(hit.sum())}
    votes = np.sum(np.stack(evidence), axis=0)
    candidates = np.flatnonzero(votes >= args.min_views)
    if len(candidates) > len(points) * args.max_fraction:
        raise ValueError("Image masks jointly select too much of the scene")
    seeds = coherent_component(points, candidates, radius=args.cluster_radius,
                               minimum=args.min_seeds)
    color_match, color_report = learned_appearance(rgb[seeds], rgb)
    distance = cKDTree(points[seeds]).query(points, workers=-1)[0]
    semantic_cutoff = float(np.quantile(similarity[seeds], 0.10))
    chosen = (distance <= args.grow_radius) & color_match & (votes > 0) & \
        (similarity >= semantic_cutoff)
    chosen[seeds] = True
    if chosen.sum() > len(points) * args.max_fraction:
        raise ValueError("Expanded object mask is too broad")
    report = {"source": str(Path(args.scene).resolve()), "text": args.text,
              "views": details, "vector_only_count": int(vector_only.sum()),
              "vector_only_fraction": float(vector_only.mean()),
              "vector_only_safe": bool(vector_only.mean() <= args.max_fraction),
              "cross_view_candidates": int(len(candidates)), "coherent_seeds": int(len(seeds)),
              "learned_appearance": color_report, "semantic_cutoff": semantic_cutoff,
              "selected_points": int(chosen.sum()), "selected_fraction": float(chosen.mean()),
              "approved": False, "quality": "needs_visual_review",
              "warning": "Automatic appearance is not proof of complete object identity; hidden background is unfilled."}
    output.mkdir(parents=True)
    for view in views:
        mask, image = masks[view], photos[view].copy()
        image[mask] = np.rint(0.55 * image[mask] +
                               0.45 * np.array([255, 45, 45])).astype(np.uint8)
        Image.fromarray(image).save(output / f"{view}-mask-overlay.png")
    np.save(output / "selected-indices.npy", np.flatnonzero(chosen))
    np.save(output / "vector-only-indices.npy", np.flatnonzero(vector_only))
    write_vertices(output / "pruned-preview.ply", vertices[~chosen].copy(), source,
                   ["unreviewed automatic text-and-learned-appearance removal"])
    if report["vector_only_safe"]:
        write_vertices(output / "vector-only-pruned-preview.ply", vertices[~vector_only].copy(),
                       source, ["unreviewed text-vector-only removal"])
    with open(output / "preview.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), "views": views,
                      "vector_only": report["vector_only_count"],
                      "vector_only_safe": report["vector_only_safe"],
                      "appearance_selection": report["selected_points"],
                      "approved": False}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--pca-path", required=True)
    p.add_argument("--sam-model", default="mobile_sam.pt")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--view-count", type=int, default=2)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--min-baseline", type=float, default=0.5)
    p.add_argument("--exclude-views", nargs="*", default=[])
    p.add_argument("--image-width", type=int, default=540)
    p.add_argument("--sam-size", type=int, default=640)
    p.add_argument("--vector-threshold", type=float, default=0.05)
    p.add_argument("--depth-tolerance", type=float, default=0.35)
    p.add_argument("--cluster-radius", type=float, default=0.5)
    p.add_argument("--min-seeds", type=int, default=200)
    p.add_argument("--grow-radius", type=float, default=0.35)
    p.add_argument("--max-fraction", type=float, default=0.20)
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
