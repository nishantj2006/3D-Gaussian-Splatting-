"""Audit learned Gaussian semantic vectors and write non-destructive previews."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from utils.ply_semantic_utils import numbered_fields, read_vertices, write_vertices
from utils.semantic_utils import encode_text, load_pca


DEFAULT_LABELS = [
    "water bottle", "carpet", "wooden furniture", "clothes", "shoes",
    "refrigerator", "wall", "floor", "background",
]
DEFAULT_POSITIVES = ["bottle", "water bottle", "plastic bottle", "metal water bottle"]
DEFAULT_NEGATIVES = [
    "carpet", "floor", "wall", "wooden furniture", "clothes", "shoes",
    "refrigerator", "background", "room", "clutter",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pca-path", default="data/my_scene/pca_model_128.pkl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--cluster-eps", type=float, default=0.12)
    parser.add_argument("--cluster-min-samples", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=30000)
    return parser.parse_args()


def audit(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ply, vertices = read_vertices(args.input)
    semantic_fields = numbered_fields(vertices.dtype.names, "semantic_")
    raw = np.column_stack([vertices[name] for name in semantic_fields]).astype(np.float32)
    finite_rows = np.isfinite(raw).all(axis=1)
    norms = np.linalg.norm(raw, axis=1)
    normalized = raw / np.maximum(norms[:, None], 1e-12)
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)

    rng = np.random.default_rng(42)
    sample_count = min(args.sample_size, len(vertices))
    sample_ids = rng.choice(len(vertices), sample_count, replace=False)
    sampled_features = normalized[sample_ids]
    sampled_xyz = xyz[sample_ids]
    singular_values = np.linalg.svd(sampled_features - sampled_features.mean(0), compute_uv=False)
    variance = singular_values ** 2
    probability = variance / variance.sum()
    effective_rank = float(np.exp(-(probability * np.log(probability + 1e-20)).sum()))

    neighbors = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(sampled_xyz)
    nearest_ids = neighbors.kneighbors(return_distance=False)[:, 1]
    nearest_cosine = np.sum(sampled_features * sampled_features[nearest_ids], axis=1)
    random_ids = rng.integers(0, sample_count, sample_count)
    random_cosine = np.sum(sampled_features * sampled_features[random_ids], axis=1)

    pca = load_pca(args.pca_path)
    prompts = DEFAULT_LABELS + DEFAULT_POSITIVES + DEFAULT_NEGATIVES
    query = encode_text(prompts, pca, args.device).cpu().numpy()
    label_query = query[:len(DEFAULT_LABELS)]
    positive_query = query[len(DEFAULT_LABELS):len(DEFAULT_LABELS) + len(DEFAULT_POSITIVES)]
    negative_query = query[-len(DEFAULT_NEGATIVES):]

    label_scores = normalized @ label_query.T
    winners = label_scores.argmax(axis=1)
    winner_counts = {
        label: int((winners == index).sum()) for index, label in enumerate(DEFAULT_LABELS)
    }
    positive_score = (normalized @ positive_query.T).max(axis=1)
    negative_score = (normalized @ negative_query.T).max(axis=1)
    mask = (positive_score > args.threshold) & (positive_score >= negative_score + args.margin)

    candidate_xyz = xyz[mask]
    cluster_labels = DBSCAN(
        eps=args.cluster_eps, min_samples=args.cluster_min_samples, n_jobs=-1
    ).fit_predict(candidate_xyz) if len(candidate_xyz) else np.empty(0, dtype=np.int32)
    clusters = []
    for label in sorted(set(cluster_labels) - {-1}):
        local_mask = cluster_labels == label
        points = candidate_xyz[local_mask]
        clusters.append({
            "label": int(label), "points": int(len(points)),
            "center": points.mean(0).tolist(), "minimum": points.min(0).tolist(),
            "maximum": points.max(0).tolist(),
        })
    clusters.sort(key=lambda item: item["points"], reverse=True)

    candidate_path = output_dir / "bottle-candidates-margin-020.ply"
    write_vertices(candidate_path, vertices[mask].copy(), ply,
                   ["semantic bottle selection preview; source is unchanged"])
    largest_path = None
    if clusters:
        largest_label = clusters[0]["label"]
        selected_indices = np.flatnonzero(mask)
        largest_mask = cluster_labels == largest_label
        largest_path = output_dir / "bottle-largest-spatial-cluster.ply"
        write_vertices(largest_path, vertices[selected_indices[largest_mask]].copy(), ply,
                       ["largest spatial cluster from bottle semantic preview"])

    report = {
        "input": str(Path(args.input).resolve()),
        "point_count": int(len(vertices)),
        "semantic_dimensions": int(len(semantic_fields)),
        "finite_rows": int(finite_rows.sum()),
        "zero_norm_rows": int((norms <= 1e-12).sum()),
        "norm": {"minimum": float(norms.min()), "median": float(np.median(norms)),
                 "maximum": float(norms.max())},
        "active_dimensions": int((raw.std(0) > 1e-6).sum()),
        "effective_rank_sample": effective_rank,
        "spatial_coherence": {
            "nearest_neighbor_cosine_mean": float(nearest_cosine.mean()),
            "nearest_neighbor_cosine_median": float(np.median(nearest_cosine)),
            "random_pair_cosine_mean": float(random_cosine.mean()),
            "random_pair_cosine_median": float(np.median(random_cosine)),
        },
        "label_winner_counts": winner_counts,
        "bottle_selection": {
            "threshold": args.threshold, "margin": args.margin,
            "selected_points": int(mask.sum()), "selected_fraction": float(mask.mean()),
            "score_minimum": float(positive_score.min()),
            "score_median": float(np.median(positive_score)),
            "score_maximum": float(positive_score.max()),
            "noise_points": int((cluster_labels < 0).sum()),
            "clusters": clusters[:20],
            "candidate_ply": str(candidate_path.resolve()),
            "largest_cluster_ply": str(largest_path.resolve()) if largest_path else None,
        },
    }
    report_path = output_dir / "semantic-audit.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    audit(parse_args())
