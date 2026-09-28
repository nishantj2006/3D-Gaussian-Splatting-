"""Permanently prune Gaussian points by stable object ID or semantic text."""

import argparse
from pathlib import Path
import numpy as np
import torch

from utils.ply_semantic_utils import (
    numbered_fields, read_object_manifest, read_vertices, write_object_manifest, write_vertices,
)
from utils.semantic_utils import encode_text, load_pca


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Must differ from input; the source is preserved")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--object-id", type=int, help="Exact removal for generated/merged assets")
    selector.add_argument("--text", nargs="+", help="One or more positive text prompts")
    selector.add_argument("--bbox-only", action="store_true",
                          help="Remove every Gaussian in --bbox without using semantic scores")
    parser.add_argument("--negative", nargs="*", default=[], help="Background/confuser prompts")
    parser.add_argument("--pca-path", default="data/my_scene/pca_model_128.pkl")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.05,
                        help="Positive score must exceed best negative by this margin")
    parser.add_argument("--bbox", type=float, nargs=6,
                        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    parser.add_argument("--grow-component-eps", type=float,
                        help="Grow semantic seeds to their best connected 3D component inside --bbox")
    parser.add_argument("--component-min-samples", type=int, default=5)
    parser.add_argument("--component-anchor", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Choose the connected component nearest this known object point")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def semantic_mask(vertices, args):
    fields = numbered_fields(vertices.dtype.names, "semantic_")
    if not fields:
        raise ValueError("Input PLY contains no semantic_N properties")
    pca = load_pca(args.pca_path)
    if len(fields) != int(pca.n_components_):
        raise ValueError(f"PLY is {len(fields)}D but PCA is {pca.n_components_}D")
    features = np.column_stack([vertices[name] for name in fields]).astype(np.float32, copy=False)
    features = torch.from_numpy(features).to(args.device)
    features = torch.nn.functional.normalize(features, dim=-1, eps=1e-12)
    prompts = list(args.text) + list(args.negative)
    encodings = encode_text(prompts, pca, args.device)
    positives = encodings[:len(args.text)]
    positive_score = (features @ positives.T).amax(dim=1)
    mask = positive_score > args.threshold
    if args.negative:
        negative_score = (features @ encodings[len(args.text):].T).amax(dim=1)
        mask &= positive_score >= negative_score + args.margin
    return mask.cpu().numpy(), positive_score.detach().cpu().numpy()


def remove(args):
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Refusing to overwrite the source PLY; choose a new --output")
    if args.bbox_only and not args.bbox:
        raise ValueError("--bbox-only requires --bbox")
    ply, vertices = read_vertices(args.input)
    if args.object_id is not None:
        if "object_id" not in vertices.dtype.names:
            raise ValueError("PLY has no object_id property; use --text for a trained scene")
        mask = np.rint(vertices["object_id"]).astype(np.int64) == args.object_id
        scores = None
    elif args.bbox_only:
        mask = np.ones(len(vertices), dtype=bool)
        scores = None
    else:
        mask, scores = semantic_mask(vertices, args)

    bbox_mask = np.ones(len(vertices), dtype=bool)
    if args.bbox:
        lo = np.asarray(args.bbox[:3], dtype=np.float32)
        hi = np.asarray(args.bbox[3:], dtype=np.float32)
        xyz = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
        bbox_mask = ((xyz >= lo) & (xyz <= hi)).all(axis=1)
        mask &= bbox_mask

    if args.grow_component_eps is not None:
        if not args.bbox:
            raise ValueError("--grow-component-eps requires --bbox to limit spatial growth")
        if args.bbox_only:
            raise ValueError("--grow-component-eps requires semantic --text seeds")
        if args.object_id is not None:
            raise ValueError("Component growth is unnecessary with exact --object-id removal")
        from sklearn.cluster import DBSCAN

        bbox_indices = np.flatnonzero(bbox_mask)
        labels = DBSCAN(
            eps=args.grow_component_eps,
            min_samples=args.component_min_samples,
            n_jobs=-1,
        ).fit_predict(xyz[bbox_indices])
        seed_counts = []
        for label in set(labels) - {-1}:
            component_indices = bbox_indices[labels == label]
            seed_counts.append((int(mask[component_indices].sum()), len(component_indices), label))
        if not seed_counts or max(seed_counts)[0] == 0:
            raise ValueError("No connected component contains any semantic seed points")
        if args.component_anchor is not None:
            anchor = np.asarray(args.component_anchor, dtype=np.float32)
            non_noise = labels >= 0
            if not non_noise.any():
                raise ValueError("No connected components found near --component-anchor")
            candidate_points = xyz[bbox_indices[non_noise]]
            nearest = np.square(candidate_points - anchor).sum(axis=1).argmin()
            best_label = int(labels[non_noise][nearest])
            component_indices = bbox_indices[labels == best_label]
            seed_count = int(mask[component_indices].sum())
            component_size = int(len(component_indices))
        else:
            seed_count, component_size, best_label = max(seed_counts)
        mask[:] = False
        mask[bbox_indices[labels == best_label]] = True
        print(
            f"Grew {seed_count:,} semantic seeds to connected component "
            f"{best_label} containing {component_size:,} points"
        )

    count = int(mask.sum())
    print(f"Selected {count:,}/{len(vertices):,} points ({100.0 * count / max(1, len(vertices)):.2f}%)")
    if scores is not None and len(scores):
        print(f"Positive similarity: min={scores.min():.4f}, median={np.median(scores):.4f}, max={scores.max():.4f}")
    if args.dry_run:
        print("Dry run: no file written")
        return

    kept = vertices[~mask].copy()
    write_vertices(args.output, kept, ply, [f"pruned {count} points from {args.input}"])
    manifest = read_object_manifest(args.input)
    if args.object_id is not None:
        manifest.get("objects", {}).pop(str(args.object_id), None)
    write_object_manifest(args.output, manifest)
    print(f"Wrote pruned cloud with {len(kept):,} points to {args.output}")


if __name__ == "__main__":
    remove(parse_args())
