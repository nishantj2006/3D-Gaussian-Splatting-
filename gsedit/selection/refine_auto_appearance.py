"""Grow an unapproved mask using its automatically learned local appearance.

Input seeds must come from an existing text-and-image preview. This second
stage deliberately does not require the newly added splat centers to project
inside the 2D masks, because broad Gaussians may contribute from outside them.
The output is always a new, unapproved preview folder.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from gsedit.selection.auto_object_preview import learned_appearance
from utils.ply_semantic_utils import numbered_fields, read_vertices, write_vertices
from utils.semantic_utils import encode_text, load_pca


def refine(points, rgb, similarity, seeds, radius):
    color_match, color_details = learned_appearance(rgb[seeds], rgb)
    distance = cKDTree(points[seeds]).query(points, workers=-1)[0]
    semantic_cutoff = float(np.quantile(similarity[seeds], 0.10))
    selected = (distance <= radius) & color_match & (similarity >= semantic_cutoff)
    selected[seeds] = True
    return selected, {"learned_appearance": color_details,
                      "semantic_cutoff": semantic_cutoff,
                      "appearance_added": int(selected.sum() - len(seeds))}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    if args.radius <= 0 or args.max_growth_factor < 1:
        raise ValueError("Radius must be positive and maximum growth factor at least 1")
    source, vertices = read_vertices(args.scene)
    seed_folder = Path(args.seed_preview).resolve()
    with open(seed_folder / "preview.json", encoding="utf-8") as handle:
        seed_report = json.load(handle)
    if Path(seed_report["source"]).resolve() != Path(args.scene).resolve():
        raise ValueError("Seed preview was generated from a different scene")
    if seed_report.get("approved") is not False:
        raise ValueError("Expected an unapproved seed preview")
    seeds = np.load(seed_folder / "selected-indices.npy", allow_pickle=False)
    if seeds.ndim != 1 or not np.issubdtype(seeds.dtype, np.integer) or len(seeds) < 200:
        raise ValueError("Seed indices are invalid or too sparse")
    if (seeds < 0).any() or (seeds >= len(vertices)).any() or len(np.unique(seeds)) != len(seeds):
        raise ValueError("Seed indices are duplicated or outside the scene")
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")])
    rgb = np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]) * 0.2820947918 + 0.5
    fields = numbered_fields(vertices.dtype.names, "semantic_")
    pca = load_pca(args.pca_path)
    if len(fields) != pca.n_components_:
        raise ValueError("Scene semantic dimensions do not match PCA")
    features = np.column_stack([vertices[k] for k in fields]).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
    text = args.text or seed_report["text"]
    query = encode_text([text], pca, device="cpu").cpu().numpy()[0]
    similarity = features @ query
    selected, details = refine(points, rgb, similarity, seeds, args.radius)
    if selected.sum() > len(seeds) * args.max_growth_factor:
        raise ValueError("Appearance growth exceeds the safe factor; no preview written")
    if selected.mean() > args.max_scene_fraction:
        raise ValueError("Appearance growth selects too much of the scene")
    output.mkdir(parents=True)
    np.save(output / "selected-indices.npy", np.flatnonzero(selected))
    write_vertices(output / "pruned-preview.ply", vertices[~selected].copy(), source,
                   ["unreviewed learned-appearance removal preview"])
    report = {"source": str(Path(args.scene).resolve()), "seed_preview": str(seed_folder),
              "text": text, "seed_points": int(len(seeds)), "selected_points": int(selected.sum()),
              "selected_fraction": float(selected.mean()), "radius": args.radius,
              **details, "approved": False, "quality": "needs_visual_review",
              "warning": "Appearance can leak into similar nearby surfaces; background is not reconstructed."}
    with open(output / "preview.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), "seed_points": len(seeds),
                      "selected_points": int(selected.sum()),
                      "appearance_added": details["appearance_added"],
                      "approved": False}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--seed-preview", required=True)
    p.add_argument("--text", help="Defaults to the text stored in the seed preview")
    p.add_argument("--pca-path", required=True)
    p.add_argument("--radius", type=float, default=0.35)
    p.add_argument("--max-growth-factor", type=float, default=1.5)
    p.add_argument("--max-scene-fraction", type=float, default=0.20)
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
