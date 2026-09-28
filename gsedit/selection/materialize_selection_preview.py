"""Write a schema-preserving, explicitly unapproved pruned PLY preview."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from utils.ply_semantic_utils import read_vertices, write_vertices


def validate_indices(indices, count, max_fraction):
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("Selection must be a nonempty one-dimensional array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Selection indices must be integers")
    if indices.min() < 0 or indices.max() >= count or len(np.unique(indices)) != len(indices):
        raise ValueError("Selection indices are out of range or duplicated")
    if len(indices) / count > max_fraction:
        raise ValueError("Selection exceeds scene safety fraction")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    source = Path(args.scene).resolve()
    indices_path = Path(args.indices).resolve()
    if output == source.parent or source in output.parents:
        raise ValueError("Preview folder cannot contain or replace source scene")
    source_ply, vertices = read_vertices(source)
    indices = np.load(indices_path, allow_pickle=False)
    validate_indices(indices, len(vertices), args.max_scene_fraction)
    keep = np.ones(len(vertices), dtype=bool)
    keep[indices] = False
    output.mkdir(parents=True)
    preview = output / "pruned-preview.ply"
    write_vertices(preview, vertices[keep].copy(), source_ply,
                   ["UNAPPROVED preview; original scene preserved"])
    _, verified = read_vertices(preview)
    if verified.dtype != vertices.dtype or len(verified) != int(keep.sum()):
        raise RuntimeError("Preview PLY schema/count verification failed")
    digest = hashlib.sha256(preview.read_bytes()).hexdigest()
    report = {"source": str(source), "indices": str(indices_path),
              "preview": str(preview), "original_splats": len(vertices),
              "removed_splats": len(indices), "remaining_splats": len(verified),
              "property_count": len(vertices.dtype.names),
              "semantic_dimensions": sum(name.startswith("semantic_") for name in vertices.dtype.names),
              "preview_sha256": digest, "approved": False}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--indices", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-scene-fraction", type=float, default=.2)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
