"""Transform and merge a generated Gaussian asset into a trained scene."""

import argparse
import math
from pathlib import Path
import numpy as np

from utils.ply_semantic_utils import (
    add_float_property, align_dtype, read_object_manifest, read_vertices,
    write_object_manifest, write_vertices,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--translate", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0, 0, 0))
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def merge(args):
    output = Path(args.output).resolve()
    if output.exists() or output.with_suffix(".objects.json").exists():
        raise ValueError("Refusing to overwrite an existing merge output")
    if not np.isfinite(args.scale) or args.scale <= 0:
        raise ValueError("--scale must be finite and positive")
    if not np.all(np.isfinite(args.translate)):
        raise ValueError("--translate must be finite")
    if args.object_id <= 0:
        raise ValueError("--object-id must be positive")
    scene_ply, scene = read_vertices(args.scene)
    _, asset = read_vertices(args.asset)
    if "object_id" in scene.dtype.names and np.any(np.isclose(scene["object_id"], args.object_id)):
        raise ValueError(f"Scene already contains object_id {args.object_id}")
    if "object_id" not in scene.dtype.names:
        scene = add_float_property(scene, "object_id", 0.0)
    asset = add_float_property(asset, "object_id", float(args.object_id))
    asset = align_dtype(asset, scene.dtype)
    for field, offset in zip(("x", "y", "z"), args.translate):
        asset[field] = asset[field] * args.scale + offset
    for field in (name for name in asset.dtype.names if name.startswith("scale_")):
        asset[field] += math.log(args.scale)

    merged = np.concatenate((scene, asset))
    write_vertices(args.output, merged, scene_ply, [f"object {args.object_id}: {args.label}"])
    manifest = read_object_manifest(args.scene)
    manifest.setdefault("objects", {})[str(args.object_id)] = {
        "label": args.label,
        "source": str(Path(args.asset).resolve()),
        "point_count": int(len(asset)),
    }
    manifest["background_object_id"] = 0
    manifest_path = write_object_manifest(args.output, manifest)
    print(f"Merged {len(asset):,} asset points with {len(scene):,} scene points -> {args.output}")
    print(f"Object manifest: {manifest_path}")


if __name__ == "__main__":
    merge(parse_args())
