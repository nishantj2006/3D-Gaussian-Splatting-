"""Align a Gaussian asset to a scene floor and a target 3D bounding box.

Only the asset PLY is transformed. The source PLY and scene remain unchanged.
Uniform scaling preserves proportions; Gaussian quaternions are rotated with
their centers, so anisotropic splats keep the correct covariance.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from reconstruct_flat import removed_indices
from utils.ply_semantic_utils import read_vertices, write_vertices


def xyz_of(vertices):
    return np.column_stack([vertices[field] for field in ("x", "y", "z")]).astype(np.float64)


def plane_frame(origin, normal):
    origin = np.asarray(origin, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    if origin.shape != (3,) or normal.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("Invalid ground-plane origin or normal")
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("Ground-plane normal is invalid")
    normal = normal / length
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    u = reference - np.dot(reference, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return origin, np.column_stack((u, v, normal))


def asset_up_rotation(axis):
    vectors = {
        "x": (1, 0, 0), "-x": (-1, 0, 0),
        "y": (0, 1, 0), "-y": (0, -1, 0),
        "z": (0, 0, 1), "-z": (0, 0, -1),
    }
    if axis not in vectors:
        raise ValueError(f"Unsupported asset up axis {axis!r}")
    source_up = np.asarray(vectors[axis], dtype=np.float64)
    rotation, _ = Rotation.align_vectors([[0, 0, 1]], [source_up])
    return rotation.as_matrix()


def target_from_removal(original, pruned, origin, frame, min_height=0.15):
    missing = removed_indices(original, pruned)
    points = xyz_of(original[missing])
    local = (points - origin) @ frame
    above = local[:, 2] > min_height
    if above.sum() < 100:
        raise ValueError("Removal has too few object Gaussians above the floor")
    object_points = local[above]
    low = np.quantile(object_points[:, :2], 0.01, axis=0)
    high = np.quantile(object_points[:, :2], 0.99, axis=0)
    height = np.quantile(object_points[:, 2], 0.995)
    size = np.array([high[0] - low[0], high[1] - low[1], height])
    if np.any(size <= 0.01):
        raise ValueError("Removed object bounding box is degenerate")
    return (low + high) / 2, size, int(above.sum())


def target_from_world_box(bounds, origin, frame):
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
        raise ValueError("--target-bbox requires six finite numbers")
    low, high = bounds[:3], bounds[3:]
    if np.any(low >= high):
        raise ValueError("--target-bbox min must be below max on each axis")
    corners = np.asarray(list(itertools.product(
        (low[0], high[0]), (low[1], high[1]), (low[2], high[2]))))
    local = (corners - origin) @ frame
    low_uv, high_uv = local[:, :2].min(axis=0), local[:, :2].max(axis=0)
    height = local[:, 2].max()
    size = np.array([*(high_uv - low_uv), height])
    if np.any(size <= 0.01):
        raise ValueError("Target box does not sit above the ground plane")
    return (low_uv + high_uv) / 2, size


def fit_transform(asset_xyz, origin, frame, target_uv, target_size, up_axis="z",
                  yaw_deg=None, padding=0.05, clearance=0.005):
    points = np.asarray(asset_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("Asset must contain at least four 3D points")
    if not np.all(np.isfinite(points)):
        raise ValueError("Asset contains non-finite coordinates")
    target_uv = np.asarray(target_uv, dtype=np.float64)
    target_size = np.asarray(target_size, dtype=np.float64)
    if target_uv.shape != (2,) or target_size.shape != (3,) or np.any(target_size <= 0):
        raise ValueError("Target center and size must be finite positive dimensions")
    if not np.all(np.isfinite(target_uv)) or not np.all(np.isfinite(target_size)):
        raise ValueError("Target center and size must be finite")
    if not 0 <= padding < 0.5 or not 0 <= clearance <= 1:
        raise ValueError("Padding or clearance is out of range")
    up = asset_up_rotation(up_axis)
    canonical = points @ up.T
    low = np.quantile(canonical, 0.005, axis=0)
    high = np.quantile(canonical, 0.995, axis=0)
    source_size = high - low
    if np.any(source_size <= 1e-5):
        raise ValueError("Asset bounding box is degenerate")
    options = [float(yaw_deg)] if yaw_deg is not None else [0.0, 90.0]
    best = None
    for yaw in options:
        angle = math.radians(yaw)
        yaw_matrix = Rotation.from_euler("z", angle).as_matrix()
        turned = canonical @ yaw_matrix.T
        extents = np.quantile(turned, 0.995, axis=0) - np.quantile(turned, 0.005, axis=0)
        scale = (1 - padding) * float(np.min(target_size / extents))
        candidate = (scale, yaw, yaw_matrix)
        if best is None or candidate[0] > best[0]:
            best = candidate
    scale, yaw, yaw_matrix = best
    rotation = frame @ yaw_matrix @ up
    source_anchor = up.T @ np.array([(low[0] + high[0]) / 2,
                                     (low[1] + high[1]) / 2, low[2]])
    world_anchor = origin + frame @ np.array([target_uv[0], target_uv[1], clearance])
    details = {
        "scale": scale, "yaw_degrees": yaw, "asset_up_axis": up_axis,
        "source_bbox_size": source_size.tolist(), "target_bbox_size": target_size.tolist(),
        "target_plane_uv": target_uv.tolist(), "ground_clearance": clearance,
        "rotation_matrix": rotation.tolist(), "world_anchor": world_anchor.tolist(),
    }
    return rotation, scale, source_anchor, world_anchor, details


def transform_gaussians(vertices, rotation, scale, source_anchor, world_anchor):
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Scale must be finite and positive")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Rotation must be orthonormal")
    if np.linalg.det(rotation) < 0.999:
        raise ValueError("Rotation cannot reflect Gaussian splats")
    fields = vertices.dtype.names
    required = {"x", "y", "z", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"}
    if not required.issubset(fields):
        raise ValueError(f"Asset PLY lacks Gaussian fields: {sorted(required - set(fields))}")
    if not np.allclose(rotation, np.eye(3), atol=1e-6):
        rest = [name for name in fields if name.startswith("f_rest_")]
        if rest and any(np.any(np.abs(vertices[name]) > 1e-6) for name in rest):
            raise ValueError("Rotating nonzero SH coefficients is unsupported; use a DC-only generated asset")
    result = vertices.copy()
    transformed = (xyz_of(vertices) - source_anchor) @ rotation.T * scale + world_anchor
    for i, field in enumerate(("x", "y", "z")):
        result[field] = transformed[:, i]
        result[f"scale_{i}"] += math.log(scale)
    local_q = np.column_stack([vertices[f"rot_{i}"] for i in (1, 2, 3, 0)])
    if np.any(np.linalg.norm(local_q, axis=1) < 1e-8):
        raise ValueError("Asset contains zero-length Gaussian quaternions")
    composed = (Rotation.from_matrix(rotation) * Rotation.from_quat(local_q)).as_quat()
    for i, field in enumerate(("rot_1", "rot_2", "rot_3", "rot_0")):
        result[field] = composed[:, i]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plane-json", required=True,
                        help="A flat-fill preview.json containing a fitted ground plane")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-bbox", type=float, nargs=6,
                        metavar=("MIN_X", "MIN_Y", "MIN_Z", "MAX_X", "MAX_Y", "MAX_Z"))
    target.add_argument("--replace-original", help="Original scene before object removal")
    target.add_argument("--target-center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Bottom-center world point; pair with --target-size")
    parser.add_argument("--replace-pruned", help="Ordered pruned scene; pair with --replace-original")
    parser.add_argument("--target-size", type=float, nargs=3, metavar=("WIDTH", "DEPTH", "HEIGHT"))
    parser.add_argument("--asset-up-axis", choices=("x", "-x", "y", "-y", "z", "-z"), default="z")
    parser.add_argument("--yaw-deg", type=float, help="Yaw on the ground plane; default chooses 0 or 90")
    parser.add_argument("--fit-padding", type=float, default=0.05)
    parser.add_argument("--clearance", type=float, default=0.005)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if bool(args.replace_original) != bool(args.replace_pruned):
        parser.error("--replace-original and --replace-pruned must be supplied together")
    if bool(args.target_center) != bool(args.target_size):
        parser.error("--target-center and --target-size must be supplied together")
    output = Path(args.output).resolve()
    if output.exists() or output.with_suffix(".alignment.json").exists():
        raise ValueError("Refusing to overwrite an existing alignment output")
    with open(args.plane_json, encoding="utf-8") as handle:
        plane = json.load(handle)
    origin, frame = plane_frame(plane["plane_origin"],
                                plane["plane_normal_toward_removed_object"])
    source_ply, asset = read_vertices(args.asset)
    if args.replace_original:
        _, original = read_vertices(args.replace_original)
        _, pruned = read_vertices(args.replace_pruned)
        target_uv, target_size, count = target_from_removal(
            original, pruned, origin, frame)
        provenance = {"mode": "removed_object", "object_points": count,
                      "original": str(Path(args.replace_original).resolve()),
                      "pruned": str(Path(args.replace_pruned).resolve())}
    elif args.target_bbox:
        target_uv, target_size = target_from_world_box(args.target_bbox, origin, frame)
        provenance = {"mode": "world_bbox", "bounds": args.target_bbox}
    else:
        target_uv = (np.asarray(args.target_center) - origin) @ frame[:, :2]
        target_size = np.asarray(args.target_size)
        provenance = {"mode": "plane_size", "center": args.target_center}
    rotation, scale, anchor, world, details = fit_transform(
        xyz_of(asset), origin, frame, target_uv, target_size,
        args.asset_up_axis, args.yaw_deg, args.fit_padding, args.clearance)
    details.update({"target": provenance, "plane_json": str(Path(args.plane_json).resolve()),
                    "source_asset": str(Path(args.asset).resolve())})
    if args.dry_run:
        print(json.dumps(details, indent=2))
        return
    aligned = transform_gaussians(asset, rotation, scale, anchor, world)
    write_vertices(output, aligned, source_ply, ["auto-aligned Gaussian asset"])
    with open(output.with_suffix(".alignment.json"), "w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)
        handle.write("\n")
    print(f"Aligned {len(aligned):,} Gaussians -> {output}")


if __name__ == "__main__":
    main()
