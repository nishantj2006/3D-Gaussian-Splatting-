"""Schema-preserving helpers for semantic Gaussian PLY files."""

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def numbered_fields(dtype_names, prefix):
    fields = [name for name in dtype_names if name.startswith(prefix)]
    return sorted(fields, key=lambda name: int(name[len(prefix):]))


def read_vertices(path):
    ply = PlyData.read(str(path))
    return ply, ply["vertex"].data.copy()


def add_float_property(vertices, name, value=0.0):
    if name in vertices.dtype.names:
        result = vertices.copy()
        result[name] = value
        return result
    dtype = np.dtype(vertices.dtype.descr + [(name, "<f4")])
    result = np.empty(vertices.shape, dtype=dtype)
    for field in vertices.dtype.names:
        result[field] = vertices[field]
    result[name] = value
    return result


def align_dtype(vertices, target_dtype):
    missing = set(target_dtype.names) - set(vertices.dtype.names)
    extra = set(vertices.dtype.names) - set(target_dtype.names)
    if missing or extra:
        raise ValueError(f"PLY schemas differ; missing={sorted(missing)}, extra={sorted(extra)}")
    result = np.empty(vertices.shape, dtype=target_dtype)
    for field in target_dtype.names:
        result[field] = vertices[field]
    return result


def write_vertices(path, vertices, source_ply=None, comments=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if source_ply is not None:
        kwargs.update(text=source_ply.text, byte_order=source_ply.byte_order,
                      comments=list(source_ply.comments), obj_info=list(source_ply.obj_info))
    if comments:
        kwargs["comments"] = kwargs.get("comments", []) + list(comments)
    PlyData([PlyElement.describe(vertices, "vertex")], **kwargs).write(str(path))


def sidecar_path(ply_path):
    return Path(ply_path).with_suffix(".objects.json")


def read_object_manifest(ply_path):
    path = sidecar_path(ply_path)
    if not path.exists():
        return {"objects": {}}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_object_manifest(ply_path, manifest):
    path = sidecar_path(ply_path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
