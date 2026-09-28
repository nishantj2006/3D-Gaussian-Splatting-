"""Combine independently identified object instances into a reversible edit.

Each instance keeps its own masks and source Gaussian IDs.  The union is only
used for removal and image inpainting; it is not a new semantic object ID.
"""

import argparse
import json
import re
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

from utils.ply_semantic_utils import read_vertices


def validated_ids(path, count):
    ids = np.load(path, allow_pickle=False)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"Invalid Gaussian IDs: {path}")
    ids = np.unique(ids)
    if not len(ids) or ids[0] < 0 or ids[-1] >= count:
        raise ValueError(f"Out-of-range or empty Gaussian IDs: {path}")
    return ids


def combine_masks(entries, output, *, min_views=3, max_fraction=.8):
    """Build union masks without changing or discarding component masks."""
    manifests = {}
    for label, manifest_path, _ in entries:
        if label in manifests:
            raise ValueError(f"Duplicate instance label: {label}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifests[label] = json.load(handle)["views"]
    views = sorted(set.union(*(set(m) for m in manifests.values())))
    output.mkdir(parents=True, exist_ok=True)
    merged = {}
    for view in views:
        masks, evidence = [], {}
        for label, manifest_path, _ in entries:
            item = manifests[label].get(view, {})
            if not item.get("accepted"):
                continue
            path = Path(item["mask_path"])
            mask = np.asarray(Image.open(path).convert("L")) > 127
            if masks and mask.shape != masks[0].shape:
                raise ValueError(f"Incompatible mask sizes in {view}")
            masks.append(mask)
            evidence[label] = {"mask_path": str(path.resolve()),
                               "fraction": float(mask.mean())}
        if not masks:
            merged[view] = {"accepted": False, "instances": evidence}
            continue
        union = np.logical_or.reduce(masks)
        if union.mean() > max_fraction:
            merged[view] = {"accepted": False, "reason": "unsafe_mask_fraction",
                            "instances": evidence}
            continue
        path = output / f"{view}-union.png"
        Image.fromarray((union * 255).astype(np.uint8)).save(path)
        merged[view] = {"accepted": True, "mask_path": str(path),
                        "complete": len(evidence) == len(entries),
                        "mask_fraction": float(union.mean()), "instances": evidence}
    if sum(item["accepted"] for item in merged.values()) < min_views:
        raise ValueError("Too few accepted union-mask views")
    return merged


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    began = time.perf_counter()
    _, vertices = read_vertices(args.scene)
    entries = [(label, Path(manifest), Path(ids))
               for label, manifest, ids in args.instance]
    if len(entries) < 2:
        raise ValueError("At least two independently identified instances required")
    for label, _, _ in entries:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError(f"Unsafe instance label: {label!r}")
    protected = (np.unique(np.load(args.protected_indices, allow_pickle=False))
                 if args.protected_indices else np.empty(0, np.int64))
    if len(protected) and (protected[0] < 0 or protected[-1] >= len(vertices)):
        raise ValueError("Protected IDs out of range")
    ids_by_label = {}
    for label, _, path in entries:
        ids = validated_ids(path, len(vertices))
        conflict = np.intersect1d(ids, protected)
        if len(conflict) / len(ids) > args.max_protected_conflict:
            raise ValueError(f"Instance {label} conflicts with protected objects")
        ids_by_label[label] = np.setdiff1d(ids, protected)
    union = np.unique(np.concatenate(list(ids_by_label.values())))
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    views = combine_masks(entries, output / "masks", min_views=args.min_views)
    for label, ids in ids_by_label.items():
        np.save(output / f"{label}-indices.npy", ids)
        np.savez_compressed(output / f"{label}-provenance.npz",
                            ids=ids, vertices=vertices[ids])
    np.save(output / "selected-indices.npy", union)
    provenance = {}
    for label, manifest, path in entries:
        provenance[label] = {
            "manifest": str(manifest.resolve()),
            "source_ids": str(path.resolve()),
            "selected_ids": str(output / f"{label}-indices.npy"),
            "source_records": str(output / f"{label}-provenance.npz"),
            "count": len(ids_by_label[label])}
    report = {"source": str(Path(args.scene).resolve()), "instances": provenance,
              "views": views, "selected_count": len(union),
              "incomplete_views": sorted(v for v, item in views.items()
                                         if item["accepted"] and not item["complete"]),
              "overlapping_ids": int(sum(map(len, ids_by_label.values())) - len(union)),
              "protected_count": len(protected), "approved": False,
              "elapsed_seconds": time.perf_counter() - began,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--instance", nargs=3, action="append", required=True,
                   metavar=("LABEL", "MASK_MANIFEST", "SOURCE_IDS"))
    p.add_argument("--protected-indices")
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--max-protected-conflict", type=float, default=.05)
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
