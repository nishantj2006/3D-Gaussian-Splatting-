"""Materialize saved local opacity gates in a new, reversible PLY preview."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from utils.ply_semantic_utils import read_vertices, write_vertices


def apply_gates(vertices, indices, gates):
    if indices.ndim != 1 or gates.shape != indices.shape or not len(indices):
        raise ValueError("Indices and gates must be equal-length nonempty vectors")
    if indices.min() < 0 or indices.max() >= len(vertices) or len(np.unique(indices)) != len(indices):
        raise ValueError("Indices are invalid or duplicated")
    if not np.isfinite(gates).all() or (gates < 0).any() or (gates > 1).any():
        raise ValueError("Gates must be finite in [0,1]")
    result = vertices.copy()
    changed = gates < 1
    ids = indices[changed]
    raw = np.asarray(vertices["opacity"][ids], dtype=np.float64)
    base = 1/(1+np.exp(-raw))
    adjusted = np.clip(base*gates[changed], 1e-6, 1-1e-6)
    result["opacity"][ids] = np.log(adjusted/(1-adjusted))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--parent-report", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    output = Path(a.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    seed_ply, vertices = read_vertices(a.seed)
    with np.load(a.evidence, allow_pickle=False) as evidence:
        indices, gates = evidence["indices"], evidence["gate"]
    result = apply_gates(vertices, indices, gates)
    output.mkdir(parents=True)
    candidate = output / "candidate.ply"
    write_vertices(candidate, result, seed_ply,
                   ["UNAPPROVED local-opacity preview; exact untouched-property preservation"])
    with open(a.parent_report, encoding="utf-8") as handle:
        parent = json.load(handle)
    report = {"candidate": str(candidate), "seed": str(Path(a.seed).resolve()),
              "evidence": str(Path(a.evidence).resolve()), "parent_report": str(Path(a.parent_report).resolve()),
              "training_views": parent["training_views"],
              "holdout_views": parent["holdout_views"],
              "confirmed_removed": parent["confirmed_removed"],
              "boundary_attenuated": parent["boundary_attenuated"],
              "properties": len(result.dtype.names),
              "semantic_dimensions": len([x for x in result.dtype.names if x.startswith("semantic_")]),
              "changed_opacities": int(np.count_nonzero(result["opacity"] != vertices["opacity"])),
              "non_opacity_fields_identical": all(np.array_equal(result[k], vertices[k])
                                                  for k in result.dtype.names if k != "opacity"),
              "approved": False, "elapsed_seconds": time.perf_counter()-start,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
