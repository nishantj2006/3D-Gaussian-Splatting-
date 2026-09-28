# Editing benchmark — 2026-09-24

Run `benchmark_edits.py` with the `gaussian-orin` Python environment to repeat
the CPU-side measurements. The only independent capture in this workspace is
`data/my_scene`; the bed, bottle, and inserted basketball are cases within that
one room. This is not a multi-scene accuracy evaluation.

| Case | Selected / input splats | Wall time | Peak process RSS |
| --- | ---: | ---: | ---: |
| Bed, text vector only | 4,541 / 593,814 | 14.608 s | 1,862.3 MB |
| Bed, semantic + blue-photo cue | 8,353 / 593,814 | 18.512 s | 2,548.8 MB |
| Bed, seeded 3-view MobileSAM | 8,353 / 593,814 | 52.508 s | 2,932.2 MB |
| Inserted ball, exact object ID | 60,000 / 653,814 | 3.331 s | 1,282.0 MB |

The seeded MobileSAM step added **zero** splats. Its selected-index Jaccard
agreement with the previous bed edit was 1.0. The fused selector also
reproduced the prior bed edit exactly. MobileSAM's recall of the coherent
*seed projections* was 80.8%, 87.5%, and 88.7% in frames 0135, 0139, and
0141; this is not recall of the complete physical bed.

Additional selector comparisons, using previous edit selections as **proxies,
not ground truth**:

| Case | Vector selection | Previous edit | Intersection | Vector outside previous edit | Previous edit missed by vector | Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bed | 4,541 | 8,353 | 3,806 | 735 | 4,547 | 0.4188 |
| Bottle | 118,521 / 475,825 scene splats | 4,385 | 1,156 | 117,365 | 3,229 | 0.0095 |

The bottle cut points matched exact original-scene XYZ values. The prior cuts
are not human-annotated complete-object masks, so these overlap values are **not
precision, recall, IoU against ground truth, or unintended-deletion rates**.

Tests: 30/30 passed. Original scene PLY unchanged. All 128 semantic properties
were retained in the multi-view labeled preview. Runtime and RSS are from this
CPU-only sandbox and should not be reported as Jetson GPU timings or total
system/unified-memory use. New image rendering was unavailable because CUDA
could not initialize here, so no comparable visual-quality score was measured.
