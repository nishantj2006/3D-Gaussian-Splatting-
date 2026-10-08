# GaussianCut bed selection benchmark — 2026-09-28

## Decision

Keep the existing preview. None of the new edits is approved. The fresh,
unseeded graph-cut result is worse than the existing selection in the held-out
mask/footprint comparison. Seeded graph refinement gains a little coverage but
exceeds the furniture-preservation gate. Actual RGB renders still show a
recognizable bed-shaped haze, blue edges, and wooden base/frame.

Background reconstruction was not run: the existing wall fit is unsupported
(19.583% inliers versus the 30% gate). Removing additional splats does not
establish the hidden wall depth. No paid API, image generation, model-weight
download, full training, or approved replacement was performed.

## What was executed

- Downloaded the official GaussianCut repository at commit
  `93d24a4ba02944ffae7848f9882b083cd9e04640` into `external_tools/GaussianCut`.
- Called its unchanged `graphcut_segmentation` implementation on all 593,814
  source Gaussians, using its 10-neighbor graph, five source/sink clusters, and
  default position/color energy. No replacement graph solver was substituted.
- Installed only PyMaxflow 1.3.2 in `gaussian-orin` with `--no-deps`.
  NumPy remains 1.26.4 and PyTorch remains 2.8.0.
- Used our rasterizer and camera JSON/masks as adapters, rather than installing
  the upstream segmentation and CUDA training stack. Protected furniture IDs
  received strong sink capacities and a post-solver exclusion check.
- Materialized retained original records ourselves, preserving all 190 PLY
  properties and 128D semantic features; bypassed upstream's geometry-only
  writer. All sources and previews that existed before this work are unchanged.

This is an adapted solver benchmark, not an unmodified end-to-end reproduction
of the published method. The three held-out views (`frame_0134`, `frame_0141`,
`frame_0143`) were excluded from evidence collection and calibration.

## Fresh unseeded comparison

Collected intact-original-scene contributions in 13 training views at width 270,
with no deletion opacity applied. This supplied 5,071 confident positive seeds
and 400,256 negative cluster seeds. The 17,253 independently protected source
IDs were retained. No old bed-selection IDs were injected as positive seeds.

| Held-out view | Current deleted-front recall | Fresh GaussianCut recall | Current furniture changed pixels | Fresh GaussianCut furniture changed pixels |
|---|---:|---:|---:|---:|
| frame_0134 | 91.60% | 81.35% | 0.00% | 0.00% |
| frame_0141 | 98.45% | 95.70% | 0.00% | 0.00% |
| frame_0143 | 96.33% | 94.57% | 0.75% | 2.52% |

Mean recall: current 95.46%, fresh GaussianCut 90.54%. Mean footprint IoU:
current 86.56%, fresh GaussianCut 66.16%. Fresh selection removes 7,391 bed
splats (7,612 including the independently recorded frame selection), versus
7,557 bed splats in the existing selection. It adds 691 bed IDs and restores 857.

These are **deleted front-footprint metrics against automatic pseudo-masks**,
not whole-bed removal accuracy. Furniture changed pixels use a maximum RGB
channel difference greater than 0.08 relative to the original render. The
furniture mask is not verified single-instance dresser ground truth: some views
also include other furniture/frame regions. Zero protected source IDs deleted
does not guarantee unchanged furniture pixels when broad/shared splats change.

Visual review of all three RGB renders shows more surviving blue bedding and
stretched dark edges than the current preview. The near-overhead diagnostic is
poorly reconstructed/outside reliable capture coverage and is not evidence of
successful removal. No visual acceptance claim is made.

## Incremental seeded refinement

An additional trial supplied the current bed IDs as positive evidence, with
terminal evidence weight 10. This is incremental refinement, not independent
segmentation. It selected 8,258 bed splats: 735 additions and 34 restorations.
Its maximum furniture-mask changed fraction was 1.13%, above the 1% gate.

A training-only revealed-footprint guard examined the additions across 15
non-held-out protection views and retained 14 ambiguous source IDs. The guarded
selection removes 8,244 bed splats (8,466 total). Mean source-deletion footprint
recall reaches 96.84%, but maximum furniture changed pixels remains about 1.13%.

Then 16 strongly supported shared parents were calibrated as 256 children over
all 13 training views, with 52 optimization steps at width 270. Of these children,
139 were suppressed. Unrelated records remained exact and all child 128D
semantic fields exactly copied their parent fields.

The split step alone passed its **incremental** held-out preservation checks:
no additional furniture pixels crossed the 0.08 change threshold, and the
maximum additional outside-mask changed fraction was 0.1203%. This does not
undo the graph stage's error. Comparing the entire refined edit with the original
still gives up to 1.18% changed furniture-mask pixels. The RGB renders retain
the bed-shaped brown/grey haze and blue boundary. The result remains unapproved.

## Runtime and memory on the Orin

| Stage | Seconds | Peak process RSS MiB | Peak PyTorch GPU allocation MiB |
|---|---:|---:|---:|
| Fresh intact training evidence | 23.54 | 2,110.47 | 972.44 |
| Fresh unseeded official graph solver + PLY materialization | 179.76 | 2,228.76 | 0 |
| Fresh held-out RGB/footprint/overhead validation | 20.20 | 4,274.80 | 1,361.01 |
| Seeded official graph solver + PLY materialization | 179.03 | 2,239.27 | 0 |
| Revealed furniture guard | 25.75 | 3,130.00 | 964.49 |
| Shared-splat calibration and suppression | 99.13 | 5,630.77 | 1,493.32 |
| Incremental shared-splat held-out validation | 15.35 | 3,836.45 | 1,346.01 |

RSS is per-process high-water memory, not whole-machine RAM. Orin uses unified
memory: do not add process RSS and GPU allocation as independent physical RAM.
Measurements are not isolated hardware/power-mode-controlled research timing.
The solver stages exclude mask/model preparation performed in other stages.

## Provenance and artifacts

- Existing baseline: `output/bed-ops/bed-shared-split-calibrated-v3/candidate.ply`.
- Fresh evidence: `output/bed-ops/bed-gaussiancut-intact-evidence-v1/`.
- Fresh unseeded PLY and comparison:
  `output/bed-ops/bed-gaussiancut-intact-protected-v1/`.
- Seeded trial: `output/bed-ops/bed-gaussiancut-seeded-protected-v3/`.
- Guarded trial: `output/bed-ops/bed-gaussiancut-revealed-guard-v1/`.
- Refined diagnostic PLY, separate split validation, and final RGB comparison:
  `output/bed-ops/bed-gaussiancut-shared-refined-v1/`.

Source IDs, retained IDs, separate bed/frame labels, deleted records, and
parent-child split provenance are saved alongside the candidates. Candidate
PLY filenames do not imply approval.

The initial `bed-gaussiancut-protected-v1` trial used post-removal cached
evidence (only 710 positive seeds). Its low coverage is not a fair unseeded
comparison; use the fresh result above. A setup-correction trial in
`bed-gaussiancut-seeded-protected-v2` was interrupted before producing a result
and is excluded. Original PLYs, prior previews, and photos were never overwritten.

Non-inference regression suite: 120 passed, 1 skipped. CUDA checks/rendering ran
successfully in the GPU-enabled execution context; pytest's sandbox-only CUDA
warning is not a GPU-stage failure. New graph trials and the larger split were
not rerun for bitwise reproducibility; KMeans random state and calibration seed
were fixed. The fresh intact evidence benchmark should be the reproducible
starting point for future parameter/instance-mask studies.
