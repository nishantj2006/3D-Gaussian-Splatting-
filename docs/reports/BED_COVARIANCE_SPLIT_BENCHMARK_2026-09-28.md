# Covariance-aware grouping and calibrated shared-splat splitting

## Result: partial improvement, unapproved

The new covariance-aware graph adds 452 source splats to the existing 7,327
removal IDs, for 7,779 removed source splats. It catches all four broad
unprotected examples from the prior diagnosis, including IDs 325426 and 302685.
Protected IDs cannot enter this removal group.

The eight-parent local split produces 128 smaller Gaussians and suppresses 88
of them. Smaller replacements are calibrated against the covariance preview's
RGB before removal; the source scene itself is never trained or overwritten.
Only shared parent records and their children can change. Unrelated records
remain exact, the 190-property schema is preserved, and the 128D semantic
features of every daughter exactly match its parent.

Actual held-out RGB shows less central blue bedding, but the bed-shaped outline,
lower wooden frame and some blue edges/shading remain. This is not a complete
bed removal. No hidden-background reconstruction or final approval is claimed.

## Method

`covariance_instance_graph.py` uses covariance-normalized proximity instead of
a fixed center radius. It accounts for scale and orientation and requires
strong training-view footprint evidence; the kernel is an approximation, not
an exact ellipsoid collision test. Hard protected-object exclusions still apply.
It reuses the existing 13-view footprint cache and does not re-run the mask
detector or the entire attribution pipeline.

`refine_shared_splats.py` chooses shared protected parents with independent
bed-footprint support, constructs moment-preserving mixtures along the two broad
axes, and locally calibrates child opacity and DC appearance. Geometry and
128D features are fixed during this calibration. Children may be suppressed
only with strong bed agreement in at least three views and negligible detected
protected-mask contribution. If no child is supported for a parent, its entire
split is reverted. Preservation failures are written as rejected diagnostics.

The measured trials used six non-held-out training views at width 180 to keep
the benchmark bounded under another workload. Thus these are sampled-view
experiments, not exhaustive validation across the capture. The default CLI
supports using all available training views by omitting `--max-views`.

## Orin stage benchmarks

| Stage | Time | Peak process RSS | Peak PyTorch GPU allocation |
| --- | ---: | ---: | ---: |
| Previous cached center-based expansion | 3.01 s | 1,517 MiB | 0 |
| Cached covariance growth/materialization | 2.26 s | 1,411 MiB | 0 |
| 4 shared parents, 64 children, 16 calibration steps | 45.15 s | 5,531 MiB | 1,380 MiB |
| 8 shared parents, 128 children, 24 calibration steps | 51.12 s | 5,515 MiB | 1,380 MiB |
| Eight-parent held-out RGB benchmark, width 540 | 15.11 s | 3,833 MiB | 1,346 MiB |

Timings are **under competing load**, not isolated performance: another conda
training process occupied roughly 38 GB of RAM. It was not stopped or changed.
An initial 13-view, width-270 split attempt terminated with exit code 143 before
producing a completion report; its cause was not established. Only its
calibration-input PLY exists, not a completed edit. Reduced-cost runs completed.
RSS and CUDA allocation overlap on Orin unified memory; do not add them.

## Held-out checks

Covariance-only removal compared with the intact source:

| View | Previous front-footprint recall | New recall | Dresser-mask changed pixels |
| --- | ---: | ---: | ---: |
| frame_0134 | 76.25% | 76.47% | 0% |
| frame_0141 | 91.50% | 94.57% | 0% |
| frame_0143 | 72.65% | 76.18% | 0.75% |

The new split compared with the covariance-only preview:

| View | Incremental dresser RGB MAE | Dresser changed pixels | Outside-target changed pixels |
| --- | ---: | ---: | ---: |
| frame_0134 | 0 | 0% | 0.129% |
| frame_0141 | 0.000232 | 0% | 0% |
| frame_0143 | 0 | 0% | 0% |

RGB error is normalized to [0,1]. A changed pixel exceeds 0.08 in at least one
channel; zero changed pixels does not mean every pixel is identical. These
checks passed their numerical preservation gates, but cannot approve the edit.
Masks are detector-generated pseudo-labels. Front-footprint recall is not
whole-object deletion accuracy.

The four-parent combined result's detected bed-mask area fell from 109,579 to
86,783 pixels in frame_0134, but rose from 118,564 to 119,992 in frame_0141 and
barely changed (100,367 to 100,057) in frame_0143. That mixed result reinforces
the visible finding: residue is not resolved consistently. This detector
comparison is not ground truth and was not used to fit an edit.

## Repeatability and tests

- 112 tests passed with one existing skip; actual GPU stages were tested on the
  Orin separately from the CPU-safe unit suite.
- Repeated covariance materialization is byte-identical, SHA256:
  `a7491fa5c3083e0581f791d8a619b562a151741d15682e8ae56268a0a58eb431`.
- Repeated eight-parent calibration at seed 0 selects identical parents and
  suppressed children. PLY schema/count match; maximum float-field difference
  is 4.53e-6. GPU calibration is therefore numerically repeatable, not
  byte-identical.

## New artifacts

- Latest unapproved split preview:
  `output/bed-ops/bed-shared-split-calibrated-v3/candidate.ply`
- Its `provenance.npz` stores original parent records, source IDs, child IDs,
  parent-child mapping and measured removal evidence.
- Its `validation/` folder contains paired `baseline/` and `split/` RGB renders
  and a numerical report for frame_0134, frame_0141 and frame_0143.
- Covariance-only preview:
  `output/bed-ops/bed-covariance-protected-v1/candidate.ply`.

Source and earlier previews remain unchanged. The split replaces eight shared
protected parents; unlike covariance-only removal, it does not claim every
protected parent row is unchanged. Instead, it preserves unrelated rows and
tests the protected object's rendered appearance.
