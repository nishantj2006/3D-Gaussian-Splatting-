# Bed learned-evidence fusion: unapproved component result

## Change and scope

`gsedit.selection.fuse_instance_scores` now adds learned bed evidence to the
existing covariance-aware instance selection. A new source Gaussian requires a
strong learned bed probability and class margin, rendered bed footprint support
in multiple training views, limited contradictory exterior evidence, proximity
to the existing 3D bed component, and covariance-aware overlap. Protected IDs,
already classified frame splats, and unrelated source records are excluded.
The learned 16D object-feature classifier is auxiliary; the scene's 128D
semantic features are unchanged. This is a separate preview stage, not an
approved edit or a replacement of the autonomous runner.

Inputs: `bed-covariance-protected-v1` selection, `bed-spatial-dresser-protected-v2`
footprint evidence, and the 1,000-step Inpaint360GS adapted trial's source
probabilities. `frame_0134`, `frame_0141`, and `frame_0143` remain held out.
All source scenes and earlier previews are unchanged.

## Measured result on the Orin

| Measure | Result |
|---|---:|
| Parent selected source splats | 7,779 |
| Learned + footprint eligible splats | 294 |
| Near the existing bed component | 280 |
| Added after covariance overlap | 263 |
| Total selected in unapproved preview | 8,042 |
| Protected source IDs removed | 0 |
| Raw high-bed-score IDs among protected source IDs | 574 |
| Fusion stage wall time / peak process RSS | 2.25 s / 1,382 MiB |
| Held-out render/validation wall time / peak RSS / Torch GPU allocation | 17.02 s / 3,799 MiB / 945 MiB |

The 574 protected raw scores demonstrate why a learned-score-only deletion rule
is unsafe. The new candidate preserves all 190 PLY properties and all 128
semantic dimensions; disk-read validation shows every retained original record
is exactly unchanged.

At a visible RGB-change threshold of 0.08, **incremental change from the 263
added deletions** was:

| Held-out view | Pixels changed inside bed mask | Changed inside protected mask | Changed outside both masks | Added splat footprint precision in bed mask |
|---|---:|---:|---:|---:|
| `frame_0134` | 0.00% | 0.00% | 0.00% | no rendered footprint |
| `frame_0141` | 6.37% | 0.00% | 0.16% | 86.96% |
| `frame_0143` | 8.13% | 0.00% | 0.22% | 88.70% |

These are changed-pixel fractions, **not** true-positive bed-removal recall.
The protected mask is existing furniture evidence, not independently verified
dresser ground truth. Inspection of actual RGB renders at `frame_0141` and
`frame_0143` still shows a substantial recognizable bed-like body. Thus the
fusion is a modest, reasonably localized improvement in two views, not a
solution to whole-bed deletion. Its small extra change outside masks also
deserves review. No visual approval is granted.

## Background geometry decision

The pre-existing floor plane has 96.78% inliers, but the relevant wall fit has
only 19.58% and `geometry_supported: false`. The selected region spans both
floor and wall, and bed-like splats are still visible. This experiment therefore
does **not** create replacement Gaussians or claim a completed 3D wall/floor
repair. Existing floor-atlas experiments are not evidence of reliable hidden
wall depth. The next bottleneck is attribution of newly revealed/broad splats,
then independently validated depth before a shared-atlas fill is materialized.

## Artifacts and checks

- `output/bed-ops/bed-learned-fusion-v1/candidate.ply`: unapproved removal-only
  preview. Source ID arrays, deleted records, and provenance are adjacent.
- `output/bed-ops/bed-learned-fusion-v1/validation/comparison.png`: source,
  parent selection, and fusion in the three held-out views; per-view images and
  machine-readable metrics are adjacent.
- `output/bed-ops/bed-learned-fusion-v1/report.json` and
  `validation/report.json`: source, parameters, stage timing and checks.
- Regression suite: 138 passed, 1 skipped. The non-escalated test run warned
  about CUDA initialization; the escalated GPU rendering/validation completed.

Reproduce with:

```bash
python -m gsedit.selection.fuse_instance_scores \
  --edit-dir output/bed-ops/bed-covariance-protected-v1 \
  --evidence-dir output/bed-ops/bed-spatial-dresser-protected-v2 \
  --probabilities output/bed-ops/research-2026-inpaint-component-1000-v1/source-instance-probabilities.npy \
  --output-dir output/bed-ops/NEW-FUSION-PREVIEW
```

The command refuses an existing output directory.
