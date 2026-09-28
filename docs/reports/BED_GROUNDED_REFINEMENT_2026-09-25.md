# Text-grounded bed selection experiment — 2026-09-25

This experiment is preview-only. The source 5k PLY was not modified. The target
instance here is bedding, anchored to the existing 6,328-splat semantic seed;
wooden frame and bedding remain separate, not grouped into one object ID.

## Method

1. Grounding DINO tiny located user-supplied phrases `bed`, `bedding`, and
   `bedspread` in source photos. SAM 2.1 tiny segmented each candidate box.
   Every candidate mask was saved independently; the mask with the strongest
   seed agreement was associated with object ID 1 across 22 views.
2. A five-sample anisotropic footprint lift with depth and multi-view evidence
   was tested. It added only 77 splats and decreased held-out mask overlap in
   two of three views; this was rejected.
3. A differentiable Gaussian rasterizer optimized only per-splat membership
   probabilities against 19 photo masks. Geometry, colors, opacity, 128D
   semantic features, and the source PLY stayed fixed. Three views were held
   out completely from optimization.

## Results

| Held-out view | Baseline IoU | Approximate lift IoU | Renderer-refined IoU |
|---|---:|---:|---:|
| frame_0134 | 0.813 | 0.813 | 0.851 |
| frame_0141 | 0.789 | 0.780 | 0.809 |
| frame_0143 | 0.779 | 0.777 | 0.799 |

Mean held-out IoU rose from 0.794 to 0.820 (+0.026). Mean precision rose
from 0.850 to 0.865 and mean recall from 0.924 to 0.940. The refined preview
selects 6,498 of 593,814 splats: 155 added and 40 removed relative to the
6,383-splat baseline. PLY schema verification found 190 properties, including
all 128 semantic dimensions, and 587,316 remaining splats.

Grounded 2D masks took 46.3 s, peaking at 4,697 MiB process RAM and 2,221 MiB
allocated GPU memory. Renderer refinement took 88.5 s, peaking at 2,725 MiB
process RAM and 1,237 MiB allocated GPU memory. These were separate runs, not
additive peaks.

## Limitation and decision

IoU is measured against automatic SAM 2 masks, **not human-labeled ground
truth**. More importantly, RGB renders after deletion show blurred blue bed
remnants in all three held-out views. The optimized selection removes much of
the originally visible bedding but not occluded/revealed bed layers. It must
not be claimed as complete whole-bed deletion or approved for automatic
application. An iterative, appearance-constrained revealed-layer pass and
independent assessment of the bed frame are needed. Removing every splat
under the 2D mask would also delete legitimate hidden wall/floor and is unsafe.

## Files

- Mask candidates and cross-view association: `output/bed-ops/bed-grounded-sam2-masks-v1/manifest.json`
- Lift ablation: `output/bed-ops/bed-grounded-lift-v1/preview.json`
- Exact rasterizer baseline/lift comparison: `output/bed-ops/bed-grounded-render-eval-v1/report.json`
- Refined membership and held-out metrics: `output/bed-ops/bed-render-refined-v1/report.json`
- Unapproved pruned PLY: `output/bed-ops/bed-render-refined-ply-preview-v1/pruned-preview.ply`
- RGB before/after renders: `output/bed-ops/bed-render-refined-rgb-v1/`

The source PLY remains
`output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply`.
