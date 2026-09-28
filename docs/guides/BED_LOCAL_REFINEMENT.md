# Bed removal: local depth-guided preview (not approved)

This workflow uses the 5k PLY, existing SAM 2 bed masks, nearby wall masks, camera
poses, and original photographs. It never reruns the 5k training job or overwrites
the source. All output paths below are new preview folders.

## Pipeline

1. `grounded_surface_masks.py` finds candidate wall pixels independently from
   the bed; `filter_surface_masks.py` rejects masks overlapping the foreground.
2. `fit_depth_wall.py` renders source depth in training wall masks, back-projects
   the pixels, and fits a vertical wall plane in the established floor frame.
   It rejects poor inlier ratio, plane error, or cross-view support.
3. `build_depth_background.py` seeds a brown wall plane from wall-photo colors
   and 128D wall semantics, plus carpet Gaussians copied from intact floor.
   This is an extrapolation of unseen surfaces, not ground-truth recovery.
4. `refine_local_background.py` tests differentiable local opacity optimization.
   Inside bed masks it uses a rendered replacement target; outside it uses the
   original photos. The geometry and semantic features are frozen.
5. `attribute_revealed_bed.py` exposes the source splats by hiding the synthetic
   fill, then attributes their rendered footprint to the bed mask across views.
   It peels confident layers and attenuates ambiguous boundary splats. The
   `materialize_local_gates.py` step writes an exact-property PLY from that
   evidence, leaving untouched splats bit-for-bit unchanged.
6. `render_local_components.py` separates retained scene, wall, and floor;
   `render_ply_preview.py` checks held-out RGB views. Never approve from mask
   scores alone.

## Current run

- Source: `output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply`
- Existing selection: `output/bed-ops/bed-revealed-pass-v1/selected-indices.npy`
- Wall fit: `output/bed-ops/bed-depth-wall-fit-v1/report.json`
- Replacement seed: `output/bed-ops/bed-depth-background-v1/surface-seeded-preview.ply`
- Latest *unapproved* PLY: `output/bed-ops/bed-revealed-depth-preview-v2/candidate.ply`
- Held-out renders: `frame_0134.png`, `frame_0141.png`, `frame_0143.png` in the v2 folder.

The wall fit had 16.8% inliers from 12 training views and a 0.263-unit 90th
percentile plane error. The seed removed the initial 6,788 selected bed splats
and added 651 carpet and 5,400 wall splats. Cross-view peeling identified
1,615 additional splats for strong suppression and 616 for partial attenuation.
The v2 PLY changes only those 2,231 opacity values; it retains all 190 PLY
properties and all 128 semantic-feature dimensions. The peel took 172 seconds,
with 4,016 MiB peak process RSS and 972 MiB peak GPU allocation. The separate
gradient-opacity test took 120 seconds, 4,477 MiB RSS, and 1,259 MiB GPU.

The result **does not pass visual review**. A flat bed-shaped brown rectangle,
dark bottom edge, and some colored streaks remain in oblique views. Component
renders show both retained bed geometry and a replacement plane clipped to the
bed silhouette. The current wall color is extrapolated up to 3.8 scene units
from observed wall-photo samples (90th percentile), so its edge/texture should
not be called accurate. The synthetic top-down camera is outside the captured
trajectory and is too blurred to validate the repair.

## Safety and next work

The source and older previews are unchanged. The latest preview is reversible
and explicitly `approved: false`. No edit should be committed until a broader
wall/floor reconstruction removes the rectangular boundary without covering
intact furniture. The present boundary treatment attenuates broad splats; it
does not yet geometrically split them. A credible next version needs a larger,
photo-consistent wall surface and floor continuation, plus an independent mask
for the wooden frame before treating that frame as removable or protected.

Run the safety checks with:

```bash
/home/nishantj/miniforge3/envs/gaussian-orin/bin/python -m pytest -q tests/test_local_bed_background.py tests/test_materialize_local_gates.py
```
