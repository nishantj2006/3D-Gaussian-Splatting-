# Bed + frame removal and shared-background experiment (unapproved)

The original 5k scene and all earlier previews were left unchanged. This run
keeps the bed body and wooden frame as separate instance IDs, masks, and full
190-property source-Gaussian records. Their union selected 17,097 source splats
(6,221 bed, 10,896 frame, 20 overlaps); the resulting unapproved PLY preserves
128 semantic dimensions. Six accepted camera views lack one instance mask,
including `frame_0138`, so they cannot validate complete two-part removal.

The shared floor atlas uses 14 non-held-out photos and a robust ground-footprint
bound from the deleted bed IDs. It transfers observed carpet detail once, then
projects the same texture into every diagnostic view. Local Stable Diffusion
inpaints the wall; a final step locks floor pixels back to the shared atlas.
`frame_0134`, `frame_0141`, and `frame_0143` were excluded from atlas fitting.

## Measured result

| Stage/check | Result |
| --- | --- |
| Shared atlas | 20.61% directly observed texels; 4.95 s; 1,322 MiB peak process RSS |
| Five-view local diffusion | 43.93 s; 6,615 MiB peak RSS; 2,539 MiB peak PyTorch GPU allocation |
| Frame+bed revealed-layer pass | 771 additional splats strongly suppressed, 1,752 attenuated; 73.88 s; 3,942 MiB RSS; 952 MiB GPU allocation |
| Two-view carpet RGB L1 | 0.0431, below the 0.08 gate and down from the prior 0.116 |
| All usable view-pair carpet RGB L1 | 0.0418–0.0548; unsupported pairs have no geometric overlap |
| Wall refit | **19.58% inliers**, below the required 30%; geometry unsupported |
| Held-out 3D renders | Bed body still plainly visible; **fails visual review** |
| Held-out pixels changed outside union mask | 4.31%, 2.71%, 3.87% at `frame_0134`, `frame_0141`, `frame_0143` (difference >16/255) |

The shared atlas improved cross-view carpet *color* agreement, but the 2D
preview still shows a visible wall/carpet boundary. The original photographs
also still show the gray foreground bottle; it is not restored to the edited
3D PLY. The all-view image-consistency gate fails because one wall pair has
0.0907 RGB L1 and some camera pairs have no overlap. The RGB metric is
conditioned on a speculative wall plane and does **not** validate depth.

The 3D wall is not reliable and the local deletion still leaves a recognizable
bed. Therefore **no wall/floor replacement Gaussians were generated or
approved**. Threshold relaxation would conceal, not solve, these failures.
Additional unobstructed wall/floor views or a stronger independently validated
3D layout are needed for an approvable 3D reconstruction.

## Main artifacts

- Separate instance records and masks: `output/bed-ops/bed-body-frame-union-v2/`
- Unapproved first-pass PLY: `output/bed-ops/bed-body-frame-pruned-v1/pruned-preview.ply`
- Unapproved revealed-layer PLY: `output/bed-ops/bed-body-frame-revealed-v1/candidate.ply`
- 2D generated views and shared floor: `output/bed-ops/bed-body-frame-shared-locked-v2/` and `output/bed-ops/bed-body-frame-shared-guides-v6/floor-atlas.png`
- Geometry rejection: `output/bed-ops/bed-body-frame-wall-refit-v1/report.json`
- Held-out 3D views and 2D consistency: `output/bed-ops/bed-body-frame-renders-v2/` and `output/bed-ops/bed-body-frame-consistency-all-v1/report.json`

The conda environment used was `gaussian-orin`; the full Python suite passed
92 tests. Process RSS and PyTorch GPU allocation are different measurements
on unified memory and must not be added together as physical RAM.
