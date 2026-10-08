# Deleted-neighbor context trial: no material visual gain

The user proposed that a Gaussian surrounded by deleted bed Gaussians is more
likely to be bed. We tested this on the strict second-pass preview without
modifying that preview, the 5k source, or any prior output. Both new candidate
PLYs are **unapproved**.

## Method

1. Re-rendered the strict preview in 13 training views. Grounding DINO Tiny and
   SAM2.1 Hiera Tiny detected remaining bed and frame in those edited renders.
   Excluding independent frame and protected-furniture pixels left 10 safe
   bed-residue views. The held-out frames `0134`, `0141`, and `0143` were never
   used for training/mask selection.
2. Queried each surviving local splat's nearest source Gaussians. Weighted
   neighbors by the overlap implied by both splats' 3D covariance, then scored
   the fraction already confirmed as bed. Calibrated the threshold using the
   source bed and 2,969 protected nearby negatives. Hard-excluded protected
   source IDs and intact splats close to the independently fitted carpet plane.
3. Required the learned bed class to beat background and frame, plus positive
   rendered contribution inside the new bed-residue masks from multiple views
   and high inside/outside attribution agreement. Only then reduced opacity.

The rule is not "delete every spatial neighbor": carpet, wall, frame, and
dresser can all be close to a selected bed splat. The test uses no color-name
or scene-specific coordinate rule.

## Results

| Setting | Local unprotected pool | Graph + learned + floor supported | Accepted by multi-view residue | Held-out changed bed pixels (>0.08 RGB) | Protected source IDs changed |
|---|---:|---:|---:|---:|---:|
| Conservative | 4,239 | 72 | 24 | 0 in all 3 views | 0 |
| Broader | 4,239 | 108 | 31 | 0 in all 3 views | 0 |

The conservative graph threshold was 0.723 (the 10th percentile of confirmed
bed neighbor scores); the broader threshold was 0.543 (the 95th percentile of
protected-neighbor scores). The carpet-plane check excluded 593 possible local
splats before attribution. Both candidates retain all 190 PLY properties and
all 128 semantic dimensions. Validation confirms the only changed source
records are the intended 24/31 opacity values, and protected source records
remain exact.

At the pixel level, the broader edit changed zero pixels in `frame_0134`, 242
pixels in `frame_0141` (maximum channel change 3/255), and 364 pixels in
`frame_0143` (maximum 12/255). The 3 m and 5 m overhead renders are **bitwise
identical** before and after. Visual review likewise finds no material cleanup
of the remaining bed-shaped blur. This is a **negative result** for the current
instance: safer neighborhood growth finds only nearly invisible residual
splats. Lowering the rule until the blur disappears would abandon its
separation from nearby background/protected splats and is not justified by
these measurements.

## Runtime and outputs

The 13-view render took 15.20 s (2,022 MiB peak process RSS, 932 MiB Torch
allocation). Residue detection took 45.73 s (4,209 MiB RSS, 2,217 MiB Torch
allocation). Each neighborhood attribution took about 20 s (4,032–4,036 MiB
RSS, 947 MiB Torch allocation); each held-out/overhead comparison took about
19–20 s. Torch allocation and process RSS overlap in Jetson unified memory and
must not be summed. Full regression suite: **142 passed, 1 skipped**; a
non-escalated test run warned about CUDA initialization, while the GPU trials
completed with host permission.

- `output/bed-ops/bed-neighbor-pass-v1/candidate.ply`: conservative,
  unapproved trial.
- `output/bed-ops/bed-neighbor-pass-broader-v1/candidate.ply`: broader,
  unapproved trial; `validation/comparison.png` and overhead images are adjacent.
- `output/bed-ops/bed-neighbor-pass-safe-masks-v1/`: protected bed-residue
  masks used for both trials.

Recommendation: retain the user-reviewed preview. The remaining artifact is
not resolved by local adjacency; investigate its contributing broad or
view-dependent splats and the 3D reconstruction/replacement limitations before
promoting a new PLY.
