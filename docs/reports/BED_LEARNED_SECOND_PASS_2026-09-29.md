# Rendered-residue second pass on the learned-fusion bed preview

This is an **unapproved** local second-pass experiment. It starts from
`output/bed-ops/bed-learned-fusion-v1/candidate.ply`; the source scene and
earlier previews are untouched. No replacement Gaussians are created.

## What ran

1. Rendered the current edited PLY in 13 accepted **training** camera views.
   `frame_0134`, `frame_0141`, and `frame_0143` were excluded.
2. Used cached Grounding DINO Tiny and SAM2.1 Hiera Tiny to detect the still
   visible bed and the independently recognized wooden frame in those renders.
   Bed residue was detected in 13/13 views. Subtracting frame and independently
   protected furniture masks left 10 safe bed-residue views.
3. Measured each nearby surviving Gaussian's exact rasterized contribution
   inside and outside these safe masks, iteratively re-rendering after opacity
   changes. Protected source IDs were excluded. The stronger candidate uses
   three supporting views, at least 0.8 total contribution, 0.92 inside/total
   agreement for strong attenuation, and 0.8 for partial boundary attenuation.
4. Rendered both candidates in the held-out views and near-overhead cameras.

This pass **attenuates opacity**; it does not physically remove source records.
All positions, colors, 128D semantics, and other non-opacity properties remain
bit-exact relative to the learned-fusion seed.

## Comparison

| Candidate | Strongly attenuated splats | Partially attenuated splats | Bed-mask pixels changed: 0134 / 0141 / 0143 | Pixels changed outside bed/protection: 0134 / 0141 / 0143 |
|---|---:|---:|---:|---:|
| Initial setting | 133 | 174 | 0.04% / 3.15% / 4.57% | 0% / 0.44% / 0.27% |
| Stricter setting | 92 | 64 | 0% / 2.32% / 2.84% | 0% / 0.067% / 0.013% |

The stricter setting is the safer preview: no source ID in the protected set
changed, and no held-out protected-mask pixel crossed the RGB-change threshold.
The looser setting has noticeably more off-mask change without a compelling
visual payoff. The protected mask is still only furniture evidence, not a
verified dresser ground-truth annotation.

Visual inspection of `frame_0141`, `frame_0143`, and near-overhead renders shows
**only modest local cleanup**. A large bed-shaped/blurred region remains in these
diagnostic views, and the overhead appearance does not materially improve.
The strict preview should not replace the current user-reviewed scene
automatically. This is a targeted polish trial, not complete bed removal or
background reconstruction.

## Resource and integrity checks

| Stage | Wall time | Peak process RSS | Peak Torch GPU allocation |
|---|---:|---:|---:|
| 13 edited-scene renders | 13.66 s | 2,022 MiB | 932 MiB |
| Rendered residue detection | 57.92 s | 4,225 MiB | 2,217 MiB |
| Strict local attribution | 41.96 s | 3,966 MiB | 951 MiB |
| Held-out/overhead validation | 18.08 s | 4,280 MiB | 1,345 MiB |

Peak Torch allocation is part of unified-memory use and must not be added to
peak process RSS. The PLY retains all 190 properties and 128 semantic fields;
only 156 opacity values changed in the strict candidate, all within the
attribution pool. Full regression suite: **140 passed, 1 skipped**. The
non-escalated test run reported a CUDA-initialization warning, while actual GPU
rendering and attribution completed successfully with host permission.

## Review artifacts

- `output/bed-ops/bed-learned-secondpass-strict-v1/candidate.ply` — safer,
  unapproved preview.
- `output/bed-ops/bed-learned-secondpass-strict-v1/validation/comparison.png`
  and `validation/overhead-3-0.png` / `overhead-3-1.png` — before/after views.
- `output/bed-ops/bed-learned-secondpass-candidate-v1/` — looser comparison.
- `output/bed-ops/bed-learned-secondpass-renders-v1/` and
  `bed-learned-secondpass-safe-masks-v1/` — training-view evidence.
