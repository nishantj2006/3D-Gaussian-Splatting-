# Why the current bed-like residue remains: per-splat diagnosis

This is **diagnostic only** on
`output/bed-ops/bed-learned-secondpass-strict-v1/candidate.ply`. No PLY or
selection was changed. We rendered the current candidate in the edit-validation
views `frame_0134`, `frame_0141`, and `frame_0143`; ran Grounding DINO Tiny and
SAM2.1 Hiera Tiny for residual bed/frame pseudo-masks; excluded independent
frame and protected-furniture pixels; then measured the exact rasterized
influence of every surviving splat inside and outside those masks. The
held-out masks were **only used for diagnosis**, not selection or training.

For the 20 largest individual contributors, we additionally rendered a
counterfactual with just that one splat hidden, measured RGB change by image
region, and audited its contacts with protected and bed-only masks in 12
non-held-out training views. This separates optical influence from class
identity: a wall splat can influence a bed-mask pixel without being bed.

## Overall result

- 2,268 surviving splats have measurable footprint inside the diagnostic
  masks. Just **16 splats account for 50%** of the measured masked-region
  influence; 139 account for 90%.
- The category assigned as the *first applicable survival reason* explains:

  | First survival reason | Splats | Share of masked-region influence |
  |---|---:|---:|
  | Source ID protected by furniture evidence | 1,504 | 78.63% |
  | Center outside the 1.0-scene-unit local bed group | 164 | 18.30% |
  | Mixed bed/background rendered footprint | 305 | 1.42% |
  | Weak learned bed score after earlier gates | 21 | 0.66% |
  | Already opacity-attenuated by the second pass | 63 | 0.52% |
  | Single-view evidence | 123 | 0.15% |
  | Strong remaining evidence without an identified gate | 88 | 0.32% |

  Categories are **not semantic ground truth**. A splat can satisfy several
  reasons; the table partitions by first reason to avoid double-counting.
  Protected source IDs are not necessarily dresser Gaussians. Some may be broad
  or ambiguously grouped bed splats whose footprint touches furniture masks.
- In `frame_0134`, `0141`, and `0143`, **39.1%, 61.1%, and 63.6%** of the
  measured influence inside the residual mask came from splats whose centers
  project **outside** that mask. This is why point-center location and 3D
  neighbor counting miss the dominant broad splats.

## Individual cases

The percentage in “RGB effect in residue” is from removing **one splat only**
in memory, then rendering all three diagnostic views; it is the fraction of
that splat's total absolute RGB effect that falls in the safe residual masks.
It does **not** mean deletion is visually safe or that the splat is bed.

| Source ID / rank | What survives and why | RGB effect in residue | Assessment |
|---|---|---:|---|
| `216288` / 1 | Protected; center 5.65 units from deleted bed; very broad 11.01-unit axis; learned bed probability 0.079; footprint inside/outside agreement 0.385. | 57.1% | Overlay covers wall and bed region; removing whole splat would alter substantial non-bed image. Not an isolated bed point. |
| `182259` / 2 | Unprotected but center 1.24 units away, outside the local 1.0-unit cutoff; learned bed probability 0.260. It contributes to bed-only masks in 9 training views and essentially not to protected masks. | 86.7% | Plausible residual-bed contributor missed by center locality and weak classifier score; still broad (4.36-unit axis), so test splitting/local attenuation before wholesale deletion. |
| `144990` / 4 | Protected, 2.54 units away, bed probability 0.257. In training views, only 1.6% of its combined bed/protected-mask influence is on the protected mask. | 93.0% | Protection may be over-conservative for this splat, but it cannot be unprotected from these proxy masks alone. |
| `163339` / 7 | Protected, 0.67 units away, bed probability 0.755. Training protected-mask share 1.4%; strong bed-only contact in 9 views. | 94.2% | Another likely mixed/over-protected case worth a per-view footprint split, not blanket neighbor growth. |
| `428930` / 8 | Protected, 4.60 units away, learned bed probability 0.198; footprint agreement 0.316 and an ablation changes much outside the residue. | 21.8% | Strong example to **keep**: it is principally background/protected visual content, despite overlapping the residue mask. |
| `144317` / 10 | Unprotected, center 1.09 units away, just beyond local cutoff; bed probability 0.554 and footprint agreement 0.995; near-zero protected-mask contribution. | 99.4% | High-priority targeted candidate, but first check counterfactual RGB from more angles and prevent holes. |
| `399922` / 15 | Protected, only 0.45 units from deleted bed; learned bed probability 0.998. Training protected-mask share 0.1%, but its source ID remains protected. | 96.1% | The clearest possible protection conflict; still needs independent confirmation before overriding a protected source ID. |
| `105021` / 20 | Unprotected, center 1.42 units away; learned bed probability 0.427; bed-mask agreement 0.998 and protected contact effectively zero. | 99.8% | Likely outside both the original spatial neighborhood and classifier confidence gate. |

## Interpretation and next safe experiment

The remaining image is **not** mainly a cloud of small uncertain points
surrounded by deleted ones. Its visible influence is dominated by a handful of
large splats and by conservative protection boundaries. A nearest-neighbor
delete rule selected 24–31 almost invisible splats because it did not reach
those broad, sometimes distant contributors.

The next experiment should isolate the individually suspicious IDs above,
render each single-splat ablation in additional training and held-out views,
and split or locally attenuate broad splats whose bed footprint is separable.
For protected IDs, require an explicit per-view protection audit and retain
the protected portions rather than lifting protection for an entire splat.
Do **not** simply delete all 1,504 protected contributors: the top wall-like
cases show why that would damage background or furniture. The unsupported
wall-depth fit remains a separate obstacle to convincing replacement.

## Artifacts

- `output/bed-ops/bed-current-residue-diagnosis-v1/report.json` has the top 50
  source IDs, positions, scales, learned scores, reasons, and contribution
  metrics. Adjacent `frame_*-rank*-source*.png` images show the footprint of
  top individual splats over the current render. No PLY is written there.
- `output/bed-ops/bed-current-residue-case-ablation-v1/report.json` has
  one-splat RGB effect for the top 20, with amplified RGB-difference images.
- `output/bed-ops/bed-current-residue-protection-audit-v1/report.json` has
  independent bed/protected training-view contact for those cases.
- Full test suite: **144 passed, 1 skipped**. The non-escalated test run showed
  a CUDA initialization warning; GPU diagnostics completed with host
  permission.

These numbers depend on automatic residual masks and therefore support a
diagnosis, not an authorized deletion list or a claim of true bed segmentation.
