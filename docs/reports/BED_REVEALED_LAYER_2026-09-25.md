# Revealed-layer bed-removal preview — 2026-09-25

The first bed-removal preview left blurred blue splats that became visible only
after the front bed splats were removed. This experiment tests a conservative,
repeatable second pass; it does **not** retrain the 5k scene or modify the
source PLY.

## Method

`refine_revealed_layers.py` first suppresses the confirmed first-pass splats in
the renderer, then measures each newly visible candidate's exact rendered
contribution inside and outside the training photo masks. Candidates must be
near the confirmed 3D object, appear inside its mask in at least two views,
and match its learned color and 128D semantic appearance. The thresholds are
calibrated from the confirmed splats, not a hardcoded blue color. Three views
(`frame_0134`, `frame_0141`, `frame_0143`) stay held out.

| Selection | Splats removed | Held-out IoU 0134 | 0141 | 0143 |
|---|---:|---:|---:|---:|
| First pass | 6,498 | 0.8511 | 0.8091 | 0.7991 |
| Revealed pass 1 | 6,788 (+290) | 0.8485 | 0.8428 | 0.8462 |
| Revealed pass 2 | 6,809 (+21) | 0.8485 | 0.8428 | 0.8463 |

The first revealed pass raises mean held-out automatic-mask IoU from 0.8198
to 0.8458. The second iteration is negligible, so the **first** revealed pass
was selected as the preview. Its RGB changes were concentrated within the bed
mask: the fraction of pixels changing by more than 16/255 outside the mask was
0.25%, 0.04%, and 0.03% in the three held-out views. These numbers are
automatic-mask proxies, not manually labeled accuracy.

The bed is still visibly present in the RGB after-removal views. This is not
complete whole-bed deletion, and the preview remains unapproved. Automatically
stripping all residual pixels beneath the original 2D mask would risk removing
hidden wall and floor splats. Better coverage requires new evidence or an
explicitly reviewed object boundary, not merely a lower threshold.

## Compute on this Jetson AGX Orin

The 19-view first revealed pass took **94.5 s**, with peak process RSS
**2,150 MiB** and peak PyTorch GPU allocation **1,243 MiB**. Repeating it
for a third layer took **54.9 s** but found only 21 more splats. The original
22-view Grounding DINO/SAM 2 mask stage took 46.3 s (4,697 MiB RSS,
2,221 MiB GPU allocation), and the initial renderer fit took 88.5 s
(2,725 MiB RSS, 1,237 MiB GPU allocation). A fresh two-pass workflow is
therefore about **4 minutes** on this Orin; with prior masks and first-pass
selection cached, only about **1.5 minutes**. These are separate process peaks.
On this unified-memory system, process RSS and CUDA allocation are not
independent physical-memory totals and should not be summed.

No new packages or model weights were needed for the second pass. The current
`gaussian-orin` conda environment already contains the rasterizer, PyTorch,
and required dependencies. No 5k training rerun is needed.

## Preview artifacts

- Selected indices and per-candidate evidence: `output/bed-ops/bed-revealed-pass-v1/`
- RGB original/pruned/difference renders: `output/bed-ops/bed-revealed-pass-rgb-v1/`
- Separate unapproved PLY: `output/bed-ops/bed-revealed-pass-ply-preview-v1/pruned-preview.ply`
- Diminishing-return second iteration: `output/bed-ops/bed-revealed-pass-v2/`

The preview PLY has 587,026 remaining splats and preserves all 190 properties,
including 128 semantic dimensions. The source scene and previous previews
remain unchanged.
