# Bed-selection ablation — 2026-09-24

Source: `output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply`
(593,814 Gaussians, 128 semantic dimensions). All selections below are
**unapproved previews**. The source PLY and the older preview PLYs were not
modified. Numerical details and projected-point overlays are in
`output/bed-ops/bed-ablation-visible-2026-09-24-v2/`.

The common seed is the 6,328-splat automatic bedspread selection. The benchmark
compares each stage on the same source scene and caches MobileSAM's masks in
memory for its four variants. The baseline has three views; the diverse-view
variant has six views, each required to project at least 10% of the seed splats
into frame. The footprint variant samples the projected anisotropic Gaussian
ellipse at five points, checks local depth and opacity, and is **not** a full
alpha-composited renderer. The graph variant grows only from the footprint
selection into nearby, mask-supported splats matching adaptive seed color and
128D semantic similarity.

| Variant | Selected | Added to 6,328 seed | Net vs 3-view baseline |
|---|---:|---:|---:|
| Three-view baseline | 6,383 | 55 | 0 |
| Six bed-visible views | 6,398 | 70 | +15 |
| Anisotropic footprint, six views | 6,384 | 56 | +1 |
| 3D graph growth after footprint | 6,393 | 65 | +10 |

The graph step itself added 9 points to the footprint selection. All added
points in all variants were also in the earlier 8,353-point bed candidate;
that older selection is incomplete and **not** ground truth. The validation
overlays (`*-frame_0134.png`, `*-frame_0141.png`, `*-frame_0143.png`) place most
new points along the bedding boundary. They do not show materially more of the
bed body or wooden frame being selected. Some point centers project just above
the upper bedding edge, so the previews still require visual review.

The corrected seven-image CPU benchmark took **677.3 s (11.3 min)**, of which
623.3 s were MobileSAM and CLIP mask processing. Peak process RSS was **3,127
MiB** (about 3.05 GiB); this is process memory, not whole-system RAM. The
test suite passed **39 tests**. No approved or edited scene PLY was written.

The first six-view experiment was diagnostic only: CLIP chose `frame_0116`
with zero seed splats visible and `frame_0150` with relatively little of the
bed. Visibility filtering corrected that mistake in the report above. The
remaining bottleneck appears to be segmenting and associating the missing bed
regions across views, not choosing another threshold or fitting a larger
footprint. These counts are coverage proxies; without independently annotated
ground-truth masks, they are not precision/recall measurements.
