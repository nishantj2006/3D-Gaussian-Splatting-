# Why the protected spatial preview still looks like a bed

Diagnostic only: no PLY was written or changed. Analysis uses the current
7,327-splat removal preview and separately detects visible residue in
frame_0134, frame_0141 and frame_0143. These held-out images were analyzed for
diagnosis, not used to fit or materialize another edit.

## Findings

There are two distinct blockers, not simply weak semantic vectors:

1. **The center-based graph misses broad bed-local splats.** Source ID 325426
   has 99.64% bed-footprint agreement and contribution in 12 training views,
   but its center is 1.16 scene units from the removed bed group; the graph
   connection radius is only 0.53. Its largest Gaussian scale is 1.61 scene
   units. ID 302685 has 89.76% training agreement in 13 views, a center distance
   of 3.04, and maximum scale 2.89. Their rendered extent reaches the bed even
   though their centers are disconnected from the seeds. Center proximity is
   therefore a poor proxy for overlap for these broad Gaussians.
2. **Hard protection preserves mixed bed/background/furniture contributions.**
   The protected component renders visibly include blue bedding edges and the
   bed-shaped lower frame, as well as real background/furniture. Turning off
   protection globally would undo the preservation improvement. Some protected
   IDs need stronger instance attribution or splitting, not blanket deletion.

Across the three detected residue masks, opacity-weighted contribution is:

| Category | Contributing source splats | Contribution share |
| --- | ---: | ---: |
| Protected furniture set | 1,471 | 53.87% |
| Unprotected, >=80% residue-footprint agreement and >=2 views | 831 | 38.64% |
| Other mixed footprint | 514 | 6.95% |
| Mostly exterior footprint | 238 | 0.54% |

These shares measure rendering within automatic residual masks, **not the
percentage of true bed geometry in each group**. Newly visible wall/background
can lie inside a mask. A high residue-footprint score does not prove object
identity on its own. The protected set also covers more than one neighboring
furniture proposal; it must not be equated with a verified dresser instance.

Only 222 source splats account for 90% of the measured residual contribution.
Contribution from splats whose centers are outside the detected residue mask
is 39.57%, 29.20%, and 30.07% respectively in the three views. This again shows
why center-only grouping cannot fully attribute the rendered residue.

## Rendering check

Component images preserve all retained splats' opacity and occlusion; only
colors outside the measured component are zeroed. They are additive
contributions, not isolated-object renders that expose different geometry.
Their summed RGB matches the complete renderer within 5.97e-7 maximum channel
error. Visual inspection of both protected and unprotected components confirms
that both retain parts of the visible bed-like layer.

The unit suite passed 108 tests with one existing skip. GPU attribution was
run separately on the Orin. The measured attribution stage took 49.75 seconds.

## Next targeted change

- Replace fixed center-distance edges with size/orientation-aware overlap and
  rendered-footprint evidence, while preserving strong non-target barriers.
- Revalidate the 831 unprotected bed-local candidates on training views and
  attribute newly revealed layers; do not use the held-out diagnosis as an
  automatically approved deletion list.
- For protected broad splats producing both bed and furniture, refine separate
  instance evidence and split only demonstrably shared contributors. Require
  outside-mask and furniture rendering checks before retaining any edit.

The evidence does not justify a larger unrestricted radius or box cut.

## Artifacts

- RGB components, source IDs, geometry and numeric report:
  `output/bed-ops/bed-residual-diagnosis-attribution-v1/`
- Residue masks and detector confidence:
  `output/bed-ops/bed-residual-diagnosis-masks-v2/`
- Complete current RGB renders:
  `output/bed-ops/bed-residual-diagnosis-renders-v1/`

`diagnose_residual_splats.py` reproduces the read-only contribution analysis;
`tests/test_residual_diagnosis.py` checks protected/mixed category separation.
