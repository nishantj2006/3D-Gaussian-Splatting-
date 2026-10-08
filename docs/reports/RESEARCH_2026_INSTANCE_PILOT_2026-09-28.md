# Split&Splat native bed-instance timing pilot

## Actual results

| Run | Steps | End-to-end seconds | Recent median seconds/step | Final splats | Peak process RSS MiB | Peak Torch GPU allocation MiB |
|---|---:|---:|---:|---:|---:|---:|
| Initial pilot | 300 | 21.81 | 0.0372 | 9,612 | 1,923.53 | 139.77 |
| Extended pilot | 1,000 | 54.01 | 0.0432 | 35,577 | 2,098.68 | 226.81 |

Both runs finished successfully. No training from these pilots is still running.
This is one method's native object-reconstruction stage, **not** a completed
four-method object-removal/background comparison. The other three methods'
scene-level trials have not started. Their successful synthetic GPU checks
do not constitute an editing benchmark.

At the observed late speed, another 1,000 steps on a comparably sized object
would be about 43 seconds of training, plus loading and saving. Further
densification, higher resolution, full-scene composition, completion models
and image generation can change that substantially. Do not extrapolate this
timing to the complete four-method workflow or to the earlier 128D scene
training. There is not yet a measured end-to-end ETA for that workflow.

## What was run

- Native `external_tools/Split_and_Splat/train.py` training function, RGB/SSIM
  plus instance-mask loss, standard Adam, degree-3 capacity initialized at
  degree 0, densification enabled after step 100, stopped 50 steps before end.
- Width 270, height 480; 16 independently accepted bed-mask training views.
- Initialized from 7,557 existing baseline bed centers/colors, excluding
  protected source IDs. Native scale/opacity are reinitialized from its point
  cloud routine rather than copied from the original trained Gaussian model.
- Used existing masks and seed IDs: this does **not** reproduce the paper's
  automatic SAM2/depth propagation/instance discovery stages. It is a seeded
  instance reconstruction pilot, not independent selection evidence.
- `frame_0134`, `frame_0141`, `frame_0143` are excluded from fitting. The staged
  training split contains none of them; the native test list is empty, so no
  held-out imagery enters native training evaluation. Separate visual testing
  remains to be done.
- Native Scene's wall-clock RNG override is replaced by the fixed seed 0.
  The two runs have different densification schedules; they are not duplicate
  repeatability trials.
- Depth regularization is disabled: no reliable completed depth target was
  supplied. No missing wall/floor reconstruction was run.

The native camera initializer triggered a missing cuSolver symbol on Jetson.
Its downloaded research copy now computes the same small camera-pose inverse
on CPU before copying the center to CUDA. Rasterization and optimization remain
on the GPU. Startup attempts v1/v2 failed before any training iterations; v3
and the 1,000-step run completed. No system CUDA library or working environment
was replaced.

## Files and safety

- Staged dataset: `output/bed-ops/research-2026-split-bed-dataset-v1/`.
- Completed 300-step run: `output/bed-ops/research-2026-split-bed-pilot-v3/`.
- Completed 1,000-step run: `output/bed-ops/research-2026-split-bed-pilot-1000-v1/`.
- Live/final timing: `progress.json` and `report.json` in each successful run.
- Native object PLY: `native-model/point_cloud/iteration_1000/point_cloud.ply`.

The native PLY is an unapproved bed-only reconstruction, not a scene candidate.
Its `desc_*` schema is not our 128D semantic schema (upstream initializes
descriptors to NaN until semantic assignment). It must not be merged or mistaken
for a validated semantic model. Initial source IDs are retained separately;
densified descendants have not yet been mapped back to original scene IDs.
Sources, photos, source masks and earlier scene previews were not modified.
Upstream mask recoloring is confined to the staged copies.

Peak process RSS and Torch GPU allocation overlap on unified Orin memory.
These measurements are not power-mode-controlled research timings. Reported
recent step times exclude the first 20 iterations and use the last 50 observed
wall intervals; they include previous optimizer and current rendering work.

New tests verify pose-axis roundtrip, held-out exclusion, protected-seed
exclusion, source preservation and existing-output rejection: 7 targeted tests
passed together with the earlier startup/smoke safety tests. Visual quality,
whole-object selection and cross-view replacement quality are still unmeasured.
