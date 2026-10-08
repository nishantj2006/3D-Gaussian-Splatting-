# 2026 methods on Orin: partial setup trials

## Scope and decision

All four public implementations were downloaded, and all four CUDA rasterizers
were compiled and exercised on this Orin with synthetic forward/backward tests.
**No method has yet completed a bed-removal or background-reconstruction run.**
These results are compatibility evidence, not a visual-quality benchmark and
not an end-to-end reproduction of any paper. There is no new edited scene PLY.

The unrelated training workload initially left about 11 GiB available. After
the user freed compute, available memory rose to approximately 50 GiB. We did
not pause, stop, or modify that workload ourselves.

Before full trials, resolve whether separate object-aware/Scaffold scene
reconstruction is allowed. Earlier experiments explicitly avoided full scene
retraining; Split&Splat and CoIn are not drop-in edits to our existing PLY.
Inpaint360GS needs learned instance features; GPGS needs a geometry-completion
stack and reference targets. Their exporters also need provenance/semantic
adapters before any result can meet our preservation requirements.

## Actual revisions and GPU checks

| Implementation | Git revision | Synthetic GPU forward/backward |
|---|---|---|
| Split&Splat | `9b7ba906511f75b87c438b4cf30ae3d1d5302f61` | Passed |
| Inpaint360GS | `d54c893285c6cb27788e05cce607e7d3cca6388a` | Passed after SM 8.7 rebuild |
| GPGS | `c166832667d9478493b3021e6f0ef10a52036d28` | Passed |
| CoIn | `a87cffde5b6b4d437a67e5e7a573b4aa80bce1e1` | Passed |

Repositories are under `external_tools/<implementation>/`.
An isolated conda environment was cloned from the working environment:
`external_tools/environments/research-2026-base/`.
Torch remains 2.8.0, CUDA 12.6, NumPy 1.26.4. The original `gaussian-orin`
environment was checked afterward: Open3D and wandb are still absent there;
they and other bootstrap dependencies were installed only in the clone.

Each extension is installed into its own `external_tools/research-2026-bindings/`
directory. Three upstream projects use the same Python extension name but
different argument/return contracts; never install them over the working
editing rasterizer. Build wheels are saved separately under
`output/bed-ops/research-2026-build-v1/`.

Inpaint360GS originally hardcoded `compute_86,sm_86`. Its tiny render failed
with a PyTorch CUDA allocator/NVML assertion, including with synchronous CUDA
and debug checks. Removing just that hardcoded flag and rebuilding with
`TORCH_CUDA_ARCH_LIST=8.7` yielded a successful render and backward pass.
This is evidence that the targeted rebuild resolves this test; it is not proof
that every upstream kernel or full training stage is Jetson-compatible.
The failed build is retained; the corrected wheel is under
`research-2026-build-v2/Inpaint360GS/`, installed into
`external_tools/research-2026-bindings/Inpaint360GS-sm87/`.

## Measured synthetic checks — not scene costs

Two Gaussians, 32x32 RGB, fixed seed, one forward/backward operation:

| Renderer | Test-body seconds | Peak process RSS MiB | Peak Torch CUDA allocation MiB |
|---|---:|---:|---:|
| Split&Splat | 0.665 | 671.50 | 0.100 |
| Inpaint360GS, corrected build | 0.973 | 671.79 | 0.196 |
| GPGS | 0.789 | 674.52 | 0.178 |
| CoIn | 3.210 | 720.36 | 0.093 |

These values include different startup/JIT effects and cannot estimate full
scene runtime or rank the methods' efficiency. CUDA context/driver memory is
not represented by Torch allocation alone. Unified-memory measurements overlap;
do not add process RSS and CUDA allocation as independent physical RAM.

Machine-readable successful checks:

- `output/bed-ops/research-2026-gpu-smoke-v1/Split_and_Splat.json`
- `output/bed-ops/research-2026-gpu-smoke-v3/Inpaint360GS.json`
- `output/bed-ops/research-2026-gpu-smoke-v1/GPGS.json`
- `output/bed-ops/research-2026-gpu-smoke-v1/CoIn.json`

The failed Inpaint360GS debug snapshot is retained only in
`output/bed-ops/research-2026-gpu-debug-v1/`.

## Startup and remaining scene integration

The bounded `--help` startup audit is recorded in
`output/bed-ops/research-2026-startup-v6/report.json` with per-method logs,
commit IDs, exit status, elapsed time and process RSS. Split&Splat, GPGS and
Inpaint360GS pass at this stage. CoIn requires further dependency setup; the
v6 log reports missing `torch_scatter`, for which an ARM64 source build was
initiated. Subsequent startup audits, if present, supersede v6.

- **Split&Splat:** cross-view masks followed by separate instance reconstruction
  and composition. Its PLY loader expects `id_0..2` and `desc_*`. Our `semantic_*`
  fields are not that interface. Needs an instance dataset, depth propagation,
  object-wise training and a feature/provenance adapter, not a naive rename of
  feature fields or injection of unverified object IDs.
- **Inpaint360GS:** uses 16D object features and a trained classifier/instance
  association stage, then removal, virtual views, LaMa color/depth inpainting,
  and local optimization (released common config specifies 5,000 iterations).
  Its PLY loader silently initializes absent object features to zero. Loading
  our PLY therefore does not make object selection valid. HQ-SAM/CropFormer,
  LaMa weights and associated masks/classifier state remain to be prepared.
- **GPGS:** expects a `filter_3D` field, additional appearance state and its
  depth-aware rasterizer. Requires Point-MAE weights and PointNet2/KNN/Chamfer
  compatibility work. Released scripts include 50 completion-training epochs,
  image-refinement training and a 3,000-iteration composition stage. These are
  defaults, not a measured Orin time estimate. Completed images and unseen masks
  must not include held-out views during fitting.
- **CoIn:** uses Scaffold-style anchors, offsets and learned MLPs rather than
  our explicit SH Gaussians. Its released configs include 20k/10k/30k training
  stages and diffusion-guided inpainting. It needs its own reconstruction and a
  documented materialization/semantic transfer strategy, not direct loading of
  the 128D scene as an equivalent model. Further dependencies and checkpoints
  remain beyond the successful rasterizer test.

No standalone segmentation/completion/inpainting checkpoints or generated
images were downloaded/created. The Inpaint360GS Git checkout does include
small bundled LPIPS metric weights; bootstrap libraries are otherwise setup.

## Scene-level acceptance still pending

The source remains
`output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply`
(593,814 Gaussians, 190 properties, 128 semantic fields).
The previous preview remains
`output/bed-ops/bed-shared-split-calibrated-v3/candidate.ply`.
Existing photos, masks and camera files are unchanged.

A fair next comparison must hold out `frame_0134`, `frame_0141`, `frame_0143`,
preserve independent bed/frame/dresser evidence, and review actual renders for
residue, holes, seams and collateral furniture changes. Whole-object removal,
background quality, held-out consistency, per-stage RAM/runtime and exact
preservation of unrelated splats have **not** been measured for these methods.
The current wall's 19.58% inlier fit remains unsupported; compatibility tests
do not override that gate or approve generated depth.

New tools:

- `python -m gsedit.evaluation.probe_2026_methods --help`
- `python -m gsedit.evaluation.smoke_research_rasterizer --help`

The audit refuses existing output directories; the smoke check refuses existing
reports. Regression suite after setup: **125 passed**. Full method inference,
scene-level repeatability and visual approval remain pending.
