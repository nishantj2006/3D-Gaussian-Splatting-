# Real-capture component trials: partial benchmark, not a solved edit

The Orin had approximately 50 GiB RAM available before these trials. No other
research training job was active. The trials use an isolated conda clone,
method-specific rasterizers and new output folders. No source or prior preview
was overwritten. **No method has completed end-to-end background reconstruction
or earned visual approval in this benchmark.**

## What actually ran

All images in these trials are 270 x 480. The held-out views are `frame_0134`,
`frame_0141` and `frame_0143`; none contributes training labels, images or fitting.
The component adapter uses 13 accepted training views with both independent
bed and wooden-frame masks. Missing-frame views are excluded. Overlapping class
masks have ignored labels rather than arbitrary ownership.

| Component | Steps | Total seconds including setup/output | Training seconds | Median step seconds | Peak process RSS GiB | Peak Torch GPU allocation MiB |
|---|---:|---:|---:|---:|---:|---:|
| Inpaint360GS native 16D feature/classifier distillation, adapted | 1,000 | 132.02 | 119.76 | 0.11750 | 4.020 | 741.91 |
| GPGS native RGB/depth renderer and reconstruction model, adapted loop | 300 | 19.70 | 11.59 | 0.03224 | 2.163 | 108.58 |
| CoIn native anchor/MLP RGB reconstruction model, adapted loop | 300 | 28.40 | 21.28 | 0.06582 | 1.851 | 159.80 |

The latter two initialize from the same seeded 20,000-point source sample and
do not densify. They fit the **original photographs containing the bed**, so
their RGB errors are reconstruction diagnostics, not removal or hidden-background
scores. Inpaint360GS uses all 593,814 source splats but freezes geometry and SH
appearance. It learns separate new 16D instance features and a three-class
classifier; the original 128D semantic fields are not retrained.

The adapted Inpaint loop uses class-balanced pixel cross-entropy, ignores
overlapping labels, and omits upstream 3D feature regularization. CoIn/GPGS use
their native models and differentiable renderers in a bounded RGB optimization
loop, not their full paper pipelines. The native scene-loader/GPU inverse is
bypassed with equivalent camera matrices computed on CPU to avoid the existing
Jetson cuSolver library incompatibility. These adaptations must be retained when
interpreting results; this is not a claim of official-method reproduction.

An initial GPGS run overlapped the 100-step Inpaint probe and took 32.69 seconds;
it is superseded for timing by the serialized 19.70-second run. Source sampling
is repeatable, but native CUDA optimization is not bitwise deterministic: the
repeated GPGS runs have slightly different loss/RGB values. No strict inference
determinism claim is made. On the Jetson, Torch GPU allocation and process RSS
overlap in physical unified memory; do not add them together. Context/driver RAM
is not fully represented by Torch allocation.

## Inpaint360GS selection and actual removal

The 100-step probe learned a bed image mask but selected **zero** source splats
at probability > 0.7. The 1,000-step trial selected 10,334 splats after excluding
protected source IDs. Its raw selection also included **11,686 protected source
IDs**, which were blocked. Therefore the instance model alone is not safe for
deletion; protection is doing important work, and these protected IDs/masks are
not independently verified dresser ground truth.

Held-out image classification IoU against existing unverified masks:

| View | Bed IoU | Frame IoU |
|---|---:|---:|
| frame_0134 | 0.88610 | 0.70599 |
| frame_0141 | 0.86385 | 0.46773 |
| frame_0143 | 0.91406 | 0.50446 |

The comparison uses the **working scene renderer** at 540 x 960, comparing source,
the current calibrated split preview, and the new instance-feature removal:

| View | Trial deleted-front recall | Trial protected pixels visibly changed | Current protected pixels visibly changed | Trial pixels changed outside bed/protection | Current pixels changed outside bed/protection |
|---|---:|---:|---:|---:|---:|
| frame_0134 | 93.61% | 0.000% | 0.000% | 0.792% | 0.771% |
| frame_0141 | 99.48% | 0.000% | 0.000% | 1.502% | 1.619% |
| frame_0143 | 99.52% | 0.435% | 0.749% | 1.053% | 0.993% |

“Visibly changed” means maximum channel difference > 0.08 on normalized RGB.
Deleted-front recall measures the rendered footprint of removed Gaussians, not
whole-object removal. The native removal render still contains a recognizable
bed-like hazy/blue structure: removing front contributors exposes other layers.
The new result is **not approved**. Pixel scores do not establish a sound wall,
prove every bed layer removed, or prove exact dresser preservation.

The candidate retains all 190 properties and all 128 semantic dimensions. A
separate disk-read comparison confirms every retained vertex record exactly
matches the source. Protected source IDs removed: zero. Broad retained splats
can still change the appearance of a protected area through occlusion, which is
why pixel-level checks are reported separately.

## Native reconstruction diagnostics — not inpainting quality

| View | GPGS RGB L1 | CoIn RGB L1 |
|---|---:|---:|
| frame_0134 | 0.06212 | 0.04671 |
| frame_0141 | 0.08957 | 0.05372 |
| frame_0143 | 0.08808 | 0.05609 |

CoIn's small anchor model fits these photographs better after this particular
short schedule. This does **not** establish better removal, completion, or final
quality. Both native outputs lack our full 128D scene schema and must not be
merged or substituted as approved scenes.

## Remaining end-to-end prerequisites

- Inpaint360GS: the learned-instance component is now exercised. Its LaMa RGB/
  depth checkpoint, virtual-view inputs, 3D fusion and local inpainting stages
  are still unrun. Original mask coverage and revealed-object layers remain
  weaknesses that these component trials do not repair.
- GPGS: the Chamfer extension built successfully for ARM64/SM 8.7. The official
  PointNet2 build failed because its setup overrides the architecture list with
  obsolete SM 3.7; a one-line patch changes assignment to `setdefault` so our
  explicit SM 8.7 is honored, and a rebuild was started. The referenced
  `unlimblue/KNN_CUDA` repository returns 404, so an audited alternative operator
  is needed. Point-MAE pretrained weights, scene-specific completion training,
  projected image refinement and composition remain unrun. No completed depth
  or geometry estimate has been produced.
- CoIn: native reconstruction works. Stable Diffusion 2 inpainting and Depth
  Anything checkpoint setup, view-guidance/attention/adversarial stages,
  end-to-end removal, and a documented 128D/provenance adapter remain unrun.

The official GPGS setup and checkpoint requirements are documented at
https://github.com/yongjoon99/GPGS and
https://github.com/Pang-Yatian/Point-MAE . The native checkout READMEs specify
the corresponding Inpaint360GS and CoIn requirements.

## Decision supported by these measurements

The measured components are affordable on this Orin; RAM is not their immediate
constraint. Inpaint-style rendered instance-feature supervision is usable as
**additional selection evidence**, but not as a stand-alone safe deletion rule.
Retain explicit furniture protection and refine/re-attribute revealed layers.
GPGS's depth-capable renderer is usable; its geometry-completion benefit is not
yet measured. CoIn is a feasible reconstruction component but requires replacing
our scene representation, and no editing advantage has yet been established.

These results do not resolve the existing 19.58% wall-fit inlier problem or
override the geometry confidence gate. No sound 3D background repair can be
claimed from the current trials. More iterations of a model fitting original
bed photographs are not an appropriate substitute for completing the hidden
background stages.

## Artifacts and tools

- `output/bed-ops/research-2026-inpaint-component-1000-v1/report.json`
- `output/bed-ops/research-2026-inpaint-component-1000-v1/candidate-unapproved.ply`
- `output/bed-ops/research-2026-inpaint-comparison-v1/report.json`
- `output/bed-ops/research-2026-inpaint-comparison-v1/comparison.png`
- `output/bed-ops/research-2026-inpaint-comparison-v1/overhead-2.png`
- `output/bed-ops/research-2026-gpgs-component-serial-v1/report.json`
- `output/bed-ops/research-2026-coin-component-v1/report.json`
- `output/bed-ops/research-2026-gpgs-completion-build-v1/`
- `output/bed-ops/research-2026-pointnet-build-v2.log`

New commands: `python -m gsedit.evaluation.benchmark_research_components --help`
and `python -m gsedit.evaluation.compare_instance_trial --help`. Both refuse
existing output folders. Regression suite: **131 passed**.
