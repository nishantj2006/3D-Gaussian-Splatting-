# Latest research trial results: appearance and geometry-prior addendum

This report follows `RESEARCH_2026_CAPTURE_COMPONENTS_2026-09-28.md`. Its
later checkpoint/operator results supersede the setup status in that snapshot.
**The completed trials are components, not complete reproductions or a validated
3D background repair.** Original scenes and earlier previews remain unchanged.

## Completed appearance trials

Both models ran on the same three non-held-out source photographs:
`frame_0133`, `frame_0136`, `frame_0144`. Input size is 288 x 512. Bed and wooden
frame masks are combined only for the requested edit footprint; their original
instance evidence remains separate. Masks have an 8-pixel generation margin and
exclude accepted dresser-protection pixels. No held-out photo is a generation
input. Output pixels outside the generation mask are copied exactly from the
input. The source photographs still show the bottle; these are photo diagnostics,
not edited renders of the previously bottle-removed scene.

| Appearance component | Model loading seconds | Total seconds, loading and 3 images | Per-image seconds | Peak process RSS GiB | Peak Torch CUDA allocation MiB |
|---|---:|---:|---|---:|---:|
| LaMa native FFC generator, inference-only adapter | 10.36 | 12.45 | 0.507 / 0.257 / 0.241 | 1.839 | 331.88 |
| CoIn's documented SD2 inpainting checkpoint, diffusers/DDIM adapter, 20 steps | 11.38 | 21.51 | 3.825 / 2.647 / 2.623 | 7.908 | 2832.05 |

These short probes overlapped briefly during startup; they are measured pilot
costs, not controlled comparative efficiency rankings. The initial standalone
SD2 probe took 4.212 seconds for one image and 16.28 seconds including loading,
with 8.064 GiB peak process RSS. Unified RAM and Torch GPU allocations overlap;
do not add them. Image dimensions, sampler and iteration schedules differ from
the papers' full workflows.

Visual inspection at `frame_0136`:

- LaMa plausibly extends the carpet, but the upper repair has dark/blue softness
  and visible residue-like texture. It is not an accepted bed-free replacement.
- SD2 generates a cleaner-looking empty area, but invents bars and circular
  details not supported by the capture. The other generated image inspected at
  `frame_0133` looks more plausible; there is no demonstrated cross-view agreement.
- Exact outside-mask compositing is useful, but does not prove consistency inside
  the mask, correct wall depth, preservation of shadows or seamless boundaries.

No generated image was converted into an approved scene or used to override the
wall geometry gate. This test does not run CoIn's FreeDoM/3D guidance, reference
attention, texture lifting, adversarial refinement or full Scaffold scene edit.
It does not run Inpaint360GS's complete virtual-view/depth fusion/3D optimizer.

Artifacts:

- `output/bed-ops/research-2026-lama-2d-v4/report.json`
- `output/bed-ops/research-2026-lama-2d-v4/frame_0136-generated-2d-only.png`
- `output/bed-ops/research-2026-coin-sd2-2d-threeviews-v2/report.json`
- `output/bed-ops/research-2026-coin-sd2-2d-threeviews-v2/frame_0136-generated-2d-only.png`

Earlier `*-v1/v2/v3` attempts remain intact. They record configuration-resolution,
restricted checkpoint metadata and incomplete-protection-mask failures, including
a completed first image before a three-view attempt stopped. The runner now
preflights every mask before loading a model. Its default chooses three eligible
views automatically; an explicit list must have accepted independent target and
protection evidence. The corrected three-view benchmark used an explicit common
view list only for comparison, not a hardcoded object/coordinate rule.

## Completed GPGS pretrained geometry-prior probe

The native Point-MAE model strictly loaded the pretrained checkpoint linked by
the official Point-MAE repository. Chamfer and PointNet2 were compiled for ARM64
and SM 8.7, then installed only in the research environment. PointNet2 required
one build-system change: respect the externally supplied architecture list rather
than overwrite it with obsolete architectures.

The referenced KNN_CUDA repository returns 404. The probe injects an explicitly
documented PyTorch `cdist`/`topk` index operator. Unit tests verify indices and both
tensor layouts. Native Point-MAE consumes only the returned indices; no equivalence
claim is made for the unavailable package's distance-unit convention or tied-index
ordering.

Input: 1,024 points from an intact local carpet patch, excluding existing source
deletion IDs, near the validated existing floor plane. Twenty-five of sixty-four
groups are artificially masked. The model predicts 800 points. This is not a
prediction of the wall behind the bed: masked-group centers come from the known
context, so they already constrain where reconstruction happens.

- Total probe time including loading/output: **12.05 seconds**.
- Median of four warmed forward passes: **0.04617 seconds**.
- Peak process RSS: **2.040 GiB**.
- Peak Torch CUDA allocation: **135.68 MiB**.
- Median predicted-point distance to the known plane: **0.007435 scene units**.
- 95th percentile distance: **0.009539 scene units**.

These are distances in the capture's coordinate units, not calibrated meters.
They measure proximity to the existing plane, not exact hidden-surface error.
No scene-specific completion fine-tuning, occlusion-aware hidden wall completion,
held-out depth reprojection or Gaussian insertion followed this probe. A flat
plane model may already be more appropriate than a learned prior for this floor.
The current 19.58% wall fit remains unsupported.

Artifacts:

- `output/bed-ops/research-2026-pointmae-prior-v2/report.json`
- `output/bed-ops/research-2026-pointmae-prior-v2/predicted-points-diagnostic.ply`
  (plain XYZ diagnostic points, **not Gaussian splats**).
- `output/bed-ops/research-2026-pointmae-prior-v2/context-points.npy`
- `output/bed-ops/research-2026-gpgs-completion-build-v1/chamfer-2.0.0-cp310-cp310-linux_aarch64.whl`
- `output/bed-ops/research-2026-gpgs-completion-build-v2/pointnet2_ops-3.0.0-cp310-cp310-linux_aarch64.whl`

## Checkpoints, imports and environment safety

All installs/builds target `external_tools/environments/research-2026-base/`.
The working `gaussian-orin` environment and its rasterizer are not replaced.

- Point-MAE checkpoint SHA256:
  `27ded932bb0a2625d5a8eb006df199b2578598c774aee6d86b985300b6a5fd20`.
- LaMa model source: official README's `smartywu/big-lama` link. The bundled
  Inpaint360GS LaMa checkout lacks `training/data`, so this probe uses the native
  generator from the complete official LaMa source, commit
  `786f5936b27fb3dacd2b1ad799e4de968ea697e7`.
- LaMa legacy metadata is loaded using `weights_only=True`, an explicit global
  allowlist, and inert metadata record aliases. No unrestricted pickle loading
  or execution of checkpoint trainer/config classes is enabled. Only generator
  tensors are returned; model state loading is strict. Tests reject unknown
  globals before deserialization.
- SD2 source: the checkpoint specified by CoIn's official README,
  `sd2-community/stable-diffusion-2-inpainting`.
- FeatUp's ARM64 CUDA wheel built and was installed in the research environment.
  CoIn's full guidance additionally imports PyTorch3D, which has not been ported
  or installed. A working base diffusion pipeline is not equivalent to that
  full guidance stage.

The locked views are excluded from **these new refinement/component trials**.
The pre-existing 5k source scene may already have been trained on those photos;
these are edit-validation holdouts, not a claim of capture-level unseen-data
generalization.

## What to use next — conclusions, not claims of completion

1. Use Inpaint-style rendered instance features as additional attribution evidence
   with independently checked dresser-negative evidence and explicit protected
   IDs. Its 10,334-splat protected edit improves some front-coverage/collateral
   metrics but still exposes recognizable bed layers. Shared footprint splitting
   and revealed-view attribution remain necessary.
2. Use LaMa or diffusion **inside a shared surface atlas** with observed texels
   fixed. These measurements establish affordable appearance generation, not
   permission to fit Gaussians to inconsistent independently generated photos.
   LaMa is a useful cheap texture proposal; diffusion needs checks against invented
   structure. A 2D image may suggest appearance, but must not establish wall depth.
3. Use the working depth-capable renderer and local RGB/depth optimization on
   validated geometry. Point-MAE can now be evaluated as a constrained refinement
   prior; this probe does not justify using it to hallucinate a confident wall.
4. Keep the existing 5k/128D scene and exact retained records. Do not adopt CoIn's
   anchor representation solely because its short RGB reconstruction loss is
   lower; a materialization/provenance adapter and actual end-to-end benefit are
   still required.

No full-method winner can be ranked yet for whole-bed removal or background
quality. No replacement has been promoted. Remaining work includes revealed-layer
attribution, validated hidden depth, shared texture fusion and local RGB/depth
optimization, followed by the required held-out/overhead visual review.
