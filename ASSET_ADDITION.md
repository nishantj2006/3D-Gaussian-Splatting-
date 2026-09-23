# Add a generated asset to a Gaussian scene

`add_asset.py` joins four stages: Nano Banana or local SDXL image generation, TripoSR
single-image mesh generation, mesh-surface Gaussian conversion, and scene
placement/merge. The scene PLY is never overwritten. Each run needs a new
`--output-dir`, and writes intermediate image/mesh/PLY files plus `pipeline.json`.

Use `gaussian-orin` for the pipeline and scene conversion. For Nano Banana,
install `google-genai` in that conda environment and set `GEMINI_API_KEY` in
the shell (do not put the key in a command or source file). Image generation
uses `gemini-3.1-flash-image` by default, configurable with `--nano-model`.
The default `--image-provider nano` keeps the previous behavior.

For an API-free text-to-3D run, select `--image-provider local-sdxl`. The
optional `generate_local_image.py` stage uses SDXL through Diffusers on CUDA,
then passes its PNG to the same TripoSR/semantic/merge stages. An isolated
conda clone is ready at `output/setup/sdxl-orin`, with Jetson CUDA PyTorch,
Diffusers 0.35.2, Accelerate 1.10.1, Transformers and Safetensors. The
`gaussian-orin` training environment was not changed. SDXL's fp16 weights
are cached at `output/models/sdxl-base-1.0-fp16` on the external project
drive. Use `--image-python` and `--local-model` below to avoid another
download to the system drive. Generation uses 1024x1024, 25 steps, and
`--seed` by default; override with `--local-width`, `--local-height`,
`--local-steps`, or `--local-model`.

TripoSR is the project's pinned Git submodule at `external_tools/TripoSR`.
Its pinned dependencies conflict with the scene environment; use a separate
`triposr-orin` environment and pass its Python with `--tripo-python`. The
TripoSR model weights are downloaded on first use. On Jetson AGX Orin, the
upstream `torchmcubes` CUDA extension may require an ARM/CUDA compatibility
patch; a missing extension means the image-to-mesh stage is not ready. The
`--mesh` option bypasses that stage and remains fully usable.

To replace the removed bottle's approximate location and size, use its
original/pruned pair as the target box:

```bash
conda activate gaussian-orin
python add_asset.py \
  --prompt "red glass bottle" \
  --scene output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply \
  --plane-json output/bottle-orin-128d-5k/semantic-validation/carpet-fill-preview-distance-support/preview.json \
  --replace-original output/bottle-orin-128d-5k/point_cloud/iteration_5000/point_cloud.ply \
  --replace-pruned output/bottle-orin-128d-5k/semantic-validation/carpet-fill-preview-distance-support/refined-pruned.ply \
  --tripo-python /home/nishantj/miniforge3/envs/triposr-orin/bin/python \
  --object-id 41 --points 60000 \
  --output-dir output/asset-additions/red-glass-bottle-01
```

For the blue bottle using local SDXL instead of the Gemini API:

```bash
conda activate gaussian-orin
python add_asset.py \
  --image-provider local-sdxl \
  --image-python output/setup/sdxl-orin/bin/python \
  --local-model output/models/sdxl-base-1.0-fp16 \
  --prompt "a single upright blue reusable water bottle" \
  --scene output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply \
  --plane-json output/bottle-orin-128d-5k/semantic-validation/carpet-fill-preview-distance-support/preview.json \
  --replace-original output/bottle-orin-128d-5k/point_cloud/iteration_5000/point_cloud.ply \
  --replace-pruned output/bottle-orin-128d-5k/semantic-validation/carpet-fill-preview-distance-support/refined-pruned.ply \
  --tripo-python /home/nishantj/miniforge3/envs/triposr-orin/bin/python \
  --object-id 41 --points 60000 \
  --output-dir output/asset-additions/blue-water-bottle-local-01
```

Use a new output directory for each attempt. The failed Gemini API run already
created `blue-water-bottle-01`, so that name cannot be reused. Local image
generation can be slow on Jetson; run it separately from active training to
avoid GPU memory pressure. The SDXL model has been downloaded but no image
has been generated yet. Inspect the resulting image and aligned asset before
treating the merged PLY as final.

To resume without API access or TripoSR, provide `--image` and/or `--mesh`.
`--mesh` alone skips both external stages but still creates a CLIP/PCA semantic
descriptor from the prompt. The output's `object_id` allows exact later
removal with `remove.py --object-id 41`.

Other target options are `--target-bbox MIN_X MIN_Y MIN_Z MAX_X MAX_Y MAX_Z`
in world coordinates, or `--target-center X Y Z --target-size WIDTH DEPTH
HEIGHT`, with sizes along the fitted plane. The placement uses uniform scale,
an optional `--yaw-deg` rotation (otherwise 0 or 90 degrees is chosen), and a
small floor clearance. It rotates Gaussian covariance as well as center
positions. It does not infer artistic orientation, physical units, lighting,
or collision-free placement from a prompt alone; inspect the final PLY in
SuperSplat and adjust the target/yaw if needed. View-dependent nonzero SH
assets are rejected when rotated, since SH coefficients are not yet rotated.

The source asset's mesh vertex colors become Gaussian DC color. TripoSR's
default OBJ uses vertex colors, not a baked texture atlas. `bridge.py` samples
the mesh surface deterministically (`--points`, `--seed`) and assigns all of
one generated object the same scene-compatible 128D text/image descriptor.
Those semantics are for selecting the asset as a whole, not segmenting its
individual parts.
