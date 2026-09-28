# Autonomous generated-background preview

`run_generated_background.py` is a single entry point for the bed-replacement stages. It accepts camera poses, source photos, accepted object masks, a wall fit, and either existing replacement geometry or the inputs for `build_continuous_background.py`. It writes only to a new output directory.

The stages are:

1. Build or load surface-aligned wall/floor Gaussians. Deleted bed Gaussian positions are provenance, not replacement locations.
2. Automatically choose training views that cover the most new Gaussians, excluding held-out views and independently protected frame pixels.
3. Inpaint those views. `diffusion` uses the locally installed Diffusers inpainting pipeline; `opencv` is a fast diagnostic fallback and is **not** a learned generator. White mask pixels are repainted, while original pixels outside the bed mask are restored exactly.
4. Project generated pixels onto the shared 3D wall/floor Gaussians, fuse colors across views, and measure coverage and cross-view disagreement. The original Gaussians, new positions, PLY schema, and 128D semantic features remain unchanged.
5. Render the seed and result from held-out cameras and write inside/outside mask metrics. Nothing is promoted automatically; even passing numerical tests means `needs_visual_review`, not approved.

Example configuration keys (JSON): `seed`, `seed_report` (or a `replacement` object containing the arguments to `build_continuous_background.py`), `wall_fit`, `cameras`, `bed_manifest`, `frame_manifest`, `images`, `holdout_views`, and `generation`. The latter may specify `backend: "diffusion"`, model, resolution, seed, steps, and quality thresholds. No scene-specific coordinates or colors are embedded in the algorithm. Run:

```bash
conda activate gaussian-orin
python -m gsedit.pipelines.run_generated_background --config /path/to/config.json --output-dir /path/to/new-preview
```

The local environment now has `diffusers==0.35.2`, `accelerate==1.15.0`, and `importlib-metadata`; `pip check` is clean. Diffusion weights are **not** cached and have **not** been downloaded or run. The default model is `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`. Its first use downloads weights and runs locally, without a paid image API. The official [Diffusers inpainting guide](https://huggingface.co/docs/diffusers/main/using-diffusers/inpaint) describes this model and mask behavior.

## Current bed-scene result

The existing bed scene fails the default geometry safety gate: the wall fit has 16.8% inliers, below the 30% minimum; the previous wall/carpet texture donors are also too distant. An explicit weak-fit override was used **only** for an OpenCV integration diagnostic at `output/bed-ops/bed-autonomous-opencv-diagnostic-v1`. It textured 12,452 of 14,759 new Gaussians (84.4%), had 8,376 Gaussians seen in multiple generated views, took 49.2 s end to end, and used 1.47 GiB peak process RAM in the generation stage. The held-out `frame_0143` render still has an obvious smeared wall/bed boundary; the workflow correctly reports `automatic_checks_failed` and `approved: false`.

Do not spend compute on diffusion for this capture until geometry support improves or a user explicitly requests an experimental visual-only preview. Generated color alone cannot fix a wrong wall/floor plane or incomplete removal.
