# Place assets on a detected bed; preview bed extraction

`surface_pipeline.py` uses the scene's 128D semantic features, blue bedding
observed in source photos, camera poses, and a fitted floor plane. It chooses
bed-facing camera views automatically unless `--views` is provided. It then
estimates the observed bed top and places an existing Gaussian asset there.
This is a code-driven, scene-specific preview workflow, **not** a general
natural-language planner. It currently supports the center of the observed
bed, not instructions such as "near the pillow" or reliable placement on
unseen surfaces.

All commands require `gaussian-orin` on the Jetson. Every output folder must
be new. The original scene PLY is never overwritten.

## Place the existing basketball on the bed

```bash
/home/nishantj/miniforge3/envs/gaussian-orin/bin/python -m gsedit.pipelines.surface_pipeline place \
  --scene output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply \
  --asset output/asset-additions/basketball-local-01/asset-raw.ply \
  --asset-label basketball --object-id 43 --diameter 0.8 \
  --surface bed --color blue \
  --cameras output/bottle-orin-128d-5k/cameras.json \
  --images data/my_scene/images \
  --ground-plane output/bottle-orin-128d-5k/semantic-validation/carpet-fill-preview-distance-support/preview.json \
  --output-dir output/bed-ops/ball-on-bed-next
```

The command saves a selected-surface PLY, aligned asset, new merged scene,
multiview renders, and `preview.json`. Change `--diameter` to resize the ball.
Use `--views frame_0135 frame_0141` to choose views explicitly. A different
object or colored surface may require different positive/negative text, color,
or source views. The method rejects sparse or overly broad selections; inspect
renders even when it succeeds.

## Extract the bed for removal

```bash
/home/nishantj/miniforge3/envs/gaussian-orin/bin/python -m gsedit.pipelines.surface_pipeline extract \
  --scene output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply \
  --surface bed --color blue --grow-radius 0.4 \
  --cameras output/bottle-orin-128d-5k/cameras.json \
  --images data/my_scene/images \
  --views frame_0135 frame_0141 \
  --ground-plane output/bottle-orin-128d-5k/semantic-validation/carpet-fill-preview-distance-support/preview.json \
  --output-dir output/bed-ops/bed-extract-next
```

`extracted-object.ply` is the conservative bed splat asset.
`residual-cleanup.ply` contains broad, dark splats whose screen footprints
overlap the bed but are too ambiguous to include in the clean asset.
`pruned-preview.ply` removes both; review it from several angles. The source
PLY and earlier previews remain intact.

## Experimental hidden wall and carpet

The bed hides large areas in the photos. The following is *only a speculative
3D extrapolation* using nearby carpet and visible wall colors. It is not a
restoration of observed geometry:

```bash
/home/nishantj/miniforge3/envs/gaussian-orin/bin/python -m gsedit.reconstruction.background_preview \
  --scene output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply \
  --pruned output/bed-ops/bed-extract-v4/pruned-preview.ply \
  --bed-core output/bed-ops/bed-extract-v4/extracted-object.ply \
  --ground-plane output/bottle-orin-128d-5k/semantic-validation/carpet-fill-preview-distance-support/preview.json \
  --cameras output/bottle-orin-128d-5k/cameras.json \
  --images data/my_scene/images --views frame_0135 frame_0141 \
  --allow-extrapolation --output-dir output/bed-ops/bed-background-next
```

Without the explicit `--allow-extrapolation`, the command rejects the hidden
surface fill. It writes a new `candidate.ply`, diagnostics and renders, never
an approved scene. The current bed test shows visible wall seams and residual
smearing; it is not suitable as a final edited asset.
