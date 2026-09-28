# Automatic text-to-removal preview

`auto_remove_preview.py` accepts a text label and uses CLIP to choose source
views, MobileSAM to propose independent image masks, camera/depth projection to
find coherent 3D seeds, and a color distribution learned from those seeds to
grow the selection locally. There are no named-color choices, fixed camera
frames, bed-specific synonyms, or scene-specific negative prompts in this
workflow. Generic safety thresholds are CLI parameters. It **never** approves
or overwrites the source scene and does **not** reconstruct hidden surfaces.

```bash
/home/nishantj/miniforge3/envs/gaussian-orin/bin/python -m gsedit.pipelines.auto_remove_preview \
  --scene output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply \
  --text bed \
  --cameras output/bottle-orin-128d-5k/cameras.json \
  --images data/my_scene/images \
  --pca-path data/my_scene/pca_model_128.pkl \
  --sam-model mobile_sam.pt \
  --device cpu \
  --output-dir output/bed-ops/bed-auto-next
```

The final preview is `appearance-expanded/pruned-preview.ply`. Inspect the
`image-seeds/*-mask-overlay.png` images, `summary.json`, and the PLY from
several views before use. Use `--device cuda` only when the Jetson GPU is
accessible. Positive-vector-only selection is measured but its PLY is not
written if it exceeds the safe scene-fraction limit.

## Current bed trial

The one-command run in `output/bed-ops/bed-auto-one-command-v1` took 248.05 s
on CPU and reproducibly chose frames 0143 and 0134. It found 5,601 image/3D
seeds and added 727 nearby splats by learned appearance: **6,328 total**.
All 128 semantic properties and the PLY schema were preserved. The source
scene is unchanged. Against the earlier, incomplete 8,353-splat bed edit,
6,032 selected splats overlap and 296 lie outside; this is an agreement proxy,
not an accuracy score.

The result mainly follows the **bedspread**. It does not independently detect
the wooden bed frame or reconstruct wall/floor hidden behind the bed. A bare
positive-vector-only query chose 518,082 of 593,814 splats (87.2%) and was
correctly refused as a removal. For comparison, a separately configured
vector-only baseline with explicit negative prompts selected 4,541 splats and
is saved at `output/bed-ops/bed-vector-only-preview-v1/scene-pruned.ply`.
Neither candidate is approved as a complete bed removal.
