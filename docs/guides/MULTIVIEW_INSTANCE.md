# Seeded multi-view instance preview

`multiview_instance.py` is an experimental post-training pass. It takes an
existing 3D seed selection, prompts the local MobileSAM model in multiple
source photos, and admits additional nearby Gaussians only when their image
projections and depth agree across views. It writes an `object_id` label preview,
mask overlays, and a pruned scene in a **new** folder. It never edits the input
scene or approves the result. It does not fill any hidden background.

Example using the existing bed seed selection:

```bash
/home/nishantj/miniforge3/envs/gaussian-orin/bin/python -m gsedit.selection.multiview_instance \
  --scene output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply \
  --seed-indices output/bed-ops/bed-extract-v4/selected-indices.npy \
  --cameras output/bottle-orin-128d-5k/cameras.json \
  --images data/my_scene/images \
  --views frame_0135 frame_0139 frame_0141 \
  --model mobile_sam.pt --device cpu \
  --output-dir output/bed-ops/bed-instance-sam-next
```

CUDA can be used with `--device cuda` when the Jetson GPU is accessible. The
default renderer also requires CUDA; use `--skip-render` when it is not.
Review the `*-mask.png` overlays and `preview.json` before using either PLY.

**Current bed finding:** the two- and three-view trials in
`output/bed-ops/bed-instance-sam-preview-v1` and `-v2` each selected exactly the
existing 8,353 seed splats, adding zero new points. The 2D mask followed the
bedspread, but did not recover the remaining wooden frame. These trials are
diagnostic, not improved bed-removal results. The next method should account
for full Gaussian screen footprints and group separate object parts before
marking a whole-bed instance; do not loosen thresholds simply to increase the
selection count.
