# 3D Gaussian Splatting with semantic features

This fork stores a learned semantic descriptor on every Gaussian and renders
those descriptors with the RGB image. The Jetson AGX Orin profile uses **128
floats per point** (up from 64) and builds CUDA kernels for Orin's SM 8.7 GPU.

## Jetson AGX Orin setup

This project was configured on a 64 GB AGX Orin running Jetson Linux R36.5.
Install the R36.5 CUDA development stack once (administrator access required):

```bash
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6 libcudnn9-dev-cuda-12 cmake ninja-build
```

The user-owned environment is `/home/nishantj/miniforge3/envs/gaussian-orin`.
Activate it with:

```bash
source /home/nishantj/miniforge3/etc/profile.d/conda.sh
conda activate gaussian-orin
```

PyTorch 2.8.0 and torchvision 0.23.0 must be the Jetson CUDA 12.6 wheels from
the Jetson AI Lab `jp6/cu126` index, not generic PyPI aarch64 wheels. After the
system CUDA packages are installed, build the two CUDA extensions from source:

```bash
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-feature
```

Both extension setup scripts default `TORCH_CUDA_ARCH_LIST` to `8.7`. Set that
environment variable explicitly before installation only when compiling for a
different GPU.

## 128D feature pipeline

`extract_features.py` and `apply_pca.py` now produce channel-first 128D maps
named `<image>_fmap_CxHxW.pt`, matching the dataset loader. Train with:

```bash
python train.py \
  -s /path/to/scene \
  -m output/scene-128d \
  --semantic_feature_dim 128 \
  --resolution -1 \
  --iterations 40000 \
  --position_lr_max_steps 40000 \
  --densify_until_iter 20000
```

`--resolution -1` retains the existing 1600-pixel width safety cap and is the
recommended 64 GB setting. Use `--resolution 2` for unusually large scenes or
if densification grows beyond roughly 3–5 million Gaussians. The 128D vector is
intentional: 256D fits in capacity but approximately doubles semantic rendering
traffic and CUDA per-pixel working storage, which is a poor trade on Orin.

The CUDA rasterizer accepts runtime descriptor sizes from 1 through 128, so
older/smaller experiments remain possible by passing the matching
`--semantic_feature_dim`. The source feature maps and the argument must agree.

## Memory impact

A 128D float32 descriptor costs 512 bytes per Gaussian before gradients and
Adam state. During training, semantic parameters, gradients, and two Adam
moments are roughly 2 KiB per Gaussian (about 2 GiB for one million points),
excluding geometry, RGB spherical harmonics, raster buffers, and framework
overhead. A rendered 128D float32 map costs `128 * height * width * 4` bytes;
at 960x540 that is about 253 MiB per map.

If memory pressure is high, reduce image resolution first, shorten the
densification window with `--densify_until_iter`, or use `--speedup` (which
stores one quarter of the input feature channels per Gaussian and learns a
1x1 decoder).

## Text-to-3D insertion and removal

The insertion/editing tools use the same Hugging Face CLIP ViT-B/32 model and
the scene's saved PCA projection. Do not use a different PCA file: vectors from
different PCA fits are not comparable even when both contain 128 values.

Convert a TripoSR mesh to Gaussians. Supplying the Nano Banana source image is
recommended because the stored descriptor then blends the text and image
descriptions:

```bash
python bridge.py \
  --mesh external_tools/TripoSR/output/0/mesh.obj \
  --reference-image /path/to/generated-bottle.png \
  --label "red bottle" \
  --object-id 1 \
  --pca-path data/my_scene/pca_model_128.pkl \
  --output output/assets/red-bottle.ply
```

Merge it into a completed scene. The result stores `object_id` on every point
and writes a neighboring `.objects.json` manifest:

```bash
python merge.py \
  --scene output/bottle-orin-128d-5k/point_cloud/iteration_5000/point_cloud.ply \
  --asset output/assets/red-bottle.ply \
  --output output/composites/room-with-bottle.ply \
  --object-id 1 --label "red bottle" \
  --scale 0.5 --translate 1.5 0 -2
```

Generated assets can be removed exactly by ID. For objects learned from the
photographs, use semantic text matching; start with `--dry-run`, negatives, and
optionally a bounding box before writing the pruned copy:

```bash
python remove.py \
  --input output/composites/room-with-bottle.ply \
  --output output/composites/room-without-bottle.ply \
  --object-id 1

python remove.py \
  --input output/bottle-orin-128d-5k/point_cloud/iteration_5000/point_cloud.ply \
  --output output/bottle-orin-128d-5k/point_cloud/iteration_5000/no-bottle.ply \
  --text "bottle" "plastic bottle" \
  --negative "table" "wall" "floor" \
  --pca-path data/my_scene/pca_model_128.pkl \
  --threshold 0.05 --margin 0.05 --dry-run
```

For a known object region, semantic seeds can be expanded to one connected 3D
component. This is much safer than accepting matching points throughout the
scene: add `--bbox XMIN YMIN ZMIN XMAX YMAX ZMAX --grow-component-eps 0.06`.
If multiple components exist in that box, use
`--component-anchor X Y Z` with a point known to lie on the requested object.

`remove.py` never overwrites its input. Remove `--dry-run` only after the
selected count looks plausible. Existing 128D training does not need to be
rerun for these tools; retraining is only required after changing the feature
extractor or fitting a new PCA model.

## Reconstructing a flat surface after removal

`reconstruct_flat.py` fills a missing floor, wall, or tabletop region with a
single 3D patch derived from nearby *observed* texture. It requires the original
PLY, an order-preserving pruned PLY from `remove.py`, `cameras.json`, and the
source images. It does not recover the exact surface hidden by an object.

For the bottle scene, v3 is an overbroad diagnostic cut. `--refine-mask` first
projects the removed object onto the fitted carpet plane, restores carpet
outside that footprint, and repairs the smaller region. A chosen clear donor
view is reproducible; omit `--donor-view` to score available views automatically.

```bash
python reconstruct_flat.py preview \
  --original output/bottle-orin-128d-5k/point_cloud/iteration_5000/point_cloud.ply \
  --pruned output/bottle-orin-128d-5k/semantic-validation/no-bottle-preview-v3.ply \
  --cameras output/bottle-orin-128d-5k/cameras.json \
  --images data/my_scene/images \
  --output-dir output/bottle-orin-128d-5k/semantic-validation/carpet-fill-preview \
  --views frame_0067 frame_0115 frame_0150 \
  --refine-mask --grid-step 0.004
```

The preview directory contains `candidate.ply`, `preview.json`, and rendered
images. With `--refine-mask` it also contains `refined-pruned.ply` and renders
showing the removal before filling. Review all views: a residual shadow or a
visible texture seam means the edit is not yet final. To repair a shadow
outside the object's geometric footprint, mark only that region in a grayscale
PNG from one of the `*-pruned.png` views and rerun `preview` with
`--extra-repair-mask mask.png --extra-repair-view frame_0115`. The tool
back-projects that mask to the plane, copies it into the preview directory,
and includes it in the same 3D patch. `--extra-mask-margin` expands the marked
region to cover splats whose centers lie outside their rendered footprint.
If a few oversized residual splats still visibly overlap the fill, isolate
and review them before passing their **original PLY row indices** with
`--exclude-original-index N` (repeatable). This is an explicit, auditable
exception rather than a broad automatic deletion; `preview.json` records
the excluded indices. Never use candidate PLY row numbers here.
For a visible texture boundary, `--donor-shift U V` selects a nearby
plane-aligned donor offset (both numbers must be multiples of `--cell`).
`--local-color-match` fits a low-frequency RGB correction using intact
carpet around the hole. `--seam-blend-width 0.05` adds a 5 cm feathered
overlap of new Gaussians; wider bands may double or blur the weave.
Always compare multiple rendered views before accepting a seam adjustment.
If a viewer loses the fine fill at a distance, `--distance-support-scale 0.014`
adds a sparse, larger-footprint carpet layer just behind it. The default is
off. This is a viewer-compatibility preview option, not proof that a particular
viewer will render the repair correctly; inspect both near and far views.
The tool rejects an unreliable plane or donor patch instead of silently
applying a poor fill.

For a remaining low-frequency color boundary, `refine_fill_color.py` takes an
unapproved preview, a near-normal diagnostic camera JSON, and its rendered
image. It fits a conservative RGB gradient and saves a separate preview;
geometry, opacity, and semantic features are unchanged. Review the new PLY

Only after visual approval, copy the unchanged candidate to a new final path:

```bash
python reconstruct_flat.py commit \
  --preview-dir output/bottle-orin-128d-5k/semantic-validation/carpet-fill-preview \
  --output output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed.ply \
  --approve
```

The commit step checks the preview's hash and refuses to overwrite any PLY.
Generated fill splats inherit nearby carpet semantic vectors, so the 128D
descriptor schema is preserved. Non-planar hidden geometry needs a different
reconstruction method or new source images.
