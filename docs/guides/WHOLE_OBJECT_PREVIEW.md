# Multi-part whole-object preview

`whole_object_preview.py` starts from an existing text/image seed selection,
keeps independent MobileSAM proposals across several camera views, associates
recurring proposals in 3D, and assigns separate `part_id` values in a preview
PLY. The requested whole-object deletion is a temporary union of accepted
part IDs; the source scene is never overwritten.

Example:

```bash
/home/nishantj/miniforge3/envs/gaussian-orin/bin/python -m gsedit.selection.whole_object_preview \
  --scene output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply \
  --seed-indices output/bed-ops/bed-auto-one-command-v1/appearance-expanded/selected-indices.npy \
  --text bed \
  --cameras output/bottle-orin-128d-5k/cameras.json \
  --images data/my_scene/images \
  --sam-model mobile_sam.pt --device cpu \
  --output-dir output/bed-ops/bed-whole-parts-next
```

Review `preview.json`, `*-parts.png`, `part-labeled-preview.ply`, and
`pruned-preview.ply` before accepting any result. Candidate masks are scored
against the user-provided text, with no scene-specific color or part names.
Projected Gaussian size is approximated and bounded; this is **not** a full
rendered-contribution analysis. A proposed part must have cross-view splat
support, spatial proximity to the seed, and pass scene/part size limits.
Rejected proposals are diagnostic, not objects to delete.

The workflow does not guarantee a complete object if a part is occluded,
missing from the reconstructed scene, or repeatedly segmented with the
background. It does not reconstruct wall or floor hidden by a removed object.
