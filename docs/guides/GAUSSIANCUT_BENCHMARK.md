# GaussianCut comparison

This integration calls the official `graphcut_segmentation` function from
`external_tools/GaussianCut/gaussian-splatting/utils/graphcut.py` without modifying
its energy, neighbor graph, or maximum-flow implementation. It uses the existing
scene and does not retrain it.

## Adaptations and limitations

- This is a solver benchmark, not a full upstream end-to-end reproduction.
- The primary unseeded comparison uses fresh rasterized bed-mask footprints
  from the intact scene in 13 training views. Earlier diagnostic trials used
  cached post-removal evidence and are not an equivalent unseeded baseline.
  Upstream's SAM-and-Track/custom CUDA masking are replaced by local adapters.
- A separate seeded refinement uses our current bed IDs as positive evidence;
  it is incremental refinement, not independent segmentation.
- Independently protected furniture source IDs are strong sink constraints.
- The source PLY records are retained directly, preserving all properties and
  128D semantic features; upstream's geometry-only writer is bypassed.
- Wooden-frame source IDs remain separate. Existing safe frame removal is held
  fixed during the bed-selection comparison.
- Held-out `frame_0134`, `frame_0141`, and `frame_0143` are not used by the solver.
- Automatic masks are pseudo-labels, not human-reviewed segmentation ground
  truth. Protected furniture masks currently cover more than one instance.
- Neither footprint recall nor changed bed pixels establish complete removal:
  review the actual RGB renders for deeper bed layers and background damage.

## Setup and commands

```bash
git clone --depth 1 https://github.com/umangi-jain/gaussiancut.git external_tools/GaussianCut
python -m pip install --no-deps -r configs/requirements-gaussiancut.txt
python -m gsedit.selection.benchmark_gaussiancut --help
python -m gsedit.evaluation.compare_gaussiancut --help
python -m gsedit collect_intact_cut_evidence --help
python -m gsedit guard_graphcut_additions --help
```

The setup above was completed in the `gaussian-orin` environment for this
experiment. No upstream dependency installer, segmentation weights, or training
stack was installed. Follow upstream license terms for research/evaluation use.

Every run requires a new output folder. A result above the 15% scene-deletion
safety limit is saved as a rejected diagnostic. Passing that gate never approves
the edit; dresser/background preservation and multi-view visual review remain
required. Background reconstruction is a separate geometry-gated stage.
