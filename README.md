# Gaussian scene editing

Local, reversible semantic editing of 3D Gaussian scenes on Jetson AGX Orin.

## Layout

```text
gsedit/
  assets/          Image/mesh-to-Gaussian conversion, alignment, merging
  selection/       Masks, instance attribution, grouping, removal, splitting
  reconstruction/  Surface/depth fitting and local background refinement
  generation/      Local image/atlas generation and inpainting
  rendering/       PLY, footprint, overhead, and component previews
  evaluation/      Benchmarks, diagnostics, and validation
  pipelines/       End-to-end workflow orchestration
  preprocessing/   Feature extraction, PCA, image/video utilities, viewer
docs/
  guides/          How to run the workflows
  reports/         Dated experiment results and limitations
tests/             Non-inference regression and synthetic tests
scripts/           Environment/setup scripts
archive/
  patch-backups/   Preserved historical .orig files; not active code
```

The upstream core (`scene/`, `gaussian_renderer/`, `arguments/`, `models/`,
`utils/`) and training entry points (`train.py`, `render.py`, `convert.py`,
`metrics.py`, `full_eval.py`) remain in place. Datasets, weights, third-party
tools, and all existing `output/` scenes/previews have not been relocated.

## Commands

From the repository root, with the existing conda environment activated:

```bash
python -m gsedit --list
python -m gsedit run_autonomous_edit --help
python -m gsedit add_asset --help
python -m pytest -q
```

Fully qualified entry points also work, e.g.
`python -m gsedit.selection.refine_shared_splats --help`.
Old root script paths have been replaced by module commands; flags are unchanged.
See [the documentation index](docs/README.md) for workflows and measured results.
