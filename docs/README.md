# Documentation

Run commands from the repository root in the `gaussian-orin` conda environment.
Editing tools now use `python -m gsedit COMMAND`, or their fully qualified
module names. For example, `python remove.py ...` is now
`python -m gsedit remove ...`. Flags and output locations are unchanged.

## Usage guides

- [Asset generation and insertion](guides/ASSET_ADDITION.md)
- [Autonomous removal and reconstruction](guides/AUTONOMOUS_EDIT.md)
- [Automatic removal](guides/AUTO_REMOVE.md)
- [Surface placement and extraction](guides/SURFACE_PIPELINE.md)
- [Multi-view instance attribution](guides/MULTIVIEW_INSTANCE.md)
- [Whole-object previews](guides/WHOLE_OBJECT_PREVIEW.md)
- [Local bed refinement](guides/BED_LOCAL_REFINEMENT.md)
- [Continuous background replacement](guides/BED_CONTINUOUS_REPLACEMENT.md)
- [Generated background workflow](guides/BED_GENERATED_PIPELINE.md)
- [Image-first experiments](guides/IMAGE_FIRST_EXPERIMENT.md)

## Experiment reports

Historical results are kept in `reports/`. They are experiment records, not
approval of the resulting scenes. The latest report is
[covariance grouping and shared-splat splitting](reports/BED_COVARIANCE_SPLIT_BENCHMARK_2026-09-28.md).

Source scenes and generated previews remain in their original `output/` paths.
Old patch backups are preserved under `archive/patch-backups/`; they are not
active code.
