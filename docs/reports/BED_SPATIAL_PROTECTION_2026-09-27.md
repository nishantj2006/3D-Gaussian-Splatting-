# Spatial bed grouping with protected neighboring furniture

## Outcome: preservation improved; whole-bed removal not achieved

The new preview starts from the unchanged bottle-removed 5k source, not the
damaged prior bed preview. It removes 7,327 source splats and preserves all
17,253 protected source splats. All retained records are exactly unchanged;
the original 190 PLY properties and 128D semantic features remain intact.
The bed body and wooden-frame source IDs are separate. IDs and deleted records
are stored alongside the candidate, so the operation is reversible.

The protected-furniture masks overlap 10,233 IDs in the old initial removal:
603 old bed-body IDs and 9,649 old wooden-frame IDs. This demonstrates that the
old selection was not independent of the neighboring furniture. Automatic
"dresser/cabinet" detection sometimes selects the wooden unit and sometimes
the white cabinet/appliance. Protection intentionally retains both, but these
masks are pseudo-labels, not verified ground-truth object identities.

**RGB renders still show a large blurred bed-shaped residue.** High front
footprint recall does not establish whole-bed removal. The preview remains
unapproved; no background reconstruction was attempted in this experiment.

## Method

- Independent local DINO/SAM2 protection masks: 18 accepted views, of which
  15 are training views; the three held-out views do not affect selection.
- Exact rasterizer contribution identifies protected source splats before
  any removal; protected IDs cannot enter the bed graph.
- Bed and frame seeds are revalidated against camera silhouettes, then assigned
  separate IDs through a neighbor graph with scene-derived spacing.
- An oriented occupied envelope only limits search; it is never a box-cut mask.
- Iterative re-rendering measures newly exposed contributions. Completely hidden
  splats may grow only within a narrow seed band and consistent silhouettes.
- A cached expansion additionally uses a multi-view volume core and footprint
  evidence. Broad contradictory or disconnected splats are not forcibly erased.

Implementation: `spatial_instance_removal.py`, `expand_spatial_instance.py`,
`validate_spatial_removal.py`. Thresholds and input paths are CLI arguments;
there are no embedded scene colors or coordinates.

## Held-out validation

RGB change is relative to the intact source, normalized to [0,1]. A changed
pixel has maximum channel delta greater than 0.08. Protection masks are
automatic pseudo-labels. These are preservation diagnostics, not approval.

| View | Protected-mask mean RGB change | Protected-mask changed pixels | Removed front-footprint recall | Outside-mask changed pixels |
| --- | ---: | ---: | ---: | ---: |
| frame_0134 | 0.000083 | 0% | 76.25% | 0.73% |
| frame_0141 | 0.000042 | 0% | 91.50% | 1.01% |
| frame_0143 | 0.003389 | 0.81% | 72.65% | 0.89% |

The protected cabinet/furniture is visibly restored, but the residual bed
remains recognizable. Small image changes can occur at occlusion boundaries
even when protected source records are unchanged.

The revised grouping stage took 297.25 seconds, with 3,368 MiB peak process RSS
and 976 MiB peak PyTorch GPU allocation. Cached expansion took 3.01 seconds.
RSS and GPU allocation overlap on the Orin's unified memory and must not be
added as independent physical RAM. The suite passed 106 tests with one skip;
GPU stages were run separately on the Orin.

## Artifacts

- Candidate: `output/bed-ops/bed-spatial-dresser-protected-expanded-v2/candidate.ply`
- Per-source instance IDs, protected IDs, retained IDs and deleted records are
  in the same folder.
- RGB renders and numerical checks: its `validation/` folder.
- Source contribution evidence: `output/bed-ops/bed-spatial-dresser-protected-v2/`.
- Protection masks: `output/bed-ops/dresser-protection-masks-v1/`.

Next work must address the residual's mixed/background contributions rather
than claim that a larger spatial box is a safe whole-bed boundary. The original
source and all earlier previews remain unchanged.
