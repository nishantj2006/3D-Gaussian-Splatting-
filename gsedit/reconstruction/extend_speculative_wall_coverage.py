"""Test a connected synthetic wall surface around the main bed-hole region.

Only adds splats to a new unapproved preview.  The rectangle is a deliberately
synthetic coverage hypothesis, not measured wall geometry.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from gsedit.reconstruction.grow_speculative_wall import SH_C0
from utils.ply_semantic_utils import read_vertices, write_vertices


def run(args):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel

    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    seed = Path(args.seed).resolve()
    ply, vertices = read_vertices(seed)
    with np.load(args.provenance) as p:
        active = p['wall_active'].astype(bool)
        source_count = int(p['source_count'])
    wall_count = int(active.sum())
    if len(vertices) < source_count + wall_count:
        raise ValueError('Wall provenance does not match seed PLY')
    wall = vertices[source_count:source_count + wall_count]
    rows, cols = np.nonzero(active)

    # Only bridge the two largest wall pieces around the main bed opening.
    # The row range comes from their observed footprint; no other wall region
    # is extrapolated.  The new extent is separately reviewed across cameras.
    new_mask = np.zeros_like(active)
    new_mask[args.row_min:args.row_max + 1,
             args.col_min:args.col_max + 1] = True
    new_mask &= ~active
    new_rows, new_cols = np.nonzero(new_mask)
    lookup = np.full(active.shape, -1, np.int32)
    lookup[rows, cols] = np.arange(wall_count, dtype=np.int32)
    nearest = ndimage.distance_transform_edt(
        ~active, return_distances=False, return_indices=True)
    parent_idx = lookup[nearest[0, new_rows, new_cols],
                        nearest[1, new_rows, new_cols]]
    if (parent_idx < 0).any():
        raise AssertionError('Unmapped wall parent')
    fill = wall[parent_idx].copy()

    # The pre-existing wall is a regular fixed plane.  Fit its grid-to-world
    # affine transform, then place all new splats exactly on the same plane.
    sample = np.linspace(0, wall_count - 1, min(4000, wall_count)).astype(int)
    design = np.column_stack((np.ones(len(sample)), rows[sample], cols[sample]))
    xyz = np.column_stack([wall[k][sample] for k in ('x', 'y', 'z')])
    transform, *_ = np.linalg.lstsq(design, xyz, rcond=None)
    fit_error = float(np.max(np.abs(design @ transform - xyz)))
    if fit_error > 1e-3:
        raise ValueError(f'Existing wall is not a regular plane: {fit_error}')
    fill_xyz = np.column_stack((np.ones(len(fill)), new_rows, new_cols)) @ transform
    for axis, key in enumerate(('x', 'y', 'z')):
        fill[key] = fill_xyz[:, axis]

    observed = np.asarray(Image.open(Path(args.observed_dir) / 'wall-observed.png').convert('RGB'))
    evidence = np.asarray(Image.open(Path(args.observed_dir) / 'wall-evidence.png').convert('L')) > 127
    rgb = np.median(observed[evidence], axis=0)
    for channel in range(3):
        fill[f'f_dc_{channel}'] = (rgb[channel] / 255. - .5) / SH_C0
    # Give the new wall cells enough coverage in the previously empty gap,
    # tapering only near the outside of the tested rectangle.
    rect = np.zeros_like(active)
    rect[args.row_min:args.row_max + 1,
         args.col_min:args.col_max + 1] = True
    edge = ndimage.distance_transform_edt(rect)[new_rows, new_cols]
    opacity = np.clip(.62 * np.minimum(1., edge / 8.), .08, .62)
    fill['opacity'] = np.log(opacity / (1. - opacity))
    result = np.concatenate((vertices, fill))

    out.mkdir(parents=True)
    candidate = out / 'candidate-unapproved.ply'
    write_vertices(candidate, result, ply, comments=[
        'UNAPPROVED synthetic wall coverage and appearance; depth hypothetical.'])
    cameras = {c['img_name']: c for c in json.loads(Path(args.cameras).read_text())}
    before_model = GaussianModel(3, 128)
    before_model.load_ply(str(seed))
    model = GaussianModel(3, 128)
    model.load_ply(str(candidate))
    for item in (before_model, model):
        for key in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                    '_scaling', '_rotation', '_semantic_feature'):
            getattr(item, key).requires_grad_(False)
    review = {}
    for view in args.views:
        camera = cameras[view]
        height = round(camera['height'] * args.width / camera['width'])
        raster = GaussianRasterizer(camera_settings(camera, height, args.width)
                                    ._replace(sh_degree=3))
        with torch.no_grad():
            before = render_scene(before_model, raster, before_model.get_opacity)
            after = render_scene(model, raster, model.get_opacity)
        Image.fromarray(before).save(out / f'{view}-before.png')
        Image.fromarray(after).save(out / f'{view}-after.png')
        review[view] = {'mean_abs_rgb_change': float(np.abs(after.astype(float) - before).mean() / 255.),
                        'dark_pixels_before': int((before.mean(2) < 32).sum()),
                        'dark_pixels_after': int((after.mean(2) < 32).sum())}
    report = {'seed': str(seed), 'candidate': str(candidate),
              'source_count': source_count, 'prior_wall_count': wall_count,
              'new_wall_count': len(fill), 'plane_fit_max_error': fit_error,
              'hypothesis_grid_rectangle': [args.row_min, args.row_max,
                                            args.col_min, args.col_max],
              'visible_wall_median_rgb': rgb.tolist(),
              'prior_records_unchanged': bool(np.array_equal(vertices, result[:len(vertices)])),
              'semantic_dimensions': sum(k.startswith('semantic_') for k in result.dtype.names),
              'review': review, 'approved': False,
              'warning': 'Synthetic surface and 9.25 wall depth are hypotheses; visual approval required.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('seed', 'provenance', 'observed-dir', 'cameras', 'output-dir'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--row-min', type=int, default=868)
    p.add_argument('--row-max', type=int, default=1210)
    p.add_argument('--col-min', type=int, default=0)
    p.add_argument('--col-max', type=int, default=170)
    p.add_argument('--width', type=int, default=540)
    p.add_argument('--views', nargs='+', default=['frame_0134', 'frame_0141',
                                                  'frame_0143', 'frame_0131', 'frame_0147'])
    return p


if __name__ == '__main__':
    run(parser().parse_args())
