"""Apply a shared synthetic material texture to an unapproved wall preview.

The texture contributes only zero-mean, high-frequency color variation.  The
observed wall atlas is locked, and every non-wall Gaussian is left unchanged.
This is a plausible appearance experiment, not recovery of hidden wall detail.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from gsedit.reconstruction.grow_speculative_wall import SH_C0
from utils.ply_semantic_utils import read_vertices, write_vertices


def texture_residual(texture, rows, cols, sample_stride=3, contrast=1.5):
    """Sample a tileable, zero-mean material at shared wall-grid coordinates."""
    gray = np.asarray(texture.convert('L'), np.float32)
    gray = ndimage.gaussian_filter(gray, 4, mode='wrap')
    gray -= ndimage.gaussian_filter(gray, 40, mode='wrap')
    gray -= gray.mean()
    scale = max(float(gray.std()), 1.)
    # Integer indexing deliberately keeps the same material locked to each
    # surface point in every camera instead of using view-space noise.
    yy = (np.asarray(rows) * sample_stride) % gray.shape[0]
    xx = (np.asarray(cols) * sample_stride) % gray.shape[1]
    return np.clip(gray[yy, xx] * contrast * 5. / scale, -16., 16.)


def recolor_wall(vertices, provenance, texture, observed, evidence, contrast=1.5):
    source_count = int(provenance['source_count'])
    active = provenance['wall_active'].astype(bool)
    rows, cols = np.nonzero(active)
    wall_count = len(rows)
    if len(vertices) < source_count + wall_count:
        raise ValueError('Candidate is shorter than the wall provenance')
    actual = vertices[source_count:source_count + wall_count]
    original = np.asarray(provenance['initial_rgb'], np.uint8)
    if len(original) != wall_count:
        raise ValueError('Provenance color count does not match wall splats')
    if observed.shape[:2] != evidence.shape or not evidence.any():
        raise ValueError('Visible wall colors and evidence mask must align')
    wall_median = np.median(observed[evidence], axis=0)

    # Fine wall splats are stored in row-major grid order by the builder.
    coarse = provenance['trusted_coarse'].astype(bool)
    fine_step = float(provenance['wall_grid_step'])
    coarse_step = .075  # coarse grid step in the wall-only provenance
    coarse_rows = np.clip(np.floor((rows + .5) * fine_step / coarse_step).astype(int),
                          0, coarse.shape[0] - 1)
    coarse_cols = np.clip(np.floor((cols + .5) * fine_step / coarse_step).astype(int),
                          0, coarse.shape[1] - 1)
    # A few cells around visible evidence get a smooth transition.
    distance = ndimage.distance_transform_edt(~coarse)
    confidence_weight = np.clip(distance[coarse_rows, coarse_cols] / 4., 0., 1.)
    residual = texture_residual(texture, rows, cols, contrast=contrast)
    residual *= confidence_weight
    before_rgb = np.column_stack([
        np.clip((actual[f'f_dc_{channel}'] * SH_C0 + .5) * 255., 0, 255)
        for channel in range(3)])
    # The old affine extrapolation becomes an implausible purple field deep in
    # the hole.  Use the visible wall median there, but keep actual observed
    # texels locked and blend over a few atlas cells.
    synthetic_rgb = wall_median[None, :] + residual[:, None]
    after_rgb = np.clip(before_rgb * (1. - confidence_weight[:, None]) +
                        synthetic_rgb * confidence_weight[:, None], 0., 255.)
    result = vertices.copy()
    for channel in range(3):
        result[f'f_dc_{channel}'][source_count:source_count + wall_count] = (
            after_rgb[:, channel] / 255. - .5) / SH_C0
    return result, {
        'source_count': source_count,
        'wall_count': wall_count,
        'texture_contrast': contrast,
        'visible_wall_median_rgb': wall_median.tolist(),
        'mean_abs_wall_rgb_change': float(np.abs(after_rgb - before_rgb).mean()),
        'max_abs_wall_rgb_change': float(np.abs(after_rgb - before_rgb).max()),
        'trusted_fraction_of_wall_splats': float((confidence_weight == 0).mean()),
        'outside_wall_unchanged': bool(
            np.array_equal(vertices[:source_count], result[:source_count]) and
            np.array_equal(vertices[source_count + wall_count:], result[source_count + wall_count:])),
    }


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
    provenance_path = Path(args.provenance).resolve()
    texture_path = Path(args.texture).resolve()
    ply, vertices = read_vertices(seed)
    with np.load(provenance_path) as data:
        provenance = {key: data[key] for key in data.files}
    observed_dir = Path(args.observed_dir).resolve()
    observed = np.asarray(Image.open(observed_dir / 'wall-observed.png').convert('RGB'))
    evidence = np.asarray(Image.open(observed_dir / 'wall-evidence.png').convert('L')) > 127
    result, report = recolor_wall(vertices, provenance,
                                  Image.open(texture_path).convert('RGB'),
                                  observed, evidence, args.contrast)
    if not report['outside_wall_unchanged']:
        raise AssertionError('Non-wall Gaussian records changed')
    out.mkdir(parents=True)
    candidate = out / 'candidate-unapproved.ply'
    write_vertices(candidate, result, ply, comments=[
        'UNAPPROVED synthetic wall material; hidden appearance and depth are hypothetical.'])
    Image.open(texture_path).convert('RGB').save(out / 'synthetic-wall-swatch.png')

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
    report.update({'seed': str(seed), 'candidate': str(candidate),
                   'provenance': str(provenance_path), 'observed_dir': str(observed_dir),
                   'texture': str(out / 'synthetic-wall-swatch.png'),
                   'semantic_dimensions': sum(k.startswith('semantic_') for k in result.dtype.names),
                   'review': review, 'approved': False,
                   'warning': 'Synthetic material and wall depth are speculative; visual approval required.'})
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('seed', 'provenance', 'texture', 'observed-dir', 'cameras', 'output-dir'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--contrast', type=float, default=1.5)
    p.add_argument('--width', type=int, default=540)
    p.add_argument('--views', nargs='+', default=['frame_0134', 'frame_0141',
                                                  'frame_0143', 'frame_0131', 'frame_0147'])
    return p


if __name__ == '__main__':
    run(parser().parse_args())
