"""Rebuild a speculative wall preview with small, regularly spaced splats.

The wall plane and footprint come from an earlier unapproved wall-only preview.
Only appended records are replaced; every source record remains byte-identical.
Visible wall texels fit one robust low-frequency shared color field so sparse
photo coverage does not create nearest-neighbor color islands.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy import ndimage

from gsedit.assets.align_asset import plane_frame
from gsedit.reconstruction.build_continuous_background import gaussian_frame, surface_gaussians
from gsedit.reconstruction.grow_speculative_wall import SH_C0
from gsedit.reconstruction.refine_speculative_wall_color import robust_wall_atlas
from gsedit.reconstruction.wall_color_field import predict_wall_colors
from utils.ply_semantic_utils import read_vertices, write_vertices


def fine_grid(active, low, coarse_step, fine_step):
    """Return fine cell centers whose containing coarse cell is active."""
    extent = np.asarray(active.shape, float) * coarse_step
    shape = np.ceil(extent / fine_step).astype(int)
    ij = np.indices(tuple(shape)).reshape(2, -1).T
    uv = np.asarray(low) + (ij + .5) * fine_step
    coarse = np.floor((uv - low) / coarse_step).astype(int)
    inside = np.all((coarse >= 0) & (coarse < np.asarray(active.shape)), axis=1)
    keep = np.zeros(len(ij), bool)
    keep[inside] = active[coarse[inside, 0], coarse[inside, 1]]
    return tuple(int(x) for x in shape), ij[keep], uv[keep], coarse[keep]


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel

    started = time.perf_counter()
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    prior = Path(a.prior_dir).resolve()
    prior_report = json.loads((prior / 'report.json').read_text())
    ply, source = read_vertices(prior_report['source'])
    _, old_candidate = read_vertices(prior / 'candidate-unapproved.ply')
    if not np.array_equal(source, old_candidate[:len(source)]):
        raise ValueError('Prior preview changed source records')
    with np.load(prior / 'provenance.npz') as provenance:
        active = provenance['wall_active']
        low = provenance['wall_grid_low'].astype(float)
        coarse_step = float(provenance['wall_grid_step'])
        old_parents = provenance['wall_parent_indices']
    if len(old_candidate) - len(source) != int(active.sum()):
        raise ValueError('Prior wall provenance does not match appended records')

    refined, trusted, center = robust_wall_atlas(
        np.asarray(Image.open(prior / 'wall-observed.png').convert('RGB')),
        np.asarray(Image.open(prior / 'wall-evidence.png').convert('L')) > 127,
        a.max_robust_distance)
    shape, ij, uv, coarse = fine_grid(active, low, coarse_step, a.step)
    if len(uv) > a.max_splats:
        raise ValueError(f'Fine wall has {len(uv)} splats; limit is {a.max_splats}')

    # Use coarse atlas coordinates for a shared robust plane fit.  The fit is
    # intentionally low frequency because only a small fraction is observed.
    atlas_y = (uv[:, 0] - low[0]) / coarse_step - .5
    atlas_x = (uv[:, 1] - low[1]) / coarse_step - .5
    rgb = predict_wall_colors(refined, trusted, atlas_y, atlas_x)

    floor = json.loads(Path(a.floor_fit).read_text())
    sweep = json.loads(Path(a.sweep_report).read_text())
    origin, frame = plane_frame(floor['plane_origin'],
                                floor['plane_normal_toward_removed_object'])
    normal2 = np.asarray(sweep['plane_normal_floor_xy'], float)
    normal2 /= np.linalg.norm(normal2)
    tangent, wall_basis = gaussian_frame(frame, normal2)
    offset = float(sweep['best_offset'])
    xyz = (origin + (uv[:, 0, None] * tangent + offset * normal2) @ frame[:, :2].T +
           uv[:, 1, None] * frame[:, 2])

    parent_grid = np.full(active.shape, -1, np.int64)
    parent_grid[active] = old_parents
    parents = parent_grid[coarse[:, 0], coarse[:, 1]]
    fine_active = np.zeros(shape, bool)
    fine_active[ij[:, 0], ij[:, 1]] = True
    edge = ndimage.distance_transform_edt(fine_active)[fine_active]
    fade_cells = max(1., a.edge_fade / a.step)
    opacity = np.clip(a.opacity * np.minimum(1., edge / fade_cells), .08, .95)
    fill = surface_gaussians(source, parents, xyz, step=a.step,
                             basis=wall_basis, opacity=opacity)
    for channel in range(3):
        fill[f'f_dc_{channel}'] = (rgb[:, channel] / 255 - .5) / SH_C0
    result = np.concatenate((source, fill))

    out.mkdir(parents=True)
    atlas = np.zeros((*shape, 3), np.uint8)
    atlas[ij[:, 0], ij[:, 1]] = rgb
    Image.fromarray(atlas).save(out / 'wall-atlas-low-frequency.png')
    Image.fromarray((trusted * 255).astype(np.uint8)).save(out / 'wall-trusted-evidence.png')
    write_vertices(out / 'candidate-unapproved.ply', result, ply, comments=[
        'UNAPPROVED fine speculative wall; depth remains an unmeasured hypothesis.'])
    np.savez_compressed(out / 'provenance.npz', wall_parent_indices=parents,
                        wall_active=fine_active, wall_grid_low=low,
                        wall_grid_step=a.step, source_count=len(source),
                        initial_rgb=rgb, trusted_coarse=trusted)

    cameras = {c['img_name']: c for c in json.loads(Path(a.cameras).read_text())}
    before_model = GaussianModel(3, 128)
    before_model.load_ply(prior_report['source'])
    model = GaussianModel(3, 128)
    model.load_ply(str(out / 'candidate-unapproved.ply'))
    for item in (before_model, model):
        for key in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                    '_scaling', '_rotation', '_semantic_feature'):
            getattr(item, key).requires_grad_(False)
    validation = {}
    for view in a.views:
        camera = cameras[view]
        height = round(camera['height'] * a.width / camera['width'])
        raster = GaussianRasterizer(camera_settings(camera, height, a.width)
                                    ._replace(sh_degree=3))
        with torch.no_grad():
            before = render_scene(before_model, raster, before_model.get_opacity)
            after = render_scene(model, raster, model.get_opacity)
        Image.fromarray(before).save(out / f'{view}-before.png')
        Image.fromarray(after).save(out / f'{view}-after.png')
        validation[view] = {
            'dark_pixels_before': int((before.mean(axis=2) < a.dark_threshold).sum()),
            'dark_pixels_after': int((after.mean(axis=2) < a.dark_threshold).sum()),
            'mean_abs_rgb_change': float(np.abs(after.astype(float)-before).mean()/255),
        }
    report = {
        'source': str(Path(prior_report['source']).resolve()),
        'prior': str(prior),
        'candidate': str(out / 'candidate-unapproved.ply'),
        'source_count': len(source),
        'wall_offset_hypothesis': offset,
        'depth_constrained': False,
        'coarse_step': coarse_step,
        'fine_step': a.step,
        'wall_added': len(fill),
        'trusted_wall_texels': int(trusted.sum()),
        'observed_wall_fraction': float(trusted.mean()),
        'dominant_wall_rgb': center.tolist(),
        'color_model': 'robust shared affine atlas field',
        'source_records_unchanged': bool(np.array_equal(source, result[:len(source)])),
        'semantic_dimensions': sum(k.startswith('semantic_') for k in source.dtype.names),
        'validation': validation,
        'approved': False,
        'warning': 'Wall depth and hidden appearance remain hypothetical; visual approval required.',
        'elapsed_seconds': time.perf_counter() - started,
        'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prior-dir', 'floor-fit', 'sweep-report', 'cameras', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--views', nargs='+', default=[
        'frame_0134', 'frame_0141', 'frame_0143', 'frame_0131', 'frame_0147'])
    p.add_argument('--step', type=float, default=.035)
    p.add_argument('--opacity', type=float, default=.62)
    p.add_argument('--edge-fade', type=float, default=.3)
    p.add_argument('--max-splats', type=int, default=60000)
    p.add_argument('--max-robust-distance', type=float, default=4.)
    p.add_argument('--width', type=int, default=540)
    p.add_argument('--dark-threshold', type=float, default=32.)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
