"""Bounded multi-view appearance optimization for an appended wall preview.

Only DC color and opacity of appended wall splats are trainable. Original
records, geometry, covariance, SH residuals, and 128D semantics remain frozen.
Visible wall photos supervise appearance; the source render supervises pixels
outside the edit hole and independently segmented carpet/dresser regions.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from gsedit.reconstruction.optimize_continuous_background import render, mean_masked_l1
from scene.gaussian_model import GaussianModel
from utils.ply_semantic_utils import read_vertices, write_vertices


def manifest_mask(manifest, view, size):
    entry = manifest.get('views', {}).get(view, {})
    if not entry.get('accepted') or not entry.get('mask_path'):
        return np.zeros((size[1], size[0]), bool)
    image = Image.open(entry['mask_path']).convert('L').resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def neighbor_pairs(active):
    labels = np.full(active.shape, -1, np.int64)
    labels[active] = np.arange(active.sum())
    pairs = []
    both = active[:-1] & active[1:]
    pairs.append(np.column_stack((labels[:-1][both], labels[1:][both])))
    both = active[:, :-1] & active[:, 1:]
    pairs.append(np.column_stack((labels[:, :-1][both], labels[:, 1:][both])))
    return np.concatenate(pairs) if any(len(x) for x in pairs) else np.empty((0, 2), np.int64)


def run(a):
    started = time.perf_counter()
    output = Path(a.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    seed_dir = Path(a.seed_dir).resolve()
    report = json.loads((seed_dir / 'report.json').read_text())
    source_ply, vertices = read_vertices(seed_dir / 'candidate-unapproved.ply')
    source_count = int(report['source_count'])
    if not 0 < source_count < len(vertices):
        raise ValueError('Invalid original/appended record boundary')
    _, original = read_vertices(report['source'])
    if not np.array_equal(original, vertices[:source_count]):
        raise ValueError('Seed changed source records')
    with np.load(seed_dir / 'provenance.npz') as provenance:
        active = provenance['wall_active']
    if int(active.sum()) != len(vertices) - source_count:
        raise ValueError('Wall grid does not match appended splats')

    cameras = {c['img_name']: c for c in json.loads(Path(a.cameras).read_text())}
    sweep = json.loads(Path(a.sweep_report).read_text())
    wall_manifest = json.loads(Path(a.wall_manifest).read_text())
    floor_manifest = json.loads(Path(a.floor_manifest).read_text())
    dresser_manifest = json.loads(Path(a.dresser_manifest).read_text())
    training = [v for v in sweep['training_views'] if v in cameras]
    if len(training) < 4:
        raise ValueError('Too few training views')

    model = GaussianModel(3, 128)
    model.load_ply(str(seed_dir / 'candidate-unapproved.ply'))
    for name in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                 '_scaling', '_rotation', '_semantic_feature'):
        getattr(model, name).requires_grad_(False)
    added = torch.arange(source_count, len(vertices), dtype=torch.long, device='cuda')
    base_dc = model._features_dc.detach()
    base_rest = model._features_rest.detach()
    base_raw_alpha = model._opacity.detach()
    base_alpha = model.get_opacity.detach()
    dc_delta = torch.nn.Parameter(torch.zeros((len(added), 3), device='cuda'))
    alpha_delta = torch.nn.Parameter(torch.zeros((len(added), 1), device='cuda'))
    optimizer = torch.optim.Adam([dc_delta, alpha_delta], lr=a.learning_rate)

    pairs_np = neighbor_pairs(active)
    pairs = torch.as_tensor(pairs_np, dtype=torch.long, device='cuda')
    rasters, photos, wall_masks, preserve_masks, source_targets = {}, {}, {}, {}, {}
    coverage = {}
    for view in training:
        camera = cameras[view]
        height = round(camera['height'] * a.width / camera['width'])
        size = (a.width, height)
        rasters[view] = GaussianRasterizer(
            camera_settings(camera, height, a.width)._replace(sh_degree=3))
        photo = Image.open(Path(a.images) / f'{view}.jpg').convert('RGB').resize(
            size, Image.Resampling.LANCZOS)
        photos[view] = torch.from_numpy(
            np.asarray(photo).copy().astype(np.float32) / 255).permute(2, 0, 1).cuda()
        wall = manifest_mask(wall_manifest, view, size)
        floor = manifest_mask(floor_manifest, view, size)
        dresser = manifest_mask(dresser_manifest, view, size)
        hole_path = Path(a.hole_masks) / f'{view}.png'
        if not hole_path.exists():
            raise FileNotFoundError(hole_path)
        hole = np.asarray(Image.open(hole_path).convert('L').resize(
            size, Image.Resampling.NEAREST)) > 127
        expanded_hole = ndimage.binary_dilation(hole, iterations=a.hole_guard_px)
        # Photo supervision is limited to independently accepted, visible wall.
        wall &= ~expanded_hole
        preserve = (~expanded_hole) | floor | dresser
        wall_masks[view] = torch.from_numpy(wall.astype(np.float32)).cuda()[None]
        preserve_masks[view] = torch.from_numpy(preserve.astype(np.float32)).cuda()[None]
        with torch.no_grad():
            source_alpha = base_alpha.clone()
            source_alpha[added] = 0
            source_targets[view] = render(model, rasters[view], model.get_features,
                                          source_alpha).detach()
        coverage[view] = {'wall_pixels': int(wall.sum()),
                          'preserve_pixels': int(preserve.sum())}

    torch.cuda.reset_peak_memory_stats()
    history = []
    for epoch in range(a.epochs):
        rows = []
        for view in training:
            optimizer.zero_grad(set_to_none=True)
            limited_dc = dc_delta.clamp(-a.max_dc_shift, a.max_dc_shift)
            dc = base_dc.index_copy(0, added, base_dc[added] + limited_dc[:, None, :])
            shs = torch.cat((dc, base_rest), dim=1)
            limited_alpha = alpha_delta.clamp(-a.max_alpha_shift, a.max_alpha_shift)
            alpha = base_alpha.index_copy(
                0, added, torch.sigmoid(base_raw_alpha[added] + limited_alpha))
            prediction = render(model, rasters[view], shs, alpha)
            wall_loss = mean_masked_l1(prediction, photos[view], wall_masks[view])
            preserve_loss = mean_masked_l1(
                prediction, source_targets[view], preserve_masks[view])
            if len(pairs):
                tv = (limited_dc[pairs[:, 0]] - limited_dc[pairs[:, 1]]).square().mean()
            else:
                tv = limited_dc.new_zeros(())
            regularizer = (a.color_penalty * limited_dc.square().mean() +
                           a.alpha_penalty * limited_alpha.square().mean() +
                           a.smoothness_weight * tv)
            loss = a.wall_weight * wall_loss + a.preserve_weight * preserve_loss + regularizer
            loss.backward()
            optimizer.step()
            rows.append((float(wall_loss.detach()), float(preserve_loss.detach()),
                         float(tv.detach()), float(loss.detach())))
        item = {'epoch': epoch + 1,
                'wall_l1': float(np.mean([x[0] for x in rows])),
                'preserve_l1': float(np.mean([x[1] for x in rows])),
                'smoothness': float(np.mean([x[2] for x in rows])),
                'loss': float(np.mean([x[3] for x in rows]))}
        history.append(item)
        print(json.dumps(item), flush=True)

    result = vertices.copy()
    delta_color = dc_delta.detach().clamp(-a.max_dc_shift, a.max_dc_shift).cpu().numpy()
    for channel in range(3):
        result[f'f_dc_{channel}'][source_count:] += delta_color[:, channel]
    final_alpha = torch.sigmoid(base_raw_alpha[added] + alpha_delta.detach().clamp(
        -a.max_alpha_shift, a.max_alpha_shift)).cpu().numpy()[:, 0]
    result['opacity'][source_count:] = np.log(final_alpha / (1-final_alpha))
    if not np.array_equal(result[:source_count], original):
        raise RuntimeError('Optimization changed source records')

    output.mkdir(parents=True)
    candidate = output / 'candidate-unapproved.ply'
    write_vertices(candidate, result, source_ply, comments=[
        'UNAPPROVED bounded multi-view appearance optimization; speculative wall depth.'])
    np.savez_compressed(output / 'optimization-deltas.npz',
                        added_start=source_count, dc_delta=delta_color,
                        alpha_delta=alpha_delta.detach().cpu().numpy())

    final = GaussianModel(3, 128)
    final.load_ply(str(candidate))
    before = GaussianModel(3, 128)
    before.load_ply(report['source'])
    for item in (final, before):
        for name in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                     '_scaling', '_rotation', '_semantic_feature'):
            getattr(item, name).requires_grad_(False)
    validation = {}
    for view in a.views:
        camera = cameras[view]
        height = round(camera['height'] * a.render_width / camera['width'])
        raster = GaussianRasterizer(camera_settings(camera, height, a.render_width)
                                    ._replace(sh_degree=3))
        with torch.no_grad():
            source_image = render(before, raster, before.get_features,
                                  before.get_opacity).permute(1, 2, 0).clamp(0, 1)
            final_image = render(final, raster, final.get_features,
                                 final.get_opacity).permute(1, 2, 0).clamp(0, 1)
        source_np = np.rint(source_image.cpu().numpy() * 255).astype(np.uint8)
        final_np = np.rint(final_image.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(source_np).save(output / f'{view}-before.png')
        Image.fromarray(final_np).save(output / f'{view}-after.png')
        size = (a.render_width, height)
        floor = manifest_mask(floor_manifest, view, size)
        dresser = manifest_mask(dresser_manifest, view, size)
        delta = np.abs(final_np.astype(float) - source_np.astype(float)).mean(axis=2) / 255
        validation[view] = {
            'dark_pixels_before': int((source_np.mean(axis=2) < a.dark_threshold).sum()),
            'dark_pixels_after': int((final_np.mean(axis=2) < a.dark_threshold).sum()),
            'carpet_mean_abs_change': float(delta[floor].mean()) if floor.any() else None,
            'dresser_mean_abs_change': float(delta[dresser].mean()) if dresser.any() else None,
        }

    final_report = {
        'seed': str(seed_dir), 'source': report['source'], 'candidate': str(candidate),
        'source_count': source_count, 'added_optimized': len(added),
        'training_views': training, 'review_views': a.views,
        'wall_offset_hypothesis': report['wall_offset_hypothesis'],
        'depth_constrained': False, 'coverage': coverage, 'history': history,
        'mean_abs_dc_shift': float(np.abs(delta_color).mean()),
        'mean_abs_alpha_shift': float(np.abs(alpha_delta.detach().cpu().numpy()).mean()),
        'source_records_unchanged': True, 'validation': validation,
        'approved': False,
        'warning': 'Hidden texture and wall depth are not recoverable from this capture.',
        'elapsed_seconds': time.perf_counter() - started,
        'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }
    (output / 'report.json').write_text(json.dumps(final_report, indent=2) + '\n')
    print(json.dumps(final_report, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('seed-dir', 'cameras', 'images', 'sweep-report', 'wall-manifest',
                 'floor-manifest', 'dresser-manifest', 'hole-masks', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--views', nargs='+', default=[
        'frame_0134', 'frame_0141', 'frame_0143', 'frame_0131', 'frame_0147'])
    p.add_argument('--width', type=int, default=270)
    p.add_argument('--render-width', type=int, default=540)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=.025)
    p.add_argument('--hole-guard-px', type=int, default=3)
    p.add_argument('--wall-weight', type=float, default=2.)
    p.add_argument('--preserve-weight', type=float, default=5.)
    p.add_argument('--color-penalty', type=float, default=.05)
    p.add_argument('--alpha-penalty', type=float, default=.02)
    p.add_argument('--smoothness-weight', type=float, default=.1)
    p.add_argument('--max-dc-shift', type=float, default=.12)
    p.add_argument('--max-alpha-shift', type=float, default=.75)
    p.add_argument('--dark-threshold', type=float, default=32.)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
