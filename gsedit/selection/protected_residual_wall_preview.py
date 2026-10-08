"""Preview residual-bed attenuation while restoring protected source splats.

Only accepted training masks select source records. Held-out views are rendered
for review. The appended wall is fixed, and all changes are opacity-only.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image


def select_residue(inside, outside, support, protected_mass, protected_support,
                   protected_ids, floor_like, active, *, min_inside, min_views,
                   min_agreement, max_protected_mass, max_protected_support):
    agreement = inside / np.maximum(inside + outside, 1e-8)
    selected = ((inside >= min_inside) & (support >= min_views) &
                (agreement >= min_agreement) &
                (protected_mass <= max_protected_mass) &
                (protected_support <= max_protected_support) &
                ~protected_ids & ~floor_like & active)
    return selected, agreement


def load_mask(manifest, view, size):
    entry = manifest['views'].get(view, {})
    if not entry.get('accepted'):
        raise ValueError(f'Missing accepted protection mask: {view}')
    return np.asarray(Image.open(entry['mask_path']).convert('L').resize(
        size, Image.Resampling.NEAREST)) > 127


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.selection.refine_revealed_layers import view_contributions
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel
    from utils.ply_semantic_utils import read_vertices, write_vertices

    started = time.perf_counter()
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    seed_dir = Path(a.seed_dir).resolve()
    seed_report = json.loads((seed_dir / 'report.json').read_text())
    ply, current = read_vertices(seed_dir / 'candidate-unapproved.ply')
    _, pre_aggressive = read_vertices(a.pre_aggressive)
    source_count = int(seed_report['source_count'])
    if len(pre_aggressive) != source_count or not np.array_equal(
            current[:source_count][list(current.dtype.names[:3])],
            pre_aggressive[list(pre_aggressive.dtype.names[:3])]):
        raise ValueError('Pre-aggressive/current source index alignment failed')
    if current.dtype != pre_aggressive.dtype:
        raise ValueError('Source schema mismatch')
    retained_ids = np.load(a.retained_source_indices, allow_pickle=False)
    protected_source_ids = np.load(a.protected_source_indices, allow_pickle=False)
    if len(retained_ids) != source_count or not np.all(np.diff(retained_ids) > 0):
        raise ValueError('Retained/source ID mapping mismatch')
    protected = np.isin(retained_ids, protected_source_ids)
    restoration = protected & (current['opacity'][:source_count] !=
                               pre_aggressive['opacity'])
    if not restoration.any():
        raise ValueError('No protected splats to restore; check lineage')

    restored = current.copy()
    restored['opacity'][:source_count][protected] = pre_aggressive['opacity'][protected]
    floor = json.loads(Path(a.floor_fit).read_text())
    xyz = np.column_stack([current[key][:source_count] for key in ('x', 'y', 'z')])
    normal = np.asarray(floor['plane_normal_toward_removed_object'], float)
    normal /= np.linalg.norm(normal)
    floor_distance = np.abs((xyz - floor['plane_origin']) @ normal)
    floor_like = floor_distance <= a.floor_error_multiplier * floor['plane_error_q90']
    floor_like |= (xyz - floor['plane_origin']) @ normal < -a.floor_error_multiplier * floor['plane_error_q90']

    cameras = {item['img_name']: item for item in json.loads(Path(a.cameras).read_text())}
    bed_manifest = json.loads(Path(a.bed_manifest).read_text())
    dresser_manifest = json.loads(Path(a.dresser_manifest).read_text())
    carpet_manifest = json.loads(Path(a.carpet_manifest).read_text())
    holdouts = set(a.holdout_views)
    views = sorted(view for view, entry in bed_manifest['views'].items()
                   if entry.get('accepted') and view in cameras and view not in holdouts)
    if len(views) < a.min_training_views:
        raise ValueError('Too few accepted training masks')
    for view in views:
        if not dresser_manifest['views'].get(view, {}).get('accepted'):
            raise ValueError(f'Training view lacks dresser protection: {view}')

    model = GaussianModel(3, 128)
    model.load_ply(str(seed_dir / 'candidate-unapproved.ply'))
    for key in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                '_scaling', '_rotation', '_semantic_feature'):
        getattr(model, key).requires_grad_(False)
    original_alpha = model.get_opacity.detach().clone()
    base_logits = torch.from_numpy(restored['opacity'].copy()).cuda()[:, None]
    base_alpha = torch.sigmoid(base_logits)
    pool = torch.arange(source_count, device='cuda')
    inside = np.zeros(source_count, np.float64)
    outside = np.zeros(source_count, np.float64)
    support = np.zeros(source_count, np.uint16)
    protected_mass = np.zeros(source_count, np.float64)
    protected_support = np.zeros(source_count, np.uint16)
    torch.cuda.reset_peak_memory_stats()
    for view in views:
        camera = cameras[view]
        height = round(camera['height'] * a.width / camera['width'])
        size = (a.width, height)
        bed = load_mask(bed_manifest, view, size)
        dresser = load_mask(dresser_manifest, view, size)
        bed &= ~dresser
        raster = GaussianRasterizer(camera_settings(camera, height, a.width)
                                    ._replace(sh_degree=3))
        pin, pout = view_contributions(
            model, raster, base_alpha, pool,
            torch.from_numpy(bed.astype(np.float32)).cuda())
        protect, _ = view_contributions(
            model, raster, base_alpha, pool,
            torch.from_numpy(dresser.astype(np.float32)).cuda())
        pin, pout, protect = (np.maximum(item, 0) for item in (pin, pout, protect))
        inside += pin
        outside += pout
        protected_mass += protect
        support += pin >= a.min_view_contribution
        protected_support += protect >= a.min_view_contribution
        print(json.dumps({'view': view, 'bed_pixels': int(bed.sum()),
                          'dresser_pixels': int(dresser.sum())}), flush=True)

    active = (base_alpha[:source_count, 0].cpu().numpy() >= a.min_active_opacity)
    selected, agreement = select_residue(
        inside, outside, support, protected_mass, protected_support,
        protected, floor_like, active,
        min_inside=a.min_inside, min_views=a.min_views,
        min_agreement=a.min_agreement,
        max_protected_mass=a.max_protected_mass,
        max_protected_support=a.max_protected_support)
    selected_ids = np.flatnonzero(selected)
    if len(selected_ids) > a.max_additions:
        raise ValueError(f'Selection {len(selected_ids)} exceeds limit {a.max_additions}')
    if np.any(protected[selected_ids]) or np.any(floor_like[selected_ids]):
        raise AssertionError('Protected or floor splats selected')
    result = restored.copy()
    result['opacity'][selected_ids] = a.opacity_logit
    if not np.array_equal(result[source_count:], current[source_count:]):
        raise AssertionError('Appended wall records changed')
    for key in current.dtype.names:
        if key != 'opacity' and not np.array_equal(result[key], current[key]):
            raise AssertionError(f'Non-opacity field changed: {key}')

    out.mkdir(parents=True)
    np.save(out / 'newly-attenuated-current-indices.npy', selected_ids)
    np.save(out / 'newly-attenuated-source-indices.npy', retained_ids[selected_ids])
    np.save(out / 'restored-protected-current-indices.npy', np.flatnonzero(restoration))
    np.savez_compressed(out / 'selection-evidence.npz',
                        inside=inside.astype(np.float32),
                        outside=outside.astype(np.float32),
                        support=support,
                        protected_mass=protected_mass.astype(np.float32),
                        protected_support=protected_support,
                        agreement=agreement.astype(np.float32),
                        protected_source=protected, floor_like=floor_like,
                        active=active)
    write_vertices(out / 'candidate-unapproved.ply', result, ply, comments=[
        'UNAPPROVED dresser-protected residual-bed preview; speculative wall depth.'])
    selected_tensor = torch.as_tensor(selected_ids, device='cuda')
    final_alpha = base_alpha.clone()
    final_alpha[selected_tensor] = torch.sigmoid(torch.tensor(
        a.opacity_logit, device='cuda'))
    validation = {}
    for view in a.review_views:
        camera = cameras[view]
        height = round(camera['height'] * a.width / camera['width'])
        size = (a.width, height)
        raster = GaussianRasterizer(camera_settings(camera, height, a.width)
                                    ._replace(sh_degree=3))
        with torch.no_grad():
            current_rgb = render_scene(model, raster, original_alpha)
            restored_rgb = render_scene(model, raster, base_alpha)
            final_rgb = render_scene(model, raster, final_alpha)
        Image.fromarray(current_rgb).save(out / f'{view}-current.png')
        Image.fromarray(restored_rgb).save(out / f'{view}-restored.png')
        Image.fromarray(final_rgb).save(out / f'{view}-after.png')
        dresser = load_mask(dresser_manifest, view, size)
        carpet = load_mask(carpet_manifest, view, size)
        masks = {'dresser': dresser, 'carpet': carpet}
        for name, mask in masks.items():
            delta = np.abs(final_rgb.astype(float) - current_rgb.astype(float)).mean(axis=2) / 255
            restoration_delta = np.abs(restored_rgb.astype(float) - current_rgb.astype(float)).mean(axis=2) / 255
            masks[name] = {'mean_abs_change': float(delta[mask].mean()) if mask.any() else None,
                           'restoration_change': float(restoration_delta[mask].mean()) if mask.any() else None}
        validation[view] = {
            'dark_pixels_current': int((current_rgb.mean(axis=2) < a.dark_threshold).sum()),
            'dark_pixels_restored': int((restored_rgb.mean(axis=2) < a.dark_threshold).sum()),
            'dark_pixels_after': int((final_rgb.mean(axis=2) < a.dark_threshold).sum()),
            **masks,
        }
    report = {
        'seed': str(seed_dir), 'candidate': str(out / 'candidate-unapproved.ply'),
        'pre_aggressive': str(Path(a.pre_aggressive).resolve()),
        'training_views': views, 'holdout_views': a.holdout_views,
        'review_views': a.review_views,
        'restored_protected': int(restoration.sum()),
        'newly_attenuated': len(selected_ids),
        'newly_attenuated_protected': int(protected[selected_ids].sum()),
        'newly_attenuated_floor': int(floor_like[selected_ids].sum()),
        'newly_attenuated_source_ids': retained_ids[selected_ids].tolist(),
        'source_count': source_count, 'wall_count': len(result)-source_count,
        'wall_records_unchanged': True, 'semantic_dimensions': sum(
            key.startswith('semantic_') for key in current.dtype.names),
        'validation': validation, 'parameters': vars(a), 'approved': False,
        'warning': 'Residual masks are pseudo-labels; wall depth and hidden texture remain hypothetical.',
        'elapsed_seconds': time.perf_counter() - started,
        'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in (
        'restored_protected', 'newly_attenuated', 'newly_attenuated_protected',
        'newly_attenuated_floor', 'validation', 'elapsed_seconds')}, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('seed-dir', 'pre-aggressive', 'retained-source-indices',
                'protected-source-indices', 'floor-fit', 'cameras', 'bed-manifest',
                'dresser-manifest', 'carpet-manifest', 'output-dir'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--holdout-views', nargs='+', default=[
        'frame_0134', 'frame_0141', 'frame_0143'])
    p.add_argument('--review-views', nargs='+', default=[
        'frame_0134', 'frame_0141', 'frame_0143', 'frame_0131', 'frame_0147'])
    p.add_argument('--width', type=int, default=540)
    p.add_argument('--min-training-views', type=int, default=4)
    p.add_argument('--min-view-contribution', type=float, default=.02)
    p.add_argument('--min-inside', type=float, default=.2)
    p.add_argument('--min-views', type=int, default=3)
    p.add_argument('--min-agreement', type=float, default=.9)
    p.add_argument('--max-protected-mass', type=float, default=.02)
    p.add_argument('--max-protected-support', type=int, default=0)
    p.add_argument('--floor-error-multiplier', type=float, default=2.)
    p.add_argument('--min-active-opacity', type=float, default=.02)
    p.add_argument('--max-additions', type=int, default=500)
    p.add_argument('--opacity-logit', type=float, default=-12.)
    p.add_argument('--dark-threshold', type=float, default=32.)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
