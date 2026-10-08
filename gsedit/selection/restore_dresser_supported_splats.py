"""Restore only aggressive-pass splats with strong training dresser evidence."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image


def choose_restore(restored_ids, protected_mass, protected_support, bed_mass,
                   *, min_protected_mass, min_protected_views, min_protected_fraction):
    dresser_fraction = protected_mass / np.maximum(protected_mass + bed_mass, 1e-8)
    eligible = ((protected_mass >= min_protected_mass) &
                (protected_support >= min_protected_views) &
                (dresser_fraction >= min_protected_fraction))
    selected = restored_ids[eligible[restored_ids]]
    return selected, dresser_fraction


def read_mask(manifest, view, size):
    entry = manifest['views'].get(view, {})
    if not entry.get('accepted'):
        return np.zeros((size[1], size[0]), bool)
    return np.asarray(Image.open(entry['mask_path']).convert('L').resize(
        size, Image.Resampling.NEAREST)) > 127


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel
    from utils.ply_semantic_utils import read_vertices, write_vertices

    start = time.perf_counter()
    output = Path(a.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    seed = Path(a.seed_dir).resolve()
    evidence_dir = Path(a.evidence_dir).resolve()
    seed_report = json.loads((seed / 'report.json').read_text())
    evidence_report = json.loads((evidence_dir / 'report.json').read_text())
    if evidence_report['seed'] != str(seed):
        raise ValueError('Evidence was not measured on this seed')
    ply, current = read_vertices(seed / 'candidate-unapproved.ply')
    _, pre_aggressive = read_vertices(a.pre_aggressive)
    source_count = int(seed_report['source_count'])
    if len(pre_aggressive) != source_count or current.dtype != pre_aggressive.dtype:
        raise ValueError('Source/schema mismatch')
    for key in current.dtype.names:
        if key != 'opacity' and not np.array_equal(
                current[key][:source_count], pre_aggressive[key]):
            raise ValueError(f'Source alignment failed for {key}')
    with np.load(evidence_dir / 'selection-evidence.npz') as evidence:
        mass = evidence['protected_mass']
        support = evidence['protected_support']
        bed = evidence['inside']
        protected = evidence['protected_source']
    restored_ids = np.load(evidence_dir / 'restored-protected-current-indices.npy')
    if len(mass) != source_count or not protected[restored_ids].all():
        raise ValueError('Evidence/source index mismatch')
    chosen, dresser_fraction = choose_restore(
        restored_ids, mass, support, bed,
        min_protected_mass=a.min_protected_mass,
        min_protected_views=a.min_protected_views,
        min_protected_fraction=a.min_protected_fraction)
    if not 0 < len(chosen) <= a.max_restored:
        raise ValueError(f'Unsafe selective restoration count: {len(chosen)}')
    result = current.copy()
    result['opacity'][chosen] = pre_aggressive['opacity'][chosen]
    if not np.array_equal(result[source_count:], current[source_count:]):
        raise AssertionError('Wall records changed')
    output.mkdir(parents=True)
    np.save(output / 'restored-current-indices.npy', chosen)
    np.savez_compressed(output / 'restoration-evidence.npz',
                        protected_mass=mass[chosen], protected_support=support[chosen],
                        bed_mass=bed[chosen], dresser_fraction=dresser_fraction[chosen])
    candidate = output / 'candidate-unapproved.ply'
    write_vertices(candidate, result, ply, comments=[
        'UNAPPROVED selective dresser restoration; no new residual-bed deletion.'])

    model = GaussianModel(3, 128)
    model.load_ply(str(seed / 'candidate-unapproved.ply'))
    for name in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                 '_scaling', '_rotation', '_semantic_feature'):
        getattr(model, name).requires_grad_(False)
    before_alpha = model.get_opacity.detach().clone()
    after_alpha = torch.sigmoid(torch.from_numpy(result['opacity'].copy()).cuda()[:, None])
    cameras = {item['img_name']: item for item in json.loads(Path(a.cameras).read_text())}
    dresser_manifest = json.loads(Path(a.dresser_manifest).read_text())
    carpet_manifest = json.loads(Path(a.carpet_manifest).read_text())
    bed_manifest = json.loads(Path(a.diagnostic_bed_manifest).read_text())
    validation = {}
    for view in a.review_views:
        camera = cameras[view]
        height = round(camera['height'] * a.width / camera['width'])
        size = (a.width, height)
        raster = GaussianRasterizer(camera_settings(camera, height, a.width)
                                    ._replace(sh_degree=3))
        with torch.no_grad():
            before = render_scene(model, raster, before_alpha)
            after = render_scene(model, raster, after_alpha)
        Image.fromarray(before).save(output / f'{view}-before.png')
        Image.fromarray(after).save(output / f'{view}-after.png')
        delta = np.abs(after.astype(float) - before.astype(float)).mean(axis=2) / 255
        dresser = read_mask(dresser_manifest, view, size)
        carpet = read_mask(carpet_manifest, view, size)
        bed_mask = read_mask(bed_manifest, view, size)
        validation[view] = {
            'dark_pixels_before': int((before.mean(axis=2) < a.dark_threshold).sum()),
            'dark_pixels_after': int((after.mean(axis=2) < a.dark_threshold).sum()),
            'dresser_mean_abs_change': float(delta[dresser].mean()) if dresser.any() else None,
            'carpet_mean_abs_change': float(delta[carpet].mean()) if carpet.any() else None,
            'bed_diagnostic_mean_abs_change': float(delta[bed_mask].mean()) if bed_mask.any() else None,
        }
    report = {
        'seed': str(seed), 'evidence_dir': str(evidence_dir),
        'candidate': str(candidate), 'restored': len(chosen),
        'originally_aggressive_attenuated_protected': len(restored_ids),
        'newly_attenuated': 0,
        'training_evidence_views': evidence_report['training_views'],
        'review_views': a.review_views, 'wall_records_unchanged': True,
        'source_nonopacity_records_unchanged': True,
        'semantic_dimensions': sum(key.startswith('semantic_') for key in current.dtype.names),
        'validation': validation, 'parameters': vars(a), 'approved': False,
        'warning': 'Dresser mask is a proxy; residual-bed selection yielded zero safe whole-splat candidates.',
        'elapsed_seconds': time.perf_counter()-start,
        'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated()/1024**2,
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('seed-dir', 'evidence-dir', 'pre-aggressive', 'cameras',
                'dresser-manifest', 'carpet-manifest', 'diagnostic-bed-manifest',
                'output-dir'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--review-views', nargs='+', default=[
        'frame_0134', 'frame_0141', 'frame_0143', 'frame_0131', 'frame_0147'])
    p.add_argument('--min-protected-mass', type=float, default=1.)
    p.add_argument('--min-protected-views', type=int, default=3)
    p.add_argument('--min-protected-fraction', type=float, default=.5)
    p.add_argument('--max-restored', type=int, default=100)
    p.add_argument('--width', type=int, default=540)
    p.add_argument('--dark-threshold', type=float, default=32.)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
