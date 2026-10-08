"""Experimental bed-residue removal without furniture-ID vetoes.

Training-view footprint evidence selects splats; held-out views are rendered
only for review. Never overwrites an existing preview or approves an edit.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image


def choose(inside, outside, support, min_inside, min_views, min_agreement):
    agreement = inside / np.maximum(inside + outside, 1e-8)
    return ((inside >= min_inside) & (support >= min_views) &
            (agreement >= min_agreement)), agreement


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.selection.refine_revealed_layers import view_contributions
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel
    from utils.ply_semantic_utils import read_vertices, write_vertices

    start = time.perf_counter()
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(Path(a.mask_manifest).read_text())
    cameras = {c['img_name']: c for c in json.loads(Path(a.cameras).read_text())}
    holdouts = set(a.holdout_views)
    views = sorted(v for v, item in manifest['views'].items()
                   if item.get('accepted') and v in cameras and v not in holdouts)
    if len(views) < a.min_views:
        raise ValueError('Insufficient training masks')
    ply, vertices = read_vertices(a.scene)
    model = GaussianModel(3, 128)
    model.load_ply(a.scene)
    for name in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                 '_scaling', '_rotation', '_semantic_feature'):
        getattr(model, name).requires_grad_(False)
    pool = torch.arange(len(vertices), device='cuda')
    opacity = model.get_opacity.detach()
    inside = np.zeros(len(vertices), np.float64)
    outside = np.zeros(len(vertices), np.float64)
    support = np.zeros(len(vertices), np.uint16)
    torch.cuda.reset_peak_memory_stats()
    for view in views:
        camera = cameras[view]
        height = round(camera['height'] * a.width / camera['width'])
        mask = np.asarray(Image.open(manifest['views'][view]['mask_path']).convert('L')
                          .resize((a.width, height), Image.Resampling.NEAREST)) > 127
        raster = GaussianRasterizer(camera_settings(camera, height, a.width)
                                    ._replace(sh_degree=3))
        pin, pout = view_contributions(model, raster, opacity, pool,
                                       torch.from_numpy(mask.astype(np.float32)).cuda())
        pin, pout = np.maximum(pin, 0), np.maximum(pout, 0)
        inside += pin
        outside += pout
        support += pin >= a.min_view_contribution
        print(json.dumps({'view': view, 'visible': int((pin > 0).sum())}), flush=True)
    selected, agreement = choose(inside, outside, support, a.min_inside,
                                  a.min_views, a.min_agreement)
    count = int(selected.sum())
    if not count or count / len(vertices) > a.max_scene_fraction:
        raise ValueError(f'Unsafe selection size {count}/{len(vertices)}')
    # The protection list is diagnostic only; it never vetoes a selection.
    source_ids = None
    protected_count = None
    if a.source and a.prior_selected and a.protected_indices:
        _, source = read_vertices(a.source)
        prior = np.load(a.prior_selected, allow_pickle=False)
        retained = np.ones(len(source), bool)
        retained[prior] = False
        source_ids = np.flatnonzero(retained)
        if len(source_ids) != len(vertices):
            raise ValueError('Current/source ID alignment mismatch')
        protected = np.load(a.protected_indices, allow_pickle=False)
        protected_count = int(np.isin(source_ids[selected], protected).sum())
    out.mkdir(parents=True)
    np.save(out / 'selected-current-indices.npy', np.flatnonzero(selected))
    if source_ids is not None:
        np.save(out / 'selected-source-indices.npy', source_ids[selected])
    np.savez_compressed(out / 'footprint-evidence.npz', inside=inside.astype(np.float32),
                        outside=outside.astype(np.float32), support=support,
                        agreement=agreement.astype(np.float32))
    result = vertices.copy()
    result['opacity'][selected] = a.opacity_logit
    write_vertices(out / 'candidate-unapproved.ply', result, ply,
                   comments=['Unapproved aggressive removal; furniture protection disabled.'])
    chosen = torch.as_tensor(np.flatnonzero(selected), device='cuda')
    after = opacity.clone()
    after[chosen] = torch.sigmoid(torch.tensor(a.opacity_logit, device='cuda'))
    for view in a.holdout_views:
        camera = cameras[view]
        height = round(camera['height'] * a.width / camera['width'])
        raster = GaussianRasterizer(camera_settings(camera, height, a.width)
                                    ._replace(sh_degree=3))
        with torch.no_grad():
            before_rgb = render_scene(model, raster, opacity)
            after_rgb = render_scene(model, raster, after)
        Image.fromarray(before_rgb).save(out / f'{view}-before.png')
        Image.fromarray(after_rgb).save(out / f'{view}-after.png')
        Image.fromarray(np.abs(before_rgb.astype(np.int16)-after_rgb.astype(np.int16))
                        .astype(np.uint8)).save(out / f'{view}-difference.png')
    report = {'source': str(Path(a.scene).resolve()), 'candidate': str(out/'candidate-unapproved.ply'),
              'training_views': views, 'holdout_views': a.holdout_views,
              'selected_splats': count, 'scene_fraction': count / len(vertices),
              'previously_protected_selected': protected_count,
              'furniture_protection_used': False, 'approved': False,
              'warning': 'Bed masks are pseudo-labels. This may erase dresser/wall; inspect held-out RGB before use.',
              'elapsed_seconds': time.perf_counter()-start,
              'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated()/1024**2}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'mask-manifest', 'cameras', 'output-dir'):
        p.add_argument('--'+key, required=True)
    for key in ('source', 'prior-selected', 'protected-indices'):
        p.add_argument('--'+key)
    p.add_argument('--holdout-views', nargs='+', default=['frame_0134','frame_0141','frame_0143'])
    p.add_argument('--width', type=int, default=540)
    p.add_argument('--min-inside', type=float, default=.15)
    p.add_argument('--min-view-contribution', type=float, default=.02)
    p.add_argument('--min-views', type=int, default=2)
    p.add_argument('--min-agreement', type=float, default=.20)
    p.add_argument('--max-scene-fraction', type=float, default=.05)
    p.add_argument('--opacity-logit', type=float, default=-12.)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
