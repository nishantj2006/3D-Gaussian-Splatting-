"""Split broad mixed bed/furniture splats in an unapproved wall preview.

Training-view footprints choose parents and children. Each parent is replaced
by a calibrated local mixture, and parents without a safe child cut are
reverted. Held-out views are used only for final review.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from gsedit.reconstruction.optimize_continuous_background import render
from gsedit.selection.refine_shared_splats import split_mixture
from gsedit.selection.refine_revealed_layers import view_contributions
from utils.ply_semantic_utils import read_vertices, write_vertices
from scene.gaussian_model import GaussianModel


def choose_parents(restored_ids, inside, support, protected_mass,
                   protected_support, max_scale, *, min_bed, min_protected,
                   min_bed_views, min_protected_views, min_scale, max_parents):
    eligible = (inside[restored_ids] >= min_bed) & (protected_mass[restored_ids] >= min_protected)
    eligible &= (support[restored_ids] >= min_bed_views)
    eligible &= (protected_support[restored_ids] >= min_protected_views)
    eligible &= max_scale[restored_ids] >= min_scale
    ids = restored_ids[eligible]
    score = np.minimum(inside[ids], protected_mass[ids]) * max_scale[ids]
    return ids[np.argsort(score)[::-1][:max_parents]], score


def choose_children(inside, outside, support, protected_mass, protected_support,
                    *, min_inside, min_views, min_agreement, max_protected_mass,
                    max_protected_support):
    agreement = inside / np.maximum(inside + outside, 1e-8)
    cut = ((inside >= min_inside) & (support >= min_views) &
           (agreement >= min_agreement) &
           (protected_mass <= max_protected_mass) &
           (protected_support <= max_protected_support))
    return cut, agreement


def load_mask(manifest, view, size, *, required=True):
    entry = manifest['views'].get(view, {})
    if not entry.get('accepted'):
        if required:
            raise ValueError(f'Missing accepted mask for {view}')
        return np.zeros((size[1], size[0]), bool)
    return np.asarray(Image.open(entry['mask_path']).convert('L').resize(
        size, Image.Resampling.NEAREST)) > 127


def masked_l1(pred, target, mask):
    if mask.sum() == 0:
        return pred.new_zeros(())
    return ((pred - target).abs() * mask[None]).sum() / (3 * mask.sum())


def run(a):
    started = time.perf_counter()
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    seed = Path(a.seed_dir).resolve()
    report = json.loads((seed / 'report.json').read_text())
    evidence_dir = Path(a.evidence_dir).resolve()
    evidence_report = json.loads((evidence_dir / 'report.json').read_text())
    if evidence_report['seed'] != report['seed']:
        raise ValueError('Training evidence does not match seed lineage')
    header, base = read_vertices(seed / 'candidate-unapproved.ply')
    _, pre_aggressive = read_vertices(a.pre_aggressive)
    source_count = int(report['source_count']) if 'source_count' in report else len(pre_aggressive)
    if len(pre_aggressive) != source_count or base.dtype != pre_aggressive.dtype:
        raise ValueError('Source index/schema mismatch')
    for key in base.dtype.names:
        if key != 'opacity' and not np.array_equal(base[key][:source_count], pre_aggressive[key]):
            raise ValueError(f'Source alignment mismatch: {key}')
    with np.load(evidence_dir / 'selection-evidence.npz') as data:
        inside = data['inside']; support = data['support']
        protected_mass = data['protected_mass']; protected_support = data['protected_support']
        protected_ids = data['protected_source']
    restored_ids = np.load(evidence_dir / 'restored-protected-current-indices.npy')
    scale = np.exp(np.column_stack([pre_aggressive[f'scale_{j}'] for j in range(3)]))
    max_scale = scale.max(axis=1)
    parents, _ = choose_parents(restored_ids, inside, support,
                                protected_mass, protected_support, max_scale,
                                min_bed=a.min_parent_bed,
                                min_protected=a.min_parent_protected,
                                min_bed_views=a.min_parent_bed_views,
                                min_protected_views=a.min_parent_protected_views,
                                min_scale=a.min_parent_scale,
                                max_parents=a.max_parents)
    if len(parents) == 0 or not protected_ids[parents].all():
        raise ValueError('No safely attributed protected mixed parents')
    parts = []
    child_parents = []
    for parent in parents:
        daughters = np.asarray(pre_aggressive[parent])[None]
        for _ in range(a.split_depth):
            daughters = np.concatenate([split_mixture(row, a.scale_factor) for row in daughters])
        parts.append(daughters)
        child_parents.extend([int(parent)] * len(daughters))
    children = np.concatenate(parts)
    child_parents = np.asarray(child_parents, np.int64)
    initial = np.concatenate((base.copy(), children))
    child_ids = np.arange(len(base), len(initial))
    initial['opacity'][parents] = a.hidden_logit

    out.mkdir(parents=True)
    write_vertices(out / 'split-initial.ply', initial, header,
                   ['UNAPPROVED local mixed-splat calibration input.'])
    model = GaussianModel(3, 128)
    model.load_ply(str(out / 'split-initial.ply'))
    for name in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                 '_scaling', '_rotation', '_semantic_feature'):
        getattr(model, name).requires_grad_(False)
    base_alpha = model.get_opacity.detach()
    base_raw_alpha = model._opacity.detach()
    base_dc = model._features_dc.detach()
    base_rest = model._features_rest.detach()
    child = torch.as_tensor(child_ids, dtype=torch.long, device='cuda')
    parent_gpu = torch.as_tensor(parents, dtype=torch.long, device='cuda')
    parent_restored_alpha = torch.as_tensor(
        pre_aggressive['opacity'][parents].copy(), device='cuda')[:, None].sigmoid()
    hidden_alpha = torch.sigmoid(torch.tensor(a.hidden_logit, device='cuda'))
    teacher_alpha = base_alpha.clone()
    teacher_alpha[parent_gpu] = parent_restored_alpha
    teacher_alpha[child] = hidden_alpha
    logits = torch.nn.Parameter(base_raw_alpha[child].clone())
    color_delta = torch.nn.Parameter(torch.zeros((len(child), 3), device='cuda'))
    opt = torch.optim.Adam([logits, color_delta], lr=a.learning_rate)

    cameras = {item['img_name']: item for item in json.loads(Path(a.cameras).read_text())}
    bed_manifest = json.loads(Path(a.bed_manifest).read_text())
    dresser_manifest = json.loads(Path(a.dresser_manifest).read_text())
    carpet_manifest = json.loads(Path(a.carpet_manifest).read_text())
    heldout = set(a.holdout_views)
    training = [view for view in evidence_report['training_views']
                if view in cameras and view not in heldout]
    if len(training) < a.min_training_views:
        raise ValueError('Too few training views')
    rasters = {}; bed_masks = {}; dresser_masks = {}; carpet_masks = {}; targets = {}
    with torch.no_grad():
        for view in training:
            camera = cameras[view]
            height = round(camera['height'] * a.width / camera['width'])
            size = (a.width, height)
            bed = load_mask(bed_manifest, view, size)
            dresser = load_mask(dresser_manifest, view, size)
            carpet = load_mask(carpet_manifest, view, size)
            bed &= ~dresser & ~carpet
            raster = GaussianRasterizer(camera_settings(camera, height, a.width)
                                        ._replace(sh_degree=3))
            rasters[view] = raster
            bed_masks[view] = torch.from_numpy(bed).cuda()
            dresser_masks[view] = torch.from_numpy(dresser).cuda()
            carpet_masks[view] = torch.from_numpy(carpet).cuda()
            targets[view] = render(model, raster, model.get_features, teacher_alpha).detach()

    torch.cuda.reset_peak_memory_stats()
    history = []
    for step in range(a.calibration_steps):
        view = training[step % len(training)]
        opt.zero_grad(set_to_none=True)
        alpha = base_alpha.index_copy(0, child, torch.sigmoid(logits))
        dc = base_dc.index_copy(0, child, base_dc[child] +
                                color_delta.clamp(-a.max_dc_shift, a.max_dc_shift)[:, None, :])
        pred = render(model, rasters[view], torch.cat((dc, base_rest), dim=1), alpha)
        err = (a.outside_weight * masked_l1(pred, targets[view], ~bed_masks[view]) +
               a.dresser_weight * masked_l1(pred, targets[view], dresser_masks[view]) +
               a.carpet_weight * masked_l1(pred, targets[view], carpet_masks[view]) +
               a.bed_weight * masked_l1(pred, targets[view], bed_masks[view]))
        reg = (a.color_penalty * color_delta.square().mean() +
               a.opacity_penalty * (logits - base_raw_alpha[child]).square().mean())
        loss = err + reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            logits.clamp_(base_raw_alpha[child] - a.max_logit_shift,
                          base_raw_alpha[child] + a.max_logit_shift)
            color_delta.clamp_(-a.max_dc_shift, a.max_dc_shift)
        if (step + 1) % len(training) == 0 or step == 0:
            history.append({'step': step + 1, 'view': view,
                            'loss': float(loss.detach())})
            print(json.dumps(history[-1]), flush=True)

    calibrated = initial.copy()
    calibrated['opacity'][child_ids] = logits.detach().cpu().numpy()[:, 0]
    shifts = color_delta.detach().cpu().numpy()
    for channel in range(3):
        calibrated[f'f_dc_{channel}'][child_ids] += shifts[:, channel]
    # Footprints use the calibrated children and the intact shared wall.
    calibrated_alpha = base_alpha.index_copy(0, child, torch.sigmoid(logits.detach()))
    model._opacity.copy_(torch.from_numpy(calibrated['opacity'].copy()).cuda()[:, None])
    for channel in range(3):
        model._features_dc[child, 0, channel] += color_delta.detach()[:, channel]
    bed_inside = np.zeros(len(child_ids)); outside = np.zeros(len(child_ids))
    bed_support = np.zeros(len(child_ids), np.uint16)
    dresser_mass = np.zeros(len(child_ids)); dresser_support = np.zeros(len(child_ids), np.uint16)
    for view in training:
        raster = rasters[view]
        pin, pout = view_contributions(model, raster, calibrated_alpha, child,
                                       bed_masks[view].float())
        protect, _ = view_contributions(model, raster, calibrated_alpha, child,
                                        dresser_masks[view].float())
        pin, pout, protect = (np.maximum(item, 0) for item in (pin, pout, protect))
        bed_inside += pin; outside += pout; dresser_mass += protect
        bed_support += pin >= a.min_view_contribution
        dresser_support += protect >= a.min_view_contribution
    cut, agreement = choose_children(
        bed_inside, outside, bed_support, dresser_mass, dresser_support,
        min_inside=a.min_child_inside, min_views=a.min_child_views,
        min_agreement=a.min_child_agreement,
        max_protected_mass=a.max_child_dresser_mass,
        max_protected_support=a.max_child_dresser_views)
    active_parents = np.unique(child_parents[cut])
    result = calibrated.copy()
    for parent in parents:
        if parent not in active_parents:
            result[parent] = base[parent]
            result['opacity'][child_ids[child_parents == parent]] = a.hidden_logit
    result['opacity'][child_ids[cut]] = a.hidden_logit
    unrelated = np.ones(len(base), bool)
    unrelated[parents] = False
    if not np.array_equal(result[:len(base)][unrelated], base[unrelated]):
        raise AssertionError('Unrelated source or wall records changed')
    semantic = [key for key in base.dtype.names if key.startswith('semantic_')]
    if not all(np.array_equal(result[key][child_ids],
                              pre_aggressive[key][child_parents]) for key in semantic):
        raise AssertionError('Child semantics differ from parents')

    # Numerical preservation gate against the input preview.
    current_alpha = base_alpha.clone()
    current_alpha[parent_gpu] = torch.as_tensor(base['opacity'][parents].copy(),
                                                device='cuda')[:, None].sigmoid()
    current_alpha[child] = hidden_alpha
    result_alpha = torch.from_numpy(result['opacity'].copy()).cuda()[:, None].sigmoid()
    model._opacity.copy_(torch.from_numpy(result['opacity'].copy()).cuda()[:, None])
    training_checks = {}
    with torch.no_grad():
        for view in training:
            raster = rasters[view]
            before = render(model, raster, model.get_features, current_alpha)
            after = render(model, raster, model.get_features, result_alpha)
            dresser_l1 = float(masked_l1(after, before, dresser_masks[view]))
            carpet_l1 = float(masked_l1(after, before, carpet_masks[view]))
            exterior_l1 = float(masked_l1(after, before, ~bed_masks[view]))
            training_checks[view] = {'dresser_l1': dresser_l1,
                                     'carpet_l1': carpet_l1,
                                     'outside_bed_l1': exterior_l1}
    numerical_pass = bool(
        cut.any() and max(x['dresser_l1'] for x in training_checks.values()) <= a.max_dresser_l1
        and max(x['carpet_l1'] for x in training_checks.values()) <= a.max_carpet_l1
        and max(x['outside_bed_l1'] for x in training_checks.values()) <= a.max_outside_l1)
    candidate = out / ('candidate-unapproved.ply' if numerical_pass else 'rejected-diagnostic.ply')
    write_vertices(candidate, result, header,
                   ['UNAPPROVED protected mixed-splat split; speculative wall remains.'])
    np.savez_compressed(out / 'provenance.npz', parent_current_indices=parents,
                        parent_original_records=pre_aggressive[parents],
                        parent_input_records=base[parents], child_indices=child_ids,
                        child_parent_indices=child_parents, cut=cut,
                        bed_inside=bed_inside, outside=outside, bed_support=bed_support,
                        dresser_mass=dresser_mass, dresser_support=dresser_support,
                        agreement=agreement)

    review = {}
    for view in a.review_views:
        camera = cameras[view]
        height = round(camera['height'] * a.render_width / camera['width'])
        size = (a.render_width, height)
        raster = GaussianRasterizer(camera_settings(camera, height, a.render_width)
                                    ._replace(sh_degree=3))
        with torch.no_grad():
            before = render(model, raster, model.get_features, current_alpha).permute(1, 2, 0)
            after = render(model, raster, model.get_features, result_alpha).permute(1, 2, 0)
        before_np = np.rint(before.cpu().numpy() * 255).astype(np.uint8)
        after_np = np.rint(after.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(before_np).save(out / f'{view}-before.png')
        Image.fromarray(after_np).save(out / f'{view}-after.png')
        delta = np.abs(after_np.astype(float)-before_np.astype(float)).mean(axis=2)/255
        dresser = load_mask(dresser_manifest, view, size, required=False)
        carpet = load_mask(carpet_manifest, view, size, required=False)
        review[view] = {
            'dresser_mean_abs_change': float(delta[dresser].mean()) if dresser.any() else None,
            'carpet_mean_abs_change': float(delta[carpet].mean()) if carpet.any() else None,
            'dark_pixels_before': int((before_np.mean(axis=2) < 32).sum()),
            'dark_pixels_after': int((after_np.mean(axis=2) < 32).sum()),
        }
    final_report = {
        'seed': str(seed), 'candidate': str(candidate),
        'parent_current_indices': parents.tolist(),
        'split_parents': len(parents), 'active_parents': len(active_parents),
        'daughters': len(children), 'suppressed_daughters': int(cut.sum()),
        'numerical_preservation_passed': numerical_pass,
        'training_views': training, 'heldout_views': a.holdout_views,
        'training_preservation': training_checks, 'review': review,
        'unrelated_records_unchanged': True, 'semantic_dimensions': len(semantic),
        'approved': False, 'parameters': vars(a),
        'warning': 'Masks are proxy labels; wall geometry and hidden texture remain speculative.',
        'elapsed_seconds': time.perf_counter()-started,
        'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated()/1024**2,
    }
    (out / 'report.json').write_text(json.dumps(final_report, indent=2)+'\n')
    print(json.dumps({key: final_report[key] for key in (
        'split_parents', 'active_parents', 'daughters', 'suppressed_daughters',
        'numerical_preservation_passed', 'review', 'elapsed_seconds')}, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('seed-dir', 'evidence-dir', 'pre-aggressive', 'cameras',
                'bed-manifest', 'dresser-manifest', 'carpet-manifest', 'output-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--holdout-views', nargs='+', default=[
        'frame_0134', 'frame_0141', 'frame_0143'])
    p.add_argument('--review-views', nargs='+', default=[
        'frame_0134', 'frame_0141', 'frame_0143', 'frame_0131', 'frame_0147'])
    p.add_argument('--max-parents', type=int, default=8)
    p.add_argument('--min-parent-bed', type=float, default=30.)
    p.add_argument('--min-parent-protected', type=float, default=1.)
    p.add_argument('--min-parent-bed-views', type=int, default=3)
    p.add_argument('--min-parent-protected-views', type=int, default=2)
    p.add_argument('--min-parent-scale', type=float, default=.3)
    p.add_argument('--split-depth', type=int, default=2)
    p.add_argument('--scale-factor', type=float, default=.7)
    p.add_argument('--hidden-logit', type=float, default=-12.)
    p.add_argument('--width', type=int, default=270)
    p.add_argument('--render-width', type=int, default=540)
    p.add_argument('--min-training-views', type=int, default=4)
    p.add_argument('--calibration-steps', type=int, default=30)
    p.add_argument('--learning-rate', type=float, default=.03)
    p.add_argument('--max-logit-shift', type=float, default=1.)
    p.add_argument('--max-dc-shift', type=float, default=.08)
    p.add_argument('--color-penalty', type=float, default=.02)
    p.add_argument('--opacity-penalty', type=float, default=.002)
    p.add_argument('--outside-weight', type=float, default=2.)
    p.add_argument('--dresser-weight', type=float, default=8.)
    p.add_argument('--carpet-weight', type=float, default=8.)
    p.add_argument('--bed-weight', type=float, default=.25)
    p.add_argument('--min-view-contribution', type=float, default=.02)
    p.add_argument('--min-child-inside', type=float, default=.05)
    p.add_argument('--min-child-views', type=int, default=3)
    p.add_argument('--min-child-agreement', type=float, default=.85)
    p.add_argument('--max-child-dresser-mass', type=float, default=.02)
    p.add_argument('--max-child-dresser-views', type=int, default=0)
    p.add_argument('--max-dresser-l1', type=float, default=.01)
    p.add_argument('--max-carpet-l1', type=float, default=.01)
    p.add_argument('--max-outside-l1', type=float, default=.015)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
