"""Conservative third pass: 3D bed-neighbor context plus rendered residue.

The local neighbor score is calibrated against independently protected source
IDs; it is never sufficient by itself to attenuate a Gaussian. All proposed
splats must also explain bed-only residue in multiple current-scene renders.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from gsedit.selection.covariance_instance_graph import covariances


def deleted_neighbor_fraction(points, cov, tree, queries, bed, *, neighbors=32):
    """Gaussian-overlap weighted fraction of nearby deleted-bed neighbors."""
    result = np.zeros(len(queries), np.float32)
    if not len(queries):
        return result
    k = min(neighbors + 1, len(points))
    for begin in range(0, len(queries), 512):
        ids = queries[begin:begin+512]
        _, near = tree.query(points[ids], k=k, workers=-1)
        if near.ndim == 1:
            near = near[:, None]
        delta = points[ids, None, :] - points[near]
        joint = cov[ids, None] + cov[near] + np.eye(3) * 1e-10
        solved = np.linalg.solve(joint, delta[..., None])[..., 0]
        squared = np.sum(delta * solved, axis=-1)
        weight = np.exp(-.5*np.clip(squared, 0, 80))
        weight[squared > 9] = 0
        weight[near == ids[:, None]] = 0
        denominator = weight.sum(axis=1)
        result[begin:begin+len(ids)] = (
            (weight * bed[near]).sum(axis=1) / np.maximum(denominator, 1e-8))
    return result


def safe_candidates(score, learned, inside, outside, support, protected,
                    floor_like, *, threshold, min_inside=.1, min_views=2,
                    min_agreement=.85, min_probability=.5):
    ratio = inside / np.maximum(inside + outside, 1e-8)
    bed_best = (learned[:, 1] > learned[:, 0]) & (learned[:, 1] > learned[:, 2])
    return ((score >= threshold) & bed_best &
            (learned[:, 1] >= min_probability) &
            (inside >= min_inside) & (support >= min_views) &
            (ratio >= min_agreement) & ~protected & ~floor_like)


def run(a):
    from diff_gaussian_rasterization import GaussianRasterizer
    import torch
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.selection.refine_revealed_layers import view_contributions
    from scene.gaussian_model import GaussianModel
    from utils.ply_semantic_utils import read_vertices, write_vertices

    start = time.perf_counter()
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    header, source = read_vertices(a.source)
    seed_header, seed = read_vertices(a.seed)
    selected = np.load(a.selected_indices)
    bed_ids = np.load(a.bed_indices)
    protected_ids = np.load(a.protected_indices)
    learned = np.load(a.probabilities)
    if learned.shape != (len(source), 3) or not np.isfinite(learned).all():
        raise ValueError('Learned score/source mismatch')
    keep = np.ones(len(source), bool)
    keep[selected] = False
    retained = np.flatnonzero(keep)
    if len(seed) != len(retained) or seed.dtype != source.dtype:
        raise ValueError('Seed/source schema mismatch')
    source_to_seed = np.full(len(source), -1, np.int64)
    source_to_seed[retained] = np.arange(len(seed))
    protected = np.zeros(len(source), bool)
    protected[protected_ids] = True
    bed = np.zeros(len(source), bool)
    bed[bed_ids] = True
    points = np.column_stack([source[k] for k in ('x','y','z')]).astype(np.float64)
    bed_tree = cKDTree(points[bed_ids])
    distance = bed_tree.query(points[retained], workers=-1)[0]
    pool = retained[(distance <= a.radius) & ~protected[retained]]
    # Do not re-edit opacity modified by the preceding residue pass.
    pool = pool[seed['opacity'][source_to_seed[pool]] == source['opacity'][pool]]
    if not 0 < len(pool) <= a.max_pool:
        raise ValueError(f'Unsafe local pool size: {len(pool)}')
    cov = covariances(source)
    tree = cKDTree(points)
    protected_local = protected_ids[bed_tree.query(points[protected_ids], workers=-1)[0] <= a.radius]
    if len(protected_local) < a.min_negative_examples:
        raise ValueError('Too few nearby protected negatives for calibration')
    bed_reference = deleted_neighbor_fraction(points, cov, tree, bed_ids, bed,
                                               neighbors=a.neighbors)
    protected_reference = deleted_neighbor_fraction(points, cov, tree, protected_local, bed,
                                                     neighbors=a.neighbors)
    threshold = float(max(np.quantile(protected_reference, a.negative_quantile),
                          np.quantile(bed_reference, a.positive_quantile)))
    graph = deleted_neighbor_fraction(points, cov, tree, pool, bed,
                                      neighbors=a.neighbors)
    floor = json.loads(Path(a.floor_fit).read_text())
    origin = np.asarray(floor['plane_origin'], np.float64)
    normal = np.asarray(floor['plane_normal_toward_removed_object'], np.float64)
    normal /= np.linalg.norm(normal)
    floor_margin = a.floor_error_multiplier * float(floor['plane_error_q90'])
    floor_like = np.abs((points[pool]-origin) @ normal) <= floor_margin
    preselect = (graph >= threshold) & (learned[pool,1] >= a.min_probability) & (
        learned[pool,1] > np.maximum(learned[pool,0],learned[pool,2])) & ~floor_like
    test_ids = pool[preselect]
    if len(test_ids) > a.max_attribution:
        raise ValueError(f'Too many graph-supported candidates: {len(test_ids)}')
    if not len(test_ids):
        raise ValueError('No safely graph-supported candidate remains')
    manifest = json.loads(Path(a.mask_manifest).read_text())
    cams = {c['img_name']: c for c in json.loads(Path(a.cameras).read_text())}
    views = sorted(v for v,d in manifest['views'].items() if d.get('accepted')
                   and v in cams and v not in a.holdout_views)
    if len(views) < a.min_training_views:
        raise ValueError('Insufficient independent residue views')
    model = GaussianModel(3,128)
    model.load_ply(str(a.seed))
    for name in ('_xyz','_features_dc','_features_rest','_opacity',
                 '_scaling','_rotation','_semantic_feature'):
        getattr(model,name).requires_grad_(False)
    pool_gpu = torch.from_numpy(source_to_seed[test_ids]).cuda()
    alpha = model.get_opacity.detach()
    inside = np.zeros(len(test_ids))
    outside = np.zeros(len(test_ids))
    support = np.zeros(len(test_ids),np.uint16)
    torch.cuda.reset_peak_memory_stats()
    for view in views:
        c = cams[view]
        h = round(c['height']*a.width/c['width'])
        with Image.open(manifest['views'][view]['mask_path']) as image:
            mask = np.asarray(image.convert('L').resize((a.width,h),Image.Resampling.NEAREST)) > 127
        raster = GaussianRasterizer(camera_settings(c,h,a.width))
        pin,pout = view_contributions(model,raster,alpha,pool_gpu,
            torch.from_numpy(mask.astype(np.float32)).cuda())
        inside += np.maximum(pin,0)
        outside += np.maximum(pout,0)
        support += pin >= a.min_view_contribution
        print(json.dumps({'view':view,'visible_candidates':int((pin>=a.min_view_contribution).sum())}),flush=True)
    accepted = safe_candidates(graph[preselect],learned[test_ids],inside,outside,
                               support,protected[test_ids],floor_like[preselect],
                               threshold=threshold,min_inside=a.min_inside,
                               min_views=a.min_views,min_agreement=a.min_agreement,
                               min_probability=a.min_probability)
    additions = test_ids[accepted]
    if len(additions) > a.max_additions:
        raise ValueError('Third-pass addition cap exceeded')
    if protected[additions].any():
        raise AssertionError('Protected source selected')
    result = seed.copy()
    chosen = source_to_seed[additions]
    if len(chosen):
        original_alpha = 1/(1+np.exp(-result['opacity'][chosen].astype(np.float64)))
        adjusted = np.clip(original_alpha*a.opacity_gate,1e-6,1-1e-6)
        result['opacity'][chosen] = np.log(adjusted/(1-adjusted))
    out.mkdir(parents=True)
    write_vertices(out/'candidate.ply',result,seed_header,
                   ['UNAPPROVED neighbor-context bed residue attenuation'])
    np.save(out/'added-source-indices.npy',additions)
    np.savez_compressed(out/'candidate-evidence.npz',source_ids=test_ids,
                        graph_score=graph[preselect],bed_probability=learned[test_ids,1],
                        inside=inside.astype(np.float32),outside=outside.astype(np.float32),
                        supporting_views=support,accepted=accepted)
    report = {'approved':False,'source':str(Path(a.source).resolve()),
              'seed':str(Path(a.seed).resolve()),'candidate':str((out/'candidate.ply').resolve()),
              'training_views':views,'holdout_views':a.holdout_views,
              'local_pool':len(pool),'nearby_protected_negatives':len(protected_local),
              'graph_threshold':threshold,
              'positive_graph_q10':float(np.quantile(bed_reference,.1)),
              'protected_graph_q95':float(np.quantile(protected_reference,.95)),
              'graph_supported':len(test_ids),'newly_attenuated':len(additions),
              'protected_changed':0,'floor_rejected':int(floor_like.sum()),
              'floor_margin':floor_margin,'parameters':vars(a),
              'elapsed_seconds':time.perf_counter()-start,
              'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('local_pool','graph_threshold',
        'graph_supported','newly_attenuated','elapsed_seconds','peak_rss_mb',
        'peak_gpu_allocated_mb')}))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','seed','selected-indices','bed-indices','protected-indices',
                'probabilities','floor-fit','mask-manifest','cameras','output-dir'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--holdout-views',nargs='+',required=True)
    p.add_argument('--radius',type=float,default=1.)
    p.add_argument('--neighbors',type=int,default=32)
    p.add_argument('--negative-quantile',type=float,default=.95)
    p.add_argument('--positive-quantile',type=float,default=.1)
    p.add_argument('--min-negative-examples',type=int,default=100)
    p.add_argument('--min-probability',type=float,default=.5)
    p.add_argument('--floor-error-multiplier',type=float,default=2.)
    p.add_argument('--max-pool',type=int,default=15000)
    p.add_argument('--max-attribution',type=int,default=2500)
    p.add_argument('--max-additions',type=int,default=1000)
    p.add_argument('--width',type=int,default=270)
    p.add_argument('--min-training-views',type=int,default=4)
    p.add_argument('--min-view-contribution',type=float,default=.02)
    p.add_argument('--min-inside',type=float,default=.1)
    p.add_argument('--min-views',type=int,default=2)
    p.add_argument('--min-agreement',type=float,default=.85)
    p.add_argument('--opacity-gate',type=float,default=.001)
    return p


if __name__=='__main__':
    run(parser().parse_args())
