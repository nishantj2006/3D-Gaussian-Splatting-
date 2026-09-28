"""Preview-only seeded 3D instance grouping with hard protected-object barriers.

All coordinates and appearance come from the scene. A volume only limits the
search; membership still requires multi-view silhouette/render evidence. Source
IDs are stored separately, keeping the original PLY schema and features intact.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def seeded_group(points, eligible, seeds, protected, *, radius, neighbors=24):
    eligible = np.asarray(eligible, bool) & ~np.asarray(protected, bool)
    ids = np.flatnonzero(eligible)
    chosen = np.zeros(len(points), bool)
    if not len(ids):
        return chosen
    distance, neighbor = cKDTree(points[ids]).query(
        points[ids], k=min(neighbors, len(ids)), workers=-1)
    if distance.ndim == 1:
        distance, neighbor = distance[:, None], neighbor[:, None]
    row = np.broadcast_to(np.arange(len(ids))[:, None], neighbor.shape)
    valid = distance <= radius
    graph = coo_matrix((np.ones(valid.sum()), (row[valid], neighbor[valid])),
                       shape=(len(ids), len(ids))).tocsr()
    _, labels = connected_components(graph, directed=False)
    seed_labels = np.unique(labels[np.isin(ids, seeds)])
    chosen[ids] = np.isin(labels, seed_labels)
    return chosen


def occupied_envelope(points, seeds, margin=.15):
    """Robust oriented envelope, never used alone as a deletion mask."""
    center = np.median(points[seeds], axis=0)
    _, _, axes = np.linalg.svd(points[seeds]-center, full_matrices=False)
    local = (points-center) @ axes.T
    lo, hi = np.quantile(local[seeds], [.005, .995], axis=0)
    pad = np.maximum((hi-lo)*margin, 1e-6)
    inside = ((local >= lo-pad) & (local <= hi+pad)).all(axis=1)
    return inside, dict(center=center.tolist(), axes=axes.tolist(),
                        low=(lo-pad).tolist(), high=(hi+pad).tolist())


def load_mask(path, size):
    return np.asarray(Image.open(path).convert('L').resize(
        size, Image.Resampling.NEAREST)) > 127


def run(args):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.selection.multiview_instance import project
    from gsedit.selection.refine_revealed_layers import view_contributions
    from scene.gaussian_model import GaussianModel
    from utils.ply_semantic_utils import read_vertices, write_vertices
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    ply, vertices = read_vertices(args.scene)
    points = np.column_stack([vertices[k] for k in ('x','y','z')])
    n = len(points)
    cameras = {c['img_name']: c for c in json.loads(Path(args.cameras).read_text())}
    target = json.loads(Path(args.instance_manifest).read_text())
    protection = json.loads(Path(args.protected_manifest).read_text())
    names = list(target['instances'])
    seeds = {name: np.load(target['instances'][name]['selected_ids']) for name in names}
    for ids in seeds.values():
        if not len(ids) or ids.min() < 0 or ids.max() >= n:
            raise ValueError('Invalid source seed IDs')
    views = sorted(v for v,d in target['views'].items() if
        d.get('complete') and v in cameras and v not in args.holdout_views)
    if len(views) < 3:
        raise ValueError('Too few complete training views')
    model = GaussianModel(3, 128)
    model.load_ply(args.scene)
    for name in ('_xyz','_features_dc','_features_rest','_opacity',
                 '_scaling','_rotation','_semantic_feature'):
        getattr(model,name).requires_grad_(False)
    pool = torch.arange(n, device='cuda')
    original_alpha = model.get_opacity.detach()
    alpha = original_alpha.clone()
    torch.cuda.reset_peak_memory_stats()
    rasters, masks, protected_masks = {}, {}, {}
    protected = np.zeros(n, bool)
    protected_score = np.zeros(n, np.float32)
    # Protection is measured against the intact source, not an already pruned seed.
    protection_views = sorted(v for v,d in protection['views'].items() if
        d.get('accepted') and v in cameras and v not in args.holdout_views)
    if len(protection_views) < 2:
        raise ValueError('Too few independently detected protected-object views')
    for view in sorted(set(views) | set(protection_views)):
        c = cameras[view]
        h = round(c['height']*args.width/c['width'])
        size = (args.width,h)
        rasters[view] = GaussianRasterizer(camera_settings(c,h,args.width))
        pm = np.zeros((h,args.width), bool)
        if view in protection_views:
            pm = load_mask(protection['views'][view]['mask_path'],size)
            pm = cv2.dilate(pm.astype(np.uint8),np.ones((5,5),np.uint8)) > 0
            pin,pout = view_contributions(model,rasters[view],alpha,pool,
                torch.from_numpy(pm.astype(np.float32)).cuda())
            protected_score = np.maximum(protected_score,pin)
            protected |= (pin >= args.protect_contribution) & (
                pin/np.maximum(pin+pout,1e-8) >= args.protect_agreement)
        protected_masks[view] = pm
        if view in views:
            masks[view] = {name: load_mask(target['views'][view]['instances'][name]['mask_path'],size)
                & ~pm for name in names}
    print(json.dumps({'protected_source_splats':int(protected.sum()),
                      'protection_views':len(protection_views)}),flush=True)
    silhouettes, rejected, envelopes, radii = {}, {}, {}, {}
    for name in names:
        silhouette = np.zeros(n,np.uint16)
        contradiction = np.zeros(n,np.uint16)
        for view in views:
            x,y,_,valid = project(points,cameras[view],masks[view][name].shape)
            ids = np.flatnonzero(valid)
            mask = masks[view][name]
            silhouette[ids] += mask[y[ids],x[ids]]
            union = np.logical_or.reduce(list(masks[view].values()))
            outside = cv2.erode((~union).astype(np.uint8),np.ones((7,7),np.uint8)) > 0
            contradiction[ids] += outside[y[ids],x[ids]]
        clean_seed = seeds[name][~protected[seeds[name]] & (silhouette[seeds[name]]>=2)]
        if len(clean_seed)<50:
            raise ValueError(f'Too few revalidated {name} seeds')
        envelope, details = occupied_envelope(points,clean_seed,args.envelope_margin)
        spacing = cKDTree(points[clean_seed]).query(points[clean_seed],
            k=min(9,len(clean_seed)),workers=-1)[0][:,-1]
        radii[name] = max(float(np.quantile(spacing,.9))*args.radius_multiplier,1e-6)
        envelopes[name] = (envelope,details)
        silhouettes[name], rejected[name] = silhouette,contradiction
        seeds[name] = clean_seed
    labels = np.zeros(n,np.uint16)
    labels[protected] = len(names)+1
    history = []
    evidence = {}
    for iteration in range(args.layers):
        selected = np.zeros(n,bool)
        confidence = np.zeros(n,np.float32)
        for identity,name in enumerate(names,1):
            inside,outside = np.zeros(n),np.zeros(n)
            support = np.zeros(n,np.uint16)
            for view in views:
                a,b = view_contributions(model,rasters[view],alpha,pool,
                    torch.from_numpy(masks[view][name].astype(np.float32)).cuda())
                inside += np.maximum(a,0)
                outside += np.maximum(b,0)
                support += a >= args.min_view_contribution
            ratio = inside/np.maximum(inside+outside,1e-8)
            # Connected one-view proposals may grow a verified component, but
            # broad/background splats with contradictory footprint are excluded.
            eligible = envelopes[name][0] & (silhouettes[name]>=2) & (
                rejected[name] <= args.max_exterior_views) & (support>=1) & (
                inside>=args.min_total_contribution) & (ratio>=args.min_agreement)
            # Narrow occupied bands admit hidden splats, not an entire box.
            distance_to_seed = cKDTree(points[seeds[name]]).query(points,workers=-1)[0]
            occluded = (inside+outside < args.min_total_contribution) & (
                distance_to_seed <= radii[name]) & (silhouettes[name]>=2) & (
                rejected[name] <= args.max_exterior_views)
            eligible |= (occluded & envelopes[name][0]) | (labels == identity)
            anchor = np.unique(np.concatenate((seeds[name],np.flatnonzero(labels==identity))))
            group = seeded_group(points,eligible,anchor,protected,
                                 radius=radii[name],neighbors=args.neighbors)
            rank = np.where(occluded, .5, ratio)
            replace = group & (rank>confidence) & ~protected
            labels[replace] = identity
            confidence[replace] = rank[replace]
            selected |= group
            evidence[name] = dict(inside=inside.astype(np.float32),
                outside=outside.astype(np.float32),support=support,
                silhouette=silhouettes[name],contradictions=rejected[name])
        old = alpha[:,0].cpu().numpy()==0
        total = (labels>0)&(labels<=len(names))
        if total.mean()>args.max_scene_fraction:
            raise ValueError('Group exceeds scene safety fraction')
        alpha[torch.from_numpy(total).cuda()] = 0
        history.append(dict(layer=iteration+1,new=int((total & ~old).sum()),
                            total=int(total.sum())))
        print(json.dumps(history[-1]),flush=True)
        if not (total & ~old).any():
            break
    removed = np.flatnonzero((labels>0)&(labels<=len(names)))
    if np.intersect1d(removed,np.flatnonzero(protected)).size:
        raise AssertionError('Protected IDs selected')
    keep = np.ones(n,bool); keep[removed]=False
    out.mkdir(parents=True)
    write_vertices(out/'candidate.ply',vertices[keep],ply,
        ['UNAPPROVED spatial instance deletion; source preserved'])
    np.save(out/'selected-indices.npy',removed)
    np.save(out/'retained-source-indices.npy',np.flatnonzero(keep))
    np.save(out/'instance-ids.npy',labels)
    np.save(out/'protected-source-indices.npy',np.flatnonzero(protected))
    np.savez_compressed(out/'deleted-records.npz',source_ids=removed,vertices=vertices[removed])
    for identity,name in enumerate(names,1):
        np.save(out/f'{name}-indices.npy',np.flatnonzero(labels==identity))
        np.savez_compressed(out/f'{name}-evidence.npz',**evidence[name])
    report = dict(source=str(Path(args.scene).resolve()),approved=False,
        training_views=views,holdout_views=args.holdout_views,
        protection_views=protection_views,protected_source_splats=int(protected.sum()),
        protected_removed=0,removed=int(len(removed)),rounds=history,
        instances={name:dict(id=i,count=int((labels==i).sum()),
            radius=radii[name],envelope=envelopes[name][1]) for i,name in enumerate(names,1)},
        parameters=vars(args),
        source_properties=len(vertices.dtype.names),
        semantic_dimensions=sum(k.startswith('semantic_') for k in vertices.dtype.names),
        elapsed_seconds=time.perf_counter()-started,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','cameras','instance-manifest','protected-manifest','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--holdout-views',nargs='+',required=True)
    p.add_argument('--width',type=int,default=270)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--neighbors',type=int,default=24)
    p.add_argument('--radius-multiplier',type=float,default=1.5)
    p.add_argument('--envelope-margin',type=float,default=.15)
    p.add_argument('--min-view-contribution',type=float,default=.02)
    p.add_argument('--min-total-contribution',type=float,default=.05)
    p.add_argument('--min-agreement',type=float,default=.75)
    p.add_argument('--max-exterior-views',type=int,default=2)
    p.add_argument('--protect-contribution',type=float,default=.02)
    p.add_argument('--protect-agreement',type=float,default=.1)
    p.add_argument('--max-scene-fraction',type=float,default=.15)
    return p


if __name__=='__main__':
    run(parser().parse_args())
