"""Attribute current visible bed-like residue to surviving source Gaussians.

Diagnostic only: held-out renders/masks may be used here, but no mask, score,
or source ID from this command is permitted to train or authorize deletion.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


REASONS = ('protected_source', 'already_attenuated', 'outside_local_bed_group',
           'mixed_bed_background_footprint', 'weak_learned_bed_score',
           'near_verified_floor', 'single_view_evidence', 'unresolved_high_evidence')


def reason_tags(protected, prior_attenuated, distance, ratio, bed_probability,
                floor_like, support, *, radius=1., agreement=.8, probability=.5):
    tags=[]
    if protected:
        tags.append('protected_source')
    if prior_attenuated:
        tags.append('already_attenuated')
    if distance>radius:
        tags.append('outside_local_bed_group')
    if ratio<agreement:
        tags.append('mixed_bed_background_footprint')
    if bed_probability<probability:
        tags.append('weak_learned_bed_score')
    if floor_like:
        tags.append('near_verified_floor')
    if support<2:
        tags.append('single_view_evidence')
    return tags or ['unresolved_high_evidence']


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings, render_selection
    from gsedit.selection.multiview_instance import project
    from gsedit.selection.refine_revealed_layers import view_contributions
    from gsedit.selection.spatial_instance_removal import load_mask
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel
    from utils.ply_semantic_utils import read_vertices

    start=time.perf_counter();out=Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    _,source=read_vertices(a.source)
    _,current=read_vertices(a.current)
    selected=np.load(a.selected_indices)
    bed_ids=np.load(a.bed_indices)
    protected_ids=np.load(a.protected_indices)
    probabilities=np.load(a.probabilities)
    retained=np.ones(len(source),bool);retained[selected]=False
    source_ids=np.flatnonzero(retained)
    if len(current)!=len(source_ids) or current.dtype!=source.dtype:
        raise ValueError('Source/current schema mismatch')
    for name in source.dtype.names:
        if name!='opacity' and not np.array_equal(current[name],source[name][source_ids]):
            raise ValueError(f'Retained {name} differs from source')
    if probabilities.shape!=(len(source),3):
        raise ValueError('Classifier/source ID mismatch')
    original_xyz=np.column_stack([source[k] for k in ('x','y','z')])
    xyz=original_xyz[source_ids]
    distance=cKDTree(original_xyz[bed_ids]).query(xyz,workers=-1)[0]
    protected=np.isin(source_ids,protected_ids)
    prior_attenuated=current['opacity']!=source['opacity'][source_ids]
    floor=json.loads(Path(a.floor_fit).read_text())
    origin=np.asarray(floor['plane_origin'],np.float64)
    normal=np.asarray(floor['plane_normal_toward_removed_object'],np.float64)
    normal/=np.linalg.norm(normal)
    floor_distance=np.abs((xyz-origin)@normal)
    floor_like=floor_distance<=a.floor_error_multiplier*floor['plane_error_q90']
    model=GaussianModel(3,128);model.load_ply(str(a.current))
    for key in ('_xyz','_features_dc','_features_rest','_opacity',
                '_scaling','_rotation','_semantic_feature'):
        getattr(model,key).requires_grad_(False)
    cams={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    manifest=json.loads(Path(a.residual_manifest).read_text())
    views=sorted(v for v,d in manifest['views'].items() if d.get('accepted') and v in cams)
    if not views or not set(views).issubset(a.diagnostic_views):
        raise ValueError('Diagnostic masks/views mismatch')
    pool=torch.arange(len(current),device='cuda')
    alpha=model.get_opacity.detach()
    inside=np.zeros(len(current),np.float64)
    outside=np.zeros(len(current),np.float64)
    support=np.zeros(len(current),np.uint16)
    perview={}
    torch.cuda.reset_peak_memory_stats()
    out.mkdir(parents=True)
    for view in views:
        c=cams[view];h=round(c['height']*a.width/c['width'])
        mask=load_mask(manifest['views'][view]['mask_path'],(a.width,h))
        raster=GaussianRasterizer(camera_settings(c,h,a.width)._replace(sh_degree=3))
        pin,pout=view_contributions(model,raster,alpha,pool,
            torch.from_numpy(mask.astype(np.float32)).cuda())
        pin,pout=np.maximum(pin,0),np.maximum(pout,0)
        inside+=pin;outside+=pout;support+=pin>=a.min_view_contribution
        px,py,_,valid=project(xyz,c,mask.shape)
        center_inside=np.zeros(len(xyz),bool)
        valid_ids=np.flatnonzero(valid)
        center_inside[valid_ids]=mask[py[valid_ids],px[valid_ids]]
        perview[view]={'mask_pixels':int(mask.sum()),
                       'contribution_from_centers_outside_mask':float(pin[~center_inside].sum()/max(pin.sum(),1e-8)),
                       'visible_splats':int((pin>=a.min_inside).sum())}
        with torch.no_grad():
            Image.fromarray(render_scene(model,raster,alpha)).save(out/f'{view}-current.png')
        print(json.dumps({'view':view,'visible_splats':perview[view]['visible_splats']}),flush=True)
    ratio=inside/np.maximum(inside+outside,1e-8)
    prob=probabilities[source_ids,1]
    visible=inside>=a.min_inside
    primary=np.full(len(current),'not_visible',dtype=object)
    tags={}
    for idx in np.flatnonzero(visible):
        items=reason_tags(protected[idx],prior_attenuated[idx],distance[idx],ratio[idx],
            prob[idx],floor_like[idx],support[idx],radius=a.radius,
            agreement=a.agreement,probability=a.bed_probability)
        primary[idx]=items[0]
        tags[idx]=items
    categories={}
    for name in REASONS:
        hit=(primary==name)
        categories[name]={'splats':int(hit.sum()),
            'inside_contribution_share':float(inside[hit].sum()/max(inside.sum(),1e-8))}
    order=np.argsort(inside)[::-1]
    mass=np.cumsum(inside[order])/max(inside.sum(),1e-8)
    scales=np.column_stack([np.exp(current[f'scale_{i}']) for i in range(3)])
    top=[]
    for rank,idx in enumerate(order[:a.top_count],1):
        if not visible[idx]:
            break
        top.append({'rank':rank,'source_id':int(source_ids[idx]),
            'current_index':int(idx),'primary_reason':str(primary[idx]),
            'all_reasons':tags[idx], 'inside':float(inside[idx]),
            'outside':float(outside[idx]),'bed_mask_agreement':float(ratio[idx]),
            'support_views':int(support[idx]),'learned_bed_probability':float(prob[idx]),
            'learned_frame_probability':float(probabilities[source_ids[idx],2]),
            'protected_id':bool(protected[idx]),'previously_attenuated':bool(prior_attenuated[idx]),
            'distance_to_deleted_bed':float(distance[idx]),
            'floor_distance':float(floor_distance[idx]),
            'xyz':xyz[idx].tolist(),'scale':scales[idx].tolist(),
            'opacity':float(alpha[idx,0].cpu())})
    # Footprint overlays of the largest individual culprits in each view.
    with torch.no_grad():
        for view in views:
            c=cams[view];h=round(c['height']*a.width/c['width'])
            raster=GaussianRasterizer(camera_settings(c,h,a.width)._replace(sh_degree=3))
            rgb=np.asarray(Image.open(out/f'{view}-current.png')).copy()
            for entry in top[:min(a.overlay_count,len(top))]:
                footprint,_=render_selection(model,raster,np.array([entry['current_index']]))
                strength=np.clip(footprint.cpu().numpy(),0,1)
                overlay=rgb.astype(np.float32)
                overlay[...,0]=overlay[...,0]*(1-strength)+255*strength
                overlay[...,1]*=(1-.75*strength)
                overlay[...,2]*=(1-.75*strength)
                Image.fromarray(overlay.astype(np.uint8)).save(
                    out/f"{view}-rank{entry['rank']:02d}-source{entry['source_id']}.png")
    report={'diagnostic_only':True,'no_ply_written':True,'current':str(Path(a.current).resolve()),
            'source':str(Path(a.source).resolve()),'views':views,
            'mask_warning':'Rendered SAM masks are pseudo-labels; a contributing splat may be background.',
            'visible_splats':int(visible.sum()),'categories':categories,
            'splats_covering_50_percent':int(np.searchsorted(mass,.5)+1),
            'splats_covering_90_percent':int(np.searchsorted(mass,.9)+1),
            'per_view':perview,'top_contributors':top,
            'elapsed_seconds':time.perf_counter()-start,
            'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'evidence.npz',source_ids=source_ids,inside=inside.astype(np.float32),
        outside=outside.astype(np.float32),support=support,primary_reason=primary.astype(str),
        protected=protected,bed_probability=prob.astype(np.float32),
        distance_to_deleted_bed=distance.astype(np.float32))
    print(json.dumps({k:report[k] for k in ('visible_splats','categories',
        'splats_covering_50_percent','splats_covering_90_percent','elapsed_seconds')}),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','current','selected-indices','bed-indices','protected-indices',
                'probabilities','floor-fit','cameras','residual-manifest','output-dir'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--diagnostic-views',nargs='+',required=True)
    p.add_argument('--width',type=int,default=540)
    p.add_argument('--min-view-contribution',type=float,default=.02)
    p.add_argument('--min-inside',type=float,default=.02)
    p.add_argument('--radius',type=float,default=1.)
    p.add_argument('--agreement',type=float,default=.8)
    p.add_argument('--bed-probability',type=float,default=.5)
    p.add_argument('--floor-error-multiplier',type=float,default=2.)
    p.add_argument('--top-count',type=int,default=50)
    p.add_argument('--overlay-count',type=int,default=8)
    return p


if __name__=='__main__':
    run(parser().parse_args())
