"""Read-only residue attribution: component colors keep original occlusion.

No PLY is written. Residual masks are diagnostic pseudo-labels, not deletion
permission. Exact color derivatives identify source splat contributions.
"""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from PIL import Image


def categories(inside,outside,support,protected,min_inside=.02):
    ratio=inside/np.maximum(inside+outside,1e-8)
    labels=np.full(len(inside),4,np.uint8)
    visible=inside>=min_inside
    labels[visible & (ratio<.1)]=3
    labels[visible & (ratio>=.1)]=2
    labels[visible & (ratio>=.8) & (support>=2)]=1
    labels[protected]=0
    return labels,ratio


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.selection.refine_revealed_layers import view_contributions
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel
    from gsedit.selection.spatial_instance_removal import load_mask
    from utils.sh_utils import eval_sh
    started=time.perf_counter();out=Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    edit=Path(a.edit_dir)
    info=json.loads((edit/'report.json').read_text())
    cameras={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    model=GaussianModel(3,128);model.load_ply(info['source'])
    for key in ('_xyz','_features_dc','_features_rest','_opacity','_scaling',
                '_rotation','_semantic_feature'):
        getattr(model,key).requires_grad_(False)
    n=len(model.get_xyz)
    removed=np.load(edit/'selected-indices.npy')
    protected=np.zeros(n,bool);protected[np.load(edit/'protected-source-indices.npy')]=True
    opacity=model.get_opacity.detach().clone()
    opacity[torch.from_numpy(removed).cuda()]=0
    pool=torch.arange(n,device='cuda')
    rasters={};shapes={}
    for view in a.views:
        c=cameras[view];h=round(c['height']*a.width/c['width'])
        shapes[view]=(a.width,h)
        rasters[view]=GaussianRasterizer(camera_settings(c,h,a.width)._replace(sh_degree=3))
    out.mkdir(parents=True)
    with torch.no_grad():
        for view in a.views:
            Image.fromarray(render_scene(model,rasters[view],opacity)).save(out/f'{view}.png')
    if not a.residual_manifest:
        print(json.dumps({'renders':str(out),'no_scene_edit':True}))
        return
    masks=json.loads(Path(a.residual_manifest).read_text())
    usable=[v for v in a.views if masks['views'].get(v,{}).get('accepted')]
    if not usable:
        raise ValueError('No detected residual masks')
    inside,outside=np.zeros(n),np.zeros(n)
    support=np.zeros(n,np.uint16)
    perview={};targets={}
    torch.cuda.reset_peak_memory_stats()
    for view in usable:
        mask=load_mask(masks['views'][view]['mask_path'],shapes[view])
        targets[view]=mask
        x,y=view_contributions(model,rasters[view],opacity,pool,
            torch.from_numpy(mask.astype(np.float32)).cuda())
        x,y=np.maximum(x,0),np.maximum(y,0)
        inside+=x;outside+=y;support+=x>=.02
        perview[view]=(x,y)
    group,ratio=categories(inside,outside,support,protected)
    names={0:'protected_furniture',1:'bed_local_footprint',2:'mixed_footprint',
           3:'mostly_exterior_footprint',4:'low_contribution'}
    report=dict(source=info['source'],edit_dir=str(edit.resolve()),
        diagnostic_only=True,no_ply_written=True,
        warning='Residual masks are pseudo-labels. Pixel footprint is not object identity.',
        accepted_views=usable,categories={},views={})
    xyz=model.get_xyz.detach().cpu().numpy()
    scales=model.get_scaling.detach().cpu().numpy()
    from scipy.spatial import cKDTree
    bed_ids=np.load(edit/'bed-indices.npy')
    nearest=cKDTree(xyz[bed_ids]).query(xyz,workers=-1)[0]
    for identity,name in names.items():
        hit=group==identity
        report['categories'][name]=dict(splats=int((hit&(inside>=.02)).sum()),
            residual_footprint_share=float(inside[hit].sum()/max(inside.sum(),1e-8)))
    order=np.argsort(inside)[::-1]
    mass=np.cumsum(inside[order])/max(inside.sum(),1e-8)
    report['splats_covering_90_percent_residual_contribution']=int(np.searchsorted(mass,.9)+1)
    report['top_contributors']=[dict(source_id=int(i),inside=float(inside[i]),
        outside=float(outside[i]),agreement=float(ratio[i]),support_views=int(support[i]),
        category=names[int(group[i])],xyz=xyz[i].tolist(),scale=scales[i].tolist(),
        distance_to_removed_bed=float(nearest[i]))
        for i in order[:30]]
    with torch.no_grad():
        for view in usable:
            camera=torch.tensor(cameras[view]['position'],device='cuda')
            direction=model.get_xyz-camera
            direction=direction/direction.norm(dim=1,keepdim=True).clamp_min(1e-8)
            colors=(eval_sh(3,model.get_features.transpose(1,2),direction)+.5).clamp_min(0)
            components={};total=None
            for identity,name in names.items():
                c=colors*torch.from_numpy((group==identity).astype(np.float32)).cuda()[:,None]
                rgb,_,_,_=rasters[view](means3D=model.get_xyz,
                    means2D=torch.zeros_like(model.get_xyz),shs=None,colors_precomp=c,
                    semantic_feature=model.get_semantic_feature,opacities=opacity,
                    scales=model.get_scaling,rotations=model.get_rotation,cov3D_precomp=None)
                total=rgb if total is None else total+rgb
                components[name]=float(rgb[:,torch.from_numpy(targets[view]).cuda()].sum().cpu())
                Image.fromarray((rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(
                    out/f'{view}-{name}.png')
            raw,_,_,_=rasters[view](means3D=model.get_xyz,
                means2D=torch.zeros_like(model.get_xyz),shs=model.get_features,colors_precomp=None,
                semantic_feature=model.get_semantic_feature,opacities=opacity,
                scales=model.get_scaling,rotations=model.get_rotation,cov3D_precomp=None)
            shares={name:value/max(sum(components.values()),1e-8) for name,value in components.items()}
            from gsedit.selection.multiview_instance import project
            px,py,_,valid=project(xyz,cameras[view],targets[view].shape)
            center_inside=np.zeros(n,bool)
            valid_ids=np.flatnonzero(valid)
            center_inside[valid_ids]=targets[view][py[valid_ids],px[valid_ids]]
            outside_center_share=float(perview[view][0][~center_inside].sum()/
                max(perview[view][0].sum(),1e-8))
            report['views'][view]=dict(contribution_from_centers_outside_mask=outside_center_share,
                footprint_shares={name:float(
                perview[view][0][group==identity].sum()/max(perview[view][0].sum(),1e-8))
                for identity,name in names.items()},rgb_mass_shares=shares,
                additive_rgb_max_error=float((total-raw).abs().max().cpu()))
    np.savez_compressed(out/'evidence.npz',source_ids=np.arange(n),inside=inside,
        outside=outside,support=support,category=group,removed=removed)
    report.update(elapsed_seconds=time.perf_counter()-started,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('categories','views',
        'splats_covering_90_percent_residual_contribution','elapsed_seconds')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('edit-dir','cameras','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--views',nargs='+',required=True)
    p.add_argument('--width',type=int,default=540)
    p.add_argument('--residual-manifest')
    run(p.parse_args())
