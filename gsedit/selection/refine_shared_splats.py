"""Calibrate a small Gaussian mixture for shared splats before selective removal.

Unrelated records and all semantic features are frozen. Preservation failures
produce a rejected diagnostic, never an approved edit. No source is overwritten.
"""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.spatial.transform import Rotation


def split_mixture(record,factor=.7):
    """Four components preserve covariance moments and linear projected mass."""
    if not 0<factor<1:
        raise ValueError('Split scale must be between zero and one')
    s=np.exp([record[f'scale_{i}'] for i in range(3)])
    q=np.array([record[f'rot_{i}'] for i in range(4)],float)
    q/=np.linalg.norm(q)
    axes=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    broad=np.argsort(s)[-2:]
    xyz=np.array([record[k] for k in ('x','y','z')])
    alpha=1/(1+np.exp(-float(record['opacity'])))
    child_alpha=np.clip(alpha/(4*factor*factor),1e-6,.95)
    children=np.repeat(np.asarray(record)[None],4).copy()
    shift=np.sqrt(1-factor*factor)
    for i,(a,b) in enumerate([(-1,-1),(-1,1),(1,-1),(1,1)]):
        offset=shift*(a*s[broad[0]]*axes[:,broad[0]]+b*s[broad[1]]*axes[:,broad[1]])
        for j,k in enumerate(('x','y','z')):
            children[k][i]=xyz[j]+offset[j]
        for j in broad:
            children[f'scale_{j}'][i]=np.log(s[j]*factor)
        children['opacity'][i]=np.log(child_alpha/(1-child_alpha))
    return children


def choose_children(inside,outside,support,protected_max,*,agreement=.85,min_views=3):
    return (inside>=.05)&(support>=min_views)&(
        inside/np.maximum(inside+outside,1e-8)>=agreement)&(protected_max<.02)


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.selection.covariance_instance_graph import covariances,overlap_links
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.selection.refine_revealed_layers import view_contributions
    from scene.gaussian_model import GaussianModel
    from gsedit.selection.spatial_instance_removal import load_mask
    from utils.ply_semantic_utils import read_vertices,write_vertices
    from utils.sh_utils import eval_sh
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    started=time.perf_counter();out=Path(a.output_dir)
    if out.exists():
        raise FileExistsError(out)
    edit=Path(a.edit_dir);report=json.loads((edit/'report.json').read_text())
    _,source=read_vertices(report['source'])
    header,base=read_vertices(edit/'candidate.ply')
    kept=np.load(edit/'retained-source-indices.npy')
    protected=np.load(edit/'protected-source-indices.npy')
    e=np.load(Path(a.evidence_dir)/f'{a.instance}-evidence.npz')
    ratio=e['inside']/np.maximum(e['inside']+e['outside'],1e-8)
    eligible=protected[(ratio[protected]>=a.parent_agreement)&(e['support'][protected]>=3)&
                       (e['inside'][protected]>=.1)]
    xyz=np.column_stack([source[k] for k in ('x','y','z')])
    cov=covariances(source)
    anchor=np.load(edit/f'{a.instance}-indices.npy')
    link=overlap_links(xyz,cov,anchor,eligible,padding=report['instances'][a.instance]['radius'])
    eligible=eligible[link[eligible]]
    parents=eligible[np.argsort(e['inside'][eligible])[::-1][:a.max_parents]]
    if not len(parents):
        raise ValueError('No supported shared parents')
    inverse=np.full(len(source),-1,int);inverse[kept]=np.arange(len(kept))
    positions=inverse[parents]
    if (positions<0).any():
        raise ValueError('Shared protected parent missing from base')
    parts=[];parent_ids=[]
    for sid in parents:
        children=np.asarray(source[sid])[None]
        for _ in range(a.split_depth):
            children=np.concatenate([split_mixture(c,a.scale_factor) for c in children])
        parts.append(children);parent_ids.extend([int(sid)]*len(children))
    daughters=np.concatenate(parts);parent_ids=np.array(parent_ids)
    print(json.dumps({'stage':'parents_selected','parents':len(parents),'children':len(daughters)}),flush=True)
    initial=np.concatenate((base.copy(),daughters))
    initial['opacity'][positions]=np.log(1e-6/(1-1e-6))
    out.mkdir(parents=True)
    write_vertices(out/'split-initial.ply',initial,header,['UNAPPROVED calibration input'])
    model=GaussianModel(3,128);model.load_ply(str(out/'split-initial.ply'))
    teacher=GaussianModel(3,128);teacher.load_ply(str(edit/'candidate.ply'))
    for m in (model,teacher):
        for key in ('_xyz','_features_dc','_features_rest','_opacity','_scaling',
                    '_rotation','_semantic_feature'):
            getattr(m,key).requires_grad_(False)
    cameras={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    target=json.loads(Path(a.instance_manifest).read_text())
    protector=json.loads(Path(a.protected_manifest).read_text())
    views=[v for v in report['training_views'] if v not in report['holdout_views'] and
           target['views'][v].get('complete')]
    if a.max_views and len(views)>a.max_views:
        views=[views[i] for i in np.unique(np.linspace(0,len(views)-1,a.max_views).round().astype(int))]
    child_np=np.arange(len(base),len(initial));child=torch.from_numpy(child_np).cuda()
    logits=torch.nn.Parameter(model._opacity[child].detach().clone())
    color_shift=torch.nn.Parameter(torch.zeros(len(child),3,device='cuda'))
    optimizer=torch.optim.Adam([logits,color_shift],lr=a.learning_rate)
    rasters={};masks={};pmasks={};references={};colors={}
    def raster(m,r,alpha,color=None):
        return r(means3D=m.get_xyz,means2D=torch.zeros_like(m.get_xyz),
            shs=m.get_features if color is None else None,colors_precomp=color,
            semantic_feature=m.get_semantic_feature,opacities=alpha,
            scales=m.get_scaling,rotations=m.get_rotation,cov3D_precomp=None)[0]
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for v in views:
            c=cameras[v];h=round(c['height']*a.width/c['width']);size=(a.width,h)
            rasters[v]=GaussianRasterizer(camera_settings(c,h,a.width)._replace(sh_degree=3))
            pm=np.zeros((h,a.width),bool)
            if protector['views'].get(v,{}).get('accepted'):
                pm=load_mask(protector['views'][v]['mask_path'],size)
            tm=load_mask(target['views'][v]['instances'][a.instance]['mask_path'],size)&~pm
            masks[v]=torch.from_numpy(tm).cuda();pmasks[v]=torch.from_numpy(pm).cuda()
            references[v]=raster(teacher,rasters[v],teacher.get_opacity).detach().clamp(0,1)
            direction=model.get_xyz-torch.tensor(c['position'],device='cuda')
            direction/=direction.norm(dim=1,keepdim=True).clamp_min(1e-8)
            colors[v]=(eval_sh(3,model.get_features.transpose(1,2),direction)+.5).clamp_min(0)
    del teacher
    print(json.dumps({'stage':'references_cached','views':len(views)}),flush=True)
    base_alpha=model.get_opacity.detach()
    def render_fit(v):
        alpha=base_alpha.index_copy(0,child,torch.sigmoid(logits))
        color=colors[v].index_copy(0,child,(colors[v][child]+color_shift).clamp_min(0))
        return raster(model,rasters[v],alpha,color).clamp(0,1)
    def region_loss(error,mask):
        return (error*mask[None]).sum()/(3*mask.sum().clamp_min(1))
    for step in range(a.calibration_steps):
        v=views[step%len(views)]
        error=(render_fit(v)-references[v]).abs()
        loss=region_loss(error,~masks[v])+8*region_loss(error,pmasks[v])+(
            .25*region_loss(error,masks[v]))
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            color_shift.clamp_(-.05,.05)
        if step%4==0:
            print(json.dumps({'stage':'calibration','step':step,'loss':float(loss.detach().cpu())}),flush=True)
    fitted=initial.copy()
    fitted['opacity'][child_np]=logits.detach().cpu().numpy()[:,0]
    for j in range(3):
        fitted[f'f_dc_{j}'][child_np]+=color_shift.detach().cpu().numpy()[:,j]/.2820947918
    model._opacity[child]=logits.detach()
    for j in range(3):
        model._features_dc[child,0,j]+=color_shift.detach()[:,j]/.2820947918
    ins,outs=np.zeros(len(child)),np.zeros(len(child))
    support=np.zeros(len(child),np.uint16);pmax=np.zeros(len(child))
    before=[]
    with torch.no_grad():
        for v in views:
            delta=(render_fit(v)-references[v]).abs()
            before.append(float(region_loss(delta,pmasks[v]).cpu()))
    for v in views:
        x,y=view_contributions(model,rasters[v],model.get_opacity.detach(),child,masks[v].float())
        ins+=np.maximum(x,0);outs+=np.maximum(y,0);support+=x>=.02
        if pmasks[v].any():
            p,_=view_contributions(model,rasters[v],model.get_opacity.detach(),child,pmasks[v].float())
            pmax=np.maximum(pmax,p)
    cut=choose_children(ins,outs,support,pmax,agreement=a.child_agreement)
    active_parents=np.unique(parent_ids[cut])
    # No deletion evidence: revert that entire parent's approximation.
    for sid,pos in zip(parents,positions):
        if sid not in active_parents:
            fitted[pos]=base[pos]
            fitted['opacity'][child_np[parent_ids==sid]]=np.log(1e-6/(1-1e-6))
    fitted['opacity'][child_np[cut]]=np.log(1e-6/(1-1e-6))
    model._opacity.copy_(torch.from_numpy(fitted['opacity'].copy()).cuda()[:,None])
    after=[];outside_errors=[];dresser_changed=[]
    with torch.no_grad():
        for v in views:
            delta=(raster(model,rasters[v],model.get_opacity).clamp(0,1)-references[v]).abs()
            after.append(float(region_loss(delta,pmasks[v]).cpu()))
            outside_errors.append(float(region_loss(delta,~masks[v]).cpu()))
            p=pmasks[v]
            dresser_changed.append(float(((delta.max(dim=0).values>.08)&p).sum().cpu()/max(int(p.sum()),1)))
    # Baseline matching and post-cut preservation are both mandatory.
    passed=bool(cut.any() and max(before)<.01 and max(after)<.01 and
                max(dresser_changed)<.01 and max(outside_errors)<.01)
    untouched=np.ones(len(base),bool);untouched[positions]=False
    assert np.array_equal(fitted[:len(base)][untouched],base[untouched])
    semantic=[k for k in base.dtype.names if k.startswith('semantic_')]
    assert all(np.array_equal(fitted[k][child_np],source[k][parent_ids]) for k in semantic)
    candidate=out/('candidate.ply' if passed else 'rejected-diagnostic.ply')
    write_vertices(candidate,fitted,header,['UNAPPROVED targeted shared-splat split'])
    np.savez_compressed(out/'provenance.npz',parent_source_ids=parents,
        parent_base_positions=positions,parent_records=source[parents],
        child_indices=child_np,child_parent_source_ids=parent_ids,cut=cut,
        inside=ins,outside=outs,support=support,protected_max=pmax)
    result=dict(source=str(edit.resolve()),candidate=str(candidate.resolve()),
        approved=False,numerical_preservation_passed=passed,parents=len(parents),
        active_parents=len(active_parents),children=len(child),suppressed_children=int(cut.sum()),
        training_views=views,holdout_views=report['holdout_views'],parameters=vars(a),
        pre_cut_dresser_mae_max=max(before),post_cut_dresser_mae_max=max(after),
        dresser_changed_fraction_max=max(dresser_changed),outside_mae_max=max(outside_errors),
        unrelated_records_unchanged=True,semantic_dimensions=len(semantic),
        elapsed_seconds=time.perf_counter()-started,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('edit-dir','evidence-dir','instance','cameras','instance-manifest',
                 'protected-manifest','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--max-parents',type=int,default=8)
    p.add_argument('--parent-agreement',type=float,default=.65)
    p.add_argument('--child-agreement',type=float,default=.85)
    p.add_argument('--scale-factor',type=float,default=.7)
    p.add_argument('--split-depth',type=int,default=2)
    p.add_argument('--calibration-steps',type=int,default=40)
    p.add_argument('--learning-rate',type=float,default=.15)
    p.add_argument('--width',type=int,default=270)
    p.add_argument('--max-views',type=int,default=0)
    p.add_argument('--seed',type=int,default=0)
    run(p.parse_args())
