"""Recheck graph-cut additions against revealed protected footprints, training only."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np


def ambiguous_additions(inside, support, min_inside=.05, min_views=2):
    return (inside >= min_inside) & (support >= min_views)


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.selection.refine_revealed_layers import view_contributions
    from gsedit.selection.spatial_instance_removal import load_mask
    from scene.gaussian_model import GaussianModel
    from utils.ply_semantic_utils import read_vertices,write_vertices
    start=time.perf_counter();out=Path(a.output_dir)
    if out.exists():
        raise FileExistsError(out)
    graph=Path(a.graph_dir);prior=Path(a.prior_dir)
    report=json.loads((graph/'report.json').read_text())
    header,source=read_vertices(report['source'])
    ids=np.load(graph/'selected-indices.npy')
    prior_ids=np.load(prior/'selected-indices.npy')
    additions=np.setdiff1d(ids,prior_ids)
    protection=json.loads(Path(a.protected_manifest).read_text())
    cameras={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    views=sorted(v for v,d in protection['views'].items() if d.get('accepted') and v in cameras and v not in report['holdout_views'])
    m=GaussianModel(3,128);m.load_ply(report['source'])
    for k in ('_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_semantic_feature'):
        getattr(m,k).requires_grad_(False)
    alpha=m.get_opacity.detach().clone();alpha[torch.as_tensor(prior_ids,device='cuda')]=0
    pool=torch.as_tensor(additions,device='cuda')
    total=np.zeros(len(additions));support=np.zeros(len(additions),np.uint16)
    torch.cuda.reset_peak_memory_stats()
    for v in views:
        c=cameras[v];h=round(c['height']*a.width/c['width'])
        mask=load_mask(protection['views'][v]['mask_path'],(a.width,h))
        r=GaussianRasterizer(camera_settings(c,h,a.width))
        pin,_=view_contributions(m,r,alpha,pool,torch.from_numpy(mask.astype(np.float32)).cuda())
        total+=np.maximum(pin,0);support+=pin>=.01
        print(json.dumps({'view':v,'contributing_additions':int((pin>=.01).sum())}),flush=True)
    ambiguous=additions[ambiguous_additions(total,support)]
    labels=np.load(graph/'instance-ids.npy');labels[ambiguous]=0
    selected=np.setdiff1d(ids,ambiguous)
    protected=np.union1d(np.load(graph/'protected-source-indices.npy'),ambiguous)
    retained=np.setdiff1d(np.arange(len(source)),selected)
    out.mkdir(parents=True)
    write_vertices(out/'candidate.ply',source[retained],header,['UNAPPROVED graphcut with training-only revealed furniture guard'])
    for name,data in [('selected-indices',selected),('bed-indices',np.flatnonzero(labels==1)),
        ('wooden_frame-indices',np.flatnonzero(labels==2)),('instance-ids',labels),
        ('protected-source-indices',protected),('retained-source-indices',retained),
        ('ambiguous-added-indices',ambiguous)]:
        np.save(out/f'{name}.npy',data)
    np.savez_compressed(out/'guard-evidence.npz',source_ids=additions,protected_inside=total,protected_support=support)
    np.savez_compressed(out/'deleted-records.npz',source_ids=selected,vertices=source[selected])
    for name,d in report['instances'].items():
        d['count']=int((labels==d['id']).sum())
    report.update(candidate=str((out/'candidate.ply').resolve()),approved=False,removed=len(selected),
        graph_parent=str(graph.resolve()),guard_training_views=views,ambiguous_additions_retained=len(ambiguous),
        stage='training_only_revealed_furniture_guard',parameters=vars(a),
        elapsed_seconds=time.perf_counter()-start,protected_removed=0,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('removed','ambiguous_additions_retained','elapsed_seconds','peak_rss_mb','peak_gpu_allocated_mb')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('graph-dir','prior-dir','protected-manifest','cameras','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--width',type=int,default=270)
    run(p.parse_args())
