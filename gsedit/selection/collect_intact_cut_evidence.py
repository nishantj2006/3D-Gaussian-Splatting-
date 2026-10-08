"""Fresh intact-scene, non-held-out bed footprint evidence for GaussianCut."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.selection.refine_revealed_layers import view_contributions
    from gsedit.selection.spatial_instance_removal import load_mask
    from gsedit.selection.multiview_instance import project
    from scene.gaussian_model import GaussianModel
    out=Path(a.output_dir)
    if out.exists():
        raise FileExistsError(out)
    start=time.perf_counter()
    prior=json.loads((Path(a.edit_dir)/'report.json').read_text())
    cameras={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    target=json.loads(Path(a.target_manifest).read_text())
    protection=json.loads(Path(a.protected_manifest).read_text())
    views=prior['training_views']
    assert not set(views).intersection(prior['holdout_views'])
    m=GaussianModel(3,128);m.load_ply(prior['source'])
    for k in ('_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_semantic_feature'):
        getattr(m,k).requires_grad_(False)
    pool=torch.arange(len(m.get_xyz),device='cuda')
    xyz=m.get_xyz.detach().cpu().numpy()
    ins=np.zeros(len(xyz));outs=np.zeros(len(xyz));support=np.zeros(len(xyz),np.uint16)
    silhouette=np.zeros(len(xyz),np.uint16);contradictions=np.zeros(len(xyz),np.uint16)
    torch.cuda.reset_peak_memory_stats()
    for v in views:
        c=cameras[v];h=round(c['height']*a.width/c['width']);size=(a.width,h)
        mask=load_mask(target['views'][v]['instances']['bed']['mask_path'],size)
        if protection['views'].get(v,{}).get('accepted'):
            mask &= ~load_mask(protection['views'][v]['mask_path'],size)
        r=GaussianRasterizer(camera_settings(c,h,a.width))
        inside,outside=view_contributions(m,r,m.get_opacity.detach(),pool,torch.from_numpy(mask.astype(np.float32)).cuda())
        ins+=np.maximum(inside,0);outs+=np.maximum(outside,0);support+=inside>=.02
        x,y,_,valid=project(xyz,c,mask.shape);ids=np.flatnonzero(valid)
        silhouette[ids]+=mask[y[ids],x[ids]]
        contradictions[ids]+=~mask[y[ids],x[ids]]
        print(json.dumps({'view':v,'intact_bed_contributors':int((inside>=.02).sum())}),flush=True)
    out.mkdir(parents=True)
    np.savez_compressed(out/'bed-evidence.npz',inside=ins,outside=outs,support=support,
        silhouette=silhouette,contradictions=contradictions)
    report=dict(source=prior['source'],opacity='intact original; no deletions',training_views=views,
        holdout_views=prior['holdout_views'],width=a.width,approved=False,
        elapsed_seconds=time.perf_counter()-start,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('edit-dir','cameras','target-manifest','protected-manifest','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--width',type=int,default=270)
    run(p.parse_args())
