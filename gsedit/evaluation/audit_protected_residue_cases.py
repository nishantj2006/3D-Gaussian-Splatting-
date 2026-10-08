"""Read-only training-view audit of protected and bed footprints for top cases."""
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
    from scene.gaussian_model import GaussianModel

    start=time.perf_counter();out=Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    diagnosis=json.loads((Path(a.diagnosis_dir)/'report.json').read_text())
    cases=diagnosis['top_contributors'][:a.cases]
    cams={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    target=json.loads(Path(a.target_manifest).read_text())
    protected=json.loads(Path(a.protected_manifest).read_text())
    views=sorted(v for v,d in target['views'].items() if d.get('complete')
                 and v in cams and protected['views'].get(v,{}).get('accepted')
                 and v not in a.holdout_views)
    if len(views)<4:
        raise ValueError('Too few independent training views')
    model=GaussianModel(3,128);model.load_ply(str(a.current))
    for key in ('_xyz','_features_dc','_features_rest','_opacity',
                '_scaling','_rotation','_semantic_feature'):
        getattr(model,key).requires_grad_(False)
    pool=torch.as_tensor([c['current_index'] for c in cases],device='cuda')
    alpha=model.get_opacity.detach()
    p_in=np.zeros(len(cases));b_in=np.zeros(len(cases));p_support=np.zeros(len(cases),np.uint16)
    b_support=np.zeros(len(cases),np.uint16)
    torch.cuda.reset_peak_memory_stats()
    for view in views:
        c=cams[view];h=round(c['height']*a.width/c['width']);size=(a.width,h)
        pm=load_mask(protected['views'][view]['mask_path'],size)
        bed=load_mask(target['views'][view]['instances']['bed']['mask_path'],size)
        frame=load_mask(target['views'][view]['instances']['wooden_frame']['mask_path'],size)
        bed &= ~pm & ~frame
        raster=GaussianRasterizer(camera_settings(c,h,a.width))
        pin,_=view_contributions(model,raster,alpha,pool,
            torch.from_numpy(pm.astype(np.float32)).cuda())
        bin_,_=view_contributions(model,raster,alpha,pool,
            torch.from_numpy(bed.astype(np.float32)).cuda())
        p_in+=np.maximum(pin,0);b_in+=np.maximum(bin_,0)
        p_support+=pin>=a.min_view_contribution
        b_support+=bin_>=a.min_view_contribution
        print(json.dumps({'view':view}),flush=True)
    result=[]
    for i,case in enumerate(cases):
        result.append({'rank':case['rank'],'source_id':case['source_id'],
            'primary_reason':case['primary_reason'],
            'protected_mask_contribution':float(p_in[i]),
            'bed_only_mask_contribution':float(b_in[i]),
            'protected_support_views':int(p_support[i]),
            'bed_support_views':int(b_support[i]),
            'protected_fraction_of_two_masks':float(p_in[i]/max(p_in[i]+b_in[i],1e-8))})
    out.mkdir(parents=True)
    report={'diagnostic_only':True,'no_ply_written':True,'training_views':views,
            'holdout_views':a.holdout_views,'cases':result,
            'warning':'Protected mask contribution is not ground-truth dresser identity.',
            'elapsed_seconds':time.perf_counter()-start,
            'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(result),'views':len(views),
                      'elapsed_seconds':report['elapsed_seconds']}))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('diagnosis-dir','current','cameras','target-manifest',
                'protected-manifest','output-dir'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--holdout-views',nargs='+',required=True)
    p.add_argument('--width',type=int,default=270)
    p.add_argument('--cases',type=int,default=20)
    p.add_argument('--min-view-contribution',type=float,default=.02)
    return p


if __name__=='__main__':
    run(parser().parse_args())
