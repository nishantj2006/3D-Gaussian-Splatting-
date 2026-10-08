"""Read-only one-splat-at-a-time RGB ablation of ranked residue contributors."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image


def region_effect(before, after, mask):
    delta=np.abs(before.astype(np.float32)-after.astype(np.float32))/255
    return {'sum_abs_rgb':float(delta[mask].sum()),
            'mean_abs_rgb':float(delta[mask].mean()) if mask.any() else 0.,
            'changed_pixels_gt_0_08':int((delta[mask].max(axis=1)>.08).sum()) if mask.any() else 0}


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_pruned_preview import render_scene
    from gsedit.selection.spatial_instance_removal import load_mask
    from scene.gaussian_model import GaussianModel
    from utils.ply_semantic_utils import read_vertices

    started=time.perf_counter();out=Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    diagnosis=json.loads((Path(a.diagnosis_dir)/'report.json').read_text())
    cases=diagnosis['top_contributors'][:a.cases]
    if not cases:
        raise ValueError('No contributor cases')
    _,source=read_vertices(a.source)
    selected=np.load(a.selected_indices)
    kept=np.ones(len(source),bool);kept[selected]=False
    mapping=np.full(len(source),-1,np.int64)
    mapping[np.flatnonzero(kept)]=np.arange(kept.sum())
    model=GaussianModel(3,128);model.load_ply(str(a.current))
    for key in ('_xyz','_features_dc','_features_rest','_opacity',
                '_scaling','_rotation','_semantic_feature'):
        getattr(model,key).requires_grad_(False)
    alpha=model.get_opacity.detach()
    cams={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    masks=json.loads(Path(a.residual_manifest).read_text())
    protected_masks=json.loads(Path(a.protected_manifest).read_text())
    out.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    results={int(c['source_id']):{'source_id':int(c['source_id']),
        'primary_reason':c['primary_reason'],'all_reasons':c['all_reasons'],
        'per_view':{}} for c in cases}
    with torch.no_grad():
        for view in diagnosis['views']:
            c=cams[view];h=round(c['height']*a.width/c['width'])
            raster=GaussianRasterizer(camera_settings(c,h,a.width)._replace(sh_degree=3))
            before=render_scene(model,raster,alpha)
            target=load_mask(masks['views'][view]['mask_path'],(a.width,h))
            protected=load_mask(protected_masks['views'][view]['mask_path'],(a.width,h))
            regions={'safe_bed_residue':target & ~protected,
                     'protected_furniture':protected,
                     'outside_bed_and_protection':~target & ~protected}
            for case in cases:
                sid=int(case['source_id']);idx=mapping[sid]
                if idx<0:
                    raise AssertionError('Ranked contributor was deleted')
                changed_alpha=alpha.clone();changed_alpha[idx]=0
                after=render_scene(model,raster,changed_alpha)
                effects={name:region_effect(before,after,mask) for name,mask in regions.items()}
                mass=sum(e['sum_abs_rgb'] for e in effects.values())
                effects['safe_residue_effect_share']=effects['safe_bed_residue']['sum_abs_rgb']/max(mass,1e-8)
                effects['protected_effect_share']=effects['protected_furniture']['sum_abs_rgb']/max(mass,1e-8)
                results[sid]['per_view'][view]=effects
                if case['rank']<=a.save_top:
                    delta=np.abs(before.astype(np.int16)-after.astype(np.int16))
                    Image.fromarray(np.clip(delta*5,0,255).astype(np.uint8)).save(
                        out/f"{view}-rank{case['rank']:02d}-source{sid}-rgb-delta.png")
            print(json.dumps({'view':view,'cases':len(cases)}),flush=True)
    summary=[]
    for case in cases:
        sid=int(case['source_id']);entry=results[sid]
        inside=sum(v['safe_bed_residue']['sum_abs_rgb'] for v in entry['per_view'].values())
        outside=sum(v['outside_bed_and_protection']['sum_abs_rgb'] for v in entry['per_view'].values())
        furniture=sum(v['protected_furniture']['sum_abs_rgb'] for v in entry['per_view'].values())
        entry.update(rank=int(case['rank']),safe_bed_rgb_mass=inside,
                     outside_rgb_mass=outside,protected_rgb_mass=furniture,
                     safe_bed_effect_share=inside/max(inside+outside+furniture,1e-8))
        summary.append(entry)
    report={'diagnostic_only':True,'no_ply_written':True,'cases':summary,
            'elapsed_seconds':time.perf_counter()-started,
            'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
            'warning':'Ablation measures RGB influence, not semantic object identity or safe deletion.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(summary),'elapsed_seconds':report['elapsed_seconds']}))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('diagnosis-dir','source','current','selected-indices','cameras',
                'residual-manifest','protected-manifest','output-dir'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--cases',type=int,default=20)
    p.add_argument('--save-top',type=int,default=8)
    p.add_argument('--width',type=int,default=540)
    return p


if __name__=='__main__':
    run(parser().parse_args())
