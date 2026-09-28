"""Held-out preservation benchmark for a locally split preview, never approval."""
import argparse
import json
from pathlib import Path
import time
import resource
import numpy as np
from PIL import Image


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel
    from gsedit.selection.spatial_instance_removal import load_mask
    from utils.ply_semantic_utils import read_vertices
    start=time.perf_counter();out=Path(a.output_dir)
    if out.exists():
        raise FileExistsError(out)
    trial=Path(a.split_dir);details=json.loads((trial/'report.json').read_text())
    baseline=Path(details['source'])/'candidate.ply'
    _,base=read_vertices(baseline);_,result=read_vertices(details['candidate'])
    p=np.load(trial/'provenance.npz')
    keep=np.ones(len(base),bool);keep[p['parent_base_positions']]=False
    preserved=base.dtype==result.dtype and np.array_equal(base[keep],result[:len(base)][keep])
    semantics=[k for k in base.dtype.names if k.startswith('semantic_')]
    parent_index={int(sid):i for i,sid in enumerate(p['parent_source_ids'])}
    expected=p['parent_records'][[parent_index[int(sid)] for sid in p['child_parent_source_ids']]]
    semantic_ok=all(np.array_equal(result[k][p['child_indices']],expected[k]) for k in semantics)
    if not preserved or not semantic_ok:
        raise AssertionError('Unrelated records/schema/semantic preservation failed')
    cameras={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    target=json.loads(Path(a.target_manifest).read_text())
    protect=json.loads(Path(a.protected_manifest).read_text())
    models=[]
    for path in [baseline,details['candidate']]:
        m=GaussianModel(3,128);m.load_ply(str(path))
        for key in ('_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_semantic_feature'):
            getattr(m,key).requires_grad_(False)
        models.append(m)
    out.mkdir(parents=True);(out/'baseline').mkdir();(out/'split').mkdir()
    report=dict(approved=False,unrelated_records_unchanged=preserved,
        semantic_dimensions=len(semantics),daughter_semantics_preserved=semantic_ok,views={})
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for v in a.views:
            c=cameras[v];h=round(c['height']*a.width/c['width']);size=(a.width,h)
            r=GaussianRasterizer(camera_settings(c,h,a.width)._replace(sh_degree=3))
            images=[render_scene(m,r,m.get_opacity) for m in models]
            for label,image in zip(['baseline','split'],images):
                Image.fromarray(image).save(out/label/f'{v}.png')
            delta=np.abs(images[0].astype(float)-images[1].astype(float))/255
            tm=load_mask(target['views'][v]['instances'][a.instance]['mask_path'],size)
            pm=load_mask(protect['views'][v]['mask_path'],size)
            outside=~tm & ~pm
            metric=dict(dresser_mae=float(delta[pm].mean()),
                dresser_changed_fraction=float((delta[pm].max(axis=1)>.08).mean()),
                outside_mae=float(delta[outside].mean()),
                outside_changed_fraction=float((delta[outside].max(axis=1)>.08).mean()),
                target_changed_fraction=float((delta[tm].max(axis=1)>.08).mean()))
            report['views'][v]=metric
    report['heldout_preservation_passed']=all(m['dresser_mae']<.01 and
        m['dresser_changed_fraction']<.01 and m['outside_changed_fraction']<.005
        for m in report['views'].values())
    report.update(elapsed_seconds=time.perf_counter()-start,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('split-dir','cameras','target-manifest','protected-manifest','instance','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--views',nargs='+',required=True)
    p.add_argument('--width',type=int,default=540)
    run(p.parse_args())
