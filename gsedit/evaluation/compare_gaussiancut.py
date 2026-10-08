"""Held-out RGB comparison of the current split preview and GaussianCut edit."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image, ImageDraw


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings, render_selection, scores
    from gsedit.rendering.render_pruned_preview import render_scene
    from gsedit.rendering.render_overhead_preview import overhead_camera
    from gsedit.selection.spatial_instance_removal import load_mask
    from scene.gaussian_model import GaussianModel
    from utils.ply_semantic_utils import read_vertices
    out=Path(a.output_dir)
    if out.exists():
        raise FileExistsError(out)
    started=time.perf_counter()
    graph=Path(a.graph_dir);report=json.loads((graph/'report.json').read_text())
    if not report['size_gate_passed']:
        raise ValueError('Graph edit failed size gate; no promotable comparison')
    source=report['source']
    ids=np.load(graph/'selected-indices.npy');retained=np.load(graph/'retained-source-indices.npy')
    protected=np.load(graph/'protected-source-indices.npy')
    header,vertices=read_vertices(source);_,candidate=read_vertices(graph/'candidate.ply')
    exact=vertices.dtype==candidate.dtype and np.array_equal(vertices[retained],candidate)
    if not exact or np.intersect1d(ids,protected).size:
        raise AssertionError('Source/protected preservation failed')
    cameras={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    targets=json.loads(Path(a.target_manifest).read_text())
    protection=json.loads(Path(a.protected_manifest).read_text())
    baseline_dir=Path(a.baseline_dir)
    baseline_report=json.loads((baseline_dir/'report.json').read_text())
    base_deletions=np.load(Path(baseline_report['source'])/'selected-indices.npy')
    models=[]
    for path in (source,str(baseline_dir/'candidate.ply')):
        m=GaussianModel(3,128);m.load_ply(path)
        for key in ('_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_semantic_feature'):
            getattr(m,key).requires_grad_(False)
        models.append(m)
    original,baseline=models
    replacement = None
    if a.candidate_ply:
        replacement=GaussianModel(3,128);replacement.load_ply(a.candidate_ply)
        for key in ('_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_semantic_feature'):
            getattr(replacement,key).requires_grad_(False)
    edited=original.get_opacity.detach().clone();edited[torch.as_tensor(ids,device='cuda')]=0
    out.mkdir(parents=True)
    for name in ('original','current','gaussiancut'):
        (out/name).mkdir()
    metrics=dict(approved=False,heldout_views=a.views,source_properties=len(vertices.dtype.names),
        semantic_dimensions=sum(k.startswith('semantic_') for k in vertices.dtype.names),
        retained_records_exact=exact,protected_removed=0,views={},
        warnings=['Masks are unverified pseudo-labels, not object-removal ground truth.',
                  'Deleted front-footprint recall does not measure remaining bed layers.',
                  'Current preview includes calibrated children; footprint score uses its source deletion IDs only.'])
    torch.cuda.reset_peak_memory_stats()
    sheets=[]
    with torch.no_grad():
        for v in a.views:
            c=cameras[v];h=round(c['height']*a.width/c['width']);size=(a.width,h)
            r=GaussianRasterizer(camera_settings(c,h,a.width)._replace(sh_degree=3))
            images=[render_scene(original,r,original.get_opacity),
                    render_scene(baseline,r,baseline.get_opacity),
                    render_scene(replacement,r,replacement.get_opacity) if replacement is not None else render_scene(original,r,edited)]
            mask=load_mask(targets['views'][v]['instances']['bed']['mask_path'],size)
            pm=load_mask(protection['views'][v]['mask_path'],size)
            clean=mask & ~pm
            metrics['views'][v]={}
            for name,rgb,selection in zip(('original','current','gaussiancut'),images,(None,base_deletions,ids)):
                Image.fromarray(rgb).save(out/name/f'{v}.png')
                if selection is not None:
                    delta=np.abs(images[0].astype(float)-rgb.astype(float))/255
                    footprint,_=render_selection(original,r,selection)
                    metrics['views'][v][name]=dict(
                        dresser_mae=float(delta[pm].mean()),
                        dresser_changed_fraction=float((delta[pm].max(axis=1)>.08).mean()),
                        outside_changed_fraction=float((delta[~mask & ~pm].max(axis=1)>.08).mean()),
                        target_rgb_change=float(delta[clean].mean()),
                        target_black_fraction=float((rgb[clean].max(axis=1)<8).mean()),
                        front_footprint=scores(footprint,torch.from_numpy(clean).cuda(),.1))
            strip=Image.new('RGB',(3*a.width,h+28),'white');draw=ImageDraw.Draw(strip)
            for i,(name,rgb) in enumerate(zip(('Original','Current split preview','GaussianCut'),images)):
                strip.paste(Image.fromarray(rgb),(i*a.width,28));draw.text((i*a.width+8,6),v+' / '+name,fill='black')
            sheets.append(strip)
        floor=json.loads(Path(a.floor_fit).read_text())
        normal=np.asarray(floor['plane_normal_toward_removed_object'])
        target=np.median(np.column_stack([vertices[k][base_deletions] for k in ('x','y','z')]),axis=0)
        entry=overhead_camera(target,normal,cameras[a.views[1]],5.,a.width)
        r=GaussianRasterizer(camera_settings(entry,a.width,a.width)._replace(sh_degree=3))
        final_model=replacement if replacement is not None else original
        final_alpha=replacement.get_opacity if replacement is not None else edited
        for name,m,alpha in [('original',original,original.get_opacity),('current',baseline,baseline.get_opacity),('gaussiancut',final_model,final_alpha)]:
            Image.fromarray(render_scene(m,r,alpha)).save(out/name/'overhead.png')
    sheet=Image.new('RGB',(sheets[0].width,sum(s.height for s in sheets)),'white');y=0
    for strip in sheets:
        sheet.paste(strip,(0,y));y+=strip.height
    sheet.save(out/'comparison.png')
    metrics.update(elapsed_seconds=time.perf_counter()-started,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
    (out/'report.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('graph-dir','baseline-dir','cameras','target-manifest','protected-manifest','floor-fit','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--views',nargs='+',default=['frame_0134','frame_0141','frame_0143'])
    p.add_argument('--width',type=int,default=540)
    p.add_argument('--candidate-ply')
    run(p.parse_args())
