"""Validate an opacity-only neighbor-context edit against its edited seed."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image, ImageDraw


def changed_fraction(before, after, mask, threshold=.08):
    if not mask.any():
        return None
    delta=np.abs(before.astype(np.float32)-after.astype(np.float32))/255
    return float((delta[mask].max(axis=1)>threshold).mean())


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_overhead_preview import overhead_camera
    from gsedit.rendering.render_pruned_preview import render_scene
    from gsedit.selection.spatial_instance_removal import load_mask
    from scene.gaussian_model import GaussianModel
    from utils.ply_semantic_utils import read_vertices

    out=Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    start=time.perf_counter()
    report=json.loads((Path(a.pass_dir)/'report.json').read_text())
    if not set(a.views).issubset(report['holdout_views']):
        raise ValueError('Attribution used a validation view')
    _,source=read_vertices(a.source)
    _,seed=read_vertices(a.seed)
    _,edited=read_vertices(Path(a.pass_dir)/'candidate.ply')
    selected=np.load(a.selected_indices)
    protected=np.load(a.protected_indices)
    added=np.load(Path(a.pass_dir)/'added-source-indices.npy')
    keep=np.ones(len(source),bool);keep[selected]=False
    retained=np.flatnonzero(keep)
    if len(seed)!=len(retained) or seed.dtype!=source.dtype or edited.dtype!=seed.dtype:
        raise AssertionError('Source/seed/candidate schema mismatch')
    for name in source.dtype.names:
        if name!='opacity' and (not np.array_equal(source[name][retained],seed[name]) or
                                not np.array_equal(seed[name],edited[name])):
            raise AssertionError(f'Unrelated property changed: {name}')
    source_to_seed=np.full(len(source),-1,np.int64)
    source_to_seed[retained]=np.arange(len(seed))
    changed=np.flatnonzero(seed['opacity']!=edited['opacity'])
    if not np.array_equal(np.sort(changed),np.sort(source_to_seed[added])):
        raise AssertionError('Changed opacity does not match attributed source IDs')
    if np.intersect1d(changed,source_to_seed[protected]).size:
        raise AssertionError('Protected source opacity changed')
    cams={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    target=json.loads(Path(a.target_manifest).read_text())
    protect=json.loads(Path(a.protected_manifest).read_text())
    models=[]
    for path in (a.seed,Path(a.pass_dir)/'candidate.ply'):
        model=GaussianModel(3,128);model.load_ply(str(path))
        for name in ('_xyz','_features_dc','_features_rest','_opacity',
                     '_scaling','_rotation','_semantic_feature'):
            getattr(model,name).requires_grad_(False)
        models.append(model)
    out.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    metrics={};strips=[]
    with torch.no_grad():
        for view in a.views:
            c=cams[view];h=round(c['height']*a.width/c['width'])
            raster=GaussianRasterizer(camera_settings(c,h,a.width)._replace(sh_degree=3))
            before,after=[render_scene(m,raster,m.get_opacity) for m in models]
            bed=load_mask(target['views'][view]['instances']['bed']['mask_path'],(a.width,h))
            furniture=load_mask(protect['views'][view]['mask_path'],(a.width,h))
            metrics[view]={'bed_changed':changed_fraction(before,after,bed & ~furniture),
                           'protected_changed':changed_fraction(before,after,furniture),
                           'outside_changed':changed_fraction(before,after,~bed & ~furniture)}
            strip=Image.new('RGB',(a.width*2,h+30),'white')
            draw=ImageDraw.Draw(strip)
            for i,(label,image) in enumerate((('Strict second pass',before),('Neighbor pass',after))):
                strip.paste(Image.fromarray(image),(i*a.width,30))
                draw.text((i*a.width+5,6),view+' / '+label,fill='black')
                Image.fromarray(image).save(out/f'{view}-{i}.png')
            strips.append(strip)
        floor=json.loads(Path(a.floor_fit).read_text())
        xyz=np.column_stack([source[k][selected] for k in ('x','y','z')])
        center=np.median(xyz,axis=0)
        normal=np.asarray(floor['plane_normal_toward_removed_object'])
        for distance in (3.,5.):
            overhead=overhead_camera(center,normal,cams['frame_0141'],distance,a.width)
            raster=GaussianRasterizer(camera_settings(overhead,a.width,a.width)._replace(sh_degree=3))
            for i,m in enumerate(models):
                Image.fromarray(render_scene(m,raster,m.get_opacity)).save(
                    out/f'overhead-{distance:g}-{i}.png')
    image=Image.new('RGB',(strips[0].width,sum(s.height for s in strips)),'white')
    offset=0
    for strip in strips:
        image.paste(strip,(0,offset));offset+=strip.height
    image.save(out/'comparison.png')
    result={'approved':False,'changed_opacities':len(changed),
            'protected_source_ids_changed':0,'all_nonopacity_fields_exact':True,
            'source_properties':len(seed.dtype.names),
            'semantic_dimensions':sum(k.startswith('semantic_') for k in seed.dtype.names),
            'views':metrics,'elapsed_seconds':time.perf_counter()-start,
            'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
            'warning':'Changed-pixel masks are unverified proxies, not whole-object accuracy.'}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pass-dir','source','seed','selected-indices','protected-indices',
                'cameras','target-manifest','protected-manifest','floor-fit','output-dir'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--views',nargs='+',default=['frame_0134','frame_0141','frame_0143'])
    p.add_argument('--width',type=int,default=540)
    return p


if __name__=='__main__':
    run(parser().parse_args())
