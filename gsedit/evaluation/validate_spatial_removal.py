"""Held-out RGB and footprint checks for an unapproved protected removal."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings, render_selection, scores
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel
    from gsedit.selection.spatial_instance_removal import load_mask
    from utils.ply_semantic_utils import read_vertices
    out=Path(a.output_dir)
    if out.exists():
        raise FileExistsError(out)
    cameras={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    target=json.loads(Path(a.target_manifest).read_text())
    protect=json.loads(Path(a.protected_manifest).read_text())
    ids=np.load(Path(a.edit_dir)/'selected-indices.npy')
    protected=np.load(Path(a.edit_dir)/'protected-source-indices.npy')
    retained=np.load(Path(a.edit_dir)/'retained-source-indices.npy')
    _,source=read_vertices(a.scene)
    _,candidate=read_vertices(Path(a.edit_dir)/'candidate.ply')
    exact=(source.dtype==candidate.dtype and np.array_equal(source[retained],candidate))
    if not exact or len(np.intersect1d(ids,protected)):
        raise AssertionError('Preservation validation failed')
    model=GaussianModel(3,128);model.load_ply(a.scene)
    for name in ('_xyz','_features_dc','_features_rest','_opacity','_scaling',
                 '_rotation','_semantic_feature'):
        getattr(model,name).requires_grad_(False)
    opacity=model.get_opacity.detach()
    edited=opacity.clone();edited[torch.from_numpy(ids).cuda()]=0
    out.mkdir(parents=True)
    report=dict(approved=False,unchanged_retained_records=exact,
                protected_removed=0,source_properties=len(source.dtype.names),views={})
    with torch.no_grad():
        for view in a.views:
            c=cameras[view];h=round(c['height']*a.width/c['width'])
            size=(a.width,h)
            raster=GaussianRasterizer(camera_settings(c,h,a.width)._replace(sh_degree=3))
            original=render_scene(model,raster,opacity)
            result=render_scene(model,raster,edited)
            Image.fromarray(original).save(out/f'{view}-original.png')
            Image.fromarray(result).save(out/f'{view}-pruned.png')
            delta=np.abs(original.astype(float)-result.astype(float))/255
            Image.fromarray(np.rint(delta*255).astype(np.uint8)).save(out/f'{view}-difference.png')
            t=load_mask(target['views'][view]['mask_path'],size)
            p=load_mask(protect['views'][view]['mask_path'],size)
            clean=t & ~p
            footprint,_=render_selection(model,raster,ids)
            metrics=scores(footprint,torch.from_numpy(clean).cuda(),.1)
            Image.fromarray((footprint.cpu().numpy()*255).astype(np.uint8)).save(out/f'{view}-removed-footprint.png')
            report['views'][view]=dict(
                dresser_mean_rgb_change=float(delta[p].mean()),
                dresser_changed_fraction=float((delta[p].max(axis=1)>.08).mean()),
                outside_changed_fraction=float((delta[~t & ~p].max(axis=1)>.08).mean()),
                bed_front_footprint=metrics,
                warning='Footprint recall is not whole-object removal accuracy.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','edit-dir','cameras','target-manifest','protected-manifest','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--views',nargs='+',required=True)
    p.add_argument('--width',type=int,default=540)
    run(p.parse_args())
