"""Reject color outliers in an observed wall atlas, then recolor new splats.

This changes only appended Gaussian DC colors in a new unapproved preview.
Geometry and every source-scene record remain bit-exact.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy import ndimage

from gsedit.reconstruction.grow_speculative_wall import SH_C0
from utils.ply_semantic_utils import read_vertices, write_vertices


def robust_wall_atlas(observed, evidence, max_robust_distance=4.):
    values=observed[evidence].astype(float)
    if len(values)<100:
        raise ValueError('Too little observed wall color')
    center=np.median(values,axis=0)
    mad=np.median(np.abs(values-center),axis=0)
    sigma=np.maximum(1.4826*mad,5.)
    distance=np.sqrt((((observed.astype(float)-center)/sigma)**2).sum(axis=2))
    trusted=evidence & (distance<=max_robust_distance)
    if trusted.sum()<100 or trusted.sum()/evidence.sum()<.5:
        raise ValueError('Observed wall colors have no dominant cluster')
    nearest=ndimage.distance_transform_edt(~trusted,return_distances=False,
                                             return_indices=True)
    extended=observed[nearest[0],nearest[1]]
    smooth=ndimage.gaussian_filter(extended.astype(float),sigma=(1,1,0))
    smooth[trusted]=observed[trusted]
    return np.rint(np.clip(smooth,0,255)).astype(np.uint8),trusted,center


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel

    start=time.perf_counter()
    out=Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    prior=Path(a.prior_dir)
    report=json.loads((prior/'report.json').read_text())
    _,source=read_vertices(report['source'])
    ply,candidate=read_vertices(prior/'candidate-unapproved.ply')
    if not np.array_equal(source,candidate[:len(source)]):
        raise ValueError('Prior output modified source Gaussian records')
    observed=np.asarray(Image.open(prior/'wall-observed.png').convert('RGB'))
    evidence=np.asarray(Image.open(prior/'wall-evidence.png').convert('L'))>127
    refined,trusted,center=robust_wall_atlas(observed,evidence,a.max_robust_distance)
    with np.load(prior/'provenance.npz') as provenance:
        active=provenance['wall_active']
    if len(candidate)-len(source)!=int(active.sum()) or refined.shape[:2]!=active.shape:
        raise ValueError('Wall atlas and Gaussian grid do not align')
    result=candidate.copy()
    for j in range(3):
        result[f'f_dc_{j}'][len(source):]=(refined[active,j]/255-.5)/SH_C0
    out.mkdir(parents=True)
    Image.fromarray(refined).save(out/'wall-atlas-filtered.png')
    Image.fromarray((trusted*255).astype(np.uint8)).save(out/'wall-trusted-evidence.png')
    write_vertices(out/'candidate-unapproved.ply',result,ply,comments=[
        'Unapproved speculative wall with robust observed-color filtering.'])
    model=GaussianModel(3,128)
    model.load_ply(str(out/'candidate-unapproved.ply'))
    for key in ('_xyz','_features_dc','_features_rest','_opacity',
                '_scaling','_rotation','_semantic_feature'):
        getattr(model,key).requires_grad_(False)
    cameras={c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    validation={}
    for view in a.views:
        camera=cameras[view]
        height=round(camera['height']*a.width/camera['width'])
        raster=GaussianRasterizer(camera_settings(camera,height,a.width)
                                  ._replace(sh_degree=3))
        with torch.no_grad():
            image=render_scene(model,raster,model.get_opacity)
        Image.fromarray(image).save(out/f'{view}-after.png')
        validation[view]={'dark_pixels':int((image.mean(axis=2)<a.dark_threshold).sum())}
    final={'source':report['source'],'prior':str(prior.resolve()),
           'candidate':str(out/'candidate-unapproved.ply'),
           'wall_offset_hypothesis':report['wall_offset_hypothesis'],
           'depth_constrained':False,'wall_added':len(candidate)-len(source),
           'trusted_wall_texels':int(trusted.sum()),
           'observed_wall_texels':int(evidence.sum()),
           'dominant_wall_rgb':center.tolist(),
           'source_records_unchanged':True,
           'only_new_dc_colors_changed':True,
           'heldout':validation,'approved':False,
           'elapsed_seconds':time.perf_counter()-start,
           'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
           'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2}
    (out/'report.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(final),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior-dir',required=True)
    p.add_argument('--cameras',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--views',nargs='+',default=['frame_0134','frame_0141','frame_0143'])
    p.add_argument('--max-robust-distance',type=float,default=4.)
    p.add_argument('--width',type=int,default=540)
    p.add_argument('--dark-threshold',type=float,default=32.)
    return p


if __name__=='__main__':
    run(parser().parse_args())
