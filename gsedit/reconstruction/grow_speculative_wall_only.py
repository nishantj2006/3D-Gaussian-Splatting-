"""Grow an unapproved wall-only Gaussian patch into multi-view dark holes.

Wall depth is a hypothesis. Existing scene splats and the reliable carpet are
preserved byte-for-byte in the output's leading PLY records.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy import ndimage

from gsedit.assets.align_asset import plane_frame
from gsedit.reconstruction.build_continuous_background import gaussian_frame, surface_gaussians
from gsedit.reconstruction.build_local_background import mask_votes
from gsedit.reconstruction.grow_speculative_wall import (
    SH_C0, cell_indices, grid_from_rays, recursive_donors, supported_grid,
    wall_photo_atlas)
from utils.ply_semantic_utils import read_vertices, write_vertices


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel

    started = time.perf_counter()
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    ply, source = read_vertices(a.scene)
    xyz = np.column_stack([source[k] for k in ('x','y','z')]).astype(float)
    cameras = {c['img_name']:c for c in json.loads(Path(a.cameras).read_text())}
    wall_entries = json.loads(Path(a.wall_manifest).read_text())
    sweep = json.loads(Path(a.sweep_report).read_text())
    floor = json.loads(Path(a.floor_fit).read_text())
    origin, frame = plane_frame(floor['plane_origin'],
                                floor['plane_normal_toward_removed_object'])
    normal2 = np.asarray(sweep['plane_normal_floor_xy'], float)
    normal2 /= np.linalg.norm(normal2)
    tangent, wall_basis = gaussian_frame(frame, normal2)
    world_tangent = frame[:, :2] @ tangent
    world_normal = frame[:, :2] @ normal2
    offset = float(sweep['best_offset'])
    train = sweep['training_views']
    per_view = {}
    hole_pixels = {}
    for view in train:
        dark = np.asarray(Image.open(Path(a.hole_masks)/f'{view}.png').convert('L')) > 127
        camera = cameras[view]
        height,width = dark.shape
        yy,xx = np.where(dark[::a.ray_stride,::a.ray_stride])
        pixels = np.column_stack((xx*a.ray_stride+.5, yy*a.ray_stride+.5))
        rays = np.column_stack(((pixels[:,0]-width/2)/(camera['fx']*width/camera['width']),
                                (pixels[:,1]-height/2)/(camera['fy']*height/camera['height']),
                                np.ones(len(pixels)))) @ np.asarray(camera['rotation']).T
        pos = np.asarray(camera['position'])
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (offset-(pos-origin) @ world_normal)/(rays @ world_normal)
            floor_depth = ((origin-pos) @ frame[:,2])/(rays @ frame[:,2])
        hit = np.isfinite(depth) & (depth>.05) & (depth<50)
        hit &= (depth<floor_depth) | ~np.isfinite(floor_depth) | (floor_depth<=0)
        points = pos+np.where(hit,depth,0)[:,None]*rays
        th = np.column_stack(((points-origin) @ world_tangent,
                              (points-origin) @ frame[:,2]))
        hit &= (th[:,1]>=0) & (th[:,1]<=6)
        per_view[view] = th[hit]
        hole_pixels[view] = int(dark.sum())
    rays = np.concatenate(list(per_view.values()))
    low, shape = grid_from_rays(rays,a.step,a.margin,a.max_cells,height_floor=True)
    core,active,votes = supported_grid(per_view,low,shape,a.step,a.min_views,a.grow_cells)
    color, known, observed = wall_photo_atlas(
        a.images,wall_entries['views'],train,cameras,origin,world_normal,offset,
        low,shape,a.step,world_tangent,frame[:,2],a.photo_stride)
    reference = np.median(color[known],axis=0)/255
    wall_votes = mask_votes(xyz,cameras,wall_entries,train)
    local = (xyz-origin) @ frame
    distance = np.abs(local[:,:2] @ normal2-offset)
    rgb = np.clip(np.column_stack([source[f'f_dc_{j}'] for j in range(3)])*SH_C0+.5,0,1)
    alpha = 1/(1+np.exp(-source['opacity']))
    candidates = np.flatnonzero((wall_votes>=2)&(local[:,2]>.5)&(alpha>.1))
    priority = distance[candidates]+np.linalg.norm(rgb[candidates]-reference,axis=1)
    confident = candidates[(distance[candidates]<=a.max_donor_plane_distance)&
                           (np.linalg.norm(rgb[candidates]-reference,axis=1)
                            <=a.max_donor_color_distance)]
    if len(confident)<a.min_donors:
        confident = candidates[np.argsort(priority)[:a.min_donors]]
    if len(confident)<3:
        raise ValueError('No viable wall Gaussian donors')
    donor_uv = np.column_stack((local[confident,:2] @ tangent,local[confident,2]))
    inherited = recursive_donors(active,low,a.step,donor_uv)
    parent = confident[inherited[active]]
    ij = np.argwhere(active)
    uv = low+(ij+.5)*a.step
    world = origin+(uv[:,0,None]*tangent+offset*normal2) @ frame[:,:2].T+(
        uv[:,1,None]*frame[:,2])
    edge = ndimage.distance_transform_edt(active)[active]
    opacity = np.clip(a.opacity*np.minimum(1,edge/a.grow_cells),.08,.95)
    fill = surface_gaussians(source,parent,world,step=a.step,basis=wall_basis,
                             opacity=opacity)
    for j in range(3):
        fill[f'f_dc_{j}'] = (color[active,j]/255-.5)/SH_C0
    candidate = np.concatenate((source,fill))
    out.mkdir(parents=True)
    Image.fromarray(color).save(out/'wall-atlas.png')
    Image.fromarray(observed).save(out/'wall-observed.png')
    Image.fromarray((known*255).astype(np.uint8)).save(out/'wall-evidence.png')
    write_vertices(out/'candidate-unapproved.ply',candidate,ply,comments=[
        'Speculative wall-only recursive donor growth; wall depth is not measured.'])
    np.savez_compressed(out/'provenance.npz',wall_parent_indices=parent,
                        wall_grid_votes=votes,wall_core=core,wall_active=active,
                        wall_grid_low=low,wall_grid_step=a.step)
    final = GaussianModel(3,128)
    final.load_ply(str(out/'candidate-unapproved.ply'))
    for key in ('_xyz','_features_dc','_features_rest','_opacity',
                '_scaling','_rotation','_semantic_feature'):
        getattr(final,key).requires_grad_(False)
    original = GaussianModel(3,128)
    original.load_ply(a.scene)
    for key in ('_xyz','_features_dc','_features_rest','_opacity',
                '_scaling','_rotation','_semantic_feature'):
        getattr(original,key).requires_grad_(False)
    validation={}
    for view in sweep['holdout_views']:
        camera=cameras[view]
        height=round(camera['height']*a.width/camera['width'])
        raster=GaussianRasterizer(camera_settings(camera,height,a.width)
                                  ._replace(sh_degree=3))
        with torch.no_grad():
            before=render_scene(original,raster,original.get_opacity)
            after=render_scene(final,raster,final.get_opacity)
        Image.fromarray(before).save(out/f'{view}-before.png')
        Image.fromarray(after).save(out/f'{view}-after.png')
        dark_before=(before.mean(axis=2)<a.dark_threshold)
        dark_after=(after.mean(axis=2)<a.dark_threshold)
        validation[view]={'dark_pixels_before':int(dark_before.sum()),
                          'dark_pixels_after':int(dark_after.sum()),
                          'mean_abs_rgb_change':float(np.abs(before.astype(float)-after).mean()/255)}
    report={'source':str(Path(a.scene).resolve()),
            'candidate':str(out/'candidate-unapproved.ply'),
            'wall_offset_hypothesis':offset,
            'depth_constrained':bool(sweep['depth_constrained']),
            'wall_seeds':len(confident),'wall_seed_indices':confident.tolist(),
            'wall_added':len(fill),'floor_added':0,
            'observed_wall_atlas_fraction':float(known.mean()),
            'hole_pixels_by_training_view':hole_pixels,
            'heldout':validation,'semantic_dimensions':sum(k.startswith('semantic_') for k in source.dtype.names),
            'source_records_unchanged':True,'approved':False,
            'warning':'Unmeasured wall depth and generated hidden appearance; inspect RGB from multiple views.',
            'elapsed_seconds':time.perf_counter()-started,
            'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('wall_seeds','wall_added','floor_added',
          'observed_wall_atlas_fraction','heldout','elapsed_seconds',
          'peak_rss_mb','peak_gpu_allocated_mb')},indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','cameras','images','wall-manifest','floor-fit','sweep-report',
                 'hole-masks','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--width',type=int,default=540)
    p.add_argument('--ray-stride',type=int,default=4)
    p.add_argument('--photo-stride',type=int,default=4)
    p.add_argument('--step',type=float,default=.075)
    p.add_argument('--margin',type=float,default=.4)
    p.add_argument('--max-cells',type=int,default=140000)
    p.add_argument('--min-views',type=int,default=2)
    p.add_argument('--grow-cells',type=int,default=4)
    p.add_argument('--max-donor-plane-distance',type=float,default=1.)
    p.add_argument('--max-donor-color-distance',type=float,default=.3)
    p.add_argument('--min-donors',type=int,default=12)
    p.add_argument('--opacity',type=float,default=.65)
    p.add_argument('--dark-threshold',type=float,default=32.)
    return p


if __name__=='__main__':
    run(parser().parse_args())
