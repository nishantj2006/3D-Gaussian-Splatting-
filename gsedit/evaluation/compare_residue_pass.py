"""Check a local residue-opacity edit against its seed in held-out RGB views."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image, ImageDraw


def fractions(first, second, mask, threshold=.08):
    if not mask.any():
        return None
    delta = np.abs(first.astype(np.float32)-second.astype(np.float32))/255
    return float((delta[mask].max(axis=1) > threshold).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('pass-dir','source','seed','selected-indices','protected-indices',
                'cameras','target-manifest','protected-manifest','floor-fit','output-dir'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--views', nargs='+', default=['frame_0134','frame_0141','frame_0143'])
    p.add_argument('--width', type=int, default=540)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_pruned_preview import render_scene
    from gsedit.rendering.render_overhead_preview import overhead_camera
    from gsedit.selection.spatial_instance_removal import load_mask
    from scene.gaussian_model import GaussianModel
    from utils.ply_semantic_utils import read_vertices

    start = time.perf_counter()
    report = json.loads((a.pass_dir/'report.json').read_text())
    if not set(a.views).issubset(report['holdout_views']):
        raise ValueError('Validation view used for attribution')
    _, source = read_vertices(a.source)
    _, seed = read_vertices(a.seed)
    _, edited = read_vertices(a.pass_dir/'candidate.ply')
    ids = np.load(a.selected_indices)
    protected = np.load(a.protected_indices)
    keep = np.ones(len(source), bool)
    keep[ids] = False
    if not np.array_equal(source[keep], seed):
        raise AssertionError('Seed does not match source minus selected IDs')
    if len(seed) != len(edited) or seed.dtype != edited.dtype:
        raise AssertionError('Candidate schema changed')
    evidence = np.load(a.pass_dir/'evidence.npz')
    changed = np.flatnonzero(seed['opacity'] != edited['opacity'])
    allowed = evidence['indices'][evidence['gate'] < 1]
    if not np.isin(changed, allowed).all():
        raise AssertionError('Opacity changed outside measured residue candidates')
    for name in seed.dtype.names:
        if name != 'opacity' and not np.array_equal(seed[name], edited[name]):
            raise AssertionError(f'{name} changed')
    source_to_candidate = np.full(len(source), -1, dtype=np.int64)
    source_to_candidate[np.flatnonzero(keep)] = np.arange(len(seed))
    protected_candidate = source_to_candidate[protected]
    if np.intersect1d(changed, protected_candidate[protected_candidate >= 0]).size:
        raise AssertionError('Protected source ID opacity changed')
    cameras = {c['img_name']: c for c in json.loads(a.cameras.read_text())}
    target = json.loads(a.target_manifest.read_text())
    protection = json.loads(a.protected_manifest.read_text())
    models = []
    for ply in (a.seed,a.pass_dir/'candidate.ply'):
        m = GaussianModel(3,128)
        m.load_ply(str(ply))
        for name in ('_xyz','_features_dc','_features_rest','_opacity',
                     '_scaling','_rotation','_semantic_feature'):
            getattr(m,name).requires_grad_(False)
        models.append(m)
    a.output_dir.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    metrics, strips = {}, []
    with torch.no_grad():
        for view in a.views:
            c = cameras[view]
            h = round(c['height']*a.width/c['width'])
            raster = GaussianRasterizer(camera_settings(c,h,a.width)._replace(sh_degree=3))
            before, after = [render_scene(m,raster,m.get_opacity) for m in models]
            bed = load_mask(target['views'][view]['instances']['bed']['mask_path'],(a.width,h))
            protected_mask = load_mask(protection['views'][view]['mask_path'],(a.width,h))
            metrics[view] = {'bed_changed':fractions(before,after,bed & ~protected_mask),
                             'protected_changed':fractions(before,after,protected_mask),
                             'outside_changed':fractions(before,after,~bed & ~protected_mask)}
            strip = Image.new('RGB',(2*a.width,h+30),'white')
            draw = ImageDraw.Draw(strip)
            for i,(label,image) in enumerate((('Before residue pass',before),('After residue pass',after))):
                strip.paste(Image.fromarray(image),(i*a.width,30))
                draw.text((i*a.width+5,6),view+' / '+label,fill='black')
                Image.fromarray(image).save(a.output_dir/f'{view}-{i}.png')
            strips.append(strip)
        floor = json.loads(a.floor_fit.read_text())
        xyz = np.column_stack([source[k][ids] for k in ('x','y','z')])
        center = np.median(xyz,axis=0)
        normal = np.asarray(floor['plane_normal_toward_removed_object'])
        for distance in (3.,5.):
            overhead = overhead_camera(center,normal,cameras['frame_0141'],distance,a.width)
            raster = GaussianRasterizer(camera_settings(overhead,a.width,a.width)._replace(sh_degree=3))
            for i,m in enumerate(models):
                Image.fromarray(render_scene(m,raster,m.get_opacity)).save(
                    a.output_dir/f'overhead-{distance:g}-{i}.png')
    composite=Image.new('RGB',(strips[0].width,sum(x.height for x in strips)),'white')
    offset=0
    for strip in strips:
        composite.paste(strip,(0,offset))
        offset+=strip.height
    composite.save(a.output_dir/'comparison.png')
    result={'approved':False,'views':metrics,'changed_opacities':len(changed),
            'protected_source_ids_changed':0,'schema_properties':len(seed.dtype.names),
            'semantic_dimensions':sum(k.startswith('semantic_') for k in seed.dtype.names),
            'all_nonopacity_fields_exact':True,
            'elapsed_seconds':time.perf_counter()-start,
            'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
            'cautions':['Image masks are pseudo-labels, not ground truth.',
                        'Changed pixels do not establish whole-object removal.']}
    (a.output_dir/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
