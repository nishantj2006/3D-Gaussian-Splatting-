"""Batch render a pruned scene in accepted training views for residue detection."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('ply', 'cameras', 'instance-manifest', 'output-dir'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--holdout-views', nargs='+', required=True)
    p.add_argument('--width', type=int, default=540)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel
    started = time.perf_counter()
    cams = {c['img_name']: c for c in json.loads(a.cameras.read_text())}
    manifest = json.loads(a.instance_manifest.read_text())
    views = sorted(v for v, d in manifest['views'].items() if d.get('complete')
                   and v in cams and v not in a.holdout_views)
    if len(views) < 4:
        raise ValueError('Too few complete training views')
    model = GaussianModel(3, 128)
    model.load_ply(str(a.ply))
    for key in ('_xyz','_features_dc','_features_rest','_opacity',
                '_scaling','_rotation','_semantic_feature'):
        getattr(model, key).requires_grad_(False)
    a.output_dir.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for view in views:
            c = cams[view]
            h = round(c['height'] * a.width / c['width'])
            raster = GaussianRasterizer(camera_settings(c, h, a.width)._replace(sh_degree=3))
            Image.fromarray(render_scene(model, raster, model.get_opacity)).save(
                a.output_dir / f'{view}.png')
            print(json.dumps({'rendered': view}), flush=True)
    report = {'ply': str(a.ply.resolve()), 'views': views,
              'holdout_views': a.holdout_views, 'width': a.width,
              'elapsed_seconds': time.perf_counter()-started,
              'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated()/1024**2}
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
