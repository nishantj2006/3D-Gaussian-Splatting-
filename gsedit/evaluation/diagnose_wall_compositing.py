"""Render retained scene, appended wall, and their composite for blur diagnosis."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel

    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    seed = Path(a.seed_dir).resolve()
    report = json.loads((seed / 'report.json').read_text())
    source_count = int(report['source_count'])
    model = GaussianModel(3, 128)
    model.load_ply(str(seed / 'candidate-unapproved.ply'))
    for key in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                '_scaling', '_rotation', '_semantic_feature'):
        getattr(model, key).requires_grad_(False)
    if not 0 < source_count < len(model.get_xyz):
        raise ValueError('Source/wall boundary mismatch')
    cameras = {item['img_name']: item for item in json.loads(Path(a.cameras).read_text())}
    alpha = model.get_opacity.detach()
    source_alpha = alpha.clone(); source_alpha[source_count:] = 0
    wall_alpha = alpha.clone(); wall_alpha[:source_count] = 0
    out.mkdir(parents=True)
    per_view = {}
    for view in a.views:
        camera = cameras[view]
        height = round(camera['height'] * a.width / camera['width'])
        raster = GaussianRasterizer(camera_settings(camera, height, a.width)
                                    ._replace(sh_degree=3))
        with torch.no_grad():
            source = render_scene(model, raster, source_alpha)
            wall = render_scene(model, raster, wall_alpha)
            combined = render_scene(model, raster, alpha)
        Image.fromarray(source).save(out / f'{view}-source-only.png')
        Image.fromarray(wall).save(out / f'{view}-wall-only.png')
        Image.fromarray(combined).save(out / f'{view}-combined.png')
        influence = np.abs(combined.astype(np.int16)-source.astype(np.int16)).mean(axis=2)
        Image.fromarray(np.clip(influence * a.influence_gain, 0, 255).astype(np.uint8)).save(
            out / f'{view}-wall-influence.png')
        per_view[view] = {
            'wall_influence_pixels_gt_8': int((influence > 8).sum()),
            'source_dark_pixels': int((source.mean(axis=2) < 32).sum()),
            'combined_dark_pixels': int((combined.mean(axis=2) < 32).sum()),
        }
    result = {'seed': str(seed), 'source_count': source_count,
              'wall_count': len(model.get_xyz) - source_count,
              'views': per_view, 'diagnostic_only': True, 'approved': False}
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-dir', required=True)
    p.add_argument('--cameras', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--views', nargs='+', default=[
        'frame_0134', 'frame_0141', 'frame_0143', 'frame_0131', 'frame_0147'])
    p.add_argument('--width', type=int, default=540)
    p.add_argument('--influence-gain', type=float, default=5.)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
