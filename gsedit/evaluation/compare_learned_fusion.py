"""Held-out RGB comparison for an instance-score fusion against its parent edit."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image, ImageDraw


def changed_fraction(original, edited, mask, threshold=.08):
    if not mask.any():
        return None
    delta = np.abs(original.astype(np.float32)-edited.astype(np.float32)) / 255
    return float((delta[mask].max(axis=1) > threshold).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('fusion-dir', 'cameras', 'target-manifest', 'protected-manifest', 'output-dir'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--views', nargs='+', default=['frame_0134', 'frame_0141', 'frame_0143'])
    p.add_argument('--width', type=int, default=540)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from scene.gaussian_model import GaussianModel
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings, render_selection, scores
    from gsedit.rendering.render_pruned_preview import render_scene
    from gsedit.selection.spatial_instance_removal import load_mask
    from utils.ply_semantic_utils import read_vertices

    start = time.perf_counter()
    fusion = json.loads((a.fusion_dir / 'report.json').read_text())
    source = Path(fusion['source'])
    prior = Path(fusion['parent'])
    prior_ids = np.load(prior / 'selected-indices.npy')
    added = np.load(a.fusion_dir / 'learned-additions.npy')
    fused_ids = np.load(a.fusion_dir / 'selected-indices.npy')
    if not np.array_equal(np.union1d(prior_ids, added), fused_ids):
        raise AssertionError('Fusion IDs do not match the parent plus additions')
    header, original = read_vertices(source)
    _, candidate = read_vertices(a.fusion_dir / 'candidate.ply')
    keep = np.ones(len(original), bool)
    keep[fused_ids] = False
    if original.dtype != candidate.dtype or not np.array_equal(original[keep], candidate):
        raise AssertionError('Candidate did not preserve retained original records')
    model = GaussianModel(3, 128)
    model.load_ply(str(source))
    for key in ('_xyz','_features_dc','_features_rest','_opacity',
                '_scaling','_rotation','_semantic_feature'):
        getattr(model, key).requires_grad_(False)
    alpha = model.get_opacity.detach()
    parent_alpha, fused_alpha = alpha.clone(), alpha.clone()
    parent_alpha[torch.from_numpy(prior_ids).cuda()] = 0
    fused_alpha[torch.from_numpy(fused_ids).cuda()] = 0
    cameras = {c['img_name']: c for c in json.loads(a.cameras.read_text())}
    target = json.loads(a.target_manifest.read_text())
    protection = json.loads(a.protected_manifest.read_text())
    a.output_dir.mkdir(parents=True)
    metrics, strips = {}, []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for name in a.views:
            if name not in fusion['holdout_views']:
                raise ValueError(f'{name} is not held out')
            c = cameras[name]
            h = round(c['height'] * a.width / c['width'])
            raster = GaussianRasterizer(camera_settings(c, h, a.width)._replace(sh_degree=3))
            rgb = [render_scene(model, raster, opacity) for opacity in (alpha, parent_alpha, fused_alpha)]
            bed = load_mask(target['views'][name]['instances']['bed']['mask_path'], (a.width, h))
            protected = load_mask(protection['views'][name]['mask_path'], (a.width, h))
            bed_only = bed & ~protected
            outside = ~bed & ~protected
            footprint, _ = render_selection(model, raster, added)
            metrics[name] = dict(
                additions_front_footprint=scores(footprint, torch.from_numpy(bed_only).cuda(), .1),
                added_change_in_bed=changed_fraction(rgb[1], rgb[2], bed_only),
                added_change_in_protected=changed_fraction(rgb[1], rgb[2], protected),
                added_change_outside=changed_fraction(rgb[1], rgb[2], outside),
                parent_protected_change_from_source=changed_fraction(rgb[0], rgb[1], protected),
                fusion_protected_change_from_source=changed_fraction(rgb[0], rgb[2], protected),
                parent_outside_change_from_source=changed_fraction(rgb[0], rgb[1], outside),
                fusion_outside_change_from_source=changed_fraction(rgb[0], rgb[2], outside),
            )
            strip = Image.new('RGB', (3*a.width, h+30), 'white')
            labels = ('Source', 'Parent selection', 'Learned fusion')
            draw = ImageDraw.Draw(strip)
            for i, (label, image) in enumerate(zip(labels, rgb)):
                strip.paste(Image.fromarray(image), (i*a.width,30))
                draw.text((i*a.width+6,8), name+' / '+label, fill='black')
                Image.fromarray(image).save(a.output_dir / f'{name}-{i}.png')
            strips.append(strip)
    sheet = Image.new('RGB', (strips[0].width, sum(s.height for s in strips)), 'white')
    offset = 0
    for strip in strips:
        sheet.paste(strip, (0, offset))
        offset += strip.height
    sheet.save(a.output_dir / 'comparison.png')
    result = dict(approved=False, exact_retained_records=True,
                  source_properties=len(original.dtype.names),
                  semantic_dimensions=sum(k.startswith('semantic_') for k in original.dtype.names),
                  protected_source_ids_removed=0, views=metrics,
                  elapsed_seconds=time.perf_counter()-start,
                  peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                  cautions=['Protected masks are not independently verified dresser ground truth.',
                            'Changed pixels and front footprints do not prove whole-bed removal.',
                            'No background replacement is created by this selection trial.'])
    (a.output_dir / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
