"""Exclude independent frame and protected furniture from rendered bed residue."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def safe_bed_mask(bed, frame, furniture, radius=3):
    bed = np.asarray(bed, bool)
    frame = np.asarray(frame, bool)
    furniture = np.asarray(furniture, bool)
    if bed.shape != frame.shape or bed.shape != furniture.shape:
        raise ValueError('Instance masks must have identical dimensions')
    if radius < 0 or radius > 32:
        raise ValueError('Invalid exclusion radius')
    forbidden = frame | furniture
    if radius:
        side = 2*radius+1
        forbidden = cv2.dilate(forbidden.astype(np.uint8),
                               np.ones((side, side), np.uint8)) > 0
    return bed & ~forbidden


def run(a):
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    bed = json.loads(Path(a.bed_manifest).read_text())
    frame = json.loads(Path(a.frame_manifest).read_text())
    furniture = json.loads(Path(a.protected_manifest).read_text())
    out.mkdir(parents=True)
    (out / 'masks').mkdir()
    result = {'views': {}, 'approved': False, 'source_bed_manifest': str(Path(a.bed_manifest).resolve()),
              'frame_manifest': str(Path(a.frame_manifest).resolve()),
              'protected_manifest': str(Path(a.protected_manifest).resolve()),
              'exclusion_radius': a.exclusion_radius}
    for view, item in sorted(bed['views'].items()):
        if not item.get('accepted'):
            result['views'][view] = {'accepted': False, 'reason': 'no_bed_detection'}
            continue
        if view in a.holdout_views:
            raise ValueError('Held-out view in residue masks')
        with Image.open(item['mask_path']) as im:
            bed_mask = np.asarray(im.convert('L')) > 127
        frame_item = frame['views'].get(view, {})
        if not frame_item.get('accepted'):
            result['views'][view] = {'accepted': False, 'reason': 'no_independent_frame_mask'}
            continue
        with Image.open(frame_item['mask_path']) as im:
            frame_mask = np.asarray(im.convert('L').resize(
                (bed_mask.shape[1], bed_mask.shape[0]), Image.Resampling.NEAREST)) > 127
        furniture_item = furniture['views'].get(view, {})
        if not furniture_item.get('accepted'):
            result['views'][view] = {'accepted': False, 'reason': 'no_protected_furniture_mask'}
            continue
        with Image.open(furniture_item['mask_path']) as im:
            furniture_mask = np.asarray(im.convert('L').resize(
                (bed_mask.shape[1], bed_mask.shape[0]), Image.Resampling.NEAREST)) > 127
        safe = safe_bed_mask(bed_mask, frame_mask, furniture_mask, a.exclusion_radius)
        if safe.mean() < a.minimum_area:
            result['views'][view] = {'accepted': False, 'reason': 'too_little_safe_residue',
                                     'pixels': int(safe.sum())}
            continue
        path = out / 'masks' / f'{view}-bed-safe.png'
        Image.fromarray(safe.astype(np.uint8)*255).save(path)
        result['views'][view] = {'accepted': True, 'mask_path': str(path),
                                 'bed_pixels': int(bed_mask.sum()),
                                 'safe_pixels': int(safe.sum()),
                                 'excluded_pixels': int((bed_mask & ~safe).sum())}
    if sum(x['accepted'] for x in result['views'].values()) < a.minimum_views:
        raise ValueError('Insufficient safely masked training views')
    (out / 'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'safe_views': sum(x['accepted'] for x in result['views'].values()),
                      'total_views': len(result['views'])}))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bed-manifest', 'frame-manifest', 'protected-manifest', 'output-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--holdout-views', nargs='+', required=True)
    p.add_argument('--exclusion-radius', type=int, default=3)
    p.add_argument('--minimum-area', type=float, default=.001)
    p.add_argument('--minimum-views', type=int, default=4)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
