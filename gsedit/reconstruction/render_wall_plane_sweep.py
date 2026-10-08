"""Visualize visible-wall reprojection for several candidate plane depths.

This is a diagnostic overlay, not a 3D replacement or an approved wall fit.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gsedit.assets.align_asset import plane_frame
from gsedit.reconstruction.sweep_visible_wall import intersect, project


def mask(entry, shape):
    if not entry or not entry.get('accepted'):
        return np.zeros(shape, bool)
    return np.asarray(Image.open(entry['mask_path']).convert('L').resize(
        (shape[1], shape[0]), Image.Resampling.NEAREST)) > 127


def run(a):
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    cams = {c['img_name']: c for c in json.loads(Path(a.cameras).read_text())}
    wall_entries = json.loads(Path(a.wall_manifest).read_text())['views']
    floor_entries = json.loads(Path(a.floor_manifest).read_text())['views']
    object_entries = json.loads(Path(a.object_manifest).read_text())['views']
    report = json.loads(Path(a.sweep_report).read_text())
    floor = json.loads(Path(a.floor_fit).read_text())
    origin, frame = plane_frame(floor['plane_origin'],
                                floor['plane_normal_toward_removed_object'])
    world_normal = frame[:, :2] @ np.asarray(report['plane_normal_floor_xy'])
    training = report['training_views']
    source_masks = {v: mask(wall_entries.get(v), (a.height, a.width))
                    for v in training}
    out.mkdir(parents=True)
    stats = {}
    for view in a.views:
        camera = cams[view]
        height = round(camera['height']*a.width/camera['width'])
        shape = (height, a.width)
        if shape != (a.height, a.width):
            raise ValueError('Set --height to the rescaled image height')
        photo = Path(a.images) / (view+'.jpg')
        base = np.asarray(Image.open(photo).convert('RGB').resize(
            (a.width, height), Image.Resampling.LANCZOS))
        target_wall = mask(wall_entries.get(view), shape)
        target_floor = mask(floor_entries.get(view), shape)
        target_object = mask(object_entries.get(view), shape)
        stats[view] = {}
        for offset in a.offsets:
            votes = np.zeros(shape, np.uint16)
            for source in training:
                wall = source_masks[source]
                yy, xx = np.where(wall[::a.stride, ::a.stride])
                if not len(xx):
                    continue
                pixels = np.column_stack((xx*a.stride+.5, yy*a.stride+.5))
                xyz, valid = intersect(cams[source], pixels, wall.shape,
                                       origin, world_normal, offset)
                x, y, projected = project(xyz[valid], camera, shape)
                np.add.at(votes, (y, x), 1)
            supported = cv2.dilate((votes > 0).astype(np.uint8),
                                   np.ones((5, 5), np.uint8)) > 0
            overlay = base.copy()
            good = supported & target_wall
            wrong_floor = supported & target_floor & ~target_wall
            unseen_wall = target_wall & ~supported
            occluded = supported & target_object & ~target_wall
            for region, color in ((good, (25, 220, 50)),
                                  (wrong_floor, (255, 45, 20)),
                                  (unseen_wall, (255, 50, 220)),
                                  (occluded, (50, 135, 255))):
                overlay[region] = np.rint(.55*base[region]+.45*np.asarray(color)).astype(np.uint8)
            Image.fromarray(overlay).save(out/f'{view}-offset{offset:g}.png')
            stats[view][str(offset)] = {
                'visible_wall_covered': float(good.sum()/max(target_wall.sum(), 1)),
                'projected_on_floor_pixels': int(wrong_floor.sum()),
                'projected_on_object_pixels': int(occluded.sum())}
    (out/'report.json').write_text(json.dumps({
        'source_sweep': str(Path(a.sweep_report).resolve()),
        'colors': {'green': 'projected wall agrees with visible wall',
                   'red': 'projected wall overlaps floor',
                   'magenta': 'visible wall not covered by projection',
                   'blue': 'projection hidden by foreground object'},
        'views': stats, 'approved': False}, indent=2)+'\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cameras', 'images', 'wall-manifest', 'floor-manifest',
                 'object-manifest', 'floor-fit', 'sweep-report', 'output-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--views', nargs='+', default=['frame_0134','frame_0141','frame_0143'])
    p.add_argument('--offsets', nargs='+', type=float, default=[5.25, 9.25, 14.])
    p.add_argument('--width', type=int, default=540)
    p.add_argument('--height', type=int, default=960)
    p.add_argument('--stride', type=int, default=4)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
