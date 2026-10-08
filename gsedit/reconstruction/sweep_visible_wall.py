"""Test whether visible-wall masks constrain a wall plane's depth.

Only training views select the offset. Held-out views are evaluated afterward.
No Gaussian PLY is written: a flat score curve is a depth-ambiguity result.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from PIL import Image

from gsedit.assets.align_asset import plane_frame


def samples(mask, limit, seed):
    # Include mask boundaries, where parallax is most informative, and interior.
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    rng = np.random.default_rng(seed)
    points = []
    for region in ((mask & (distance <= 30)), (mask & (distance > 30))):
        yy, xx = np.where(region)
        if len(xx) > limit:
            take = rng.choice(len(xx), limit, replace=False)
            xx, yy = xx[take], yy[take]
        points.append(np.column_stack((xx, yy)))
    return np.concatenate(points) if any(len(p) for p in points) else np.empty((0, 2))


def intersect(camera, pixels, shape, origin, wall_normal, offset):
    height, width = shape
    rays = np.column_stack(((pixels[:, 0]-width/2) /
                            (camera['fx']*width/camera['width']),
                            (pixels[:, 1]-height/2) /
                            (camera['fy']*height/camera['height']),
                            np.ones(len(pixels)))) @ np.asarray(camera['rotation']).T
    start = np.asarray(camera['position'])
    denominator = rays @ wall_normal
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (offset-(start-origin) @ wall_normal)/denominator
    valid = np.isfinite(depth) & (depth > .05) & (depth < 60)
    xyz = start + np.where(valid, depth, 0)[:, None]*rays
    return xyz, valid


def project(xyz, camera, shape):
    height, width = shape
    local = (xyz-np.asarray(camera['position'])) @ np.asarray(camera['rotation'])
    z = local[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.rint(local[:, 0]/z*camera['fx']*width/camera['width']+width/2)
        y = np.rint(local[:, 1]/z*camera['fy']*height/camera['height']+height/2)
    good = np.isfinite(x) & np.isfinite(y) & (z > .05) & (z < 60) & (
        x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[good].astype(int), y[good].astype(int), good


def score_target(target, source_views, cameras, masks, pixel_samples,
                 origin, wall_normal, offset):
    pixels = pixel_samples[target]
    xyz, on_plane = intersect(cameras[target], pixels, masks[target].shape,
                              origin, wall_normal, offset)
    wall_votes = np.zeros(len(pixels), np.uint16)
    comparable = np.zeros(len(pixels), np.uint16)
    for source in source_views:
        if source == target:
            continue
        x, y, in_frame = project(xyz, cameras[source], masks[source].shape)
        ids = np.flatnonzero(on_plane & in_frame)
        if len(ids) != len(x):
            # Projection validity also includes rays that missed the plane.
            x, y, projected = project(xyz[on_plane], cameras[source], masks[source].shape)
            ids = np.flatnonzero(on_plane)[projected]
        hit = masks[source][y, x]
        wall_votes[ids] += hit
        comparable[ids] += 1
    supported = wall_votes >= 2
    valid = on_plane & (comparable >= 2)
    return {'coverage': float(supported[valid].mean()) if valid.any() else 0.,
            'valid_fraction': float(valid.mean()),
            'wall_support_mean': float(wall_votes[valid].mean()) if valid.any() else 0.,
            'sample_count': int(len(pixels))}


def run(a):
    started = time.perf_counter()
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    cameras = {c['img_name']: c for c in json.loads(Path(a.cameras).read_text())}
    entries = json.loads(Path(a.wall_manifest).read_text())['views']
    masks = {v: np.asarray(Image.open(d['mask_path']).convert('L')) > 127
             for v, d in entries.items() if d.get('accepted') and v in cameras}
    holdouts = set(a.holdout_views)
    training = sorted(set(masks)-holdouts)
    testing = sorted(set(masks)&holdouts)
    if len(training) < 4 or len(testing) < 2:
        raise ValueError('Need at least four training and two held-out wall views')
    floor = json.loads(Path(a.floor_fit).read_text())
    hypothesis = json.loads(Path(a.wall_hypothesis).read_text())
    origin, frame = plane_frame(floor['plane_origin'],
                                floor['plane_normal_toward_removed_object'])
    normal2 = np.asarray(hypothesis['wall_normal_floor_xy'], float)
    normal2 /= np.linalg.norm(normal2)
    wall_normal = frame[:, :2] @ normal2
    sample = {v: samples(masks[v], a.samples_per_region,
                         a.seed+int(v.split('_')[-1])) for v in masks}
    offsets = np.arange(a.min_offset, a.max_offset+a.step/2, a.step)
    rows = []
    for offset in offsets:
        per_view = {v: score_target(v, training, cameras, masks, sample,
                                    origin, wall_normal, float(offset))
                    for v in training}
        score = float(np.mean([x['coverage'] for x in per_view.values()]))
        rows.append({'offset': float(offset), 'training_coverage': score,
                     'training_views': per_view})
    best = max(rows, key=lambda row: row['training_coverage'])
    heldout = {v: score_target(v, training, cameras, masks, sample,
                               origin, wall_normal, best['offset']) for v in testing}
    scores = np.asarray([row['training_coverage'] for row in rows])
    near = scores >= max(scores)-a.ambiguity_margin
    report = {'plane_normal_floor_xy': normal2.tolist(), 'best_offset': best['offset'],
              'best_training_coverage': best['training_coverage'],
              'heldout_at_best': heldout, 'offsets_within_margin':
              [float(offsets[near].min()), float(offsets[near].max())],
              'ambiguity_margin': a.ambiguity_margin,
              'depth_constrained': bool((offsets[near].max()-offsets[near].min()) <= a.max_ambiguity_width
                                        and best['offset'] not in (offsets[0], offsets[-1])),
              'training_views': training, 'holdout_views': testing,
              'all_scores': [{'offset': r['offset'], 'coverage': r['training_coverage']}
                             for r in rows], 'approved': False,
              'warning': 'Mask agreement is not measured depth; a narrow optimum is necessary but not sufficient.',
              'elapsed_seconds': time.perf_counter()-started,
              'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('best_offset', 'best_training_coverage',
          'offsets_within_margin', 'depth_constrained', 'heldout_at_best',
          'elapsed_seconds')}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cameras', 'wall-manifest', 'floor-fit', 'wall-hypothesis', 'output-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--holdout-views', nargs='+', default=['frame_0134','frame_0141','frame_0143'])
    p.add_argument('--min-offset', type=float, default=4.)
    p.add_argument('--max-offset', type=float, default=14.)
    p.add_argument('--step', type=float, default=.25)
    p.add_argument('--samples-per-region', type=int, default=1200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ambiguity-margin', type=float, default=.02)
    p.add_argument('--max-ambiguity-width', type=float, default=1.)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
