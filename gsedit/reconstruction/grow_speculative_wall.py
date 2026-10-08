"""Grow a speculative wall/floor Gaussian patch from verified local donors.

The wall depth is a hypothesis, not measured geometry. Source records are
unchanged; output is a new unapproved PLY plus evidence and held-out renders.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from gsedit.assets.align_asset import plane_frame
from gsedit.reconstruction.build_continuous_background import gaussian_frame, surface_gaussians
from gsedit.reconstruction.build_local_background import mask_votes
from gsedit.reconstruction.sweep_visible_wall import intersect
from utils.ply_semantic_utils import read_vertices, write_vertices


SH_C0 = .28209479177387814


def read_mask(entries, view, shape):
    item = entries.get(view)
    if not item or not item.get('accepted'):
        return np.zeros(shape, bool)
    return np.asarray(Image.open(item['mask_path']).convert('L').resize(
        (shape[1], shape[0]), Image.Resampling.NEAREST)) > 127


def grid_from_rays(points, step, margin, max_cells, height_floor=False):
    if len(points) < 100:
        raise ValueError('Too few rays for replacement surface')
    low = np.quantile(points, .01, axis=0)-margin
    high = np.quantile(points, .99, axis=0)+margin
    if height_floor:
        low[1] = max(0, low[1])
        high[1] = min(high[1], 6.)
    shape = np.ceil((high-low)/step).astype(int)
    if np.any(shape <= 0) or int(np.prod(shape)) > max_cells:
        raise ValueError('Replacement patch exceeds safety grid limit')
    return low, tuple(int(x) for x in shape)


def cell_indices(points, low, shape, step):
    ij = np.floor((points-low)/step).astype(int)
    good = np.all((ij >= 0) & (ij < np.asarray(shape)), axis=1)
    return ij, good


def supported_grid(per_view_points, low, shape, step, min_views, grow_cells):
    votes = np.zeros(shape, np.uint16)
    for points in per_view_points.values():
        if not len(points):
            continue
        ij, good = cell_indices(points, low, shape, step)
        flat = np.unique(np.ravel_multi_index(ij[good].T, shape))
        votes.ravel()[flat] += 1
    core = votes >= min_views
    if core.sum() < 50:
        raise ValueError('Too few cells supported by multiple hole views')
    expanded = ndimage.binary_dilation(core, iterations=grow_cells)
    return core, expanded, votes


def recursive_donors(active, low, step, donor_uv):
    """Multi-source 4-neighbor growth copies donor identity into each cell."""
    ij = np.argwhere(active)
    uv = low+(ij+.5)*step
    labels = np.full(active.shape, -1, np.int32)
    queue = deque()
    # Start at the active cell nearest to every donor; seed each disconnected
    # component as well, then propagate only through the requested patch.
    tree = cKDTree(uv)
    _, nearest_cell = tree.query(donor_uv)
    for donor, cell in enumerate(np.atleast_1d(nearest_cell)):
        i, j = ij[cell]
        if labels[i, j] < 0:
            labels[i, j] = donor
            queue.append((int(i), int(j)))
    components, count = ndimage.label(active)
    seeded = set(int(components[i, j]) for i, j in queue)
    for component in range(1, count+1):
        if component in seeded:
            continue
        candidates = np.argwhere(components == component)
        donor = int(cKDTree(donor_uv).query(low+(candidates.mean(axis=0)+.5)*step)[1])
        i, j = candidates[len(candidates)//2]
        labels[i, j] = donor
        queue.append((int(i), int(j)))
    while queue:
        i, j = queue.popleft()
        for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
            if 0 <= ni < active.shape[0] and 0 <= nj < active.shape[1] and (
                    active[ni, nj] and labels[ni, nj] < 0):
                labels[ni, nj] = labels[i, j]
                queue.append((ni, nj))
    if np.any(active & (labels < 0)):
        raise RuntimeError('Donor propagation missed active cells')
    return labels


def wall_photo_atlas(images, entries, views, cameras, origin, normal, offset,
                     low, shape, step, tangent, up, sample_stride):
    sums = np.zeros((*shape, 3), np.float64)
    counts = np.zeros(shape, np.int32)
    for view in views:
        wall = read_mask(entries, view, (960, 540))
        if not wall.any():
            continue
        image = np.asarray(Image.open(Path(images)/(view+'.jpg')).convert('RGB').resize(
            (wall.shape[1], wall.shape[0]), Image.Resampling.LANCZOS))
        yy, xx = np.where(wall[::sample_stride, ::sample_stride])
        pixels = np.column_stack((xx*sample_stride+.5, yy*sample_stride+.5))
        xyz, valid = intersect(cameras[view], pixels, wall.shape, origin, normal, offset)
        xy = np.column_stack(((xyz[valid]-origin) @ tangent,
                              (xyz[valid]-origin) @ up))
        ij, inside = cell_indices(xy, low, shape, step)
        yy_img = np.minimum(yy*sample_stride, wall.shape[0]-1)
        xx_img = np.minimum(xx*sample_stride, wall.shape[1]-1)
        color = image[yy_img, xx_img][valid][inside]
        np.add.at(sums, (ij[inside, 0], ij[inside, 1]), color)
        np.add.at(counts, (ij[inside, 0], ij[inside, 1]), 1)
    known = counts > 0
    if known.sum() < 20:
        raise ValueError('Visible-wall photos supply too few atlas texels')
    observed = np.zeros((*shape, 3), np.uint8)
    observed[known] = np.rint(sums[known]/counts[known, None]).astype(np.uint8)
    # Large blank regions are recursively extended from the nearest observed
    # texel; a small inpaint pass softens only their internal seams.
    nearest = ndimage.distance_transform_edt(~known, return_distances=False,
                                              return_indices=True)
    extended = observed[nearest[0], nearest[1]]
    missing = (~known).astype(np.uint8)
    color = cv2.inpaint(extended, missing, 3, cv2.INPAINT_TELEA)
    color[known] = observed[known]
    return color, known, observed


def read_floor_atlas(path, report_path, uv):
    atlas = np.asarray(Image.open(path).convert('RGB'))
    info = json.loads(Path(report_path).read_text())
    low = np.asarray(info['uv_low'])
    high = np.asarray(info['uv_high'])
    xy = (uv-low)/(high-low)
    coverage = np.all((xy >= 0) & (xy <= 1), axis=1)
    x = np.clip(np.rint(xy[:, 0]*(atlas.shape[1]-1)).astype(int), 0, atlas.shape[1]-1)
    y = np.clip(np.rint(xy[:, 1]*(atlas.shape[0]-1)).astype(int), 0, atlas.shape[0]-1)
    return atlas[y, x], coverage


def run(a):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings
    from gsedit.rendering.render_pruned_preview import render_scene
    from scene.gaussian_model import GaussianModel

    start = time.perf_counter()
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    source_ply, source = read_vertices(a.scene)
    xyz = np.column_stack([source[k] for k in ('x', 'y', 'z')]).astype(float)
    floor = json.loads(Path(a.floor_fit).read_text())
    sweep = json.loads(Path(a.sweep_report).read_text())
    origin, frame = plane_frame(floor['plane_origin'],
                                floor['plane_normal_toward_removed_object'])
    normal2 = np.asarray(sweep['plane_normal_floor_xy'])
    normal2 /= np.linalg.norm(normal2)
    tangent, wall_basis = gaussian_frame(frame, normal2)
    wall_normal = frame[:, :2] @ normal2
    offset = sweep['best_offset']
    cameras = {c['img_name']: c for c in json.loads(Path(a.cameras).read_text())}
    wall_entries = json.loads(Path(a.wall_manifest).read_text())
    object_entries = json.loads(Path(a.object_manifest).read_text())
    training = [v for v in sweep['training_views'] if v in object_entries['views']]
    holdouts = sweep['holdout_views']
    model = GaussianModel(3, 128)
    model.load_ply(a.scene)
    for key in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                '_scaling', '_rotation', '_semantic_feature'):
        getattr(model, key).requires_grad_(False)
    out.mkdir(parents=True)
    (out/'hole-masks').mkdir()
    per_wall, per_floor, hole_counts = {}, {}, {}
    for view in training:
        camera = cameras[view]
        height = round(camera['height']*a.width/camera['width'])
        raster = GaussianRasterizer(camera_settings(camera, height, a.width)
                                    ._replace(sh_degree=3))
        with torch.no_grad():
            image = render_scene(model, raster, model.get_opacity)
        object_mask = read_mask(object_entries['views'], view, image.shape[:2])
        dark = (image.astype(float).mean(axis=2) < a.hole_brightness) & object_mask
        dark = ndimage.binary_dilation(dark, iterations=a.hole_dilate_px)
        dark &= ndimage.binary_dilation(object_mask, iterations=a.hole_dilate_px)
        Image.fromarray((dark*255).astype(np.uint8)).save(out/'hole-masks'/f'{view}.png')
        hole_counts[view] = int(dark.sum())
        yy, xx = np.where(dark[::a.ray_stride, ::a.ray_stride])
        pixels = np.column_stack((xx*a.ray_stride+.5, yy*a.ray_stride+.5))
        rays = np.column_stack(((pixels[:, 0]-a.width/2)/(camera['fx']*a.width/camera['width']),
                                (pixels[:, 1]-height/2)/(camera['fy']*height/camera['height']),
                                np.ones(len(pixels)))) @ np.asarray(camera['rotation']).T
        pos = np.asarray(camera['position'])
        with np.errstate(divide='ignore', invalid='ignore'):
            wd = (offset-(pos-origin) @ wall_normal)/(rays @ wall_normal)
            fd = ((origin-pos) @ frame[:, 2])/(rays @ frame[:, 2])
        wvalid = np.isfinite(wd) & (wd > .05) & (wd < 50)
        fvalid = np.isfinite(fd) & (fd > .05) & (fd < 50)
        wall_xyz = pos+wd[:, None]*rays
        floor_xyz = pos+fd[:, None]*rays
        wall_height = (wall_xyz-origin) @ frame[:, 2]
        is_wall = wvalid & (wall_height >= 0) & (wall_height <= 6) & (
            (wd < fd) | ~fvalid)
        is_floor = fvalid & ((fd < wd) | ~wvalid)
        per_wall[view] = np.column_stack(((wall_xyz[is_wall]-origin) @
                                           (frame[:, :2] @ tangent), wall_height[is_wall]))
        per_floor[view] = (floor_xyz[is_floor]-origin) @ frame[:, :2]
    wall_points = np.concatenate(list(per_wall.values()))
    floor_points = np.concatenate(list(per_floor.values()))
    wall_low, wall_shape = grid_from_rays(wall_points, a.wall_step, a.wall_margin,
                                           a.max_cells, height_floor=True)
    floor_low, floor_shape = grid_from_rays(floor_points, a.floor_step, a.floor_margin,
                                             a.max_cells)
    wall_core, wall_active, wall_votes_grid = supported_grid(
        per_wall, wall_low, wall_shape, a.wall_step, a.min_hole_views, a.grow_cells)
    floor_core, floor_active, floor_votes_grid = supported_grid(
        per_floor, floor_low, floor_shape, a.floor_step, a.min_hole_views, a.grow_cells)
    local = (xyz-origin) @ frame
    wall_votes = mask_votes(xyz, cameras, wall_entries, training)
    rgb = np.clip(np.column_stack([source[f'f_dc_{j}'] for j in range(3)])*SH_C0+.5, 0, 1)
    wall_reference = np.median(rgb[(wall_votes >= 2) & (local[:, 2] > .5)], axis=0)
    if not np.isfinite(wall_reference).all():
        raise ValueError('No visible wall Gaussian color reference')
    alpha = 1/(1+np.exp(-source['opacity']))
    wall_distance = np.abs(local[:, :2] @ normal2-offset)
    donor_mask = (wall_votes >= 2) & (wall_distance <= a.donor_plane_distance) & (
        local[:, 2] > .5) & (alpha > .1) & (
        np.linalg.norm(rgb-wall_reference, axis=1) <= a.donor_color_distance)
    wall_donors = np.flatnonzero(donor_mask)
    if len(wall_donors) < a.min_wall_donors:
        # Fall back to the closest high-vote plane splats but keep the result
        # visibly flagged as low confidence in the report.
        candidates = np.flatnonzero((wall_votes >= 2) & (local[:, 2] > .5) & (alpha > .1))
        rank = wall_distance[candidates]+np.linalg.norm(rgb[candidates]-wall_reference, axis=1)
        wall_donors = candidates[np.argsort(rank)[:a.min_wall_donors]]
    if len(wall_donors) < 3:
        raise ValueError('Too few wall Gaussian seeds even for a speculative fill')
    wall_ij = np.argwhere(wall_active)
    wall_uv = wall_low+(wall_ij+.5)*a.wall_step
    wall_donor_uv = np.column_stack((local[wall_donors, :2] @ tangent,
                                     local[wall_donors, 2]))
    inherited = recursive_donors(wall_active, wall_low, a.wall_step, wall_donor_uv)
    parent = wall_donors[inherited[wall_active]]
    wall_xyz = (origin+(wall_uv[:, 0, None]*tangent+offset*normal2) @ frame[:, :2].T+
                wall_uv[:, 1, None]*frame[:, 2])
    wall_edge = ndimage.distance_transform_edt(wall_active)
    wall_opacity = np.clip(a.wall_opacity*np.minimum(1, wall_edge[wall_active]/a.grow_cells),
                           .08, .95)
    wall_fill = surface_gaussians(source, parent, wall_xyz, step=a.wall_step,
                                   basis=wall_basis, opacity=wall_opacity)
    wall_color, known, observed = wall_photo_atlas(
        a.images, wall_entries['views'], training, cameras, origin, wall_normal,
        offset, wall_low, wall_shape, a.wall_step, frame[:, :2] @ tangent,
        frame[:, 2], a.photo_stride)
    Image.fromarray(wall_color).save(out/'wall-atlas.png')
    Image.fromarray(observed).save(out/'wall-observed.png')
    Image.fromarray((known*255).astype(np.uint8)).save(out/'wall-evidence.png')
    for j in range(3):
        wall_fill[f'f_dc_{j}'] = (wall_color[wall_active, j]/255-.5)/SH_C0
    floor_ij = np.argwhere(floor_active)
    floor_uv = floor_low+(floor_ij+.5)*a.floor_step
    floor_xyz = origin+floor_uv @ frame[:, :2].T+a.floor_lift*frame[:, 2]
    floor_mask = (np.abs(local[:, 2]) < a.floor_donor_height) & (alpha > .1)
    floor_donors = np.flatnonzero(floor_mask)
    _, nearest_floor = cKDTree(local[floor_donors, :2]).query(floor_uv, workers=-1)
    floor_edge = ndimage.distance_transform_edt(floor_active)
    floor_opacity = np.clip(a.floor_opacity*np.minimum(1, floor_edge[floor_active]/a.grow_cells),
                            .08, .95)
    floor_fill = surface_gaussians(source, floor_donors[nearest_floor], floor_xyz,
                                    step=a.floor_step, basis=frame, opacity=floor_opacity)
    floor_rgb, atlas_covered = read_floor_atlas(a.floor_atlas, a.floor_atlas_report,
                                                floor_uv)
    if atlas_covered.mean() < .9:
        raise ValueError('Existing shared carpet atlas does not cover the hole')
    for j in range(3):
        floor_fill[f'f_dc_{j}'] = (floor_rgb[:, j]/255-.5)/SH_C0
    candidate = np.concatenate((source, floor_fill, wall_fill))
    write_vertices(out/'candidate-unapproved.ply', candidate, source_ply,
                   comments=['Speculative wall depth; do not treat as measured geometry.'])
    np.savez_compressed(out/'provenance.npz', wall_parent_indices=parent,
                        floor_parent_indices=floor_donors[nearest_floor],
                        wall_core=wall_core, floor_core=floor_core,
                        wall_grid_votes=wall_votes_grid, floor_grid_votes=floor_votes_grid)
    del model
    torch.cuda.empty_cache()
    final = GaussianModel(3, 128)
    final.load_ply(str(out/'candidate-unapproved.ply'))
    for key in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                '_scaling', '_rotation', '_semantic_feature'):
        getattr(final, key).requires_grad_(False)
    for view in holdouts:
        camera = cameras[view]
        height = round(camera['height']*a.width/camera['width'])
        raster = GaussianRasterizer(camera_settings(camera, height, a.width)
                                    ._replace(sh_degree=3))
        with torch.no_grad():
            image = render_scene(final, raster, final.get_opacity)
        Image.fromarray(image).save(out/f'{view}-candidate.png')
    report = {'source': str(Path(a.scene).resolve()),
              'candidate': str(out/'candidate-unapproved.ply'),
              'wall_offset_hypothesis': offset,
              'wall_depth_constrained': sweep['depth_constrained'],
              'wall_seed_gaussians': len(wall_donors),
              'wall_seed_indices': wall_donors.tolist(),
              'wall_added': len(wall_fill), 'floor_added': len(floor_fill),
              'wall_photo_texels_observed_fraction': float(known.mean()),
              'floor_atlas_coverage': float(atlas_covered.mean()),
              'hole_pixels_by_view': hole_counts,
              'schema_properties': len(source.dtype.names),
              'semantic_dimensions': sum(k.startswith('semantic_') for k in source.dtype.names),
              'source_records_unchanged': True, 'approved': False,
              'warning': 'Speculative 3D fill from a depth-ambiguous plane; visual review required.',
              'elapsed_seconds': time.perf_counter()-start,
              'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated()/1024**2}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('wall_seed_gaussians', 'wall_added',
          'floor_added', 'wall_photo_texels_observed_fraction', 'elapsed_seconds',
          'peak_rss_mb', 'peak_gpu_allocated_mb')}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'cameras', 'images', 'wall-manifest', 'object-manifest',
                 'floor-fit', 'sweep-report', 'floor-atlas', 'floor-atlas-report', 'output-dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--width', type=int, default=540)
    p.add_argument('--hole-brightness', type=float, default=32.)
    p.add_argument('--hole-dilate-px', type=int, default=5)
    p.add_argument('--ray-stride', type=int, default=5)
    p.add_argument('--wall-step', type=float, default=.08)
    p.add_argument('--floor-step', type=float, default=.06)
    p.add_argument('--wall-margin', type=float, default=.4)
    p.add_argument('--floor-margin', type=float, default=.3)
    p.add_argument('--max-cells', type=int, default=140000)
    p.add_argument('--min-hole-views', type=int, default=2)
    p.add_argument('--grow-cells', type=int, default=4)
    p.add_argument('--donor-plane-distance', type=float, default=1.)
    p.add_argument('--donor-color-distance', type=float, default=.3)
    p.add_argument('--min-wall-donors', type=int, default=12)
    p.add_argument('--floor-donor-height', type=float, default=.07)
    p.add_argument('--floor-lift', type=float, default=.006)
    p.add_argument('--floor-opacity', type=float, default=.68)
    p.add_argument('--wall-opacity', type=float, default=.68)
    p.add_argument('--photo-stride', type=int, default=4)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
