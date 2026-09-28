"""Preview code-driven object extraction and support-surface placement.

Text features propose a target; observed color and camera projections localize
it. Geometry places an asset on its top. Outputs are previews in a new folder.
"""

from gsedit.runtime import PROJECT_ROOT, module_command

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from gsedit.assets.align_asset import fit_transform, plane_frame, transform_gaussians
from utils.ply_semantic_utils import numbered_fields, read_vertices, write_vertices
from utils.semantic_utils import encode_text, load_pca


def xyz_of(vertices):
    return np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float64)


def color_mask(rgb, color):
    r, g, b = [rgb[..., i] for i in range(3)]
    if color == "blue":
        return (b > r * 1.15) & (b > g * 1.06) & (b > 0.16)
    if color == "red":
        return (r > g * 1.20) & (r > b * 1.15) & (r > 0.18)
    if color == "green":
        return (g > r * 1.15) & (g > b * 1.12) & (g > 0.16)
    raise ValueError(f"Unsupported color cue: {color}")


def photo_region(path, color, width=540):
    with Image.open(path) as source:
        height = round(source.height * width / source.width)
        photo = np.asarray(source.convert("RGB").resize((width, height)),
                           dtype=np.float32) / 255
    rough = ndimage.binary_opening(color_mask(photo, color), iterations=2)
    labels, count = ndimage.label(rough)
    if count == 0:
        raise ValueError(f"No {color} region in {path}")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    region = labels == sizes.argmax()
    fraction = float(region.mean())
    if not 0.005 <= fraction <= 0.65:
        raise ValueError(f"Color region is too small/broad in {path}: {fraction:.2%}")
    return region, fraction


def project_mask(xyz, camera, mask):
    height, width = mask.shape
    local = (xyz - np.asarray(camera["position"])) @ np.asarray(camera["rotation"])
    depth = np.maximum(local[:, 2], 1e-6)
    x = np.floor(local[:, 0] / depth * camera["fx"] * width /
                 camera["width"] + width / 2).astype(int)
    y = np.floor(local[:, 1] / depth * camera["fy"] * height /
                 camera["height"] + height / 2).astype(int)
    valid = (local[:, 2] > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    hit = np.zeros(len(xyz), dtype=bool)
    ids = np.flatnonzero(valid)
    hit[ids] = mask[y[ids], x[ids]]
    return hit


def semantic_margin(vertices, label, negatives, pca_path):
    fields = numbered_fields(vertices.dtype.names, "semantic_")
    pca = load_pca(pca_path)
    if len(fields) != pca.n_components_:
        raise ValueError("Scene and PCA semantic dimensions differ")
    features = np.column_stack([vertices[f] for f in fields]).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
    positives = ([label, label + " mattress", label + " cover", label + " blanket"]
                 if label.lower() == "bed" else [label, label + " surface"])
    queries = encode_text(positives + negatives, pca, device="cpu").cpu().numpy()
    scores = features @ queries.T
    return scores[:, :len(positives)].max(axis=1) - scores[:, len(positives):].max(axis=1)


def detect(vertices, cameras, images, views, ground, *, label, color,
           negatives, pca_path, min_margin, grow_radius):
    xyz = xyz_of(vertices)
    rgb = np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]) * 0.2820947918 + 0.5
    point_color = color_mask(rgb, color)
    margin = semantic_margin(vertices, label, negatives, pca_path)
    height = (xyz - ground[0]) @ ground[1]
    votes = np.zeros(len(vertices), dtype=np.uint8)
    view_details = {}
    for view in views:
        if view not in cameras:
            raise ValueError(f"Unknown camera: {view}")
        paths = list(Path(images).glob(view + ".*"))
        if len(paths) != 1:
            raise ValueError(f"Expected one source photo for {view}")
        region, fraction = photo_region(paths[0], color)
        hit = project_mask(xyz, cameras[view], region)
        votes += hit.astype(np.uint8)
        view_details[view] = {"color_fraction": fraction, "projected_points": int(hit.sum())}
    seeds = (votes > 0) & point_color & (margin >= min_margin) & (height > 0.25)
    seed_ids = np.flatnonzero(seeds)
    if len(seed_ids) < 300:
        raise ValueError("Too few supported object seeds")
    labels = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1).fit_predict(xyz[seed_ids])
    counts = np.bincount(labels[labels >= 0])
    if len(counts) == 0 or counts.max() < 300:
        raise ValueError("No coherent 3D object component")
    seed_ids = seed_ids[labels == counts.argmax()]
    distance = cKDTree(xyz[seed_ids]).query(xyz, workers=-1)[0]
    selected = (distance <= grow_radius) & (margin >= min_margin - 0.15) & (height > 0.5)
    selected[seed_ids] = True
    if selected.sum() > len(vertices) * 0.20:
        raise ValueError("Selection is too broad for safe extraction")
    report = {"label": label, "color": color, "views": view_details,
              "seed_points": int(len(seed_ids)), "selected_points": int(selected.sum()),
              "selected_fraction": float(selected.mean()),
              "xyz_quantiles": np.quantile(xyz[selected], [0.02, 0.5, 0.98], axis=0).tolist()}
    return selected, report


def top_anchor(points, ground):
    origin, normal = ground
    height = (points - origin) @ normal
    if len(points) < 300 or np.quantile(height, 0.9) - np.quantile(height, 0.1) > 3:
        raise ValueError("Support surface is too sparse or vertically ambiguous")
    upper = points[height >= np.quantile(height, 0.72)]
    center = np.median(upper, axis=0)
    nearby = upper[cKDTree(upper).query(center, k=min(200, len(upper)))[1]]
    local_height = np.quantile((nearby - origin) @ normal, 0.35)
    anchor = center + (local_height - (center - origin) @ normal) * normal
    return anchor, {"height_above_ground": float(local_height),
                    "support_points": int(len(nearby))}



def overlapping_splats(vertices, cameras, images, views, color, ground, margin):
    """Catch broad Gaussians whose screen footprint covers the object mask."""
    xyz = xyz_of(vertices)
    rgb = np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]) * 0.2820947918 + 0.5
    radius_world = np.exp(np.column_stack([vertices[f"scale_{i}"] for i in range(3)])).max(axis=1)
    height = (xyz - ground[0]) @ ground[1]
    spill = np.zeros(len(vertices), dtype=bool)
    for view in views:
        camera = cameras[view]
        photo = next(Path(images).glob(view + ".*"))
        mask, _ = photo_region(photo, color)
        screen_distance = ndimage.distance_transform_edt(~mask)
        h, w = mask.shape
        local = (xyz - np.asarray(camera["position"])) @ np.asarray(camera["rotation"])
        depth = np.maximum(local[:, 2], 1e-6)
        px = np.rint(local[:, 0] / depth * camera["fx"] * w / camera["width"] + w / 2).astype(int)
        py = np.rint(local[:, 1] / depth * camera["fy"] * h / camera["height"] + h / 2).astype(int)
        inside = (local[:, 2] > 0) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        ids = np.flatnonzero(inside)
        projected_radius = 3 * radius_world[ids] * camera["fx"] * w / camera["width"] / depth[ids]
        distance = screen_distance[py[ids], px[ids]]
        plausible_color = (rgb[ids].mean(axis=1) < 0.45) | color_mask(rgb[ids], color)
        hit = (projected_radius > 8) & (distance < np.minimum(projected_radius + 10, 180))
        hit &= (height[ids] > 0.5) & (margin[ids] > -0.2) & plausible_color
        spill[ids[hit]] = True
    return spill


def choose_views(cameras, images, color, count=3):
    """Rank photos by coherent target-color area and camera baseline."""
    ranked = []
    for name, camera in cameras.items():
        paths = list(Path(images).glob(name + ".*"))
        if len(paths) != 1:
            continue
        try:
            _, fraction = photo_region(paths[0], color, width=270)
        except ValueError:
            continue
        if fraction >= 0.05:
            ranked.append((fraction, name, np.asarray(camera["position"])))
    ranked.sort(reverse=True)
    chosen = []
    for _, name, position in ranked:
        if all(np.linalg.norm(position - old) >= 0.30 for _, old in chosen):
            chosen.append((name, position))
        if len(chosen) == count:
            break
    if len(chosen) < 2:
        raise ValueError("Auto view selection found fewer than two distinct views")
    return [name for name, _ in chosen]

def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output folder: {output}")
    if not 0.05 <= args.grow_radius <= 0.5:
        raise ValueError("--grow-radius must be in [0.05, 0.5]")
    if args.action == "place" and (not args.asset or args.diameter <= 0):
        raise ValueError("place requires --asset and positive --diameter")
    ply, scene = read_vertices(args.scene)
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    if args.views == ["auto"]:
        args.views = choose_views(cameras, args.images, args.color)
    with open(args.ground_plane, encoding="utf-8") as handle:
        plane = json.load(handle)
    origin, frame = plane_frame(plane["plane_origin"],
                                plane["plane_normal_toward_removed_object"])
    ground = (origin, frame[:, 2])
    selected, report = detect(scene, cameras, args.images, args.views, ground,
                              label=args.surface, color=args.color,
                              negatives=args.negative, pca_path=args.pca_path,
                              min_margin=args.min_margin, grow_radius=args.grow_radius)
    asset_selection = selected.copy()
    if args.action == "extract":
        margin = semantic_margin(scene, args.surface, args.negative, args.pca_path)
        spill = overlapping_splats(scene, cameras, args.images, args.views,
                                   args.color, ground, margin)
        report["screen_overlap_added"] = int((spill & ~selected).sum())
        selected |= spill
        if selected.mean() > 0.20:
            raise ValueError("Expanded object mask is too broad")
        report["selected_points"] = int(selected.sum())
        report["selected_fraction"] = float(selected.mean())
    output.mkdir(parents=True)
    indices = np.flatnonzero(selected)
    np.save(output / "selected-indices.npy", indices)
    write_vertices(output / "selected-surface.ply", scene[selected].copy(), ply,
                   ["unreviewed semantic multi-view selection"])
    if args.action == "extract":
        write_vertices(output / "extracted-object.ply", scene[asset_selection].copy(), ply,
                       ["unreviewed separate object splat asset"])
        write_vertices(output / "residual-cleanup.ply", scene[selected & ~asset_selection].copy(),
                       ply, ["unreviewed broad-splat cleanup; not part of clean asset"])
        report["clean_asset_points"] = int(asset_selection.sum())
        write_vertices(output / "pruned-preview.ply", scene[~selected].copy(), ply,
                       ["unreviewed object removal"])
    else:
        anchor, support = top_anchor(xyz_of(scene[selected]), ground)
        _, asset = read_vertices(args.asset)
        _, bed_frame = plane_frame(anchor, ground[1])
        rotation, scale, source_anchor, world_anchor, details = fit_transform(
            xyz_of(asset), anchor, bed_frame, [0, 0], [args.diameter] * 3,
            up_axis=args.asset_up_axis, yaw_deg=args.yaw_deg)
        aligned = transform_gaussians(asset, rotation, scale, source_anchor, world_anchor)
        write_vertices(output / "asset-aligned.ply", aligned, ply,
                       ["unreviewed asset placement on detected support"])
        report["anchor"] = anchor.tolist()
        report["support"] = support
        report["alignment"] = details
        merged = output / "scene-with-asset.ply"
        subprocess.run([*module_command("merge.py"),
                        "--scene", args.scene, "--asset", str(output / "asset-aligned.ply"),
                        "--output", str(merged), "--object-id", str(args.object_id),
                        "--label", args.asset_label], check=True)
        report["merged_preview"] = str(merged)
        report["object_id"] = args.object_id
        report["asset_label"] = args.asset_label
    report["source"] = str(Path(args.scene).resolve())
    report["selected_indices"] = "selected-indices.npy"
    report["approved"] = False
    with open(output / "preview.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not args.skip_render:
        targets = (("pruned-preview.ply", "pruned"), ("extracted-object.ply", "bed-only")) if args.action == "extract" else (("scene-with-asset.ply", "scene"),)
        for name in args.views:
            for ply_name, suffix in targets:
                subprocess.run([*module_command("render_ply_preview.py"),
                                "--ply", str(output / ply_name), "--cameras", args.cameras,
                                "--image-name", name,
                                "--output", str(output / (name + "-" + suffix + ".png")),
                                "--width", str(args.render_width)], check=True)
    print(json.dumps({"output": str(output), "selected_points": report["selected_points"],
                      "anchor": report.get("anchor"), "approved": False}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("place", "extract"))
    p.add_argument("--scene", required=True)
    p.add_argument("--asset")
    p.add_argument("--surface", default="bed")
    p.add_argument("--color", choices=("blue", "red", "green"), default="blue")
    p.add_argument("--cameras", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--views", nargs="+", default=["auto"], help="Camera names or auto")
    p.add_argument("--ground-plane", required=True)
    p.add_argument("--pca-path", default="data/my_scene/pca_model_128.pkl")
    p.add_argument("--negative", nargs="+", default=["carpet", "floor", "wall", "clothes", "dresser"])
    p.add_argument("--min-margin", type=float, default=-0.05)
    p.add_argument("--grow-radius", type=float, default=0.22)
    p.add_argument("--diameter", type=float, default=0.8)
    p.add_argument("--asset-up-axis", choices=("x", "-x", "y", "-y", "z", "-z"), default="z")
    p.add_argument("--yaw-deg", type=float)
    p.add_argument("--object-id", type=int, default=43)
    p.add_argument("--asset-label", default="generated asset")
    p.add_argument("--render-width", type=int, default=540)
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
