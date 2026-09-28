"""Render a reproducible top-down diagnostic camera over a selected object."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from gsedit.reconstruction.refine_local_background import render_rgb
from scene.gaussian_model import GaussianModel
from utils.ply_semantic_utils import read_vertices


def overhead_camera(target, plane_normal, reference, distance, width):
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    forward = -normal
    up_hint = np.asarray(reference["rotation"], dtype=np.float64)[:, 1]
    up = up_hint - np.dot(up_hint, forward)*forward
    if np.linalg.norm(up) < .1:
        up_hint = np.array([1., 0., 0.])
        up = up_hint - np.dot(up_hint, forward)*forward
    up /= np.linalg.norm(up)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    rotation = np.column_stack((right, up, forward))
    entry = dict(reference)
    entry["position"] = (target + distance*normal).tolist()
    entry["rotation"] = rotation.tolist()
    entry["width"] = width
    entry["height"] = width
    entry["fx"] = reference["fx"]*width/reference["width"]
    entry["fy"] = reference["fy"]*width/reference["height"]
    return entry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ply", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--selected-indices", required=True)
    p.add_argument("--floor-fit", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--reference-view", default="frame_0141")
    p.add_argument("--distance", type=float, default=3.)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    _, vertices = read_vertices(a.source)
    selected = np.load(a.selected_indices, allow_pickle=False)
    xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")])
    target = np.median(xyz[selected], axis=0)
    with open(a.floor_fit, encoding="utf-8") as handle:
        fit = json.load(handle)
    normal = np.asarray(fit["frame"])[:, 2]
    if normal[2] < 0:
        normal = -normal
    with open(a.cameras, encoding="utf-8") as handle:
        cameras = {c["img_name"]: c for c in json.load(handle)}
    entry = overhead_camera(target, normal, cameras[a.reference_view], a.distance, a.width)
    model = GaussianModel(3, 128)
    model.load_ply(a.ply)
    settings = camera_settings(entry, a.width, a.width)._replace(sh_degree=3)
    with torch.no_grad():
        rgb = render_rgb(model, GaussianRasterizer(settings), model.get_opacity)
    pixels = (rgb.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(a.output)
    print(json.dumps({"output": str(Path(a.output).resolve()), "target": target.tolist(),
                      "camera": entry["position"], "distance": a.distance}))


if __name__ == "__main__":
    main()
