"""Render one trained camera view from a Gaussian PLY for edit verification."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth-output", help="Optional NumPy depth map for analyzing visible splats")
    parser.add_argument("--debug-y-bins", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Color Gaussians red, green, or blue according to world Y")
    parser.add_argument("--debug-plane-json", help="reconstruct_flat.py preview.json with fitted plane")
    parser.add_argument("--debug-plane-bins", type=float, nargs=2, default=(0.05, 0.5),
                        metavar=("LOW", "HIGH"), help="Signed plane-distance thresholds")
    parser.add_argument("--width", type=int, default=540)
    args = parser.parse_args()

    with open(args.cameras, encoding="utf-8") as handle:
        cameras = json.load(handle)
    entry = next((cam for cam in cameras if cam["img_name"] == args.image_name), None)
    if entry is None:
        raise ValueError(f"Camera {args.image_name!r} not found")

    scale = args.width / entry["width"]
    height = round(entry["height"] * scale)
    fovx = 2 * math.atan(entry["width"] / (2 * entry["fx"]))
    fovy = 2 * math.atan(entry["height"] / (2 * entry["fy"]))
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = entry["rotation"]
    c2w[:3, 3] = entry["position"]
    w2c = np.linalg.inv(c2w)
    world_view = torch.from_numpy(w2c).transpose(0, 1).contiguous().cuda()
    projection = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).cuda()
    camera = SimpleNamespace(
        image_width=args.width,
        image_height=height,
        FoVx=fovx,
        FoVy=fovy,
        world_view_transform=world_view,
        full_proj_transform=world_view @ projection,
        camera_center=torch.tensor(entry["position"], dtype=torch.float32, device="cuda"),
    )
    model = GaussianModel(3, semantic_feature_dim=128)
    model.load_ply(args.ply)
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    background = torch.zeros(3, device="cuda")
    override_color = None
    if args.debug_y_bins:
        low, high = args.debug_y_bins
        if low >= high:
            raise ValueError("LOW must be less than HIGH")
        y = model.get_xyz[:, 1]
        override_color = torch.zeros((len(y), 3), device="cuda")
        override_color[y < low, 0] = 1.0
        override_color[(y >= low) & (y < high), 1] = 1.0
        override_color[y >= high, 2] = 1.0
    if args.debug_plane_json:
        with open(args.debug_plane_json, encoding="utf-8") as handle:
            details = json.load(handle)
        origin = torch.tensor(details["plane_origin"], dtype=torch.float32, device="cuda")
        normal = torch.tensor(details["plane_normal_toward_removed_object"],
                              dtype=torch.float32, device="cuda")
        low, high = args.debug_plane_bins
        if low >= high:
            raise ValueError("Plane LOW must be less than HIGH")
        distance = ((model.get_xyz - origin) * normal).sum(dim=1)
        override_color = torch.zeros((len(distance), 3), device="cuda")
        override_color[distance < low, 2] = 1.0
        override_color[(distance >= low) & (distance < high), 1] = 1.0
        override_color[distance >= high, 0] = 1.0
    with torch.inference_mode():
        result = render(camera, model, pipeline, background, override_color=override_color)
        rgb = result["render"].clamp(0, 1)
    image = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output)
    if args.depth_output:
        Path(args.depth_output).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.depth_output, result["depth"].detach().cpu().numpy())
    print(f"Rendered {args.image_name} to {args.output}")


if __name__ == "__main__":
    main()
