"""Render original and preview-pruned scene from held-out camera poses."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from gsedit.evaluation.evaluate_rendered_masks import camera_settings
from scene.gaussian_model import GaussianModel


def render_scene(model, rasterizer, opacity):
    image, _, _, _ = rasterizer(
        means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
        shs=model.get_features, colors_precomp=None,
        semantic_feature=model.get_semantic_feature,
        opacities=opacity, scales=model.get_scaling,
        rotations=model.get_rotation, cov3D_precomp=None)
    return (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    model = GaussianModel(args.sh_degree, args.semantic_dimensions)
    model.load_ply(args.scene)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                 "_scaling", "_rotation", "_semantic_feature"):
        getattr(model, name).requires_grad_(False)
    indices = np.load(args.indices, allow_pickle=False)
    if indices.ndim != 1 or len(indices) == 0 or indices.min() < 0 or indices.max() >= len(model.get_xyz):
        raise ValueError("Invalid selection")
    opacity = model.get_opacity
    pruned_opacity = opacity.clone()
    pruned_opacity[torch.as_tensor(indices, device="cuda", dtype=torch.long)] = 0
    output.mkdir(parents=True)
    with torch.no_grad():
        for view in args.views:
            camera = cameras[view]
            width = args.image_width
            height = round(camera["height"] * width / camera["width"])
            settings = camera_settings(camera, height, width)._replace(sh_degree=model.active_sh_degree)
            rasterizer = GaussianRasterizer(settings)
            original = render_scene(model, rasterizer, opacity)
            pruned = render_scene(model, rasterizer, pruned_opacity)
            Image.fromarray(original).save(output / f"{view}-original.png")
            Image.fromarray(pruned).save(output / f"{view}-pruned.png")
            Image.fromarray(np.abs(original.astype(np.int16)-pruned).astype(np.uint8)).save(
                output / f"{view}-difference.png")
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump({"source": str(Path(args.scene).resolve()),
                   "indices": str(Path(args.indices).resolve()),
                   "views": args.views, "approved": False}, handle, indent=2)
        handle.write("\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--indices", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--image-width", type=int, default=540)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--semantic-dimensions", type=int, default=128)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
