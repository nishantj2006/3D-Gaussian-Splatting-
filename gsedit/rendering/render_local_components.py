"""Render retained and newly synthesized Gaussian components separately."""

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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ply", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--original-count", required=True, type=int)
    p.add_argument("--floor-count", required=True, type=int)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--width", default=540, type=int)
    a = p.parse_args()
    output = Path(a.output_dir)
    if output.exists():
        raise FileExistsError(output)
    with open(a.cameras, encoding="utf-8") as handle:
        cameras = {x["img_name"]: x for x in json.load(handle)}
    model = GaussianModel(3, 128)
    model.load_ply(a.ply)
    base = model.get_opacity.detach()
    n = len(base)
    if not 0 < a.original_count < a.original_count+a.floor_count < n:
        raise ValueError("Invalid component offsets")
    parts = {"retained": (0, a.original_count),
             "floor": (a.original_count, a.original_count+a.floor_count),
             "wall": (a.original_count+a.floor_count, n)}
    output.mkdir(parents=True)
    with torch.no_grad():
        for view in a.images:
            camera = cameras[view]
            height = round(a.width*camera["height"]/camera["width"])
            settings = camera_settings(camera, height, a.width)._replace(sh_degree=3)
            rasterizer = GaussianRasterizer(settings)
            for label, (begin, end) in parts.items():
                opacity = torch.zeros_like(base)
                opacity[begin:end] = base[begin:end]
                image = render_rgb(model, rasterizer, opacity)
                pixels = (image.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                Image.fromarray(pixels).save(output / f"{view}-{label}.png")
    print(json.dumps({"components": {k: e-b for k, (b,e) in parts.items()},
                      "images": a.images, "output_dir": str(output)}))


if __name__ == "__main__":
    main()
