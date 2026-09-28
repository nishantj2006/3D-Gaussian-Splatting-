"""Preview-only exact rasterizer evaluation of Gaussian object selections."""

import argparse
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def camera_settings(camera, height, width):
    """Match this repository's camera JSON and training rasterizer convention."""
    rotation = np.asarray(camera["rotation"], dtype=np.float32)
    position = np.asarray(camera["position"], dtype=np.float32)
    translation = -rotation.T @ position
    view = torch.from_numpy(getWorld2View2(rotation, translation)).T.cuda()
    fovx = 2 * math.atan(camera["width"] / (2 * camera["fx"]))
    fovy = 2 * math.atan(camera["height"] / (2 * camera["fy"]))
    projection = getProjectionMatrix(.01, 100., fovx, fovy).T.cuda()
    return GaussianRasterizationSettings(
        image_height=height, image_width=width,
        tanfovx=math.tan(fovx / 2), tanfovy=math.tan(fovy / 2),
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.,
        viewmatrix=view, projmatrix=view @ projection, sh_degree=0,
        campos=torch.from_numpy(position).cuda(), prefiltered=False, debug=False)


def render_selection(model, rasterizer, indices):
    n = len(model.get_xyz)
    colors = torch.zeros((n, 3), device="cuda", dtype=torch.float32)
    colors[torch.as_tensor(indices, device="cuda", dtype=torch.long)] = 1.
    image, _, radii, _ = rasterizer(
        means3D=model.get_xyz,
        means2D=torch.zeros_like(model.get_xyz),
        shs=None, colors_precomp=colors,
        semantic_feature=model.get_semantic_feature,
        opacities=model.get_opacity, scales=model.get_scaling,
        rotations=model.get_rotation, cov3D_precomp=None)
    return image[0].clamp(0, 1), int((radii > 0).sum())


def scores(rendered, target, threshold):
    predicted = rendered >= threshold
    intersection = int((predicted & target).sum())
    predicted_count = int(predicted.sum())
    target_count = int(target.sum())
    return {"iou": intersection / max(predicted_count + target_count - intersection, 1),
            "precision": intersection / max(predicted_count, 1),
            "recall": intersection / max(target_count, 1),
            "rendered_pixels": predicted_count, "target_pixels": target_count}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    start = time.perf_counter()
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    with open(args.mask_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    model = GaussianModel(args.sh_degree, args.semantic_dimensions)
    model.load_ply(args.scene)
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                 "_scaling", "_rotation", "_semantic_feature"):
        getattr(model, name).requires_grad_(False)
    selections = {name: np.load(path, allow_pickle=False)
                  for name, path in args.selection}
    count = len(model.get_xyz)
    for name, ids in selections.items():
        if ids.ndim != 1 or len(ids) == 0 or ids.min() < 0 or ids.max() >= count:
            raise ValueError(f"Invalid selection: {name}")
    output.mkdir(parents=True)
    report = {"source": str(Path(args.scene).resolve()), "selections": {},
              "threshold": args.threshold, "approved": False,
              "metric_warning": "Automatic SAM 2 masks are pseudo-labels, not ground truth."}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for view in args.views:
            detail = manifest["views"][view]
            if not detail["accepted"]:
                raise ValueError(f"Mask not accepted in {view}")
            target = np.asarray(Image.open(detail["mask_path"]).convert("L")) > 127
            height, width = target.shape
            rasterizer = GaussianRasterizer(camera_settings(cameras[view], height, width))
            target_tensor = torch.from_numpy(target).cuda()
            report["selections"][view] = {}
            for name, indices in selections.items():
                rendered, visible = render_selection(model, rasterizer, indices)
                image = (rendered.cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(image).save(output / f"{view}-{name}.png")
                report["selections"][view][name] = {
                    **scores(rendered >= args.threshold, target_tensor, .5),
                    "selected_splats": len(indices), "visible_scene_splats": visible}
    report["elapsed_seconds"] = time.perf_counter() - start
    report["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--selection", nargs=2, action="append", metavar=("NAME", "INDICES"), required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--threshold", type=float, default=.1)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--semantic-dimensions", type=int, default=128)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
