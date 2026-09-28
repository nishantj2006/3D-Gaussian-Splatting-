"""Preview-only image-first removal on selected masked camera views.

No scene or original image is modified. The resulting images are diagnostics,
not 3D supervision until multi-view consistency has been checked.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from PIL import Image
import torch

from gsedit.generation.generate_background_views import inpaint_diffusion, load_diffusion


def resized_size(image, width):
    height = round(image.height * width / image.width / 8) * 8
    return width, height


def prepare_mask(path, size, close_px, dilate_px):
    mask = np.asarray(Image.open(path).convert("L").resize(
        size, Image.Resampling.NEAREST)) > 127
    if close_px:
        kernel = np.ones((close_px, close_px), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                kernel) > 0
    if dilate_px:
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    if not .01 < mask.mean() < .8:
        raise ValueError(f"Mask coverage unsafe: {mask.mean():.1%}")
    return mask


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.width < 256 or args.width % 8:
        raise ValueError("Width must be at least 256 and divisible by 8")
    with open(args.mask_manifest, encoding="utf-8") as handle:
        entries = json.load(handle)["views"]
    output.mkdir(parents=True)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    pipe = None
    report = {"model": args.model if args.backend == "diffusion" else None,
              "backend": args.backend,
              "source_manifest": str(Path(args.mask_manifest).resolve()),
              "init_image_dir": str(Path(args.init_image_dir).resolve()) if args.init_image_dir else None,
              "strength": args.strength, "steps": args.steps, "seed": args.seed,
              "prompt": args.prompt, "negative_prompt": args.negative_prompt,
              "approved": False, "views": {}}
    try:
        pipe = load_diffusion(args.model) if args.backend == "diffusion" else None
        for index, view in enumerate(args.views):
            entry = entries.get(view, {})
            if not entry.get("accepted"):
                raise ValueError(f"No accepted object mask for {view}")
            paths = [p for p in Path(args.images).glob(view + ".*") if
                     p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if len(paths) != 1:
                raise ValueError(f"Expected one source image for {view}")
            source = Image.open(paths[0]).convert("RGB")
            size = resized_size(source, args.width)
            photo = np.asarray(source.resize(size, Image.Resampling.LANCZOS))
            mask = prepare_mask(entry["mask_path"], size,
                                args.close_px, args.dilate_px)
            init = photo
            if args.init_image_dir:
                guide_path = Path(args.init_image_dir) / f"{view}-guide.png"
                init = np.asarray(Image.open(guide_path).convert("RGB"))
                if init.shape != photo.shape or np.any(init[~mask] != photo[~mask]):
                    raise ValueError(f"Guide changes exterior pixels or shape in {view}")
            began = time.perf_counter()
            if pipe:
                result = inpaint_diffusion(pipe, init, mask, args.prompt,
                                          args.negative_prompt, args.steps,
                                          args.seed + index, strength=args.strength)
            else:
                result = cv2.inpaint(photo, mask.astype(np.uint8),
                                     args.opencv_radius, cv2.INPAINT_TELEA)
                result[~mask] = photo[~mask]
            Image.fromarray(photo).save(output / f"{view}-source.png")
            if args.init_image_dir:
                Image.fromarray(init).save(output / f"{view}-guide.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                output / f"{view}-mask.png")
            Image.fromarray(result).save(output / f"{view}-inpainted.png")
            report["views"][view] = {"mask_fraction": float(mask.mean()),
                                      "size": size,
                                      "seconds": time.perf_counter() - began,
                                      "inpainted": str(output / f"{view}-inpainted.png")}
            print(json.dumps({"view": view, **report["views"][view]}), flush=True)
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        with open(output / "report.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True)
    p.add_argument("--mask-manifest", required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backend", choices=("diffusion", "opencv"), default="diffusion")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="duplicated objects, deformed surfaces, black hole")
    p.add_argument("--init-image-dir")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--close-px", type=int, default=5)
    p.add_argument("--dilate-px", type=int, default=2)
    p.add_argument("--opencv-radius", type=int, default=7)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
