"""Independently segment a named background surface in source photographs.

Each candidate is saved; only a mask that avoids the foreground-object mask
can serve as a wall/floor donor.  This does not edit the 3D scene.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
import torch
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          Sam2Model, Sam2Processor)


def quality(mask, excluded, detector_score, *, min_area=.01,
            max_area=.8, max_overlap=.12):
    area = float(mask.mean())
    if area < min_area or area > max_area:
        return None
    overlap = float((mask & excluded).sum()/max(mask.sum(), 1))
    if overlap > max_overlap:
        return None
    return {"area_fraction": area, "excluded_overlap": overlap,
            "score": float(detector_score) + .15*min(area/.25, 1.) - overlap}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse mask folder: {output}")
    start = time.perf_counter()
    with open(args.foreground_manifest, encoding="utf-8") as handle:
        foreground = json.load(handle)
    views = sorted(view for view, item in foreground["views"].items() if item["accepted"])
    detector_processor = AutoProcessor.from_pretrained(args.detector_model)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.detector_model).to(args.device).eval()
    segmenter_processor = Sam2Processor.from_pretrained(args.segmenter_model)
    segmenter = Sam2Model.from_pretrained(args.segmenter_model).to(args.device).eval()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    (output / "overlays").mkdir()
    report = {"prompts": args.prompts, "foreground_manifest": str(
                  Path(args.foreground_manifest).resolve()),
              "detector_model": args.detector_model,
              "segmenter_model": args.segmenter_model,
              "views": {}, "approved": False}
    for view in views:
        matches = [p for p in Path(args.images).glob(view+".*") if
                   p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if len(matches) != 1:
            raise ValueError(f"Expected one source image for {view}")
        with Image.open(matches[0]) as source:
            image = source.convert("RGB")
            if image.width != args.image_width:
                image = image.resize((args.image_width,
                                      round(image.height*args.image_width/image.width)))
        excluded = np.asarray(Image.open(foreground["views"][view]["mask_path"])
                              .convert("L").resize(image.size)) > 127
        entry = {"accepted": False, "candidates": []}
        inputs = detector_processor(images=image, text=[args.prompts],
                                    return_tensors="pt").to(args.device)
        with torch.inference_mode():
            detection = detector(**inputs)
        boxes = detector_processor.post_process_grounded_object_detection(
            detection, inputs.input_ids, threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(image.height, image.width)])[0]
        candidates = sorted(zip(boxes["boxes"], boxes["scores"],
                                boxes["text_labels"]),
                            key=lambda item: float(item[1]), reverse=True)[:args.max_boxes]
        if candidates:
            segmenter_inputs = segmenter_processor(
                images=image, input_boxes=[[[float(z) for z in box] for box, _, _ in candidates]],
                return_tensors="pt").to(args.device)
            with torch.inference_mode():
                masks_out = segmenter(**segmenter_inputs, multimask_output=False)
            masks = segmenter_processor.post_process_masks(
                masks_out.pred_masks.cpu(), segmenter_inputs["original_sizes"])[0]
            best = None
            for index, (_, score, label) in enumerate(candidates):
                mask = (masks[index, 0] > 0).numpy()
                rank = quality(mask, excluded, float(score),
                               max_overlap=args.max_foreground_overlap)
                path = output / "masks" / f"{view}-candidate-{index+1}.png"
                Image.fromarray((mask*255).astype(np.uint8)).save(path)
                entry["candidates"].append({"id": index+1, "label": str(label),
                                            "score": float(score), "quality": rank,
                                            "mask_path": str(path)})
                if rank is not None and (best is None or rank["score"] > best[0]):
                    best = (rank["score"], index, mask)
            if best is not None:
                _, index, mask = best
                path = output / "masks" / f"{view}-surface.png"
                Image.fromarray((mask*255).astype(np.uint8)).save(path)
                overlay = np.asarray(image).copy()
                overlay[mask] = np.rint(.55*overlay[mask] +
                                        .45*np.array([255, 30, 30])).astype(np.uint8)
                Image.fromarray(overlay).save(output / "overlays" / f"{view}.png")
                entry.update(accepted=True, selected_candidate=index+1,
                             mask_path=str(path), quality=entry["candidates"][index]["quality"])
        report["views"][view] = entry
        print(json.dumps({"view": view, "accepted": entry["accepted"],
                          "candidates": len(entry["candidates"])}), flush=True)
    report["accepted_views"] = sum(v["accepted"] for v in report["views"].values())
    report["elapsed_seconds"] = time.perf_counter()-start
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    report["peak_gpu_allocated_mb"] = (torch.cuda.max_memory_allocated()/1024**2
                                        if args.device == "cuda" else None)
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"accepted_views": report["accepted_views"],
                      "elapsed_seconds": report["elapsed_seconds"],
                      "peak_gpu_allocated_mb": report["peak_gpu_allocated_mb"]}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True)
    p.add_argument("--foreground-manifest", required=True)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--segmenter-model", default="facebook/sam2.1-hiera-tiny")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--image-width", type=int, default=540)
    p.add_argument("--box-threshold", type=float, default=.2)
    p.add_argument("--text-threshold", type=float, default=.18)
    p.add_argument("--max-boxes", type=int, default=5)
    p.add_argument("--max-foreground-overlap", type=float, default=.12)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
