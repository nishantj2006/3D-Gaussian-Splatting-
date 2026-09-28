"""Find still-visible object instances in renders after an initial 3D edit.

Detections are restricted to separately recorded instance footprints. Their
mask is evidence for another attribution pass, not permission to delete points.
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
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          Sam2Model, Sam2Processor)


def rank_residual(candidate, prior, score, *, min_area, min_overlap):
    area = float(candidate.mean())
    if area < min_area or area > .6:
        return None
    hit = candidate & prior
    overlap = hit.sum()/max(candidate.sum(), 1)
    if overlap < min_overlap:
        return None
    return {"detector_score": float(score), "candidate_area": area,
            "inside_instance_fraction": float(overlap),
            "rank": float(score) + float(overlap)}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    with open(args.instance_manifest, encoding="utf-8") as handle:
        known = json.load(handle)
    prompts = dict(pair.split("=", 1) for pair in args.prompt)
    if set(prompts) != set(known["instances"]):
        raise ValueError("Supply exactly one prompt for each separate instance label")
    detector_processor = AutoProcessor.from_pretrained(args.detector_model)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.detector_model).to(args.device).eval()
    sam_processor = Sam2Processor.from_pretrained(args.segmenter_model)
    sam = Sam2Model.from_pretrained(args.segmenter_model).to(args.device).eval()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    reports = {label: {"views": {}, "prompt": prompt, "approved": False}
               for label, prompt in prompts.items()}
    for view in args.views:
        if view in args.holdout_views:
            raise ValueError("Held-out views cannot train residual attribution")
        entry = known["views"].get(view, {})
        if not entry.get("complete"):
            for label in prompts:
                reports[label]["views"][view] = {"accepted": False,
                    "reason": "incomplete_independent_masks"}
            continue
        source = Path(args.renders) / f"{view}.png"
        image = Image.open(source).convert("RGB")
        if image.width != args.width:
            image = image.resize((args.width, round(image.height * args.width/image.width)))
        for label, prompt in prompts.items():
            prior = np.asarray(Image.open(entry["instances"][label]["mask_path"])
                               .convert("L").resize(image.size, Image.Resampling.NEAREST)) > 127
            dilated = cv2.dilate(prior.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            inputs = detector_processor(images=image, text=[[prompt]],
                                        return_tensors="pt").to(args.device)
            with torch.inference_mode():
                prediction = detector(**inputs)
            detections = detector_processor.post_process_grounded_object_detection(
                prediction, inputs.input_ids, threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[(image.height, image.width)])[0]
            ranked = sorted(zip(detections["boxes"], detections["scores"]),
                            key=lambda x: float(x[1]), reverse=True)[:args.max_boxes]
            accepted = None
            candidates = []
            if ranked:
                boxes = [[float(z) for z in box] for box, _ in ranked]
                sam_inputs = sam_processor(images=image, input_boxes=[boxes],
                                           return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    output_masks = sam(**sam_inputs, multimask_output=False)
                masks = sam_processor.post_process_masks(
                    output_masks.pred_masks.cpu(), sam_inputs["original_sizes"])[0]
                for index, (_, score) in enumerate(ranked):
                    mask = (masks[index, 0] > 0).numpy()
                    quality = rank_residual(mask, dilated, score,
                        min_area=args.min_area, min_overlap=args.min_overlap)
                    candidates.append(quality)
                    if quality and (accepted is None or quality["rank"] > accepted[0]):
                        accepted = (quality["rank"], mask & dilated, quality)
            if accepted is None:
                reports[label]["views"][view] = {"accepted": False,
                    "reason": "no_safe_rendered_detection", "candidates": candidates}
                continue
            _, mask, quality = accepted
            path = output / "masks" / f"{view}-{label}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            reports[label]["views"][view] = {"accepted": True, "mask_path": str(path),
                "quality": quality, "candidates": candidates,
                "residual_pixels": int(mask.sum())}
    for label, report in reports.items():
        path = output / f"{label}-manifest.json"
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report["peak_gpu_allocated_mb"] = (torch.cuda.max_memory_allocated() / 1024**2
                                           if args.device == "cuda" else 0.)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return {label: {"detected_views": sum(d["accepted"] for d in report["views"].values()),
                    "views": report["views"]} for label, report in reports.items()}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--renders", required=True)
    p.add_argument("--instance-manifest", required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--prompt", action="append", required=True,
                   help="LABEL=TEXT, one for each independently identified instance")
    p.add_argument("--holdout-views", nargs="+", default=[])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--segmenter-model", default="facebook/sam2.1-hiera-tiny")
    p.add_argument("--width", type=int, default=540)
    p.add_argument("--box-threshold", type=float, default=.2)
    p.add_argument("--text-threshold", type=float, default=.18)
    p.add_argument("--min-area", type=float, default=.002)
    p.add_argument("--min-overlap", type=float, default=.5)
    p.add_argument("--max-boxes", type=int, default=5)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
