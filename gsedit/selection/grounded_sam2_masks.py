"""Text-grounded SAM 2 masks anchored to an existing 3D object seed.

Every candidate stays separate in the manifest; only the best seed-consistent
mask per view is promoted to the target instance. This never edits a scene.
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

from gsedit.selection.multiview_instance import project
from utils.ply_semantic_utils import read_vertices


def rank_mask(mask, seed_x, seed_y, seed_valid, detector_score):
    """Rank by 3D-seed agreement, then detector confidence and mask compactness."""
    area = float(mask.mean())
    if area < .003 or area > .6 or not seed_valid.any():
        return None
    recall = float(mask[seed_y[seed_valid], seed_x[seed_valid]].mean())
    quality = .7 * recall + .25 * detector_score - .05 * area
    return {"seed_recall": recall, "area_fraction": area, "quality": quality}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse mask folder: {output}")
    start = time.perf_counter()
    _, vertices = read_vertices(args.scene)
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float32)
    seed = np.unique(np.load(args.seed_indices, allow_pickle=False))
    if seed.ndim != 1 or not len(seed) or seed.min() < 0 or seed.max() >= len(points):
        raise ValueError("Invalid seed indices")
    with open(args.cameras, encoding="utf-8") as handle:
        cameras = {item["img_name"]: item for item in json.load(handle)}
    min_visible = max(args.min_visible_points, int(np.ceil(args.min_visible_fraction * len(seed))))
    eligible = []
    for name, camera in cameras.items():
        visible = project(points[seed], camera,
                          (int(camera["height"]), int(camera["width"])))[3]
        if visible.sum() >= min_visible:
            eligible.append((name, int(visible.sum())))
    eligible.sort()
    if args.max_views and len(eligible) > args.max_views:
        order = np.linspace(0, len(eligible)-1, args.max_views).round().astype(int)
        eligible = [eligible[i] for i in np.unique(order)]
    if len(eligible) < args.min_accepted_views:
        raise ValueError("Too few seed-visible source views")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; rerun with --device cpu")
    detector_processor = AutoProcessor.from_pretrained(args.detector_model)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.detector_model).to(device).eval()
    segmenter_processor = Sam2Processor.from_pretrained(args.segmenter_model)
    segmenter = Sam2Model.from_pretrained(args.segmenter_model).to(device).eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    (output / "overlays").mkdir()
    report = {"source": str(Path(args.scene).resolve()), "seed_indices": str(Path(args.seed_indices).resolve()),
              "prompts": args.prompts, "detector_model": args.detector_model,
              "segmenter_model": args.segmenter_model, "device": device,
              "min_visible": min_visible, "views": {}, "approved": False}
    for view, projected_seed_count in eligible:
        camera = cameras[view]
        matches = [p for p in Path(args.images).glob(view + ".*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one source image for {view}")
        with Image.open(matches[0]) as source:
            image = source.convert("RGB")
            if image.width > args.image_width:
                image = image.resize((args.image_width,
                                      round(image.height * args.image_width / image.width)))
        sx, sy, _, valid = project(points[seed], camera, (image.height, image.width))
        entry = {"seed_visible": projected_seed_count, "candidates": [],
                 "accepted": False, "object_id": None}
        detector_inputs = detector_processor(images=image, text=[args.prompts],
                                             return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = detector(**detector_inputs)
        detections = detector_processor.post_process_grounded_object_detection(
            outputs, detector_inputs.input_ids, threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(image.height, image.width)])[0]
        candidates = sorted(zip(detections["boxes"], detections["scores"],
                                detections["text_labels"]),
                            key=lambda item: float(item[1]), reverse=True)[:args.max_boxes]
        if candidates:
            boxes = [box.cpu().tolist() for box, _, _ in candidates]
            segmenter_inputs = segmenter_processor(images=image, input_boxes=[boxes],
                                                   return_tensors="pt").to(device)
            with torch.inference_mode():
                masks_output = segmenter(**segmenter_inputs, multimask_output=False)
            masks = segmenter_processor.post_process_masks(
                masks_output.pred_masks.cpu(), segmenter_inputs["original_sizes"])[0]
            best = None
            for index, (box, score, label) in enumerate(candidates):
                mask = (masks[index, 0] > 0).numpy()
                quality = rank_mask(mask, sx, sy, valid, float(score))
                metadata = {"candidate_id": index + 1, "label": str(label),
                            "box": [float(x) for x in box], "detector_score": float(score),
                            "mask_path": None, "quality": quality}
                if quality is not None:
                    candidate_path = output / "masks" / f"{view}-candidate-{index+1}.png"
                    Image.fromarray((mask * 255).astype(np.uint8)).save(candidate_path)
                    metadata["mask_path"] = str(candidate_path)
                    if best is None or quality["quality"] > best[0]:
                        best = (quality["quality"], index, mask)
                entry["candidates"].append(metadata)
            if best is not None and entry["candidates"][best[1]]["quality"]["seed_recall"] >= args.min_seed_recall:
                _, index, mask = best
                mask_path = output / "masks" / f"{view}-object-1.png"
                Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                overlay = np.asarray(image).copy()
                overlay[mask] = np.rint(.55 * overlay[mask] + .45 * np.array([255, 30, 30])).astype(np.uint8)
                Image.fromarray(overlay).save(output / "overlays" / f"{view}.png")
                entry.update(accepted=True, object_id=1, selected_candidate=index+1,
                             mask_path=str(mask_path), seed_recall=entry["candidates"][index]["quality"]["seed_recall"],
                             mask_fraction=entry["candidates"][index]["quality"]["area_fraction"])
        report["views"][view] = entry
        print(json.dumps({"view": view, "accepted": entry["accepted"],
                          "candidates": len(entry["candidates"]),
                          "seed_recall": entry.get("seed_recall")}), flush=True)
    report["accepted_views"] = sum(item["accepted"] for item in report["views"].values())
    report["elapsed_seconds"] = time.perf_counter() - start
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report["peak_gpu_allocated_mb"] = (torch.cuda.max_memory_allocated() / 1024**2
                                        if device == "cuda" else None)
    report["quality"] = ("needs_3d_validation" if report["accepted_views"] >= args.min_accepted_views
                         else "rejected_insufficient_views")
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), "accepted_views": report["accepted_views"],
                      "eligible_views": len(eligible), "quality": report["quality"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--seed-indices", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--prompts", nargs="+", required=True,
                   help="User-supplied object phrases; no scene-specific names are coded")
    p.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--segmenter-model", default="facebook/sam2.1-hiera-tiny")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--image-width", type=int, default=540)
    p.add_argument("--max-views", type=int, default=0)
    p.add_argument("--min-visible-points", type=int, default=300)
    p.add_argument("--min-visible-fraction", type=float, default=.1)
    p.add_argument("--min-accepted-views", type=int, default=3)
    p.add_argument("--box-threshold", type=float, default=.22)
    p.add_argument("--text-threshold", type=float, default=.2)
    p.add_argument("--max-boxes", type=int, default=5)
    p.add_argument("--min-seed-recall", type=float, default=.35)
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
