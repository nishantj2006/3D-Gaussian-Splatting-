"""Rerank saved surface-mask candidates using foreground-relative geometry."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def rank_candidate(surface, foreground, detection_score, *, max_overlap=.05,
                   min_above_fraction=.8, max_area=.45):
    if surface.shape != foreground.shape or not foreground.any():
        return None
    area = float(surface.mean())
    if area < .01 or area > max_area:
        return None
    y_top = int(np.quantile(np.where(foreground)[0], .05))
    above = float(surface[:y_top].sum()/max(surface.sum(), 1))
    overlap = float((surface & foreground).sum()/max(surface.sum(), 1))
    if above < min_above_fraction or overlap > max_overlap:
        return None
    return {"above_foreground_fraction": above,
            "foreground_overlap": overlap, "area_fraction": area,
            "score": float(detection_score) + .1*min(area/.2, 1.) - overlap}


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse filtered-mask folder: {output}")
    with open(args.candidate_manifest, encoding="utf-8") as handle:
        candidates = json.load(handle)
    with open(args.foreground_manifest, encoding="utf-8") as handle:
        foreground = json.load(handle)
    output.mkdir(parents=True)
    (output / "overlays").mkdir()
    report = {"source_candidates": str(Path(args.candidate_manifest).resolve()),
              "foreground_manifest": str(Path(args.foreground_manifest).resolve()),
              "views": {}, "approved": False}
    for view, detail in candidates["views"].items():
        front = np.asarray(Image.open(foreground["views"][view]["mask_path"])
                           .convert("L")) > 127
        best = None
        for item in detail["candidates"]:
            mask = np.asarray(Image.open(item["mask_path"]).convert("L")) > 127
            rank = rank_candidate(mask, front, item["score"],
                                  max_overlap=args.max_overlap,
                                  min_above_fraction=args.min_above_fraction,
                                  max_area=args.max_area)
            if rank is not None and (best is None or rank["score"] > best[0]):
                best = (rank["score"], item, rank, mask)
        entry = {"accepted": best is not None}
        if best is not None:
            _, item, rank, mask = best
            entry.update(mask_path=item["mask_path"], candidate_id=item["id"],
                         quality=rank)
            matches = list(Path(args.images).glob(view+".*"))
            if len(matches) == 1:
                image = np.asarray(Image.open(matches[0]).convert("RGB").resize(
                    (mask.shape[1], mask.shape[0]))).copy()
                image[mask] = np.rint(.55*image[mask] +
                                      .45*np.array([255, 20, 20])).astype(np.uint8)
                Image.fromarray(image).save(output / "overlays" / f"{view}.png")
        report["views"][view] = entry
    report["accepted_views"] = sum(x["accepted"] for x in report["views"].values())
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"accepted_views": report["accepted_views"],
                      "accepted": sorted(k for k,v in report["views"].items()
                                         if v["accepted"])}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--foreground-manifest", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-overlap", type=float, default=.05)
    p.add_argument("--min-above-fraction", type=float, default=.8)
    p.add_argument("--max-area", type=float, default=.45)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
