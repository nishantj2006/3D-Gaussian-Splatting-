"""Reproject one shared carpet atlas after independent diffusion previews.

Diffusion may independently repaint each view.  Keep its wall result, but use
the same atlas-derived floor in every view.  This is a 2D preview only.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image


FLOOR_LAYOUT = np.array([50, 170, 50], np.uint8)


def lock_floor(generated, guide, layout, source, target_mask):
    if (generated.shape != guide.shape or generated.shape != source.shape or
            generated.shape != layout.shape or generated.shape[:2] != target_mask.shape):
        raise ValueError("Preview dimensions differ")
    floor = np.all(layout == FLOOR_LAYOUT, axis=-1)
    if not floor.any() or np.any(floor & ~target_mask):
        raise ValueError("Invalid floor layout or mask")
    result = generated.copy()
    result[floor] = guide[floor]
    result[~target_mask] = source[~target_mask]
    return result, floor


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    with open(args.inpaint_report, encoding="utf-8") as handle:
        inpaint = json.load(handle)
    with open(args.guide_report, encoding="utf-8") as handle:
        guides = json.load(handle)
    if inpaint["approved"] or guides["approved"]:
        raise ValueError("Expected unapproved preview inputs")
    views = sorted(set(inpaint["views"]) & set(guides["views"]))
    if not views:
        raise ValueError("No common generated views")
    output.mkdir(parents=True)
    started = time.perf_counter()
    details = {}
    for view in views:
        base = Path(args.inpaint_report).parent
        guide_base = Path(args.guide_report).parent
        generated = np.asarray(Image.open(base/f"{view}-inpainted.png").convert("RGB"))
        guide = np.asarray(Image.open(guide_base/f"{view}-guide.png").convert("RGB"))
        layout = np.asarray(Image.open(guide_base/f"{view}-layout.png").convert("RGB"))
        source = np.asarray(Image.open(base/f"{view}-source.png").convert("RGB"))
        mask = np.asarray(Image.open(base/f"{view}-mask.png").convert("L")) > 127
        result, floor = lock_floor(generated, guide, layout, source, mask)
        path = output/f"{view}-inpainted.png"
        Image.fromarray(result).save(path)
        details[view] = {"floor_pixels_locked": int(floor.sum()), "output": str(path),
                         "exterior_unchanged": bool(np.array_equal(result[~mask], source[~mask]))}
    report = {"guide_report": str(Path(args.guide_report).resolve()),
              "inpaint_report": str(Path(args.inpaint_report).resolve()),
              "views": details, "approved": False, "geometry_supported": False,
              "elapsed_seconds": time.perf_counter()-started,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    with open(output/"report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inpaint-report", required=True)
    p.add_argument("--guide-report", required=True)
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
