"""Keep only visible wall pixels outside removed objects and observed floor.

The floor mask supplies a conservative wall/floor boundary.  Missing or
conflicting masks reject the view rather than inventing hidden wall depth.
"""

import argparse
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
from PIL import Image


def read_mask(item, size):
    if not item or not item.get("accepted"):
        return None
    image = Image.open(item["mask_path"]).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def clean_wall(wall, target, floor, margin):
    if wall.shape != target.shape or wall.shape != floor.shape:
        raise ValueError("Surface-mask dimensions differ")
    forbidden = (target | floor).astype(np.uint8)
    if margin:
        kernel = np.ones((2*margin+1, 2*margin+1), np.uint8)
        forbidden = cv2.dilate(forbidden, kernel)
    return wall & ~forbidden.astype(bool)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    began = time.perf_counter()
    entries = []
    for path in (args.wall_manifest, args.target_manifest, args.floor_manifest):
        with open(path, encoding="utf-8") as handle:
            entries.append(json.load(handle)["views"])
    wall_entries, target_entries, floor_entries = entries
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    views = {}
    for view, wall_entry in wall_entries.items():
        wall = read_mask(wall_entry, None)
        if wall is None:
            views[view] = {"accepted": False, "reason": "missing_wall_mask"}
            continue
        size = (wall.shape[1], wall.shape[0])
        target = read_mask(target_entries.get(view), size)
        floor = read_mask(floor_entries.get(view), size)
        if target is None or floor is None:
            views[view] = {"accepted": False, "reason": "missing_target_or_floor_mask"}
            continue
        intact = clean_wall(wall, target, floor, args.margin)
        if intact.sum() < args.min_pixels:
            views[view] = {"accepted": False, "reason": "too_little_intact_wall",
                           "intact_pixels": int(intact.sum())}
            continue
        path = output / "masks" / f"{view}-intact-wall.png"
        Image.fromarray((intact*255).astype(np.uint8)).save(path)
        views[view] = {"accepted": True, "mask_path": str(path),
                       "intact_pixels": int(intact.sum()),
                       "rejected_fraction": float(1-intact.sum()/max(wall.sum(), 1))}
    report = {"wall_manifest": str(Path(args.wall_manifest).resolve()),
              "target_manifest": str(Path(args.target_manifest).resolve()),
              "floor_manifest": str(Path(args.floor_manifest).resolve()),
              "views": views,
              "accepted_views": sum(v["accepted"] for v in views.values()),
              "approved": False, "elapsed_seconds": time.perf_counter()-began,
              "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    if report["accepted_views"] < args.min_views:
        raise ValueError("Too few intact wall views")
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("wall-manifest", "target-manifest", "floor-manifest", "output-dir"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--margin", type=int, default=3)
    p.add_argument("--min-pixels", type=int, default=500)
    p.add_argument("--min-views", type=int, default=3)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
