"""Reproduce CPU-side edit cost and selector counts for the available room scene.

This is not an accuracy benchmark: the repository has no human-annotated 3D
instance masks and only one independent captured scene. All generated PLYs
live in a temporary directory and are discarded after counts are collected.
"""

from gsedit.runtime import PROJECT_ROOT, module_command

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil
from plyfile import PlyData


ROOT = PROJECT_ROOT
PYTHON = sys.executable
ROOM = ROOT / "output/bottle-orin-128d-5k/semantic-validation/no-bottle-reconstructed-color-matched.ply"
CAMERAS = ROOT / "output/bottle-orin-128d-5k/cameras.json"
IMAGES = ROOT / "data/my_scene/images"
PLANE = ROOT / "output/bottle-orin-128d-5k/semantic-validation/carpet-fill-preview-distance-support/preview.json"
PCA = ROOT / "data/my_scene/pca_model_128.pkl"
SEEDS = ROOT / "output/bed-ops/bed-extract-v4/selected-indices.npy"
MERGED = ROOT / "output/bed-ops/ball-on-bed-auto-v2/scene-with-asset.ply"


def measure(command):
    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    peak = 0
    while process.poll() is None:
        try:
            family = [psutil.Process(process.pid)] + psutil.Process(process.pid).children(recursive=True)
            peak = max(peak, sum(p.memory_info().rss for p in family if p.is_running()))
        except psutil.Error:
            pass
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    if process.returncode:
        raise RuntimeError(f"Benchmark command failed ({process.returncode}):\n{stderr[-2000:]}")
    return {"wall_seconds": round(time.perf_counter() - start, 3),
            "peak_process_rss_mb": round(peak / 1048576, 1),
            "stdout": stdout.strip()}


def count_from_dry_run(output):
    match = re.search(r"Selected ([\d,]+)/([\d,]+) points", output)
    if not match:
        raise ValueError("Could not parse selector count")
    return int(match.group(1).replace(",", "")), int(match.group(2).replace(",", ""))


def main():
    required = (ROOM, CAMERAS, IMAGES, PLANE, PCA, SEEDS, MERGED)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark inputs: {missing}")
    result = {"scene_count": 1, "device": "cpu", "accuracy_ground_truth": False,
              "measurements": {}}
    scene_points = len(PlyData.read(str(ROOM))["vertex"])
    result["scene_points"] = scene_points
    with tempfile.TemporaryDirectory(prefix="gaussian-edit-benchmark-") as folder:
        scratch = Path(folder)
        cases = {
            "bed_vector_only": [*module_command("remove.py", python=PYTHON), "--input", str(ROOM),
                                "--output", str(scratch / "unused.ply"), "--text", "bed",
                                "--negative", "carpet", "floor", "wall", "clothes", "dresser",
                                "--pca-path", str(PCA), "--dry-run", "--device", "cpu"],
            "ball_exact_id": [*module_command("remove.py", python=PYTHON), "--input", str(MERGED),
                              "--output", str(scratch / "unused2.ply"), "--object-id", "43",
                              "--dry-run"],
            "bed_semantic_color": [*module_command("surface_pipeline.py", python=PYTHON), "extract",
                                   "--scene", str(ROOM), "--surface", "bed", "--color", "blue",
                                   "--grow-radius", "0.4", "--cameras", str(CAMERAS),
                                   "--images", str(IMAGES), "--views", "frame_0135", "frame_0141",
                                   "--ground-plane", str(PLANE), "--pca-path", str(PCA),
                                   "--skip-render", "--output-dir", str(scratch / "fused")],
            "bed_multiview_sam": [*module_command("multiview_instance.py", python=PYTHON),
                                  "--scene", str(ROOM), "--seed-indices", str(SEEDS),
                                  "--cameras", str(CAMERAS), "--images", str(IMAGES),
                                  "--views", "frame_0135", "frame_0139", "frame_0141",
                                  "--model", str(ROOT / "mobile_sam.pt"), "--device", "cpu",
                                  "--skip-render", "--output-dir", str(scratch / "sam")],
        }
        for name, command in cases.items():
            measurement = measure(command)
            output = measurement.pop("stdout")
            if name.endswith("only") or name == "ball_exact_id":
                measurement["selected_points"], measurement["input_points"] = count_from_dry_run(output)
            else:
                report = json.loads((scratch / ("fused" if name == "bed_semantic_color" else "sam") /
                                     "preview.json").read_text())
                measurement["selected_points"] = report["selected_points"]
                measurement["input_points"] = scene_points
                if name == "bed_multiview_sam":
                    measurement["added_to_seed"] = report["added_to_seed"]
                    measurement["mask_seed_recall"] = {
                        k: round(v["seed_recall"], 3) for k, v in report["views"].items()}
            result["measurements"][name] = measurement
            print(json.dumps({name: measurement}), flush=True)
        old = np.load(SEEDS, allow_pickle=False)
        fused = np.load(scratch / "fused/selected-indices.npy", allow_pickle=False)
        sam = np.load(scratch / "sam/selected-indices.npy", allow_pickle=False)
        result["selector_agreement"] = {
            "fused_vs_previous_jaccard": round(len(np.intersect1d(fused, old)) /
                                               len(np.union1d(fused, old)), 4),
            "sam_vs_previous_jaccard": round(len(np.intersect1d(sam, old)) /
                                             len(np.union1d(sam, old)), 4),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
