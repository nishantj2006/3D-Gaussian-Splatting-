"""One-command, non-destructive wall/floor replacement and generation pipeline.

The JSON config names existing camera poses, masks, photos, fit, and either an
existing replacement seed or the inputs to build one. All outputs live under a
new directory. The runner generates a candidate and held-out RGB diagnostics,
but never promotes it to an approved scene automatically.
"""

from gsedit.runtime import PROJECT_ROOT, module_command

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image


ROOT = PROJECT_ROOT


def cli_options(values):
    args = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        args.append("--" + key.replace("_", "-"))
        if value is True:
            continue
        if isinstance(value, list):
            args.extend(map(str, value))
        else:
            args.append(str(value))
    return args


def run_script(name, arguments):
    cmd = [*module_command(name, python=sys.executable)] + cli_options(arguments)
    subprocess.run(cmd, cwd=ROOT, check=True)


def masked_metrics(before, after, mask):
    old = np.asarray(Image.open(before).convert("RGB"), dtype=np.float32)/255
    new = np.asarray(Image.open(after).convert("RGB"), dtype=np.float32)/255
    if old.shape != new.shape:
        raise ValueError("Held-out render dimensions differ")
    bed = np.asarray(Image.open(mask).convert("L").resize(
        (old.shape[1], old.shape[0]), Image.Resampling.NEAREST)) > 127
    if not bed.any() or bed.all():
        raise ValueError("Invalid held-out bed mask")
    delta = np.abs(new-old).mean(axis=2)
    return {"inside_l1_change": float(delta[bed].mean()),
            "outside_l1_change": float(delta[~bed].mean()),
            "outside_fraction_above_0.1": float((delta[~bed] > .1).mean()),
            "inside_near_black_fraction": float((new[bed].max(axis=1) < .02).mean())}


def run(config_path, output_dir):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")
    with open(config_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    holdouts = cfg["holdout_views"]
    if len(holdouts) < 2 or len(set(holdouts)) != len(holdouts):
        raise ValueError("Provide at least two distinct held-out cameras")
    if "replacement" not in cfg and not {"seed", "seed_report"} <= cfg.keys():
        raise ValueError("Config needs replacement inputs or seed + seed_report")
    if "replacement" in cfg and ("seed" in cfg or "seed_report" in cfg):
        raise ValueError("Choose either replacement inputs or an existing seed")
    required = ("wall_fit", "cameras", "bed_manifest", "images")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    output.mkdir(parents=True)
    report = {"config": str(Path(config_path).resolve()), "output": str(output),
              "approved": False, "status": "running", "stages": {}}
    started = time.perf_counter()
    try:
        if "replacement" in cfg:
            replacement = dict(cfg["replacement"])
            replacement["output_dir"] = str(output / "replacement")
            replacement["holdout_views"] = holdouts
            run_script("build_continuous_background.py", replacement)
            seed = output / "replacement" / "candidate.ply"
            seed_report = output / "replacement" / "report.json"
            report["stages"]["replacement"] = "built"
        else:
            seed = Path(cfg["seed"]).resolve()
            seed_report = Path(cfg["seed_report"]).resolve()
            report["stages"]["replacement"] = "existing_unapproved_seed"
        generation = dict(cfg.get("generation", {}))
        generation.update({"seed": str(seed), "seed_report": str(seed_report),
                           "wall_fit": cfg["wall_fit"], "cameras": cfg["cameras"],
                           "bed_manifest": cfg["bed_manifest"],
                           "frame_manifest": cfg.get("frame_manifest"),
                           "images": cfg["images"], "holdout_views": holdouts,
                           "output_dir": str(output / "generated")})
        run_script("generate_background_views.py", generation)
        with open(output / "generated" / "report.json", encoding="utf-8") as handle:
            generated_report = json.load(handle)
        report["stages"]["generation"] = generated_report
        with open(cfg["bed_manifest"], encoding="utf-8") as handle:
            masks = json.load(handle)["views"]
        (output / "heldout-seed").mkdir()
        (output / "heldout-generated").mkdir()
        measurements = {}
        for view in holdouts:
            if view not in masks or not masks[view].get("accepted"):
                raise ValueError(f"No accepted held-out mask for {view}")
            before = output / "heldout-seed" / f"{view}.png"
            after = output / "heldout-generated" / f"{view}.png"
            common = {"cameras": cfg["cameras"], "image_name": view,
                      "width": cfg.get("render_width", 540)}
            run_script("render_ply_preview.py", dict(common, ply=str(seed),
                                                      output=str(before)))
            run_script("render_ply_preview.py", dict(common,
                       ply=generated_report["candidate"], output=str(after)))
            measurements[view] = masked_metrics(
                before, after, masks[view]["mask_path"])
        report["stages"]["heldout"] = measurements
        report["status"] = ("needs_visual_review" if
                            generated_report["automatic_checks_pass"] else
                            "automatic_checks_failed")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        with open(output / "workflow-report.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="Resume a configured replacement seed")
    p.add_argument("--scene", help="Original 128D scene PLY for text-targeted mode")
    p.add_argument("--images")
    p.add_argument("--cameras")
    p.add_argument("--text", help="Object to remove")
    p.add_argument("--pca-path")
    p.add_argument("--holdout-views", nargs="+")
    p.add_argument("--sam-model", default="mobile_sam.pt")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--mask-device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--backend", choices=("diffusion", "opencv"), default="diffusion")
    p.add_argument("--model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--min-wall-inlier-ratio", type=float, default=.3)
    p.add_argument("--skip-optimization", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if bool(args.config) == bool(args.scene):
        raise ValueError("Provide exactly one of --config or --scene")
    if args.config:
        result = run(args.config, args.output_dir)
    else:
        if not all((args.images, args.cameras, args.text, args.pca_path)):
            raise ValueError("Text mode needs images, cameras, text, and pca-path")
        from gsedit.pipelines.run_autonomous_edit import run as run_text_target
        result = run_text_target(args)
    print(json.dumps(result, indent=2))
