"""Run text-to-mask and learned-appearance removal as one unapproved workflow."""

from gsedit.runtime import PROJECT_ROOT, module_command

import argparse
import json
from pathlib import Path
import subprocess
import sys


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse preview folder: {output}")
    source = Path(args.scene).resolve()
    script_dir = PROJECT_ROOT
    seeds = output / "image-seeds"
    expanded = output / "appearance-expanded"
    subprocess.run([*module_command("auto_object_preview.py", python=sys.executable),
        "--scene", str(source), "--text", args.text,
        "--cameras", args.cameras, "--images", args.images,
        "--pca-path", args.pca_path, "--sam-model", args.sam_model,
        "--device", args.device, "--view-count", str(args.view_count),
        "--exclude-views", *args.exclude_views,
        "--min-views", str(args.min_views), "--output-dir", str(seeds),
    ], check=True)
    subprocess.run([*module_command("refine_auto_appearance.py", python=sys.executable),
        "--scene", str(source), "--seed-preview", str(seeds),
        "--pca-path", args.pca_path, "--radius", str(args.grow_radius),
        "--output-dir", str(expanded),
    ], check=True)
    with open(seeds / "preview.json", encoding="utf-8") as handle:
        seed_report = json.load(handle)
    with open(expanded / "preview.json", encoding="utf-8") as handle:
        expanded_report = json.load(handle)
    summary = {"source": str(source), "text": args.text,
               "automatic_views": list(seed_report["views"]),
               "positive_vector_only_points": seed_report["vector_only_count"],
               "positive_vector_only_safe": seed_report["vector_only_safe"],
               "image_seed_points": seed_report["selected_points"],
               "appearance_added": expanded_report["appearance_added"],
               "final_selected_points": expanded_report["selected_points"],
               "candidate_ply": str(expanded / "pruned-preview.ply"),
               "approved": False,
               "warning": "Preview only; inspect masks and multiple views before removing an object."}
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--pca-path", required=True)
    p.add_argument("--sam-model", default="mobile_sam.pt")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--view-count", type=int, default=2)
    p.add_argument("--min-views", type=int, default=1)
    p.add_argument("--exclude-views", nargs="*", default=[])
    p.add_argument("--grow-radius", type=float, default=0.35)
    p.add_argument("--output-dir", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
