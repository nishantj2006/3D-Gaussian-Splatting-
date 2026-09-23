"""Generate, align, and merge a semantic Gaussian asset into an existing scene.

Provide --image/--mesh to resume at an intermediate stage. Nano Banana needs
GEMINI_API_KEY; local SDXL uses an optional Python environment. TripoSR must
be installed separately. The source scene is never modified, and each run
requires a new output directory.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def generate_nano_image(prompt, output, model):
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY in the running environment for Nano Banana")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install google-genai in the active conda environment") from exc
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    result = client.models.generate_content(
        model=model,
        contents=("A single complete 3D object: " + prompt + ". "
                  "Centered, fully visible, isolated on a plain white background; "
                  "no floor, shadows, text, or other objects. Three-quarter view."),
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    for candidate in result.candidates or []:
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if part.inline_data and part.inline_data.data:
                output.write_bytes(part.inline_data.data)
                return
    raise RuntimeError("Nano Banana returned no image; inspect the prompt and API response")


def local_image_command(args, output):
    """Build the isolated image-generation step without loading model weights."""
    return [str(args.image_python), str(ROOT / "generate_local_image.py"),
            "--prompt", args.prompt, "--output", str(output),
            "--model", args.local_model, "--steps", str(args.local_steps),
            "--width", str(args.local_width), "--height", str(args.local_height),
            "--seed", str(args.seed)]


def validate_generation_requirements(args):
    """Check the chosen provider before creating the output directory."""
    if args.mesh or args.image:
        return
    if args.image_provider == "nano" and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Nano Banana requires GEMINI_API_KEY, or use "
                           "--image-provider local-sdxl/--image/--mesh")
    if args.image_provider == "local-sdxl":
        if not args.image_python.is_file():
            raise FileNotFoundError(f"SDXL Python not found: {args.image_python}")
        if not (1 <= args.local_steps <= 150):
            raise ValueError("--local-steps must be between 1 and 150")
        for label, size in (("width", args.local_width), ("height", args.local_height)):
            if size < 512 or size > 2048 or size % 8:
                raise ValueError(f"--local-{label} must be a multiple of 8 from 512 to 2048")


def run_stage(command, expected):
    subprocess.run(command, check=True, cwd=ROOT)
    if not expected.is_file() or expected.stat().st_size == 0:
        raise RuntimeError(f"Stage completed but expected output is missing: {expected}")


def alignment_target(args):
    if args.replace_original:
        if not args.replace_pruned:
            raise ValueError("--replace-pruned is required with --replace-original")
        return ["--replace-original", str(args.replace_original),
                "--replace-pruned", str(args.replace_pruned)]
    if args.replace_pruned:
        raise ValueError("--replace-original is required with --replace-pruned")
    if args.target_bbox:
        return ["--target-bbox", *map(str, args.target_bbox)]
    if args.target_center and args.target_size:
        return ["--target-center", *map(str, args.target_center),
                "--target-size", *map(str, args.target_size)]
    raise ValueError("Choose a removal pair, target bounding box, or target center and size")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--prompt", required=True, help="Object description and semantic label")
    cli.add_argument("--scene", type=Path, required=True, help="Scene PLY to add to")
    cli.add_argument("--plane-json", type=Path, required=True, help="Fitted ground-plane preview.json")
    cli.add_argument("--output-dir", type=Path, required=True, help="New, nonexistent result directory")
    cli.add_argument("--object-id", type=int, required=True)
    cli.add_argument("--image", type=Path, help="Existing generated/reference image; skip image generation")
    cli.add_argument("--mesh", type=Path, help="Existing TripoSR mesh; skip image generation and TripoSR")
    cli.add_argument("--image-provider", choices=("nano", "local-sdxl"), default="nano",
                     help="Text-to-image backend when --image/--mesh is absent")
    cli.add_argument("--nano-model", default="gemini-3.1-flash-image")
    cli.add_argument("--image-python", type=Path, default=Path(sys.executable),
                     help="Python with diffusers/PyTorch for local SDXL")
    cli.add_argument("--local-model", default="stabilityai/stable-diffusion-xl-base-1.0",
                     help="SDXL model ID or local model directory")
    cli.add_argument("--local-steps", type=int, default=25)
    cli.add_argument("--local-width", type=int, default=1024)
    cli.add_argument("--local-height", type=int, default=1024)
    cli.add_argument("--tripo-python", type=Path, default=Path(sys.executable))
    cli.add_argument("--tripo-script", type=Path, default=ROOT / "external_tools/TripoSR/run.py")
    target = cli.add_mutually_exclusive_group(required=True)
    target.add_argument("--replace-original", type=Path)
    target.add_argument("--target-bbox", type=float, nargs=6)
    target.add_argument("--target-center", type=float, nargs=3)
    cli.add_argument("--replace-pruned", type=Path)
    cli.add_argument("--target-size", type=float, nargs=3)
    cli.add_argument("--asset-up-axis", choices=("x", "-x", "y", "-y", "z", "-z"), default="z")
    cli.add_argument("--yaw-deg", type=float)
    cli.add_argument("--points", type=int, default=60000)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--pca-path", type=Path, default=ROOT / "data/my_scene/pca_model_128.pkl")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    for field in ("scene", "plane_json", "pca_path", "image", "mesh", "image_python", "tripo_python",
                  "tripo_script", "replace_original", "replace_pruned"):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).resolve())
    target_flags = alignment_target(args)
    if args.target_center and not args.target_size:
        raise ValueError("--target-size is required with --target-center")
    if args.target_size and not args.target_center:
        raise ValueError("--target-center is required with --target-size")
    if args.object_id <= 0:
        raise ValueError("--object-id must be positive")
    if args.image and not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.mesh and not args.mesh.is_file():
        raise FileNotFoundError(args.mesh)
    for required in (args.scene, args.plane_json, args.pca_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not args.mesh and not args.tripo_script.is_file():
        raise FileNotFoundError("TripoSR not installed; provide --mesh or install its run.py")
    validate_generation_requirements(args)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")
    output.mkdir(parents=True)

    image = args.image.resolve() if args.image else None
    mesh = args.mesh.resolve() if args.mesh else None
    record = {"prompt": args.prompt, "scene": str(args.scene.resolve()),
              "object_id": args.object_id, "target": target_flags,
              "image_provider": args.image_provider, "nano_model": args.nano_model,
              "local_model": args.local_model if args.image_provider == "local-sdxl" else None,
              "local_steps": args.local_steps if args.image_provider == "local-sdxl" else None,
              "local_width": args.local_width if args.image_provider == "local-sdxl" else None,
              "local_height": args.local_height if args.image_provider == "local-sdxl" else None,
              "image_python": str(args.image_python) if args.image_provider == "local-sdxl" else None,
              "points": args.points, "seed": args.seed}
    try:
        if mesh is None:
            if image is None:
                if args.image_provider == "nano":
                    image = output / "nano.png"
                    generate_nano_image(args.prompt, image, args.nano_model)
                else:
                    image = output / "local-sdxl.png"
                    run_stage(local_image_command(args, image), image)
            tripo_output = output / "tripo"
            run_stage([str(args.tripo_python), str(args.tripo_script), str(image),
                       "--output-dir", str(tripo_output), "--model-save-format", "obj"],
                      tripo_output / "0/mesh.obj")
            mesh = tripo_output / "0/mesh.obj"
        record.update({"image": str(image) if image else None, "mesh": str(mesh)})
        raw = output / "asset-raw.ply"
        bridge_command = [sys.executable, str(ROOT / "bridge.py"), "--mesh", str(mesh),
                          "--output", str(raw), "--label", args.prompt,
                          "--object-id", str(args.object_id), "--pca-path", str(args.pca_path),
                          "--points", str(args.points), "--seed", str(args.seed)]
        if image:
            bridge_command += ["--reference-image", str(image)]
        run_stage(bridge_command, raw)
        aligned = output / "asset-aligned.ply"
        align_command = [sys.executable, str(ROOT / "align_asset.py"),
                         "--asset", str(raw), "--output", str(aligned),
                         "--plane-json", str(args.plane_json),
                         "--asset-up-axis", args.asset_up_axis, *target_flags]
        if args.yaw_deg is not None:
            align_command += ["--yaw-deg", str(args.yaw_deg)]
        run_stage(align_command, aligned)
        merged = output / "scene-with-asset.ply"
        run_stage([sys.executable, str(ROOT / "merge.py"), "--scene", str(args.scene),
                   "--asset", str(aligned), "--output", str(merged),
                   "--object-id", str(args.object_id), "--label", args.prompt], merged)
        record["outputs"] = {"raw": str(raw), "aligned": str(aligned), "merged": str(merged)}
    finally:
        (output / "pipeline.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Asset inserted into {merged}; source scene was not changed")


if __name__ == "__main__":
    main()
