"""Generate an isolated object image with local SDXL for add_asset.py.

This optional stage runs in its own Python interpreter, so diffusers need not
be installed in the scene-training environment. Model weights are downloaded
by Diffusers on first use and cached by Hugging Face.
"""

import argparse
from pathlib import Path


def object_prompt(description):
    return ("A single complete 3D object: " + description + ". "
            "Centered and fully visible, isolated on a plain white background, "
            "three-quarter product-photo view, natural proportions, sharp detail. "
            "No floor, cast shadow, hands, text, or other objects.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def validate_args(args):
    if args.output.suffix.lower() != ".png":
        raise ValueError("--output must be a PNG path")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not 1 <= args.steps <= 150:
        raise ValueError("--steps must be between 1 and 150")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    for label, size in (("width", args.width), ("height", args.height)):
        if size < 512 or size > 2048 or size % 8:
            raise ValueError(f"--{label} must be a multiple of 8 from 512 to 2048")


def generate(args):
    validate_args(args)
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Local SDXL requires torch, diffusers, transformers, accelerate, "
            "and safetensors in --image-python's environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Local SDXL requires a CUDA-enabled PyTorch environment")

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.inference_mode():
        image = pipeline(
            prompt=object_prompt(args.prompt),
            negative_prompt=("multiple objects, cropped object, busy background, "
                             "floor, shadows, people, hands, text, logo, watermark"),
            num_inference_steps=args.steps,
            width=args.width,
            height=args.height,
            generator=generator,
        ).images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG")
    print(f"Saved local SDXL image to {args.output}")


if __name__ == "__main__":
    generate(parse_args())
