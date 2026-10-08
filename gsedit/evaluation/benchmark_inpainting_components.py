"""Benchmark local LaMa/CoIn-SD2 appearance components, never 3D approval.

Uses training views only. Models inpaint independently here; a plausible image
is not evidence of cross-view depth or a faithful end-to-end method reproduction.
"""
import argparse
import contextlib
import json
from pathlib import Path
import resource
import sys
import time

HOLDOUT = {"frame_0134", "frame_0141", "frame_0143"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=["lama", "coin-sd2"], required=True)
    for key in ["checkpoint", "images", "manifest", "protected-manifest", "output-dir"]:
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--upstream-dir", type=Path)
    p.add_argument("--views", nargs="+", default=None)
    p.add_argument("--width", type=int, default=288)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--margin", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default="an empty indoor room, uninterrupted carpet floor and plain wall, realistic photograph, no furniture in the empty area")
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    if set(a.views or []) & HOLDOUT:
        raise ValueError("Held-out photos must never become generation inputs")
    a.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(provider=a.provider, status="loading", approved=False, is_2d_diagnostic=True,
                  consistent_depth_established=False, source_scene_modified=False,
                  scope="appearance_component_not_end_to_end_method", seed=a.seed,
                  held_out=sorted(HOLDOUT), views=[], width=a.width, steps=a.steps)
    with (a.output_dir / "run.log").open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            import numpy as np
            import torch
            import torch.nn.functional as F
            from PIL import Image, ImageFilter
            torch.manual_seed(a.seed)
            manifest = json.loads(a.manifest.read_text())["views"]
            protection = json.loads(a.protected_manifest.read_text())["views"]
            paths = {f.stem: f for f in a.images.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}}
            eligible = sorted(n for n, v in manifest.items() if n not in HOLDOUT and n in paths and v.get("accepted")
                              and all(v.get("instances", {}).get(k, {}).get("mask_path") for k in ["bed", "wooden_frame"])
                              and protection.get(n, {}).get("accepted") and protection[n].get("mask_path"))
            if a.views is None:
                if len(eligible) < 3:
                    raise ValueError("Need three accepted target/protection views")
                a.views = [eligible[i] for i in np.linspace(0, len(eligible)-1, 3, dtype=int)]
            if not set(a.views) <= set(eligible):
                raise ValueError("Missing accepted target or protection evidence: " + repr(set(a.views)-set(eligible)))
            report["generation_views"] = a.views
            if a.provider == "lama":
                if a.upstream_dir is None:
                    raise ValueError("LaMa source directory is required")
                sys.path.insert(0, str(a.upstream_dir.resolve()))
                import yaml
                from saicinpainting.training.modules import make_generator
                config = yaml.safe_load((a.checkpoint / "config.yaml").read_text())
                from omegaconf import OmegaConf
                params = OmegaConf.to_container(OmegaConf.create(config).generator, resolve=True)
                kind = params.pop("kind")
                model = make_generator(config, kind, **params)
                from gsedit.evaluation.checkpoint_metadata import load_lama_tensor_state
                state = load_lama_tensor_state(a.checkpoint / "models/best.ckpt")
                model.load_state_dict(state, strict=True)
                model.eval().cuda()
                report.update(checkpoint_loaded_strictly=True, adapted_inference="generator only; no Lightning trainer or refinement")
            else:
                from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
                model = StableDiffusionInpaintPipeline.from_single_file(
                    str(a.checkpoint), config="sd2-community/stable-diffusion-2-inpainting",
                    torch_dtype=torch.float16, safety_checker=None)
                model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
                model.to("cuda")
                report.update(prompt=a.prompt, adapted_inference="documented CoIn checkpoint via diffusers; no FreeDoM/3D guidance")
            report["model_load_seconds"] = time.perf_counter()-started
            (a.output_dir / "progress.json").write_text(json.dumps(report, indent=2))
            for i, name in enumerate(a.views):
                if not manifest[name].get("accepted"):
                    raise ValueError("Unaccepted mask: " + name)
                original = Image.open(paths[name]).convert("RGB")
                height = round(original.height*a.width/original.width)
                original = original.resize((a.width, height), Image.Resampling.LANCZOS)
                masks = []
                for instance in ["bed", "wooden_frame"]:
                    path = manifest[name].get("instances", {}).get(instance, {}).get("mask_path")
                    if not path:
                        raise ValueError("Missing independent instance mask")
                    masks.append(np.asarray(Image.open(path).convert("L").resize(original.size, Image.Resampling.NEAREST))>127)
                mask = Image.fromarray((np.logical_or(*masks)*255).astype(np.uint8))
                if a.margin:
                    mask = mask.filter(ImageFilter.MaxFilter(2*a.margin+1))
                protected = np.asarray(Image.open(protection[name]["mask_path"]).convert("L").resize(original.size, Image.Resampling.NEAREST))>127
                mask_array = np.asarray(mask).copy()
                mask_array[protected] = 0
                mask = Image.fromarray(mask_array)
                if not mask_array.any():
                    raise ValueError("Empty unprotected edit mask")
                original.save(a.output_dir / (name+"-input.png"))
                mask.save(a.output_dir / (name+"-mask.png"))
                torch.cuda.synchronize()
                before = time.perf_counter()
                with torch.inference_mode():
                    if a.provider == "lama":
                        image = torch.from_numpy(np.asarray(original).copy()).permute(2, 0, 1).float()[None].cuda()/255
                        tensor_mask = torch.from_numpy(mask_array.copy()).float()[None, None].cuda()/255
                        inputs = torch.cat([image*(1-tensor_mask), tensor_mask], dim=1)
                        pw, ph = (-a.width)%8, (-height)%8
                        predicted = model(F.pad(inputs, (0, pw, 0, ph), mode="reflect"))[:, :, :height, :a.width].clamp(0, 1)
                        generated = Image.fromarray((predicted[0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
                    else:
                        generated = model(prompt=a.prompt, negative_prompt="bed, mattress, wooden frame, dresser, furniture",
                                          image=original, mask_image=mask, width=a.width, height=height,
                                          num_inference_steps=a.steps, guidance_scale=10., eta=1.,
                                          generator=torch.Generator(device="cuda").manual_seed(a.seed+i)).images[0]
                torch.cuda.synchronize()
                duration = time.perf_counter()-before
                # Preserve original image pixels exactly outside the generation mask.
                pixels = np.asarray(generated).copy()
                pixels[mask_array == 0] = np.asarray(original)[mask_array == 0]
                Image.fromarray(pixels).save(a.output_dir / (name+"-generated-2d-only.png"))
                strip = Image.new("RGB", (2*a.width, height), "white")
                strip.paste(original, (0, 0))
                strip.paste(Image.fromarray(pixels), (a.width, 0))
                strip.save(a.output_dir / (name+"-comparison.png"))
                item = dict(view=name, seconds=duration, mask_fraction=float((mask_array>0).mean()),
                            outside_mask_exact=bool(np.array_equal(pixels[mask_array==0], np.asarray(original)[mask_array==0])))
                report["views"].append(item)
                report.update(status="generating", peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                              gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                              total_seconds=time.perf_counter()-started)
                (a.output_dir / "progress.json").write_text(json.dumps(report, indent=2))
                print(json.dumps(item), flush=True)
            report.update(status="completed_2d_component", total_seconds=time.perf_counter()-started)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            report.update(status="failed", error=repr(exc), total_seconds=time.perf_counter()-started)
        (a.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        (a.output_dir / "progress.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
