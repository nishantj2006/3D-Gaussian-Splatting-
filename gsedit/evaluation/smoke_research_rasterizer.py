"""Tiny synthetic forward/backward check, not an object-removal benchmark."""
import argparse
import importlib
import json
from pathlib import Path
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["Split_and_Splat", "Inpaint360GS", "GPGS", "CoIn"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    torch.manual_seed(0)
    started = time.perf_counter()
    module = importlib.import_module("diff_gaussian_rasterization_inpaint360gs"
                                    if args.method == "Inpaint360GS" else "diff_gaussian_rasterization")
    xyz = torch.tensor([[0., 0., 2.], [.2, .1, 2.5]], device="cuda", requires_grad=True)
    xy = torch.zeros_like(xyz, requires_grad=True)
    opacity = torch.full((2, 1), .8, device="cuda", requires_grad=True)
    color = torch.tensor([[.8, .2, .1], [.2, .8, .1]], device="cuda", requires_grad=True)
    scale = torch.full((2, 3), .1, device="cuda", requires_grad=True)
    rotation = torch.tensor([[1., 0., 0., 0.]] * 2, device="cuda", requires_grad=True)
    projection = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                               [0., 0., 1.001, -.01001], [0., 0., 1., 0.]], device="cuda").T.contiguous()
    settings = dict(image_height=32, image_width=32, tanfovx=1., tanfovy=1.,
                    bg=torch.zeros(3, device="cuda"), scale_modifier=1.,
                    viewmatrix=torch.eye(4, device="cuda"), projmatrix=projection,
                    sh_degree=0, campos=torch.zeros(3, device="cuda"), prefiltered=False, debug=args.debug)
    if args.method == "Split_and_Splat":
        settings["antialiasing"] = False
    elif args.method == "GPGS":
        settings.update(kernel_size=.3, require_coord=True, require_depth=True)
    elif args.method == "Inpaint360GS":
        settings["confidence"] = torch.ones_like(opacity)
    kwargs = dict(means3D=xyz, means2D=xy, opacities=opacity,
                  colors_precomp=color, scales=scale, rotations=rotation)
    if args.method == "Inpaint360GS":
        kwargs["sh_objs"] = torch.ones((2, 1, 16), device="cuda", requires_grad=True)
    elif args.method == "CoIn":
        kwargs["uncertainties"] = torch.zeros_like(opacity, requires_grad=True)
    raster = module.GaussianRasterizer(module.GaussianRasterizationSettings(**settings))
    result = raster(**kwargs)
    rgb = result[0]
    assert torch.isfinite(rgb).all() and rgb.sum() > 0, "Invalid or empty rendering"
    rgb.square().mean().backward()
    torch.cuda.synchronize()
    assert all(t.grad is not None and torch.isfinite(t.grad).all()
               for t in [xyz, color, opacity, scale, rotation]), "Missing/nonfinite gradient"
    report = dict(method=args.method, scope="synthetic_renderer_only", inference_run=False,
                  removal_quality_measured=False, forward_backward_passed=True,
                  renderer_path=module.__file__, returned_shapes=[list(t.shape) for t in result],
                  seconds=time.perf_counter() - started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  device_capability=list(torch.cuda.get_device_capability()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
