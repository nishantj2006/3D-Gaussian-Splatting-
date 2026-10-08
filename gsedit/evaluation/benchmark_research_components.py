"""Bounded real-capture component trials, not end-to-end paper reproductions.

Inpaint360GS trains its native 16D object features with separate bed/frame labels
on the frozen source. GPGS and CoIn exercise their native reconstruction models.
No hidden geometry is invented and no scene is overwritten or approved.
"""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace

HOLDOUT = {"frame_0134", "frame_0141", "frame_0143"}


def label_instances(bed, frame):
    import numpy as np
    labels = np.zeros(bed.shape, dtype=np.uint8)
    labels[bed] = 1
    labels[frame] = 2
    labels[bed & frame] = 255  # Ambiguous evidence is not a supervised class.
    return labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["Inpaint360GS", "GPGS", "CoIn"], required=True)
    for key in ["upstream-dir", "source-ply", "cameras", "images", "manifest", "protected-ids", "output-dir"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--width", type=int, default=270)
    p.add_argument("--points", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    if a.steps < 30 or a.points < 100 or a.width < 64:
        raise ValueError("Insufficient pilot steps, points, or resolution")
    out.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(method=a.method, status="preparing", scope="component_trial_not_end_to_end_method",
                  held_out=sorted(HOLDOUT), source_modified=False, approved=False,
                  background_reconstructed=False, steps=a.steps, width=a.width, seed=a.seed)
    with (out / "run.log").open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            sys.path.insert(0, str(a.upstream_dir.resolve()))
            import numpy as np
            import torch
            from PIL import Image
            from plyfile import PlyData, PlyElement
            from gaussian_renderer import render
            from scene.gaussian_model import GaussianModel, BasicPointCloud
            from utils.graphics_utils import getProjectionMatrix
            from arguments import ModelParams, OptimizationParams
            from utils.loss_utils import ssim
            torch.manual_seed(a.seed)
            np.random.seed(a.seed)
            rng = np.random.default_rng(a.seed)
            source = PlyData.read(str(a.source_ply), mmap="r")
            vertices = source["vertex"].data
            report["source_vertex_sha256"] = hashlib.sha256(vertices.tobytes()).hexdigest()
            manifest = json.loads(a.manifest.read_text())["views"]
            camera_entries = {c["img_name"]: c for c in json.loads(a.cameras.read_text())}
            paths = {f.stem: f for f in a.images.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}}
            train_names = sorted(n for n, v in manifest.items() if n not in HOLDOUT and v.get("accepted") and n in paths
                                 and all(v.get("instances", {}).get(k, {}).get("mask_path") for k in ["bed", "wooden_frame"]))
            if len(train_names) < 3:
                raise ValueError("Need at least three non-held-out accepted views")

            def camera(name, uid):
                c = camera_entries[name]
                height = round(a.width * c["height"] / c["width"])
                rgb = np.asarray(Image.open(paths[name]).convert("RGB").resize((a.width, height), Image.Resampling.LANCZOS)).copy()
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = c["rotation"]
                c2w[:3, 3] = c["position"]
                w2c = np.linalg.inv(c2w)  # CPU inversion avoids Jetson cuSolver incompatibility.
                view = torch.from_numpy(w2c).T.contiguous().cuda()
                fx, fy = 2*math.atan(c["width"]/(2*c["fx"])), 2*math.atan(c["height"]/(2*c["fy"]))
                projection = getProjectionMatrix(.01, 100., fx, fy).T.cuda()
                result = SimpleNamespace(image_width=a.width, image_height=height, FoVx=fx, FoVy=fy,
                                         world_view_transform=view, full_proj_transform=view @ projection,
                                         camera_center=torch.tensor(c["position"], dtype=torch.float32, device="cuda"),
                                         R=np.asarray(c["rotation"]), T=w2c[:3, 3], uid=uid, image_name=name,
                                         original_image=torch.from_numpy(rgb).permute(2, 0, 1).float().cuda()/255)
                masks = []
                for instance in ["bed", "wooden_frame"]:
                    path = manifest.get(name, {}).get("instances", {}).get(instance, {}).get("mask_path")
                    if not path:
                        raise ValueError("Missing independent instance mask: " + name + " " + instance)
                    masks.append(np.asarray(Image.open(path).convert("L").resize((a.width, height), Image.Resampling.NEAREST)) > 127)
                result.objects = torch.from_numpy(label_instances(*masks).astype(np.int64)).cuda()
                return result

            cams = [camera(n, i) for i, n in enumerate(train_names)]
            report["training_views"] = train_names
            parser = argparse.ArgumentParser()
            mp, op = ModelParams(parser), OptimizationParams(parser)
            args = parser.parse_args([])
            options = op.extract(args)
            options.iterations = a.steps
            pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
            background = torch.zeros(3, device="cuda")
            classifier = None
            if a.method == "Inpaint360GS":
                model = GaussianModel(3)
                model.load_ply(str(a.source_ply))
                model.spatial_lr_scale = 1.
                model.training_setup_distill(options)
                classifier = torch.nn.Conv2d(model.num_objects, 3, 1).cuda()
                cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
                # Balanced labels are an explicit adaptation, not the upstream
                # uniform-pixel loss. Retain ambiguous overlaps as ignored.
                counts = torch.bincount(torch.cat([c.objects.flatten() for c in cams]), minlength=256)[:3].float()
                weights = counts.sum() / counts.clamp_min(1)
                weights /= weights.mean()
                criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=255)
                report.update(component="native_object_feature_distillation_adapted", geometry_frozen=True,
                              regularization_3d=False, original_semantics_untouched=True,
                              label_classes={"0": "background", "1": "bed", "2": "wooden_frame"})
            else:
                ids = np.sort(rng.choice(len(vertices), min(a.points, len(vertices)), replace=False))
                xyz = np.column_stack([vertices[k][ids] for k in ["x", "y", "z"]])
                rgb = np.clip(.5 + .28209479177387814 * np.column_stack([vertices[f"f_dc_{i}"][ids] for i in range(3)]), 0, 1)
                cloud = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros_like(xyz))
                np.save(out / "initial-source-ids.npy", ids)
                if a.method == "CoIn":
                    conf = SimpleNamespace(**vars(mp.extract(args)))
                    for key, value in vars(options).items():
                        setattr(conf, key, value)
                    for key, value in dict(use_attn=False, use_inmask=False, add_opacity_dist=False,
                                           add_cov_dist=False, add_color_dist=False, appearance_dim=0,
                                           full_iteration=a.steps, voxel_size=.005).items():
                        setattr(conf, key, value)
                    model = GaussianModel(conf)
                    model.create_from_pcd(cloud, cams, 1.)
                else:
                    model = GaussianModel(3)
                    model.create_from_pcd(cloud, 1.)
                    model.compute_3D_filter(cams)
                model.training_setup(options)
                report.update(component="native_rgb_reconstruction_adapted_loop", densification=False,
                              initial_sample_points=len(ids), raw_model_not_128d=True)
            times, losses = [], []
            training_start = time.perf_counter()
            for iteration in range(1, a.steps+1):
                cam = cams[int(rng.integers(len(cams)))]
                torch.cuda.synchronize()
                step_start = time.perf_counter()
                model.optimizer.zero_grad(set_to_none=True)
                if classifier is not None:
                    cls_optimizer.zero_grad(set_to_none=True)
                if a.method == "GPGS":
                    pkg = render(cam, model, pipe, background, .3, require_depth=True, require_coord=False)
                else:
                    pkg = render(cam, model, pipe, background)
                if classifier is not None:
                    logits = classifier(pkg["render_object"])
                    loss = criterion(logits[None], cam.objects[None])
                else:
                    image = pkg["render"]
                    loss = .8*(image-cam.original_image).abs().mean()+.2*(1-ssim(image, cam.original_image))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                model.optimizer.step()
                if classifier is not None:
                    cls_optimizer.step()
                torch.cuda.synchronize()
                times.append(time.perf_counter()-step_start)
                losses.append(float(loss.item()))
                if iteration % 10 == 0 or iteration == a.steps:
                    report.update(status="training", iteration=iteration, loss=losses[-1],
                                  median_step_seconds=float(np.median(times[20:] or times)),
                                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                                  gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                                  total_seconds=time.perf_counter()-started)
                    (out / "progress.json").write_text(json.dumps(report, indent=2))
                    print(json.dumps(report), flush=True)
            report["training_seconds"] = time.perf_counter()-training_start
            (out / "renders").mkdir()
            evaluations = []
            for i, name in enumerate(sorted(HOLDOUT)):
                if name not in manifest or name not in paths:
                    raise ValueError("Missing held-out evidence: " + name)
                cam = camera(name, i)
                with torch.no_grad():
                    if a.method == "GPGS":
                        pkg = render(cam, model, pipe, background, .3, require_depth=True, require_coord=False)
                    else:
                        pkg = render(cam, model, pipe, background)
                    rgb = pkg["render"].clamp(0, 1)
                    Image.fromarray((rgb.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)).save(out / "renders" / (name+"-rgb.png"))
                    item = dict(view=name, rgb_l1=float((rgb-cam.original_image).abs().mean().item()))
                    if classifier is not None:
                        prediction = classifier(pkg["render_object"]).argmax(dim=0)
                        Image.fromarray((prediction.cpu().numpy()*100).astype(np.uint8)).save(out / "renders" / (name+"-labels.png"))
                        for label, title in [(1, "bed"), (2, "frame")]:
                            valid = cam.objects != 255
                            target, selected = (cam.objects == label)&valid, (prediction == label)&valid
                            item[title+"_iou"] = float((target&selected).sum().item()/max(1, (target|selected).sum().item()))
                    evaluations.append(item)
            report["held_out_metrics"] = evaluations
            if classifier is not None:
                with torch.no_grad():
                    probabilities = torch.softmax(classifier(model._objects_dc.permute(2, 0, 1)).squeeze(-1).T, dim=1).cpu().numpy()
                np.save(out / "source-instance-probabilities.npy", probabilities)
                protected = np.load(a.protected_ids)
                selected = np.flatnonzero((probabilities[:, 1] > .7) | (probabilities[:, 2] > .7))
                report["raw_selected_protected"] = len(np.intersect1d(selected, protected))
                selected = np.setdiff1d(selected, protected)
                np.save(out / "selected-source-ids.npy", selected)
                keep = np.ones(len(vertices), dtype=bool)
                keep[selected] = False
                elements = [PlyElement.describe(vertices[keep].copy(), "vertex") if e.name == "vertex" else e for e in source.elements]
                candidate = PlyData(elements, text=source.text, byte_order=source.byte_order, comments=source.comments, obj_info=source.obj_info)
                candidate.write(str(out / "candidate-unapproved.ply"))
                report.update(removed=len(selected), protected_removed=0,
                              retained_vertex_sha256=hashlib.sha256(vertices[keep].tobytes()).hexdigest(),
                              properties_preserved=len(vertices.dtype.names),
                              semantics_preserved=sum(n.startswith("semantic_") for n in vertices.dtype.names))
                # Save removal RGB using exactly the same native renderer.
                model._opacity.data[torch.from_numpy(selected).cuda()] = -100
                with torch.no_grad():
                    for i, name in enumerate(sorted(HOLDOUT)):
                        rgb = render(camera(name, i), model, pipe, background)["render"].clamp(0, 1)
                        Image.fromarray((rgb.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)).save(out / "renders" / (name+"-removed.png"))
                torch.save(dict(features=model._objects_dc.detach().cpu(), classifier=classifier.state_dict()), out / "object-feature-checkpoint.pt")
            else:
                model.save_ply(str(out / "native-reconstruction.ply"))
                if a.method == "CoIn":
                    model.save_mlp_checkpoints(str(out))
            report.update(status="completed_component_trial", total_seconds=time.perf_counter()-started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                          gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        except Exception as exc:
            traceback.print_exc()
            report.update(status="failed", error=repr(exc), total_seconds=time.perf_counter()-started)
        (out / "report.json").write_text(json.dumps(report, indent=2))
        (out / "progress.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
