"""Compare an instance-feature trial with the current preview in held-out RGB."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["trial-dir", "baseline-dir", "cameras", "target-manifest", "protected-manifest", "floor-fit", "output-dir"]:
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--width", type=int, default=540)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    from scene.gaussian_model import GaussianModel
    from gsedit.evaluation.evaluate_rendered_masks import camera_settings, render_selection, scores
    from gsedit.rendering.render_pruned_preview import render_scene
    from gsedit.rendering.render_overhead_preview import overhead_camera
    from gsedit.selection.spatial_instance_removal import load_mask
    from plyfile import PlyData
    started = time.perf_counter()
    prep = json.loads((a.trial_dir / "report.json").read_text())
    if prep["status"] != "completed_component_trial":
        raise ValueError("Trial is incomplete")
    source = json.loads((a.baseline_dir / "report.json").read_text())["source"]
    source = json.loads((Path(source) / "report.json").read_text())["source"]
    vertices = PlyData.read(source)["vertex"].data
    ids = np.load(a.trial_dir / "selected-source-ids.npy")
    candidate_vertices = PlyData.read(str(a.trial_dir / "candidate-unapproved.ply"))["vertex"].data
    keep = np.ones(len(vertices), bool)
    keep[ids] = False
    if vertices.dtype != candidate_vertices.dtype or not np.array_equal(vertices[keep], candidate_vertices):
        raise AssertionError("Retained records differ")
    a.output_dir.mkdir(parents=True)
    models = []
    for path in [source, a.baseline_dir / "candidate.ply", a.trial_dir / "candidate-unapproved.ply"]:
        model = GaussianModel(3, 128)
        model.load_ply(str(path))
        models.append(model)
    cameras = {c["img_name"]: c for c in json.loads(a.cameras.read_text())}
    targets, protection = json.loads(a.target_manifest.read_text()), json.loads(a.protected_manifest.read_text())
    sheets, metrics = [], {}
    with torch.no_grad():
        for name in ["frame_0134", "frame_0141", "frame_0143"]:
            c = cameras[name]
            height = round(c["height"]*a.width/c["width"])
            raster = GaussianRasterizer(camera_settings(c, height, a.width)._replace(sh_degree=3))
            rgbs = [render_scene(m, raster, m.get_opacity) for m in models]
            target = load_mask(targets["views"][name]["instances"]["bed"]["mask_path"], (a.width, height))
            protected = load_mask(protection["views"][name]["mask_path"], (a.width, height))
            clean = target & ~protected
            metrics[name] = {}
            sheet = Image.new("RGB", (3*a.width, height+30), "white")
            draw = ImageDraw.Draw(sheet)
            for i, (label, rgb) in enumerate(zip(["Source", "Current preview", "Inpaint features"], rgbs)):
                sheet.paste(Image.fromarray(rgb), (i*a.width, 30))
                draw.text((i*a.width+5, 6), name+" / "+label, fill="black")
                Image.fromarray(rgb).save(a.output_dir / (name+"-"+str(i)+".png"))
                if i:
                    delta = np.abs(rgbs[0].astype(float)-rgb.astype(float))/255
                    metrics[name][label] = dict(protected_rgb_mae=float(delta[protected].mean()),
                                               protected_changed_fraction=float((delta[protected].max(axis=1)>.08).mean()),
                                               outside_changed_fraction=float((delta[~target & ~protected].max(axis=1)>.08).mean()),
                                               target_black_fraction=float((rgb[clean].max(axis=1)<8).mean()))
            footprint, _ = render_selection(models[0], raster, ids)
            metrics[name]["trial_front_footprint"] = scores(footprint, torch.from_numpy(clean).cuda(), .1)
            sheets.append(sheet)
        floor = json.loads(a.floor_fit.read_text())
        center = np.median(np.column_stack([vertices[k][ids] for k in ["x", "y", "z"]]), axis=0)
        overhead = overhead_camera(center, np.asarray(floor["plane_normal_toward_removed_object"]), cameras["frame_0141"], 5., a.width)
        raster = GaussianRasterizer(camera_settings(overhead, a.width, a.width)._replace(sh_degree=3))
        for i, model in enumerate(models):
            Image.fromarray(render_scene(model, raster, model.get_opacity)).save(a.output_dir / ("overhead-"+str(i)+".png"))
    comparison = Image.new("RGB", (3*a.width, sum(s.height for s in sheets)), "white")
    offset = 0
    for sheet in sheets:
        comparison.paste(sheet, (0, offset))
        offset += sheet.height
    comparison.save(a.output_dir / "comparison.png")
    report = dict(approved=False, retained_records_exact=True, source_properties=len(vertices.dtype.names),
                  semantic_dimensions=sum(k.startswith("semantic_") for k in vertices.dtype.names),
                  views=metrics, seconds=time.perf_counter()-started,
                  warnings=["Protection masks are furniture evidence, not verified dresser ground truth.",
                            "Front-footprint recall does not measure revealed bed layers.",
                            "This is a removal-only comparison, not background reconstruction."])
    (a.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
