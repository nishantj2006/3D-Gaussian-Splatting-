"""Match a flat-fill preview's low-frequency color to its surrounding surface.

This edits only the RGB coefficients of the added fill splats. Geometry,
opacity, and semantic features stay byte-identical. The source preview is
never overwritten, and the result still requires visual approval.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from reconstruct_flat import sha256
from utils.ply_semantic_utils import read_vertices, write_vertices


def project(xyz, camera, width, height):
    local = (xyz - np.asarray(camera["position"])) @ np.asarray(camera["rotation"])
    x = (camera["fx"] * local[:, 0] / local[:, 2] + camera["width"] / 2) * width / camera["width"]
    y = (camera["fy"] * local[:, 1] / local[:, 2] + camera["height"] / 2) * height / camera["height"]
    return x, y, local[:, 2] > 0


def fill_mask(vertices, fine_count, added_count, camera, width, height):
    first = len(vertices) - added_count
    fine = vertices[first:first + fine_count]
    xyz = np.column_stack([fine[field] for field in ("x", "y", "z")])
    x, y, front = project(xyz, camera, width, height)
    ix, iy = np.rint(x).astype(int), np.rint(y).astype(int)
    valid = front & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    mask = np.zeros((height, width), dtype=bool)
    mask[iy[valid], ix[valid]] = True
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_dilation(mask, iterations=2)
    fraction = float(mask.mean())
    if not 0.005 < fraction < 0.6:
        raise ValueError("Reference view does not show a localized fill")
    return mask


def boundary_correction(image, mask):
    """Fit a robust affine RGB correction to low-pass color across the edge."""
    if image.shape[:2] != mask.shape or image.shape[2] != 3:
        raise ValueError("Reference render and projected fill mask have different sizes")
    height, width = mask.shape
    smooth = ndimage.gaussian_filter(image.astype(np.float64) / 255,
                                     sigma=(6, 6, 0))
    signed = ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)
    grad_y, grad_x = np.gradient(signed)
    edge = mask & ~ndimage.binary_erosion(mask)
    yy, xx = np.where(edge)
    length = np.hypot(grad_y[yy, xx], grad_x[yy, xx])
    valid = ((length > 0.1) & (xx > 20) & (xx < width - 20)
             & (yy > 20) & (yy < height - 20))
    yy, xx, length = yy[valid], xx[valid], length[valid]
    if len(xx) < 100:
        raise ValueError("Too little visible fill boundary for color matching")
    direction_y = grad_y[yy, xx] / length
    direction_x = grad_x[yy, xx] / length
    offset = 8
    inner_y = np.clip(yy + offset * direction_y, 0, height - 1)
    inner_x = np.clip(xx + offset * direction_x, 0, width - 1)
    outer_y = np.clip(yy - offset * direction_y, 0, height - 1)
    outer_x = np.clip(xx - offset * direction_x, 0, width - 1)
    inside = ndimage.map_coordinates(mask.astype(np.float32),
                                     [inner_y, inner_x], order=0) > 0
    outside = ndimage.map_coordinates(mask.astype(np.float32),
                                      [outer_y, outer_x], order=0) < 1
    good = inside & outside
    yy, xx = yy[good], xx[good]
    if len(xx) < 100:
        raise ValueError("Not enough intact carpet immediately outside the fill")
    inner = np.column_stack([
        ndimage.map_coordinates(smooth[:, :, channel],
                                [inner_y[good], inner_x[good]], order=1)
        for channel in range(3)])
    outer = np.column_stack([
        ndimage.map_coordinates(smooth[:, :, channel],
                                [outer_y[good], outer_x[good]], order=1)
        for channel in range(3)])
    residual = np.clip(outer - inner, -0.15, 0.15)
    design = design_matrix(xx, yy, width, height)
    coefficients = np.linalg.lstsq(design, residual, rcond=None)[0]
    for _ in range(5):
        error = np.mean(np.abs(design @ coefficients - residual), axis=1)
        weights = np.clip(0.03 / np.maximum(error, 1e-6), 0.15, 1)
        root = np.sqrt(weights)
        matrix = np.vstack((design * root[:, None], np.diag([0, 0.4, 0.4])))
        values = np.vstack((residual * root[:, None], np.zeros((3, 3))))
        coefficients = np.linalg.lstsq(matrix, values, rcond=None)[0]
    return coefficients, {
        "boundary_samples": int(len(xx)),
        "median_absolute_boundary_rgb": np.median(np.abs(inner - outer), axis=0).tolist(),
    }


def design_matrix(x, y, width, height):
    return np.column_stack((np.ones(len(x)),
                            (x - width / 2) / (width / 3),
                            (y - height / 2) / (height / 3)))


def adjust_added_colors(vertices, added_count, camera, width, height,
                        coefficients, strength):
    if not 0 < strength <= 2:
        raise ValueError("--strength must be in (0, 2]")
    result = vertices.copy()
    xyz = np.column_stack([vertices[field][-added_count:]
                           for field in ("x", "y", "z")])
    x, y, _ = project(xyz, camera, width, height)
    correction = np.clip(strength * (design_matrix(x, y, width, height)
                                     @ coefficients), -0.1, 0.1)
    for channel in range(3):
        field = f"f_dc_{channel}"
        rgb = result[field][-added_count:] * 0.2820947918 + 0.5
        rgb = np.clip(rgb + correction[:, channel], 0, 1)
        result[field][-added_count:] = (rgb - 0.5) / 0.2820947918
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", required=True)
    parser.add_argument("--camera-json", required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--reference-render", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    parent = Path(args.preview_dir).resolve()
    destination = Path(args.output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("--output-dir must be absent or empty")
    with open(parent / "preview.json", encoding="utf-8") as handle:
        details = json.load(handle)
    source_ply = parent / "candidate.ply"
    if details.get("approved") or sha256(source_ply) != details["candidate_sha256"]:
        raise ValueError("Source must be an unchanged, unapproved preview")
    with open(args.camera_json, encoding="utf-8") as handle:
        camera = next((entry for entry in json.load(handle)
                       if entry["img_name"] == args.image_name), None)
    if camera is None:
        raise ValueError("Reference camera was not found")
    reference = np.asarray(Image.open(args.reference_render).convert("RGB"))
    ply, vertices = read_vertices(source_ply)
    added = int(details["added_donor_gaussians"])
    fine = int(details.get("fine_fill_gaussians", added))
    if not 0 < fine <= added < len(vertices):
        raise ValueError("Invalid fill count in source preview")
    height, width = reference.shape[:2]
    mask = fill_mask(vertices, fine, added, camera, width, height)
    coefficients, diagnostics = boundary_correction(reference, mask)
    candidate = adjust_added_colors(vertices, added, camera, width, height,
                                    coefficients, args.strength)
    destination.mkdir(parents=True, exist_ok=True)
    output_ply = destination / "candidate.ply"
    write_vertices(output_ply, candidate, ply,
                   ["unapproved boundary-color-matched preview"])
    sidecar = source_ply.with_suffix(".objects.json")
    if sidecar.is_file():
        shutil.copy2(sidecar, output_ply.with_suffix(".objects.json"))
    Image.fromarray((mask * 255).astype(np.uint8)).save(destination / "fill-mask.png")
    shutil.copy2(args.reference_render, destination / "reference-before.png")
    shutil.copy2(args.camera_json, destination / "diagnostic-cameras.json")
    tool = Path(__file__).with_name("render_ply_preview.py")
    def render(cameras, name, output, width):
        subprocess.run([sys.executable, str(tool), "--ply", str(output_ply),
                        "--cameras", str(cameras), "--image-name", name,
                        "--output", str(output), "--width", str(width)], check=True)
    render(destination / "diagnostic-cameras.json", args.image_name,
           destination / "diagnostic-after.png", width)
    for name in details["views"]:
        render(details["cameras"], name, destination / f"{name}.png", 540)
    with Image.open(args.reference_render) as before, Image.open(destination / "diagnostic-after.png") as after:
        comparison = Image.new("RGB", (width * 2, height))
        comparison.paste(before.convert("RGB"), (0, 0))
        comparison.paste(after.convert("RGB"), (width, 0))
        comparison.save(destination / "before-after.jpg", quality=90)
    details.pop("view_metrics", None)
    details.update({
        "parent_preview": str(parent),
        "candidate_sha256": sha256(output_ply),
        "approved": False,
        "boundary_color_match": {
            "reference_camera": args.image_name,
            "reference_render_sha256": sha256(args.reference_render),
            "strength": args.strength,
            "coefficients": coefficients.tolist(),
            **diagnostics,
        },
    })
    with open(destination / "preview.json", "w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2)
        handle.write("\n")
    print(f"Unapproved color-matched preview: {output_ply}")


if __name__ == "__main__":
    main()
