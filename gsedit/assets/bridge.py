"""Convert a TripoSR mesh into a semantic 3D Gaussian surface asset."""

import argparse
import math

import numpy as np
from PIL import Image
import torch
import trimesh

from scene.gaussian_model import GaussianModel
from utils.ply_semantic_utils import add_float_property, read_vertices, write_vertices
from utils.semantic_utils import blend_embeddings, encode_image, encode_text, load_clip, load_pca

SH_C0 = 0.28209479177387814


def quaternions_from_z(normals):
    """Return wxyz quaternions rotating each Gaussian's local Z onto a normal."""
    normals = np.asarray(normals, dtype=np.float64)
    length = np.linalg.norm(normals, axis=1)
    if np.any(length < 1e-8):
        raise ValueError("Mesh has zero-length face normals")
    unit = normals / length[:, None]
    result = np.column_stack((1 + unit[:, 2], -unit[:, 1],
                              unit[:, 0], np.zeros(len(unit))))
    opposite = result[:, 0] < 1e-7
    result[opposite] = [0, 1, 0, 0]
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result.astype(np.float32)


def sample_mesh_gaussians(mesh, count, seed=0, splat_log_scale=None):
    """Sample colored mesh faces and make locally tangent, thin Gaussians."""
    if count < 100 or count > 2_000_000:
        raise ValueError("--points must be between 100 and 2,000,000")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0 or mesh.area <= 0:
        raise ValueError("Asset must be a nonempty surface mesh")
    points, face_index, colors = trimesh.sample.sample_surface(
        mesh, count, sample_color=True, seed=seed)
    normals = np.asarray(mesh.face_normals)[face_index]
    quaternions = quaternions_from_z(normals)
    if splat_log_scale is None:
        tangent_radius = math.sqrt(float(mesh.area) / count) * 0.8
    else:
        tangent_radius = math.exp(splat_log_scale)
    if not np.isfinite(tangent_radius) or tangent_radius <= 0:
        raise ValueError("Gaussian radius is invalid")
    scales = np.tile(np.log([tangent_radius, tangent_radius,
                             tangent_radius * 0.25]), (count, 1)).astype(np.float32)
    if colors is None or len(colors) != count:
        rgb = np.full((count, 3), 0.7, dtype=np.float32)
    else:
        rgb = np.asarray(colors)[:, :3].astype(np.float32) / 255
    return points.astype(np.float32), rgb, scales, quaternions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="TripoSR OBJ/GLB mesh")
    parser.add_argument("--output", required=True, help="Output Gaussian PLY")
    parser.add_argument("--label", required=True, help="Text label, e.g. 'red bottle'")
    parser.add_argument("--reference-image", help="Generated/source image for semantic blending")
    parser.add_argument("--pca-path", default="data/my_scene/pca_model_128.pkl")
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--points", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splat-log-scale", type=float,
                        help="Override area-based tangent log scale")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def convert(args):
    from pathlib import Path
    output = Path(args.output).resolve()
    if output.exists() or args.object_id <= 0:
        raise ValueError("Output must not exist and --object-id must be positive")
    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    points, colors, scales, quaternions = sample_mesh_gaussians(
        mesh, args.points, args.seed, args.splat_log_scale)

    pca = load_pca(args.pca_path)
    semantic_dim = int(pca.n_components_)
    model, processor = load_clip(args.device)
    text_embedding = encode_text(args.label, pca, args.device, model, processor)
    image_embedding = None
    if args.reference_image:
        with Image.open(args.reference_image) as image:
            image_embedding = encode_image(image.convert("RGB"), pca,
                                           args.device, model, processor)
    semantic = blend_embeddings(text_embedding, image_embedding, args.text_weight)

    count = len(points)
    gaussians = GaussianModel(args.sh_degree, semantic_feature_dim=semantic_dim)
    gaussians._xyz = torch.as_tensor(points, device=args.device)
    gaussians._features_dc = torch.as_tensor(
        (colors - 0.5) / SH_C0, device=args.device).unsqueeze(1)
    rest_coefficients = (args.sh_degree + 1) ** 2 - 1
    gaussians._features_rest = torch.zeros(
        (count, rest_coefficients, 3), device=args.device)
    gaussians._scaling = torch.as_tensor(scales, device=args.device)
    gaussians._rotation = torch.as_tensor(quaternions, device=args.device)
    opacity = math.log(0.85 / 0.15)
    gaussians._opacity = torch.full((count, 1), opacity, device=args.device)
    gaussians._semantic_feature = semantic.reshape(
        1, 1, semantic_dim).repeat(count, 1, 1)
    gaussians.save_ply(str(output))

    ply, vertices = read_vertices(output)
    vertices = add_float_property(vertices, "object_id", float(args.object_id))
    write_vertices(output, vertices, ply,
                   [f"object {args.object_id}: {args.label}"])
    print(f"Saved {count:,} semantic surface Gaussians "
          f"({semantic_dim}D, object_id={args.object_id}) to {output}")


if __name__ == "__main__":
    convert(parse_args())
