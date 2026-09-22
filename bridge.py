"""Convert a TripoSR mesh into semantic 3D Gaussian points."""

import argparse
import numpy as np
from PIL import Image
import torch
import trimesh

from scene.gaussian_model import GaussianModel
from utils.ply_semantic_utils import add_float_property, read_vertices, write_vertices
from utils.semantic_utils import blend_embeddings, encode_image, encode_text, load_clip, load_pca

SH_C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="TripoSR OBJ/GLB mesh")
    parser.add_argument("--output", required=True, help="Output Gaussian PLY")
    parser.add_argument("--label", required=True, help="Text label, e.g. 'red bottle'")
    parser.add_argument("--reference-image", help="Optional generated/source image")
    parser.add_argument("--pca-path", default="data/my_scene/pca_model_128.pkl")
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--splat-log-scale", type=float, default=-4.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def convert(args):
    pca = load_pca(args.pca_path)
    semantic_dim = int(pca.n_components_)
    model, processor = load_clip(args.device)
    text_embedding = encode_text(args.label, pca, args.device, model, processor)
    image_embedding = None
    if args.reference_image:
        with Image.open(args.reference_image) as image:
            image_embedding = encode_image(image.convert("RGB"), pca, args.device, model, processor)
    semantic = blend_embeddings(text_embedding, image_embedding, args.text_weight)

    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    xyz = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=args.device)
    num_points = xyz.shape[0]
    vertex_colors = getattr(mesh.visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) == num_points:
        colors = torch.as_tensor(np.asarray(vertex_colors)[:, :3], dtype=torch.float32, device=args.device) / 255.0
    else:
        colors = torch.full_like(xyz, 0.5)

    gaussians = GaussianModel(args.sh_degree, semantic_feature_dim=semantic_dim)
    gaussians._xyz = xyz
    gaussians._features_dc = ((colors - 0.5) / SH_C0).unsqueeze(1)
    rest_coefficients = (args.sh_degree + 1) ** 2 - 1
    gaussians._features_rest = torch.zeros((num_points, rest_coefficients, 3), device=args.device)
    gaussians._scaling = torch.full((num_points, 3), args.splat_log_scale, device=args.device)
    gaussians._rotation = torch.zeros((num_points, 4), device=args.device)
    gaussians._rotation[:, 0] = 1.0
    gaussians._opacity = torch.full((num_points, 1), 5.0, device=args.device)
    gaussians._semantic_feature = semantic.reshape(1, 1, semantic_dim).repeat(num_points, 1, 1)
    gaussians.save_ply(args.output)

    ply, vertices = read_vertices(args.output)
    vertices = add_float_property(vertices, "object_id", float(args.object_id))
    write_vertices(args.output, vertices, ply, [f"object {args.object_id}: {args.label}"])
    print(f"Saved {num_points:,} semantic Gaussians ({semantic_dim}D, object_id={args.object_id}) to {args.output}")


if __name__ == "__main__":
    convert(parse_args())
