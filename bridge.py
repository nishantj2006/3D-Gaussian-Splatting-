import torch
import trimesh
import numpy as np
from scene.gaussian_model import GaussianModel
import os

# --- CONFIGURATION ---
input_mesh_path = "external_tools/TripoSR/output/0/mesh.obj"
output_gaussian_path = os.path.abspath("cat_gaussians.ply")

def convert_mesh_to_gaussians():
    print(f"Loading TripoSR Mesh from {input_mesh_path}...")
    mesh = trimesh.load(input_mesh_path, process=False)
    
    # 1. Extract XYZ coordinates
    xyz = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
    num_pts = xyz.shape[0]
    
    # 2. Extract Colors
    # TripoSR saves textures directly to the mesh vertices as RGB colors
    if hasattr(mesh.visual, 'vertex_colors'):
        colors = torch.tensor(mesh.visual.vertex_colors[:, :3], dtype=torch.float32, device="cuda") / 255.0
    else:
        colors = torch.ones_like(xyz) * 0.5
        
    # 3DGS uses Spherical Harmonics (SH) for color. 
    # The math to convert standard RGB to the base SH DC component: dc = (color - 0.5) / 0.28209
    SH_C0 = 0.28209479177387814
    f_dc = (colors - 0.5) / SH_C0
    f_dc = f_dc.unsqueeze(1) # Shape must be (N, 1, 3)

    # 3. Build the Gaussian Splats
    print("Shattering mesh into 3D Gaussians...")
    new_obj = GaussianModel(sh_degree=1) # Matching your room's sh_degree
    new_obj._xyz = xyz
    new_obj._features_dc = f_dc
    
    # Pad out the rest of the required Gaussian math
    new_obj._features_rest = torch.zeros((num_pts, 3, 3), device="cuda") # Empty view-dependent colors
    new_obj._scaling = torch.ones((num_pts, 3), device="cuda") * -4.5    # Make splats small enough to retain mesh shape
    new_obj._rotation = torch.zeros((num_pts, 4), device="cuda")
    new_obj._rotation[:, 0] = 1.0                                        # Default identity rotation
    new_obj._opacity = torch.ones((num_pts, 1), device="cuda") * 5.0     # Maximum opacity

    #needs some features in order to add
    new_obj._semantic_feature = torch.zeros((num_pts, 1, 64), device="cuda")
    # 4. Save
    new_obj.save_ply(output_gaussian_path)
    print(f"Success! Native 3D Gaussians saved to {output_gaussian_path}")

if __name__ == "__main__":
    convert_mesh_to_gaussians()