import torch
from scene.gaussian_model import GaussianModel

# --- CONFIGURATION ---
room_path = "output/semantic_run_64D/point_cloud/iteration_20000/cleaned_cloud.ply"
# Make sure this points to the output from your bridge script!
new_object_path = "cat_gaussians.ply" 
output_path = "output/semantic_run_64D/point_cloud/iteration_20000/room_with_cat.ply"

device = "cuda" if torch.cuda.is_available() else "cpu"

def inject_new_object():
    print("Loading your room...")
    room = GaussianModel(sh_degree=1)
    room.load_ply(room_path)
    
    print("Loading the new AI-generated object...")
    new_obj = GaussianModel(sh_degree=1) # Match the SH degree of your room!
    new_obj.load_ply(new_object_path)

    # --- POSITION AND SCALE THE NEW OBJECT ---
    scale_factor = 0.5 
    new_obj._xyz *= scale_factor
    new_obj._scaling -= abs(torch.log(torch.tensor(scale_factor))) 
    
    # Move it to where you want it in the room (X, Y, Z)
    new_obj._xyz[:, 0] += 1.5 
    new_obj._xyz[:, 1] += 0.0 
    new_obj._xyz[:, 2] += -2.0 

    # --- THE MERGE ---
    print("Gluing the objects together...")
    with torch.no_grad():
        room._xyz = torch.cat([room._xyz, new_obj._xyz], dim=0)
        room._features_dc = torch.cat([room._features_dc, new_obj._features_dc], dim=0)
        room._features_rest = torch.cat([room._features_rest, new_obj._features_rest], dim=0)
        room._opacity = torch.cat([room._opacity, new_obj._opacity], dim=0)
        room._scaling = torch.cat([room._scaling, new_obj._scaling], dim=0)
        room._rotation = torch.cat([room._rotation, new_obj._rotation], dim=0)
        
        # Merge the 64D semantics we already created in bridge.py!
        if hasattr(room, '_semantic_feature') and hasattr(new_obj, '_semantic_feature'):
            room._semantic_feature = torch.cat([room._semantic_feature, new_obj._semantic_feature], dim=0)

    room.save_ply(output_path)
    print(f"Done! Open {output_path} to see your newly generated object in the room.")

if __name__ == "__main__":
    inject_new_object()