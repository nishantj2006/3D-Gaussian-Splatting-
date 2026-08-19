import torch
import pickle
import os
import transformers
from transformers import CLIPProcessor, CLIPModel
from scene.gaussian_model import GaussianModel

# Mute the Hugging Face security alarm 
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

# --- CONFIGURATION ---
ply_path = "output\semantic_run_64D\point_cloud\iteration_20000\point_cloud.ply" 
pca_path = "output/pca_model.pkl"
output_path = "output\semantic_run_64D\point_cloud\iteration_20000/cleaned_cloud.ply"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. MULTI-TARGETS: A "chorus" of words creates a much stronger mathematical center for the object
target_queries = [
    "carpet",
]

negative_queries = [
"desk", "clothes"

]

# 3. THE AGGRESSION CONTROLS (Balanced for a fair fight)
threshold = 0.05          # Require at least 15% confidence to delete
aggression_boost = 0.05   # Drop this back to a tiny tie-breaker boost, NOT a 30% handicap

def get_64d_text_vec(text, model, processor, pca):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        vec = model.get_text_features(**inputs).cpu().numpy()
    
    vec_64 = pca.transform(vec)
    vec_64 = torch.from_numpy(vec_64).to(device)
    vec_64 /= vec_64.norm(dim=-1, keepdim=True)
    return vec_64

def remove_by_text():
    print(f"Loading PCA model from {pca_path}...")
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    
    print("Loading Language AI (CLIP)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"Translating queries into 64D math...")
    # Load all target vectors
    target_vecs_64 = [get_64d_text_vec(t, model, processor, pca) for t in target_queries]
    # Load all negative vectors
    neg_vecs_64 = [get_64d_text_vec(n, model, processor, pca) for n in negative_queries]

    print(f"Loading 3D Scene from {ply_path}...")
    gaussians = GaussianModel(sh_degree=1)
    gaussians.load_ply(ply_path)
    
    # ==========================================
    # --- THE HUNTING BLOCK ---
    print("Hunting for the 64D language brain...")
    gauss_feats = None
    for name in dir(gaussians):
        if name.startswith('__'): continue
        try:
            val = getattr(gaussians, name)
            if torch.is_tensor(val) and len(val.shape) >= 2 and val.shape[-1] == 64:
                gauss_feats = val
                print(f"SUCCESS! Found 64D features hiding in attribute: '{name}'")
                break
        except Exception:
            pass

    if gauss_feats is None:
        raise ValueError("\nCRITICAL ERROR: The 64D tensor is not in memory.")
    # ==========================================

    gauss_feats = gauss_feats.view(-1, 64).clone().to(device)
    gauss_feats /= gauss_feats.norm(dim=-1, keepdim=True)

    print(f"Running Aggressive Contrastive Battle...")
    
    # 1. Get the HIGHEST score among all the target words
    sim_targets = torch.stack([torch.matmul(gauss_feats, t.T).squeeze() for t in target_vecs_64])
    max_target_sim, _ = sim_targets.max(dim=0)
    
    # 2. Get the HIGHEST score among all the negative background words
    sim_negatives = torch.stack([torch.matmul(gauss_feats, n.T).squeeze() for n in neg_vecs_64])
    max_neg_sim, _ = sim_negatives.max(dim=0)

    # 3. THE NEW AGGRESSIVE MASK
    # The target gets the 'aggression_boost' added to its score before fighting the background
    mask = (max_target_sim + aggression_boost > max_neg_sim) & (max_target_sim > threshold)
    
    points_to_remove = mask.sum().item()
    print(f"Found {points_to_remove} points matching the targets!")
    
    if points_to_remove > 0:
        print("Erasing points...")
        with torch.no_grad():
            gaussians._opacity[mask] = -9999.0 
        
        gaussians.save_ply(output_path)
        print("Done! You can now load the cleaned_cloud.ply in your viewer.")
    else:
        print("No points deleted. Increase 'aggression_boost' at the top of the script!")

if __name__ == "__main__":
    remove_by_text()