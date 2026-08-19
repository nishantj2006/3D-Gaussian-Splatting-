import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import gc
import pickle # Added for saving the model

def compress_features():
    temp_dir = "output/perm512/"
    output_dir = "output/feature_maps/"
    
    # Keeping this at 16 for your 6GB GPU
    target_dim = 64 
    
    os.makedirs(output_dir, exist_ok=True)

    print("Phase 1: Gathering data to train PCA...")
    pt_files = [f for f in os.listdir(temp_dir) if f.endswith('.pt')]
    
    if not pt_files:
        print(f"No .pt files found in {temp_dir}!")
        return

    sampled_features = []
    
    for f in tqdm(pt_files, desc="Sampling"):
        pt_path = os.path.join(temp_dir, f)
        feat_map = torch.load(pt_path).numpy() 
        flat_map = feat_map.reshape(-1, 512)

        norms = np.linalg.norm(flat_map, axis=1)
        valid_features = flat_map[norms > 0.01]

        if len(valid_features) > 2000:
            indices = np.random.choice(len(valid_features), 2000, replace=False)
            valid_features = valid_features[indices]

        sampled_features.append(valid_features)

    # Train PCA
    all_features = np.vstack(sampled_features).astype(np.float32)
    print(f"\nPhase 2: Training PCA on {len(all_features)} data points...")
    pca = PCA(n_components=target_dim)
    pca.fit(all_features)

    # --- NEW: SAVE THE PCA MODEL ---
    # This allows us to use your text input ("water bottle") later!
    print("Saving PCA model to output/pca_model.pkl...")
    with open("output/pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
    # -------------------------------

    del all_features, sampled_features
    gc.collect()

    print(f"\nPhase 3: Compressing maps to {target_dim}D...")
    for f in tqdm(pt_files, desc="Compressing"):
        pt_path = os.path.join(temp_dir, f)
        
        clean_name = f.replace('_features.pt', '.pt')
        final_path = os.path.join(output_dir, clean_name)
        
        feat_map_512 = torch.load(pt_path).numpy()
        h, w = feat_map_512.shape[:2]
        
        flat_map = feat_map_512.reshape(-1, 512).astype(np.float32)
        compressed_flat = pca.transform(flat_map)
        
        feat_map_compressed = torch.from_numpy(compressed_flat.reshape((h, w, target_dim))).half()
        torch.save(feat_map_compressed, final_path)
        
        del feat_map_512, flat_map, compressed_flat, feat_map_compressed
        gc.collect()

    print("\nCompression complete! 16D Feature maps and pca_model.pkl saved.")

if __name__ == "__main__":
    compress_features()