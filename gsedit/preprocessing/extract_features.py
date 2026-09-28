import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import SAM
from tqdm import tqdm
from sklearn.decomposition import PCA
import gc
import pickle
import argparse

def extract_features(input_dir, output_dir, temp_dir, target_dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Load Models
    print("Loading MobileSAM...")
    sam = SAM('mobile_sam.pt') 
    
    print("Loading CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    images = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    if not images:
        print("No images found!")
        return

    print(f"Phase 1: Extracting 512D features and saving temporarily...")
    all_unique_features = [] # Stores a lightweight list of object vectors for PCA training

    # ==========================================
    # PHASE 1: Process Images & Collect Features
    # ==========================================
    for img_name in tqdm(images):
        img_path = os.path.join(input_dir, img_name)
        
        cv2_img = cv2.imread(img_path)
        cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_img_rgb)
        
        h, w = cv2_img.shape[:2]
        
        scale = 4
        feat_h, feat_w = h // scale, w // scale
        
        # Initialize a SMALLER empty feature map
        feature_map = torch.zeros((feat_h, feat_w, 512), dtype=torch.float16, device="cpu")
        
        results = sam(cv2_img)
        
        if results and results[0].masks is not None and len(results[0].masks) > 0:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy() 

            object_crops = []
            object_masks = []
            for i, mask in enumerate(masks):
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                x1, y1, x2, y2 = map(int, boxes[i])
                
                # SAFETY CHECK: if too small dont waste your time and VRAM on tiny objects that won't contribute much to the final result
                area = (x2 - x1) * (y2 - y1)
                if area < 900: 
                    continue
                    
                object_crops.append(pil_img.crop((x1, y1, x2, y2)))
                small_mask = cv2.resize(mask.astype(np.uint8), (feat_w, feat_h), interpolation=cv2.INTER_NEAREST)
                object_masks.append(torch.from_numpy(small_mask.astype(bool)))

            if object_crops:
                inputs = clip_processor(
                    images=object_crops, return_tensors="pt", padding=True
                ).to(device)
                with torch.inference_mode():
                    clip_features = clip_model.get_image_features(**inputs)
                    clip_features /= clip_features.norm(dim=-1, keepdim=True)
                    clip_features = clip_features.half().cpu()
                for mask_tensor, clip_feature in zip(object_masks, clip_features):
                    feature_map[mask_tensor] = clip_feature
                    all_unique_features.append(clip_feature.numpy())
                
        # Save the heavy 512D map to disk temporarily so we don't run out of RAM
        temp_save_path = os.path.join(temp_dir, f"{img_name.split('.')[0]}_features.pt")
        torch.save(feature_map.cpu(), temp_save_path)
        
        # Clean up VRAM
        del feature_map
        torch.cuda.empty_cache()

    # ==========================================
    # PHASE 2: Train Global PCA (512 -> 128)
    # ==========================================
    print("\nPhase 2: Training Global PCA...")
    
    # Delete heavy models from VRAM to make room
    del sam
    del clip_model
    torch.cuda.empty_cache()
    gc.collect()

    if len(all_unique_features) == 0:
        print("No features extracted! Check your images and SAM model.")
        return
        
    # Stack all unique object vectors into a single matrix (N_objects, 512)
    all_unique_features = np.vstack(all_unique_features).astype(np.float32)
    
    if len(all_unique_features) < target_dim:
        raise ValueError(
            f"PCA needs at least {target_dim} object samples, got {len(all_unique_features)}"
        )
    pca = PCA(n_components=target_dim)
    pca.fit(all_unique_features)
    pca_path = os.path.join(
        os.path.dirname(output_dir.rstrip(os.sep)), f"pca_model_{target_dim}.pkl"
    )
    with open(pca_path, "wb") as pca_file:
        pickle.dump(pca, pca_file)

    # ==========================================
    # PHASE 3: Apply PCA to spatial maps & Save
    # ==========================================
    print(f"\nPhase 3: Compressing feature maps to {target_dim}D...")
    
    for img_name in tqdm(images):
        base_name = img_name.split('.')[0]
        temp_path = os.path.join(temp_dir, f"{base_name}_features.pt")
        final_path = os.path.join(output_dir, f"{base_name}_fmap_CxHxW.pt")
        
        if not os.path.exists(temp_path):
            continue
            
        # Load the 512D map from disk
        feat_map_512 = torch.load(temp_path).numpy() # Shape: (H, W, 512)
        h, w = feat_map_512.shape[:2]
        
        # Flatten map to apply PCA: (H*W, 512)
        flat_map = feat_map_512.reshape(-1, 512).astype(np.float32)
        
        # Transform flat map to 128D
        compressed_flat = pca.transform(flat_map)
        
        # Reshape back to spatial map and convert back to Float16 tensor
        feat_map_128 = torch.from_numpy(
            compressed_flat.reshape((h, w, target_dim))
        ).permute(2, 0, 1).contiguous().half()
        
        # Save the final 128D PyTorch tensor!
        torch.save(feat_map_128, final_path)
        
        # Delete the temporary 512D file to free disk space
        os.remove(temp_path)
        del feat_map_512, flat_map, compressed_flat, feat_map_128
        gc.collect()

    # Remove the empty temp directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    print(f"\nExtraction complete! {target_dim}D Feature maps saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dense SAM+CLIP feature maps")
    parser.add_argument("--input-dir", default="data/my_scene/images")
    parser.add_argument("--output-dir", default="data/my_scene/sam_embeddings_128")
    parser.add_argument("--temp-dir", default="output/temp_512_128")
    parser.add_argument("--target-dim", type=int, default=128)
    args = parser.parse_args()
    extract_features(args.input_dir, args.output_dir, args.temp_dir, args.target_dim)
