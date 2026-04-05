import os
import sys
from PIL import Image
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
import numpy as np

# --- Custom Utilities ---
try:
    from utils import load_config, seed_everything, get_device, ensure_dir
except ImportError:
    print("[ERROR] Could not import 'utils.py'. Ensure it exists in the project root.")
    sys.exit(1)

def process_clean_dataset(source_path, dest_path, prefix, target_size, max_images):
    """
    Processes high-fidelity, pre-aligned image datasets (e.g., CelebA-HQ, FFHQ).
    
    This function standardizes the spatial dimensions using Lanczos resampling 
    to preserve high-frequency textural details, and converts the output to 
    lossless PNG format to prevent compression artifacts during UAP optimization.
    """
    if not os.path.exists(source_path):
        print(f"[WARNING] Source directory not found: {source_path}")
        return

    print(f"[*] Processing {prefix.upper()} dataset from {source_path}...")
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # Lexicographical sort ensures deterministic subset selection across separate runs
    files = [f for f in os.listdir(source_path) if f.lower().endswith(valid_exts)]
    files.sort()
    
    processed_count = 0
    
    for fname in tqdm(files, desc=f"Standardizing {prefix.upper()}"):
        if processed_count >= max_images:
            break
            
        try:
            src_file = os.path.join(source_path, fname)
            img = Image.open(src_file).convert('RGB')
            
            # Spatial Standardization via High-Fidelity Resampling
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Lossless serialization
            save_name = f"{prefix}_{processed_count:05d}.png"
            save_path = os.path.join(dest_path, save_name)
            
            img.save(save_path, optimize=True)
            processed_count += 1
                
        except Exception as e:
            print(f"[Error] Failed to process {fname}: {e}")

def process_wild_dataset(source_path, dest_path, target_size, max_images, device):
    """
    Processes 'in-the-wild' unaligned datasets (e.g., Labeled Faces in the Wild - LFW).
    
    Utilizes a Multi-task Cascaded Convolutional Network (MTCNN) to detect and 
    spatially align facial landmarks. This ensures the zero-shot transferability 
    testing is conducted on geometrically standardized inputs.
    """
    if not os.path.exists(source_path):
        print(f"[WARNING] Source directory not found: {source_path}")
        return

    # Initialize the biometric alignment network
    # A margin of 20 pixels preserves peripheral topological features (hairline, jawline)
    mtcnn = MTCNN(
        image_size=target_size[0], 
        margin=20, 
        post_process=False,  # Bypass internal standard-deviation normalization
        device=device
    )
    
    file_list = []
    for root, _, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                file_list.append(os.path.join(root, file))
    
    file_list.sort()
    processed_count = 0

    for src_file in tqdm(file_list, desc="Aligning LFW (In-the-Wild)"):
        if processed_count >= max_images:
            break
            
        try:
            img = Image.open(src_file).convert('RGB')
            
            # MTCNN returns a cropped tensor of shape (3, H, W)
            img_cropped = mtcnn(img)
            
            if img_cropped is not None:
                # Transpose dimensions: (3, H, W) -> (H, W, 3)
                img_np = img_cropped.permute(1, 2, 0).cpu().numpy()
                
                # Radiometric Correction:
                # MTCNN (post_process=False) yields unnormalized floats in [0, 255].
                # We strictly clip to valid 8-bit boundaries to avoid contrast distortion.
                img_clipped = np.clip(img_np, 0, 255).astype('uint8')
                
                img_out = Image.fromarray(img_clipped)
                save_name = f"lfw_{processed_count:05d}.png"
                img_out.save(os.path.join(dest_path, save_name))
                
                processed_count += 1
        except Exception:
            # Silently skip images where MTCNN fails to detect a valid face
            continue

def main():
    """
    Data Pipeline Orchestrator.
    
    Establishes the experimental data splits by preprocessing the high-fidelity 
    training manifold (CelebA-HQ + FFHQ) and the heterogeneous testing manifold (LFW).
    """
    config = load_config()
    seed_everything(config['system']['seed'])
    device = get_device()
    
    print(f"--- Experimental Data Preprocessing Pipeline ---")
    print(f"[*] Hardware Target: {device}")
    
    # Establish Output Topography
    train_dir = config['paths']['train_data_output']
    test_dir = config['paths']['test_data_output']
    ensure_dir(train_dir)
    ensure_dir(test_dir)
    
    # Extract structural constraints from hyperparameter configuration
    target_size = (config['data']['image_size'], config['data']['image_size'])
    total_train_images = config['data']['train_subset_size']
    total_test_images = config['data']['test_subset_size']
    
    # Distribute the training manifold equally across generating distributions
    half_train = total_train_images // 2
    
    # Execute Phase 1: Training Manifold Standardization
    process_clean_dataset(
        config['paths']['celeba_hq_source'], 
        train_dir, 
        prefix="celeba", 
        target_size=target_size, 
        max_images=half_train
    )
    
    process_clean_dataset(
        config['paths']['ffhq_source'], 
        train_dir, 
        prefix="ffhq", 
        target_size=target_size, 
        max_images=half_train
    )
    
    # Execute Phase 2: Testing Manifold Alignment
    process_wild_dataset(
        config['paths']['lfw_source'], 
        test_dir, 
        target_size=target_size, 
        max_images=total_test_images,
        device=device
    )
    
    # Pipeline Verification
    num_train = len([f for f in os.listdir(train_dir) if f.endswith('.png')])
    num_test = len([f for f in os.listdir(test_dir) if f.endswith('.png')])
    
    print(f"\n[SUCCESS] Pipeline Execution Complete.")
    print(f" -> Protected Training Manifold (CelebA+FFHQ): {num_train} images mapped to {train_dir}")
    print(f" -> Vulnerable Testing Manifold (Aligned LFW): {num_test} images mapped to {test_dir}")

if __name__ == "__main__":
    main()