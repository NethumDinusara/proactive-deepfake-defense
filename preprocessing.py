import os
import sys
from PIL import Image
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
import numpy as np

# Import our custom utilities
# Ensure utils.py is in the same directory
try:
    from utils import load_config, seed_everything, get_device, ensure_dir
except ImportError:
    print("Error: Could not import 'utils.py'. Please ensure utils.py is created in the project root.")
    sys.exit(1)

def process_clean_dataset(source_path, dest_path, prefix, target_size, max_images):
    """
    Processes high-quality, pre-aligned datasets (CelebA-HQ, FFHQ).
    Action: Resizes to target_size and converts to PNG.
    """
    if not os.path.exists(source_path):
        print(f"[WARNING] Source path not found: {source_path}")
        return

    print(f"Processing {prefix} from {source_path}...")
    
    # Supported extensions
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # Get all files and sort them to ensure the same 2500 images are picked every time
    files = [f for f in os.listdir(source_path) if f.lower().endswith(valid_exts)]
    files.sort()
    
    count = 0
    processed_count = 0
    
    for fname in tqdm(files, desc=f"Processing {prefix}"):
        if processed_count >= max_images:
            break
            
        try:
            src_file = os.path.join(source_path, fname)
            img = Image.open(src_file).convert('RGB')
            
            # High-quality resize (Lanczos)
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save as PNG (lossless)
            save_name = f"{prefix}_{processed_count:05d}.png"
            save_path = os.path.join(dest_path, save_name)
            
            img.save(save_path, optimize=True)
            processed_count += 1
                
        except Exception as e:
            print(f"Error processing {fname}: {e}")

def process_wild_dataset(source_path, dest_path, target_size, max_images, device):
    if not os.path.exists(source_path):
        print(f"[WARNING] Source path not found: {source_path}")
        return

    # margin=20 provides a bit of "breathing room" around the face, 
    # which helps StyleGAN2 learn the hairline/ears better.
    mtcnn = MTCNN(
        image_size=target_size[0], 
        margin=20, 
        post_process=False, # We handle normalization manually below
        device=device
    )
    
    file_list = []
    for root, _, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                file_list.append(os.path.join(root, file))
    
    file_list.sort()
    processed_count = 0

    for src_file in tqdm(file_list, desc="Aligning LFW"):
        if processed_count >= max_images:
            break
            
        try:
            img = Image.open(src_file).convert('RGB')
            # MTCNN returns (3, H, W) tensor
            img_cropped = mtcnn(img)
            
            if img_cropped is not None:
                # Convert (3, H, W) -> (H, W, 3)
                img_np = img_cropped.permute(1, 2, 0).cpu().numpy()
                
                # FIXED NORMALIZATION:
                # Since post_process=False, we must manually scale from raw to 0-255
                img_min, img_max = img_np.min(), img_np.max()
                img_normalized = (img_np - img_min) / (img_max - img_min) # Scale to 0-1
                img_final = (img_normalized * 255).astype('uint8') # Scale to 0-255
                
                img_out = Image.fromarray(img_final)
                save_name = f"lfw_{processed_count:05d}.png"
                img_out.save(os.path.join(dest_path, save_name))
                
                processed_count += 1
        except Exception:
            continue

def main():
    # 1. Load Configuration & Setup
    config = load_config()
    seed_everything(config['system']['seed'])
    device = get_device()
    
    print(f"--- Data Preprocessing Started ---")
    print(f"Device: {device}")
    
    # 2. Create Output Directories
    train_dir = config['paths']['train_data_output']
    test_dir = config['paths']['test_data_output']
    ensure_dir(train_dir)
    ensure_dir(test_dir)
    
    # 3. Get Settings from Config
    target_size = (config['data']['image_size'], config['data']['image_size'])
    total_train_images = config['data']['train_subset_size']
    total_test_images = config['data']['test_subset_size']
    
    # Split training quota evenly between CelebA-HQ and FFHQ
    half_train = total_train_images // 2
    
    # 4. Process Training Data
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
    
    # 5. Process Testing Data (LFW)
    process_wild_dataset(
        config['paths']['lfw_source'], 
        test_dir, 
        target_size=target_size, 
        max_images=total_test_images,
        device=device
    )
    
    # 6. Final Summary
    num_train = len([f for f in os.listdir(train_dir) if f.endswith('.png')])
    num_test = len([f for f in os.listdir(test_dir) if f.endswith('.png')])
    
    print(f"\n[DONE] Preprocessing Complete.")
    print(f"Training Images (CelebA+FFHQ): {num_train} saved to {train_dir}")
    print(f"Testing Images (Aligned LFW): {num_test} saved to {test_dir}")

if __name__ == "__main__":
    main()