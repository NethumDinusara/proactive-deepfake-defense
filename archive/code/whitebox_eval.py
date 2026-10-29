import os
import argparse
import sys
import types
from unittest.mock import MagicMock

# --- Windows Environment Compatibility Fix ---
aim_module = types.ModuleType("aim")
aim_module.__spec__ = MagicMock()
aim_module.__path__ = []
sys.modules["aim"] = aim_module
aim_sdk_module = types.ModuleType("aim.sdk")
aim_sdk_module.__spec__ = MagicMock()
sys.modules["aim.sdk"] = aim_sdk_module
# ---------------------------------------------

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SurrogateEnsemble
from utils import load_config, get_device
from evaluate import FlatFolderDataset

def calculate_whitebox_asr(run_folder, num_test_images=500):
    config = load_config()
    device = get_device()
    print(f"--- Executing White-Box ASR Evaluation on {device} ---")

    # 1. Load the White-Box Optimization Manifold
    train_dir = config['paths']['train_data_output']
    if not os.path.exists(train_dir):
        print(f"[ERROR] Training directory not found: {train_dir}")
        return

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = FlatFolderDataset(train_dir, transform=transform)
    
    subset_size = min(num_test_images, len(train_dataset))
    subset = torch.utils.data.Subset(train_dataset, range(subset_size))
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    # 2. Load the SDSM-UAP Tensor
    target_epoch = config['training']['epochs']
    results_dir = os.path.join(config['paths']['results_dir'], run_folder)
    uap_path = os.path.join(results_dir, f"perturbation_epoch_{target_epoch}.pt")
    
    if not os.path.exists(uap_path):
        print(f"[ERROR] UAP tensor not found at: {uap_path}")
        return
        
    print(f"[*] Loading UAP Shield: {uap_path}")
    v = torch.load(uap_path, map_location=device, weights_only=True).requires_grad_(False)

    # 3. Initialize Tri-Architecture Surrogate Ensemble
    ensemble = SurrogateEnsemble(device)
    ensemble.facenet.to(device)
    if ensemble.unet: ensemble.unet.to(device)
    if ensemble.stargan: ensemble.stargan.to(device)

    # Evaluation Thresholds from Thesis Protocol
    THRESH_COS = 0.10
    THRESH_MSE = 0.0020

    success_facenet = 0
    success_unet = 0
    success_stargan = 0

    print("\n[*] Measuring Generative Collapse on Training Manifold...")
    with torch.no_grad():
        for x_clean, _ in tqdm(loader, desc="Evaluating White-Box"):
            x_clean = x_clean.to(device)
            x_pert = torch.clamp(x_clean + v, 0, 1)

            # --- Model 1: FaceNet (Biometric Identity) ---
            x_c_160 = F.interpolate((x_clean - 0.5)/0.5, size=(160, 160), mode='bilinear')
            x_p_160 = F.interpolate((x_pert - 0.5)/0.5, size=(160, 160), mode='bilinear')
            f_c = ensemble.facenet(x_c_160)
            f_p = ensemble.facenet(x_p_160)
            cos_dist = 1 - F.cosine_similarity(f_c, f_p).item()
            if cos_dist > THRESH_COS: 
                success_facenet += 1

            # --- Model 2: DDPM U-Net (Latent Diffusion) ---
            if ensemble.unet:
                t = torch.tensor([800], device=device).long()
                unet_c = ensemble.unet(x_clean, t).sample
                unet_p = ensemble.unet(x_pert, t).sample
                if F.mse_loss(unet_c, unet_p).item() > THRESH_MSE: 
                    success_unet += 1

            # --- Model 3: StarGAN (Spatial Synthesis) ---
            if ensemble.stargan:
                # 1. Normalize inputs to [-1, 1] for StarGAN
                star_x_c = (x_clean - 0.5) / 0.5
                star_x_p = (x_pert - 0.5) / 0.5
                
                # 2. Provide the target attribute change
                c_trg = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=device)
                
                # 3. Generate the Deepfake edits
                out_c = ensemble.stargan(star_x_c, c_trg)
                out_p = ensemble.stargan(star_x_p, c_trg)
                
                star_c = out_c[0] if isinstance(out_c, tuple) else out_c
                star_p = out_p[0] if isinstance(out_p, tuple) else out_p
                
                # 4. Measure the catastrophic failure
                if F.mse_loss(star_c, star_p).item() > THRESH_MSE: 
                    success_stargan += 1

    # Format Output for Thesis Documentation
    print("\n" + "="*60)
    print("          WHITE-BOX ATTACK SUCCESS RATES (ASR)          ")
    print("="*60)
    print(f"Evaluation Manifold: {subset_size} images (CelebA-HQ / FFHQ)")
    print(f"FaceNet (Biometric Evasion):   {(success_facenet/subset_size)*100:.1f}%")
    if ensemble.stargan:
        print(f"StarGAN (Spatial Collapse):    {(success_stargan/subset_size)*100:.1f}%")
    if ensemble.unet:
        print(f"DDPM U-Net (Latent Collapse):  {(success_unet/subset_size)*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate White-Box Attack Success Rates")
    parser.add_argument('--run_folder', type=str, required=True, help="Target run folder located in /results/")
    args = parser.parse_args()
    
    calculate_whitebox_asr(args.run_folder)