import sys
import types
from unittest.mock import MagicMock
import argparse

# --- ROBUST WINDOWS FIX ---
aim_module = types.ModuleType("aim")
aim_module.__spec__ = MagicMock()
aim_module.__path__ = []
sys.modules["aim"] = aim_module
aim_sdk_module = types.ModuleType("aim.sdk")
aim_sdk_module.__spec__ = MagicMock()
sys.modules["aim.sdk"] = aim_sdk_module
# ---------------------------

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import lpips
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models import SurrogateEnsemble
from utils import load_config, get_device, ensure_dir

class FlatFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg'))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

def plot_thesis_graphs(results_dir, lpips_list, rand_div, prot_div, rand_mse, prot_mse):
    """Generates publication-quality graphs tailored for the SDSM-Directional methodology."""
    print("\n[Generating Thesis Graphs]...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Convergence Curve (from CSV)
    csv_path = os.path.join(results_dir, "training_log.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(10, 5))
        plt.plot(df['Epoch'], df['Geom_Loss_GAN'], marker='o', label='GAN Disruption', color='#27ae60')
        plt.plot(df['Epoch'], df['Geom_Loss_Diff'], marker='s', label='Diffusion Disruption', color='#2980b9')
        plt.plot(df['Epoch'], df['Geom_Loss_FaceNet'], marker='^', label='FaceNet Disruption', color='#e74c3c')
        plt.title('Optimization Convergence (Tri-Surrogate)', fontweight='bold')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Geometric Disruption', fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'graph_1_convergence.png'), dpi=300)
        plt.close()

    # 2. LPIPS Distribution Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(lpips_list, bins=30, color='#2980b9', edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(lpips_list), color='#e74c3c', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(lpips_list):.4f}')
    plt.axvline(x=0.05, color='#27ae60', linestyle='dotted', linewidth=2.5, label='Invisibility Threshold (0.05)')
    plt.title('Distribution of Stealth Scores (LPIPS)', fontweight='bold')
    plt.xlabel('LPIPS Score (Lower = More Imperceptible)', fontweight='bold')
    plt.ylabel('Number of Images', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'graph_2_lpips_histogram.png'), dpi=300)
    plt.close()

    # 3. Geometric Disruption Bar Chart (Cosine Distance)
    plt.figure(figsize=(7, 6))
    bars = plt.bar(['Random Noise\n(Baseline)', 'SDSM UAP\n(Proposed)'], 
                   [np.mean(rand_div), np.mean(prot_div)], 
                   color=['#95a5a6', '#e74c3c'], edgecolor='black', width=0.5)
    plt.title('Geometric Disruption (Angular Divergence)', fontweight='bold')
    plt.ylabel('Cosine Distance', fontweight='bold')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.01), round(yval, 4), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'graph_3_geometric_disruption.png'), dpi=300)
    plt.close()

    # 4. Attack Success Rate (ASR) Graph
    thresh_cos = 0.10
    thresh_mse = 0.0020
    
    # Calculate Percentages
    asr_cos_rand = (np.array(rand_div) > thresh_cos).mean() * 100
    asr_cos_prot = (np.array(prot_div) > thresh_cos).mean() * 100
    asr_mse_rand = (np.array(rand_mse) > thresh_mse).mean() * 100
    asr_mse_prot = (np.array(prot_mse) > thresh_mse).mean() * 100

    categories = ['Biometric Evasion\n(Cosine > 0.10)', 'Spatial Disruption\n(MSE > 0.0020)']
    baseline_asr = [asr_cos_rand, asr_mse_rand]
    proposed_asr = [asr_cos_prot, asr_mse_prot]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, baseline_asr, width, label='Baseline (Random Noise)', color='#95a5a6', edgecolor='black')
    bars2 = ax.bar(x + width/2, proposed_asr, width, label='Proposed (SDSM UAP)', color='#2980b9', edgecolor='black')

    ax.set_ylabel('Attack Success Rate (%)', fontweight='bold')
    ax.set_title('Target Defeat: Attack Success Rate (ASR)', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper left', frameon=True, shadow=True)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'graph_4_asr_chart.png'), dpi=300)
    plt.close()
    
    print(f"-> All 4 Thesis Graphs saved to {results_dir}")

def evaluate(run_folder):
    config = load_config()
    device = get_device()
    print(f"--- Starting SDSM-Directional Evaluation on {device} ---")
    
    results_dir = os.path.join(config['paths']['results_dir'], run_folder)
    if not os.path.exists(results_dir):
        print(f"[ERROR] Could not find run folder: {results_dir}")
        return

    vis_dir = os.path.join(results_dir, "evaluation_images")
    ensure_dir(vis_dir)

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = FlatFolderDataset(config['paths']['test_data_output'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    target_epoch = config['training']['epochs']
    epsilon = config['training']['epsilon']
    
    perturbation_path = os.path.join(results_dir, f"perturbation_epoch_{target_epoch}.pt")
    if not os.path.exists(perturbation_path):
        print(f"[ERROR] Could not find {perturbation_path}.")
        return

    print(f"Loading UAP from: {perturbation_path}")
    v = torch.load(perturbation_path, map_location=device, weights_only=True)
    v.requires_grad = False

    ensemble = SurrogateEnsemble(device)
    if ensemble.unet: ensemble.unet.to(device)
    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    metrics = {'lpips': [], 'psnr': [], 'ssim': [], 'div_random': [], 'div_protected': [], 'mse_random': [], 'mse_protected': []}
    
    print("\n[Running Mathematical Evaluation on LFW Dataset]...")
    with torch.no_grad():
        for i, (x_clean, fname) in enumerate(tqdm(test_loader)):
            x_clean = x_clean.to(device)
            
            # Apply UAPs
            x_pert = torch.clamp(x_clean + v, 0, 1)
            v_random = torch.empty_like(v).uniform_(-epsilon, epsilon).to(device)
            x_random = torch.clamp(x_clean + v_random, 0, 1)

            # STEALTH METRICS
            lpips_val = loss_fn_alex((x_clean * 2 - 1).float(), (x_pert * 2 - 1).float()).item()
            metrics['lpips'].append(lpips_val)
            
            img_c = x_clean.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img_p = x_pert.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            metrics['psnr'].append(psnr(img_c, img_p, data_range=1.0))
            metrics['ssim'].append(ssim(img_c, img_p, data_range=1.0, channel_axis=2, win_size=3))

            # PROTECTION METRICS
            if ensemble.unet:
                t = torch.tensor([800], device=device).long()
                feat_clean = ensemble.unet(x_clean, t).sample
                feat_pert = ensemble.unet(x_pert, t).sample
                feat_rand = ensemble.unet(x_random, t).sample
                
                mse_prot = F.mse_loss(feat_clean, feat_pert).item()
                mse_rand = F.mse_loss(feat_clean, feat_rand).item()
                metrics['mse_protected'].append(mse_prot)
                metrics['mse_random'].append(mse_rand)
                
                cos_sim_prot = F.cosine_similarity(feat_clean, feat_pert, dim=1) 
                cos_sim_rand = F.cosine_similarity(feat_clean, feat_rand, dim=1)
                
                metrics['div_protected'].append(1 - cos_sim_prot.mean().item())
                metrics['div_random'].append(1 - cos_sim_rand.mean().item())

            if i < 3:
                noise_vis = (v[0] - v.min()) / (v.max() - v.min())
                combined = torch.cat((x_clean[0], noise_vis, x_pert[0]), dim=2)
                transforms.ToPILImage()(combined.cpu()).save(os.path.join(vis_dir, f"eval_sample_{i}.png"))

    # Generate Graphs
    plot_thesis_graphs(results_dir, metrics['lpips'], metrics['div_random'], metrics['div_protected'], metrics['mse_random'], metrics['mse_protected'])

    # --- CALCULATE ASR FOR REPORT ---
    thresh_cos = 0.10
    thresh_mse = 0.0020
    asr_cos_rand = (np.array(metrics['div_random']) > thresh_cos).mean() * 100
    asr_cos_prot = (np.array(metrics['div_protected']) > thresh_cos).mean() * 100
    asr_mse_rand = (np.array(metrics['mse_random']) > thresh_mse).mean() * 100
    asr_mse_prot = (np.array(metrics['mse_protected']) > thresh_mse).mean() * 100

    # Print Final Report
    print("\n" + "="*50)
    print("        FINAL ABLATION STUDY RESULTS        ")
    print("="*50)
    print(f"Tested on: {len(test_dataset)} images (LFW Dataset)")
    print(f"UAP Source: Epoch {target_epoch} (Epsilon={epsilon})")
    print("-" * 50)
    print("1. STEALTH (Invisibility to Humans)")
    print(f"   * Average LPIPS: {np.mean(metrics['lpips']):.4f}  (Target: < 0.05)")
    print(f"   * Average PSNR:  {np.mean(metrics['psnr']):.2f} dB (Target: > 35 dB)")
    print(f"   * Average SSIM:  {np.mean(metrics['ssim']):.4f}  (Target: > 0.95)")
    print("-" * 50)
    print("2. GEOMETRIC PROTECTION (Average Magnitudes)")
    print(f"   * Baseline Cosine Distance: {np.mean(metrics['div_random']):.4f}")
    print(f"   * SDSM UAP Cosine Distance: {np.mean(metrics['div_protected']):.4f}")
    print(f"   * Baseline Spatial MSE:     {np.mean(metrics['mse_random']):.4f}")
    print(f"   * SDSM UAP Spatial MSE:     {np.mean(metrics['mse_protected']):.4f}")
    print("-" * 50)
    print("3. ATTACK SUCCESS RATE (ASR)")
    print(f"   * Biometric Evasion (Cosine > {thresh_cos}):")
    print(f"       - Baseline: {asr_cos_rand:.1f}% | SDSM UAP: {asr_cos_prot:.1f}%")
    print(f"   * Spatial Disruption (MSE > {thresh_mse}):")
    print(f"       - Baseline: {asr_mse_rand:.1f}% | SDSM UAP: {asr_mse_prot:.1f}%")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UAP Defense")
    parser.add_argument('--run_folder', type=str, required=True, help="Name of the run folder inside /results/")
    args = parser.parse_args()
    
    evaluate(args.run_folder)