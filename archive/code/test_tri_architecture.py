import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models import SurrogateEnsemble
from utils import load_config, get_device, ensure_dir

def get_heatmap(tensor):
    """Averages across channels to create a 2D visual heatmap of the model's 'brain'."""
    hm = tensor.squeeze(0).mean(dim=0).cpu().detach().numpy()
    hm = (hm - np.min(hm)) / (np.max(hm) - np.min(hm) + 1e-8)
    return hm

def run_tri_architecture_test(run_folder, image_name=None):
    config = load_config()
    device = get_device()
    
    print(f"\n--- Starting Tri-Architecture Latent Evaluation on {device} ---")
    
    results_dir = os.path.join(config['paths']['results_dir'], run_folder)
    test_output_dir = os.path.join(results_dir, "whitebox_tests")
    ensure_dir(test_output_dir)
    
    target_epoch = config['training']['epochs']
    perturbation_path = os.path.join(results_dir, f"perturbation_epoch_{target_epoch}.pt")
    v = torch.load(perturbation_path, map_location=device, weights_only=True).requires_grad_(False)
    
    print("[*] Loading FaceNet, DDPM, and StarGAN...")
    ensemble = SurrogateEnsemble(device)
    if ensemble.unet: ensemble.unet.to(device)
    if ensemble.stargan: ensemble.stargan.to(device)
    ensemble.facenet.to(device)
    
    ffhq_path = config['paths']['ffhq_source']
    if image_name is None:
        image_name = [f for f in os.listdir(ffhq_path) if f.lower().endswith(('.png', '.jpg'))][83]
    
    img_path = os.path.join(ffhq_path, image_name)
    preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    x_clean = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    x_protected = torch.clamp(x_clean + v, 0.0, 1.0)
    
    print("[*] Extracting Latent Feature Maps from all 3 models...")
    with torch.no_grad():
        # --- 1. FACENET (Identity Disruption) ---
        x_160_c = F.interpolate(x_clean, size=(160, 160), mode='bilinear')
        x_160_p = F.interpolate(x_protected, size=(160, 160), mode='bilinear')
        id_c = ensemble.facenet((x_160_c - 0.5) / 0.5)
        id_p = ensemble.facenet((x_160_p - 0.5) / 0.5)
        id_cosine_sim = F.cosine_similarity(id_c, id_p).item()
        
        # --- 2. DIFFUSION U-NET (Synthesis Disruption) ---
        t = torch.tensor([500]).long().to(device)
        noise_pred_c = ensemble.unet(x_clean, t).sample
        noise_pred_p = ensemble.unet(x_protected, t).sample
        ddpm_mse = F.mse_loss(noise_pred_c, noise_pred_p).item()
        
        hm_ddpm_c = get_heatmap(noise_pred_c)
        hm_ddpm_p = get_heatmap(noise_pred_p)
        
        # --- 3. STARGAN (GAN Feature Disruption) ---
        c_trg = torch.zeros(1, 5).to(device)
        x_stargan_c = (x_clean - 0.5) / 0.5
        x_stargan_p = (x_protected - 0.5) / 0.5
        
        out_c = ensemble.stargan(x_stargan_c, c_trg)
        out_p = ensemble.stargan(x_stargan_p, c_trg)
        
        # Grab a middle feature map (index 2) from the list of internal layers
        feat_c = out_c[1][2] if isinstance(out_c, tuple) else out_c
        feat_p = out_p[1][2] if isinstance(out_p, tuple) else out_p
        gan_mse = F.mse_loss(feat_c, feat_p).item()
        
        hm_gan_c = get_heatmap(feat_c)
        hm_gan_p = get_heatmap(feat_p)

    # --- VISUALIZATION DASHBOARD ---
    plt.figure(figsize=(15, 12))
    plt.suptitle("Tri-Architecture Latent Space Disruption", fontsize=18, fontweight='bold')

    def to_pil(tensor):
        return transforms.ToPILImage()(tensor.squeeze(0).cpu().clamp(0, 1))

    # Row 1: The Images
    plt.subplot(3, 3, 1)
    plt.imshow(to_pil(x_clean))
    plt.title("Clean Original", fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(to_pil(x_protected))
    plt.title("Protected (+UAP)", fontweight='bold')
    plt.axis('off')
    
    # FaceNet Identity Metric Panel
    plt.subplot(3, 3, 3)
    plt.text(0.5, 0.7, "FaceNet Identity Shift", ha='center', va='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.4, f"Cosine Similarity:\n{id_cosine_sim:.4f}", ha='center', va='center', fontsize=16, color='red' if id_cosine_sim < 0.5 else 'orange')
    plt.text(0.5, 0.1, "(Lower = Identity Destroyed)", ha='center', va='center', fontsize=10)
    plt.axis('off')

    # Row 2: Diffusion Disruption
    plt.subplot(3, 3, 4)
    plt.imshow(hm_ddpm_c, cmap='viridis')
    plt.title("DDPM Clean Features", fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(hm_ddpm_p, cmap='inferno') # Using a different colormap to highlight the chaos
    plt.title("DDPM Disrupted Features", fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.text(0.5, 0.5, f"DDPM MSE:\n{ddpm_mse:.4f}", ha='center', va='center', fontsize=16, color='red')
    plt.axis('off')

    # Row 3: StarGAN Disruption
    plt.subplot(3, 3, 7)
    plt.imshow(hm_gan_c, cmap='viridis')
    plt.title("StarGAN Clean Features", fontweight='bold')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(hm_gan_p, cmap='inferno')
    plt.title("StarGAN Disrupted Features", fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.text(0.5, 0.5, f"StarGAN MSE:\n{gan_mse:.4f}", ha='center', va='center', fontsize=16, color='red')
    plt.axis('off')

    save_path = os.path.join(test_output_dir, f"tri_arch_eval_{image_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\n[+] Tri-Architecture Dashboard saved to:\n    {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder', type=str, required=True)
    parser.add_argument('--image_name', type=str, default=None)
    args = parser.parse_args()
    run_tri_architecture_test(args.run_folder, args.image_name)