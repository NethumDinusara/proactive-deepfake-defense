import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models import SurrogateEnsemble
from utils import load_config, get_device, ensure_dir

# --- HF Diffusers for DDPM ---
from diffusers import DDPMPipeline

def norm_for_models(x):
    """Maps [0, 1] to [-1, 1] for model inputs"""
    return (x - 0.5) / 0.5

def denorm_for_visuals(x):
    """Maps [-1, 1] back to [0, 1] and strictly clamps"""
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)

def get_facenet_attention(model, img_tensor):
    """Generates a visual heatmap of what FaceNet is looking at using Gradient Saliency"""
    img_tensor = img_tensor.clone().detach().requires_grad_(True)
    img_norm = norm_for_models(F.interpolate(img_tensor, size=(160, 160), mode='bilinear'))
    
    embedding = model(img_norm)
    score = embedding.norm() # Magnitude of the identity
    
    model.zero_grad()
    score.backward()
    
    # Extract the gradients and create a 2D heatmap
    saliency, _ = torch.max(img_tensor.grad.data.abs(), dim=1, keepdim=True)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency.squeeze().cpu().numpy()

def run_fixed_generators(run_folder, image_name=None):
    config = load_config()
    device = get_device()
    
    print(f"\n--- Initiating Fixed Generative White-Box Test on {device} ---")
    
    results_dir = os.path.join(config['paths']['results_dir'], run_folder)
    test_output_dir = os.path.join(results_dir, "whitebox_tests")
    ensure_dir(test_output_dir)
    
    target_epoch = config['training']['epochs']
    perturbation_path = os.path.join(results_dir, f"perturbation_epoch_{target_epoch}.pt")
    
    v = torch.load(perturbation_path, map_location=device, weights_only=True).requires_grad_(False)
    
    print("[*] Loading Pretrained Weights for StarGAN, FaceNet, and DDPM...")
    ensemble = SurrogateEnsemble(device)
    ensemble.stargan.to(device)
    ensemble.facenet.to(device)
    
    pipeline = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to(device)
    scheduler = pipeline.scheduler
    
    ffhq_path = config['paths']['ffhq_source']
    if image_name is None:
        image_name = [f for f in os.listdir(ffhq_path) if f.lower().endswith(('.png', '.jpg'))][2]
    
    img_path = os.path.join(ffhq_path, image_name)
    
    # FIX 1: Add CenterCrop to zoom in on the face and eliminate StarGAN's green background artifacts
    preprocess = transforms.Compose([
        transforms.CenterCrop(190), 
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])
    x_clean = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    x_protected = torch.clamp(x_clean + v, 0.0, 1.0)
    
    x_clean_norm = norm_for_models(x_clean)
    x_prot_norm = norm_for_models(x_protected)

    print("[*] 1. Executing StarGAN Generation (Green Artifacts Fixed)...")
    with torch.no_grad():
        c_trg = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0]]).to(device) # Black Hair & Gender Swap
        out_c = ensemble.stargan(x_clean_norm, c_trg)
        out_p = ensemble.stargan(x_prot_norm, c_trg)
        
        stargan_clean_gen = denorm_for_visuals(out_c[0] if isinstance(out_c, tuple) else out_c)
        stargan_prot_gen = denorm_for_visuals(out_p[0] if isinstance(out_p, tuple) else out_p)

    print("[*] 2. Generating FaceNet Visual Attention Maps...")
    # Generate Saliency Heatmaps
    facenet_clean_map = get_facenet_attention(ensemble.facenet, x_clean)
    facenet_prot_map = get_facenet_attention(ensemble.facenet, x_protected)

    print("[*] 3. Executing Diffusion Generation (Attack Window Optimized)...")
    # FIX 2: Lower timesteps to 150 so the UAP isn't completely washed away by denoising
    t_start = 150 
    scheduler.set_timesteps(1000)
    timesteps = scheduler.timesteps[-t_start:] 
    start_timestep = timesteps[0].unsqueeze(0)
    
    torch.manual_seed(42)
    noise = torch.randn_like(x_clean_norm)
    noisy_clean = scheduler.add_noise(x_clean_norm, noise, start_timestep)
    noisy_prot = scheduler.add_noise(x_prot_norm, noise, start_timestep)
    
    def denoise(noisy_img, desc):
        img = noisy_img.clone()
        for t in tqdm(timesteps, desc=desc):
            with torch.no_grad():
                res = pipeline.unet(img, t).sample
            img = scheduler.step(res, t, img).prev_sample
        return img

    ddpm_clean_gen = denorm_for_visuals(denoise(noisy_clean, "Clean DDPM"))
    ddpm_prot_gen = denorm_for_visuals(denoise(noisy_prot, "Protected DDPM"))

    # =========================================================
    # GENERATE THE VISUAL OUTPUT GRID
    # =========================================================
    def to_pil(tensor):
        return transforms.ToPILImage()(tensor.squeeze(0).cpu())

    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    fig.patch.set_facecolor('#ffffff')
    plt.suptitle("Deepfake Output Failure Across Tri-Architecture Models", fontsize=22, fontweight='bold', y=0.96)

    col_titles = ["Original Input", "Protected (+UAP)", "Deepfake Target", "Distorted Output"]
    for ax, col in zip(axes[0], col_titles):
        ax.set_title(col, fontweight='bold', fontsize=14)

    # --- ROW 1: STARGAN ---
    axes[0, 0].imshow(to_pil(x_clean))
    axes[0, 0].set_ylabel("StarGAN\n(Attribute Edit)", fontweight='bold', fontsize=14)
    axes[0, 1].imshow(to_pil(x_protected))
    axes[0, 2].imshow(to_pil(stargan_clean_gen)) 
    axes[0, 3].imshow(to_pil(stargan_prot_gen))  
    
    # --- ROW 2: DIFFUSION ---
    axes[1, 0].imshow(to_pil(x_clean))
    axes[1, 0].set_ylabel("DDPM\n(Synthesis)", fontweight='bold', fontsize=14)
    axes[1, 1].imshow(to_pil(x_protected))
    axes[1, 2].imshow(to_pil(ddpm_clean_gen))
    axes[1, 3].imshow(to_pil(ddpm_prot_gen))

    # --- ROW 3: FACENET ---
    axes[2, 0].imshow(to_pil(x_clean))
    axes[2, 0].set_ylabel("FaceNet\n(Identity Extractor)", fontweight='bold', fontsize=14)
    axes[2, 1].imshow(to_pil(x_protected))
    
    # Show the Saliency Heatmaps (FIX 3)
    axes[2, 2].imshow(to_pil(x_clean).convert("L"), cmap='gray', alpha=0.5)
    axes[2, 2].imshow(facenet_clean_map, cmap='inferno', alpha=0.7)
    axes[2, 2].text(0.5, 0.05, "Focused on Facial Features", ha='center', va='bottom', color='white', fontweight='bold', transform=axes[2,2].transAxes)

    axes[2, 3].imshow(to_pil(x_protected).convert("L"), cmap='gray', alpha=0.5)
    axes[2, 3].imshow(facenet_prot_map, cmap='inferno', alpha=0.7)
    axes[2, 3].text(0.5, 0.05, "Attention Shattered", ha='center', va='bottom', color='white', fontweight='bold', transform=axes[2,3].transAxes)

    # Clean up axes
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    save_path = os.path.join(test_output_dir, f"fixed_generators_{image_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[+] Fixed visual grid saved to:\n    {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder', type=str, required=True)
    parser.add_argument('--image_name', type=str, default=None)
    args = parser.parse_args()
    run_fixed_generators(args.run_folder, args.image_name)