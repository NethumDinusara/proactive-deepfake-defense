import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusers import DDPMPipeline
from utils import load_config, get_device, ensure_dir

def run_generative_failure_test(run_folder, image_name=None):
    config = load_config()
    device = get_device()
    
    print(f"\n--- Starting Generative Synthesis Evaluation (DDPM) on {device} ---")
    
    results_dir = os.path.join(config['paths']['results_dir'], run_folder)
    test_output_dir = os.path.join(results_dir, "whitebox_tests")
    ensure_dir(test_output_dir)
    
    # 1. Load the UAP
    target_epoch = config['training']['epochs']
    perturbation_path = os.path.join(results_dir, f"perturbation_epoch_{target_epoch}.pt")
    print(f"[*] Loading UAP from Epoch {target_epoch}...")
    v = torch.load(perturbation_path, map_location=device, weights_only=True).requires_grad_(False)
    
    # 2. Load the Generative Diffusion Pipeline
    print("[*] Loading DDPM Generative Pipeline...")
    pipeline = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to(device)
    scheduler = pipeline.scheduler
    
    # 3. Load and Preprocess Image
    ffhq_path = config['paths']['ffhq_source']
    if image_name is None:
        image_name = [f for f in os.listdir(ffhq_path) if f.lower().endswith(('.png', '.jpg'))][83]
    
    img_path = os.path.join(ffhq_path, image_name)
    print(f"[*] Loading test image: {image_name}")
    
    preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    x_clean = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    x_protected = torch.clamp(x_clean + v, 0.0, 1.0)
    
    # 4. Map from [0, 1] to [-1, 1] for Diffusion Math
    img_clean_norm = (x_clean * 2) - 1
    img_prot_norm = (x_protected * 2) - 1
    
    # 5. Image-to-Image Generation Setup (SDEdit style)
    # We add 40% noise to the image, then ask the AI to generate a face from it
    t_start = 400 # 400 out of 1000 timesteps
    scheduler.set_timesteps(1000)
    
    # Get the exact timesteps to loop over (from 400 down to 0)
    timesteps = scheduler.timesteps[-t_start:] 
    start_timestep = timesteps[0].unsqueeze(0)
    
    # Create identical starting noise for a fair comparison
    torch.manual_seed(42)
    noise = torch.randn_like(img_clean_norm)
    
    noisy_clean = scheduler.add_noise(img_clean_norm, noise, start_timestep)
    noisy_prot = scheduler.add_noise(img_prot_norm, noise, start_timestep)
    
    # 6. The Generative Loop
    def denoise_image(noisy_img, desc="Generating"):
        img = noisy_img.clone()
        for t in tqdm(timesteps, desc=desc):
            with torch.no_grad():
                residual = pipeline.unet(img, t).sample
            img = scheduler.step(residual, t, img).prev_sample
        return img

    print("\n[*] Synthesizing Baseline Deepfake...")
    gen_clean = denoise_image(noisy_clean, "Baseline Gen")
    
    print("\n[*] Synthesizing Disrupted Deepfake...")
    gen_prot = denoise_image(noisy_prot, "Disrupted Gen")
    
    # Map back to visual range [0, 1]
    gen_clean = (gen_clean / 2 + 0.5).clamp(0, 1)
    gen_prot = (gen_prot / 2 + 0.5).clamp(0, 1)
    
    # 7. Visualization
    def to_pil(tensor):
        return transforms.ToPILImage()(tensor.squeeze(0).cpu())

    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(to_pil(x_clean))
    plt.title("1. Clean Original", fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(to_pil(x_protected))
    plt.title("2. Protected Image (+UAP)", fontweight='bold', color='green')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(to_pil(gen_clean))
    plt.title("3. Baseline Generative Output\n(Normal Synthesis)", fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(to_pil(gen_prot))
    plt.title("4. CATASTROPHIC FAILURE\n(UAP Derails Synthesis)", fontweight='bold', color='red')
    plt.axis('off')

    save_path = os.path.join(test_output_dir, f"generative_failure_{image_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[+] Visual representation saved successfully to:\n    {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder', type=str, required=True)
    parser.add_argument('--image_name', type=str, default=None)
    args = parser.parse_args()
    run_generative_failure_test(args.run_folder, args.image_name)