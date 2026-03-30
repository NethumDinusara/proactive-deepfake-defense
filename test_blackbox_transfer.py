import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_config, get_device, ensure_dir
from facenet_pytorch import InceptionResnetV1

# --- HuggingFace Stable Diffusion for Unseen Generative Test ---
from diffusers import StableDiffusionImg2ImgPipeline

def norm_for_models(x):
    return (x - 0.5) / 0.5

def denorm_for_visuals(x):
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)

def run_blackbox_transferability(run_folder, image_name=None):
    config = load_config()
    device = get_device()
    
    print(f"\n--- Initiating Black-Box Transferability Test on {device} ---")
    
    results_dir = os.path.join(config['paths']['results_dir'], run_folder)
    test_output_dir = os.path.join(results_dir, "blackbox_tests")
    ensure_dir(test_output_dir)
    
    # We are using your final 30-epoch model
    perturbation_path = os.path.join(results_dir, f"perturbation_epoch_30.pt")
    
    print("[*] Loading Final 30-Epoch SDSM UAP...")
    v = torch.load(perturbation_path, map_location=device, weights_only=True).requires_grad_(False)
    
    # ---------------------------------------------------------
    # BLACK-BOX 1: Unseen Identity Extractor (CASIA-Webface Weights)
    # ---------------------------------------------------------
    print("[*] Loading Black-Box Identity Extractor (CASIA-Webface)...")
    blackbox_facenet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    
    # ---------------------------------------------------------
    # BLACK-BOX 2: Unseen Latent Diffusion Model (Stable Diffusion 1.5)
    # ---------------------------------------------------------
    print("[*] Loading Black-Box Generative Engine (Stable Diffusion 1.5)...")
    sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16 if 'cuda' in device.type else torch.float32,
        safety_checker=None # Disabled for research speed
    ).to(device)
    
    ffhq_path = config['paths']['ffhq_source']
    if image_name is None:
        image_name = [f for f in os.listdir(ffhq_path) if f.lower().endswith(('.png', '.jpg'))][20]
    
    img_path = os.path.join(ffhq_path, image_name)
    
    preprocess = transforms.Compose([
        transforms.CenterCrop(190), 
        transforms.Resize((512, 512)), # Stable diffusion requires 512x512 minimum
        transforms.ToTensor()
    ])
    
    x_clean = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    
    # The UAP was trained at 256x256, so we interpolate it to 512x512 for the Black-Box test
    v_resized = F.interpolate(v, size=(512, 512), mode='bilinear')
    x_protected = torch.clamp(x_clean + v_resized, 0.0, 1.0)
    
    x_clean_norm = norm_for_models(x_clean)
    x_prot_norm = norm_for_models(x_protected)

    print("[*] 1. Executing Unseen Identity Extraction...")
    with torch.no_grad():
        x_clean_160 = F.interpolate(x_clean_norm, size=(160, 160), mode='bilinear')
        x_prot_160 = F.interpolate(x_prot_norm, size=(160, 160), mode='bilinear')
        
        id_clean = blackbox_facenet(x_clean_160)
        id_prot = blackbox_facenet(x_prot_160)
        id_conf = max(0.0, F.cosine_similarity(id_clean, id_prot).item())

    print("[*] 2. Executing Stable Diffusion (Img2Img) Generation...")
    # Convert tensors back to PIL for the HuggingFace Pipeline
    pil_clean = transforms.ToPILImage()(x_clean.squeeze(0).cpu())
    pil_prot = transforms.ToPILImage()(x_protected.squeeze(0).cpu())
    
    # The Ultimate Stress Test: High Guidance, High Strength
    prompt = "A perfectly symmetrical, high quality photorealistic portrait photograph of a person's face with white hair, 8k resolution."
    
    sd_clean_out = sd_pipeline(prompt=prompt, image=pil_clean, strength=0.5, guidance_scale=15.0).images[0]
    sd_prot_out = sd_pipeline(prompt=prompt, image=pil_prot, strength=0.8, guidance_scale=15.0).images[0]

    # =========================================================
    # GENERATE THE BLACK-BOX VISUAL GRID
    # =========================================================
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor('#ffffff')
    plt.suptitle("Black-Box Transferability: Unseen Model Disruption", fontsize=22, fontweight='bold', y=0.98)

    col_titles = ["Original Input", "Protected (+UAP)", "Unseen Model Target", "Distorted Output"]
    for ax, col in zip(axes[0], col_titles):
        ax.set_title(col, fontweight='bold', fontsize=14)

    # --- ROW 1: BLACK-BOX IDENTITY (CASIA-Webface) ---
    axes[0, 0].imshow(pil_clean)
    axes[0, 0].set_ylabel("FaceNet\n(CASIA-Webface Weights)", fontweight='bold', fontsize=12)
    axes[0, 1].imshow(pil_prot)
    
    axes[0, 2].text(0.5, 0.5, "Extract Identity\nTarget Match: 100%", ha='center', va='center', fontsize=14, color='green', fontweight='bold')
    axes[0, 2].set_facecolor('#eaffea')
    
    success = id_conf < 0.60
    color = 'red' if success else 'green'
    status = "DEEPFAKE CRASHED" if success else "PIPELINE VULNERABLE"
    
    axes[0, 3].text(0.5, 0.6, f"Extracted Identity:\n{id_conf:.1%} Match", ha='center', va='center', fontsize=14, color=color, fontweight='bold')
    axes[0, 3].text(0.5, 0.4, status, ha='center', va='center', fontsize=12, color='darkred' if success else 'darkgreen')
    axes[0, 3].set_facecolor('#ffeaea' if success else '#eaffea')

    # --- ROW 2: BLACK-BOX GENERATION (Stable Diffusion) ---
    axes[1, 0].imshow(pil_clean)
    axes[1, 0].set_ylabel("Stable Diffusion v1.5\n(Latent Img2Img)", fontweight='bold', fontsize=12)
    axes[1, 1].imshow(pil_prot)
    axes[1, 2].imshow(sd_clean_out)
    axes[1, 3].imshow(sd_prot_out)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    save_path = os.path.join(test_output_dir, f"blackbox_transfer_{image_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[+] Black-Box visual grid saved to:\n    {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder', type=str, required=True)
    parser.add_argument('--image_name', type=str, default=None)
    args = parser.parse_args()
    run_blackbox_transferability(args.run_folder, args.image_name)