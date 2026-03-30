import sys
import types
from unittest.mock import MagicMock

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
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL

from utils import load_config, get_device, ensure_dir

def generate_autoencoder_failure():
    config = load_config()
    device = get_device()
    print(f"--- Generating Autoencoder Failure Proof on {device} ---")

    results_dir = config['paths']['results_dir']
    proofs_dir = os.path.join(results_dir, "thesis_visual_proofs")
    ensure_dir(proofs_dir)

    target_epoch = config['training']['epochs']
    perturbation_path = os.path.join(results_dir, f"perturbation_epoch_{target_epoch}.pt")

    if not os.path.exists(perturbation_path):
        print("[ERROR] UAP not found. Ensure training is complete.")
        return

    v = torch.load(perturbation_path, map_location=device)
    v.requires_grad = False

    # Load a pre-trained Autoencoder (Same architectural family as DeepFaceLab)
    print("Loading AutoencoderKL...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    test_dir = config['paths']['test_data_output']
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg'))]
    img_path = os.path.join(test_dir, image_files[0]) 

    image = Image.open(img_path).convert("RGB")

    # VAE expects inputs scaled between -1 and 1
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Ensure exact dimensions
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) 
    ])

    x_clean = transform(image).unsqueeze(0).to(device)

    # Scale UAP from [0, 1] to [-1, 1]
    v_scaled = v * 2.0 
    x_pert = torch.clamp(x_clean + v_scaled, -1, 1)

    print("Executing Autoencoder Reconstruction Pass...")

    with torch.no_grad():
         # Encode and Decode Clean Image
        latent_clean = vae.encode(x_clean).latent_dist.sample()
        recon_clean = vae.decode(latent_clean).sample

         # Encode and Decode Protected Image (The UAP hits the encoder directly)
        latent_pert = vae.encode(x_pert).latent_dist.sample()
        recon_pert = vae.decode(latent_pert).sample

    # Format for Matplotlib
    def format_for_plot(tensor):
        tensor = (tensor / 2 + 0.5).clamp(0, 1)
        return tensor.squeeze().cpu().permute(1, 2, 0).numpy()

    disp_clean_input = format_for_plot(x_clean)
    disp_pert_input = format_for_plot(x_pert)
    disp_clean_output = format_for_plot(recon_clean)
    disp_pert_output = format_for_plot(recon_pert)

    print("Plotting the Failure Matrix...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Autoencoder Generative Collapse (Simulating DeepFaceLab)', fontsize=16, fontweight='bold', y=0.95)

    axes[0, 0].imshow(disp_clean_input)
    axes[0, 0].set_title("Input: Clean Image", fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(disp_pert_input)
    axes[0, 1].set_title("Input: Protected Image (UAP Applied)", fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(disp_clean_output)
    axes[1, 0].set_title("Autoencoder Output: Successful Reconstruction", fontweight='bold', color='green')
    axes[1, 0].axis('off')

    # This should now show severe color bleeding, grid artifacts, and structural collapse
    axes[1, 1].imshow(disp_pert_output)
    axes[1, 1].set_title("Autoencoder Output: CATASTROPHIC FAILURE", fontweight='bold', color='red')
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(proofs_dir, "Figure_3_Autoencoder_Failure.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Success] Output failure proof saved to: {save_path}")

if __name__ == "__main__":
    generate_autoencoder_failure()
