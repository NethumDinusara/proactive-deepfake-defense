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
import csv
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import lpips 

from models import SurrogateEnsemble
from utils import load_config, seed_everything, get_device, ensure_dir

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
        return image

def project_linf(v, epsilon):
    """Constrains perturbation magnitude to maintain visual stealth."""
    v.data = torch.clamp(v.data, -epsilon, epsilon)

def train():
    config = load_config()
    seed_everything(config['system']['seed'])
    device = get_device()
    
    print(f"--- Tri-Surrogate SDA-T3 UAP Optimization on {device} ---")
    
    epsilon = config['training']['epsilon']
    epochs = config['training']['epochs']
    lambda_lpips = config['training']['lambda_lpips']
    lambda_geom = config['training']['lambda_geom']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_ep{epochs}_eps{epsilon}_geom{lambda_geom}_lpips{lambda_lpips}_{timestamp}"
    run_dir = os.path.join(config['paths']['results_dir'], run_folder_name)
    
    ensure_dir(run_dir)
    print(f"[*] Saving all training data and logs to:\n    -> {run_dir}")
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FlatFolderDataset(config['paths']['train_data_output'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    
    ensemble = SurrogateEnsemble(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    v = torch.empty(1, 3, 256, 256).uniform_(-epsilon, epsilon).to(device)
    v.requires_grad = True
    
    optimizer = optim.Adam([v], lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    csv_file = os.path.join(run_dir, "training_log.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Geom_Loss_FaceNet', 'Geom_Loss_Diff', 'Geom_Loss_GAN', 'LPIPS_Penalty', 'Learning_Rate'])

    for epoch in range(1, epochs + 1):
        total_facenet = 0.0
        total_diff = 0.0
        total_gan = 0.0
        total_lpips = 0.0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for x_clean in progress:
            x_clean = x_clean.to(device)
            
            # 1. CLEAR PREVIOUS GRADIENTS
            optimizer.zero_grad()
            
            # --- 2. LPIPS Stealth Penalty ---
            x_pert = torch.clamp(x_clean + v, 0, 1)
            loss_lpips = lambda_lpips * loss_fn_vgg(x_clean * 2 - 1, x_pert * 2 - 1).mean()
            loss_lpips.backward() # BACKWARD PASS 1
            
            # --- 3. VRAM-Efficient Gradient Accumulation ---
            
            # Anchor A: FaceNet
            ensemble.facenet.to(device)
            loss_facenet = ensemble.forward_loss(x_clean, v, mode="facenet")
            loss_facenet.backward() # BACKWARD PASS 2 (While FaceNet is still on GPU)
            ensemble.facenet.to("cpu")
            torch.cuda.empty_cache()
            
            # Anchor B: Diffusion 
            loss_diff = torch.tensor(0.0, device=device)
            if hasattr(ensemble, 'unet') and ensemble.unet is not None:
                ensemble.unet.to(device)
                loss_diff = ensemble.forward_loss(x_clean, v, mode="diffusion")
                loss_diff.backward() # BACKWARD PASS 3 (While UNet is still on GPU)
                ensemble.unet.to("cpu")
                torch.cuda.empty_cache()
                
            # Anchor C: StarGAN
            loss_gan = torch.tensor(0.0, device=device)
            if hasattr(ensemble, 'stargan') and ensemble.stargan is not None:
                ensemble.stargan.to(device)
                loss_gan = ensemble.forward_loss(x_clean, v, mode="gan")
                loss_gan.backward() # BACKWARD PASS 4 (While StarGAN is still on GPU)
                ensemble.stargan.to("cpu")
                torch.cuda.empty_cache()
            
            # --- 4. STEP OPTIMIZER ---
            # All gradients are now safely accumulated inside `v.grad`!
            optimizer.step()
            project_linf(v, epsilon)
            
            # Logging tracking variables
            total_facenet += loss_facenet.item()
            total_diff += loss_diff.item()
            total_gan += loss_gan.item()
            total_lpips += loss_lpips.item()
            
            progress.set_postfix(
                fn=f"{loss_facenet.item():.2f}", 
                df=f"{loss_diff.item():.2f}", 
                gan=f"{loss_gan.item():.2f}", 
                lpips=f"{loss_lpips.item():.2f}"
            )

        scheduler.step()
        num_batches = len(dataloader)
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                total_facenet / num_batches, 
                total_diff / num_batches, 
                total_gan / num_batches, 
                total_lpips / num_batches, 
                scheduler.get_last_lr()[0]
            ])
        
        torch.save(v.detach().cpu(), os.path.join(run_dir, f"perturbation_epoch_{epoch}.pt"))
        v_vis = (v.detach().cpu() + epsilon) / (2 * epsilon)
        transforms.ToPILImage()(v_vis.squeeze(0)).save(os.path.join(run_dir, f"vis_epoch_{epoch}.png"))

if __name__ == "__main__":
    train()