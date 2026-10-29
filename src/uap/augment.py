import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class RobustnessAugmentation(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- 1. Random Resize (Spatial Compression) ---
        # Simulates a platform downscaling the image, then the model upscaling it
        scale_factor = torch.empty(1).uniform_(0.5, 0.9).item() 
        scaled_H, scaled_W = int(H * scale_factor), int(W * scale_factor)
        
        x_aug = F.interpolate(x, size=(scaled_H, scaled_W), mode='bilinear', align_corners=False)
        x_aug = F.interpolate(x_aug, size=(H, W), mode='bilinear', align_corners=False)
        
        # --- 2. Color / Brightness Jitter (Simulating Platform Filters) ---
        # Forces the UAP to be robust against slight color grading
        if torch.rand(1).item() < 0.5:
            brightness_factor = torch.empty(1).uniform_(0.8, 1.2).item()
            x_aug = TF.adjust_brightness(x_aug, brightness_factor)
        
        # --- 3. Gaussian Blur (Simulating Optics/Downscaling) ---
        if torch.rand(1).item() < 0.5:
            sigma = torch.empty(1).uniform_(0.1, 1.0).item()
            x_aug = TF.gaussian_blur(x_aug, kernel_size=[3, 3], sigma=[sigma, sigma])
            
        # --- 4. Differentiable Quantization (Simulating JPEG Compression) ---
        # Simulates the information loss of 8-bit/JPEG quantization without breaking gradients
        quantization_noise = torch.empty_like(x_aug).uniform_(-1/255.0, 1/255.0)
        x_aug = x_aug + quantization_noise
        
        # --- 5. Random Crop/Shift (Simulating Misalignment) ---
        # Forces the noise to be spatially invariant
        padding = 4
        x_padded = F.pad(x_aug, (padding, padding, padding, padding), mode='reflect')
        h_start = torch.randint(0, 2*padding, (1,)).item()
        w_start = torch.randint(0, 2*padding, (1,)).item()
        x_shifted = x_padded[:, :, h_start:h_start+H, w_start:w_start+W]
        
        return torch.clamp(x_shifted, 0, 1)