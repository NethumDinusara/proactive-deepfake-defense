import sys
import types
from unittest.mock import MagicMock

# --- Windows Environment Compatibility Fix ---
# Mocks the 'aim' tracking module to prevent import errors on Windows systems.
aim_module = types.ModuleType("aim")
aim_module.__spec__ = MagicMock()
aim_module.__path__ = []
sys.modules["aim"] = aim_module
aim_sdk_module = types.ModuleType("aim.sdk")
aim_sdk_module.__spec__ = MagicMock()
sys.modules["aim.sdk"] = aim_sdk_module
# ---------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMPipeline
from facenet_pytorch import InceptionResnetV1

# Requires the 'stargan' module from the CMUA-Watermark repository
try:
    from stargan.model import Generator
except ImportError:
    print("[ERROR] Could not import StarGAN. Please ensure the 'stargan' folder is in your project root.")

from utils import load_config
from robustness import RobustnessAugmentation 

class SurrogateEnsemble(nn.Module):
    """
    Tri-Architecture Surrogate Ensemble.
    
    This class loads and manages three distinct generative and biometric architectures 
    (FaceNet, DDPM, StarGAN) to optimize a Universal Adversarial Perturbation (UAP).
    To bypass strict VRAM constraints, models are initialized in evaluation mode, frozen, 
    and held in CPU RAM (Cold Storage) for Sequential Gradient Accumulation.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.config = load_config()
        self.augmentor = RobustnessAugmentation(device)
        
        print("\n[Ensemble] Initializing Geometrically-Aware Tri-Surrogates...")
        
        # 1. Biometric Identity Bottleneck (FaceNet)
        print("[Ensemble] Loading FaceNet Identity Surrogate...")
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().requires_grad_(False).to("cpu")

        # 2. Latent Diffusion Pipeline (DDPM U-Net)
        print("[Ensemble] Loading Diffusion U-Net...")
        try:
            pipeline = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
            self.unet = pipeline.unet.eval().requires_grad_(False).to("cpu")
        except Exception as e:
            print(f"[Warning] Diffusion load failed: {e}")
            self.unet = None

        # 3. Spatial Convolutional Synthesis (StarGAN)
        print("[Ensemble] Loading StarGAN Surrogate...")
        try:
            self.stargan = Generator(conv_dim=64, c_dim=5, repeat_num=6)
            stargan_weights_path = self.config['paths']['stargan_weights']            
            self.stargan.load_state_dict(torch.load(stargan_weights_path, map_location='cpu'))
            self.stargan.eval().requires_grad_(False).to("cpu")
        except Exception as e:
            print(f"[Warning] StarGAN load failed. Check weights at '{stargan_weights_path}': {e}")
            self.stargan = None

    def get_directional_loss(self, feat_clean, feat_pert):
        """
        Computes the Geometrically-Aware Directional Loss based on the SDSM strategy.
        
        Minimizes cosine similarity to maximize angular divergence in the deep feature space, 
        while subtracting a scaled Mean Squared Error (MSE) penalty to simultaneously 
        maximize absolute magnitude distance and prevent local minima collapse.
        """
        mse_loss = F.mse_loss(feat_clean, feat_pert)
        
        # Calculate angular distance (Cosine Similarity)
        if len(feat_clean.shape) == 4:
            cos_sim = F.cosine_similarity(feat_clean, feat_pert, dim=1).mean()
        else:
            cos_sim = F.cosine_similarity(feat_clean, feat_pert, dim=1).mean()
            
        return cos_sim - (0.1 * mse_loss)

    def forward_loss(self, x_clean, v, mode="facenet"):
        """
        Executes a forward pass for a specified surrogate model and returns the divergence loss.
        Gradients are accumulated externally to facilitate VRAM-efficient sequential training.
        """
        # Inject the universal shield and enforce digital image boundaries
        x_pert = torch.clamp(x_clean + v, 0, 1)
        
        # Expectation over Transformation (EoT): Apply stochastic spatial augmentations
        # to ensure the perturbation survives real-world pipeline transformations.
        combined = torch.cat([x_clean, x_pert], dim=0)
        combined_aug = self.augmentor(combined)
        x_aug, x_p = torch.chunk(combined_aug, 2, dim=0)
        
        w_geom = self.config['training']['lambda_geom']
        
        if mode == "facenet":
            # Preprocessing: Dimensional rigidity and standardization for FaceNet extraction
            x_aug_160 = F.interpolate(x_aug, size=(160, 160), mode='bilinear')
            x_p_160 = F.interpolate(x_p, size=(160, 160), mode='bilinear')
            
            x_aug_160 = (x_aug_160 - 0.5) / 0.5
            x_p_160 = (x_p_160 - 0.5) / 0.5

            with torch.no_grad():
                f_c = self.facenet(x_aug_160)
            f_p = self.facenet(x_p_160)
            return w_geom * self.get_directional_loss(f_c, f_p)
            
        elif mode == "diffusion" and self.unet is not None:
            # Random timestep sampling to target the Markovian noise prediction process
            t = torch.randint(500, 1000, (x_clean.shape[0],), device=self.device).long()
            with torch.no_grad():
                f_c = self.unet(x_aug, t).sample
            f_p = self.unet(x_p, t).sample
            return w_geom * self.get_directional_loss(f_c, f_p)

        elif mode == "gan" and self.stargan is not None:
            # Null conditional vector as the target is generalized structural disruption
            c_trg = torch.zeros(x_aug.size(0), 5, device=self.device)
            
            with torch.no_grad():
                out_c = self.stargan(x_aug, c_trg)
            out_p = self.stargan(x_p, c_trg)
            
            # StarGAN returns (final_image, feature_maps). 
            # Optimize across all intermediate convolutional layers to shatter spatial maps.
            if isinstance(out_c, tuple) and len(out_c) > 1:
                feat_list_c = out_c[1]
                feat_list_p = out_p[1]
                
                gan_loss = 0.0
                for f_c, f_p in zip(feat_list_c, feat_list_p):
                    gan_loss += self.get_directional_loss(f_c, f_p)
                
                return w_geom * (gan_loss / len(feat_list_c))
            else:
                # Fallback for alternative GAN architecture implementations
                return w_geom * self.get_directional_loss(out_c, out_p)
            
        # Failsafe tensor for unloaded models
        return torch.tensor(0.0, device=self.device, requires_grad=True)