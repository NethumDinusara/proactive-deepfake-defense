# test_memory.py
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

import torch
from models import SurrogateEnsemble
from utils import get_device

def test_memory():
    print("--- Starting 4GB VRAM Memory Test (Gradient Accumulation) ---")
    try:
        device = get_device()
        print(f"Device: {device}")
        
        # 1. Initialize Ensemble
        ensemble = SurrogateEnsemble(device)
        
        # 2. Create Dummy Data
        x = torch.randn(1, 3, 256, 256).to(device)
        
        # Perturbation v must be a leaf tensor with grad enabled
        v = torch.zeros(1, 3, 256, 256).to(device)
        v.requires_grad = True
        
        # 3. Run Compute & Backward
        print("Running Swap-Compute-Backward...")
        
        # This function handles the backward pass internally!
        loss_val = ensemble.compute_and_backward(x, v)
        
        print(f"Loss Value: {loss_val}")
        
        # 4. Check if gradients exist
        if v.grad is not None:
            print(f"[SUCCESS] Gradients computed! v.grad norm: {v.grad.norm().item()}")
        else:
            print("[FAIL] No gradients found in v.")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n[FAIL] GPU Out of Memory!")
        else:
            print(f"\n[FAIL] Runtime Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_memory()