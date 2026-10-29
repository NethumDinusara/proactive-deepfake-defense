"""
Perturbation loading and the matched-epsilon noise control.

NUMERICAL CONTRACT -- do not "clean up" `matched_noise`.

The random-noise baseline is the reference every reported multiplier is
measured against. Nine separate scripts generated it inline, and they agreed
only because each repeated the same four operations in the same order:

    torch.Generator(device="cpu").manual_seed(1000 + index)
    torch.rand(shape, generator=g) * 2 - 1      # on CPU, in float32
    .to(device)                                 # move BEFORE scaling
    * epsilon

Seeding per image index (not per run) is what makes the control reproducible
and lets separate scripts -- evaluate, arcface, robustness, the baselines --
compare against bitwise-identical noise for the same image. Changing the seed
base, the operation order, or the device the draw happens on silently shifts
every baseline in the paper.
"""
import os

import torch


def load_uap(path, device):
    """Load a saved perturbation tensor, frozen, with its measured L-inf."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"UAP tensor not found: {path}")
    v = torch.load(path, map_location=device, weights_only=True)
    v.requires_grad_(False)
    return v, v.abs().max().item()


def uap_path(results_dir, run_folder, epoch):
    """Conventional checkpoint location for a run."""
    return os.path.join(results_dir, run_folder, f"perturbation_epoch_{epoch}.pt")


def matched_noise(shape, epsilon, index, device):
    """Uniform noise in [-epsilon, epsilon], deterministic per image index.

    See the NUMERICAL CONTRACT above before altering anything here.
    """
    g = torch.Generator(device="cpu").manual_seed(1000 + index)
    return (torch.rand(shape, generator=g) * 2 - 1).to(device) * epsilon


def apply(x_clean, v, scale=1.0):
    """Add the perturbation and clamp back to valid image range."""
    return torch.clamp(x_clean + scale * v, 0, 1)


def variant_name(scale):
    """`uap` at full strength, `uap<k>` when scaled -- the CSV convention."""
    return "uap" if abs(scale - 1.0) < 1e-9 else f"uap{scale:g}"


def build_variants(x_clean, v, scales, epsilon, index):
    """{variant_name: perturbed_image} for each scale, plus the noise control."""
    out = {variant_name(k): apply(x_clean, v, k) for k in scales}
    out["rand"] = torch.clamp(
        x_clean + matched_noise(v.shape, epsilon, index, v.device), 0, 1
    )
    return out
