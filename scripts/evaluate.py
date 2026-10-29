#!/usr/bin/env python
"""
evaluate.py -- corrected evaluation for the SDSM-UAP proactive defense.

Differences from archive/code/evaluate.py, all deliberate:

  1. Each model is measured on ITSELF. FaceNet cosine distance comes from
     FaceNet, StarGAN MSE from StarGAN, DDPM MSE from the DDPM U-Net.
     archive/code/evaluate.py derived every "protection" number from the U-Net alone.

  2. ASR is the empirical fraction of images exceeding a threshold, computed
     from the per-image values. No distribution is fitted.

  3. Both FaceNet weight sets are evaluated: vggface2 (the training surrogate,
     white-box) and casia-webface (held-out weights). They are reported
     separately and labelled as such.

  4. The DDPM U-Net receives inputs in [-1, 1], which is what
     google/ddpm-celebahq-256 expects. train.py and archive/code/evaluate.py fed it [0, 1].
     Pass --unet-input 01 to reproduce the original behaviour for comparison.

  5. SSIM uses win_size=11 with gaussian_weights=True, matching Wang et al.
     archive/code/evaluate.py used win_size=3.

  6. LPIPS is reported for both alex and vgg backbones. Training constrained
     vgg; archive/code/evaluate.py reported alex.

  7. Random-noise baseline uses the same epsilon as the UAP, drawn fresh per
     image with a fixed seed for reproducibility.

  8. Optional delta-scaling sweep (--scales) produces the stealth/robustness
     operating curve from the single saved tensor. No retraining.

Every per-image value is written to results_v2.csv in long format
(image, variant, metric, value) so statistics can be recomputed offline.

Usage
-----
  # headline run, ~15 min on a 4 GB card
  python scripts/evaluate.py --run_folder run_ep30_eps0.04_geom1.2_lpips1.8_20260309_233825

  # operating curve on a subset, run this second
  python scripts/evaluate.py --run_folder <same> --n 250 \
      --scales 0.25,0.5,0.75,1.0 --out results_v2_curve.csv

  # denoising-timestep sensitivity
  python scripts/evaluate.py --run_folder <same> --n 250 --timesteps 100,300,500,700,900
"""

import _path  # noqa: F401  -- puts src/ and third_party/ on sys.path
from uap import compat  # noqa: F401  -- stubs `aim` before torch/diffusers

import sys
import argparse


import os
import csv
import gc

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from uap.config import load_config, get_device, ensure_dir
from uap.data import list_images, load_image
from uap.perturbation import build_variants, load_uap, uap_path


# ----------------------------------------------------------------------
# Recorder
# ----------------------------------------------------------------------

class Recorder:
    """Long-format accumulator: (image, variant, metric) -> value."""

    def __init__(self):
        self.rows = []

    def add(self, image, variant, metric, value):
        self.rows.append((image, variant, metric, float(value)))

    def write(self, path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "variant", "metric", "value"])
            w.writerows(self.rows)
        print(f"\n[+] {len(self.rows)} per-image records written to {path}")

    def summary(self):
        agg = {}
        for _, variant, metric, value in self.rows:
            agg.setdefault((metric, variant), []).append(value)
        return agg


# ----------------------------------------------------------------------
# Stealth metrics
# ----------------------------------------------------------------------

def pass_stealth(files, test_dir, v, scales, epsilon, device, rec):
    import lpips as lpips_lib

    print("\n[1/4] Stealth metrics (LPIPS alex + vgg, PSNR, SSIM)")
    fn_alex = lpips_lib.LPIPS(net="alex").to(device)
    fn_vgg = lpips_lib.LPIPS(net="vgg").to(device)

    with torch.no_grad():
        for i, fname in enumerate(tqdm(files, desc="  stealth")):
            x = load_image(test_dir, fname, device)
            ref = x.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            for name, xp in build_variants(x, v, scales, epsilon, i).items():
                rec.add(fname, name, "lpips_alex",
                        fn_alex(x * 2 - 1, xp * 2 - 1).item())
                rec.add(fname, name, "lpips_vgg",
                        fn_vgg(x * 2 - 1, xp * 2 - 1).item())

                cand = xp.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                rec.add(fname, name, "psnr", sk_psnr(ref, cand, data_range=1.0))
                rec.add(fname, name, "ssim",
                        sk_ssim(ref, cand, data_range=1.0, channel_axis=2,
                                win_size=11, gaussian_weights=True))

    del fn_alex, fn_vgg
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# FaceNet -- measured on FaceNet
# ----------------------------------------------------------------------

def pass_facenet(files, test_dir, v, scales, epsilon, device, rec, weights):
    from facenet_pytorch import InceptionResnetV1

    tag = "facenet_vggface2" if weights == "vggface2" else "facenet_casia"
    role = "white-box surrogate" if weights == "vggface2" else "held-out weights"
    print(f"\n[2/4] FaceNet ({weights}) -- {role}")

    net = InceptionResnetV1(pretrained=weights).eval().requires_grad_(False).to(device)

    def embed(img):
        z = F.interpolate(img, size=(160, 160), mode="bilinear", align_corners=False)
        return net((z - 0.5) / 0.5)

    with torch.no_grad():
        for i, fname in enumerate(tqdm(files, desc=f"  {tag}")):
            x = load_image(test_dir, fname, device)
            e_clean = embed(x)
            for name, xp in build_variants(x, v, scales, epsilon, i).items():
                cos = F.cosine_similarity(e_clean, embed(xp)).item()
                rec.add(fname, name, f"{tag}_cosdist", 1.0 - cos)

    del net
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# StarGAN -- measured on StarGAN
# ----------------------------------------------------------------------

def pass_stargan(files, test_dir, v, scales, epsilon, device, rec, weights_path):
    try:
        from stargan.model import Generator
    except ImportError:
        print("\n[3/4] StarGAN -- SKIPPED (stargan module not importable)")
        return

    print("\n[3/4] StarGAN -- white-box surrogate")
    try:
        net = Generator(conv_dim=64, c_dim=5, repeat_num=6)
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        net = net.eval().requires_grad_(False).to(device)
    except Exception as exc:
        print(f"      SKIPPED (weight load failed: {exc})")
        return

    # Black hair, matching archive/code/whitebox_eval.py. A null vector is not an edit.
    c_trg = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=device)

    def generate(img):
        out = net((img - 0.5) / 0.5, c_trg)
        img_out = out[0] if isinstance(out, tuple) else out
        feats = out[1] if (isinstance(out, tuple) and len(out) > 1) else None
        return img_out, feats

    with torch.no_grad():
        for i, fname in enumerate(tqdm(files, desc="  stargan")):
            x = load_image(test_dir, fname, device)
            g_clean, f_clean = generate(x)

            for name, xp in build_variants(x, v, scales, epsilon, i).items():
                g_pert, f_pert = generate(xp)
                # Output-space disruption: how wrecked is the edited image.
                rec.add(fname, name, "stargan_out_mse",
                        F.mse_loss(g_clean, g_pert).item())
                # Feature-space disruption: what the loss actually optimised.
                if f_clean is not None:
                    dists = [1.0 - F.cosine_similarity(a, b, dim=1).mean().item()
                             for a, b in zip(f_clean, f_pert)]
                    rec.add(fname, name, "stargan_feat_cosdist",
                            float(np.mean(dists)))

    del net
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# DDPM U-Net -- measured on the U-Net
# ----------------------------------------------------------------------

def pass_ddpm(files, test_dir, v, scales, epsilon, device, rec,
              timesteps, unet_input):
    from diffusers import DDPMPipeline

    print(f"\n[4/4] DDPM U-Net -- timesteps {timesteps}, input range "
          f"{'[-1,1]' if unet_input == 'pm1' else '[0,1]'}")

    try:
        unet = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").unet
        unet = unet.eval().requires_grad_(False).to(device)
    except Exception as exc:
        print(f"      SKIPPED (load failed: {exc})")
        return

    def prep(img):
        return img * 2 - 1 if unet_input == "pm1" else img

    with torch.no_grad():
        for i, fname in enumerate(tqdm(files, desc="  ddpm")):
            x = load_image(test_dir, fname, device)
            variants = build_variants(x, v, scales, epsilon, i)

            for t_val in timesteps:
                t = torch.tensor([t_val], device=device).long()
                n_clean = unet(prep(x), t).sample

                for name, xp in variants.items():
                    n_pert = unet(prep(xp), t).sample
                    rec.add(fname, name, f"ddpm_mse_t{t_val}",
                            F.mse_loss(n_clean, n_pert).item())
                    cos = F.cosine_similarity(n_clean, n_pert, dim=1).mean().item()
                    rec.add(fname, name, f"ddpm_cosdist_t{t_val}", 1.0 - cos)

    del unet
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# Console summary
# ----------------------------------------------------------------------

def print_summary(rec, n_images):
    agg = rec.summary()
    metrics = sorted({m for m, _ in agg})
    variants = sorted({v for _, v in agg})

    print("\n" + "=" * 78)
    print(f"  MEAN +/- SD OVER {n_images} HELD-OUT LFW IMAGES")
    print("=" * 78)
    header = f"{'metric':<28}" + "".join(f"{v:>16}" for v in variants)
    print(header)
    print("-" * len(header))
    for m in metrics:
        line = f"{m:<28}"
        for v in variants:
            vals = agg.get((m, v))
            line += f"{np.mean(vals):>9.4f}+-{np.std(vals):<5.4f}" if vals else f"{'--':>16}"
        print(line)
    print("=" * 78)
    print("Thresholded rates are NOT printed here. Compute them from the CSV in\n"
          "stats.py, where the threshold is stated explicitly and swept.")


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_folder", required=True)
    ap.add_argument("--n", type=int, default=None, help="limit image count")
    ap.add_argument("--scales", default="1.0",
                    help="comma-separated delta multipliers, e.g. 0.25,0.5,0.75,1.0")
    ap.add_argument("--timesteps", default="500",
                    help="comma-separated DDPM timesteps")
    ap.add_argument("--unet-input", choices=["pm1", "01"], default="pm1",
                    help="pm1 = correct [-1,1]; 01 = reproduce original bug")
    ap.add_argument("--skip", default="", help="comma list: stealth,facenet,stargan,ddpm")
    ap.add_argument("--out", default="results_v2.csv")
    args = ap.parse_args()

    cfg = load_config()
    device = get_device()
    scales = [float(s) for s in args.scales.split(",")]
    timesteps = [int(t) for t in args.timesteps.split(",")]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    run_dir = os.path.join(cfg["paths"]["results_dir"], args.run_folder)
    epoch = cfg["training"]["epochs"]
    epsilon = cfg["training"]["epsilon"]
    tensor_path = uap_path(cfg["paths"]["results_dir"], args.run_folder, epoch)
    try:
        v, measured_eps = load_uap(tensor_path, device)
    except FileNotFoundError as exc:
        sys.exit(f"[ERROR] {exc}")

    if abs(measured_eps - epsilon) > 1e-4:
        print(f"[WARN] config epsilon={epsilon} but tensor L_inf={measured_eps:.5f}. "
              f"Using tensor value for the noise baseline.")
        epsilon = measured_eps

    test_dir = cfg["paths"]["test_data_output"]
    files = list_images(test_dir, args.n)
    if not files:
        sys.exit(f"[ERROR] no images in {test_dir}")

    print("=" * 78)
    print("  SDSM-UAP CORRECTED EVALUATION")
    print("=" * 78)
    print(f"  device        : {device}")
    print(f"  UAP           : {tensor_path}")
    print(f"  L_inf         : {measured_eps:.5f}  ({measured_eps * 255:.2f}/255)")
    print(f"  eval set      : {len(files)} held-out LFW images from {test_dir}")
    print(f"  delta scales  : {scales}")
    print(f"  baseline      : uniform noise, epsilon matched at {epsilon:.5f}")

    rec = Recorder()
    ensure_dir(run_dir)

    if "stealth" not in skip:
        pass_stealth(files, test_dir, v, scales, epsilon, device, rec)
    if "facenet" not in skip:
        pass_facenet(files, test_dir, v, scales, epsilon, device, rec, "vggface2")
        pass_facenet(files, test_dir, v, scales, epsilon, device, rec, "casia-webface")
    if "stargan" not in skip:
        pass_stargan(files, test_dir, v, scales, epsilon, device, rec,
                     cfg["paths"]["stargan_weights"])
    if "ddpm" not in skip:
        pass_ddpm(files, test_dir, v, scales, epsilon, device, rec,
                  timesteps, args.unet_input)

    out_path = os.path.join(run_dir, args.out)
    rec.write(out_path)
    print_summary(rec, len(files))


if __name__ == "__main__":
    main()
