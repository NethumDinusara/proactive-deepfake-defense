#!/usr/bin/env python
"""
eval_heldout_ddpm.py -- does generative disruption transfer to an UNSEEN
diffusion model?

Every generative number in the paper is currently white-box: both the
DDPM U-Net and StarGAN were surrogates. Reviewer 1 of the previous
submission asked specifically about transferability to diffusion models
not used during optimisation. This script answers it.

Deliberately standalone. evaluate.py produced the paper's headline
table and is not touched.

CHOOSING A CHECKPOINT -- this determines what you can claim
----------------------------------------------------------
  google/ddpm-ema-celebahq-256
      Same architecture and training data, EMA weights instead of raw.
      Analogous to the VGGFace2 -> CASIA relationship on the biometric
      side: a held-out CHECKPOINT. Cleanest test, no confound.

  google/ddpm-church-256   /   google/ddpm-bedroom-256
      Different training domain entirely. These models never saw a face,
      so their noise prediction on facial input is already out of
      distribution. Any "disruption" measured is confounded with that
      domain shift, and a reviewer may say so. If you use one of these,
      disclose the confound explicitly.

The script tries the candidates in order and reports which loaded, so
you can see what you actually measured rather than assuming.

Usage
-----
    python scripts/eval_heldout_ddpm.py --run_folder <run> --n 250
    python scripts/eval_heldout_ddpm.py --run_folder <run> --n 250 \
        --model google/ddpm-church-256 --out results_ddpm_church.csv
"""

import _path  # noqa: F401  -- puts src/ and third_party/ on sys.path
from uap import compat  # noqa: F401  -- stubs `aim` before torch/diffusers

import sys
import os
import csv
import argparse


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from uap.config import load_config, get_device
from uap.perturbation import matched_noise

CANDIDATES = [
    ("google/ddpm-ema-celebahq-256",
     "held-out CHECKPOINT: same architecture and data, EMA weights"),
    ("google/ddpm-church-256",
     "held-out DOMAIN: never trained on faces -- confounded, disclose it"),
    ("google/ddpm-bedroom-256",
     "held-out DOMAIN: never trained on faces -- confounded, disclose it"),
]


def load_unet(explicit, device):
    from diffusers import DDPMPipeline

    tries = [(explicit, "user-specified")] if explicit else CANDIDATES
    for name, note in tries:
        try:
            print(f"[*] trying {name} ...")
            unet = DDPMPipeline.from_pretrained(name).unet
            unet = unet.eval().requires_grad_(False).to(device)
            print(f"    loaded. {note}")
            return unet, name, note
        except Exception as exc:
            print(f"    failed: {type(exc).__name__}: {str(exc)[:90]}")
    sys.exit("\n[ERROR] no checkpoint loaded. Check network access, or pass "
             "--model with one you know is available.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_folder", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--timesteps", default="500")
    ap.add_argument("--out", default="results_ddpm_heldout.csv")
    args = ap.parse_args()

    cfg = load_config()
    device = get_device()
    timesteps = [int(t) for t in args.timesteps.split(",")]

    run_dir = os.path.join(cfg["paths"]["results_dir"], args.run_folder)
    uap_path = os.path.join(run_dir,
                            f"perturbation_epoch_{cfg['training']['epochs']}.pt")
    if not os.path.exists(uap_path):
        sys.exit(f"[ERROR] UAP not found: {uap_path}")
    v = torch.load(uap_path, map_location=device, weights_only=True)
    v.requires_grad_(False)
    epsilon = v.abs().max().item()

    test_dir = cfg["paths"]["test_data_output"]
    files = sorted(f for f in os.listdir(test_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[:args.n]

    print("=" * 74)
    print("  HELD-OUT DIFFUSION MODEL")
    print("=" * 74)
    print(f"  images    : {len(files)}")
    print(f"  epsilon   : {epsilon:.5f}")
    print(f"  timesteps : {timesteps}")
    print()

    unet, model_name, note = load_unet(args.model, device)
    tag = model_name.split("/")[-1].replace("-", "_")

    rows = []
    with torch.no_grad():
        for i, fname in enumerate(tqdm(files, desc="  heldout ddpm")):
            x = transforms.ToTensor()(
                Image.open(os.path.join(test_dir, fname)).convert("RGB")
            ).unsqueeze(0).to(device)

            x_uap = torch.clamp(x + v, 0, 1)
            noise = matched_noise(v.shape, epsilon, i, device)
            x_rand = torch.clamp(x + noise, 0, 1)

            for t_val in timesteps:
                t = torch.tensor([t_val], device=device).long()
                ref = unet(x * 2 - 1, t).sample          # correct [-1,1] range
                for name, xp in (("uap", x_uap), ("rand", x_rand)):
                    pred = unet(xp * 2 - 1, t).sample
                    rows.append((fname, name, f"{tag}_mse_t{t_val}",
                                 F.mse_loss(ref, pred).item()))
                    cos = F.cosine_similarity(ref, pred, dim=1).mean().item()
                    rows.append((fname, name, f"{tag}_cosdist_t{t_val}",
                                 1.0 - cos))

    out_path = os.path.join(run_dir, args.out)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "variant", "metric", "value"])
        w.writerows(rows)
    print(f"\n[+] {len(rows)} records -> {out_path}")

    # ---------------- summary ----------------
    import collections
    from scipy import stats as st

    agg = collections.defaultdict(dict)
    for image, variant, metric, value in rows:
        agg[(image, metric)][variant] = value
    metrics = sorted({m for _, m in agg})

    print("\n" + "=" * 74)
    print(f"  {model_name}")
    print(f"  {note}")
    print("=" * 74)
    print(f"  {'metric':<28}{'noise':>10}{'uap':>10}{'ratio':>9}"
          f"{'win%':>8}{'p':>11}")
    print("  " + "-" * 64)
    for m in metrics:
        keys = [k for k in agg if k[1] == m]
        a = np.array([agg[k]["uap"] for k in keys])
        b = np.array([agg[k]["rand"] for k in keys])
        _, p = st.wilcoxon(a, b)
        print(f"  {m:<28}{b.mean():>10.5f}{a.mean():>10.5f}"
              f"{a.mean()/b.mean():>8.2f}x{100*(a>b).mean():>7.1f}%{p:>11.1e}")

    mse = [m for m in metrics if "_mse_" in m]
    if mse:
        keys = [k for k in agg if k[1] == mse[0]]
        b = np.array([agg[k]["rand"] for k in keys])
        a = np.array([agg[k]["uap"] for k in keys])
        tau = np.percentile(b, 99)
        print(f"\n  tau_99 = {tau:.5f}")
        print(f"  success at tau_99:  noise 1.0%   UAP {100*(a>tau).mean():.1f}%")

    print("\n  For comparison, the SURROGATE DDPM (Table II, n=1000):")
    print("    MSE 0.0072 -> 0.0114, ratio 1.57x, 100% at tau_99")
    print()
    print("  If the ratio here is comfortably above 1.0, generative disruption")
    print("  transfers and the corresponding limitation can be removed. If it")
    print("  sits at or below 1.0, report that: a negative result on held-out")
    print("  generative transfer is a real finding, and it is the question")
    print("  Reviewer 1 actually asked.")


if __name__ == "__main__":
    main()