#!/usr/bin/env python
"""
eval_ablation.py -- recover the hyperparameter ablation with correct metrics.

The original Table IV was measured with the evaluation that reported
DDPM U-Net statistics under FaceNet and StarGAN labels. Its numbers
cannot be used. The perturbation tensors from those runs survived, so
the table can be rebuilt properly at inference cost only.

Each run is evaluated against a noise baseline matched to ITS OWN
epsilon, since Run A used 0.045 while the others used 0.040. Comparing
all runs against a single baseline would confound the hyperparameter
effect with the budget.

Metrics per run: LPIPS/PSNR/SSIM, FaceNet-VGGFace2 (surrogate),
FaceNet-CASIA (held-out weights), ArcFace R50 (held-out architecture),
and DDPM U-Net MSE. StarGAN is skipped -- at 1.16x in the main
evaluation there is no headroom for an ablation to resolve.

Usage
-----
  python scripts/eval_ablation.py --run_folder <main_run> --n 250 ^
      --arcface models\\buffalo_l\\w600k_r50.onnx ^
      --run "A:run_ep5_eps0.045_geom2.0_lpips0.8_20260307_010242:5" ^
      --run "B:run_ep5_eps0.04_geom1.5_lpips1.2_20260307_184602:5" ^
      --run "C:run_ep5_eps0.04_geom1.2_lpips1.8_20260308_140040:5" ^
      --run "D:run_ep30_eps0.04_geom1.2_lpips1.8_20260309_233825:30" ^
      --run "E:run_ep5_eps0.04_geom1.0_lpips5.0_20260312_105648:5"

--run takes label:folder:epoch. Hyperparameters are parsed from the
folder name for the table; anything unparseable prints as "?".
"""

import _path  # noqa: F401  -- puts src/ and third_party/ on sys.path

import sys
import os
import re
import csv
import gc
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

from uap.config import load_config, get_device
from uap.perturbation import matched_noise


def parse_hparams(folder):
    def g(pat):
        m = re.search(pat, folder)
        return m.group(1) if m else "?"
    return g(r"eps([\d.]+?)_"), g(r"geom([\d.]+?)_"), g(r"lpips([\d.]+?)_")


def load_image(path, device):
    return transforms.ToTensor()(
        Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_folder", required=True, help="where the CSV is written")
    ap.add_argument("--run", action="append", required=True,
                    help="label:folder:epoch  (repeatable)")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--arcface", default=None)
    ap.add_argument("--skip-ddpm", action="store_true")
    ap.add_argument("--out", default="results_ablation.csv")
    args = ap.parse_args()

    import lpips as lpips_lib
    from facenet_pytorch import InceptionResnetV1

    cfg = load_config()
    device = get_device()
    results_root = cfg["paths"]["results_dir"]

    specs = []
    for spec in args.run:
        label, folder, epoch = spec.rsplit(":", 2)
        path = os.path.join(results_root, folder, f"perturbation_epoch_{epoch}.pt")
        if not os.path.exists(path):
            sys.exit(f"[ERROR] missing: {path}")
        v = torch.load(path, map_location=device, weights_only=True)
        v.requires_grad_(False)
        eps = v.abs().max().item()
        e, g, l = parse_hparams(folder)
        specs.append(dict(label=label, v=v, eps=eps, epoch=int(epoch),
                          e=e, g=g, l=l))

    test_dir = cfg["paths"]["test_data_output"]
    files = sorted(f for f in os.listdir(test_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[:args.n]

    print("=" * 78)
    print("  ABLATION UNDER THE CORRECTED PROTOCOL")
    print("=" * 78)
    print(f"  images: {len(files)}")
    for s in specs:
        print(f"    {s['label']}  eps={s['e']:>5}  geom={s['g']:>4}  "
              f"lpips={s['l']:>4}  ep={s['epoch']:>2}   "
              f"tensor L_inf={s['eps']:.4f}")

    rows = []          # run, image, variant, metric, value

    def variants(x, s, i):
        """UAP for this run, plus noise matched to THIS run's epsilon."""
        n = matched_noise(s["v"].shape, s["eps"], i, device)
        return {"uap": torch.clamp(x + s["v"], 0, 1),
                "rand": torch.clamp(x + n, 0, 1)}

    # ---------------- stealth ----------------
    print("\n[1/4] perceptual cost")
    fn = lpips_lib.LPIPS(net="alex").to(device)
    with torch.no_grad():
        for i, f in enumerate(tqdm(files, desc="  stealth")):
            x = load_image(os.path.join(test_dir, f), device)
            ref = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
            for s in specs:
                for name, xp in variants(x, s, i).items():
                    rows.append((s["label"], f, name, "lpips_alex",
                                 fn(x * 2 - 1, xp * 2 - 1).item()))
                    c = xp.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    rows.append((s["label"], f, name, "psnr",
                                 sk_psnr(ref, c, data_range=1.0)))
                    rows.append((s["label"], f, name, "ssim",
                                 sk_ssim(ref, c, data_range=1.0, channel_axis=2,
                                         win_size=11, gaussian_weights=True)))
    del fn; gc.collect(); torch.cuda.empty_cache()

    # ---------------- FaceNet ----------------
    for j, w in enumerate(("vggface2", "casia-webface"), start=2):
        tag = "facenet_vggface2" if w == "vggface2" else "facenet_casia"
        print(f"\n[{j}/4] {tag}")
        net = InceptionResnetV1(pretrained=w).eval().requires_grad_(False).to(device)

        def embed(img):
            z = F.interpolate(img, size=(160, 160), mode="bilinear", align_corners=False)
            return net((z - 0.5) / 0.5)

        with torch.no_grad():
            for i, f in enumerate(tqdm(files, desc=f"  {tag}")):
                x = load_image(os.path.join(test_dir, f), device)
                e_ref = embed(x)
                for s in specs:
                    for name, xp in variants(x, s, i).items():
                        rows.append((s["label"], f, name, f"{tag}_cosdist",
                                     1.0 - F.cosine_similarity(e_ref, embed(xp)).item()))
        del net; gc.collect(); torch.cuda.empty_cache()

    # ---------------- ArcFace ----------------
    if args.arcface:
        print("\n[+] ArcFace R50 (held-out architecture)")
        import onnxruntime as ort
        from eval_arcface import preprocess, cosdist
        so = ort.SessionOptions(); so.log_severity_level = 3
        sess = ort.InferenceSession(args.arcface, so,
                                    providers=["CPUExecutionProvider"])
        i_n, o_n = sess.get_inputs()[0].name, sess.get_outputs()[0].name
        for i, f in enumerate(tqdm(files, desc="  arcface")):
            x = load_image(os.path.join(test_dir, f), device)
            batch, keys = [preprocess(x)], []
            for s in specs:
                for name, xp in variants(x, s, i).items():
                    batch.append(preprocess(xp)); keys.append((s["label"], name))
            feats = sess.run([o_n], {i_n: np.stack(batch)})[0]
            for (lab, name), ft in zip(keys, feats[1:]):
                rows.append((lab, f, name, "arcface_buffalo_l_cosdist",
                             cosdist(feats[0], ft)))

    # ---------------- DDPM ----------------
    if not args.skip_ddpm:
        print("\n[4/4] DDPM U-Net")
        from diffusers import DDPMPipeline
        unet = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").unet
        unet = unet.eval().requires_grad_(False).to(device)
        t = torch.tensor([500], device=device).long()
        with torch.no_grad():
            for i, f in enumerate(tqdm(files, desc="  ddpm")):
                x = load_image(os.path.join(test_dir, f), device)
                n_ref = unet(x * 2 - 1, t).sample
                for s in specs:
                    for name, xp in variants(x, s, i).items():
                        rows.append((s["label"], f, name, "ddpm_mse_t500",
                                     F.mse_loss(n_ref, unet(xp * 2 - 1, t).sample).item()))
        del unet; gc.collect(); torch.cuda.empty_cache()

    out_path = os.path.join(results_root, args.run_folder, args.out)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "image", "variant", "metric", "value"])
        w.writerows(rows)
    print(f"\n[+] {len(rows)} records -> {out_path}")

    # ---------------- table ----------------
    import pandas as pd
    df = pd.DataFrame(rows, columns=["run", "image", "variant", "metric", "value"])
    piv = df.pivot_table(index=["run", "image", "metric"], columns="variant",
                         values="value").reset_index()

    def cell(lab, metric, col="uap"):
        s = piv[(piv["run"] == lab) & (piv["metric"] == metric)]
        return s[col].mean() if len(s) else float("nan")

    print("\n" + "=" * 96)
    print(f"  ABLATION  (n = {len(files)}; each run vs noise matched to its own epsilon)")
    print("=" * 96)
    hdr = (f"  {'Run':<4}{'eps':>7}{'geom':>6}{'lpips':>7}{'ep':>4}"
           f"{'LPIPS':>9}{'SSIM':>8}{'VGGFace2':>11}{'CASIA':>10}"
           f"{'ArcFace':>10}{'DDPM':>10}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for s in specs:
        lab = s["label"]
        print(f"  {lab:<4}{s['e']:>7}{s['g']:>6}{s['l']:>7}{s['epoch']:>4}"
              f"{cell(lab,'lpips_alex'):>9.4f}{cell(lab,'ssim'):>8.4f}"
              f"{cell(lab,'facenet_vggface2_cosdist'):>11.4f}"
              f"{cell(lab,'facenet_casia_cosdist'):>10.4f}"
              f"{cell(lab,'arcface_buffalo_l_cosdist'):>10.4f}"
              f"{cell(lab,'ddpm_mse_t500'):>10.5f}")

    print("\n  Multipliers over each run's own matched-epsilon baseline:")
    hdr2 = f"  {'Run':<4}{'VGGFace2':>11}{'CASIA':>10}{'ArcFace':>10}{'DDPM':>10}"
    print(hdr2); print("  " + "-" * (len(hdr2) - 2))
    for s in specs:
        lab = s["label"]
        out = f"  {lab:<4}"
        for m in ("facenet_vggface2_cosdist", "facenet_casia_cosdist",
                  "arcface_buffalo_l_cosdist", "ddpm_mse_t500"):
            u, r = cell(lab, m, "uap"), cell(lab, m, "rand")
            out += f"{(u / r if r else float('nan')):>9.1f}x" if not np.isnan(u) else f"{'--':>10}"
        print(out)

    print("\n  Read the LPIPS column against the disruption columns. The old")
    print("  Table IV asserted one setting was optimal; this shows the actual")
    print("  trade-off, and C vs D isolates the effect of training length at")
    print("  identical hyperparameters.")


if __name__ == "__main__":
    main()
