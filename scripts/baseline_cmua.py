#!/usr/bin/env python
"""
baseline_cmua.py -- CMUA-Watermark under our protocol.

CMUA (Huang et al., AAAI 2022) is the closest published comparison: it is
also a cross-model universal adversarial watermark for facial images. It
is also the only strong baseline obtainable without re-running someone
else's optimisation, because the released artefact is a pre-computed
perturbation tensor -- applying it is a tensor addition.

    https://github.com/VDIGPKU/CMUA-Watermark

Download their released perturbation and pass its path with --cmua.

TWO COMPARISONS, and the second is the one worth reporting
----------------------------------------------------------
1. Native. CMUA at its released magnitude vs our UAP at ours. Honest but
   confounded: if the budgets differ, the comparison is partly about
   epsilon rather than about method.

2. Matched perceptual cost. Because our perturbation scales
   (results_v2_curve.csv), we can pick the k whose LPIPS most closely
   matches CMUA's and compare there. Very few proactive-defence papers
   do this, and the operating curve is what makes it possible. Report
   this as the primary comparison and the native one as context.

Whatever the outcome, report it. A baseline you ran and lost to is worth
more than a baseline you omitted -- the first is a finding, the second is
the gap that got the previous submission rejected.

VERIFY BEFORE TRUSTING THE NUMBERS
----------------------------------
Check the released tensor's own conventions, which this script prints on
load. In particular confirm whether it is stored in [0,1] or [-1,1]
image scale; if the latter, pass --cmua-scale 0.5 to convert. Their
evaluation resolution may also differ from 256; the tensor is bilinearly
resized if needed, which is disclosed in the output.

Usage
-----
    python scripts/baseline_cmua.py --run_folder <run> --cmua path/to/cmua.pt --n 1000
"""

import _path  # noqa: F401  -- puts src/ and third_party/ on sys.path

import sys
import os
import csv
import gc
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from uap.config import load_config, get_device
from uap.perturbation import matched_noise


def load_cmua(path, device, scale):
    """Accept .pt/.pth/.npy holding a tensor, or a dict with one inside."""
    if path.endswith(".npy"):
        obj = torch.from_numpy(np.load(path))
    else:
        obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        cands = {k: t for k, t in obj.items()
                 if torch.is_tensor(t) and t.dim() in (3, 4)}
        if not cands:
            sys.exit(f"[ERROR] no perturbation tensor in {path}. Keys: {list(obj)}")
        key = next((k for k in cands if "pert" in k.lower() or "water" in k.lower()),
                   next(iter(cands)))
        print(f"[*] using key '{key}' from checkpoint")
        obj = cands[key]

    t = obj.detach().float()
    if t.dim() == 3:
        t = t.unsqueeze(0)

    print(f"[*] CMUA tensor: shape {tuple(t.shape)}  "
          f"range [{t.min():.4f}, {t.max():.4f}]  L_inf {t.abs().max():.4f}")
    if scale != 1.0:
        t = t * scale
        print(f"    scaled by {scale} -> L_inf {t.abs().max():.4f}")
    if t.abs().max() > 0.5:
        print("    [WARN] L_inf > 0.5 suggests [-1,1] or 0-255 storage.")
        print("           Re-run with --cmua-scale 0.5 (or 1/255) if so.")
    return t.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_folder", required=True)
    ap.add_argument("--cmua", required=True, help="path to released CMUA tensor")
    ap.add_argument("--cmua-scale", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--match-k", type=float, default=None,
                    help="also emit our UAP at this scale, for matched-LPIPS")
    ap.add_argument("--out", default="results_cmua.csv")
    args = ap.parse_args()

    import lpips as lpips_lib
    from facenet_pytorch import InceptionResnetV1

    cfg = load_config()
    device = get_device()

    run_dir = os.path.join(cfg["paths"]["results_dir"], args.run_folder)
    uap_path = os.path.join(run_dir, f"perturbation_epoch_{cfg['training']['epochs']}.pt")
    v = torch.load(uap_path, map_location=device, weights_only=True)
    v.requires_grad_(False)
    epsilon = v.abs().max().item()

    cmua = load_cmua(args.cmua, device, args.cmua_scale)
    if cmua.shape[-2:] != v.shape[-2:]:
        print(f"[*] resizing CMUA {tuple(cmua.shape[-2:])} -> {tuple(v.shape[-2:])} "
              f"(bilinear; disclose this in the paper)")
        cmua = F.interpolate(cmua, size=v.shape[-2:], mode="bilinear",
                             align_corners=False)

    test_dir = cfg["paths"]["test_data_output"]
    files = sorted(f for f in os.listdir(test_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[:args.n]

    print("=" * 74)
    print("  CMUA-WATERMARK BASELINE, IDENTICAL PROTOCOL")
    print("=" * 74)
    print(f"  images : {len(files)}")
    print(f"  ours   : L_inf {epsilon:.4f}   CMUA: L_inf {cmua.abs().max():.4f}")

    def build(x, i):
        out = {"uap": torch.clamp(x + v, 0, 1),
               "cmua": torch.clamp(x + cmua, 0, 1)}
        if args.match_k:
            out[f"uap{args.match_k:g}"] = torch.clamp(x + args.match_k * v, 0, 1)
        n = matched_noise(v.shape, epsilon, i, device)
        out["rand"] = torch.clamp(x + n, 0, 1)
        return out

    rows = []

    # ---- perceptual ----
    print("\n[1/2] perceptual cost")
    fn = lpips_lib.LPIPS(net="alex").to(device)
    with torch.no_grad():
        for i, fname in enumerate(tqdm(files, desc="  lpips")):
            x = transforms.ToTensor()(
                Image.open(os.path.join(test_dir, fname)).convert("RGB")
            ).unsqueeze(0).to(device)
            for name, xp in build(x, i).items():
                rows.append((fname, name, "lpips_alex",
                             fn(x * 2 - 1, xp * 2 - 1).item()))
    del fn; gc.collect(); torch.cuda.empty_cache()

    # ---- biometric, both weight sets ----
    for weights in ("vggface2", "casia-webface"):
        tag = "facenet_vggface2" if weights == "vggface2" else "facenet_casia"
        print(f"\n[2/2] {tag}")
        net = InceptionResnetV1(pretrained=weights).eval().requires_grad_(False).to(device)

        def embed(img):
            z = F.interpolate(img, size=(160, 160), mode="bilinear", align_corners=False)
            return net((z - 0.5) / 0.5)

        with torch.no_grad():
            for i, fname in enumerate(tqdm(files, desc=f"  {tag}")):
                x = transforms.ToTensor()(
                    Image.open(os.path.join(test_dir, fname)).convert("RGB")
                ).unsqueeze(0).to(device)
                e_ref = embed(x)
                for name, xp in build(x, i).items():
                    cos = F.cosine_similarity(e_ref, embed(xp)).item()
                    rows.append((fname, name, f"{tag}_cosdist", 1.0 - cos))

        del net; gc.collect(); torch.cuda.empty_cache()

    out_path = os.path.join(run_dir, args.out)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "variant", "metric", "value"])
        w.writerows(rows)
    print(f"\n[+] {len(rows)} records -> {out_path}")

    # ---- summary ----
    import pandas as pd
    from scipy import stats as st

    df = pd.DataFrame(rows, columns=["image", "variant", "metric", "value"])
    wide = df.pivot_table(index=["image", "metric"], columns="variant",
                          values="value").reset_index()
    variants = [c for c in wide.columns if c not in ("image", "metric")]

    print("\n" + "=" * 74)
    print("  MEANS")
    print("=" * 74)
    header = f"  {'metric':<26}" + "".join(f"{v:>13}" for v in variants)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in sorted(wide["metric"].unique()):
        sub = wide[wide["metric"] == m]
        print(f"  {m:<26}" + "".join(f"{sub[v].mean():>13.4f}" for v in variants))

    lp = wide[wide["metric"] == "lpips_alex"]
    print(f"\n  Perceptual cost: ours {lp['uap'].mean():.4f}   "
          f"CMUA {lp['cmua'].mean():.4f}")
    if lp["cmua"].mean() < lp["uap"].mean():
        print("  CMUA is perceptually cheaper. The fair comparison is at matched")
        print("  LPIPS: read the k off results_v2_curve.csv whose LPIPS is closest")
        print("  to CMUA's, re-run with --match-k <k>, and compare THAT column.")
    else:
        print("  Our perturbation is perceptually cheaper, so a like-for-like")
        print("  comparison at full strength already favours us on stealth.")

    print("\n  Paired tests vs CMUA (biometric):")
    for m in ("facenet_vggface2_cosdist", "facenet_casia_cosdist"):
        sub = wide[wide["metric"] == m]
        for ours in [c for c in variants if c.startswith("uap")]:
            a, b = sub[ours].values, sub["cmua"].values
            _, p = st.wilcoxon(a, b)
            verdict = "ours higher" if a.mean() > b.mean() else "CMUA higher"
            print(f"    {m:<26} {ours:<9} {a.mean():.4f} vs {b.mean():.4f}  "
                  f"({verdict}, p={p:.1e}, win {100*(a>b).mean():.1f}%)")


if __name__ == "__main__":
    main()
