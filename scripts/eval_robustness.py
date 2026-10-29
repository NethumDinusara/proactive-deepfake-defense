#!/usr/bin/env python
"""
eval_robustness.py -- does the protection survive normal image handling?

Reviewer 2 asked for "additional evaluations on realistic image
transformations". This measures FaceNet embedding displacement after the
protected image has been through the kind of processing a social platform
or a scraper actually applies:

    none          reference (matches evaluate.py)
    jpeg_90/70/50 JPEG re-encode at three qualities
    resize_0.5    downsample to 128 and back to 256 (bilinear)
    resize_0.25   downsample to 64 and back to 256
    blur_0.5/1.0  Gaussian blur
    bitdepth_5    quantise to 5 bits per channel

Each transform is applied to the CLEAN and PROTECTED image identically,
so what is measured is the displacement that survives the transform --
not the displacement caused by the transform itself.

Both FaceNet weight sets are evaluated. StarGAN and the diffusion U-Net
are skipped: at 1.16x and 1.57x there is no headroom for a transform to
eat into, and reporting their degradation would be noise.

Usage
-----
  python scripts/eval_robustness.py --run_folder <run> --n 250
  python scripts/stats.py --csv results/<run>/results_robust.csv
"""

import _path  # noqa: F401  -- puts src/ and third_party/ on sys.path
from uap import compat  # noqa: F401  -- stubs `aim` before torch/diffusers

import sys
import argparse
import io
import gc


import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as TF

from uap.config import load_config, get_device
from uap.perturbation import matched_noise


# ----------------------------------------------------------------------
# Transforms. Each takes and returns a (1,3,H,W) tensor in [0,1].
# ----------------------------------------------------------------------

def t_none(x):
    return x


def _jpeg(x, quality):
    """Real JPEG round-trip through PIL. Not differentiable; fine here."""
    pil = transforms.ToPILImage()(x.squeeze(0).cpu().clamp(0, 1))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = transforms.ToTensor()(Image.open(buf).convert("RGB"))
    return out.unsqueeze(0).to(x.device)


def _resize(x, factor):
    h, w = x.shape[-2:]
    small = F.interpolate(x, size=(int(h * factor), int(w * factor)),
                          mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def _blur(x, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def _bitdepth(x, bits):
    levels = 2 ** bits - 1
    return torch.round(x.clamp(0, 1) * levels) / levels


TRANSFORMS = {
    "none":        t_none,
    "jpeg_90":     lambda x: _jpeg(x, 90),
    "jpeg_70":     lambda x: _jpeg(x, 70),
    "jpeg_50":     lambda x: _jpeg(x, 50),
    "resize_0.5":  lambda x: _resize(x, 0.5),
    "resize_0.25": lambda x: _resize(x, 0.25),
    "blur_0.5":    lambda x: _blur(x, 0.5),
    "blur_1.0":    lambda x: _blur(x, 1.0),
    "bitdepth_5":  lambda x: _bitdepth(x, 5),
}


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_folder", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", default="results_robust.csv")
    args = ap.parse_args()

    from facenet_pytorch import InceptionResnetV1

    cfg = load_config()
    device = get_device()

    run_dir = os.path.join(cfg["paths"]["results_dir"], args.run_folder)
    epoch = cfg["training"]["epochs"]
    uap_path = os.path.join(run_dir, f"perturbation_epoch_{epoch}.pt")
    if not os.path.exists(uap_path):
        sys.exit(f"[ERROR] UAP not found: {uap_path}")

    v = torch.load(uap_path, map_location=device, weights_only=True)
    v.requires_grad_(False)
    epsilon = v.abs().max().item()

    test_dir = cfg["paths"]["test_data_output"]
    files = sorted(f for f in os.listdir(test_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[:args.n]

    print("=" * 74)
    print("  TRANSFORMATION ROBUSTNESS")
    print("=" * 74)
    print(f"  images     : {len(files)}")
    print(f"  epsilon    : {epsilon:.5f}")
    print(f"  transforms : {', '.join(TRANSFORMS)}")

    rows = []

    for weights in ("vggface2", "casia-webface"):
        tag = "facenet_vggface2" if weights == "vggface2" else "facenet_casia"
        print(f"\n[*] {tag}")
        net = InceptionResnetV1(pretrained=weights).eval().requires_grad_(False).to(device)

        def embed(img):
            z = F.interpolate(img, size=(160, 160), mode="bilinear", align_corners=False)
            return net((z - 0.5) / 0.5)

        with torch.no_grad():
            for i, fname in enumerate(tqdm(files, desc=f"  {tag}")):
                img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
                x = transforms.ToTensor()(img).unsqueeze(0).to(device)

                x_uap = torch.clamp(x + v, 0, 1)
                noise = matched_noise(v.shape, epsilon, i, device)
                x_rand = torch.clamp(x + noise, 0, 1)

                for tname, fn in TRANSFORMS.items():
                    # Transform is applied identically to the reference and
                    # the perturbed image, so the transform's own effect
                    # cancels and only surviving displacement is measured.
                    e_ref = embed(fn(x))
                    for vname, xp in (("uap", x_uap), ("rand", x_rand)):
                        cos = F.cosine_similarity(e_ref, embed(fn(xp))).item()
                        rows.append((fname, vname, f"{tag}_cosdist", tname, 1.0 - cos))

        del net
        gc.collect()
        torch.cuda.empty_cache()

    out_path = os.path.join(run_dir, args.out)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "variant", "metric", "transform", "value"])
        w.writerows(rows)
    print(f"\n[+] {len(rows)} records -> {out_path}")

    # ---- console summary: retention relative to the untransformed case ----
    import collections
    agg = collections.defaultdict(list)
    for image, variant, metric, tname, value in rows:
        agg[(metric, tname, variant)].append(value)

    for metric in ("facenet_vggface2_cosdist", "facenet_casia_cosdist"):
        base = np.mean(agg[(metric, "none", "uap")])
        print("\n" + "=" * 74)
        print(f"  {metric}   (untransformed UAP mean = {base:.4f})")
        print("=" * 74)
        print(f"  {'transform':<14}{'UAP':>10}{'noise':>10}{'ratio':>9}{'retained':>11}")
        print("  " + "-" * 52)
        for tname in TRANSFORMS:
            u = np.mean(agg[(metric, tname, "uap")])
            r = np.mean(agg[(metric, tname, "rand")])
            print(f"  {tname:<14}{u:>10.4f}{r:>10.4f}"
                  f"{u / r:>8.1f}x{100 * u / base:>10.1f}%")
    print("\n  'retained' = fraction of the untransformed displacement that")
    print("  survives the transform. Report this column in the paper.")


if __name__ == "__main__":
    main()
