#!/usr/bin/env python
"""
eval_arcface.py -- does the perturbation transfer to a different ARCHITECTURE?

Both FaceNet weight sets evaluated so far (VGGFace2, CASIA-Webface) are
InceptionResNetV1: they differ in training data, not structure. A reviewer
will read "held-out weights" correctly as one architecture, two checkpoints.

ArcFace is a genuinely different backbone (IResNet), trained with an
angular-margin loss on a different corpus. Nothing about it was seen
during optimisation.

    buffalo_l    w600k_r50.onnx    IResNet-50,  WebFace600K
    antelopev2   glintr100.onnx    IResNet-100, Glint360K

NO insightface DEPENDENCY. That package pins numpy>=2, which breaks
facenet-pytorch, lpips and scikit-image and would silently invalidate
every number already in the paper. All it provided was a file download
and an image normalisation, both reproduced here in a few lines. The
ONNX model is called directly through onnxruntime.

Preprocessing matches insightface's ArcFaceONNX.get_feat exactly:
112x112, RGB channel order, (x - 127.5) / 127.5, NCHW float32. The
embedding is L2-normalised here because the raw ONNX output is not.

Getting the model
-----------------
Download buffalo_l.zip from the insightface releases page

    https://github.com/deepinsight/insightface/releases

and unzip so that w600k_r50.onnx sits in one of:

    C:\\Users\\<you>\\.insightface\\models\\buffalo_l\\
    ./models/buffalo_l/

or pass --model path\\to\\w600k_r50.onnx directly.

Alignment note: insightface expects its own 5-point 112x112 template;
our LFW images are MTCNN-aligned at 256x256, so we resize rather than
re-align. Applied identically to clean and perturbed, so the paired
comparison holds -- it measures displacement under a fixed crop.
Re-aligning per image would let the detector land differently on the two
inputs and confound the measurement. Disclose this in the paper.

Usage
-----
    python scripts/eval_arcface.py --run_folder <run> --n 1000
    python scripts/eval_arcface.py --run_folder <run> --n 250 \
        --scales 0.25,0.5,0.75,1.0 --out results_arcface_curve.csv
"""

import _path  # noqa: F401  -- puts src/ and third_party/ on sys.path

import sys
import os
import csv
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from uap.config import load_config, get_device
from uap.perturbation import matched_noise

PACKS = {
    "buffalo_l":  ("w600k_r50.onnx", "ArcFace IResNet-50 / WebFace600K"),
    "antelopev2": ("glintr100.onnx", "ArcFace IResNet-100 / Glint360K"),
}


def find_model(pack, explicit):
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"[ERROR] not found: {explicit}")
        return explicit

    fname, _ = PACKS[pack]
    for root in (os.path.join(os.path.expanduser("~"), ".insightface", "models", pack),
                 os.path.join("models", pack),
                 os.path.join("models", "arcface"),
                 pack, "."):
        cand = os.path.join(root, fname)
        if os.path.exists(cand):
            return cand

    sys.exit(
        f"\n[ERROR] {fname} not found.\n\n"
        f"  1. Download {pack}.zip from\n"
        f"       https://github.com/deepinsight/insightface/releases\n"
        f"  2. Unzip so that {fname} sits in\n"
        f"       {os.path.join(os.path.expanduser('~'), '.insightface', 'models', pack)}\n"
        f"  3. Or pass --model path\\to\\{fname}\n")


def preprocess(x):
    """(1,3,256,256) float [0,1] -> (3,112,112) float32, ArcFace convention."""
    arr = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)      # 8-bit, as a file
    arr = cv2.resize(arr, (112, 112), interpolation=cv2.INTER_LINEAR)
    arr = (arr.astype(np.float32) - 127.5) / 127.5             # already RGB
    return arr.transpose(2, 0, 1)


def cosdist(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_folder", required=True)
    ap.add_argument("--pack", choices=list(PACKS), default="buffalo_l")
    ap.add_argument("--model", default=None, help="explicit path to the .onnx")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--scales", default="1.0")
    ap.add_argument("--out", default="results_arcface.csv")
    args = ap.parse_args()

    import onnxruntime as ort

    cfg = load_config()
    device = get_device()
    scales = [float(s) for s in args.scales.split(",")]

    run_dir = os.path.join(cfg["paths"]["results_dir"], args.run_folder)
    uap_path = os.path.join(run_dir, f"perturbation_epoch_{cfg['training']['epochs']}.pt")
    if not os.path.exists(uap_path):
        sys.exit(f"[ERROR] UAP not found: {uap_path}")
    v = torch.load(uap_path, map_location=device, weights_only=True)
    v.requires_grad_(False)
    epsilon = v.abs().max().item()

    model_path = find_model(args.pack, args.model)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    test_dir = cfg["paths"]["test_data_output"]
    files = sorted(f for f in os.listdir(test_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[:args.n]

    print("=" * 74)
    print("  HELD-OUT ARCHITECTURE EVALUATION")
    print("=" * 74)
    print(f"  model   : {PACKS[args.pack][1]}")
    print(f"            {model_path}")
    print(f"  input   : {in_name} {sess.get_inputs()[0].shape}")
    print(f"  images  : {len(files)}   epsilon {epsilon:.5f}   scales {scales}")

    metric = f"arcface_{args.pack}_cosdist"
    rows = []

    for i, fname in enumerate(tqdm(files, desc="  arcface")):
        img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0)
        x = x.unsqueeze(0).to(device)

        variants = {}
        for k in scales:
            name = "uap" if abs(k - 1.0) < 1e-9 else f"uap{k:g}"
            variants[name] = torch.clamp(x + k * v, 0, 1)

        # identical per-image seed to evaluate.py, so baselines line up
        noise = matched_noise(v.shape, epsilon, i, device)
        variants["rand"] = torch.clamp(x + noise, 0, 1)

        batch = np.stack([preprocess(x)] + [preprocess(t) for t in variants.values()])
        feats = sess.run([out_name], {in_name: batch})[0]

        ref, rest = feats[0], feats[1:]
        for name, f in zip(variants.keys(), rest):
            rows.append((fname, name, metric, cosdist(ref, f)))

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
    for image, variant, _, value in rows:
        agg[image][variant] = value
    imgs = sorted(agg)
    rand = np.array([agg[i]["rand"] for i in imgs])

    print("\n" + "=" * 74)
    print(f"  {metric}   (n = {len(imgs)})")
    print("=" * 74)
    print(f"  {'variant':<10}{'mean':>10}{'sd':>9}{'ratio':>9}{'win%':>8}{'p':>12}")
    print("  " + "-" * 58)
    print(f"  {'rand':<10}{rand.mean():>10.4f}{rand.std():>9.4f}"
          f"{1.0:>8.1f}x{'--':>8}{'--':>12}")

    order = sorted((k for k in agg[imgs[0]] if k != "rand"),
                   key=lambda s: 1.0 if s == "uap" else float(s[3:]))
    for name in order:
        a = np.array([agg[i][name] for i in imgs])
        _, p = st.wilcoxon(a, rand)
        print(f"  {name:<10}{a.mean():>10.4f}{a.std():>9.4f}"
              f"{a.mean() / rand.mean():>8.1f}x{100 * (a > rand).mean():>7.1f}%"
              f"{p:>12.1e}")

    tau = np.percentile(rand, 99)
    full = np.array([agg[i]["uap"] for i in imgs])
    print(f"\n  tau_99 (99th pct of noise) = {tau:.5f}")
    print(f"  success at tau_99:  noise 1.0%   UAP {100 * (full > tau).mean():.1f}%")
    print("\n  For comparison (Table I, n=1000):")
    print("    FaceNet / VGGFace2  0.2032   78.6x   surrogate")
    print("    FaceNet / CASIA     0.2682   34.6x   held-out weights, same arch")
    print("\n  A multiplier well above 1 means the perturbation transfers across")
    print("  ARCHITECTURES, not merely checkpoints. Near 1.0 means it is bound to")
    print("  the InceptionResNet family -- report that plainly; a specific")
    print("  limitation is worth more to a reader than silence.")


if __name__ == "__main__":
    main()