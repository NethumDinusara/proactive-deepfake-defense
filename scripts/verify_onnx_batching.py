#!/usr/bin/env python
"""
verify_onnx_batching.py -- is the ONNX batch-shape warning benign?

eval_arcface.py feeds 3 images per call (clean, uap, rand). The model
declares a fixed output batch of 1, so onnxruntime warns:

    Expected shape from model of {1,512} does not match actual shape
    of {3,512} for output 683

The ops involved are batch-independent, so this should be metadata only.
"Should be" is not adequate for a number going into a table. This script
computes every embedding twice -- once batched, once one image at a time
-- and reports the largest disagreement.

    max |cos_batched - cos_single|  <  1e-5   -> batching is safe
    anything larger                           -> stop, do not use batching

Usage
-----
    python scripts/verify_onnx_batching.py --run_folder <run>
"""

import _path  # noqa: F401  -- puts src/ and third_party/ on sys.path

import os
import argparse

import numpy as np
import torch
import onnxruntime as ort
from PIL import Image

from uap.config import load_config, get_device
from uap.perturbation import matched_noise
from eval_arcface import preprocess, cosdist, find_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_folder", required=True)
    ap.add_argument("--pack", default="buffalo_l")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config()
    device = get_device()

    run_dir = os.path.join(cfg["paths"]["results_dir"], args.run_folder)
    v = torch.load(
        os.path.join(run_dir, f"perturbation_edge.pt")
        if False else
        os.path.join(run_dir, f"perturbation_epoch_{cfg['training']['epochs']}.pt"),
        map_location=device, weights_only=True)
    v.requires_grad_(False)
    epsilon = v.abs().max().item()

    so = ort.SessionOptions()
    so.log_severity_level = 3                     # silence the shape warning
    sess = ort.InferenceSession(find_model(args.pack, args.model), so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    test_dir = cfg["paths"]["test_data_output"]
    files = sorted(f for f in os.listdir(test_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[:args.n]

    print(f"[*] comparing batched vs single-image inference on {len(files)} images")

    worst = 0.0
    for i, fname in enumerate(files):
        img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0)
        x = x.unsqueeze(0).to(device)

        x_uap = torch.clamp(x + v, 0, 1)
        noise = matched_noise(v.shape, epsilon, i, device)
        x_rand = torch.clamp(x + noise, 0, 1)

        tensors = [x, x_uap, x_rand]
        pre = [preprocess(t) for t in tensors]

        batched = sess.run([out_name], {in_name: np.stack(pre)})[0]
        single = np.stack([sess.run([out_name], {in_name: p[None]})[0][0]
                           for p in pre])

        for j in (1, 2):
            worst = max(worst,
                        abs(cosdist(batched[0], batched[j])
                            - cosdist(single[0], single[j])))

    print(f"\n  max |batched - single| cosine distance = {worst:.3e}")
    if worst < 1e-5:
        print("  -> batching is safe. The warning is metadata only.")
        print("     Add these two lines to eval_arcface.py to silence it:")
        print("       so = ort.SessionOptions(); so.log_severity_level = 3")
        print("       sess = ort.InferenceSession(model_path, so, providers=[...])")
    else:
        print("  -> DISAGREEMENT. Do not use batched inference; rerun one at a time.")


if __name__ == "__main__":
    main()
