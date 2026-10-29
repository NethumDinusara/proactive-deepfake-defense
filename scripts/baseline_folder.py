#!/usr/bin/env python
"""
baseline_folder.py -- evaluate ANY published defence under our protocol.

Works with whatever you can actually obtain. Point it at a folder of
protected images produced by someone else's method and it measures them
on exactly the same pipeline as our UAP:

    Fawkes    fawkes.exe -d <dir> --mode low
              writes <name>_cloaked.png
    LowKey    their release script, writes <name>_attacked.png
    CMUA      apply the released tensor and save, or use baseline_cmua.py
    anything  any folder of protected images with matching filenames

WHY A FOLDER AND NOT A TENSOR
-----------------------------
Per-image methods (Fawkes, LowKey) have no single perturbation to load.
Comparing at the image level is the only fair way to include them, and
it also removes any question of whether we applied their method
correctly -- their own code produced the files.

THE COMPARISON TO REPORT
------------------------
Not the raw numbers side by side. Published defences generally operate
at much lower perceptual cost than ours, so a full-strength comparison
is partly a comparison of budgets. Report instead at MATCHED perceptual
cost: find the scale k in results_v2_curve.csv whose LPIPS is closest to
the baseline's, and compare there. Pass it with --match-k.

    k=0.25 -> LPIPS 0.013
    k=0.50 -> LPIPS 0.055
    k=0.75 -> LPIPS 0.109
    k=1.00 -> LPIPS 0.162

Report the outcome whichever way it falls. A baseline you ran and lost
to at matched cost is a finding; a baseline you omitted is the gap that
got the last submission rejected.

Usage
-----
    python scripts/baseline_folder.py --run_folder <run> \
        --protected fawkes_out --name Fawkes --n 250 --match-k 0.25
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

# suffixes third-party tools append to the stem
# Decorated suffixes are tried FIRST. The bare stem is only considered when
# the protected directory is not the clean directory -- tools like Fawkes
# write their output alongside the originals, and matching the bare stem
# there pairs every image with itself.
SUFFIXES = ["_low_cloaked", "_mid_cloaked", "_high_cloaked", "_cloaked",
            "_attacked", "_protected", "_adv", "_cmua", "_perturbed"]
EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def build_index(folder):
    """stem -> path, so filename decoration does not break the pairing."""
    idx = {}
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        if ext.lower() in EXTS:
            idx[stem.lower()] = os.path.join(folder, f)
    return idx


def match(stem, idx, allow_exact):
    s = stem.lower()
    for suf in SUFFIXES:
        if s + suf in idx:
            return idx[s + suf]
    if allow_exact and s in idx:
        return idx[s]
    hits = [p for k, p in idx.items() if k.startswith(s) and k != s]
    return hits[0] if len(hits) == 1 else None


def load_as(path, size, device):
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_folder", required=True)
    ap.add_argument("--protected", required=True, help="dir of protected images")
    ap.add_argument("--name", default="baseline", help="label, e.g. Fawkes")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--match-k", type=float, default=None,
                    help="also emit our UAP at this scale (matched-LPIPS)")
    ap.add_argument("--arcface", default=None, help="path to w600k_r50.onnx")
    ap.add_argument("--exclude-unchanged", action="store_true",
                    help="drop pairs the external tool left unmodified")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import lpips as lpips_lib
    from facenet_pytorch import InceptionResnetV1

    cfg = load_config()
    device = get_device()
    tag = args.name.lower().replace(" ", "_")
    out_name = args.out or f"results_{tag}.csv"

    run_dir = os.path.join(cfg["paths"]["results_dir"], args.run_folder)
    v = torch.load(os.path.join(run_dir,
                   f"perturbation_epoch_{cfg['training']['epochs']}.pt"),
                   map_location=device, weights_only=True)
    v.requires_grad_(False)
    epsilon = v.abs().max().item()

    test_dir = cfg["paths"]["test_data_output"]
    clean_files = sorted(f for f in os.listdir(test_dir)
                         if f.lower().endswith((".png", ".jpg", ".jpeg")))

    idx = build_index(args.protected)
    same_dir = os.path.abspath(args.protected) == os.path.abspath(test_dir)
    pairs = []
    for f in clean_files:
        p = match(os.path.splitext(f)[0], idx, allow_exact=not same_dir)
        if p and os.path.abspath(p) != os.path.abspath(os.path.join(test_dir, f)):
            pairs.append((f, p))
        if len(pairs) >= args.n:
            break

    if not pairs:
        sys.exit(f"[ERROR] no filename matches between\n  {test_dir}\n  "
                 f"{args.protected}\nProtected dir holds e.g. "
                 f"{list(idx)[:3]}")

    # Check EVERY pair, not a sample. Two distinct failure modes look the
    # same on disk: the matcher resolving to the original, and the external
    # tool copying an image through unmodified when its detector fails.
    # Both yield LPIPS 0 and PSNR inf, which silently deflate the baseline.
    unchanged = []
    for cf, pf in tqdm(pairs, desc="  checking pairs"):
        a = np.asarray(Image.open(os.path.join(test_dir, cf)).convert("RGB"))
        b = np.asarray(Image.open(pf).convert("RGB"))
        if a.shape == b.shape and np.array_equal(a, b):
            unchanged.append(cf)

    if len(unchanged) == len(pairs):
        sys.exit(f"\n[ERROR] all {len(pairs)} pairs are bit-identical: the matcher "
                 f"resolved protected images to the originals.\n"
                 f"        Example: {pairs[0][0]} -> {os.path.basename(pairs[0][1])}\n")

    if unchanged:
        print(f"\n  [!] {len(unchanged)} of {len(pairs)} protected images are identical "
              f"to their source:")
        print(f"      {', '.join(unchanged[:8])}"
              f"{' ...' if len(unchanged) > 8 else ''}")
        print(f"      The external tool left these unmodified, most likely a failed")
        print(f"      face detection. They contribute zero disruption and deflate the")
        print(f"      baseline's mean.")
        if args.exclude_unchanged:
            pairs = [(c, p) for c, p in pairs if c not in set(unchanged)]
            print(f"      Excluded; {len(pairs)} pairs remain. Report this count.")
        else:
            print(f"      Retained. Pass --exclude-unchanged to drop them, and state")
            print(f"      in the paper which convention you used.")

    size = Image.open(os.path.join(test_dir, pairs[0][0])).size
    print("=" * 74)
    print(f"  {args.name.upper()} UNDER OUR PROTOCOL")
    print("=" * 74)
    print(f"  matched pairs : {len(pairs)} of {len(clean_files)} clean images")
    print(f"  reference size: {size}")
    if len(pairs) < args.n:
        print(f"  [WARN] only {len(pairs)} matched; check the naming scheme")
    print("  example pairing:")
    for cf, pf in pairs[:2]:
        print(f"    {cf}  ->  {os.path.basename(pf)}")

    def variants(x, xp_base, i):
        out = {tag: xp_base, "uap": torch.clamp(x + v, 0, 1)}
        if args.match_k:
            out[f"uap{args.match_k:g}"] = torch.clamp(x + args.match_k * v, 0, 1)
        n = matched_noise(v.shape, epsilon, i, device)
        out["rand"] = torch.clamp(x + n, 0, 1)
        return out

    rows = []

    # ---------------- perceptual ----------------
    print("\n[1/3] perceptual cost")
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    fn = lpips_lib.LPIPS(net="alex").to(device)
    with torch.no_grad():
        for i, (cf, pf) in enumerate(tqdm(pairs, desc="  lpips")):
            x = load_as(os.path.join(test_dir, cf), size, device)
            xb = load_as(pf, size, device)
            ref = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
            for name, xp in variants(x, xb, i).items():
                rows.append((cf, name, "lpips_alex",
                             fn(x * 2 - 1, xp * 2 - 1).item()))
                cand = xp.squeeze(0).permute(1, 2, 0).cpu().numpy()
                rows.append((cf, name, "psnr", sk_psnr(ref, cand, data_range=1.0)))
                rows.append((cf, name, "ssim",
                             sk_ssim(ref, cand, data_range=1.0, channel_axis=2,
                                     win_size=11, gaussian_weights=True)))
    del fn; gc.collect(); torch.cuda.empty_cache()

    base_lpips = np.mean([v for _, n, m, v in rows
                          if n == tag and m == "lpips_alex"])
    if base_lpips < 1e-6:
        sys.exit(
            f"\n[ERROR] {args.name} LPIPS is {base_lpips:.2e} -- the protected\n"
            f"        images are identical to the clean ones, so the pairing is\n"
            f"        wrong. Check the 'example pairing' lines above: each clean\n"
            f"        file must map to a DECORATED filename, not to itself.\n")
    print(f"  {args.name} mean LPIPS = {base_lpips:.4f}")

    # ---------------- FaceNet, both weight sets ----------------
    for j, weights in enumerate(("vggface2", "casia-webface"), start=2):
        mtag = "facenet_vggface2" if weights == "vggface2" else "facenet_casia"
        print(f"\n[{j}/3] {mtag}")
        net = InceptionResnetV1(pretrained=weights).eval().requires_grad_(False).to(device)

        def embed(img):
            z = F.interpolate(img, size=(160, 160), mode="bilinear", align_corners=False)
            return net((z - 0.5) / 0.5)

        with torch.no_grad():
            for i, (cf, pf) in enumerate(tqdm(pairs, desc=f"  {mtag}")):
                x = load_as(os.path.join(test_dir, cf), size, device)
                xb = load_as(pf, size, device)
                e_ref = embed(x)
                for name, xp in variants(x, xb, i).items():
                    rows.append((cf, name, f"{mtag}_cosdist",
                                 1.0 - F.cosine_similarity(e_ref, embed(xp)).item()))
        del net; gc.collect(); torch.cuda.empty_cache()

    # ---------------- ArcFace, optional ----------------
    if args.arcface:
        print("\n[+] ArcFace (held-out architecture)")
        import onnxruntime as ort
        from eval_arcface import preprocess, cosdist
        so = ort.SessionOptions(); so.log_severity_level = 3
        sess = ort.InferenceSession(args.arcface, so,
                                    providers=["CPUExecutionProvider"])
        i_n, o_n = sess.get_inputs()[0].name, sess.get_outputs()[0].name
        for i, (cf, pf) in enumerate(tqdm(pairs, desc="  arcface")):
            x = load_as(os.path.join(test_dir, cf), size, device)
            xb = load_as(pf, size, device)
            vs = variants(x, xb, i)
            batch = np.stack([preprocess(x)] + [preprocess(t) for t in vs.values()])
            feats = sess.run([o_n], {i_n: batch})[0]
            for name, f in zip(vs.keys(), feats[1:]):
                rows.append((cf, name, "arcface_buffalo_l_cosdist",
                             cosdist(feats[0], f)))

    out_path = os.path.join(run_dir, out_name)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "variant", "metric", "value"])
        w.writerows(rows)
    print(f"\n[+] {len(rows)} records -> {out_path}")

    # ---------------- summary ----------------
    import pandas as pd
    from scipy import stats as st

    df = pd.DataFrame(rows, columns=["image", "variant", "metric", "value"])
    wide = df.pivot_table(index=["image", "metric"], columns="variant",
                          values="value").reset_index()
    cols = [c for c in wide.columns if c not in ("image", "metric")]

    print("\n" + "=" * 74)
    print(f"  MEANS  (n = {len(pairs)})")
    print("=" * 74)
    head = f"  {'metric':<26}" + "".join(f"{c:>14}" for c in cols)
    print(head); print("  " + "-" * (len(head) - 2))
    for m in sorted(wide["metric"].unique()):
        s = wide[wide["metric"] == m]
        print(f"  {m:<26}" + "".join(f"{s[c].mean():>14.4f}" for c in cols))

    lp = wide[wide["metric"] == "lpips_alex"]
    b, o = lp[tag].mean(), lp["uap"].mean()
    print(f"\n  Perceptual cost: {args.name} {b:.4f}   ours (k=1) {o:.4f}")
    if b < o and not args.match_k:
        best = min([(0.25, 0.013), (0.5, 0.055), (0.75, 0.109), (1.0, 0.162)],
                   key=lambda t: abs(t[1] - b))
        print(f"  {args.name} is perceptually cheaper. Re-run with "
              f"--match-k {best[0]:g} (LPIPS {best[1]:.3f}) and report THAT.")

    print(f"\n  Paired tests vs {args.name}:")
    for m in sorted(x for x in wide["metric"].unique() if "cosdist" in x):
        s = wide[wide["metric"] == m]
        for ours in [c for c in cols if c.startswith("uap")]:
            a, bb = s[ours].values, s[tag].values
            _, p = st.wilcoxon(a, bb)
            who = "ours" if a.mean() > bb.mean() else args.name
            print(f"    {m:<26} {ours:<9} {a.mean():.4f} vs {bb.mean():.4f}  "
                  f"({who} higher, p={p:.1e}, win {100*(a>bb).mean():.1f}%)")


if __name__ == "__main__":
    main()