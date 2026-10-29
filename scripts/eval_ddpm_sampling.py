#!/usr/bin/env python
"""
eval_ddpm_sampling.py -- does the noise-prediction disruption actually change
what the model generates?

Reviewer 2 asked for exactly this, as an alternative to multi-seed runs:

    "include visual examples of generated images (clean vs. perturbed
     input) and measure FID or LPIPS on outputs to demonstrate whether
     the noise-prediction disruption actually degrades generated image
     quality"

Reviewer 3 asked for the same thing in different words ("more direct
evidence of generative-model disruption"). One experiment answers both.

It also settles the question raised by the `t_start = 150` comment in
the earlier codebase: the effect was known to depend on where denoising
begins, but that dependence was never measured.

WHAT IT DOES
------------
For each image and each starting timestep t_start:

  1. noise the clean image and the protected image to t_start using the
     SAME noise sample, so the comparison is paired and any difference
     comes from the perturbation rather than the draw
  2. run the reverse diffusion from t_start down to 0 for each
  3. measure LPIPS between the two generated images

A large LPIPS between generated-from-clean and generated-from-protected
means the perturbation changed the output. Small means it did not, and
the protection is confined to the noise-prediction statistic already
reported in Table I.

LPIPS rather than FID: FID needs thousands of samples to be stable, and
this runs on tens. Reviewer 2 allowed either.

COST
----
Generation is t_start U-Net forwards. With the default sweep
(50+100+200+400 = 750 steps) and three conditions, budget roughly two
hours for 20 images on a 4 GB card. Reduce --n first if that is too
long; the qualitative panel needs only a handful of images.

Usage
-----
  python scripts/eval_ddpm_sampling.py --run_folder <run> --n 20
  python scripts/eval_ddpm_sampling.py --run_folder <run> --n 8 --t-starts 100,200
"""

import _path  # noqa: F401  -- puts src/ and third_party/ on sys.path
from uap import compat  # noqa: F401  -- stubs `aim` before torch/diffusers

import os
import csv
import argparse


import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from uap.config import load_config, get_device
from uap.perturbation import matched_noise


def to_pil(x):
    a = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
    return Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))


@torch.no_grad()
def generate(unet, sched, x01, t_start, noise, device):
    """Noise x to t_start with the GIVEN noise, then denoise to 0."""
    x = x01 * 2 - 1                                   # model expects [-1,1]
    t = torch.tensor([t_start], device=device).long()
    xt = sched.add_noise(x, noise, t)

    steps = [s for s in sched.timesteps if s <= t_start]
    for s in steps:
        eps = unet(xt, s).sample
        xt = sched.step(eps, s, xt).prev_sample

    return ((xt / 2) + 0.5).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_folder", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--t-starts", default="50,100,200,400")
    ap.add_argument("--model", default="google/ddpm-celebahq-256")
    ap.add_argument("--panel-images", type=int, default=4)
    ap.add_argument("--out", default="results_ddpm_sampling.csv")
    args = ap.parse_args()

    import lpips as lpips_lib
    from diffusers import DDPMPipeline

    cfg = load_config()
    device = get_device()
    t_starts = [int(t) for t in args.t_starts.split(",")]

    run_dir = os.path.join(cfg["paths"]["results_dir"], args.run_folder)
    v = torch.load(os.path.join(run_dir,
                   f"perturbation_epoch_{cfg['training']['epochs']}.pt"),
                   map_location=device, weights_only=True)
    v.requires_grad_(False)
    epsilon = v.abs().max().item()

    test_dir = cfg["paths"]["test_data_output"]
    files = sorted(f for f in os.listdir(test_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[:args.n]

    total = len(files) * len(t_starts) * sum(t_starts) / sum(t_starts) * 3
    print("=" * 74)
    print("  DDPM SAMPLING: DOES THE OUTPUT ACTUALLY CHANGE?")
    print("=" * 74)
    print(f"  images    : {len(files)}")
    print(f"  t_starts  : {t_starts}")
    print(f"  epsilon   : {epsilon:.5f}")
    print(f"  forwards  : ~{len(files) * sum(t_starts) * 3:,} U-Net passes")
    print()

    pipe = DDPMPipeline.from_pretrained(args.model)
    unet = pipe.unet.eval().requires_grad_(False).to(device)
    sched = pipe.scheduler
    sched.set_timesteps(1000)

    lp = lpips_lib.LPIPS(net="alex").to(device)
    panel_dir = os.path.join(run_dir, "sampling_panels")
    os.makedirs(panel_dir, exist_ok=True)

    rows = []
    for i, fname in enumerate(tqdm(files, desc="  images")):
        x = transforms.ToTensor()(
            Image.open(os.path.join(test_dir, fname)).convert("RGB")
        ).unsqueeze(0).to(device)

        x_uap = torch.clamp(x + v, 0, 1)
        rnd = matched_noise(v.shape, epsilon, i, device)
        x_rnd = torch.clamp(x + rnd, 0, 1)

        for t_start in t_starts:
            # one noise draw shared by all three conditions: any
            # difference in the output is caused by the perturbation
            gn = torch.Generator(device="cpu").manual_seed(50000 + i)
            noise = torch.randn(x.shape, generator=gn).to(device)

            gen_clean = generate(unet, sched, x,     t_start, noise, device)
            gen_uap   = generate(unet, sched, x_uap, t_start, noise, device)
            gen_rnd   = generate(unet, sched, x_rnd, t_start, noise, device)

            def d(a, b):
                return lp(a * 2 - 1, b * 2 - 1).item()

            for name, gen in (("uap", gen_uap), ("rand", gen_rnd)):
                rows.append((fname, name, f"lpips_vs_genclean_t{t_start}",
                             d(gen_clean, gen)))
                rows.append((fname, name, f"lpips_vs_original_t{t_start}",
                             d(x, gen)))
            rows.append((fname, "clean", f"lpips_vs_original_t{t_start}",
                         d(x, gen_clean)))

            if i < args.panel_images:
                strip = Image.new("RGB", (256 * 4, 256))
                for k, img in enumerate([to_pil(x), to_pil(x_uap),
                                         to_pil(gen_clean), to_pil(gen_uap)]):
                    strip.paste(img, (256 * k, 0))
                strip.save(os.path.join(
                    panel_dir, f"panel_{os.path.splitext(fname)[0]}_t{t_start}.png"))

    out_path = os.path.join(run_dir, args.out)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "variant", "metric", "value"])
        w.writerows(rows)
    print(f"\n[+] {len(rows)} records -> {out_path}")
    print(f"[+] panels -> {panel_dir}")
    print("    column order: original | protected | generated from clean | "
          "generated from protected")

    # ---------------- summary ----------------
    import collections
    from scipy import stats as st

    agg = collections.defaultdict(dict)
    for image, variant, metric, value in rows:
        agg[(image, metric)][variant] = value

    print("\n" + "=" * 74)
    print("  LPIPS between generated-from-clean and generated-from-perturbed")
    print("=" * 74)
    print(f"  {'t_start':>8}{'noise':>10}{'UAP':>10}{'ratio':>9}"
          f"{'win%':>8}{'p':>11}")
    print("  " + "-" * 56)
    for t in t_starts:
        m = f"lpips_vs_genclean_t{t}"
        keys = [k for k in agg if k[1] == m]
        a = np.array([agg[k]["uap"] for k in keys])
        b = np.array([agg[k]["rand"] for k in keys])
        _, p = st.wilcoxon(a, b) if len(a) > 5 else (0, float("nan"))
        print(f"  {t:>8}{b.mean():>10.4f}{a.mean():>10.4f}"
              f"{a.mean()/b.mean():>8.2f}x{100*(a>b).mean():>7.1f}%{p:>11.1e}")

    print("\n  How to read this. The column is how far the generated image")
    print("  moves when the input is perturbed. If the UAP column is well")
    print("  above the noise column, the perturbation changes what the model")
    print("  produces and not merely its internal noise estimate. If the two")
    print("  are close, say so: the protection is confined to the statistic")
    print("  already reported in Table I, which is a narrower claim than the")
    print("  current Limitations section concedes.")
    print()
    print("  Expect the effect to shrink as t_start rises, since more of the")
    print("  perturbation is destroyed before denoising begins. That curve is")
    print("  the finding, whichever way it falls.")


if __name__ == "__main__":
    main()