#!/usr/bin/env python
"""
plot_operating_curve.py -- the stealth/robustness operating curve.

Replaces the deleted radar chart (old Fig. 9).

DEFAULT: one IEEE-column figure showing disruption as a multiplier over
the matched-epsilon noise baseline. That is the whole argument --
biometric curves stay above parity at every scale, generative curves
cross it only near full strength -- and absolute values already appear
in Tables I and III, so a second panel would repeat them.

--panels 2 adds an absolute panel if you have the width. Note it must
pool cosine distance with MSE on one log axis, so only within-series
comparisons are meaningful there. That caveat is why one panel is the
default.

The noise baseline sits at 1.0x by construction but at LPIPS 0.345,
roughly twice the full-strength cost. Plotting it in-axes would leave
half the figure empty, so it is reported as an off-scale annotation.

Usage
-----
  python scripts/plot_operating_curve.py --csv results/<run>/results_v2_curve.csv \
                          results/<run>/results_arcface_curve.csv
"""

import _path  # noqa: F401  -- puts src/ and third_party/ on sys.path

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [
    ("facenet_vggface2_cosdist", "VGGFace2 (surrogate)", "#e67e22", "s", "-"),
    ("facenet_casia_cosdist",    "CASIA (held-out weights)", "#c0392b", "o", "-"),
    ("arcface_buffalo_l_cosdist", "ArcFace R50 (held-out arch.)",   "#8e44ad", "v", "-"),
    ("ddpm_mse_t500",            "DDPM U-Net",                     "#2980b9", "^", "--"),
    ("stargan_out_mse",          "StarGAN",                        "#7f8c8d", "D", ":"),
]
TOP = "facenet_vggface2_cosdist"      # curve the k labels sit above


def load(csv_paths):
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    w = df.pivot_table(index=["image", "metric"], columns="variant",
                       values="value").reset_index()
    variants = [c for c in w.columns if c.startswith("uap")]
    order = sorted(variants, key=lambda s: 1.0 if s == "uap" else float(s[3:]))
    scales = [1.0 if s == "uap" else float(s[3:]) for s in order]

    lp = w[w["metric"] == "lpips_alex"]
    lpips = [lp[c].mean() for c in order]
    lpips_noise = lp["rand"].mean()

    data = {}
    for metric, *_ in SERIES:
        sub = w[w["metric"] == metric]
        if not sub.empty:
            data[metric] = {"y": [sub[c].mean() for c in order],
                            "noise": sub["rand"].mean()}
    return scales, lpips, lpips_noise, data


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7, "figure.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def k_labels(ax, lpips, scales, ys):
    for x, k, y in zip(lpips, scales, ys):
        ax.annotate(f"{k:g}", xy=(x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=6.5, color="#8a8a8a")
    ax.text(0.015, 0.975, "scale factor $k$", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, color="#8a8a8a",
            style="italic")


def draw_ratio(ax, lpips, lpips_noise, data, scales, label_k=True):
    for metric, label, colour, marker, ls in SERIES:
        if metric not in data:
            continue
        ratio = np.array(data[metric]["y"]) / data[metric]["noise"]
        ax.plot(lpips, ratio, marker=marker, linestyle=ls, color=colour,
                label=label, markersize=4, linewidth=1.4,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)

    ax.set_yscale("log")
    ax.set_xlim(0, max(lpips) * 1.20)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.9)                       # room for the k labels
    ax.axhspan(lo, 1.0, color="#000000", alpha=0.05, zorder=0, lw=0)
    ax.axhline(1.0, color="#333333", linewidth=0.9, zorder=2)

    ax.annotate(f"matched-$\\epsilon$ noise = 1.0$\\times$\nat LPIPS {lpips_noise:.2f}"
                " (off scale)",
                xy=(0.97, 0.055), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=6.5, color="#666666",
                bbox=dict(boxstyle="round,pad=0.32", fc="white",
                          ec="#cccccc", lw=0.5))

    if label_k:
        stacked = np.vstack([np.array(data[m]["y"]) / data[m]["noise"]
                             for m, *_ in SERIES if m in data])
        k_labels(ax, lpips, scales, list(stacked.max(axis=0)))

    ax.set_xticks([0.00, 0.05, 0.10, 0.15])
    ax.set_xlabel("Perceptual cost (LPIPS)")
    ax.set_ylabel("Disruption relative to\nmatched-$\\epsilon$ noise")


def draw_absolute(ax, lpips, lpips_noise, data, scales):
    for metric, label, colour, marker, ls in SERIES:
        if metric not in data:
            continue
        ax.plot(lpips, data[metric]["y"], marker=marker, linestyle=ls,
                color=colour, label=label, markersize=4, linewidth=1.4,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(0, max(lpips) * 1.20)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.9)
    stacked = np.vstack([data[m]["y"] for m, *_ in SERIES if m in data])
    k_labels(ax, lpips, scales, list(stacked.max(axis=0)))
    ax.set_xticks([0.00, 0.05, 0.10, 0.15])
    ax.set_xlabel("Perceptual cost (LPIPS)")
    ax.set_ylabel("Cosine distance / MSE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, nargs="+")
    ap.add_argument("--panels", type=int, choices=(1, 2), default=1)
    ap.add_argument("--out", default="fig_operating_curve.pdf")
    args = ap.parse_args()

    scales, lpips, lpips_noise, data = load(args.csv)
    style()

    n_series = sum(1 for m, *_ in SERIES if m in data)

    if args.panels == 1:
        ncol = 2
        rows = -(-n_series // ncol)                      # ceil
        bottom = 0.135 + 0.062 * rows
        fig, ax = plt.subplots(figsize=(3.45, 2.75 + 0.22 * rows))
        draw_ratio(ax, lpips, lpips_noise, data, scales)
        first = ax
        fig.subplots_adjust(left=0.215, right=0.975, top=0.965, bottom=bottom)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.05))
        draw_absolute(axs[0], lpips, lpips_noise, data, scales)
        draw_ratio(axs[1], lpips, lpips_noise, data, scales, label_k=False)
        axs[0].set_title("(a) Absolute", fontsize=9, fontweight="bold")
        axs[1].set_title("(b) Relative to noise", fontsize=9, fontweight="bold")
        ncol = n_series if n_series <= 5 else 3
        rows = -(-n_series // ncol)
        first = axs[0]
        fig.subplots_adjust(left=0.075, right=0.988, top=0.905,
                            bottom=0.155 + 0.070 * rows, wspace=0.27)

    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=ncol, frameon=False, handlelength=2.0,
               columnspacing=1.3, handletextpad=0.5)

    plt.savefig(args.out)
    plt.savefig(args.out.replace(".pdf", ".png"), dpi=400)
    print(f"[+] {args.out} (and .png)  --  {args.panels} panel(s)")
    print("\nCaption:")
    print("  Stealth-robustness operating curve, obtained by scaling the optimised")
    print("  perturbation by k (250 held-out LFW images). Disruption is normalised")
    print("  by the matched-epsilon random-noise baseline; the shaded region marks")
    print("  where the perturbation gives no benefit over that baseline. Biometric")
    print("  disruption exceeds it at every scale tested, including k=0.25 where")
    print("  perceptual cost is an order of magnitude lower; generative disruption")
    print("  crosses parity only near full strength, and StarGAN never")
    print("  meaningfully does.")


if __name__ == "__main__":
    main()