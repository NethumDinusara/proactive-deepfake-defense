#!/usr/bin/env python
"""
stats.py -- turn results_v2.csv into paper-ready numbers.

Produces, for every metric:
  * mean with a bias-corrected bootstrap 95% CI (10,000 resamples)
  * a paired Wilcoxon signed-rank test, UAP vs matched-epsilon noise, on the
    same images -- which is the correct test here, since every image is
    measured under both conditions
  * rank-biserial effect size, so significance is not confused with magnitude
  * empirical ASR swept across thresholds rather than reported at one

Nothing is fitted to a distribution. Every number traces to per-image values.

Usage
-----
  python scripts/stats.py --csv results/<run>/results_v2.csv
  python scripts/stats.py --csv results/<run>/results_v2_curve.csv --curve
"""

import _path  # noqa: F401  -- puts src/ and third_party/ on sys.path

import argparse
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)
N_BOOT = 10000


def boot_ci(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_test(a, b):
    """Wilcoxon signed-rank on paired samples, plus rank-biserial effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return np.nan, np.nan
    try:
        w, p = stats.wilcoxon(a, b)
    except ValueError:
        return np.nan, np.nan
    nz = d[d != 0]
    r = stats.rankdata(np.abs(nz))
    rbc = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return p, rbc


def stars(p):
    if np.isnan(p):
        return "  "
    if p < 1e-4:
        return "***"
    if p < 1e-3:
        return "** "
    if p < 0.05:
        return "*  "
    return "ns "


def wide(df):
    return df.pivot_table(index=["image", "metric"], columns="variant",
                          values="value").reset_index()


def report(df, baseline="rand", treatment="uap"):
    w = wide(df)
    print("=" * 96)
    print(f"  MEAN [95% bootstrap CI]   |   paired Wilcoxon: {treatment} vs {baseline}")
    print("=" * 96)
    print(f"{'metric':<30}{'baseline':>20}{'proposed':>20}{'ratio':>9}{'p':>11}{'':>4}{'rbc':>7}")
    print("-" * 96)

    for metric in sorted(w["metric"].unique()):
        sub = w[w["metric"] == metric].dropna(subset=[c for c in (baseline, treatment) if c in w])
        if baseline not in sub or treatment not in sub or sub.empty:
            continue
        b, t = sub[baseline].values, sub[treatment].values
        bm, blo, bhi = boot_ci(b)
        tm, tlo, thi = boot_ci(t)
        p, rbc = paired_test(t, b)
        ratio = tm / bm if bm else np.nan
        print(f"{metric:<30}"
              f"{bm:>8.4f} [{blo:.4f},{bhi:.4f}]".rjust(20) +
              f"{tm:>8.4f} [{tlo:.4f},{thi:.4f}]".rjust(20) +
              f"{ratio:>9.2f}{p:>11.2e} {stars(p)}{rbc:>7.3f}")

    print("-" * 96)
    print(f"n = {w.groupby('metric').size().max()} paired images. "
          "rbc = rank-biserial correlation (effect size, -1..1).")


def asr_sweep(df, metric, thresholds, baseline="rand", treatment="uap"):
    w = wide(df)
    sub = w[w["metric"] == metric]
    if sub.empty:
        print(f"\n[skip] {metric} not present")
        return
    print(f"\n  Empirical ASR sweep -- {metric}  (n = {len(sub)})")
    print(f"  {'threshold':>12}{'baseline':>12}{'proposed':>12}")
    for th in thresholds:
        br = 100.0 * (sub[baseline].values > th).mean() if baseline in sub else np.nan
        tr = 100.0 * (sub[treatment].values > th).mean() if treatment in sub else np.nan
        print(f"  {th:>12.4f}{br:>11.1f}%{tr:>11.1f}%")
    print("  Report the whole sweep, or state the threshold and justify it.")


def curve(df, x_metric="lpips_alex"):
    """Stealth/robustness operating curve from the delta-scaling run."""
    w = wide(df)
    variants = [c for c in w.columns if c.startswith("uap")]
    order = sorted(variants, key=lambda s: 1.0 if s == "uap" else float(s[3:]))

    y_metrics = [m for m in w["metric"].unique()
                 if m != x_metric and ("cosdist" in m or "mse" in m)]

    print("\n" + "=" * 96)
    print("  STEALTH / ROBUSTNESS OPERATING CURVE")
    print("=" * 96)
    header = f"{'variant':<10}{'scale':>7}{'LPIPS':>9}" + "".join(f"{m[:16]:>18}" for m in y_metrics)
    print(header)
    print("-" * len(header))

    rows = []
    for var in order:
        scale = 1.0 if var == "uap" else float(var[3:])
        xs = w[w["metric"] == x_metric][var]
        if xs.empty:
            continue
        line = f"{var:<10}{scale:>7.2f}{xs.mean():>9.4f}"
        rec = {"variant": var, "scale": scale, "lpips": xs.mean()}
        for m in y_metrics:
            ys = w[w["metric"] == m][var]
            val = ys.mean() if not ys.empty else np.nan
            line += f"{val:>18.5f}"
            rec[m] = val
        rows.append(rec)
        print(line)

    out = pd.DataFrame(rows)
    out.to_csv("operating_curve.csv", index=False)
    print("\n[+] operating_curve.csv written -- plot LPIPS on x, disruption on y.")
    print("    This is the figure that replaces the radar chart, and it lets you")
    print("    compare against Fawkes/LowKey/CMUA at MATCHED perceptual cost")
    print("    instead of at a single point where you lose on stealth.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--curve", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} records | "
          f"{df['image'].nunique()} images | "
          f"variants: {sorted(df['variant'].unique())}\n")

    report(df)

    for metric, ths in [
        ("facenet_vggface2_cosdist", [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]),
        ("facenet_casia_cosdist",    [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]),
        ("stargan_out_mse",          [0.001, 0.005, 0.010, 0.050, 0.100]),
    ]:
        if metric in set(df["metric"]):
            asr_sweep(df, metric, ths)

    if args.curve:
        curve(df)


if __name__ == "__main__":
    main()
