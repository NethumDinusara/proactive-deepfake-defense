# Results

Five optimisation runs. `run_ep30_eps0.04_geom1.2_lpips1.8_20260309_233825` is
**Run D**, which supplies every headline number in the paper; the four 5-epoch
runs are Runs A, B, C and E of the ablation.

| Paper label | Folder | Epoch used |
|---|---|---|
| A | `run_ep5_eps0.045_geom2.0_lpips0.8_20260307_010242` | 5 |
| B | `run_ep5_eps0.04_geom1.5_lpips1.2_20260307_184602` | 5 |
| C | `run_ep5_eps0.04_geom1.2_lpips1.8_20260308_140040` | 5 |
| **D** | `run_ep30_eps0.04_geom1.2_lpips1.8_20260309_233825` | 30 |
| E | `run_ep5_eps0.04_geom1.0_lpips5.0_20260312_105648` | 5 |

## Which CSV backs which table

All inside Run D's folder. Every value below was re-derived from these files and
matches the paper as printed.

| Paper element | Source CSV | n |
|---|---|---|
| Table II — perceptual + both FaceNet rows | `results_v2_full.csv` | 13,232 |
| Table II — ArcFace row | `results_arcface.csv` | 1,000 |
| Table II — DDPM + StarGAN rows | `results_v2_FINAL.csv` | 1,000 |
| Table III — DDPM sampling sweep | `results_ddpm_sampling.csv` | 20 |
| Table IV — transform robustness | `results_robust_FINAL.csv` | 250 |
| Table V — ablation | `results_ablation.csv` | 250 |
| Table VI — Fawkes comparison | `results_fawkes.csv` | 99 |
| Fig. 3 — operating curve | `results_v2_curve_FINAL.csv` + `results_arcface_curve.csv` | 250 |
| §V-C — `[0,1]` vs `[-1,1]` convention check | `results_v2_unet01.csv` | 250 |
| §V-C — held-out EMA checkpoint | `results_ddpm_heldout.csv` | 250 |

> **The `_FINAL` suffix is misleading.** `results_v2_FINAL.csv` is the
> 1,000-image run; Table II's perceptual and FaceNet rows come from
> `results_v2_full.csv` (13,232 images), which carries no suffix. The suffix
> marks *overwrite protection* — `scripts/evaluate.py` defaults to `--out
> results_v2.csv` — not the paper of record. Pass `--csv` explicitly to
> `scripts/stats.py` and `scripts/plot_operating_curve.py`; their docstrings still name the
> unsuffixed files.

All CSVs are long format (`image, variant, metric, value`), so any statistic can
be recomputed offline without re-running a model. `variant` is `uap` (full
strength), `uap0.25`/`uap0.5`/`uap0.75` (scaled), `rand` (matched-ε noise), or a
baseline's name.

## Other contents

- `perturbation_epoch_*.pt` — per-epoch checkpoints. Epoch 30 is the distributed
  perturbation; the rest are training provenance.
- `training_log.csv` — per-epoch losses for all three surrogates plus the LPIPS
  penalty.
- `training_console_log.txt` — console output of the ~60 h optimisation run.
- `sampling_panels/` — Fig. 2 source images (original | protected | generated
  from clean | generated from protected).
- `graph_*.png`, `evaluation_images/`, `whitebox_tests/`, `blackbox_tests/` —
  outputs of the superseded evaluation (see `../archive/README.md`). The
  `blackbox_tests/` images came from the asymmetric script and should not be
  used as evidence.

## Regenerating

`data/raw/` was removed to save 2.2 GB; re-extract from `../Data sets/*.zip` and
run `preprocessing.py` if you need it. `data/processed/` is retained — every
number above depends on that exact 13,232-image evaluation set.
