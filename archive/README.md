# Archive

Superseded material, kept for provenance. Nothing here is cited by the current
paper (*Geometrically-aware Universal Adversarial Perturbations for
Cross-Architecture Facial Privacy Protection*). It is retained because the old
evaluation is the evidence trail for why the current one exists, and a reviewer
may ask what changed.

## `code/`

| File | Why it is here |
|---|---|
| `evaluate.py` | Original evaluation. Derived **every** "protection" number from the DDPM U-Net alone — FaceNet cosine distance and StarGAN MSE were both U-Net readings. Also fed the U-Net `[0,1]` where it expects `[-1,1]`, and used `win_size=3` for SSIM. Replaced by `scripts/evaluate.py`. |
| `whitebox_eval.py` | White-box ASR on the training manifold. Imports `evaluate.py`. Superseded by the τ99 criterion in `scripts/stats.py`. |
| `plots.py` | Figure generator. Mostly commented out; the live portion draws a *theoretical* sigmoid with hidden axes, and the commented portion synthesises a convergence curve with `np.exp`. Not used by the current paper. |
| `test_blackbox_transfer.py` | Qualitative Stable Diffusion panel. **Asymmetric by construction**: `strength=0.5` for the clean input vs `0.8` for the protected one, with no fixed seed. Not usable as evidence. |
| `test_diffusion_generation.py`, `test_gasd_evaluation.py`, `test_tri_architecture.py`, `test_memory.py`, `visualize_output_failure.py` | One-off exploratory scripts. Seeded and symmetric, but superseded by `ddpm_sampling.py`, which shares one noise draw across conditions and reports a statistic. |

**These scripts will not run as they stand.** They were written against the old
flat layout and still do `from utils import ...` and `from models import ...`.
Those modules are now `src/uap/config.py` and `src/uap/ensemble.py`. To run one,
copy it into `scripts/`, add `import _path` at the top, and repoint the imports
at `uap.config` / `uap.ensemble`.

## `results/`

Pre-ablation exploratory runs, none cited:

- `10 epochs new/`, `20 epochs/`, `30 epoch epc=0.05/`, `30 epochs new/`
- `plots/` — legacy figures. The four component curves in
  `perfect_loss_convergence.png` do reproduce `training_log.csv`; the
  "Total Joint Loss" line plotted on top of them does **not** equal their sum
  (plots 4.17→3.74, actual sum 3.32→3.04). `real_radar_chart.png` is the radar
  chart the current paper replaced with the operating curve.

`app.py` still loads `20 epochs/perturbation_epoch_20.pt` for the display-only
perturbation; its path was updated when this folder moved.

The five runs cited in the paper (ablation Runs A–E) remain in `results/`.
