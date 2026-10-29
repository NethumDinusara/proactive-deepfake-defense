# Geometrically-aware Universal Adversarial Perturbations for Cross-Architecture Facial Privacy Protection

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Nethum Dinusara Perera
**Institution:** Informatics Institute of Technology (IIT) affiliated with Robert Gordon University (RGU), Aberdeen, UK
**Supervisor:** Akarshani Amarasinghe
**Degree:** BSc. (Hons) in Artificial Intelligence and Data Science (2025/2026)

> Submitted as *A Proactive Tri-Architecture Black-Box Defense Against Deepfakes
> Using Geometrically-Aware Universal Perturbations* for the BSc thesis. The
> paper was retitled when the evaluation was rebuilt; see
> [Evaluation](#-evaluation-and-benchmarking).

---

## 📖 Project Overview

Facial images posted publicly can be scraped to enrol identities in
face-recognition systems without consent, or used as source material for
generative manipulation. Reactive detection of manipulated media generalises
poorly to generators unseen during training, which motivates *proactive*
defences: perturbing an image before it is shared so that downstream models
fail on it.

This repository implements a **Universal Adversarial Perturbation (UAP)**: one
fixed tensor, computed once offline and added to any image thereafter. It is
optimised in **feature space** against a heterogeneous ensemble of three frozen
networks (a FaceNet identity encoder, a StarGAN generator and a DDPM U-Net),
using a directional objective that minimises cosine similarity while maximising
mean-squared error between clean and perturbed features. Sequential gradient
accumulation keeps peak memory below **4 GB**.

## ✨ What It Does

* **Biometric evasion that transfers across architectures.** On 13,232 held-out
  LFW images the perturbation raises embedding distance by **77.2x** over an
  L-infinity-matched noise baseline on the surrogate encoder, and by **33.6x**
  and **18.6x** on two recognisers never seen during optimisation, exceeding the
  baseline on *every* image.
* **Lower perceptual cost than the noise it is compared against.** LPIPS 0.1723
  against 0.3617 for matched-epsilon random noise. The perturbation is not
  smaller than noise; it is shaped so the same magnitude falls in directions a
  perceptual metric discounts.
* **Survives ordinary image handling.** Retains at least 91% of its displacement
  under JPEG re-encoding down to quality 50, Gaussian blur and 5-bit
  quantisation; 78% in the worst case of 4x downsampling.
* **O(1) at protection time.** One 60-hour optimisation, after which protection
  is a tensor addition, with no optimiser, gradients or model at the point of
  use.
* **Two negative results, reported as such.** The perturbation does **not**
  meaningfully disrupt the StarGAN generator (1.16x), and extending training
  from 5 to 30 epochs reallocates capacity rather than improving the
  perturbation on every axis.

## 🧠 Required Models and Weights

To execute the defense pipeline, the following pre-trained model weights and checkpoints must be downloaded and placed within the designated directories. 

1.  **FaceNet Checkpoints:**
    * Used for identity feature extraction and biometric disruption.
    * **Source:** `facenet-pytorch` (InceptionResnetV1 trained on VGGFace2 or CASIA-Webface).
    * **Path:** Automatically downloaded via PyTorch hub, or place manually in `models/facenet/`.
2.  **StarGAN Generator Weights:**
    * Used for spatial and attribute manipulation simulation.
    * **Required File:** `200000-G.ckpt` (or corresponding trained weights for CelebA 256x256).
    * **Path:** Place within the `stargan_celeba_256/models/` directory.
3.  **DDPM U-Net (diffusion surrogate):**
    * Used for latent diffusion disruption during optimisation and evaluation.
    * **Source:** Hugging Face `google/ddpm-celebahq-256`.
    * Stable Diffusion v1.5 is used **only** by the demo in `app/app.py`; no
      reported result depends on it.
    * **Path:** Downloaded dynamically via the `diffusers` library pipeline. Ensure a valid Hugging Face access token is configured if required.

## 📂 Required Datasets

For evaluation and testing, the system relies on standard facial recognition and manipulation datasets:
* **CelebA-HQ + FFHQ:** 5,000 images (2,500 each) for optimising the perturbation.
* **Labeled Faces in the Wild (LFW):** 13,232 MTCNN-aligned images for evaluation, sharing no identities with the optimisation set.
* The held-out recognisers are FaceNet/CASIA-Webface (different weights, same backbone) and ArcFace R50 / WebFace600K (different backbone, loss family and corpus).

## 🛠️ System Requirements & Installation

### Prerequisites
* OS: Windows 11 or Linux. Developed and verified on Windows 11.
* GPU: NVIDIA GPU with CUDA 12.1. **4 GB VRAM is sufficient** for the full
  tri-surrogate optimisation; sequential gradient accumulation exists precisely
  so a larger card is not required. Verified on an RTX 3050 (4 GB). Evaluation
  runs on CPU, more slowly.
* Python 3.10 (verified on 3.10.11).

### Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/NethumDinusara/proactive-deepfake-defense.git
    cd proactive-deepfake-defense
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    The CUDA builds of torch need PyTorch's own index:
    ```bash
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121
    ```

    **Do not upgrade numpy past 1.26.4.** numpy 2.x breaks facenet-pytorch,
    lpips and scikit-image in this stack, and it does so silently: results
    change rather than raising an error. `requirements-lock.txt` holds the full
    environment freeze if you need a byte-exact reproduction.

## 🏗️ Repository Structure

```text
defense_model/
├── config.yaml                 # Paths and hyperparameters, read by every script
├── requirements.txt            # Pinned direct dependencies
├── requirements-lock.txt       # Full environment freeze
│
├── src/uap/                    # Shared library. No entry points.
│   ├── config.py               #   load_config / seeding / device / PROJECT_ROOT
│   ├── compat.py               #   `aim` stub, imported before torch
│   ├── data.py                 #   image listing and loading, FlatFolderDataset
│   ├── perturbation.py         #   UAP loading, matched-epsilon noise control
│   ├── augment.py              #   Expectation-over-Transformation augmentations
│   └── ensemble.py             #   Tri-architecture surrogate ensemble
│
├── scripts/                    # Entry points. Run from anywhere.
│   ├── preprocess.py           #   Build the train and evaluation sets
│   ├── train.py                #   Optimise the UAP
│   ├── evaluate.py             #   Stealth + per-model disruption (Table II)
│   ├── eval_arcface.py         #   Held-out architecture (Table II)
│   ├── eval_heldout_ddpm.py    #   Held-out diffusion checkpoint (§V-C)
│   ├── eval_ddpm_sampling.py   #   Does the generated image change? (Table III)
│   ├── eval_robustness.py      #   JPEG / resize / blur / bit-depth (Table IV)
│   ├── eval_ablation.py        #   Runs A-E (Table V)
│   ├── baseline_folder.py      #   Any published defence, from a folder (Table VI)
│   ├── baseline_cmua.py        #   CMUA-Watermark from its released tensor
│   ├── verify_onnx_batching.py #   Confirms batched ONNX == single-image
│   ├── stats.py                #   Bootstrap CIs, paired Wilcoxon, effect sizes
│   └── plot_operating_curve.py #   Fig. 3
│
├── app/app.py                  # Streamlit demonstration interface
│
├── data/                       # processed/ (evaluation set), Demo/
├── models/                     # ArcFace ONNX weights
├── stargan_celeba_256/         # StarGAN generator checkpoint
├── third_party/stargan/        # Vendored StarGAN source
├── results/                    # Runs and per-image CSVs -> results/README.md
├── figures/                    # Paper figures and operating-curve data
├── notebooks/                  # Exploratory work
└── archive/                    # Superseded code and runs -> archive/README.md
```

Scripts put `src/` and `third_party/` on `sys.path` via `scripts/_path.py`, and
`config.yaml` paths resolve against the project root, so any script runs from
any working directory.

**Start with [`results/README.md`](results/README.md)** — it maps every table and
figure in the paper to the exact CSV it was computed from.

## 🚀 Usage

Generate the perturbation (~60 h on a 4 GB card):

```bash
python scripts/preprocess.py                    # build train/test sets first
python scripts/train.py
```

Reproduce the evaluation against an existing run:

```bash
RUN=run_ep30_eps0.04_geom1.2_lpips1.8_20260309_233825

python scripts/evaluate.py      --run_folder $RUN                  # Table II
python scripts/eval_arcface.py  --run_folder $RUN --n 1000         # Table II
python scripts/eval_robustness.py --run_folder $RUN --n 250        # Table IV
python scripts/eval_ddpm_sampling.py --run_folder $RUN --n 20      # Table III

python scripts/stats.py --csv results/$RUN/results_v2_FINAL.csv    # CIs + tests
python scripts/plot_operating_curve.py     --csv results/$RUN/results_v2_curve_FINAL.csv           results/$RUN/results_arcface_curve.csv                   # Fig. 3
```

Launch the demonstration interface:

```bash
streamlit run app/app.py
```

## 📊 Evaluation and Benchmarking

Every figure below is reproducible from the per-image CSVs in `results/`; see
[`results/README.md`](results/README.md) for which file backs which table. The
baseline is uniform random noise at the same L-infinity bound, drawn per image
with a fixed seed. Significance is a paired Wilcoxon signed-rank test. The
success criterion tau-99 is the 99th percentile of the noise baseline, which the
baseline meets for 1.0% of images by construction.

### Headline results (13,232 held-out LFW images)

| Metric | Noise | Ours | Ratio |
|---|---|---|---|
| LPIPS (lower better) | 0.3617 | **0.1723** | 0.48x |
| SSIM (higher better) | 0.7566 | **0.7839** | 1.04x |
| PSNR dB (higher better) | **32.76** | 29.27 | 0.89x |
| FaceNet-VGGFace2 cosine distance *(surrogate)* | 0.0028 | **0.2125** (100% at tau-99) | **77.2x** |
| FaceNet-CASIA *(held-out weights)* | 0.0081 | **0.2738** (100%) | **33.6x** |
| ArcFace R50 *(held-out architecture, n=1,000)* | 0.0143 | **0.2662** (100%) | **18.6x** |
| DDPM U-Net MSE *(surrogate)* | 0.0072 | **0.0114** (100%) | 1.57x |
| StarGAN output MSE *(surrogate)* | 0.0015 | 0.0017 (0.6%) | 1.16x |

The multiplier falls monotonically with distance from the surrogate
(77.2 -> 33.6 -> 18.6), driven entirely by the baseline noise floor rising as
the recogniser becomes more sensitive to unstructured perturbation. Absolute
displacement is close to architecture-independent.

### Generative disruption is narrow

StarGAN disruption is **not established**: at tau-99 the perturbation succeeds
on 0.6% of images, below the baseline own rate of 1%. The paired test is
significant (p is about 5e-139) but the shift is small relative to the baseline
spread. Statistical reliability and practical relevance are not the same thing.

Diffusion disruption is real but confined to low denoising strength. Sampling
from the model directly, the protected input moves the output 1.78x further than
matched noise at t_start = 50, 1.08x at 200, and 1.02x at 400, where it is no
longer distinguishable from the baseline (p = 0.083).

### Comparison with Fawkes

At **matched perceptual cost** Fawkes is the stronger method for biometric
evasion by roughly threefold: CASIA displacement 0.1050 at LPIPS 0.0283, against
about 0.035 for ours scaled to the same cost. The gap narrows to 1.4x on the
held-out architecture. Neither method dominates. Fawkes runs a separate
optimisation per image at protection time; ours is a single tensor addition and
carries a measured effect on a diffusion model. Where minimal visible change is
the priority and only face recognition is in scope, Fawkes is the better choice
on this evidence.

### Limitations

One published baseline (Fawkes, low mode, 99 images); CMUA-Watermark and LowKey
were not obtained. A published perturbation is a known perturbation: an
adversary who obtains it can subtract it or train against it, and no claim is
made against an adaptive adversary. The method degrades face recognition without
distinguishing between authorised and unauthorised uses. Single optimisation run
(seed 42); perceptual cost is assessed by metric, with no human study. Still
images at 256x256 only.
