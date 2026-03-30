# A Proactive Tri-Architecture Black-Box Defense Against Deepfakes Using Geometrically-Aware Universal Perturbations

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Nethum Dinusara Perera  
**Institution:** Informatics Institute of Technology (IIT) affiliated with Robert Gordon University (RGU), Aberdeen, UK  
**Supervisor:** Akarshani Amarasinghe  
**Degree:** BSc. (Hons) in Artificial Intelligence and Data Science (2025/2026)

---

## 📖 Project Overview

The rapid evolution of generative artificial intelligence has rendered conventional, reactive deepfake detection mechanisms largely obsolete. As synthetic media generation capabilities outpace detection algorithms, our digital ecosystem requires a paradigm shift toward proactive, mathematical, and engineered defense mechanisms. 

This repository contains the official implementation of the **Geometrically-Aware Universal Disruption Algorithm**, a novel proactive defense designed to immunize facial photographs against malicious manipulation before they are distributed. Moving adversarial optimization away from precarious pixel space and deeply embedding it within the geometric decision space, this framework utilizes a **Shortest-Distance Soft Maximum (SDSM)** strategy. This systematically attacks the internal latent feature space of generative algorithms, shattering the generation pipelines of both spatial GANs and Diffusion Probabilistic Models under strict black-box conditions.

## ✨ Core Features

* **Proactive Immunization:** Applies imperceptible or minimally visible Universal Adversarial Perturbations (UAPs) to source images, rendering them unusable for downstream deepfake generation.
* **Tri-Architecture Surrogate Ensemble:** Ensures robust real-world generalization across disparate generative paradigms by optimizing against a tri-fold architectural bottleneck:
    1.  **FaceNet:** Simulates the biometric identity extraction bottleneck.
    2.  **StarGAN:** Simulates adversarial synthesis and spatial generation.
    3.  **DDPM (Stable Diffusion v1.5):** Simulates state-of-the-art latent diffusion processes.
* **Resource-Efficient Optimization:** Implements strict memory management combined with Sequential Gradient Accumulation, democratizing adversarial security by enabling state-of-the-art cryptographic shield generation without requiring multi-million-dollar hardware clusters.
* **Catastrophic Structural Collapse:** Forces severe geometric hallucination and identity-swapping pipeline failure across unseen models.

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
3.  **Stable Diffusion v1.5 (DDPM):**
    * Used for latent diffusion disruption.
    * **Source:** Hugging Face `runwayml/stable-diffusion-v1-5`.
    * **Path:** Downloaded dynamically via the `diffusers` library pipeline. Ensure a valid Hugging Face access token is configured if required.

## 📂 Required Datasets

For evaluation and testing, the system relies on standard facial recognition and manipulation datasets:
* **Labeled Faces in the Wild (LFW):** Used for unconstrained empirical testing of structural collapse.
* **CASIA-Webface:** Utilized for evaluating black-box transferability and biometric extraction failure.

## 🛠️ System Requirements & Installation

### Prerequisites
* OS: Linux / Windows (WSL2 recommended)
* GPU: NVIDIA GPU with CUDA support (Minimum 8GB VRAM for Sequential Gradient Accumulation)
* Python 3.8+

### Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/proactive-deepfake-defense.git](https://github.com/YourUsername/proactive-deepfake-defense.git)
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

## 🏗️ Repository Structure

```text
proactive-deepfake-defense/
│
├── app.py                 # Main entry point for the defense interface
├── config.yaml            # Configuration parameters (learning rate, SDSM margins, LPIPS thresholds)
├── train.py               # UAP optimization loop over the Tri-Architecture ensemble
├── evaluate.py            # Evaluation scripts for benchmarking UAP efficacy
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore file ensuring models and venv are excluded
│
├── models/                # Directory for FaceNet and other auxiliary checkpoints
├── stargan_celeba_256/    # StarGAN specific weights and configurations
├── data/                  # Input directories for LFW / CASIA-Webface samples
└── results/               # Output directory for immunized images and generated graphs


```
## 🚀 Usage
1. Generating the Universal Adversarial Perturbation (UAP)
To begin the optimization process and generate a protective UAP for a given set of images, configure your paths in config.yaml and run:

```Bash
python train.py --config config.yaml
```
Note: The optimization leverages Sequential Gradient Accumulation to maintain VRAM efficiency across the three massive architectures.

2. Applying the Defense
To apply the generated UAP to new, unseen images (proactive immunization):

```Bash
python app.py --input data/raw_images --output results/immunized_images
```
## 📊 Evaluation and Benchmarking
This research validates the theoretical framework of the Geometrically-Aware Universal Disruption Algorithm against established academic standards. Rather than relying on isolated experimental runs, the evaluation protocol utilizes recognized benchmarks from existing state-of-the-art literature to measure defensive efficacy.

* Structural Disruption: Evaluated against established baseline metrics for StarGAN and Stable Diffusion v1.5, proving that the generated UAPs force generative pipelines into catastrophic structural collapse.

* Stealth-Robustness Trade-off (LPIPS): Visual imperceptibility and structural integrity are measured using Learned Perceptual Image Patch Similarity (LPIPS). The defense mathematically establishes the necessity of a specific geometric artifact threshold (LPIPS: ~0.16) required to maintain the >1.5× angular divergence necessary to consistently crash identity-swapping bottlenecks without degrading the original image's human-perceived quality.

* Transferability: Performance drops on unseen models (e.g., CASIA-Webface confidence degradation to 52.6%) align with and exceed established literature benchmarks for black-box adversarial transferability.
