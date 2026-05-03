# CellFlow

A comparative study of deep learning approaches for single-cell RNA sequencing trajectory inference on the Embryoid Body (EB) dataset.

## Overview

This project investigates whether nonlinear dimensionality reduction improves Neural ODE-based trajectory inference in single-cell transcriptomics. We implement and compare four methods:

1. **Neural ODE on PCA space** — direct trajectory modeling in 50D PCA space
2. **Sequential VAE + Neural ODE** — VAE compression followed by ODE in latent space
3. **Joint VAE + Neural ODE** — co-trained VAE and ODE with dynamic regularization
4. **Flow Matching** — simulation-free continuous normalizing flow

All methods are evaluated using Wasserstein distance in a shared 50D PCA coordinate space on held-out time points from the EB differentiation dataset.

## Results

| Method | Avg Wasserstein (50D PCA) |
|--------|--------------------------|
| Neural ODE on 50D PCA | 0.2275 ± 0.0037 |
| Flow Matching | 0.3457 ± 0.0036 |
| Joint VAE + ODE | 0.3854 ± 0.0044 |
| Sequential VAE + ODE | 0.7770 ± 0.0036 |

**Key findings:**
- Direct Neural ODE on PCA space outperforms VAE-based approaches on this dataset
- Sequential VAE + ODE fails due to decoder generalization errors at unobserved time points
- Joint training (scNODE-inspired) partially recovers performance but does not close the gap
- Flow Matching achieves competitive results with simpler, more stable training
- Dimensionality ablation shows 50D PCA is optimal among tested configurations (5, 10, 20, 30, 50)

## Dataset

Embryoid Body (EB) single-cell RNA-seq dataset from Moon et al. (2019), accessed via the [TrajectoryNet package](https://github.com/KrishnaswamyLab/TrajectoryNet). 16,819 cells across 5 time points, represented as 50-dimensional PCA embeddings.

## Project Structure

```
cellflow/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA, PCA visualization
│   ├── 02_vae.ipynb                   # VAE training (50D → 10D latent)
│   ├── 03_neural_ode.ipynb            # Neural ODE (PCA + sequential VAE)
│   ├── 04_flow_matching.ipynb         # Flow Matching (original + OT-CFM)
│   ├── 05_joint_training.ipynb        # Joint VAE + Neural ODE
│   └── 06_comparison.ipynb            # Final comparison, vector field, ablation
├── models/                            # Saved model weights (not tracked by git)
├── data/                              # Dataset files (not tracked by git)
├── figures/                           # Generated plots
└── requirements.txt
```

## Setup

```bash
# clone the repo
git clone https://github.com/jamehrotra/cellflow.git
cd cellflow

# create virtual environment
python3 -m venv cellflow_venv
source cellflow_venv/bin/activate  # Linux/WSL
# or: cellflow_venv\Scripts\activate  # Windows

# install dependencies
pip install -r requirements.txt
pip install "jax[cuda12]"  # optional, Linux/WSL only
```

## Dependencies

- PyTorch + CUDA
- torchdiffeq
- torchcfm
- scanpy
- anndata
- TrajectoryNet
- scikit-learn
- scipy
- matplotlib
- seaborn

See `requirements.txt` for full list with versions.

## Methods

### Neural ODE
Learns a continuous vector field f(z, t) = dz/dt over cell state space using the adjoint sensitivity method via torchdiffeq. Trained with MMD loss between predicted and observed cell distributions at each time point, with energy regularization to prevent velocity collapse.

### Flow Matching
Learns a velocity field v(z, t) that transports the source cell distribution (t=0) to target distributions using simulation-free conditional flow matching via torchcfm. Trained on consecutive time point pairs.

### Joint VAE + Neural ODE
Co-trains a VAE and Neural ODE simultaneously using a combined loss: VAE reconstruction + KL + dynamic regularization (ODE latent vs encoded observations) + prediction loss (decoded ODE vs observed cells). Inspired by scNODE (Zhang et al., 2024).

## References

- Tong et al. (2020). TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics. ICML.
- Lipman et al. (2022). Flow Matching for Generative Modeling. arXiv:2210.02747.
- Chen et al. (2018). Neural Ordinary Differential Equations. NeurIPS.
- Lopez et al. (2018). Deep generative modeling for single-cell transcriptomics. Nature Methods.
- Zhang et al. (2024). Modeling single-cell dynamics with neural ODEs. Bioinformatics.

## Course

MEAM 4600 — AI for Science and Engineering: From Data to Discovery
University of Pennsylvania, Spring 2026
