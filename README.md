# AttentionMS-Net

**An Attention-Enhanced Multi-Scale Framework for Alzheimer's Disease Classification with Subject-Level Validation**

*Osman Yildiz and Abdulhamit Subasi*  
University at Albany, State University of New York

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the complete experimental pipeline for our paper on Alzheimer's disease classification from 2D MRI slices. The study makes three key contributions:

1. **Data leakage quantification**: We demonstrate that image-level splitting inflates accuracy by ~19 percentage points compared to proper subject-level splitting (99.9% vs 80.8%).
2. **Three-level evaluation hierarchy**: slice-level → subject-level → subject-level with prediction aggregation, each progressively reducing bias.
3. **CNN vs. Transformer comparison**: Systematic evaluation of AttentionMS-Net (end-to-end CNN with CBAM + multi-scale fusion) against frozen Swin Transformer + ML classifiers under identical 10-fold subject-level cross-validation.

## Key Results

| Model | Accuracy | F1 Macro | AUC-ROC | Sensitivity | Specificity |
|-------|----------|----------|---------|-------------|-------------|
| AttentionMS-Net (slice) | 0.805±0.047 | 0.726±0.057 | 0.866±0.056 | 0.609±0.146 | 0.861±0.069 |
| **AttentionMS-Net (subject)** | **0.824±0.070** | **0.753±0.106** | **0.889±0.055** | **0.653±0.243** | **0.876±0.066** |
| Swin+Stacking (slice) | 0.815±0.045 | 0.741±0.050 | 0.879±0.043 | 0.618±0.064 | 0.872±0.051 |
| **Swin+Stacking (subject)** | **0.841±0.050** | **0.784±0.068** | **0.904±0.042** | **0.701±0.123** | **0.883±0.045** |

> Wilcoxon signed-rank test: p = 1.000 — no statistically significant difference between the two approaches. AttentionMS-Net provides gradient-based interpretability (Grad-CAM++) that frozen pipelines cannot.

## Dataset

We use the [OASIS-1](https://www.oasis-brains.org/) cross-sectional MRI dataset (Marcus et al., 2007), preprocessed as 2D axial slices:

- **347 subjects** (271 Non-Demented, 76 Demented)
- **86,437 slices** total (~250 slices per subject)
- Binary classification: Non-Demented vs. Demented

The dataset is available via [Kaggle](https://www.kaggle.com/datasets/ninadaithal/imagesoasis). Download and place in `data/raw/`.

## Project Structure

```
AttentionMSNet/
├── configs/
│   └── config.yaml                 # Hyperparameters and paths
├── data/
│   ├── raw/                        # Downloaded MRI slices (not tracked)
│   └── splits/
│       ├── full_data.csv           # Master file with subject IDs and labels
│       └── cv_folds.json           # 10-fold subject-level split definitions
├── src/
│   ├── data/
│   │   ├── dataset.py              # AlzDataset (PyTorch Dataset)
│   │   ├── splitter.py             # Subject-level stratified splitting
│   │   └── transforms.py           # Albumentations pipelines
│   ├── models/
│   │   ├── attention.py            # CBAM (Channel + Spatial Attention)
│   │   ├── backbone.py             # EfficientNet-B3 feature extractor
│   │   ├── multiscale.py           # Multi-scale feature fusion
│   │   ├── hybrid_model.py         # AttentionMS-Net (full architecture)
│   │   ├── swin_attention.py       # Swin Transformer with attention fusion
│   │   └── swin_gaf.py             # SwinGAF (experimental, abandoned)
│   ├── training/
│   │   └── losses.py               # Focal Loss, class-weighted CrossEntropy
│   ├── evaluation/
│   │   └── metrics.py              # Accuracy, F1, AUC, sensitivity, specificity
│   ├── ensemble/
│   │   └── ensemble.py             # Stacking ensemble pipeline
│   ├── utils/
│   │   ├── helpers.py              # Seed, device, config loading
│   │   └── visualization.py        # Plotting utilities
│   └── xai/                        # Explainability (Grad-CAM++, SHAP wrappers)
├── experiments/
│   ├── 01_prepare_data.py          # Dataset preparation and subject ID extraction
│   ├── 02_train_attentionmsnet.py  # 10-fold CV training for AttentionMS-Net
│   ├── 03_ablation_study.py        # CNN ablation: baseline → +CW → +CBAM → +MS
│   ├── 04_swin_feature_extraction.py # Extract frozen Swin multi-scale features
│   ├── 05_swin_e2e_training.py     # Swin end-to-end fine-tuning
│   ├── 06_swin_cbam_fusion.py      # Swin frozen + CBAM fusion heads
│   ├── 07_swin_ml_classifiers.py   # Swin frozen + RF/XGB/Stacking
│   ├── 08_leakage_experiment.py    # Image-level vs subject-level leakage test
│   ├── 09_subject_voting_attmsnet.py # Subject-level prediction aggregation (CNN)
│   ├── 10_subject_voting_swin.py   # Subject-level prediction aggregation (Swin+ML)
│   ├── 11_gradcam_analysis.py      # Grad-CAM++ heatmap generation
│   ├── 12_shap_analysis.py         # SHAP feature importance analysis
│   ├── data_utils.py               # Shared augmentation/transform utilities
│   ├── train_utils.py              # Shared training loop utilities
│   └── model_builders.py           # Model factory functions
├── slurm/
│   └── run_experiment.sh           # SLURM job template (DGX A100)
├── paper/
│   └── figures/                    # Publication figures
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/osmanyildiz/AttentionMSNet.git
cd AttentionMSNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Reproducing Results

### 1. Data Preparation
```bash
# Download OASIS dataset from Kaggle, place in data/raw/
python experiments/01_prepare_data.py
```

### 2. CNN Experiments (AttentionMS-Net)
```bash
# Full 10-fold CV training
python experiments/02_train_attentionmsnet.py

# Ablation study (4 configurations)
python experiments/03_ablation_study.py
```

### 3. Swin Transformer Experiments
```bash
# Extract frozen Swin features (one-time, ~30 min)
python experiments/04_swin_feature_extraction.py

# End-to-end Swin fine-tuning
python experiments/05_swin_e2e_training.py

# CBAM fusion on frozen features
python experiments/06_swin_cbam_fusion.py

# ML classifiers (RF, XGBoost, Stacking)
python experiments/07_swin_ml_classifiers.py
```

### 4. Evaluation
```bash
# Data leakage experiment
python experiments/08_leakage_experiment.py

# Subject-level prediction aggregation
python experiments/09_subject_voting_attmsnet.py
python experiments/10_subject_voting_swin.py
```

### 5. Explainability
```bash
# Grad-CAM++ heatmaps
python experiments/11_gradcam_analysis.py

# SHAP analysis
python experiments/12_shap_analysis.py
```

### Running on SLURM (DGX Cluster)

```bash
# Edit slurm/run_experiment.sh with the desired script
sbatch slurm/run_experiment.sh
```

## Hardware

All experiments were conducted on:
- **GPU**: NVIDIA A100-SXM4-80GB (DGX cluster)
- **CPU**: AMD EPYC 7742 (for ML classifiers)
- **Software**: Python 3.10, PyTorch 2.1, CUDA 12.1

## Citation

If you use this code, please cite:

```bibtex
@article{yildiz2026attentionmsnet,
  title={AttentionMS-Net: An Attention-Enhanced Multi-Scale Framework for 
         Alzheimer's Disease Classification with Subject-Level Validation},
  author={Yildiz, Osman and Subasi, Abdulhamit},
  journal={Applied Sciences},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- OASIS dataset: [Marcus et al., 2007](https://doi.org/10.1162/jocn.2007.19.9.1498)
- University at Albany Research IT for DGX cluster access
