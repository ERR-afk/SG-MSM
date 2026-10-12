# SG-MSM: SNR-Guided Data Augmentation for Mitigating Domain Shift in Avian Bioacoustics

This repository contains the official PyTorch implementation of **SG-MSM**, a multi-stage data augmentation pipeline that explicitly mitigates the focal-to-passive domain shift for avian bioacoustics.

## Overview
Passive acoustic monitoring (PAM) models often suffer from SNR distribution shifts when trained on high-quality focal recordings. SG-MSM utilizes an SNR-Guided Mixing algorithm to blend focal calls with target-site passive noise to construct a robust, multi-label training distribution under low-resource constraints.

## Repository Structure
- `src/training.py`: Main PyTorch training script utilizing an EfficientNet-B3 backbone.
- `src/training_utils.py`: Contains the core SG-MSM framework implementations (SG-Mix algorithm via `noisereduce`, Multi-SNR Domain NoiseMix, Iterative Multi-Label Overlapping, and Cross-SNR Spectrogram Splicing).
- `configs/`: The 5 base experimental configuration files corresponding to the ablation studies and the main model in the paper.
- `preprocessing/`: Python scripts used for data downloading, noise mining, and tensor conversion.
- `prepare_all_data.sh`: Bash script to fully automate the downloading and preprocessing of data across all 6 evaluation domains (PER, HSN, SSW, SNE, NES, UHH).
- `run_all_experiments.py`: Python launcher script to automate running the 5 experimental configurations across all 6 domains (30 total runs).
- `perch_ezhini_sgmsm_env.yml`: Conda environment definition for exact reproducibility.
- `perch_ezhini_sgmsm_requirements.txt`: Pip requirements list.

## Getting Started

### 1. Environment Setup
You can recreate the exact environment used on the DGX-2 system using either Conda or Pip:

**Using Conda (Recommended):**
```bash
conda env create -f perch_ezhini_sgmsm_env.yml
conda activate perch_ezhini_sgmsm
```

**Using Pip:**
```bash
pip install -r perch_ezhini_sgmsm_requirements.txt
```

### 2. Data Preparation
To download and preprocess the data (noise mining, NPY conversion) for all 6 domains, simply run the provided bash script:
```bash
./prepare_all_data.sh
```

### 3. Training the Models
To automatically run the entire suite of experiments (30 total runs spanning the 5 ablation/full models across all 6 domains) ensuring WandB logs and checkpoints are correctly separated:
```bash
python3 run_all_experiments.py
```

*Alternatively, to run a single specific config manually:*
```bash
python3 src/training.py -c configs/config_5_effnet_full_sg_msm.json
```

### 4. Evaluation
The script `src/evaluate.py` contains the complete inference pipeline to calculate ROC-AUC, cmAP, and Top-1 Accuracy metrics. It perfectly mirrors the PyTorch Lightning data preprocessing of BirdSet to ensure a fair evaluation baseline.

## Citation
If you find this code useful, please cite our TMLR 2026 paper.
