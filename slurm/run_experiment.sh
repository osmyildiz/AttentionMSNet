#!/bin/bash
#SBATCH --job-name=alzxai
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# SLURM Job Template for AttentionMS-Net Experiments
# Target: NVIDIA DGX with A100-SXM4-80GB GPUs
# ============================================================

# Load required modules
module purge
module load cuda12.1/toolkit/12.1.1 cudnn8.9-cuda12.1/8.9.7.29

# Set working directory
cd /network/rit/dgx/dgx_subasi_lab/osman/AlzAttentionXAI_DGX

# Activate virtual environment
source alzxai_env/bin/activate

# Environment variables
export PYTHONUNBUFFERED=1
export TMPDIR=/network/rit/dgx/dgx_subasi_lab/osman/tmp

# Create log directory if it doesn't exist
mkdir -p logs

# ============================================================
# Run the experiment
# Usage: sbatch slurm/run_experiment.sh experiments/02_train_attentionmsnet.py
# Or edit the line below:
# ============================================================

SCRIPT=${1:-"experiments/02_train_attentionmsnet.py"}
echo "Running: $SCRIPT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"

python -u $SCRIPT 2>&1

echo "Finished: $(date)"
