#!/bin/bash
#SBATCH --job-name=he_inference
#SBATCH --account=st-ipor-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/st-ipor-1/lwqz/Interactive_HardExudates_Segmentation/hpc_logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/lwqz/Interactive_HardExudates_Segmentation/hpc_logs/%x-%j.err
#SBATCH --mail-user=lwqz@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

module purge
module load intel-oneapi-compilers/2023.1.0
module load cuda/11.8.0
module load cudnn/8.6.0.163-11.8

source /home/lwqz/miniconda3/etc/profile.d/conda.sh
conda activate 2-stage-cnn
PYTHON=/home/lwqz/miniconda3/envs/2-stage-cnn/bin/python

cd /scratch/st-ipor-1/lwqz/Interactive_HardExudates_Segmentation

# Redirect PyTorch cache to writable scratch storage
export TORCH_HOME=/scratch/st-ipor-1/lwqz/.cache/torch
mkdir -p "$TORCH_HOME"

# Run inference
"$PYTHON" demo.py --checkpoint=RITM_EX.pth --automatic_weight='' --gpu=0
