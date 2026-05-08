#!/bin/bash
#SBATCH --job-name=TrainAttentionV2
#SBATCH --output=logs/%x.o%j
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100

# Module and env setup
module load anaconda3/2023.09-0/none-none
module load cuda/13.0.2/none-none

source activate ml_env

# Script
python3 v2.py
