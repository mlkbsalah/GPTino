#!/bin/bash
#SBATCH --job-name=TrainGPTino
#SBATCH --output=logs/%x.o%j
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --partition=gpua100

module load anaconda3/2023.09-0/none-none
module load cuda/13.0.2/none-none

source activate ml_env

time python3 train_gpt2.py
