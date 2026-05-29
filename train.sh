#!/bin/bash
#SBATCH --job-name=TrainGPT
#SBATCH --output=train_gpt_output_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=ai
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=36G

cd /home/m-ben-salah/repos/GPTino

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export NCCL_SOCKET_IFNAME=ens18

time srun apptainer exec --nv \
        gptino.sif \
        torchrun \
            --nnodes=2 \
            --nproc_per_node=2 \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            --rdzv_id=gpt2_job_${SLURM_JOB_ID} \
            train_gpt2.py