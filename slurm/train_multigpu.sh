#!/bin/bash
#SBATCH --job-name=TrainGPT
#SBATCH --output=runs/.slurm/%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=36G

IMAGE=train_gym.sif

cd /home/m-ben-salah/repos/GPTino
RUN_DIR=runs/$(date "+%Y-%m-%d_%H-%M-%S")_job_${SLURM_JOB_ID}
mkdir -p ${RUN_DIR}/metadata/
ln -s ../.slurm/${SLURM_JOB_ID}.out ${RUN_DIR}/slurm.out


git diff HEAD > ${RUN_DIR}/metadata/git_diff.patch
apptainer exec ${IMAGE} pip freeze > ${RUN_DIR}/metadata/env.txt

write_meta() {
    cat > ${RUN_DIR}/metadata/meta.json <<EOF
{
    "status": "$1",
    "start_time": "${START_TIME}",
    "last_update": "$(date -Iseconds)",
    "git_commit": "$(git rev-parse HEAD)",
    "slurm_job_id": "${SLURM_JOB_ID}",
    "slurm_nodelist": "${SLURM_JOB_NODELIST}"
}
EOF
}
START_TIME=$(date -Iseconds)
write_meta running

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export NCCL_SOCKET_IFNAME=ens18

srun apptainer exec --nv --env PYTHONPATH=src ${IMAGE} \
        torchrun \
            --nnodes=${SLURM_NNODES} \
            --nproc_per_node="gpu" \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            --rdzv_id=gpt2_job_${SLURM_JOB_ID} \
            scripts/train_gpt2.py \
                --run_dir ${RUN_DIR} \
                
[ $? -eq 0 ] && write_meta completed || write_meta failed