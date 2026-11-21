#!/bin/bash
set -eu # -u forces bash to error when it reaches an unset variable (apparently for bash if a variable does not exists it will consider it as "")

if [[ ${1:-} == --help ]]; then
    echo 'Usage: bash slurm/sweep_lr.sh RATE [RATE ...]'
    exit 0
fi

for lr in "$@"; do
    bash slurm/submit_train.sh --run_dir runs/gpt2-124M-adamw10b/lr-sweep --label "lr-$lr" --set "training.lr_scheduler.peak_lr=$lr"
done
