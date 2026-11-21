#!/bin/bash
set -eu # -u forces bash to error when it reaches an unset variable (apparently for bash if a variable does not exists it will consider it as "")

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash slurm/sweep_lr.sh RATE [RATE ...]'
    exit 0
fi

if (( $# == 0 )); then
    echo 'Usage: bash slurm/sweep_lr.sh 1e-4 3e-4 6e-4' >&2
    exit 2
fi

for lr in "$@"; do
    bash slurm/submit_train.sh --run_dir runs/lr-sweep --label "lr-$lr" --set "training.lr_scheduler.peak_lr=$lr"
done
