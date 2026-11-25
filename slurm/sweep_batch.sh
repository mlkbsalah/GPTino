#!/bin/bash
set -eu # -u forces bash to error when it reaches an unset variable (apparently for bash if a variable does not exists it will consider it as "")

if [[ ${1:-} == --help ]]; then
    echo 'Usage: bash slurm/sweep_batch.sh BATCH_SIZE [BATCH_SIZE ...]'
    exit 0
fi

for batch_size in "$@"; do
    bash slurm/submit_train.sh --run_dir runs/gpt2-124M-adamw10b/batch-sweep --label "batch-$batch_size" --set "training.train_batch_tokens=$batch_size"
done
