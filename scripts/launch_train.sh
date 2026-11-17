#!/bin/bash
set -e

export RUN_SUFFIX=""
if [[ "$1" == "--label" ]]; then
    RUN_SUFFIX="_${2:?Usage: --label NAME [--debug]}"
    shift 2
fi
for arg in "$@"; do
    if [[ "$arg" == "--debug" ]]; then
        RUN_SUFFIX="${RUN_SUFFIX}_debug"
        break
    fi
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs/.slurm
sbatch slurm/train_multigpu.sh "$@"
