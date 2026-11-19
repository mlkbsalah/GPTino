#!/bin/bash
set -e

runs_root=runs
label_suffix=""
debug_suffix=""
export CHECKPOINT=""
train_args=()
while (($#)); do
    case "$1" in
        --run_dir)
            runs_root="${2:?Usage: --run_dir DIRECTORY}"
            shift 2
            ;;
        --resume)
            CHECKPOINT="${2:?Usage: --resume CHECKPOINT}"
            shift 2
            ;;
        --label)
            label_suffix="_${2:?Usage: --label NAME}"
            shift 2
            ;;
        *)
            if [[ "$1" == "--debug" ]]; then
                debug_suffix="_debug"
            fi
            train_args+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$CHECKPOINT" ]]; then
    if [[ -n "$debug_suffix" ]]; then
        echo "Cannot use --debug and --resume together" >&2
        exit 1
    fi
    RUN_DIR=$(dirname "$(dirname "$CHECKPOINT")")
    train_args+=(--resume "$CHECKPOINT")
else
    timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
    RUN_DIR="$runs_root/$timestamp${debug_suffix:-$label_suffix}"
    mkdir -p "$runs_root"
    mkdir "$RUN_DIR"
    train_args+=(--run_dir "$RUN_DIR")
fi

export RUN_DIR
sbatch --output="$RUN_DIR/slurm-%j.out" slurm/train_multigpu.sh "${train_args[@]}"
