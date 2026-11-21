#!/bin/bash
set -e # makes sure that the script exits if it encounters an error (usually bash does not it fails then runs the next command)

runs_root=runs
label_suffix=""
debug_suffix=""
export CHECKPOINT=""
train_args=()

## This part is what parses the arguments
while (($# > 0)); do
    case "$1" in
        -h|--help)
            echo "Usage: WIP"
            exit 0
            ;;
        --run_dir)
            runs_root="${2:?}" # ?  error if is not set (option --run_dir not used ), :  error if is empty (used --run_dir but without an argument)
            if [["$run_root" == -*]]; then
                echo "Expected a directory got: ${run_root}" >&2
                exit 2
            fi
            shift 2
            ;;
        --resume)
            CHECKPOINT="${2:?}"
            if [[ "$CHECKPOINT" == -* ]]; then
                echo "Expected a checkpoint path got: ${CHECKPOINT}" >&2
                exit 2
            fi
            shift 2
            ;;
        --label)
            label_suffix="_${2:?}"
            if [[ "$label_suffix" == -* ]]; then
                echo "Expected a label name got: ${label_suffix}" >&2
                exit 2
            fi
            shift 2
            ;;
        --debug)
            debug_suffix="_debug"
            train_args+=("$1")
            shift 1
            ;;
        *)
            train_args+=("$1")
            shift 1
            ;;
    esac
done

if [[ -n "$CHECKPOINT" ]]; then # -n means a non empty string
    if [[ -n "$debug_suffix" ]]; then
        echo "Cannot use --debug and --resume together" >&2
        exit 1
    fi
    RUN_DIR=$(dirname "$(dirname "$CHECKPOINT")")
    train_args+=(--resume "$CHECKPOINT")
else
    timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
    RUN_DIR="$runs_root/$timestamp${debug_suffix:-$label_suffix}" # if the debug suffix is missing or : i,e "empty string" () use the label suffix
    mkdir -p "$runs_root"
    mkdir "$RUN_DIR"
    train_args+=(--run_dir "$RUN_DIR")
fi

export RUN_DIR
mkdir -p "$RUN_DIR/.slurm"
sbatch --output="$RUN_DIR/.slurm/slurm-%j.out" slurm/train_multigpu.sh "${train_args[@]}"
