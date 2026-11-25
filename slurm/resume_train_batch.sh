#!/bin/bash

train_args=()
runs_dir=""

while (( $# > 0 )); do
    case "$1" in
        --help)
            cat <<EOF

Usage: resume_train_batch.sh [options]

--runs_dir <runs_dir>: Directory where runs are saved
EOF
            exit 0
            ;;
        --runs_dir)
            echo "runs_dir: $2"
            runs_dir="$2"
            shift
            ;;
        *)
            train_args+=("$1")
            ;;
    esac
    shift
done

if [[ -z "$runs_dir" ]]; then
    echo "--runs_dir is required" >&2
    exit 1
fi

for checkpoint in "$runs_dir"/*/checkpoints/latest.pt; do
    if [[ -f "$checkpoint" ]]; then
        bash slurm/submit_train.sh --resume "$checkpoint" "${train_args[@]}"
    fi
done
