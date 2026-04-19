#!/bin/bash
# Usage: bash train.sh <DATASET> <SAVE_NAME> [GPU_ID]

set -e

DATASET=${1:?"Usage: bash train.sh DATASET SAVE_NAME [GPU_ID]"}
SAVE_NAME=${2:?"Usage: bash train.sh DATASET SAVE_NAME [GPU_ID]"}
GPU_ID=${3:-0}

OVERRIDE=$(mktemp /tmp/override_XXXX.yaml)
trap "rm -f $OVERRIDE" EXIT

run() {
    local MODEL=$1 DATASET_ARG=$2 CKPT=$3
    printf "gpu_id: %s\ncheckpoint_dir: %s\n" "$GPU_ID" "$CKPT" > "$OVERRIDE"
    python run_recbole.py --model "$MODEL" --dataset "$DATASET_ARG" \
        --config_files "config.yaml $OVERRIDE"
}

echo "=== [1/5] SASRec baseline ==="
run SASRec "$DATASET" "./saved/sasrec_${SAVE_NAME}"

echo "=== [2/5] price ==="
run SASRec_AddInfo "$DATASET" "./saved/price_${SAVE_NAME}"

echo "=== [3/5] sales_rank ==="
run SASRec_AddInfo "$DATASET" "./saved/salesrank_${SAVE_NAME}"

echo "=== [4/5] review_emb ==="
run SASRec_AddInfo "${DATASET}_review" "./saved/review_${SAVE_NAME}"

echo "=== [5/5] desc_emb ==="
run SASRec_AddInfo "${DATASET}_desc" "./saved/desc_${SAVE_NAME}"

echo "=== Done ==="
