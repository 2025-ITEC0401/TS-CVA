#!/bin/bash
# Store LLM Embeddings for UEA dataset
# Usage: ./store_emb.sh <dataset_name> [gpu_id]

DATASET=${1:-"BasicMotions"}
GPU=${2:-0}

echo "======================================"
echo "Storing LLM Embeddings"
echo "Dataset: $DATASET"
echo "GPU: $GPU"
echo "======================================"

cd /hdd/intern/daniel/TS-CVA

python storage/store_emb_uea.py \
    --dataset $DATASET \
    --loader UEA \
    --gpu $GPU \
    --batch_size 1

echo "Done!"
