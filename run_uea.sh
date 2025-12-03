#!/bin/bash
# TS-CVA Training Script for UEA datasets

# Default settings
GPU=0
EPOCHS=100
BATCH_SIZE=8
LR=0.001
REPR_DIMS=320

# Dataset to run
DATASET=${1:-"BasicMotions"}
RUN_NAME=${2:-"ts_cva"}

echo "======================================"
echo "TS-CVA Training"
echo "Dataset: $DATASET"
echo "Run Name: $RUN_NAME"
echo "======================================"

cd /hdd/intern/daniel/TS-CVA

python train.py $DATASET $RUN_NAME \
    --loader UEA \
    --gpu $GPU \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --repr-dims $REPR_DIMS \
    --eval

echo "Training completed!"
