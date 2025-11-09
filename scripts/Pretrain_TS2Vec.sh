#!/bin/bash
# TS2Vec Pre-training Script
# Pre-train the vector encoder with contrastive learning

export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "TS2Vec Pre-training"
echo "========================================="
echo ""

# Common parameters
seq_len=96
batch_size=256
epochs=50
learning_rate=1e-3
d_vector=320
hidden_dims=64
depth=10

# Pre-train on ETTm1
data_path="ETTm1"
echo "Pre-training on ${data_path}..."
python pretrain_ts2vec.py \
  --data_path $data_path \
  --seq_len $seq_len \
  --num_nodes 7 \
  --d_vector $d_vector \
  --hidden_dims $hidden_dims \
  --depth $depth \
  --epochs $epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --temperature 0.05 \
  --n_augmentations 2 \
  --aug_prob 0.6 \
  --seed 2024

echo ""

# Pre-train on ETTh1
data_path="ETTh1"
echo "Pre-training on ${data_path}..."
python pretrain_ts2vec.py \
  --data_path $data_path \
  --seq_len $seq_len \
  --num_nodes 7 \
  --d_vector $d_vector \
  --hidden_dims $hidden_dims \
  --depth $depth \
  --epochs $epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --temperature 0.05 \
  --n_augmentations 2 \
  --aug_prob 0.6 \
  --seed 2024

echo ""

# Pre-train on ETTm2
data_path="ETTm2"
echo "Pre-training on ${data_path}..."
python pretrain_ts2vec.py \
  --data_path $data_path \
  --seq_len $seq_len \
  --num_nodes 7 \
  --d_vector $d_vector \
  --hidden_dims $hidden_dims \
  --depth $depth \
  --epochs $epochs \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --temperature 0.05 \
  --n_augmentations 2 \
  --aug_prob 0.6 \
  --seed 2024

echo ""
echo "========================================="
echo "Pre-training completed!"
echo "Checkpoints saved in: ./checkpoints/ts2vec_pretrain/"
echo "========================================="
