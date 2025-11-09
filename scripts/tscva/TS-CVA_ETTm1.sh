#!/bin/bash
# TS-CVA Training Script for ETTm1 Dataset
# Multi-task learning with contrastive loss and forecasting loss

export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Auto-select free GPU
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FREE_GPU=$(bash ${SCRIPT_DIR}/../utils/find_free_gpu.sh)
export CUDA_VISIBLE_DEVICES=${FREE_GPU}
echo "Using GPU: ${FREE_GPU}"

data_path="ETTm1"
seq_len=96
batch_size=128

# TS-CVA specific parameters
contrastive_weight=0.3
forecast_weight=0.7
use_augmentation="--use_augmentation"
use_triple_align="--use_triple_align"
fusion_mode="gated"

# Create results directory
log_path="./Results/TS-CVA/${data_path}/"
mkdir -p $log_path

echo "========================================="
echo "TS-CVA Training for ${data_path}"
echo "========================================="
echo ""

# =====================================================
# Prediction length: 96
# =====================================================
pred_len=96
learning_rate=1e-4
channel=32
e_layer=2
d_layer=1
dropout_n=0.2
d_vector=320

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_cw${contrastive_weight}.log"

echo "Starting training: pred_len=${pred_len}"
nohup python train_tscva.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 100 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --d_vector $d_vector \
  --contrastive_weight $contrastive_weight \
  --forecast_weight $forecast_weight \
  --fusion_mode $fusion_mode \
  $use_augmentation \
  $use_triple_align > $log_file 2>&1 &

echo "  Job started. Logging to $log_file"
sleep 2

# =====================================================
# Prediction length: 192
# =====================================================
pred_len=192
learning_rate=1e-4
channel=32
e_layer=2
d_layer=1
dropout_n=0.2

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_cw${contrastive_weight}.log"

echo "Starting training: pred_len=${pred_len}"
nohup python train_tscva.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 100 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --d_vector $d_vector \
  --contrastive_weight $contrastive_weight \
  --forecast_weight $forecast_weight \
  --fusion_mode $fusion_mode \
  $use_augmentation \
  $use_triple_align > $log_file 2>&1 &

echo "  Job started. Logging to $log_file"
sleep 2

# =====================================================
# Prediction length: 336
# =====================================================
pred_len=336
learning_rate=1e-4
channel=32
e_layer=2
d_layer=1
dropout_n=0.3

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_cw${contrastive_weight}.log"

echo "Starting training: pred_len=${pred_len}"
nohup python train_tscva.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 100 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --d_vector $d_vector \
  --contrastive_weight $contrastive_weight \
  --forecast_weight $forecast_weight \
  --fusion_mode $fusion_mode \
  $use_augmentation \
  $use_triple_align > $log_file 2>&1 &

echo "  Job started. Logging to $log_file"
sleep 2

# =====================================================
# Prediction length: 720
# =====================================================
pred_len=720
learning_rate=1e-4
channel=64
e_layer=2
d_layer=1
dropout_n=0.3

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_cw${contrastive_weight}.log"

echo "Starting training: pred_len=${pred_len}"
nohup python train_tscva.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 100 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --d_vector $d_vector \
  --contrastive_weight $contrastive_weight \
  --forecast_weight $forecast_weight \
  --fusion_mode $fusion_mode \
  $use_augmentation \
  $use_triple_align > $log_file 2>&1 &

echo "  Job started. Logging to $log_file"

echo ""
echo "========================================="
echo "All training jobs started!"
echo "Monitor logs in: $log_path"
echo "========================================="
