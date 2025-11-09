#!/bin/bash
# Find free GPU with lowest memory usage
# Usage: CUDA_ID=$(bash find_free_gpu.sh)

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
  awk -F', ' '{print $2, $1}' | \
  sort -n | \
  head -1 | \
  awk '{print $2}'
