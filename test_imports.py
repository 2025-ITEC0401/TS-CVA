"""
TS-CVA Import Test Script
"""
import sys
import os

print("=" * 60)
print("TS-CVA Import Test")
print("=" * 60)

# Test PyTorch
print("\n[1] Testing PyTorch...")
import torch
print(f"    PyTorch version: {torch.__version__}")
print(f"    CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    CUDA version: {torch.version.cuda}")
    print(f"    GPU: {torch.cuda.get_device_name(0)}")

# Test NumPy/SciPy
print("\n[2] Testing NumPy/SciPy...")
import numpy as np
import scipy
print(f"    NumPy version: {np.__version__}")
print(f"    SciPy version: {scipy.__version__}")

# Test Transformers (for LLM)
print("\n[3] Testing Transformers...")
import transformers
print(f"    Transformers version: {transformers.__version__}")

# Test einops
print("\n[4] Testing einops...")
import einops
print(f"    einops version: {einops.__version__}")

# Test TS-CVA Models
print("\n[5] Testing TS-CVA Models...")
from models import TSCVA, TSCVAEncoder
from models.losses import hierarchical_contrastive_loss
print(f"    TSCVA: {TSCVA}")
print(f"    TSCVAEncoder: {TSCVAEncoder}")

# Test TS-CVA Wrapper
print("\n[6] Testing TS-CVA Wrapper...")
from ts_cva import TSCVAWrapper
print(f"    TSCVAWrapper: {TSCVAWrapper}")

# Test datautils
print("\n[7] Testing datautils...")
import datautils
print(f"    datautils: {datautils}")

# Test creating a model
print("\n[8] Testing model creation...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TSCVA(
    input_dims=9,
    output_dims=320,
    hidden_dims=64,
    depth=10,
    use_cross_modal=True
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    Model created on: {device}")
print(f"    Total parameters: {total_params:,}")
print(f"    Trainable parameters: {trainable_params:,}")

# Test forward pass
print("\n[9] Testing forward pass...")
batch_size = 4
seq_len = 100
n_features = 9
d_llm = 768
n_tokens = 10

x = torch.randn(batch_size, seq_len, n_features).to(device)
llm_emb = torch.randn(batch_size, n_tokens, d_llm).to(device)

with torch.no_grad():
    # Without LLM embeddings
    out1 = model(x)
    print(f"    Output shape (without LLM): {out1.shape}")
    
    # With LLM embeddings
    out2 = model(x, llm_emb)
    print(f"    Output shape (with LLM): {out2.shape}")

print("\n" + "=" * 60)
print("All tests passed! TS-CVA is ready to use.")
print("=" * 60)
