"""
TS-CVA Import Test
"""

import sys
sys.path.insert(0, 'TS2Vec')

print("Testing imports...")

try:
    from models import TSCVA, TSCVAEncoder
    print("✓ TSCVA, TSCVAEncoder imported")
except Exception as e:
    print(f"✗ TSCVA import failed: {e}")

try:
    from models.losses import hierarchical_contrastive_loss
    print("✓ hierarchical_contrastive_loss imported")
except Exception as e:
    print(f"✗ losses import failed: {e}")

try:
    from ts_cva import TSCVAWrapper
    print("✓ TSCVAWrapper imported")
except Exception as e:
    print(f"✗ TSCVAWrapper import failed: {e}")

try:
    from TS2Vec import datautils
    print("✓ datautils imported")
except Exception as e:
    print(f"✗ datautils import failed: {e}")

print("\n--- Testing model creation ---")

import torch

try:
    model = TSCVA(
        input_dims=9,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        use_cross_modal=False
    )
    print("✓ TSCVA model created (without cross-modal)")
    
    # Test forward
    x = torch.randn(4, 100, 9)
    out = model(x)
    print(f"✓ Forward pass: input {x.shape} -> output {out.shape}")
    
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Testing with cross-modal ---")

try:
    model_cm = TSCVA(
        input_dims=9,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        use_cross_modal=True,
        d_llm=768
    )
    print("✓ TSCVA model created (with cross-modal)")
    
    # Test forward with LLM embeddings
    x = torch.randn(4, 100, 9)
    llm_emb = torch.randn(4, 10, 768)
    out = model_cm(x, llm_emb)
    print(f"✓ Forward pass with LLM: input {x.shape}, llm {llm_emb.shape} -> output {out.shape}")
    
except Exception as e:
    print(f"✗ Cross-modal model failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== All tests completed ===")
