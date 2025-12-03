"""
Store LLM Embeddings for UEA/UCR datasets

This script generates and saves GPT-2 embeddings for time series data.
The embeddings are used for cross-modal alignment in TS-CVA.

Usage:
    python storage/store_emb_uea.py --dataset BasicMotions --loader UEA --gpu 0
"""

import torch
import os
import sys
import time
import h5py
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.gen_prompt_emb_uea import GenPromptEmbUEA
import datautils


def parse_args():
    parser = argparse.ArgumentParser(description='Store LLM embeddings for UEA/UCR datasets')
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., BasicMotions)")
    parser.add_argument("--loader", type=str, default="UEA", choices=["UEA", "UCR"], help="Data loader type")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for embedding generation")
    parser.add_argument("--d_model", type=int, default=768, help="GPT-2 hidden dimension")
    parser.add_argument("--model_name", type=str, default="gpt2", help="LLM model name")
    parser.add_argument("--save_dir", type=str, default="./Embeddings", help="Directory to save embeddings")
    parser.add_argument("--fixed_token_len", type=int, default=64, help="Fixed token length for embeddings")
    return parser.parse_args()


def store_embeddings(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading {args.loader} dataset: {args.dataset}")
    if args.loader == 'UEA':
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
    else:
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
    
    print(f"Train data shape: {train_data.shape}")  # [N, T, C]
    print(f"Test data shape: {test_data.shape}")
    
    # Initialize embedding generator
    seq_len = train_data.shape[1]
    n_features = train_data.shape[2]
    
    gen_prompt_emb = GenPromptEmbUEA(
        device=device,
        input_len=seq_len,
        n_features=n_features,
        dataset_name=args.dataset,
        model_name=args.model_name,
        d_model=args.d_model,
        fixed_token_len=args.fixed_token_len
    ).to(device)
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, f"{args.dataset}_{args.loader}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Process train data
    print("\nGenerating train embeddings...")
    train_embeddings = []
    t1 = time.time()
    
    for i in range(0, len(train_data), args.batch_size):
        batch = train_data[i:i+args.batch_size]
        batch_tensor = torch.from_numpy(batch).float().to(device)
        
        with torch.no_grad():
            emb = gen_prompt_emb.generate_embeddings(batch_tensor)
        train_embeddings.append(emb.cpu().numpy())
        
        print(f"  Processing {min(i+args.batch_size, len(train_data))}/{len(train_data)}", end='\r')
    
    train_embeddings = np.concatenate(train_embeddings, axis=0)
    print(f"\nTrain embeddings shape: {train_embeddings.shape}")
    
    # Process test data
    print("\nGenerating test embeddings...")
    test_embeddings = []
    
    for i in range(0, len(test_data), args.batch_size):
        batch = test_data[i:i+args.batch_size]
        batch_tensor = torch.from_numpy(batch).float().to(device)
        
        with torch.no_grad():
            emb = gen_prompt_emb.generate_embeddings(batch_tensor)
        test_embeddings.append(emb.cpu().numpy())
        
        print(f"  Processing {min(i+args.batch_size, len(test_data))}/{len(test_data)}", end='\r')
    
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    print(f"\nTest embeddings shape: {test_embeddings.shape}")
    
    t2 = time.time()
    print(f"\nTotal embedding generation time: {t2-t1:.2f}s")
    
    # Save embeddings
    save_path = os.path.join(save_dir, "embeddings.npz")
    np.savez(
        save_path,
        train=train_embeddings,
        test=test_embeddings,
        train_labels=train_labels,
        test_labels=test_labels
    )
    print(f"Embeddings saved to: {save_path}")
    
    # Also save as h5 for compatibility
    h5_path = os.path.join(save_dir, "embeddings.h5")
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('train', data=train_embeddings)
        hf.create_dataset('test', data=test_embeddings)
        hf.create_dataset('train_labels', data=train_labels)
        hf.create_dataset('test_labels', data=test_labels)
    print(f"H5 file saved to: {h5_path}")
    
    return train_embeddings, test_embeddings


if __name__ == "__main__":
    args = parse_args()
    store_embeddings(args)
