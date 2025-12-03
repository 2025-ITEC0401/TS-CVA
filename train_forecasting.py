"""
TS-CVA Forecasting Training Script for Yahoo Finance Data

This script trains TS-CVA representations and evaluates forecasting performance.

Usage:
    python train_forecasting.py tech --epochs 100 --gpu 0 --eval
    python train_forecasting.py indices --epochs 100 --use-cross-modal --eval
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import time
import datetime

from ts_cva import TSCVAWrapper
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, string_save, list_save


class ForecastingHead(nn.Module):
    """Linear forecasting head on top of TS-CVA representations."""
    
    def __init__(self, repr_dim, seq_len, pred_len, n_features):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        
        # Simple linear projection from representations to predictions
        self.fc = nn.Linear(repr_dim * seq_len, pred_len * n_features)
    
    def forward(self, repr):
        # repr: [B, T, D]
        B = repr.shape[0]
        repr_flat = repr.reshape(B, -1)  # [B, T*D]
        pred = self.fc(repr_flat)  # [B, pred_len * n_features]
        pred = pred.reshape(B, self.pred_len, self.n_features)
        return pred


def train_forecasting(
    model,
    X_train, y_train,
    X_val, y_val,
    repr_dim,
    pred_len,
    n_features,
    device,
    epochs=100,
    batch_size=32,
    lr=0.001
):
    """
    Train forecasting head on top of frozen TS-CVA representations.
    """
    seq_len = X_train.shape[1]
    
    # Create forecasting head
    head = ForecastingHead(repr_dim, seq_len, pred_len, n_features).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Pre-compute representations
    print("Computing representations...")
    with torch.no_grad():
        train_repr = model.encode(X_train, batch_size=batch_size)  # [N, T, D]
        val_repr = model.encode(X_val, batch_size=batch_size)
    
    train_repr = torch.from_numpy(train_repr).float().to(device)
    val_repr = torch.from_numpy(val_repr).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_repr, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        head.train()
        epoch_loss = 0
        for batch_repr, batch_y in train_loader:
            optimizer.zero_grad()
            pred = head(batch_repr)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        head.eval()
        with torch.no_grad():
            val_pred = head(val_repr)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = head.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}")
    
    # Load best model
    head.load_state_dict(best_state)
    
    return head, train_losses, val_losses


def evaluate_forecasting(model, head, X_test, y_test, scaler, device, batch_size=32, symbols=None):
    """
    Evaluate forecasting performance with comprehensive metrics.
    """
    # Compute representations
    with torch.no_grad():
        test_repr = model.encode(X_test, batch_size=batch_size)
    
    test_repr = torch.from_numpy(test_repr).float().to(device)
    
    head.eval()
    with torch.no_grad():
        pred = head(test_repr).cpu().numpy()
    
    # Calculate metrics on normalized data
    mse_norm = np.mean((pred - y_test) ** 2)
    mae_norm = np.mean(np.abs(pred - y_test))
    
    # Inverse transform if scaler available
    if scaler is not None:
        # Reshape for inverse transform
        pred_shape = pred.shape
        pred_flat = pred.reshape(-1, pred.shape[-1])
        y_test_flat = y_test.reshape(-1, y_test.shape[-1])
        
        pred_inv = scaler.inverse_transform(pred_flat).reshape(pred_shape)
        y_test_inv = scaler.inverse_transform(y_test_flat).reshape(pred_shape)
        
        mse_raw = np.mean((pred_inv - y_test_inv) ** 2)
        mae_raw = np.mean(np.abs(pred_inv - y_test_inv))
    else:
        mse_raw = mse_norm
        mae_raw = mae_norm
        pred_inv = pred
        y_test_inv = y_test
    
    # MAPE on raw prices (avoid division by zero)
    mape = np.mean(np.abs((y_test_inv - pred_inv) / (np.abs(y_test_inv) + 1e-8))) * 100
    
    # RMSE
    rmse_raw = np.sqrt(mse_raw)
    
    # Direction accuracy (predicting up/down over prediction horizon)
    pred_dir = (pred_inv[:, -1, :] - pred_inv[:, 0, :]) > 0
    target_dir = (y_test_inv[:, -1, :] - y_test_inv[:, 0, :]) > 0
    direction_acc = (pred_dir == target_dir).mean() * 100
    
    results = {
        'mse_norm': mse_norm,
        'mae_norm': mae_norm,
        'mse_raw': mse_raw,
        'mae_raw': mae_raw,
        'rmse_raw': rmse_raw,
        'mape': mape,
        'direction_acc': direction_acc,
        'predictions': pred_inv,
        'targets': y_test_inv
    }
    
    # Per-symbol analysis if symbols provided
    if symbols is not None:
        results['per_symbol'] = {}
        for i, sym in enumerate(symbols):
            sym_pred = pred_inv[:, :, i]
            sym_target = y_test_inv[:, :, i]
            
            sym_mape = np.mean(np.abs((sym_target - sym_pred) / (np.abs(sym_target) + 1e-8))) * 100
            sym_mae = np.mean(np.abs(sym_pred - sym_target))
            sym_rmse = np.sqrt(np.mean((sym_pred - sym_target) ** 2))
            
            # Per-symbol direction accuracy
            sym_pred_dir = (sym_pred[:, -1] - sym_pred[:, 0]) > 0
            sym_target_dir = (sym_target[:, -1] - sym_target[:, 0]) > 0
            sym_dir_acc = (sym_pred_dir == sym_target_dir).mean() * 100
            
            results['per_symbol'][sym] = {
                'mape': sym_mape,
                'mae': sym_mae,
                'rmse': sym_rmse,
                'direction_acc': sym_dir_acc
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='TS-CVA Forecasting Training')
    
    # Dataset arguments
    parser.add_argument('dataset', help='Yahoo Finance dataset name (tech, indices, etc.)')
    parser.add_argument('--run-name', type=str, default='forecasting', help='Run name')
    
    # Training arguments
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--repr-epochs', type=int, default=100, help='Epochs for representation learning')
    parser.add_argument('--forecast-epochs', type=int, default=100, help='Epochs for forecasting head')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model arguments
    parser.add_argument('--repr-dims', type=int, default=320, help='Representation dimension')
    parser.add_argument('--hidden-dims', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=10, help='Encoder depth')
    
    # Cross-modal arguments
    parser.add_argument('--use-cross-modal', action='store_true', help='Enable cross-modal alignment')
    parser.add_argument('--llm-embeddings', type=str, default=None, help='Path to LLM embeddings')
    
    # Evaluation
    parser.add_argument('--eval', action='store_true', help='Evaluate after training')
    parser.add_argument('--load-model', type=str, default=None, help='Load pre-trained model')
    
    args = parser.parse_args()
    
    # Create run directory
    run_dir = f'training/{args.dataset}__{name_with_datetime(args.run_name)}'
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 60)
    print("TS-CVA Forecasting Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Run directory: {run_dir}")
    print("=" * 60)
    
    # Initialize device
    device = init_dl_program(args.gpu, seed=args.seed)
    
    # Load data
    print('\nLoading Yahoo Finance data...')
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = \
        datautils.load_yahoo_finance_forecasting(args.dataset)
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    seq_len = X_train.shape[1]
    pred_len = y_train.shape[1]
    n_features = X_train.shape[2]
    
    print(f"\nSequence length: {seq_len}, Prediction length: {pred_len}, Features: {n_features}")
    
    # Load LLM embeddings if using cross-modal
    train_llm_emb = None
    val_llm_emb = None
    
    if args.use_cross_modal:
        llm_path = args.llm_embeddings
        if llm_path is None:
            # Try default path
            llm_path = f'datasets/yahoo_finance/{args.dataset}/llm_embeddings.npz'
        
        if os.path.exists(llm_path):
            print(f"\nLoading LLM embeddings from {llm_path}...")
            llm_data = np.load(llm_path)
            train_llm_emb = llm_data['train']
            val_llm_emb = llm_data['val']
            print(f"Train LLM embeddings: {train_llm_emb.shape}")
            print(f"Val LLM embeddings: {val_llm_emb.shape}")
        else:
            print(f"\nWarning: LLM embeddings not found at {llm_path}")
            print("Run: python storage/store_emb_yahoo.py --dataset {args.dataset}")
    
    # Load or train TS-CVA model
    if args.load_model:
        print(f"\nLoading pre-trained model from {args.load_model}...")
        model = TSCVAWrapper(
            input_dims=n_features,
            output_dims=args.repr_dims,
            hidden_dims=args.hidden_dims,
            depth=args.depth,
            device=device,
            use_cross_modal=args.use_cross_modal
        )
        model.load(args.load_model)
    else:
        print("\nTraining TS-CVA representations...")
        
        # Combine train and val for representation learning
        repr_data = np.concatenate([X_train, X_val], axis=0)
        
        # Combine LLM embeddings too if available
        repr_llm_emb = None
        if train_llm_emb is not None and val_llm_emb is not None:
            repr_llm_emb = np.concatenate([train_llm_emb, val_llm_emb], axis=0)
        
        model = TSCVAWrapper(
            input_dims=n_features,
            output_dims=args.repr_dims,
            hidden_dims=args.hidden_dims,
            depth=args.depth,
            device=device,
            batch_size=args.batch_size,
            lr=args.lr,
            use_cross_modal=args.use_cross_modal
        )
        
        # Train representation
        t = time.time()
        model.fit(repr_data, train_llm_embeddings=repr_llm_emb, n_epochs=args.repr_epochs, verbose=True)
        repr_time = time.time() - t
        print(f"\nRepresentation learning time: {datetime.timedelta(seconds=int(repr_time))}")
        
        # Load best model
        print("\nLoading best model...")
        model.load_best()
        
        # Save best model
        model.save_best(f'{run_dir}/model_best.pkl')
        model.save(f'{run_dir}/model.pkl')
    
    # Train forecasting head
    print("\n" + "=" * 60)
    print("Training Forecasting Head")
    print("=" * 60)
    
    t = time.time()
    head, train_losses, val_losses = train_forecasting(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        repr_dim=args.repr_dims,
        pred_len=pred_len,
        n_features=n_features,
        device=device,
        epochs=args.forecast_epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    forecast_time = time.time() - t
    print(f"\nForecasting training time: {datetime.timedelta(seconds=int(forecast_time))}")
    
    # Save forecasting head
    torch.save(head.state_dict(), f'{run_dir}/forecast_head.pt')
    
    # Save losses
    list_save(f'{run_dir}/forecast_train_loss.txt', train_losses)
    list_save(f'{run_dir}/forecast_val_loss.txt', val_losses)
    
    # Get symbol names for per-symbol analysis
    symbols = None
    symbols_path = f'datasets/yahoo_finance/{args.dataset}/symbols.txt'
    if os.path.exists(symbols_path):
        with open(symbols_path, 'r') as f:
            symbols = [line.strip() for line in f.readlines()]
    
    # Evaluate
    if args.eval:
        print("\n" + "=" * 60)
        print("Evaluation")
        print("=" * 60)
        
        results = evaluate_forecasting(model, head, X_test, y_test, scaler, device, args.batch_size, symbols)
        
        print(f"\n" + "-" * 60)
        print(f"전체 성능 (Overall Performance)")
        print(f"-" * 60)
        print(f"  RMSE (raw):          {results['rmse_raw']:.4f}")
        print(f"  MAE (raw):           {results['mae_raw']:.4f}")
        print(f"  MAPE:                {results['mape']:.2f}%")
        print(f"  Direction Accuracy:  {results['direction_acc']:.2f}%")
        print(f"")
        print(f"  MSE (normalized):    {results['mse_norm']:.6f}")
        print(f"  MAE (normalized):    {results['mae_norm']:.6f}")
        
        # Per-symbol analysis
        if 'per_symbol' in results:
            print(f"\n" + "-" * 60)
            print(f"종목별 성능 (Per-Symbol Performance)")
            print(f"-" * 60)
            print(f"{'Symbol':<8} {'MAPE':>8} {'MAE':>10} {'RMSE':>10} {'Dir Acc':>10}")
            print(f"-" * 60)
            for sym, metrics in results['per_symbol'].items():
                print(f"{sym:<8} {metrics['mape']:>7.2f}% {metrics['mae']:>10.2f} {metrics['rmse']:>10.2f} {metrics['direction_acc']:>9.1f}%")
        
        # Save results
        string_save(f'{run_dir}/eval_results.txt', str(results))
        np.savez(
            f'{run_dir}/predictions.npz',
            predictions=results['predictions'],
            targets=results['targets']
        )
        
        # Also save summary
        with open(f'{run_dir}/summary.txt', 'w') as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Sequence length: {seq_len}\n")
            f.write(f"Prediction length: {pred_len}\n")
            f.write(f"Features: {n_features}\n")
            f.write(f"Cross-modal: {args.use_cross_modal}\n")
            f.write(f"\n" + "=" * 50 + "\n")
            f.write(f"Overall Results:\n")
            f.write(f"  RMSE (raw): {results['rmse_raw']:.4f}\n")
            f.write(f"  MAE (raw): {results['mae_raw']:.4f}\n")
            f.write(f"  MAPE: {results['mape']:.2f}%\n")
            f.write(f"  Direction Accuracy: {results['direction_acc']:.2f}%\n")
            f.write(f"  MSE (normalized): {results['mse_norm']:.6f}\n")
            f.write(f"  MAE (normalized): {results['mae_norm']:.6f}\n")
            
            if 'per_symbol' in results:
                f.write(f"\n" + "=" * 50 + "\n")
                f.write(f"Per-Symbol Results:\n")
                for sym, metrics in results['per_symbol'].items():
                    f.write(f"  {sym}: MAPE={metrics['mape']:.2f}%, MAE={metrics['mae']:.2f}, Dir={metrics['direction_acc']:.1f}%\n")
    
    print("\n" + "=" * 60)
    print("Finished!")
    print("=" * 60)


if __name__ == '__main__':
    main()
