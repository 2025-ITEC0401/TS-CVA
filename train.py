"""
TS-CVA Training Script

Usage:
    python train.py <dataset> <run_name> --loader UEA --epochs 100 --eval
    
Example:
    python train.py BasicMotions ts_cva_exp --loader UEA --epochs 100 --gpu 0 --eval
"""

import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime

from ts_cva import TSCVAWrapper
import datautils
import tasks
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout, string_save, list_save
from visualization import plot_loss_curves


class DualOutput:
    """Redirect stdout to both terminal and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class DualErrorOutput:
    """Redirect stderr to both terminal and file."""
    def __init__(self, filename):
        self.terminal = sys.stderr
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def save_checkpoint_callback(save_every=1, unit='epoch'):
    """Create callback to save checkpoints."""
    assert unit in ('epoch', 'iter')
    def callback(model, loss, val_loss=None):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback


def save_best_model_callback(monitor='val_loss', mode='min'):
    """Create callback to save best model."""
    best_value = None
    best_epoch = None
    
    def callback(model, loss, val_loss=None):
        nonlocal best_value, best_epoch
        current_value = val_loss if monitor == 'val_loss' and val_loss is not None else loss
        
        if best_value is None or \
           (mode == 'min' and current_value < best_value) or \
           (mode == 'max' and current_value > best_value):
            best_value = current_value
            best_epoch = model.n_epochs
            model.save(f'{run_dir}/model_best.pkl')
    
    return callback


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TS-CVA Training')
    
    # Dataset arguments
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name for saving outputs')
    parser.add_argument('--loader', type=str, required=True,
                        help='Data loader: UCR, UEA, forecast_csv, forecast_csv_univar')
    
    # Training arguments
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--iters', type=int, default=None, help='Number of iterations')
    parser.add_argument('--save-every', type=int, default=None, help='Save checkpoint frequency')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='Max threads')
    
    # Model arguments
    parser.add_argument('--repr-dims', type=int, default=320, help='Representation dimension')
    parser.add_argument('--hidden-dims', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=10, help='Encoder depth')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d-llm', type=int, default=768, help='LLM embedding dimension')
    parser.add_argument('--d-ff', type=int, default=256, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max-train-length', type=int, default=3000, help='Max training sequence length')
    
    # Cross-modal arguments
    parser.add_argument('--use-cross-modal', action='store_true', help='Enable cross-modal alignment')
    parser.add_argument('--cross-modal-weight', type=float, default=0.3, help='Cross-modal loss weight')
    parser.add_argument('--llm-embeddings', type=str, default=None, help='Path to LLM embeddings')
    
    # Other arguments
    parser.add_argument('--irregular', type=float, default=0, help='Missing data ratio')
    parser.add_argument('--eval', action='store_true', help='Evaluate after training')
    
    args = parser.parse_args()
    
    # Create run directory
    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup logging
    sys.stdout = DualOutput(f'{run_dir}/output.log')
    sys.stderr = DualErrorOutput(f'{run_dir}/error.log')
    
    print("=" * 60)
    print("TS-CVA Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Arguments: {args}")
    print(f"Run directory: {run_dir}")
    print("=" * 60)
    
    # Initialize device
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    # Load data
    print('Loading data...', end=' ')
    
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = \
            datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        test_data = data[:, valid_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = \
            datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        test_data = data[:, valid_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = \
            datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        test_data = data[:, valid_slice]
        
    else:
        raise ValueError(f"Unknown loader: {args.loader}")
    
    # Handle irregular data
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Irregular data not supported for {task_type}")
    
    print('done')
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Load LLM embeddings if provided
    train_llm_embeddings = None
    test_llm_embeddings = None
    
    if args.llm_embeddings and args.use_cross_modal:
        print(f'Loading LLM embeddings from {args.llm_embeddings}...', end=' ')
        embeddings_data = np.load(args.llm_embeddings, allow_pickle=True)
        if isinstance(embeddings_data, np.lib.npyio.NpzFile):
            train_llm_embeddings = embeddings_data['train']
            test_llm_embeddings = embeddings_data['test']
        else:
            # Assume single array, split according to data
            n_train = train_data.shape[0]
            train_llm_embeddings = embeddings_data[:n_train]
            test_llm_embeddings = embeddings_data[n_train:]
        print('done')
        print(f"Train LLM embeddings shape: {train_llm_embeddings.shape}")
    
    # Setup callbacks
    best_model_callback = save_best_model_callback(monitor='val_loss', mode='min')
    
    # Model configuration
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        num_heads=args.num_heads,
        d_llm=args.d_llm,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_train_length=args.max_train_length,
        use_cross_modal=args.use_cross_modal,
        cross_modal_weight=args.cross_modal_weight,
        after_epoch_callback=best_model_callback
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config['after_iter_callback'] = save_checkpoint_callback(args.save_every, unit)
    
    # Initialize model
    print("\nInitializing TS-CVA model...")
    t = time.time()
    
    model = TSCVAWrapper(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model._net.parameters()):,}")
    
    # Train
    print("\nStarting training...")
    total_loss_log, loss_log, val_loss_log = model.fit(
        train_data,
        test_data,
        train_llm_embeddings=train_llm_embeddings,
        test_llm_embeddings=test_llm_embeddings,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    
    # Save final model
    model.save(f'{run_dir}/model.pkl')
    
    # Save loss logs
    list_save(f'{run_dir}/total_loss.txt', total_loss_log)
    list_save(f'{run_dir}/loss.txt', loss_log)
    list_save(f'{run_dir}/val_loss.txt', val_loss_log)
    
    # Plot loss curves
    try:
        plot_loss_curves(f'{run_dir}/loss_curves', total_loss_log, loss_log, val_loss_log, cutoff=1)
    except Exception as e:
        print(f"Warning: Could not plot loss curves: {e}")
    
    training_time = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=int(training_time))}")
    
    # Evaluation
    if args.eval:
        print("\n" + "=" * 60)
        print("Evaluation")
        print("=" * 60)
        
        # Load best model
        model.load(f'{run_dir}/model_best.pkl')
        
        if task_type == 'classification':
            # Classification evaluation
            print("\nClassification evaluation...")
            out_classification, eval_res_classification = tasks.eval_classification(
                model, train_data, train_labels, test_data, test_labels, eval_protocol='svm'
            )
            
            os.makedirs(f'{run_dir}/classification', exist_ok=True)
            string_save(f'{run_dir}/classification/eval_res.txt', str(eval_res_classification))
            pkl_save(f'{run_dir}/classification/out.pkl', out_classification)
            pkl_save(f'{run_dir}/classification/eval_res.pkl', eval_res_classification)
            
            print("Classification results:")
            print(eval_res_classification)
            
            # Clustering evaluation
            print("\nClustering evaluation...")
            os.makedirs(f'{run_dir}/clustering', exist_ok=True)
            out_clustering, eval_res_clustering = tasks.eval_clustering(
                f'{run_dir}/clustering', model, test_data, test_labels
            )
            
            string_save(f'{run_dir}/clustering/eval_res.txt', str(eval_res_clustering))
            pkl_save(f'{run_dir}/clustering/out.pkl', out_clustering)
            pkl_save(f'{run_dir}/clustering/eval_res.pkl', eval_res_clustering)
            
            print("Clustering results:")
            print(eval_res_clustering)
    
    print("\n" + "=" * 60)
    print("Finished!")
    print("=" * 60)
