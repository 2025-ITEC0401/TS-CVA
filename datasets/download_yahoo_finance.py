"""
Download Yahoo Finance Data for Time Series Forecasting

This script downloads stock/index data from Yahoo Finance
and prepares it for use in TS-CVA forecasting experiments.

Usage:
    python datasets/download_yahoo_finance.py --symbols AAPL GOOGL MSFT --period 5y
    python datasets/download_yahoo_finance.py --preset sp500_top10 --period 2y
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance'])
    import yfinance as yf


# Preset symbol groups
PRESETS = {
    'sp500_top10': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ'],
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL'],
    'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'SCHW', 'USB'],
    'indices': ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX'],  # S&P500, Dow, Nasdaq, Russell, VIX
    'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD'],
    'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X'],
    'commodities': ['GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F'],  # Gold, Silver, Oil, NatGas, Copper
    'etf': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'EFA', 'EEM', 'GLD', 'TLT'],
}


def download_data(
    symbols: List[str],
    period: str = '5y',
    interval: str = '1d',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Download data from Yahoo Finance.
    
    Args:
        symbols: List of ticker symbols
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with downloaded data
    """
    print(f"Downloading data for {len(symbols)} symbols...")
    print(f"Symbols: {symbols}")
    print(f"Period: {period}, Interval: {interval}")
    
    if start_date and end_date:
        data = yf.download(symbols, start=start_date, end=end_date, interval=interval, progress=True)
    else:
        data = yf.download(symbols, period=period, interval=interval, progress=True)
    
    return data


def prepare_forecasting_data(
    data: pd.DataFrame,
    feature_cols: List[str] = ['Close'],
    normalize: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    save_dir: str = './datasets/yahoo_finance'
) -> dict:
    """
    Prepare data for time series forecasting.
    
    Args:
        data: Raw Yahoo Finance DataFrame
        feature_cols: Columns to use (Close, Open, High, Low, Volume, Adj Close)
        normalize: Whether to normalize the data
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        save_dir: Directory to save processed data
        
    Returns:
        Dictionary with train/val/test data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract features
    if len(feature_cols) == 1:
        # Single feature - get all symbols
        if feature_cols[0] in data.columns.get_level_values(0):
            df = data[feature_cols[0]]
        else:
            df = data
    else:
        # Multiple features - stack them
        dfs = []
        for col in feature_cols:
            if col in data.columns.get_level_values(0):
                dfs.append(data[col])
        df = pd.concat(dfs, axis=1)
    
    # Drop NaN rows
    df = df.dropna()
    
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")
    
    # Convert to numpy array
    values = df.values.astype(np.float32)
    
    # Split data
    n = len(values)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = values[:train_end]
    val_data = values[train_end:val_end]
    test_data = values[val_end:]
    
    print(f"\nTrain: {train_data.shape}")
    print(f"Val: {val_data.shape}")
    print(f"Test: {test_data.shape}")
    
    # Normalize (fit on train, apply to all)
    if normalize:
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        
        train_data = (train_data - mean) / std
        val_data = (val_data - mean) / std
        test_data = (test_data - mean) / std
        
        # Save normalization parameters
        np.save(os.path.join(save_dir, 'mean.npy'), mean)
        np.save(os.path.join(save_dir, 'std.npy'), std)
    
    # Save data
    np.save(os.path.join(save_dir, 'train.npy'), train_data)
    np.save(os.path.join(save_dir, 'val.npy'), val_data)
    np.save(os.path.join(save_dir, 'test.npy'), test_data)
    
    # Save full data for TS2Vec format (T, N) -> needs to be (1, T, N) for batch
    full_data = np.concatenate([train_data, val_data, test_data], axis=0)
    np.save(os.path.join(save_dir, 'data.npy'), full_data)
    
    # Save as CSV too
    df.to_csv(os.path.join(save_dir, 'data.csv'))
    
    # Save metadata
    metadata = {
        'symbols': list(df.columns),
        'features': feature_cols,
        'n_samples': n,
        'n_features': df.shape[1],
        'train_end': train_end,
        'val_end': val_end,
        'date_range': f"{df.index[0]} to {df.index[-1]}",
        'normalized': normalize
    }
    
    import json
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nData saved to: {save_dir}")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'metadata': metadata
    }


def create_forecasting_dataset(
    data: pd.DataFrame,
    seq_len: int = 96,
    pred_len: int = 24,
    feature_cols: List[str] = ['Close'],
    normalize: bool = True,
    save_dir: str = './datasets/yahoo_finance'
) -> dict:
    """
    Create sliding window forecasting dataset.
    
    Args:
        data: Raw Yahoo Finance DataFrame
        seq_len: Input sequence length
        pred_len: Prediction length
        feature_cols: Columns to use
        normalize: Whether to normalize
        save_dir: Save directory
        
    Returns:
        Dictionary with X_train, y_train, etc.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract features
    if len(feature_cols) == 1 and feature_cols[0] in data.columns.get_level_values(0):
        df = data[feature_cols[0]]
    else:
        df = data
    
    df = df.dropna()
    values = df.values.astype(np.float32)
    
    print(f"\nData shape: {df.shape}")
    print(f"Creating sequences: input_len={seq_len}, pred_len={pred_len}")
    
    # Normalize
    if normalize:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std == 0] = 1
        values = (values - mean) / std
        np.save(os.path.join(save_dir, 'mean.npy'), mean)
        np.save(os.path.join(save_dir, 'std.npy'), std)
    
    # Create sequences
    X, y = [], []
    for i in range(len(values) - seq_len - pred_len + 1):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len:i+seq_len+pred_len])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"X shape: {X.shape}")  # (n_samples, seq_len, n_features)
    print(f"y shape: {y.shape}")  # (n_samples, pred_len, n_features)
    
    # Split
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"\nTrain: X={X_train.shape}, y={y_train.shape}")
    print(f"Val: X={X_val.shape}, y={y_val.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")
    
    # Save
    np.savez(
        os.path.join(save_dir, 'forecasting_data.npz'),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    
    print(f"\nData saved to: {save_dir}/forecasting_data.npz")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def main():
    parser = argparse.ArgumentParser(description='Download Yahoo Finance data')
    
    # Symbol selection
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='List of ticker symbols (e.g., AAPL GOOGL MSFT)')
    parser.add_argument('--preset', type=str, default=None,
                       choices=list(PRESETS.keys()),
                       help='Use preset symbol group')
    
    # Time range
    parser.add_argument('--period', type=str, default='5y',
                       help='Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--interval', type=str, default='1d',
                       help='Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    
    # Features
    parser.add_argument('--features', nargs='+', default=['Close'],
                       help='Features to use (Close, Open, High, Low, Volume, Adj Close)')
    
    # Forecasting settings
    parser.add_argument('--seq-len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--pred-len', type=int, default=24, help='Prediction length')
    parser.add_argument('--create-sequences', action='store_true',
                       help='Create sliding window sequences for forecasting')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='./datasets/yahoo_finance',
                       help='Directory to save data')
    parser.add_argument('--no-normalize', action='store_true', help='Do not normalize data')
    
    args = parser.parse_args()
    
    # Determine symbols
    if args.preset:
        symbols = PRESETS[args.preset]
        save_name = args.preset
    elif args.symbols:
        symbols = args.symbols
        save_name = '_'.join(symbols[:3]) if len(symbols) <= 3 else f"{len(symbols)}_stocks"
    else:
        # Default: top tech stocks
        symbols = PRESETS['tech']
        save_name = 'tech'
    
    # Update save directory
    save_dir = os.path.join(args.save_dir, save_name)
    
    print("=" * 60)
    print("Yahoo Finance Data Download")
    print("=" * 60)
    
    # Download data
    data = download_data(
        symbols=symbols,
        period=args.period,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end
    )
    
    if data.empty:
        print("Error: No data downloaded!")
        return
    
    # Process data
    if args.create_sequences:
        result = create_forecasting_dataset(
            data=data,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            feature_cols=args.features,
            normalize=not args.no_normalize,
            save_dir=save_dir
        )
    else:
        result = prepare_forecasting_data(
            data=data,
            feature_cols=args.features,
            normalize=not args.no_normalize,
            save_dir=save_dir
        )
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    
    # Print available presets
    print("\nAvailable presets:")
    for name, syms in PRESETS.items():
        print(f"  --preset {name}: {', '.join(syms[:5])}{'...' if len(syms) > 5 else ''}")


if __name__ == '__main__':
    main()
