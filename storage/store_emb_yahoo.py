"""
Store LLM Embeddings for Yahoo Finance Forecasting with News Context

This script generates GPT-2 embeddings for Yahoo Finance time series data
incorporating news headlines as external context.

Usage:
    python storage/store_emb_yahoo.py --dataset tech --gpu 0
"""

import torch
import os
import sys
import time
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.gen_prompt_emb_extended import GenPromptEmbExtended
import datautils


def load_news(data_dir: str) -> dict:
    """Load news data from JSON file."""
    news_file = os.path.join(data_dir, 'news.json')
    if os.path.exists(news_file):
        with open(news_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def create_external_info_for_samples(news_data: dict, n_samples: int, symbols: list) -> list:
    """
    Create external info dictionaries for each sample.
    
    Args:
        news_data: Loaded news data
        n_samples: Number of samples
        symbols: List of symbol names
        
    Returns:
        List of external_info dicts
    """
    if news_data is None:
        return [None] * n_samples
    
    all_headlines = news_data.get('all_headlines', [])
    news_by_symbol = news_data.get('news_by_symbol', {})
    
    external_info_list = []
    
    for i in range(n_samples):
        # Get rotating headlines
        if all_headlines:
            start_idx = (i * 3) % max(1, len(all_headlines) - 3)
            sample_headlines = all_headlines[start_idx:start_idx + 3]
        else:
            sample_headlines = []
        
        # Determine sentiment
        positive_words = ['rally', 'surge', 'gain', 'up', 'high', 'record', 'beat', 'growth', 'rise', 'bullish', 'buy']
        negative_words = ['fall', 'drop', 'down', 'low', 'crash', 'loss', 'miss', 'decline', 'fear', 'bearish', 'sell']
        
        all_text = ' '.join(sample_headlines).lower()
        pos_count = sum(1 for w in positive_words if w in all_text)
        neg_count = sum(1 for w in negative_words if w in all_text)
        
        if pos_count > neg_count + 1:
            sentiment = 'strongly bullish'
        elif pos_count > neg_count:
            sentiment = 'bullish'
        elif neg_count > pos_count + 1:
            sentiment = 'strongly bearish'
        elif neg_count > pos_count:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        # Get events from market news
        market_news = news_by_symbol.get('MARKET', [])
        events = []
        if market_news:
            for j in range(min(2, len(market_news))):
                idx = (i + j) % len(market_news)
                if market_news[idx].get('title'):
                    events.append(market_news[idx]['title'][:100])  # Truncate long titles
        
        external_info = {
            'news': sample_headlines[:3] if sample_headlines else ['Market trading session'],
            'sentiment': sentiment,
            'events': events,
            'indicators': {
                'market_condition': 'normal',
                'volatility': 'moderate'
            }
        }
        
        external_info_list.append(external_info)
    
    return external_info_list


def parse_args():
    parser = argparse.ArgumentParser(description='Store LLM embeddings for Yahoo Finance')
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (tech, indices, etc.)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--d_model", type=int, default=768, help="GPT-2 hidden dimension")
    parser.add_argument("--fixed_token_len", type=int, default=128, help="Fixed token length")
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data directory
    data_dir = f'datasets/yahoo_finance/{args.dataset}'
    
    # Load forecasting data
    print(f"Loading Yahoo Finance data: {args.dataset}")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = \
        datautils.load_yahoo_finance_forecasting(args.dataset)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Load news
    print("Loading news data...")
    news_data = load_news(data_dir)
    if news_data:
        print(f"Loaded {len(news_data.get('all_headlines', []))} unique headlines")
    else:
        print("No news data found, using default prompts")
    
    # Feature names based on dataset
    if args.dataset == 'tech':
        feature_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL']
    elif args.dataset == 'indices':
        feature_names = ['S&P500', 'DowJones', 'Nasdaq', 'Russell2000', 'VIX']
    else:
        feature_names = [f'Asset_{i}' for i in range(X_train.shape[2])]
    
    # Initialize embedding generator
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    
    gen_emb = GenPromptEmbExtended(
        device=device,
        input_len=seq_len,
        n_features=n_features,
        dataset_name=f'Yahoo Finance {args.dataset}',
        domain='finance',
        feature_names=feature_names,
        fixed_token_len=args.fixed_token_len
    ).to(device)
    
    # Create external info for all samples
    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)
    
    print("\nCreating external info from news...")
    train_external = create_external_info_for_samples(news_data, n_train, feature_names)
    val_external = create_external_info_for_samples(news_data, n_val, feature_names)
    test_external = create_external_info_for_samples(news_data, n_test, feature_names)
    
    # Generate embeddings
    def generate_embeddings_batched(data, external_info_list, batch_size, desc):
        embeddings = []
        n = len(data)
        t0 = time.time()
        
        for i in range(0, n, batch_size):
            batch = data[i:i+batch_size]
            batch_ext = external_info_list[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            
            with torch.no_grad():
                emb = gen_emb.generate_embeddings(batch_tensor, external_info=batch_ext, mode='summary')
            embeddings.append(emb.cpu().numpy())
            
            print(f"  {desc}: {min(i+batch_size, n)}/{n}", end='\r')
        
        print(f"  {desc}: {n}/{n} - {time.time()-t0:.1f}s")
        return np.concatenate(embeddings, axis=0)
    
    print("\nGenerating embeddings...")
    train_emb = generate_embeddings_batched(X_train, train_external, args.batch_size, "Train")
    val_emb = generate_embeddings_batched(X_val, val_external, args.batch_size, "Val")
    test_emb = generate_embeddings_batched(X_test, test_external, args.batch_size, "Test")
    
    print(f"\nEmbedding shapes:")
    print(f"  Train: {train_emb.shape}")
    print(f"  Val: {val_emb.shape}")
    print(f"  Test: {test_emb.shape}")
    
    # Save embeddings
    save_path = os.path.join(data_dir, 'llm_embeddings.npz')
    np.savez(
        save_path,
        train=train_emb,
        val=val_emb,
        test=test_emb
    )
    print(f"\nEmbeddings saved to: {save_path}")


if __name__ == '__main__':
    main()
