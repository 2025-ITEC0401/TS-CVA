"""
Crawl News from Yahoo Finance for Time Series Forecasting

This script downloads news headlines for stocks/indices from Yahoo Finance
and saves them for use with TS-CVA cross-modal learning.

Usage:
    python datasets/crawl_yahoo_news.py --symbols AAPL GOOGL MSFT
    python datasets/crawl_yahoo_news.py --preset tech
"""

import os
import json
import argparse
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

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
    'indices': ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX'],
    'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD'],
    'etf': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'EFA', 'EEM', 'GLD', 'TLT'],
}

# Human-readable names for indices
INDEX_NAMES = {
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones Industrial Average',
    '^IXIC': 'Nasdaq Composite',
    '^RUT': 'Russell 2000',
    '^VIX': 'CBOE Volatility Index'
}


def get_ticker_news(symbol: str, max_news: int = 100) -> List[Dict]:
    """
    Get recent news for a ticker symbol.
    
    Args:
        symbol: Ticker symbol
        max_news: Maximum number of news items
        
    Returns:
        List of news dictionaries
    """
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if news is None or len(news) == 0:
            return []
        
        processed_news = []
        for item in news[:max_news]:
            # Handle new yfinance news format
            if 'content' in item:
                content = item['content']
                processed_news.append({
                    'title': content.get('title', ''),
                    'summary': content.get('summary', ''),
                    'publisher': content.get('provider', {}).get('displayName', ''),
                    'published': content.get('pubDate', ''),
                    'symbol': symbol
                })
            else:
                # Old format fallback
                processed_news.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'publisher': item.get('publisher', ''),
                    'published': item.get('providerPublishTime', 0),
                    'symbol': symbol
                })
        
        return processed_news
    
    except Exception as e:
        print(f"Error getting news for {symbol}: {e}")
        return []


def get_market_news() -> List[Dict]:
    """Get general market news."""
    # Use SPY as proxy for market news
    return get_ticker_news('SPY', max_news=100)


def crawl_all_news(symbols: List[str], save_dir: str) -> Dict:
    """
    Crawl news for all symbols.
    
    Args:
        symbols: List of ticker symbols
        save_dir: Directory to save news
        
    Returns:
        Dictionary with all news
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_news = {
        'crawl_date': datetime.now().isoformat(),
        'symbols': symbols,
        'news_by_symbol': {},
        'all_headlines': []
    }
    
    print(f"Crawling news for {len(symbols)} symbols...")
    
    for symbol in symbols:
        print(f"  Getting news for {symbol}...", end=' ')
        news = get_ticker_news(symbol, max_news=100)
        all_news['news_by_symbol'][symbol] = news
        
        # Add to all headlines
        for item in news:
            if item['title']:
                all_news['all_headlines'].append(item['title'])
        
        print(f"{len(news)} articles")
        time.sleep(0.5)  # Rate limiting
    
    # Get market news
    print("  Getting market news...", end=' ')
    market_news = get_market_news()
    all_news['news_by_symbol']['MARKET'] = market_news
    for item in market_news:
        if item['title']:
            all_news['all_headlines'].append(item['title'])
    print(f"{len(market_news)} articles")
    
    # Remove duplicates from headlines
    all_news['all_headlines'] = list(set(all_news['all_headlines']))
    
    # Save
    news_file = os.path.join(save_dir, 'news.json')
    with open(news_file, 'w', encoding='utf-8') as f:
        json.dump(all_news, f, indent=2, ensure_ascii=False)
    
    print(f"\nTotal unique headlines: {len(all_news['all_headlines'])}")
    print(f"News saved to: {news_file}")
    
    return all_news


def create_news_context_for_samples(
    news_data: Dict,
    n_samples: int,
    symbols: List[str]
) -> List[Dict]:
    """
    Create news context for each sample in the dataset.
    
    Since we don't have historical news, we'll use current news as context.
    For real applications, you'd want to match news dates to sample dates.
    
    Args:
        news_data: Crawled news data
        n_samples: Number of samples in dataset
        symbols: List of symbols in order
        
    Returns:
        List of external_info dicts for each sample
    """
    external_info_list = []
    
    # Get all headlines
    all_headlines = news_data.get('all_headlines', [])
    
    # Create context for each sample
    for i in range(n_samples):
        # Rotate through headlines for variety
        start_idx = (i * 3) % max(1, len(all_headlines) - 3)
        sample_headlines = all_headlines[start_idx:start_idx + 3] if all_headlines else []
        
        # Get symbol-specific news
        symbol_news = []
        for sym in symbols[:3]:  # Top 3 symbols
            sym_headlines = news_data.get('news_by_symbol', {}).get(sym, [])
            if sym_headlines:
                symbol_news.append(sym_headlines[i % len(sym_headlines)].get('title', ''))
        
        # Determine sentiment from headlines
        positive_words = ['rally', 'surge', 'gain', 'up', 'high', 'record', 'beat', 'growth', 'rise']
        negative_words = ['fall', 'drop', 'down', 'low', 'crash', 'loss', 'miss', 'decline', 'fear']
        
        all_text = ' '.join(sample_headlines + symbol_news).lower()
        pos_count = sum(1 for w in positive_words if w in all_text)
        neg_count = sum(1 for w in negative_words if w in all_text)
        
        if pos_count > neg_count:
            sentiment = 'bullish'
        elif neg_count > pos_count:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        external_info = {
            'news': sample_headlines[:3] if sample_headlines else ['Market trading normally'],
            'sentiment': sentiment,
            'events': symbol_news[:2] if symbol_news else [],
            'context': f'Financial market data with {len(symbols)} assets'
        }
        
        external_info_list.append(external_info)
    
    return external_info_list


def main():
    parser = argparse.ArgumentParser(description='Crawl Yahoo Finance news')
    
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='List of ticker symbols')
    parser.add_argument('--preset', type=str, default=None,
                       choices=list(PRESETS.keys()),
                       help='Use preset symbol group')
    parser.add_argument('--save-dir', type=str, default='./datasets/yahoo_finance',
                       help='Directory to save news')
    
    args = parser.parse_args()
    
    # Determine symbols
    if args.preset:
        symbols = PRESETS[args.preset]
        save_name = args.preset
    elif args.symbols:
        symbols = args.symbols
        save_name = '_'.join(symbols[:3])
    else:
        symbols = PRESETS['tech']
        save_name = 'tech'
    
    save_dir = os.path.join(args.save_dir, save_name)
    
    print("=" * 60)
    print("Yahoo Finance News Crawler")
    print("=" * 60)
    print(f"Symbols: {symbols}")
    print("=" * 60)
    
    # Crawl news
    news_data = crawl_all_news(symbols, save_dir)
    
    print("\n" + "=" * 60)
    print("Sample Headlines:")
    print("=" * 60)
    for headline in news_data['all_headlines'][:10]:
        print(f"  • {headline}")
    
    print("\n" + "=" * 60)
    print("Crawling Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
