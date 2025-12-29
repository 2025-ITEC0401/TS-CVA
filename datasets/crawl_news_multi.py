"""
Multi-Source News Crawler for Time Series Forecasting

This script downloads news headlines from multiple sources:
1. NewsAPI (newsapi.org) - Free tier: 100 requests/day
2. Finnhub (finnhub.io) - Free tier: 60 calls/min
3. GNews (gnews.io) - Free tier: 100 requests/day
4. Yahoo Finance (fallback)

Usage:
    python datasets/crawl_news_multi.py --preset tech
    python datasets/crawl_news_multi.py --symbols AAPL GOOGL MSFT --days 30
"""

import os
import json
import argparse
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load from project root .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use environment variables only

# ============================================================
# API Keys - Loaded from .env file or environment variables
# ============================================================
# Get free API keys from:
# - NewsAPI: https://newsapi.org/register
# - Finnhub: https://finnhub.io/register
# - GNews: https://gnews.io/

NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY', 'your_newsapi_key_here')
FINNHUB_KEY = os.environ.get('FINNHUB_KEY', 'your_finnhub_key_here')
GNEWS_KEY = os.environ.get('GNEWS_KEY', 'your_gnews_key_here')

# ============================================================
# Preset symbol groups
# ============================================================
PRESETS = {
    'sp500_top10': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ'],
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL'],
    'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'SCHW', 'USB'],
    'indices': ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX'],
    'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD'],
}

# Company name mapping for better search
COMPANY_NAMES = {
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL': 'Google Alphabet',
    'AMZN': 'Amazon',
    'META': 'Meta Facebook',
    'NVDA': 'NVIDIA',
    'AMD': 'AMD Advanced Micro Devices',
    'INTC': 'Intel',
    'CRM': 'Salesforce',
    'ORCL': 'Oracle',
    'TSLA': 'Tesla',
    'JPM': 'JPMorgan',
    'BAC': 'Bank of America',
    'WFC': 'Wells Fargo',
    'GS': 'Goldman Sachs',
}


class NewsAPIClient:
    """NewsAPI.org client - Best for general news coverage."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
    def get_news(self, query: str, days: int = 30, max_results: int = 100) -> List[Dict]:
        """Get news articles from NewsAPI."""
        if self.api_key == 'your_newsapi_key_here':
            return []
            
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/everything"
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': min(max_results, 100),
                'apiKey': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                return [{
                    'title': a.get('title', ''),
                    'summary': a.get('description', ''),
                    'publisher': a.get('source', {}).get('name', ''),
                    'published': a.get('publishedAt', ''),
                    'url': a.get('url', ''),
                    'source': 'newsapi'
                } for a in articles if a.get('title')]
            else:
                print(f"NewsAPI error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"NewsAPI error: {e}")
            return []


class FinnhubClient:
    """Finnhub.io client - Best for stock-specific news."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        
    def get_company_news(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get company news from Finnhub."""
        if self.api_key == 'your_finnhub_key_here':
            return []
            
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/company-news"
            params = {
                'symbol': symbol.replace('^', '').replace('-USD', ''),
                'from': from_date,
                'to': to_date,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                articles = response.json()
                
                return [{
                    'title': a.get('headline', ''),
                    'summary': a.get('summary', ''),
                    'publisher': a.get('source', ''),
                    'published': datetime.fromtimestamp(a.get('datetime', 0)).isoformat(),
                    'url': a.get('url', ''),
                    'source': 'finnhub'
                } for a in articles[:100] if a.get('headline')]
            else:
                print(f"Finnhub error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Finnhub error: {e}")
            return []
    
    def get_market_news(self, category: str = 'general') -> List[Dict]:
        """Get market news from Finnhub."""
        if self.api_key == 'your_finnhub_key_here':
            return []
            
        try:
            url = f"{self.base_url}/news"
            params = {
                'category': category,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                articles = response.json()
                
                return [{
                    'title': a.get('headline', ''),
                    'summary': a.get('summary', ''),
                    'publisher': a.get('source', ''),
                    'published': datetime.fromtimestamp(a.get('datetime', 0)).isoformat(),
                    'url': a.get('url', ''),
                    'source': 'finnhub'
                } for a in articles[:100] if a.get('headline')]
            else:
                return []
                
        except Exception as e:
            print(f"Finnhub market news error: {e}")
            return []


class GNewsClient:
    """GNews.io client - Good for international coverage."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://gnews.io/api/v4"
        
    def get_news(self, query: str, max_results: int = 100) -> List[Dict]:
        """Get news from GNews."""
        if self.api_key == 'your_gnews_key_here':
            return []
            
        try:
            url = f"{self.base_url}/search"
            params = {
                'q': query,
                'lang': 'en',
                'max': min(max_results, 100),
                'token': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                return [{
                    'title': a.get('title', ''),
                    'summary': a.get('description', ''),
                    'publisher': a.get('source', {}).get('name', ''),
                    'published': a.get('publishedAt', ''),
                    'url': a.get('url', ''),
                    'source': 'gnews'
                } for a in articles if a.get('title')]
            else:
                print(f"GNews error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"GNews error: {e}")
            return []


class YahooFinanceClient:
    """Yahoo Finance client - Fallback option."""
    
    def get_news(self, symbol: str, max_news: int = 100) -> List[Dict]:
        """Get news from Yahoo Finance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if news is None or len(news) == 0:
                return []
            
            processed_news = []
            for item in news[:max_news]:
                if 'content' in item:
                    content = item['content']
                    processed_news.append({
                        'title': content.get('title', ''),
                        'summary': content.get('summary', ''),
                        'publisher': content.get('provider', {}).get('displayName', ''),
                        'published': content.get('pubDate', ''),
                        'url': '',
                        'source': 'yahoo'
                    })
                else:
                    processed_news.append({
                        'title': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'publisher': item.get('publisher', ''),
                        'published': str(item.get('providerPublishTime', '')),
                        'url': item.get('link', ''),
                        'source': 'yahoo'
                    })
            
            return processed_news
            
        except Exception as e:
            print(f"Yahoo Finance error for {symbol}: {e}")
            return []


class MultiSourceNewsCrawler:
    """Aggregate news from multiple sources."""
    
    def __init__(self):
        self.newsapi = NewsAPIClient(NEWSAPI_KEY)
        self.finnhub = FinnhubClient(FINNHUB_KEY)
        self.gnews = GNewsClient(GNEWS_KEY)
        self.yahoo = YahooFinanceClient()
        
    def get_symbol_news(self, symbol: str, days: int = 30, max_per_source: int = 50) -> List[Dict]:
        """Get news for a symbol from all sources."""
        all_news = []
        
        # Get company name for search
        company_name = COMPANY_NAMES.get(symbol, symbol)
        search_query = f"{company_name} stock"
        
        # 1. Finnhub (best for stock-specific news)
        print(f"    [Finnhub]", end=' ')
        finnhub_news = self.finnhub.get_company_news(symbol, days)
        print(f"{len(finnhub_news)} articles")
        all_news.extend(finnhub_news[:max_per_source])
        time.sleep(0.5)  # Rate limiting
        
        # 2. NewsAPI
        print(f"    [NewsAPI]", end=' ')
        newsapi_news = self.newsapi.get_news(search_query, days, max_per_source)
        print(f"{len(newsapi_news)} articles")
        all_news.extend(newsapi_news)
        time.sleep(0.5)
        
        # 3. GNews
        print(f"    [GNews]", end=' ')
        gnews_news = self.gnews.get_news(search_query, max_per_source)
        print(f"{len(gnews_news)} articles")
        all_news.extend(gnews_news)
        time.sleep(0.5)
        
        # 4. Yahoo Finance (fallback)
        print(f"    [Yahoo]", end=' ')
        yahoo_news = self.yahoo.get_news(symbol, max_per_source)
        print(f"{len(yahoo_news)} articles")
        all_news.extend(yahoo_news)
        
        # Add symbol to all news
        for news in all_news:
            news['symbol'] = symbol
            
        return all_news
    
    def get_market_news(self, days: int = 30) -> List[Dict]:
        """Get general market news."""
        all_news = []
        
        # Finnhub market news
        print(f"    [Finnhub Market]", end=' ')
        finnhub_market = self.finnhub.get_market_news('general')
        print(f"{len(finnhub_market)} articles")
        all_news.extend(finnhub_market)
        
        # NewsAPI market news
        print(f"    [NewsAPI Market]", end=' ')
        newsapi_market = self.newsapi.get_news('stock market finance', days, 50)
        print(f"{len(newsapi_market)} articles")
        all_news.extend(newsapi_market)
        
        for news in all_news:
            news['symbol'] = 'MARKET'
            
        return all_news


def remove_duplicates(news_list: List[Dict]) -> List[Dict]:
    """Remove duplicate news based on title similarity."""
    seen_titles = set()
    unique_news = []
    
    for news in news_list:
        title = news.get('title', '').lower().strip()
        # Simple dedup based on first 50 chars
        title_key = title[:50]
        
        if title_key and title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_news.append(news)
            
    return unique_news


def crawl_all_news(symbols: List[str], save_dir: str, days: int = 30) -> Dict:
    """Crawl news for all symbols from multiple sources."""
    os.makedirs(save_dir, exist_ok=True)
    
    crawler = MultiSourceNewsCrawler()
    
    all_news = {
        'crawl_date': datetime.now().isoformat(),
        'symbols': symbols,
        'days_back': days,
        'sources': ['finnhub', 'newsapi', 'gnews', 'yahoo'],
        'news_by_symbol': {},
        'all_headlines': []
    }
    
    print(f"\nCrawling news for {len(symbols)} symbols (last {days} days)...")
    print("=" * 60)
    
    total_articles = 0
    
    for symbol in symbols:
        print(f"\n📰 {symbol} ({COMPANY_NAMES.get(symbol, symbol)}):")
        news = crawler.get_symbol_news(symbol, days)
        news = remove_duplicates(news)
        
        all_news['news_by_symbol'][symbol] = news
        total_articles += len(news)
        
        # Add to all headlines
        for item in news:
            if item['title']:
                all_news['all_headlines'].append(item['title'])
        
        print(f"    → Total: {len(news)} unique articles")
        time.sleep(1)  # Rate limiting between symbols
    
    # Get market news
    print(f"\n📰 MARKET (General):")
    market_news = crawler.get_market_news(days)
    market_news = remove_duplicates(market_news)
    all_news['news_by_symbol']['MARKET'] = market_news
    total_articles += len(market_news)
    
    for item in market_news:
        if item['title']:
            all_news['all_headlines'].append(item['title'])
    print(f"    → Total: {len(market_news)} unique articles")
    
    # Remove duplicate headlines
    all_news['all_headlines'] = list(set(all_news['all_headlines']))
    
    # Statistics by source
    source_stats = {}
    for sym_news in all_news['news_by_symbol'].values():
        for news in sym_news:
            source = news.get('source', 'unknown')
            source_stats[source] = source_stats.get(source, 0) + 1
    
    all_news['source_statistics'] = source_stats
    
    # Save
    news_file = os.path.join(save_dir, 'news.json')
    with open(news_file, 'w', encoding='utf-8') as f:
        json.dump(all_news, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"📊 Summary:")
    print(f"   Total articles collected: {total_articles}")
    print(f"   Unique headlines: {len(all_news['all_headlines'])}")
    print(f"   Articles by source:")
    for source, count in source_stats.items():
        print(f"      - {source}: {count}")
    print(f"\n   Saved to: {news_file}")
    print("=" * 60)
    
    return all_news


def main():
    parser = argparse.ArgumentParser(description='Multi-Source News Crawler')
    
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='List of ticker symbols')
    parser.add_argument('--preset', type=str, default=None,
                       choices=list(PRESETS.keys()),
                       help='Use preset symbol group')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to look back (default: 30)')
    parser.add_argument('--save-dir', type=str, default='./datasets/yahoo_finance',
                       help='Directory to save news')
    
    args = parser.parse_args()
    
    # Check API keys
    print("=" * 60)
    print("🔑 API Key Status:")
    print("=" * 60)
    
    apis_available = 0
    if NEWSAPI_KEY != 'your_newsapi_key_here':
        print("   ✅ NewsAPI: Configured")
        apis_available += 1
    else:
        print("   ❌ NewsAPI: Not configured (set NEWSAPI_KEY env var)")
        
    if FINNHUB_KEY != 'your_finnhub_key_here':
        print("   ✅ Finnhub: Configured")
        apis_available += 1
    else:
        print("   ❌ Finnhub: Not configured (set FINNHUB_KEY env var)")
        
    if GNEWS_KEY != 'your_gnews_key_here':
        print("   ✅ GNews: Configured")
        apis_available += 1
    else:
        print("   ❌ GNews: Not configured (set GNEWS_KEY env var)")
    
    print("   ✅ Yahoo Finance: Always available (fallback)")
    
    if apis_available == 0:
        print("\n⚠️  No external APIs configured. Using Yahoo Finance only.")
        print("   For more news, get free API keys from:")
        print("   - NewsAPI: https://newsapi.org/register")
        print("   - Finnhub: https://finnhub.io/register")
        print("   - GNews: https://gnews.io/")
    
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
    
    print("\n" + "=" * 60)
    print("📈 Multi-Source News Crawler")
    print("=" * 60)
    print(f"   Symbols: {symbols}")
    print(f"   Days back: {args.days}")
    print(f"   Save directory: {save_dir}")
    
    # Crawl news
    news_data = crawl_all_news(symbols, save_dir, args.days)
    
    # Show sample headlines
    print("\n" + "=" * 60)
    print("📰 Sample Headlines:")
    print("=" * 60)
    for headline in news_data['all_headlines'][:15]:
        print(f"   • {headline[:80]}...")
    
    print("\n✅ Crawling Complete!")


if __name__ == '__main__':
    main()
