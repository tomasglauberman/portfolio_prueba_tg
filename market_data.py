import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import streamlit as st

class MarketDataFetcher:
    def __init__(self):
        """Initialize market data fetcher"""
        pass
        
    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def fetch_ticker_data(_self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch market data for a single ticker"""
        try:
            # Add buffer days to ensure we have data
            start_date = start_date - timedelta(days=10)
            end_date = end_date + timedelta(days=1)
            
            # Fetch data from yfinance
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
            
            if data.empty:
                print(f"Warning: No data found for ticker {ticker}")
                return None
            
            # Ensure we have the required columns and handle different yfinance response formats
            if 'Adj Close' not in data.columns:
                if 'Close' in data.columns:
                    data['Adj Close'] = data['Close']
                else:
                    print(f"Warning: No close price data for {ticker}")
                    return None
            
            # Ensure we have basic OHLC data
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_basic_cols = [col for col in required_columns if col not in data.columns]
            if missing_basic_cols:
                print(f"Warning: Missing columns for {ticker}: {missing_basic_cols}")
                # Create minimal required columns if missing
                for col in missing_basic_cols:
                    if col == 'Volume':
                        data[col] = 1000000  # Default volume
                    else:
                        data[col] = data.get('Adj Close', data.get('Close', 0))
            
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Add small delay to respect API limits
            time.sleep(0.1)
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def fetch_benchmark_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch benchmark data (SPY, QQQ)"""
        benchmarks = ['SPY', 'QQQ']
        benchmark_data = {}
        
        for benchmark in benchmarks:
            data = self.fetch_ticker_data(benchmark, start_date, end_date)
            if data is not None:
                benchmark_data[benchmark] = data
        
        return benchmark_data
    
    def fetch_crypto_data(self, crypto_tickers: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch cryptocurrency data"""
        crypto_data = {}
        
        for ticker in crypto_tickers:
            # Ensure proper crypto ticker format
            if not ticker.endswith('-USD'):
                if ticker in ['BTC', 'ETH', 'SOL']:
                    ticker = f"{ticker}-USD"
                else:
                    continue
            
            data = self.fetch_ticker_data(ticker, start_date, end_date)
            if data is not None:
                crypto_data[ticker] = data
        
        return crypto_data
    
    def fetch_equity_data(self, equity_tickers: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch equity/ETF data"""
        equity_data = {}
        
        for ticker in equity_tickers:
            data = self.fetch_ticker_data(ticker, start_date, end_date)
            if data is not None:
                equity_data[ticker] = data
        
        return equity_data
    
    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def fetch_all_data(_self, tickers: List[str], date_range: Tuple[datetime, datetime]) -> Dict[str, pd.DataFrame]:
        """Fetch all required market data"""
        start_date, end_date = date_range
        
        # Extend end date to today if needed
        today = datetime.now()
        if end_date < today:
            end_date = today
        
        all_data = {}
        
        # Separate tickers by type
        crypto_tickers = [t for t in tickers if any(crypto in t for crypto in ['BTC', 'ETH', 'SOL'])]
        equity_tickers = [t for t in tickers if t not in crypto_tickers]
        
        # Fetch equity data
        if equity_tickers:
            equity_data = _self.fetch_equity_data(equity_tickers, start_date, end_date)
            all_data.update(equity_data)
        
        # Fetch crypto data
        if crypto_tickers:
            crypto_data = _self.fetch_crypto_data(crypto_tickers, start_date, end_date)
            all_data.update(crypto_data)
        
        # Fetch benchmark data
        benchmark_data = _self.fetch_benchmark_data(start_date, end_date)
        all_data.update(benchmark_data)
        
        return all_data
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get latest available price for a ticker"""
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="1d")
            
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
            
        except Exception as e:
            print(f"Error getting latest price for {ticker}: {str(e)}")
            return None
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate if ticker exists and has data"""
        try:
            data = self.fetch_ticker_data(ticker, datetime.now() - timedelta(days=30), datetime.now())
            return data is not None and not data.empty
        except:
            return False
    
    def get_market_calendar(self, start_date: datetime, end_date: datetime) -> pd.DatetimeIndex:
        """Get market calendar (union of equity and crypto trading days)"""
        # For simplicity, return all days (crypto trades 24/7, equity has weekends/holidays)
        # In a production system, you'd want to use a proper market calendar
        return pd.date_range(start_date, end_date, freq='D')
    
    def align_data_to_calendar(self, data: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        """Align all data to a common calendar"""
        aligned_data = {}
        
        for ticker, df in data.items():
            if df is not None and not df.empty:
                # Reindex to calendar, forward filling only for non-trading gaps
                aligned_df = df.reindex(calendar, method='ffill')
                
                # Don't forward fill beyond reasonable gaps (e.g., more than 7 days for equities)
                max_gap_days = 7 if not any(crypto in ticker for crypto in ['BTC', 'ETH', 'SOL']) else 1
                
                # Remove forward fills that are too far from actual data
                for col in aligned_df.columns:
                    mask = aligned_df[col].notna()
                    has_data = bool(mask.any())
                    if has_data:
                        # Find gaps larger than max_gap_days
                        gaps = (~mask).astype(int).groupby(mask.cumsum()).cumsum()
                        aligned_df.loc[gaps > max_gap_days, col] = None
                
                aligned_data[ticker] = aligned_df
        
        return aligned_data
    
    def get_trading_calendar_for_ticker(self, ticker: str) -> str:
        """Get appropriate trading calendar for ticker"""
        crypto_tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        
        if ticker in crypto_tickers:
            return 'crypto'  # 24/7 trading
        else:
            return 'equity'  # Standard market hours
    
    def clear_cache(self):
        """Clear the data cache"""
        # Cache is handled by Streamlit - this method kept for compatibility
        pass
