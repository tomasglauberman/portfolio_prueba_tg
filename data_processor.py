import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        """Initialize data processor with validated CSV data"""
        self.df = self._process_dataframe(df.copy())
        self.trades_df = self._create_trades_dataframe()
        self.cash_flows_df = self._create_cash_flows_dataframe()
        
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the input dataframe"""
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Add signed quantity column
        df['Signed_Quantity'] = df.apply(
            lambda row: -row['Quantity'] if row['Buy_sell'] == 'SELL' else row['Quantity'],
            axis=1
        )
        
        # Add transaction value
        df['Transaction_Value'] = df['Price'] * df['Quantity']
        df['Signed_Transaction_Value'] = df['Price'] * df['Signed_Quantity']
        
        return df
    
    def _create_trades_dataframe(self) -> pd.DataFrame:
        """Create trades dataframe excluding cash flows"""
        return self.df[self.df['Ticker'] != 'CASH'].copy()
    
    def _create_cash_flows_dataframe(self) -> pd.DataFrame:
        """Create cash flows dataframe (deposits/withdrawals)"""
        cash_df = self.df[self.df['Ticker'] == 'CASH'].copy()
        if not cash_df.empty:
            # For cash flows, positive = deposit, negative = withdrawal
            cash_df['Cash_Flow'] = cash_df['Signed_Transaction_Value']
        return cash_df
    
    def get_unique_tickers(self) -> List[str]:
        """Get list of unique tickers excluding cash"""
        return self.trades_df['Ticker'].unique().tolist()
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Get the date range of all transactions"""
        return self.df['Date'].min(), self.df['Date'].max()
    
    def get_trades_for_ticker(self, ticker: str) -> pd.DataFrame:
        """Get all trades for a specific ticker"""
        return self.trades_df[self.trades_df['Ticker'] == ticker].copy()
    
    def get_cash_flows(self) -> pd.DataFrame:
        """Get all cash flows (deposits/withdrawals)"""
        return self.cash_flows_df.copy()
    
    def get_trades_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get trades within a specific date range"""
        mask = (self.trades_df['Date'] >= start_date) & (self.trades_df['Date'] <= end_date)
        return self.trades_df[mask].copy()
    
    def get_portfolio_tickers_on_date(self, date: datetime) -> List[str]:
        """Get list of tickers held in portfolio on a specific date"""
        trades_up_to_date = self.trades_df[self.trades_df['Date'] <= date]
        
        # Calculate net positions for each ticker
        net_positions = trades_up_to_date.groupby('Ticker')['Signed_Quantity'].sum()
        
        # Return tickers with non-zero positions
        return net_positions[net_positions != 0].index.tolist()
    
    def calculate_position_quantity(self, ticker: str, up_to_date: datetime) -> float:
        """Calculate net position quantity for a ticker up to a specific date"""
        ticker_trades = self.trades_df[
            (self.trades_df['Ticker'] == ticker) & 
            (self.trades_df['Date'] <= up_to_date)
        ]
        return ticker_trades['Signed_Quantity'].sum()
    
    def get_daily_transaction_summary(self) -> pd.DataFrame:
        """Get daily summary of all transactions"""
        # Group by date and summarize
        daily_summary = []
        
        for date in self.df['Date'].unique():
            date_trades = self.df[self.df['Date'] == date]
            
            summary = {
                'Date': date,
                'Total_Trades': len(date_trades),
                'Buy_Trades': len(date_trades[date_trades['Buy_sell'] == 'BUY']),
                'Sell_Trades': len(date_trades[date_trades['Buy_sell'] == 'SELL']),
                'Total_Value': date_trades['Transaction_Value'].sum(),
                'Net_Cash_Flow': date_trades[date_trades['Ticker'] == 'CASH']['Signed_Transaction_Value'].sum(),
                'Tickers_Traded': date_trades[date_trades['Ticker'] != 'CASH']['Ticker'].nunique()
            }
            
            daily_summary.append(summary)
        
        return pd.DataFrame(daily_summary)
    
    def validate_data_integrity(self) -> Dict[str, any]:
        """Validate data integrity and return summary"""
        validation_results = {
            'total_transactions': len(self.df),
            'unique_tickers': len(self.get_unique_tickers()),
            'date_range': self.get_date_range(),
            'has_cash_flows': len(self.cash_flows_df) > 0,
            'total_cash_flows': self.cash_flows_df['Cash_Flow'].sum() if len(self.cash_flows_df) > 0 else 0,
            'tickers_with_zero_position': [],
            'data_quality_issues': []
        }
        
        # Check for tickers with zero net position
        for ticker in self.get_unique_tickers():
            net_position = self.calculate_position_quantity(ticker, datetime.now())
            if abs(net_position) < 1e-6:  # Essentially zero
                validation_results['tickers_with_zero_position'].append(ticker)
        
        # Check for any data quality issues
        if self.df['Price'].min() <= 0:
            validation_results['data_quality_issues'].append("Found non-positive prices")
        
        if self.df['Quantity'].min() <= 0:
            validation_results['data_quality_issues'].append("Found non-positive quantities")
        
        # Check for future dates
        future_dates = self.df[self.df['Date'] > datetime.now()]
        if len(future_dates) > 0:
            validation_results['data_quality_issues'].append(f"Found {len(future_dates)} future-dated transactions")
        
        return validation_results
