import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Union

def format_currency(value: Union[float, int], currency_symbol: str = '$') -> str:
    """Format a numeric value as currency"""
    if pd.isna(value) or value is None:
        return f"{currency_symbol}0.00"
    
    # Handle very large numbers
    if abs(value) >= 1e9:
        return f"{currency_symbol}{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{currency_symbol}{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{currency_symbol}{value/1e3:.1f}K"
    else:
        return f"{currency_symbol}{value:,.2f}"

def format_percentage(value: Union[float, int], decimal_places: int = 2) -> str:
    """Format a decimal value as percentage"""
    if pd.isna(value) or value is None:
        return "0.00%"
    
    return f"{value * 100:.{decimal_places}f}%"

def format_number(value: Union[float, int], decimal_places: int = 2) -> str:
    """Format a number with proper thousand separators"""
    if pd.isna(value) or value is None:
        return "0.00"
    
    return f"{value:,.{decimal_places}f}"

def validate_csv(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate uploaded CSV file"""
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required columns
    required_columns = ['Date', 'Company', 'Ticker', 'Buy_sell', 'Price', 'Quantity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_result['valid'] = False
        validation_result['errors'].append(f"Missing required columns: {', '.join(missing_columns)}")
        return validation_result
    
    # Check for empty dataframe
    if df.empty:
        validation_result['valid'] = False
        validation_result['errors'].append("CSV file is empty")
        return validation_result
    
    # Validate data types and values
    for index, row in df.iterrows():
        row_num = int(index) + 1
        
        # Validate date format
        try:
            parsed_date = pd.to_datetime(row['Date'], format='%d-%m-%Y')
            # Check for future dates
            if parsed_date > datetime.now():
                validation_result['warnings'].append(f"Row {row_num}: Future date detected ({row['Date']})")
        except:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Row {row_num}: Invalid date format '{row['Date']}'. Expected dd-mm-yyyy")
            continue
        
        # Validate Buy_sell column
        if row['Buy_sell'] not in ['BUY', 'SELL']:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Row {row_num}: Buy_sell must be 'BUY' or 'SELL', found '{row['Buy_sell']}'")
        
        # Validate Price (must be positive number)
        try:
            price = float(row['Price'])
            if price <= 0:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Row {row_num}: Price must be positive, found {price}")
        except:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Row {row_num}: Price must be a number, found '{row['Price']}'")
        
        # Validate Quantity (must be positive number)
        try:
            quantity = float(row['Quantity'])
            if quantity <= 0:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Row {row_num}: Quantity must be positive, found {quantity}")
        except:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Row {row_num}: Quantity must be a number, found '{row['Quantity']}'")
        
        # Validate Ticker
        if pd.isna(row['Ticker']) or str(row['Ticker']).strip() == '':
            validation_result['valid'] = False
            validation_result['errors'].append(f"Row {row_num}: Ticker cannot be empty")
        
        # Check for known crypto tickers
        if row['Ticker'] in ['BTC', 'ETH', 'SOL']:
            validation_result['warnings'].append(f"Row {row_num}: Crypto ticker '{row['Ticker']}' should be '{row['Ticker']}-USD' for proper market data")
    
    # Check for duplicate rows
    duplicates = df.duplicated()
    if duplicates.any():
        duplicate_count = duplicates.sum()
        validation_result['warnings'].append(f"Found {duplicate_count} duplicate rows")
    
    # Validate cash transactions
    cash_transactions = df[df['Ticker'] == 'CASH']
    for index, row in cash_transactions.iterrows():
        if row['Company'] != 'CASH':
            validation_result['warnings'].append(f"Row {index + 1}: Cash transaction should have Company='CASH'")
        
        if abs(float(row['Price']) - 1.0) > 0.001:  # Allow small floating point differences
            validation_result['warnings'].append(f"Row {index + 1}: Cash transaction price should be 1.0")
    
    return validation_result

def calculate_business_days(start_date: datetime, end_date: datetime) -> int:
    """Calculate number of business days between two dates"""
    return pd.bdate_range(start_date, end_date).shape[0]

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    return numerator / denominator

def annualize_return(return_value: float, periods: int, periods_per_year: int = 365) -> float:
    """Annualize a return given the number of periods"""
    if periods <= 0 or pd.isna(return_value):
        return 0.0
    
    return (1 + return_value) ** (periods_per_year / periods) - 1

def annualize_volatility(volatility: float, periods_per_year: int = 365) -> float:
    """Annualize volatility"""
    if pd.isna(volatility):
        return 0.0
    
    return volatility * np.sqrt(periods_per_year)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0.0
    
    return excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio (downside deviation)"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    return excess_returns.mean() / downside_returns.std()

def calculate_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio"""
    if max_drawdown == 0:
        return 0.0
    
    return annual_return / abs(max_drawdown)

def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate Information ratio"""
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0
    
    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std()
    
    if tracking_error == 0:
        return 0.0
    
    return excess_returns.mean() / tracking_error

def get_ticker_type(ticker: str) -> str:
    """Determine ticker type (equity, crypto, cash)"""
    if ticker == 'CASH':
        return 'cash'
    elif ticker in ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BTC', 'ETH', 'SOL']:
        return 'crypto'
    else:
        return 'equity'

def format_date(date: datetime, format_string: str = '%d-%m-%Y') -> str:
    """Format datetime object as string"""
    if pd.isna(date) or date is None:
        return "N/A"
    
    return date.strftime(format_string)

def clean_ticker_symbol(ticker: str) -> str:
    """Clean and standardize ticker symbol"""
    if pd.isna(ticker):
        return ""
    
    ticker = ticker.strip().upper()
    
    # Handle crypto tickers
    crypto_mapping = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'SOL': 'SOL-USD',
        'BITCOIN': 'BTC-USD',
        'ETHEREUM': 'ETH-USD',
        'SOLANA': 'SOL-USD'
    }
    
    return crypto_mapping.get(ticker, ticker)

def validate_ticker_format(ticker: str) -> bool:
    """Validate ticker format"""
    if not ticker or pd.isna(ticker):
        return False
    
    ticker = ticker.strip()
    
    # Basic validation - alphanumeric with optional dash
    if not ticker.replace('-', '').replace('.', '').isalnum():
        return False
    
    # Check length (reasonable ticker length)
    if len(ticker) < 1 or len(ticker) > 10:
        return False
    
    return True

def create_date_range(start_date: datetime, end_date: datetime, freq: str = 'D') -> pd.DatetimeIndex:
    """Create date range with specified frequency"""
    return pd.date_range(start_date, end_date, freq=freq)

def get_month_name(month_number: int) -> str:
    """Get month name from month number"""
    months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    if 1 <= month_number <= 12:
        return months[month_number - 1]
    return 'Unknown'

def calculate_compound_return(returns: List[float]) -> float:
    """Calculate compound return from a list of period returns"""
    if not returns:
        return 0.0
    
    compound = 1.0
    for ret in returns:
        if not pd.isna(ret):
            compound *= (1 + ret)
    
    return compound - 1.0
