import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

class PortfolioCalculator:
    def __init__(self, data_processor, market_data: Dict, cost_basis_method: str = 'FIFO', 
                 risk_free_rate: float = 0.02, fee_percentage: float = 0.0):
        """Initialize portfolio calculator"""
        self.data_processor = data_processor
        self.market_data = market_data
        self.cost_basis_method = cost_basis_method
        self.risk_free_rate = risk_free_rate
        self.fee_percentage = fee_percentage
        
        # Calculate portfolio metrics
        self._calculate_portfolio_positions()
        self._calculate_daily_aum()
        self._calculate_performance_metrics()
    
    def _calculate_portfolio_positions(self):
        """Calculate current portfolio positions with lot tracking"""
        self.current_positions = {}
        self.lot_tracking = {}
        
        for ticker in self.data_processor.get_unique_tickers():
            self.current_positions[ticker] = {
                'quantity': 0,
                'cost_basis': 0,
                'lots': []
            }
            self.lot_tracking[ticker] = []
            
            trades = self.data_processor.get_trades_for_ticker(ticker)
            
            for _, trade in trades.iterrows():
                self._process_trade(ticker, trade)
    
    def _process_trade(self, ticker: str, trade: pd.Series):
        """Process individual trade and update positions"""
        quantity = trade['Signed_Quantity']
        price = trade['Price']
        date = trade['Date']
        
        if quantity > 0:  # Buy
            # Add new lot
            lot = {
                'date': date,
                'quantity': quantity,
                'price': price,
                'cost_basis': quantity * price
            }
            self.current_positions[ticker]['lots'].append(lot)
            self.current_positions[ticker]['quantity'] += quantity
            
        else:  # Sell
            sell_quantity = abs(quantity)
            self._process_sell(ticker, sell_quantity, price, date)
    
    def _process_sell(self, ticker: str, sell_quantity: float, sell_price: float, sell_date: datetime):
        """Process sell transaction using selected cost basis method"""
        remaining_to_sell = sell_quantity
        lots = self.current_positions[ticker]['lots']
        
        if self.cost_basis_method == 'FIFO':
            # First In, First Out
            lots_to_remove = []
            for i, lot in enumerate(lots):
                if remaining_to_sell <= 0:
                    break
                
                if lot['quantity'] <= remaining_to_sell:
                    # Sell entire lot
                    remaining_to_sell -= lot['quantity']
                    lots_to_remove.append(i)
                else:
                    # Partial sell
                    lot['quantity'] -= remaining_to_sell
                    lot['cost_basis'] = lot['quantity'] * lot['price']
                    remaining_to_sell = 0
            
            # Remove fully sold lots (in reverse order to maintain indices)
            for i in reversed(lots_to_remove):
                lots.pop(i)
        
        else:  # Average Cost
            if lots:
                # Calculate average cost
                total_quantity = sum(lot['quantity'] for lot in lots)
                total_cost = sum(lot['cost_basis'] for lot in lots)
                avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                
                # Reduce all lots proportionally
                reduction_ratio = sell_quantity / total_quantity
                for lot in lots:
                    lot['quantity'] *= (1 - reduction_ratio)
                    lot['cost_basis'] = lot['quantity'] * avg_price
                
                # Remove lots with zero quantity
                lots[:] = [lot for lot in lots if lot['quantity'] > 1e-6]
        
        # Update total position
        self.current_positions[ticker]['quantity'] -= sell_quantity
    
    def _calculate_daily_aum(self):
        """Calculate daily AUM including cash position"""
        start_date, end_date = self.data_processor.get_date_range()
        
        # Extend to today if needed
        today = datetime.now().date()
        if end_date.date() < today:
            end_date = datetime.combine(today, datetime.min.time())
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        self.daily_aum = []
        self.daily_positions = {}
        
        for date in date_range:
            aum_data = self._calculate_aum_for_date(date)
            self.daily_aum.append(aum_data)
        
        self.daily_aum_df = pd.DataFrame(self.daily_aum)
    
    def _calculate_aum_for_date(self, date: datetime) -> Dict:
        """Calculate AUM and positions for a specific date"""
        # Calculate cash position
        cash_flows = self.data_processor.get_cash_flows()
        cash_flows_up_to_date = cash_flows[cash_flows['Date'] <= date]
        cash_position = cash_flows_up_to_date['Cash_Flow'].sum()
        
        # Subtract cash used for trades
        trades_up_to_date = self.data_processor.trades_df[
            self.data_processor.trades_df['Date'] <= date
        ]
        cash_used_for_trades = trades_up_to_date['Signed_Transaction_Value'].sum()
        cash_position -= cash_used_for_trades
        
        # Calculate asset positions
        asset_values = {}
        total_asset_value = 0
        
        for ticker in self.data_processor.get_unique_tickers():
            quantity = self.data_processor.calculate_position_quantity(ticker, date)
            
            if abs(quantity) > 1e-6:  # Non-zero position
                # Get market price for date
                market_price = self._get_market_price(ticker, date)
                if market_price is not None:
                    market_value = quantity * market_price
                    asset_values[ticker] = {
                        'quantity': quantity,
                        'price': market_price,
                        'market_value': market_value
                    }
                    total_asset_value += market_value
        
        total_aum = cash_position + total_asset_value
        
        return {
            'date': date,
            'cash_position': cash_position,
            'total_asset_value': total_asset_value,
            'total_aum': total_aum,
            'asset_positions': asset_values
        }
    
    def _get_market_price(self, ticker: str, date: datetime) -> Optional[float]:
        """Get market price for ticker on specific date"""
        if ticker not in self.market_data:
            return None
        
        ticker_data = self.market_data[ticker]
        
        # Find closest available price
        available_dates = ticker_data.index
        
        # Handle timezone compatibility - convert both to comparable formats
        check_date = pd.to_datetime(date)
        
        # If market data has timezone info, make dates comparable
        if hasattr(available_dates, 'tz') and available_dates.tz is not None:
            # Convert our date to the same timezone as market data
            if check_date.tz is None:
                check_date = check_date.tz_localize('UTC').tz_convert(available_dates.tz)
        else:
            # Market data is timezone naive, ensure our date is too
            if check_date.tz is not None:
                check_date = check_date.tz_localize(None)
        
        # Try exact date first
        if check_date in available_dates:
            return ticker_data.loc[check_date, 'Adj Close']
        
        # Find closest previous date
        previous_dates = available_dates[available_dates <= check_date]
        if len(previous_dates) > 0:
            closest_date = previous_dates.max()
            return ticker_data.loc[closest_date, 'Adj Close']
        
        return None
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if len(self.daily_aum_df) < 2:
            self.performance_metrics = {}
            return
        
        # Calculate daily returns
        self.daily_aum_df['prev_aum'] = self.daily_aum_df['total_aum'].shift(1)
        
        # Get daily external cash flows
        cash_flows = self.data_processor.get_cash_flows()
        daily_ecf = self.daily_aum_df['date'].apply(
            lambda d: cash_flows[cash_flows['Date'] == d]['Cash_Flow'].sum()
        )
        
        # Calculate daily returns adjusted for external cash flows
        self.daily_aum_df['ecf'] = daily_ecf
        self.daily_aum_df['daily_return'] = np.where(
            self.daily_aum_df['prev_aum'] != 0,
            (self.daily_aum_df['total_aum'] - self.daily_aum_df['prev_aum'] - self.daily_aum_df['ecf']) / self.daily_aum_df['prev_aum'],
            0
        )
        
        # Remove first row (no previous AUM)
        returns_df = self.daily_aum_df.iloc[1:].copy()
        
        # Calculate Time-Weighted Return (TWR)
        twr = (1 + returns_df['daily_return']).prod() - 1
        
        # Calculate XIRR
        xirr = self._calculate_xirr()
        
        # Calculate other metrics
        daily_returns = returns_df['daily_return'].dropna()
        
        # Annualized metrics
        trading_days_per_year = 365
        
        # Performance metrics
        self.performance_metrics = {
            'twr': twr,
            'xirr': xirr,
            'daily_return': daily_returns.iloc[-1] if len(daily_returns) > 0 else 0,
            'mtd_return': self._calculate_mtd_return(),
            'ytd_return': self._calculate_ytd_return(),
            'annualized_return': (1 + twr) ** (trading_days_per_year / len(daily_returns)) - 1 if len(daily_returns) > 0 else 0,
            'volatility': daily_returns.std() * np.sqrt(trading_days_per_year) if len(daily_returns) > 1 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(daily_returns),
            'max_drawdown': self._calculate_max_drawdown(),
            'calmar_ratio': self._calculate_calmar_ratio(),
            'hit_rate': (daily_returns > 0).mean() if len(daily_returns) > 0 else 0,
            'average_gain': daily_returns[daily_returns > 0].mean() if len(daily_returns[daily_returns > 0]) > 0 else 0,
            'average_loss': daily_returns[daily_returns < 0].mean() if len(daily_returns[daily_returns < 0]) > 0 else 0,
            'profit_factor': self._calculate_profit_factor(daily_returns)
        }
    
    def _calculate_xirr(self) -> float:
        """Calculate XIRR (Internal Rate of Return)"""
        try:
            cash_flows = self.data_processor.get_cash_flows()
            
            if len(cash_flows) == 0:
                return 0.0
            
            # Prepare cash flows for XIRR calculation
            dates = []
            amounts = []
            
            # Add all external cash flows (deposits negative, withdrawals positive)
            for _, cf in cash_flows.iterrows():
                dates.append(cf['Date'])
                amounts.append(-cf['Cash_Flow'])  # Negative for deposits
            
            # Add final portfolio value as positive cash flow
            final_date = self.daily_aum_df['date'].max()
            final_aum = self.daily_aum_df[self.daily_aum_df['date'] == final_date]['total_aum'].iloc[0]
            dates.append(final_date)
            amounts.append(final_aum)
            
            # Convert to numpy arrays
            dates = pd.to_datetime(dates)
            amounts = np.array(amounts)
            
            # Calculate XIRR using Newton-Raphson method
            def xirr_func(rate):
                return sum(amt / ((1 + rate) ** ((date - dates[0]).days / 365.0)) 
                          for amt, date in zip(amounts, dates))
            
            # Initial guess
            rate_guess = 0.1
            
            # Solve for rate
            try:
                xirr_rate = fsolve(xirr_func, rate_guess)[0]
                return xirr_rate
            except:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_mtd_return(self) -> float:
        """Calculate Month-to-Date return"""
        try:
            current_date = self.daily_aum_df['date'].max()
            month_start = current_date.replace(day=1)
            
            mtd_data = self.daily_aum_df[self.daily_aum_df['date'] >= month_start]
            if len(mtd_data) < 2:
                return 0.0
            
            mtd_returns = mtd_data['daily_return'].dropna()
            return (1 + mtd_returns).prod() - 1
        except:
            return 0.0
    
    def _calculate_ytd_return(self) -> float:
        """Calculate Year-to-Date return"""
        try:
            current_date = self.daily_aum_df['date'].max()
            year_start = current_date.replace(month=1, day=1)
            
            ytd_data = self.daily_aum_df[self.daily_aum_df['date'] >= year_start]
            if len(ytd_data) < 2:
                return 0.0
            
            ytd_returns = ytd_data['daily_return'].dropna()
            return (1 + ytd_returns).prod() - 1
        except:
            return 0.0
    
    def _calculate_sharpe_ratio(self, daily_returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(daily_returns) < 2:
                return 0.0
            
            excess_returns = daily_returns - (self.risk_free_rate / 365)
            return excess_returns.mean() / excess_returns.std() * np.sqrt(365) if excess_returns.std() != 0 else 0.0
        except:
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative_returns = (1 + self.daily_aum_df['daily_return'].fillna(0)).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return drawdown.min()
        except:
            return 0.0
    
    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        try:
            annualized_return = self.performance_metrics.get('annualized_return', 0)
            max_drawdown = abs(self.performance_metrics.get('max_drawdown', 0))
            return annualized_return / max_drawdown if max_drawdown != 0 else 0.0
        except:
            return 0.0
    
    def _calculate_profit_factor(self, daily_returns: pd.Series) -> float:
        """Calculate profit factor"""
        try:
            gross_profit = daily_returns[daily_returns > 0].sum()
            gross_loss = abs(daily_returns[daily_returns < 0].sum())
            return gross_profit / gross_loss if gross_loss != 0 else 0.0
        except:
            return 0.0
    
    # Public methods for accessing calculated data
    
    def get_current_aum(self) -> float:
        """Get current total AUM"""
        if len(self.daily_aum_df) > 0:
            return self.daily_aum_df['total_aum'].iloc[-1]
        return 0.0
    
    def get_cash_position(self) -> float:
        """Get current cash position"""
        if len(self.daily_aum_df) > 0:
            return self.daily_aum_df['cash_position'].iloc[-1]
        return 0.0
    
    def get_realized_pnl(self) -> float:
        """Calculate total realized P&L from closed positions"""
        try:
            realized_pnl = 0.0
            
            for ticker in self.data_processor.get_unique_tickers():
                ticker_trades = self.data_processor.get_trades_for_ticker(ticker)
                
                if len(ticker_trades) == 0:
                    continue
                    
                # Track lots for this ticker using FIFO
                lots = []
                
                for _, trade in ticker_trades.iterrows():
                    quantity = trade['Signed_Quantity']
                    price = trade['Price']
                    
                    if quantity > 0:  # Buy
                        lots.append({'quantity': quantity, 'price': price})
                    else:  # Sell
                        sell_quantity = abs(quantity)
                        
                        while sell_quantity > 0 and lots:
                            lot = lots[0]
                            
                            if lot['quantity'] <= sell_quantity:
                                # Use entire lot
                                lot_quantity = lot['quantity']
                                cost_basis = lot['price']
                                realized_pnl += (price - cost_basis) * lot_quantity
                                sell_quantity -= lot_quantity
                                lots.pop(0)
                            else:
                                # Use partial lot
                                cost_basis = lot['price']
                                realized_pnl += (price - cost_basis) * sell_quantity
                                lot['quantity'] -= sell_quantity
                                sell_quantity = 0
            
            return realized_pnl
            
        except Exception:
            return 0.0
    
    def get_performance_metrics(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict:
        """Get performance metrics with optional date filtering"""
        if start_date is None and end_date is None:
            return self.performance_metrics
        
        # Recalculate performance metrics for the specified date range
        filtered_aum = self.get_aum_timeseries(start_date, end_date)
        
        if len(filtered_aum) < 2:
            return {}
        
        # Calculate returns for filtered period
        filtered_returns = self._calculate_returns_for_period(filtered_aum)
        
        return filtered_returns
    
    def get_detailed_performance_metrics(self) -> Dict:
        """Get detailed categorized performance metrics"""
        return {
            'returns': {
                'twr': self.performance_metrics.get('twr', 0),
                'xirr': self.performance_metrics.get('xirr', 0),
                'daily_return': self.performance_metrics.get('daily_return', 0),
                'mtd_return': self.performance_metrics.get('mtd_return', 0),
                'ytd_return': self.performance_metrics.get('ytd_return', 0),
                'annualized_return': self.performance_metrics.get('annualized_return', 0)
            },
            'risk': {
                'volatility': self.performance_metrics.get('volatility', 0),
                'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': self.performance_metrics.get('max_drawdown', 0),
                'calmar_ratio': self.performance_metrics.get('calmar_ratio', 0)
            },
            'trading': {
                'hit_rate': self.performance_metrics.get('hit_rate', 0),
                'average_gain': self.performance_metrics.get('average_gain', 0),
                'average_loss': self.performance_metrics.get('average_loss', 0),
                'profit_factor': self.performance_metrics.get('profit_factor', 0)
            },
            'benchmark': self.get_benchmark_comparison()
        }
    
    def get_current_holdings(self) -> pd.DataFrame:
        """Get current holdings dataframe"""
        holdings = []
        
        for ticker, position in self.current_positions.items():
            if position['quantity'] > 1e-6:  # Non-zero position
                current_price = self._get_market_price(ticker, datetime.now())
                if current_price is not None:
                    market_value = position['quantity'] * current_price
                    
                    # Calculate cost basis
                    total_cost = sum(lot['cost_basis'] for lot in position['lots'])
                    avg_cost = total_cost / position['quantity'] if position['quantity'] != 0 else 0
                    
                    unrealized_pnl = market_value - total_cost
                    unrealized_return = unrealized_pnl / total_cost if total_cost != 0 else 0
                    
                    # Calculate weight
                    total_aum = self.get_current_aum()
                    weight = market_value / total_aum if total_aum > 0 else 0
                    
                    holdings.append({
                        'Ticker': ticker,
                        'Quantity': position['quantity'],
                        'Avg Cost': avg_cost,
                        'Current Price': current_price,
                        'Market Value': market_value,
                        'Cost Basis': total_cost,
                        'Unrealized P&L': unrealized_pnl,
                        'Return %': unrealized_return,
                        'Weight %': weight
                    })
        
        return pd.DataFrame(holdings)
    
    def get_lot_detail(self, ticker: str) -> pd.DataFrame:
        """Get lot-level detail for a specific ticker"""
        if ticker not in self.current_positions:
            return pd.DataFrame()
        
        lots = self.current_positions[ticker]['lots']
        current_price = self._get_market_price(ticker, datetime.now())
        
        lot_details = []
        for lot in lots:
            if current_price is not None:
                current_value = lot['quantity'] * current_price
                unrealized_pnl = current_value - lot['cost_basis']
                unrealized_return = unrealized_pnl / lot['cost_basis'] if lot['cost_basis'] != 0 else 0
                
                lot_details.append({
                    'Purchase Date': lot['date'].strftime('%d-%m-%Y'),
                    'Quantity': lot['quantity'],
                    'Purchase Price': lot['price'],
                    'Cost Basis': lot['cost_basis'],
                    'Current Price': current_price,
                    'Current Value': current_value,
                    'Unrealized P&L': unrealized_pnl,
                    'Return %': unrealized_return
                })
        
        return pd.DataFrame(lot_details)
    
    def get_aum_timeseries(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get AUM time series data with optional date filtering"""
        df = self.daily_aum_df[['date', 'total_aum', 'cash_position', 'total_asset_value']].copy()
        
        if start_date is not None:
            df = df[df['date'] >= start_date]
        if end_date is not None:
            df = df[df['date'] <= end_date]
            
        return df
    
    def get_asset_trades(self, ticker: str) -> pd.DataFrame:
        """Get all trades for a specific asset"""
        return self.data_processor.get_trades_for_ticker(ticker)
    
    def get_asset_performance_metrics(self, ticker: str) -> Dict:
        """Get performance metrics for a specific asset"""
        trades = self.get_asset_trades(ticker)
        if trades.empty:
            return {}
        
        current_price = self._get_market_price(ticker, datetime.now())
        first_trade_price = trades.iloc[0]['Price']
        
        if current_price is not None and first_trade_price is not None:
            total_return = (current_price - first_trade_price) / first_trade_price
        else:
            total_return = 0
        
        return {
            'first_trade_date': trades.iloc[0]['Date'].strftime('%d-%m-%Y'),
            'last_trade_date': trades.iloc[-1]['Date'].strftime('%d-%m-%Y'),
            'total_trades': len(trades),
            'first_price': first_trade_price,
            'current_price': current_price,
            'total_return': total_return,
            'net_quantity': trades['Signed_Quantity'].sum()
        }
    
    def _calculate_returns_for_period(self, aum_data: pd.DataFrame) -> Dict:
        """Calculate performance metrics for a specific period"""
        if len(aum_data) < 2:
            return {}
        
        # Calculate daily returns
        aum_data = aum_data.copy()
        aum_data['prev_aum'] = aum_data['total_aum'].shift(1)
        aum_data['daily_return'] = (aum_data['total_aum'] - aum_data['prev_aum']) / aum_data['prev_aum']
        
        # Remove first row (NaN return)
        daily_returns = aum_data['daily_return'].dropna()
        
        if len(daily_returns) == 0:
            return {}
        
        # Calculate basic metrics
        total_return = (aum_data['total_aum'].iloc[-1] - aum_data['total_aum'].iloc[0]) / aum_data['total_aum'].iloc[0]
        
        # Annualized return
        days = len(daily_returns)
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility != 0 else 0
        
        # Max drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'period_days': days
        }

    def get_benchmark_comparison(self) -> Dict:
        """Get benchmark comparison metrics"""
        benchmarks = {}
        
        for benchmark_ticker in ['SPY', 'QQQ']:
            if benchmark_ticker in self.market_data:
                benchmark_data = self.market_data[benchmark_ticker]
                
                if not benchmark_data.empty and not self.daily_aum_df.empty:
                    # Calculate benchmark daily returns
                    benchmark_prices = benchmark_data['Adj Close'].reindex(self.daily_aum_df['date'])
                    benchmark_prices = benchmark_prices.fillna(method='ffill').fillna(method='bfill')
                    benchmark_returns = benchmark_prices.pct_change().fillna(0)
                    
                    # Get portfolio returns for the same dates
                    portfolio_returns = self.daily_aum_df['daily_return'].fillna(0)
                    
                    # Align the series
                    common_dates = benchmark_returns.index.intersection(portfolio_returns.index)
                    if len(common_dates) > 10:  # Need sufficient data
                        bench_ret_aligned = benchmark_returns.reindex(common_dates).fillna(0)
                        port_ret_aligned = portfolio_returns.reindex(common_dates).fillna(0)
                        
                        # Calculate metrics
                        correlation = port_ret_aligned.corr(bench_ret_aligned)
                        if pd.isna(correlation):
                            correlation = 0
                            
                        # Beta calculation
                        if bench_ret_aligned.var() > 0:
                            beta = port_ret_aligned.cov(bench_ret_aligned) / bench_ret_aligned.var()
                        else:
                            beta = 0
                            
                        # Alpha (annualized)
                        port_annual_ret = (1 + port_ret_aligned.mean()) ** 252 - 1
                        bench_annual_ret = (1 + bench_ret_aligned.mean()) ** 252 - 1
                        alpha = port_annual_ret - (self.risk_free_rate + beta * (bench_annual_ret - self.risk_free_rate))
                        
                        # Tracking error (annualized)
                        excess_returns = port_ret_aligned - bench_ret_aligned
                        tracking_error = excess_returns.std() * np.sqrt(252)
                        
                        # Information ratio
                        if tracking_error > 0:
                            information_ratio = excess_returns.mean() * 252 / tracking_error
                        else:
                            information_ratio = 0
                        
                        benchmarks[benchmark_ticker] = {
                            'correlation': correlation,
                            'beta': beta,
                            'alpha': alpha,
                            'tracking_error': tracking_error,
                            'information_ratio': information_ratio
                        }
                    else:
                        # Insufficient data
                        benchmarks[benchmark_ticker] = {
                            'correlation': 0,
                            'beta': 0,
                            'alpha': 0,
                            'tracking_error': 0,
                            'information_ratio': 0
                        }
                else:
                    # No data available
                    benchmarks[benchmark_ticker] = {
                        'correlation': 0,
                        'beta': 0,
                        'alpha': 0,
                        'tracking_error': 0,
                        'information_ratio': 0
                    }
        
        return benchmarks
