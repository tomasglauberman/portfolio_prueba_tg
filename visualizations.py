import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class PortfolioVisualizations:
    def __init__(self, portfolio_calculator, theme_mode: str = 'light'):
        """Initialize visualization generator"""
        self.calculator = portfolio_calculator
        self.theme_mode = theme_mode
        self.colors = self._get_color_scheme()
    
    def _get_color_scheme(self) -> Dict:
        """Get color scheme based on theme mode"""
        if self.theme_mode == 'dark':
            return {
                'background': '#2E2E2E',
                'paper': '#3E3E3E',
                'text': '#FFFFFF',
                'grid': '#4E4E4E',
                'positive': '#00CC96',
                'negative': '#FF6B6B',
                'neutral': '#AB63FA',
                'line_colors': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
            }
        else:
            return {
                'background': '#FFFFFF',
                'paper': '#FFFFFF',
                'text': '#000000',
                'grid': '#E0E0E0',
                'positive': '#00CC96',
                'negative': '#FF6B6B',
                'neutral': '#AB63FA',
                'line_colors': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
            }
    
    def _get_layout_template(self) -> Dict:
        """Get common layout template"""
        return {
            'plot_bgcolor': self.colors['background'],
            'paper_bgcolor': self.colors['paper'],
            'font': {'color': self.colors['text']},
            'xaxis': {
                'gridcolor': self.colors['grid'],
                'linecolor': self.colors['grid'],
                'tickcolor': self.colors['text']
            },
            'yaxis': {
                'gridcolor': self.colors['grid'],
                'linecolor': self.colors['grid'],
                'tickcolor': self.colors['text']
            }
        }
    
    def create_aum_chart(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[go.Figure]:
        """Create AUM evolution chart with optional date filtering"""
        try:
            aum_data = self.calculator.get_aum_timeseries(start_date, end_date)
            
            if aum_data.empty:
                return None
            
            fig = go.Figure()
            
            # AUM line
            fig.add_trace(go.Scatter(
                x=aum_data['date'],
                y=aum_data['total_aum'],
                mode='lines',
                name='Total AUM',
                line=dict(color=self.colors['line_colors'][0], width=2),
                hovertemplate='Date: %{x}<br>AUM: $%{y:,.0f}<extra></extra>'
            ))
            
            # Cash position area
            fig.add_trace(go.Scatter(
                x=aum_data['date'],
                y=aum_data['cash_position'],
                mode='lines',
                name='Cash Position',
                fill='tonexty',
                fillcolor='rgba(99, 110, 250, 0.2)',
                line=dict(color=self.colors['line_colors'][1], width=1),
                hovertemplate='Date: %{x}<br>Cash: $%{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                **self._get_layout_template(),
                title='Portfolio AUM Evolution',
                xaxis_title='Date',
                yaxis_title='Value ($)',
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating AUM chart: {str(e)}")
            return None
    
    def create_allocation_chart(self) -> Optional[go.Figure]:
        """Create current allocation pie chart"""
        try:
            holdings = self.calculator.get_current_holdings()
            
            if holdings.empty:
                return None
            
            # Add cash position
            cash_data = self.calculator.daily_aum_df
            if not cash_data.empty:
                current_cash = cash_data['cash_position'].iloc[-1]
                total_aum = self.calculator.get_current_aum()
                
                if current_cash > 0 and total_aum > 0:
                    cash_row = pd.DataFrame({
                        'Ticker': ['CASH'],
                        'Market Value': [current_cash],
                        'Weight %': [current_cash / total_aum]
                    })
                    holdings = pd.concat([holdings, cash_row], ignore_index=True)
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=holdings['Ticker'],
                values=holdings['Market Value'],
                hole=.3,
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='%{label}<br>Value: $%{value:,.0f}<br>Weight: %{percent}<extra></extra>',
                marker=dict(colors=self.colors['line_colors'][:len(holdings)])
            )])
            
            fig.update_layout(
                **self._get_layout_template(),
                title='Current Portfolio Allocation',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating allocation chart: {str(e)}")
            return None
    
    def create_allocation_chart_without_cash(self) -> Optional[go.Figure]:
        """Create current allocation pie chart excluding cash position"""
        try:
            holdings = self.calculator.get_current_holdings()
            
            if holdings.empty:
                return None
            
            # Filter out cash positions if any exist
            non_cash_holdings = holdings[holdings['Ticker'] != 'CASH'].copy()
            
            if non_cash_holdings.empty:
                return None
            
            # Recalculate weights for non-cash holdings only
            total_non_cash_value = non_cash_holdings['Market Value'].sum()
            if total_non_cash_value > 0:
                non_cash_holdings['Weight %'] = non_cash_holdings['Market Value'] / total_non_cash_value
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=non_cash_holdings['Ticker'],
                values=non_cash_holdings['Market Value'],
                hole=.3,
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='%{label}<br>Value: $%{value:,.0f}<br>Weight: %{percent}<extra></extra>',
                marker=dict(colors=self.colors['line_colors'][:len(non_cash_holdings)])
            )])
            
            fig.update_layout(
                **self._get_layout_template(),
                title='Portfolio Allocation (Excluding Cash)',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating allocation chart without cash: {str(e)}")
            return None
    
    def create_benchmark_comparison_chart(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[go.Figure]:
        """Create performance vs benchmark comparison chart with optional date filtering"""
        try:
            aum_data = self.calculator.get_aum_timeseries(start_date, end_date)
            
            if aum_data.empty:
                return None
            
            # Calculate cumulative returns (normalized to 100)
            aum_data = aum_data.copy()
            aum_data['portfolio_return'] = aum_data['total_aum'] / aum_data['total_aum'].iloc[0] * 100
            
            fig = go.Figure()
            
            # Portfolio performance
            fig.add_trace(go.Scatter(
                x=aum_data['date'],
                y=aum_data['portfolio_return'],
                mode='lines',
                name='Portfolio',
                line=dict(color=self.colors['line_colors'][0], width=2),
                hovertemplate='Date: %{x}<br>Portfolio: %{y:.1f}<extra></extra>'
            ))
            
            # Add benchmark data if available
            benchmark_colors = [self.colors['line_colors'][1], self.colors['line_colors'][2]]
            for i, benchmark_ticker in enumerate(['SPY', 'QQQ']):
                if benchmark_ticker in self.calculator.market_data:
                    benchmark_data = self.calculator.market_data[benchmark_ticker]
                    
                    if not benchmark_data.empty:
                        # Filter benchmark data by date range if specified
                        benchmark_filtered = benchmark_data.copy()
                        if start_date and end_date:
                            benchmark_filtered = benchmark_filtered.loc[start_date:end_date]
                        elif start_date:
                            benchmark_filtered = benchmark_filtered.loc[start_date:]
                        elif end_date:
                            benchmark_filtered = benchmark_filtered.loc[:end_date]
                        
                        if not benchmark_filtered.empty:
                            # Calculate benchmark cumulative returns (normalized to 100)
                            benchmark_prices = benchmark_filtered['Adj Close']
                            benchmark_returns = benchmark_prices / benchmark_prices.iloc[0] * 100
                            
                            fig.add_trace(go.Scatter(
                                x=benchmark_filtered.index,
                                y=benchmark_returns,
                                mode='lines',
                                name=benchmark_ticker,
                                line=dict(color=benchmark_colors[i], width=1.5),
                                hovertemplate=f'Date: %{{x}}<br>{benchmark_ticker}: %{{y:.1f}}<extra></extra>'
                            ))
            
            fig.update_layout(
                **self._get_layout_template(),
                title='Portfolio vs Benchmarks (Rebased to 100)',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (Base = 100)',
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating benchmark comparison chart: {str(e)}")
            return None
    
    def create_drawdown_chart(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[go.Figure]:
        """Create drawdown chart with optional date filtering"""
        try:
            aum_data = self.calculator.get_aum_timeseries(start_date, end_date)
            
            if aum_data.empty:
                return None
            
            # Calculate daily returns for the filtered data
            aum_data = aum_data.copy()
            aum_data['prev_aum'] = aum_data['total_aum'].shift(1)
            aum_data['daily_return'] = (aum_data['total_aum'] - aum_data['prev_aum']) / aum_data['prev_aum']
            daily_returns = aum_data['daily_return'].dropna()
            
            if len(daily_returns) == 0:
                return None
                
            # Calculate drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max * 100
            
            fig = go.Figure()
            
            # Drawdown area
            fig.add_trace(go.Scatter(
                x=aum_data['date'].iloc[1:],  # Skip first date since first return is NaN
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                fillcolor='rgba(255, 107, 107, 0.3)',
                line=dict(color=self.colors['negative'], width=1),
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ))
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color=self.colors['text'], opacity=0.5)
            
            fig.update_layout(
                **self._get_layout_template(),
                title='Portfolio Drawdown Analysis',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating drawdown chart: {str(e)}")
            return None
    
    def create_monthly_returns_heatmap(self) -> Optional[go.Figure]:
        """Create monthly returns heatmap"""
        try:
            aum_data = self.calculator.daily_aum_df
            
            if aum_data.empty or 'daily_return' not in aum_data.columns:
                return None
            
            # Calculate monthly returns
            aum_data['year'] = aum_data['date'].dt.year
            aum_data['month'] = aum_data['date'].dt.month
            
            monthly_returns = aum_data.groupby(['year', 'month'])['daily_return'].apply(
                lambda x: (1 + x).prod() - 1
            ).reset_index()
            monthly_returns['return_pct'] = monthly_returns['daily_return'] * 100
            
            # Pivot for heatmap
            heatmap_data = monthly_returns.pivot(
                index='year', 
                columns='month', 
                values='return_pct'
            )
            
            # Month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=[month_names[i-1] for i in heatmap_data.columns],
                y=heatmap_data.index,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(heatmap_data.values, 2),
                texttemplate='%{text}%',
                textfont={"size": 10},
                hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
            ))
            
            fig.update_layout(
                **self._get_layout_template(),
                title='Monthly Returns Heatmap',
                xaxis_title='Month',
                yaxis_title='Year'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating monthly returns heatmap: {str(e)}")
            return None
    
    def create_rolling_metrics_chart(self) -> Optional[go.Figure]:
        """Create rolling performance metrics chart"""
        try:
            aum_data = self.calculator.daily_aum_df
            
            if aum_data.empty or 'daily_return' not in aum_data.columns:
                return None
            
            # Calculate rolling metrics
            returns = aum_data['daily_return'].dropna()
            
            # 30-day rolling volatility
            rolling_vol_30 = returns.rolling(30).std() * np.sqrt(365) * 100
            
            # 30-day rolling Sharpe
            rf_daily = self.calculator.risk_free_rate / 365
            excess_returns = returns - rf_daily
            rolling_sharpe_30 = excess_returns.rolling(30).mean() / excess_returns.rolling(30).std() * np.sqrt(365)
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('30-Day Rolling Volatility (%)', '30-Day Rolling Sharpe Ratio'),
                vertical_spacing=0.1
            )
            
            # Volatility
            fig.add_trace(go.Scatter(
                x=aum_data['date'][1:],  # Skip first day
                y=rolling_vol_30,
                mode='lines',
                name='30-Day Volatility',
                line=dict(color=self.colors['line_colors'][0], width=2),
                hovertemplate='Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
            ), row=1, col=1)
            
            # Sharpe ratio
            fig.add_trace(go.Scatter(
                x=aum_data['date'][1:],
                y=rolling_sharpe_30,
                mode='lines',
                name='30-Day Sharpe',
                line=dict(color=self.colors['line_colors'][1], width=2),
                hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'
            ), row=2, col=1)
            
            # Zero line for Sharpe
            fig.add_hline(y=0, line_dash="dash", line_color=self.colors['text'], opacity=0.5, row=2, col=1)
            
            fig.update_layout(
                **self._get_layout_template(),
                title='Rolling Performance Metrics',
                showlegend=False,
                height=600
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating rolling metrics chart: {str(e)}")
            return None
    
    def create_asset_detail_chart(self, ticker: str) -> Optional[go.Figure]:
        """Create asset detail chart with trade markers"""
        try:
            # Get market data for the asset
            if ticker not in self.calculator.market_data:
                return None
            
            market_data = self.calculator.market_data[ticker]
            trades = self.calculator.get_asset_trades(ticker)
            
            if market_data.empty:
                return None
            
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=market_data.index,
                y=market_data['Adj Close'],
                mode='lines',
                name=f'{ticker} Price',
                line=dict(color=self.colors['line_colors'][0], width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
            
            # Trade markers
            if not trades.empty:
                buy_trades = trades[trades['Buy_sell'] == 'BUY']
                sell_trades = trades[trades['Buy_sell'] == 'SELL']
                
                # Buy markers
                if not buy_trades.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_trades['Date'],
                        y=buy_trades['Price'],
                        mode='markers',
                        name='Buy Orders',
                        marker=dict(
                            color=self.colors['positive'],
                            size=10,
                            symbol='triangle-up'
                        ),
                        hovertemplate='Date: %{x}<br>Buy Price: $%{y:.2f}<br>Quantity: %{text}<extra></extra>',
                        text=buy_trades['Quantity']
                    ))
                
                # Sell markers
                if not sell_trades.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_trades['Date'],
                        y=sell_trades['Price'],
                        mode='markers',
                        name='Sell Orders',
                        marker=dict(
                            color=self.colors['negative'],
                            size=10,
                            symbol='triangle-down'
                        ),
                        hovertemplate='Date: %{x}<br>Sell Price: $%{y:.2f}<br>Quantity: %{text}<extra></extra>',
                        text=sell_trades['Quantity']
                    ))
            
            fig.update_layout(
                **self._get_layout_template(),
                title=f'{ticker} Price Chart with Trade Markers',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating asset detail chart: {str(e)}")
            return None
    
    def create_detailed_benchmark_chart(self) -> Optional[go.Figure]:
        """Create detailed benchmark comparison chart"""
        return self.create_benchmark_comparison_chart()
    
    def create_excess_returns_chart(self) -> Optional[go.Figure]:
        """Create excess returns vs benchmark chart"""
        try:
            aum_data = self.calculator.daily_aum_df
            
            if aum_data.empty or 'daily_return' not in aum_data.columns:
                return None
            
            # Calculate excess returns (placeholder - would need actual benchmark data)
            portfolio_returns = aum_data['daily_return'].fillna(0)
            excess_returns = portfolio_returns * 100  # Placeholder
            
            fig = go.Figure()
            
            # Excess returns
            fig.add_trace(go.Scatter(
                x=aum_data['date'][1:],
                y=excess_returns[1:],
                mode='lines',
                name='Excess Returns vs SPY',
                line=dict(color=self.colors['line_colors'][0], width=2),
                hovertemplate='Date: %{x}<br>Excess Return: %{y:.2f}%<extra></extra>'
            ))
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color=self.colors['text'], opacity=0.5)
            
            fig.update_layout(
                **self._get_layout_template(),
                title='Excess Returns vs Benchmarks',
                xaxis_title='Date',
                yaxis_title='Excess Return (%)',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating excess returns chart: {str(e)}")
            return None
