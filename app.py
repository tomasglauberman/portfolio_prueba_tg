import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
from data_processor import DataProcessor
from portfolio_calculator import PortfolioCalculator
from market_data import MarketDataFetcher
from visualizations import PortfolioVisualizations
from utils import format_currency, format_percentage, validate_csv

# Page configuration
st.set_page_config(
    page_title="Professional Portfolio Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with better defaults
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'portfolio_calc' not in st.session_state:
    st.session_state.portfolio_calc = None
if 'market_data' not in st.session_state:
    st.session_state.market_data = {}
if 'cost_basis_method' not in st.session_state:
    st.session_state.cost_basis_method = 'FIFO'
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'light'
if 'selected_timeframe' not in st.session_state:
    st.session_state.selected_timeframe = 'All'
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = None

def get_timeframe_dates(timeframe: str, latest_date: datetime):
    """Calculate start and end dates based on timeframe selection"""
    if timeframe == 'All':
        return None, None  # No filtering
    
    end_date = latest_date
    
    if timeframe == '1D':
        start_date = end_date - timedelta(days=1)
    elif timeframe == '1W':
        start_date = end_date - timedelta(days=7)
    elif timeframe == '1M':
        start_date = end_date - timedelta(days=30)
    elif timeframe == '6M':
        start_date = end_date - timedelta(days=180)
    elif timeframe == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
    elif timeframe == '1Y':
        start_date = end_date - timedelta(days=365)
    else:
        return None, None
        
    return start_date, end_date

def main():
    st.title("📈 Professional Portfolio Tracker")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # Theme toggle
        theme_mode = st.selectbox("Theme", ["Light", "Dark"], 
                                 index=0 if st.session_state.theme_mode == 'light' else 1)
        if theme_mode.lower() != st.session_state.theme_mode:
            st.session_state.theme_mode = theme_mode.lower()
            st.rerun()
        
        # Cost basis method toggle
        cost_basis_method = st.selectbox(
            "Cost Basis Method",
            ["FIFO", "Average Cost"],
            index=0 if st.session_state.cost_basis_method == 'FIFO' else 1,
            help="Choose between First-In-First-Out (FIFO) or Average Cost basis calculation"
        )
        
        if cost_basis_method != st.session_state.cost_basis_method:
            st.session_state.cost_basis_method = cost_basis_method
            if st.session_state.portfolio_calc:
                st.session_state.portfolio_calc.cost_basis_method = cost_basis_method
                st.rerun()
        
        # Time frame filtering
        time_frames = ['1D', '1W', '1M', '6M', 'YTD', '1Y', 'All']
        selected_timeframe = st.selectbox(
            "Time Frame",
            time_frames,
            index=time_frames.index('All'),
            help="Select time frame for performance metrics and charts"
        )
        
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = 'All'
        
        if selected_timeframe != st.session_state.selected_timeframe:
            st.session_state.selected_timeframe = selected_timeframe
            st.rerun()
        
        # Use default values for risk-free rate and fees
        risk_free_rate = 0.02  # 2% default risk-free rate
        fee_percentage = 0.0   # No trading fees
        
        st.markdown("---")
        
        # CSV Upload
        st.header("Upload Portfolio Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type="csv",
            help="Upload your portfolio transactions CSV file"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                validation_result = validate_csv(df)
                
                if validation_result['valid']:
                    with st.spinner('Processing portfolio data...'):
                        # Initialize data processor
                        st.session_state.data_processor = DataProcessor(df)
                        
                        # Fetch market data and cache it
                        market_fetcher = MarketDataFetcher()
                        st.session_state.market_data = market_fetcher.fetch_all_data(
                            st.session_state.data_processor.get_unique_tickers(),
                            st.session_state.data_processor.get_date_range()
                        )
                        
                        # Initialize portfolio calculator
                        st.session_state.portfolio_calc = PortfolioCalculator(
                            st.session_state.data_processor,
                            st.session_state.market_data,
                            cost_basis_method,
                            risk_free_rate,
                            fee_percentage
                        )
                        
                        # Initialize visualizations
                        st.session_state.visualizations = PortfolioVisualizations(
                            st.session_state.portfolio_calc, 
                            st.session_state.theme_mode
                        )
                        
                    st.success("✅ Portfolio data loaded successfully!")
                    st.info(f"Loaded {len(df)} transactions")
                    
                else:
                    st.error("❌ CSV Validation Failed:")
                    for error in validation_result['errors']:
                        st.error(f"• {error}")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Main content area
    if st.session_state.portfolio_calc is None:
        st.info("👆 Please upload your portfolio CSV file to get started")
        
        # Show sample CSV format
        st.subheader("Expected CSV Format")
        sample_data = {
            'Date': ['01-01-2025', '02-01-2025', '05-01-2025'],
            'Company': ['CASH', 'Apple Inc', 'Microsoft Corp'],
            'Ticker': ['CASH', 'AAPL', 'MSFT'],
            'Buy_sell': ['BUY', 'BUY', 'BUY'],
            'Price': [1.0, 100.0, 200.0],
            'Quantity': [20000, 50, 30]
        }
        st.dataframe(pd.DataFrame(sample_data), width='stretch')
        
        st.markdown("""
        **Requirements:**
        - Date format: dd-mm-yyyy
        - Buy_sell: BUY or SELL
        - All Price and Quantity values must be positive
        - Cash deposits: CASH,CASH,BUY,1.0,amount
        - Cash withdrawals: CASH,CASH,SELL,1.0,amount
        - Supported crypto: BTC-USD, ETH-USD, SOL-USD
        """)
        return
    
    # Portfolio loaded - show main interface
    calculator = st.session_state.portfolio_calc
    visualizations = PortfolioVisualizations(calculator, st.session_state.theme_mode)
    
    # Get timeframe dates for filtering
    latest_date = datetime.now()
    if len(calculator.daily_aum_df) > 0:
        latest_date = calculator.daily_aum_df['date'].max()
    
    start_date, end_date = get_timeframe_dates(st.session_state.selected_timeframe, latest_date)
    
    # Header KPIs
    display_header_kpis(calculator, start_date, end_date)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", "📈 Performance", "💼 Holdings", 
        "📋 Trades", "🔍 Asset Detail", "📊 Benchmarks", "📄 Reports"
    ])
    
    with tab1:
        display_overview_tab(calculator, visualizations, start_date, end_date)
    
    with tab2:
        display_performance_tab(calculator, visualizations, start_date, end_date)
    
    with tab3:
        display_holdings_tab(calculator, visualizations, start_date, end_date)
    
    with tab4:
        display_trades_tab(calculator, visualizations, start_date, end_date)
    
    with tab5:
        display_asset_detail_tab(calculator, visualizations, start_date, end_date)
    
    with tab6:
        display_benchmarks_tab(calculator, visualizations, start_date, end_date)
    
    with tab7:
        display_reports_tab(calculator, visualizations, start_date, end_date)

def display_header_kpis(calculator, start_date=None, end_date=None):
    """Display header KPI metrics with optional time filtering"""
    try:
        current_aum = calculator.get_current_aum()
        cash_position = calculator.get_cash_position()
        aum_excluding_cash = current_aum - cash_position
        
        # Get realized and unrealized P&L
        holdings = calculator.get_current_holdings()
        unrealized_pnl = holdings['Unrealized P&L'].sum() if not holdings.empty else 0
        realized_pnl = calculator.get_realized_pnl()
        
        # Calculate % of P&L for the timeframe
        performance_metrics = calculator.get_performance_metrics(start_date, end_date)
        timeframe_return = performance_metrics.get('total_return', 0)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total AUM", format_currency(current_aum))
        
        with col2:
            st.metric("AUM Excluding Cash", format_currency(aum_excluding_cash))
        
        with col3:
            st.metric("Cash Position", format_currency(cash_position))
        
        with col4:
            st.metric("Realized P&L", format_currency(realized_pnl))
        
        with col5:
            st.metric("Unrealized P&L", format_currency(unrealized_pnl))
        
        with col6:
            timeframe_label = st.session_state.get('selected_timeframe', 'All')
            st.metric(f"% of P&L ({timeframe_label})", format_percentage(timeframe_return))
            
    except Exception as e:
        st.error(f"Error calculating KPIs: {str(e)}")

def display_overview_tab(calculator, visualizations, start_date=None, end_date=None):
    """Display overview tab content"""
    st.header("Portfolio Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # AUM Evolution Chart
        st.subheader("AUM Evolution")
        aum_chart = visualizations.create_aum_chart(start_date, end_date)
        if aum_chart:
            st.plotly_chart(aum_chart, use_container_width=True)
    
    with col2:
        # Current Allocation (excluding cash)
        st.subheader("Portfolio Allocation (Excluding Cash)")
        allocation_chart = visualizations.create_allocation_chart_without_cash()
        if allocation_chart:
            st.plotly_chart(allocation_chart, use_container_width=True)
    
    # Performance vs Benchmarks
    st.subheader("Performance vs Benchmarks")
    benchmark_chart = visualizations.create_benchmark_comparison_chart(start_date, end_date)
    if benchmark_chart:
        st.plotly_chart(benchmark_chart, use_container_width=True)
    
    # Drawdown Chart
    st.subheader("Drawdown Analysis")
    drawdown_chart = visualizations.create_drawdown_chart(start_date, end_date)
    if drawdown_chart:
        st.plotly_chart(drawdown_chart, use_container_width=True)

def display_performance_tab(calculator, visualizations, start_date=None, end_date=None):
    """Display performance tab content"""
    st.header("Performance Analysis")
    
    # Performance metrics table
    st.subheader("Key Performance Metrics")
    metrics = calculator.get_detailed_performance_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Return Metrics**")
        for key, value in metrics.get('returns', {}).items():
            if isinstance(value, (int, float)):
                st.metric(key.replace('_', ' ').title(), format_percentage(value))
    
    with col2:
        st.markdown("**Risk Metrics**")
        for key, value in metrics.get('risk', {}).items():
            if isinstance(value, (int, float)):
                if 'ratio' in key.lower():
                    st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                else:
                    st.metric(key.replace('_', ' ').title(), format_percentage(value))
    
    # Monthly returns heatmap
    st.subheader("Monthly Returns Heatmap")
    heatmap_chart = visualizations.create_monthly_returns_heatmap()
    if heatmap_chart:
        st.plotly_chart(heatmap_chart, use_container_width=True)
    
    # Rolling performance metrics
    st.subheader("Rolling Performance Metrics")
    rolling_chart = visualizations.create_rolling_metrics_chart()
    if rolling_chart:
        st.plotly_chart(rolling_chart, use_container_width=True)

def display_holdings_tab(calculator, visualizations, start_date=None, end_date=None):
    """Display holdings tab content"""
    st.header("Current Holdings")
    
    # Holdings table
    holdings_df = calculator.get_current_holdings()
    if not holdings_df.empty:
        # Format the dataframe for display
        display_df = holdings_df.copy()
        
        # Format currency and percentage columns
        currency_cols = ['Market Value', 'Cost Basis', 'Unrealized P&L']
        percentage_cols = ['Weight %', 'Return %']
        
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(format_currency)
        
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(format_percentage)
        
        st.dataframe(display_df, width='stretch')
        
        # Lot-level detail expander
        st.subheader("Lot-Level Detail")
        selected_ticker = st.selectbox(
            "Select ticker for lot detail:",
            options=holdings_df['Ticker'].unique()
        )
        
        if selected_ticker:
            lots_df = calculator.get_lot_detail(selected_ticker)
            if not lots_df.empty:
                st.dataframe(lots_df, width='stretch')
            else:
                st.info("No lot details available for this ticker")
    else:
        st.info("No current holdings")

def display_trades_tab(calculator, visualizations, start_date=None, end_date=None):
    """Display trades tab content"""
    st.header("Transaction History")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    trades_df = calculator.data_processor.df.copy()
    
    with col1:
        # Ticker filter
        tickers = ['All'] + list(trades_df['Ticker'].unique())
        selected_ticker = st.selectbox("Filter by Ticker", tickers)
    
    with col2:
        # Transaction type filter
        transaction_types = ['All', 'BUY', 'SELL']
        selected_type = st.selectbox("Filter by Type", transaction_types)
    
    with col3:
        # Date range filter
        min_date = trades_df['Date'].min()
        max_date = trades_df['Date'].max()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Apply filters
    filtered_df = trades_df.copy()
    
    if selected_ticker != 'All':
        filtered_df = filtered_df[filtered_df['Ticker'] == selected_ticker]
    
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['Buy_sell'] == selected_type]
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['Date'] >= pd.to_datetime(date_range[0])) &
            (filtered_df['Date'] <= pd.to_datetime(date_range[1]))
        ]
    
    # Display filtered trades
    if not filtered_df.empty:
        # Format for display - calculate total value BEFORE formatting
        display_df = filtered_df.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%d-%m-%Y')
        display_df['Total Value'] = (display_df['Price'] * display_df['Quantity']).apply(format_currency)
        display_df['Price'] = display_df['Price'].apply(format_currency)
        
        st.dataframe(display_df, width='stretch')
        
        # Trade summary
        st.subheader("Trade Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(filtered_df))
        with col2:
            buy_trades = len(filtered_df[filtered_df['Buy_sell'] == 'BUY'])
            st.metric("Buy Trades", buy_trades)
        with col3:
            sell_trades = len(filtered_df[filtered_df['Buy_sell'] == 'SELL'])
            st.metric("Sell Trades", sell_trades)
        with col4:
            unique_tickers = filtered_df['Ticker'].nunique()
            st.metric("Unique Tickers", unique_tickers)
    else:
        st.info("No trades match the selected filters")

def display_asset_detail_tab(calculator, visualizations, start_date=None, end_date=None):
    """Display asset detail tab content"""
    st.header("Asset Detail View")
    
    # Asset selection
    available_tickers = calculator.data_processor.get_unique_tickers()
    available_tickers = [t for t in available_tickers if t != 'CASH']
    
    if not available_tickers:
        st.info("No assets available for detail view")
        return
    
    selected_asset = st.selectbox("Select Asset", available_tickers)
    
    if selected_asset:
        # Asset price chart with trade markers
        st.subheader(f"{selected_asset} Price Chart with Trade Markers")
        asset_chart = visualizations.create_asset_detail_chart(selected_asset)
        if asset_chart:
            st.plotly_chart(asset_chart, use_container_width=True)
        
        # Asset performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Asset Performance")
            asset_metrics = calculator.get_asset_performance_metrics(selected_asset)
            
            for key, value in asset_metrics.items():
                if isinstance(value, (int, float)):
                    if 'return' in key.lower() or '%' in key:
                        st.metric(key.replace('_', ' ').title(), format_percentage(value))
                    elif 'price' in key.lower() or 'value' in key.lower():
                        st.metric(key.replace('_', ' ').title(), format_currency(value))
                    else:
                        st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
        
        with col2:
            st.subheader("Trade History")
            asset_trades = calculator.get_asset_trades(selected_asset)
            if not asset_trades.empty:
                display_trades = asset_trades.copy()
                display_trades['Date'] = display_trades['Date'].dt.strftime('%d-%m-%Y')
                display_trades['Price'] = display_trades['Price'].apply(format_currency)
                st.dataframe(display_trades, width='stretch')
            else:
                st.info("No trades found for this asset")

def display_benchmarks_tab(calculator, visualizations, start_date=None, end_date=None):
    """Display benchmarks comparison tab"""
    st.header("Benchmark Comparison")
    
    # Benchmark performance metrics
    benchmark_metrics = calculator.get_benchmark_comparison()
    
    if benchmark_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("vs SPY")
            spy_metrics = benchmark_metrics.get('SPY', {})
            for key, value in spy_metrics.items():
                if isinstance(value, (int, float)):
                    if 'correlation' in key.lower() or 'beta' in key.lower():
                        st.metric(key.replace('_', ' ').title(), f"{value:.3f}")
                    else:
                        st.metric(key.replace('_', ' ').title(), format_percentage(value))
        
        with col2:
            st.subheader("vs QQQ")
            qqq_metrics = benchmark_metrics.get('QQQ', {})
            for key, value in qqq_metrics.items():
                if isinstance(value, (int, float)):
                    if 'correlation' in key.lower() or 'beta' in key.lower():
                        st.metric(key.replace('_', ' ').title(), f"{value:.3f}")
                    else:
                        st.metric(key.replace('_', ' ').title(), format_percentage(value))
    
    # Benchmark comparison chart
    st.subheader("Cumulative Performance Comparison")
    benchmark_chart = visualizations.create_detailed_benchmark_chart()
    if benchmark_chart:
        st.plotly_chart(benchmark_chart, use_container_width=True, key="benchmark_comparison_chart")
    
    # Excess returns
    st.subheader("Excess Returns Analysis")
    excess_returns_chart = visualizations.create_excess_returns_chart()
    if excess_returns_chart:
        st.plotly_chart(excess_returns_chart, use_container_width=True, key="excess_returns_analysis_chart")

def display_reports_tab(calculator, visualizations, start_date=None, end_date=None):
    """Display reports and export tab"""
    st.header("Reports & Export")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Export")
        
        # AUM export
        if st.button("Export AUM Data (CSV)"):
            aum_data = calculator.get_aum_timeseries()
            csv = aum_data.to_csv(index=False)
            st.download_button(
                label="Download AUM CSV",
                data=csv,
                file_name="portfolio_aum.csv",
                mime="text/csv"
            )
        
        # Holdings export
        if st.button("Export Current Holdings (CSV)"):
            holdings_data = calculator.get_current_holdings()
            csv = holdings_data.to_csv(index=False)
            st.download_button(
                label="Download Holdings CSV",
                data=csv,
                file_name="current_holdings.csv",
                mime="text/csv"
            )
        
        # Trades export
        if st.button("Export All Trades (CSV)"):
            trades_data = calculator.data_processor.df
            csv = trades_data.to_csv(index=False)
            st.download_button(
                label="Download Trades CSV",
                data=csv,
                file_name="all_trades.csv",
                mime="text/csv"
            )
    
    with col2:
        st.subheader("Performance Summary")
        
        # Key metrics summary
        performance_metrics = calculator.get_detailed_performance_metrics()
        current_aum = calculator.get_current_aum()
        
        summary_text = f"""
        **Portfolio Performance Summary**
        
        📊 **Portfolio Overview**
        - Total AUM: {format_currency(current_aum)}
        - Cost Basis Method: {st.session_state.cost_basis_method}
        
        📈 **Return Metrics**
        - Total Return (TWR): {format_percentage(performance_metrics.get('returns', {}).get('twr', 0))}
        - YTD Return: {format_percentage(performance_metrics.get('returns', {}).get('ytd_return', 0))}
        - Annualized Return: {format_percentage(performance_metrics.get('returns', {}).get('annualized_return', 0))}
        
        ⚡ **Risk Metrics**
        - Volatility: {format_percentage(performance_metrics.get('risk', {}).get('volatility', 0))}
        - Sharpe Ratio: {performance_metrics.get('risk', {}).get('sharpe_ratio', 0):.2f}
        - Max Drawdown: {format_percentage(performance_metrics.get('risk', {}).get('max_drawdown', 0))}
        
        📊 **Benchmark Comparison**
        - Alpha vs SPY: {format_percentage(performance_metrics.get('benchmark', {}).get('spy_alpha', 0))}
        - Beta vs SPY: {performance_metrics.get('benchmark', {}).get('spy_beta', 0):.2f}
        - Correlation vs SPY: {performance_metrics.get('benchmark', {}).get('spy_correlation', 0):.3f}
        """
        
        st.markdown(summary_text)

if __name__ == "__main__":
    main()
