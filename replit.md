# Overview

Professional Portfolio Tracker is a Streamlit-based web application for comprehensive investment portfolio analysis and tracking. The application allows users to upload CSV transaction data, automatically fetches market data, and provides detailed portfolio analytics including performance metrics, visualizations, and position tracking with multiple cost basis calculation methods.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Component Structure**: Single-page application with sidebar controls and main content area
- **State Management**: Streamlit session state for maintaining user preferences and data persistence
- **Theme Support**: Light/dark mode toggle with dynamic color scheme adaptation

### Data Processing Layer
- **DataProcessor Class**: Core data handling component that processes CSV uploads, validates transaction data, and creates separate dataframes for trades and cash flows
- **Date Handling**: Standardized date parsing with DD-MM-YYYY format support
- **Transaction Processing**: Automatic calculation of signed quantities and transaction values for buy/sell operations

### Portfolio Calculation Engine
- **PortfolioCalculator Class**: Advanced portfolio analytics engine supporting multiple cost basis methods (FIFO, Average Cost)
- **Position Tracking**: Lot-based position tracking for accurate cost basis calculations
- **Performance Metrics**: Comprehensive calculation of returns, risk metrics, and portfolio statistics
- **Risk Analysis**: Integration of risk-free rate for Sharpe ratio and other risk-adjusted metrics

### Market Data Integration
- **MarketDataFetcher Class**: Handles real-time market data retrieval using Yahoo Finance API
- **Caching Strategy**: In-memory caching system to minimize API calls and improve performance
- **Data Validation**: Error handling for missing or invalid market data with fallback mechanisms

### Visualization System
- **PortfolioVisualizations Class**: Plotly-based charting system for interactive portfolio analytics
- **Theme-Aware Rendering**: Dynamic color schemes that adapt to user's theme preference
- **Chart Types**: Support for portfolio value charts, allocation pie charts, performance comparisons, and risk analytics

### Utility Functions
- **Formatting Utilities**: Standardized currency, percentage, and number formatting with scaling (K, M, B)
- **CSV Validation**: Comprehensive validation system for uploaded transaction data
- **Error Handling**: Robust error handling with user-friendly error messages

## External Dependencies

### Market Data Services
- **Yahoo Finance (yfinance)**: Primary market data source for real-time and historical stock prices
- **Data Coverage**: Supports global stock markets with automatic symbol resolution

### Python Libraries
- **Core Framework**: Streamlit for web application framework
- **Data Processing**: Pandas and NumPy for data manipulation and numerical calculations
- **Visualization**: Plotly Express and Plotly Graph Objects for interactive charting
- **Mathematical Operations**: SciPy for advanced financial calculations and optimization
- **Date/Time Handling**: Python datetime library for temporal data processing

### File Processing
- **CSV Import**: Native pandas CSV reading with custom validation
- **Data Export**: Base64 encoding for downloadable reports and data exports

### Performance Optimization
- **Caching**: In-memory data caching for market data to reduce API latency
- **Data Persistence**: Streamlit session state for maintaining user data across interactions