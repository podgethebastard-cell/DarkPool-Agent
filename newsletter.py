
import streamlit as st
import ccxt
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import scipy.stats as stats
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Hybrid Financial Dashboard", layout="wide")

st.title("📈 Hybrid Financial Dashboard")
st.markdown("""
This dashboard integrates **Crypto** and **Stock** markets into a single view. 
It uses `scipy` for statistical trend analysis and `plotly` for interactive visualization.
""")

# -----------------------------------------------------------------------------
# 2. UTILITY FUNCTIONS (Requests, Scipy, Numpy)
# -----------------------------------------------------------------------------

def check_connectivity():
    """
    Uses the 'requests' library explicitly to check internet status.
    """
    try:
        # Check a reliable public DNS or site
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except requests.RequestException:
        return False

def calculate_trend_slope(prices):
    """
    Uses 'scipy.stats.linregress' to calculate the slope of the trend.
    Returns the slope and the regression line points.
    """
    y = np.array(prices)
    x = np.arange(len(y))
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Generate the line for plotting
    trend_line = slope * x + intercept
    return slope, trend_line, r_value

# -----------------------------------------------------------------------------
# 3. DATA LOADING FUNCTIONS (CCXT, YFinance, Pandas)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_crypto_data(symbol, timeframe, limit):
    """
    Fetches Crypto data using 'ccxt'.
    SWITCHED TO KRAKEN TO AVOID GEO-BLOCKING (451 Errors).
    """
    try:
        # Switched from binance() to kraken() for better US/Global public access
        exchange = ccxt.kraken()
        
        # Fetch OHLCV data
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        # Convert to Pandas DataFrame
        df = pd.DataFrame(bars, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching Crypto data from Kraken: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_stock_data(ticker, period):
    """
    Fetches Stock data using 'yfinance'.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        # YFinance returns date as index, reset it to column for consistency
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'timestamp'}, inplace=True)
        
        # Ensure columns match our expected format
        if 'Stock Splits' in df.columns:
            df = df.drop(columns=['Dividends', 'Stock Splits'])
            
        return df
    except Exception as e:
        st.error(f"Error fetching Stock data: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 4. SIDEBAR & INPUTS
# -----------------------------------------------------------------------------

st.sidebar.header("Configuration")

# Explicit usage of requests library check
if check_connectivity():
    st.sidebar.success("✅ Network Status: Online (checked via `requests`)")
else:
    st.sidebar.error("❌ Network Status: Offline")

market_type = st.sidebar.radio("Select Market", ["Cryptocurrency", "Stocks"])

data = pd.DataFrame()
symbol = ""

if market_type == "Cryptocurrency":
    st.sidebar.subheader("Crypto Settings")
    st.sidebar.info("Using **Kraken** API (avoids Geo-blocks).")
    # Changed default to BTC/USD which is more common on Kraken than USDT
    symbol = st.sidebar.text_input("Symbol (e.g., BTC/USD)", value="BTC/USD")
    timeframe = st.sidebar.selectbox("Timeframe", ["1d", "4h", "1h", "15m"], index=0)
    limit = st.sidebar.slider("Data Points (Limit)", 50, 500, 100)
    
    if st.sidebar.button("Fetch Crypto Data"):
        with st.spinner("Fetching from CCXT (Kraken)..."):
            data = get_crypto_data(symbol, timeframe, limit)

else:
    st.sidebar.subheader("Stock Settings")
    symbol = st.sidebar.text_input("Ticker (e.g., AAPL, TSLA)", value="AAPL")
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    
    if st.sidebar.button("Fetch Stock Data"):
        with st.spinner("Fetching from YFinance..."):
            data = get_stock_data(symbol, period)

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD LOGIC
# -----------------------------------------------------------------------------

if not data.empty:
    # --- A. Data Processing & Statistics (Numpy/Scipy) ---
    st.subheader(f"Analysis for {symbol}")
    
    # Calculate Returns using Numpy
    data['Returns'] = data['Close'].pct_change()
    
    # Calculate Linear Regression Trend using Scipy
    slope, trend_line, r_squared = calculate_trend_slope(data['Close'].values)
    
    # Layout Columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"{data['Close'].iloc[-1]:.2f}")
    col2.metric("Trend Slope (Scipy)", f"{slope:.4f}", help="Positive = Upward Trend, Negative = Downward Trend")
    col3.metric("R-Squared", f"{r_squared**2:.4f}", help="Statistical strength of the trend")

    # --- B. Interactive Chart (Plotly) ---
    st.write("### Interactive Price Chart & Trend Line")
    fig = go.Figure()

    # Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=data['timestamp'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ))

    # Trend Line Trace (Scipy result)
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=trend_line,
        mode='lines',
        name='Linear Regression Trend (Scipy)',
        line=dict(color='orange', width=2, dash='dash')
    ))

    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- C. Statistical Distribution (Matplotlib) ---
    st.write("### Daily Returns Distribution (Matplotlib)")
    
    # Create Matplotlib Figure
    fig_mpl, ax = plt.subplots(figsize=(10, 4))
    
    # Clean NaNs for histogram
    clean_returns = data['Returns'].dropna()
    
    # Plot Histogram
    ax.hist(clean_returns, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title("Distribution of Daily Returns")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', alpha=0.5)
    
    # Add vertical line for mean
    mean_ret = np.mean(clean_returns)
    ax.axvline(mean_ret, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_ret:.4f}')
    ax.legend()
    
    # Render Matplotlib in Streamlit
    st.pyplot(fig_mpl)

    # --- D. Raw Data View ---
    with st.expander("View Raw Data"):
        st.dataframe(data)

elif symbol:
    st.info("Click the 'Fetch Data' button in the sidebar to generate the report.")
