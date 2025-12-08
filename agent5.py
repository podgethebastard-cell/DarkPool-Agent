import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Macro Insighter",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Excellent UI"
st.markdown("""
    <style>
    /* Metric Card Styling */
    .stMetric {
        background-color: #0E1117;
        border: 1px solid #262730;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stMetric:hover {
        border-color: #FF4B4B;
    }
    /* Remove padding at top */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA MAPPING (The Brain)
# -----------------------------------------------------------------------------
# Mapping your requirements to valid Yahoo Finance Tickers.
# Note: Economic data (CPI, M2) mapped to ETF proxies where direct feeds aren't on Yahoo.

TICKERS = {
    "15. MASTER CORE (The Non-Negotiables)": {
        "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Dollar Index (DXY)": "DX-Y.NYB",
        "US 10Y Yield": "^TNX", "US 02Y Yield": "^IRX", "VIX (Fear)": "^VIX",
        "Crude Oil (WTI)": "CL=F", "Gold": "GC=F", "Copper": "HG=F",
        "High Yield Corp (HYG)": "HYG", "Long Bonds (TLT)": "TLT",
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Global Liquidity (W M2)": "Wwow.L" # Proxy
    },
    "1. Global Equity Indices": {
        "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Dow Jones": "^DJI", "Russell 2000": "^RUT",
        "DAX (Germany)": "^GDAXI", "FTSE 100 (UK)": "^FTSE", "CAC 40 (France)": "^FCHI",
        "Nikkei 225 (Japan)": "^N225", "Hang Seng (HK)": "^HSI", "KOSPI (Korea)": "^KS11",
        "All-World (ACWI)": "ACWI", "Emerging Markets (EEM)": "EEM"
    },
    "2. Volatility & Fear": {
        "VIX (S&P 500)": "^VIX", "VXN (Nasdaq)": "^VXN", "VXD (Dow)": "^VXD",
        "MOVE Index (Bond Vol Proxy)": "MOVE.MX" # Note: Yahoo support for MOVE is spotty, sometimes requires specific ticker
    },
    "3. Interest Rates & Bonds": {
        "US 10Y Yield": "^TNX", "US 2Y Yield": "^IRX", "US 30Y Yield": "^TYX",
        "TLT (20Y+ Bond ETF)": "TLT", "IEF (7-10Y Bond ETF)": "IEF",
        "SHY (1-3Y Bond ETF)": "SHY", "LQD (Inv Grade)": "LQD",
        "HYG (High Yield)": "HYG", "TIP (Inflation Prot)": "TIP"
    },
    "4. Currencies (FX)": {
        "Dollar Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X", "USD/CNY": "USDCNY=X", "USD/MXN": "USDMXN=X"
    },
    "7. Commodities": {
        "WTI Crude": "CL=F", "Brent Crude": "BZ=F", "Natural Gas": "NG=F",
        "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F",
        "Corn": "ZC=F", "Wheat": "KE=F"
    },
    "11. Crypto Macro": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD",
        "Total Crypto Cap (Proxy)": "BTC-USD", # Yahoo lacks TOTAL index, using BTC as lead
    },
    "12. Risk Rotation Signals": {
        "Stocks vs Bonds (SPY/TLT)": "SPY", # Calculated field in logic
        "High Yield vs Safe (HYG/TLT)": "HYG",
        "Copper vs Gold": "HG=F"
    },
    "14. Geopolitics & Tail Risk": {
        "Defense (ITA)": "ITA", "Cybersecurity (HACK)": "HACK",
        "Energy (XLE)": "XLE", "Gold": "GC=F"
    }
}

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data(ttl=60) # Cache for 1 min to prevent spamming API
def get_market_data(ticker_dict, period="6mo"):
    """Fetches data for a dictionary of tickers."""
    symbols = list(ticker_dict.values())
    try:
        # Batch download is faster
        data = yf.download(symbols, period=period, progress=False)['Close']
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_metrics(df, ticker_symbol):
    """Calculates latest price and percentage change."""
    if ticker_symbol not in df.columns:
        return None, None, None
    
    series = df[ticker_symbol].dropna()
    if len(series) < 2:
        return None, None, None
    
    latest_price = series.iloc[-1]
    prev_price = series.iloc[-2]
    change = latest_price - prev_price
    pct_change = (change / prev_price) * 100
    
    return latest_price, change, pct_change, series

def plot_mini_chart(series, color):
    """Creates a sparkline chart using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f"rgba({','.join([str(c) for c in color_to_rgb(color)])}, 0.1)"
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=50,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def color_to_rgb(color_hex):
    # Simple helper for fill opacity
    if color_hex == '#00FF00': return (0, 255, 0)
    return (255, 0, 0)

# -----------------------------------------------------------------------------
# 4. MAIN APP LAYOUT
# -----------------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.header("📡 Macro Monitor")
    st.write("Institutional Risk Dashboard")
    selected_category = st.radio("Select Sector:", list(TICKERS.keys()))
    
    st.markdown("---")
    st.caption(f"Last Refresh: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("Refresh Data"):
        st.cache_data.clear()

# Main Content
st.title(f"{selected_category}")
st.markdown("---")

# Fetch Data
with st.spinner('Accessing Global Markets...'):
    market_data = get_market_data(TICKERS[selected_category])

# Display Grid
if not market_data.empty:
    # Logic to handle 3 columns grid
    cols = st.columns(3)
    
    for i, (name, ticker) in enumerate(TICKERS[selected_category].items()):
        col_idx = i % 3
        
        latest, change, pct, series = calculate_metrics(market_data, ticker)
        
        if latest is not None:
            with cols[col_idx]:
                # Determine Color
                color = "#00FF00" if change >= 0 else "#FF0000"
                
                # Render Metric Card
                with st.container():
                    st.metric(
                        label=name,
                        value=f"{latest:,.2f}",
                        delta=f"{pct:,.2f}%"
                    )
                    # Add Sparkline
                    st.plotly_chart(plot_mini_chart(series, color), use_container_width=True, config={'displayModeBar': False})
        else:
            with cols[col_idx]:
                st.warning(f"{name}: Data Unavailable")

# -----------------------------------------------------------------------------
# 5. SPECIAL: RATIO CHARTS (For Rotation Signals)
# -----------------------------------------------------------------------------
if selected_category == "12. Risk Rotation Signals":
    st.markdown("### ⚖️ Calculated Ratios (Visual)")
    
    # Custom fetching for Ratios
    spy = yf.Ticker("SPY").history(period="1y")['Close']
    tlt = yf.Ticker("TLT").history(period="1y")['Close']
    
    if not spy.empty and not tlt.empty:
        ratio = spy / tlt
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, mode='lines', name='SPY / TLT Ratio'))
        fig.update_layout(title="Risk On/Off: SPY (Stocks) vs TLT (Bonds)", height=400)
        st.plotly_chart(fig, use_container_width=True)
