import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time

# ==========================================
# 1. PAGE CONFIGURATION & TITAN UI ENGINE
# ==========================================
st.set_page_config(
    page_title="Crypto Macro Titan",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TITAN DARKPOOL CSS ARCHITECTURE ---
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Neon Glow Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
        font-weight: 800;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #00ff88;
        transform: translateY(-2px);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #00ff88 !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #8b949e !important;
        letter-spacing: 1px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-bottom: none;
        color: #8b949e;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1117;
        color: #00ff88;
        border-top: 2px solid #00ff88;
    }

    /* Custom Event Card */
    .event-card {
        background: #161b22;
        border-left: 4px solid #00ff88;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .event-date { color: #00ff88; font-weight: bold; font-size: 0.8rem; }
    .event-title { color: #fff; font-weight: bold; font-size: 1.1rem; }
    .event-desc { color: #8b949e; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA INGESTION ENGINE
# ==========================================

@st.cache_data(ttl=300)
def fetch_crypto_price_ccxt(symbol, timeframe='1d', limit=365):
    """
    Fetches Crypto OHLCV data using CCXT (Kraken Public API).
    Kraken is used for reliability without API keys.
    """
    try:
        exchange = ccxt.kraken()
        # CCXT symbol format is usually 'BTC/USD'
        formatted_symbol = symbol.replace('-', '/')
        ohlcv = exchange.fetch_ohlcv(formatted_symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"CCXT Error ({symbol}): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_macro_data_yf(ticker, period='1y'):
    """
    Fetches Traditional Finance Macro Data via yfinance.
    """
    try:
        df = yf.download(ticker, period=period, progress=False)
        # Handle MultiIndex if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Standardize column names
        df.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
        # Ensure lowercase for consistency
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"YFinance Error ({ticker}): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_fear_greed_index():
    """
    Fetches the Crypto Fear & Greed Index from Alternative.me.
    """
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url)
        data = response.json()
        return data['data'][0]
    except Exception as e:
        return {"value": "50", "value_classification": "Neutral"}

# ==========================================
# 3. MATHEMATICAL CORE & CORRELATION
# ==========================================

def align_and_correlate(crypto_df, macro_df, lookback=30):
    """
    Aligns timestamps between 24/7 Crypto markets and 5/7 TradFi markets.
    Calculates rolling correlation.
    """
    # Merge on index (date)
    # Inner join keeps only days where both markets were open (ignoring weekends for correlation accuracy)
    merged = pd.merge(crypto_df[['close']], macro_df[['close']], left_index=True, right_index=True, suffixes=('_crypto', '_macro'))
    
    # Calculate Rolling Correlation
    merged['correlation'] = merged['close_crypto'].rolling(window=lookback).corr(merged['close_macro'])
    
    # Normalize prices for comparison visualization (Start at 0%)
    merged['crypto_norm'] = (merged['close_crypto'] / merged['close_crypto'].iloc[0] - 1) * 100
    merged['macro_norm'] = (merged['close_macro'] / merged['close_macro'].iloc[0] - 1) * 100
    
    return merged

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("📡 Titan Controls")

crypto_asset = st.sidebar.selectbox("Crypto Asset", ["BTC/USD", "ETH/USD", "SOL/USD", "LINK/USD"], index=0)
macro_asset_map = {
    "DXY (US Dollar)": "DX-Y.NYB",
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Gold": "GC=F",
    "US 10Y Yield": "^TNX",
    "VIX (Volatility)": "^VIX"
}
macro_choice = st.sidebar.selectbox("Macro Correlation Asset", list(macro_asset_map.keys()))
macro_ticker = macro_asset_map[macro_choice]

lookback_period = st.sidebar.slider("Correlation Lookback (Days)", 14, 180, 30)
time_window = st.sidebar.selectbox("Chart Range", ["1y", "2y", "5y"], index=0)

# ==========================================
# 5. HEADER & SENTIMENT GAUGE
# ==========================================
st.title("⚡ Crypto Macro Titan")
st.caption("Institutional-Grade Correlation & Event Terminal")

# Fetch Data
with st.spinner("Establishing Uplink to Exchanges..."):
    df_crypto = fetch_crypto_price_ccxt(crypto_asset, limit=730 if time_window != '1y' else 365)
    df_macro = fetch_macro_data_yf(macro_ticker, period=time_window)
    fng_data = fetch_fear_greed_index()

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

# Live Prices
if not df_crypto.empty and not df_macro.empty:
    curr_crypto = df_crypto['close'].iloc[-1]
    prev_crypto = df_crypto['close'].iloc[-2]
    delta_crypto = ((curr_crypto - prev_crypto) / prev_crypto) * 100
    
    curr_macro = df_macro['close'].iloc[-1]
    prev_macro = df_macro['close'].iloc[-2]
    delta_macro = ((curr_macro - prev_macro) / prev_macro) * 100

    col1.metric(f"{crypto_asset}", f"${curr_crypto:,.2f}", f"{delta_crypto:+.2f}%")
    col2.metric(f"{macro_choice}", f"{curr_macro:,.2f}", f"{delta_macro:+.2f}%")

# Sentiment
fng_val = int(fng_data['value'])
fng_class = fng_data['value_classification']
fng_color = "#00ff88" if fng_val > 50 else "#ff0055" # Green greed, Red fear

col3.metric("Fear & Greed", f"{fng_val}/100", fng_class)

# Correlation Snapshot
aligned_data = align_and_correlate(df_crypto, df_macro, lookback_period)
if not aligned_data.empty:
    curr_corr = aligned_data['correlation'].iloc[-1]
    col4.metric(f"{lookback_period}D Correlation", f"{curr_corr:.2f}", "Strong" if abs(curr_corr) > 0.7 else "Weak")

# ==========================================
# 6. MAIN ANALYSIS TABS
# ==========================================
tab_main, tab_events, tab_matrix = st.tabs(["📊 Macro Correlation", "📅 Event Radar", "🧮 Statistical Matrix"])

# --- TAB 1: CORRELATION CHART ---
with tab_main:
    st.subheader(f"Price Action vs. {macro_choice}")
    
    if not aligned_data.empty:
        # Dual Axis Plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Trace 1: Crypto (Area)
        fig.add_trace(go.Scatter(
            x=aligned_data.index, 
            y=aligned_data['close_crypto'], 
            name=crypto_asset,
            line=dict(color='#00ff88', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ), secondary_y=False)
        
        # Trace 2: Macro (Line)
        fig.add_trace(go.Scatter(
            x=aligned_data.index, 
            y=aligned_data['close_macro'], 
            name=macro_choice,
            line=dict(color='#00aaff', width=2, dash='solid')
        ), secondary_y=True)
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            height=500,
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left")
        )
        fig.update_yaxes(title_text=crypto_asset, secondary_y=False, showgrid=True, gridcolor='#30363d')
        fig.update_yaxes(title_text=macro_choice, secondary_y=True, showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling Correlation Chart
        st.subheader(f"Rolling {lookback_period}-Day Correlation Coefficient")
        
        corr_fig = go.Figure()
        
        # Color logic for correlation (Green = Positive, Red = Negative)
        colors = np.where(aligned_data['correlation'] > 0, '#00ff88', '#ff0055')
        
        corr_fig.add_trace(go.Bar(
            x=aligned_data.index,
            y=aligned_data['correlation'],
            marker_color=colors,
            name='Correlation'
        ))
        
        corr_fig.add_hline(y=0, line_dash="dash", line_color="white")
        
        corr_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            height=300,
            yaxis_range=[-1, 1],
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(corr_fig, use_container_width=True)
        
    else:
        st.warning("Insufficient data intersection for correlation analysis.")

# --- TAB 2: EVENT RADAR ---
with tab_events:
    st.subheader("📡 Crypto-Macro Horizon")
    
    col_e1, col_e2 = st.columns([2, 1])
    
    with col_e1:
        # Simulated Event Data Structure (In a real app, integrate a Calendar API)
        events = [
            {"date": "2025-03-20", "title": "FOMC Interest Rate Decision", "type": "Macro", "impact": "High"},
            {"date": "2025-04-15", "title": "U.S. Tax Deadline", "type": "Liquidity", "impact": "Medium"},
            {"date": "2025-05-23", "title": "Ethereum Pectra Upgrade (Est)", "type": "Crypto", "impact": "High"},
            {"date": "2025-06-12", "title": "CPI Inflation Data Release", "type": "Macro", "impact": "High"},
            {"date": "2025-07-10", "title": "Large Token Unlock (APT/ARB)", "type": "Crypto", "impact": "Medium"},
        ]
        
        for e in events:
            # Color coding borders based on type
            border_color = "#00ff88" if e['type'] == "Crypto" else "#00aaff" if e['type'] == "Macro" else "#ffaa00"
            
            st.markdown(f"""
            <div class="event-card" style="border-left: 4px solid {border_color};">
                <div class="event-date">{e['date']} • {e['type']}</div>
                <div class="event-title">{e['title']}</div>
                <div class="event-desc">Expected Market Impact: {e['impact']}</div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_e2:
        st.info("""
        **Event Legend:**
        
        🟢 **Crypto Native:** Halvings, Upgrades, Unlocks.
        
        🔵 **Macro TradFi:** FOMC, CPI, PPI, Jobs Data.
        
        🟠 **Liquidity:** Tax dates, Treasury buybacks.
        """)
        
        st.markdown("---")
        st.metric("Days to Next FOMC", "14 Days", "-2 Days")

# --- TAB 3: STATISTICAL MATRIX ---
with tab_matrix:
    st.subheader("🧮 Asset Sensitivity Matrix")
    
    if not aligned_data.empty:
        # Calculate Beta
        # Beta = Covariance / Variance of Benchmark
        # Here we assume Macro asset is the benchmark for the calculation
        covariance = aligned_data['close_crypto'].rolling(60).cov(aligned_data['close_macro'])
        variance = aligned_data['close_macro'].rolling(60).var()
        beta = covariance / variance
        
        # Display Dataframe of recent stats
        stats_df = pd.DataFrame({
            "Date": aligned_data.index,
            "Price Crypto": aligned_data['close_crypto'],
            "Price Macro": aligned_data['close_macro'],
            "Correlation (30d)": aligned_data['correlation'],
            "Beta (60d)": beta
        }).sort_index(ascending=False).head(30)
        
        # Styling the dataframe for Titan look
        st.dataframe(
            stats_df.style.background_gradient(subset=['Correlation (30d)'], cmap='RdYlGn', vmin=-1, vmax=1)
            .format({"Price Crypto": "${:,.2f}", "Price Macro": "{:,.2f}", "Correlation (30d)": "{:.2f}", "Beta (60d)": "{:.2f}"}),
            use_container_width=True,
            height=600
        )
        
        st.caption("*Beta represents the volatility of Crypto relative to the selected Macro asset.*")

# ==========================================
# 7. FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b949e; font-size: 0.8rem;">
    TITAN TERMINAL | Data Sources: Kraken (CCXT), Yahoo Finance, Alternative.me | 
    <span style="color: #00ff88;">System Status: ONLINE</span>
</div>
""", unsafe_allow_html=True)
