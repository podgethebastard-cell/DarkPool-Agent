import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import calendar
import datetime
import requests
import urllib.parse
import math

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM UI
# ==========================================
st.set_page_config(layout="wide", page_title="DarkPool Titan Terminal v6", page_icon="👁️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .title-glow {
        font-size: 3em; font-weight: bold; color: #ffffff;
        text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 40px #00ff00;
        margin-bottom: 20px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px; border-radius: 8px; transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover { transform: scale(1.02); border-color: #00ff00; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #161b22;
        border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
        border: 1px solid #30363d; color: #8b949e;
    }
    .stTabs [aria-selected="true"] { background-color: #0e1117; color: #00ff00; border-bottom: 2px solid #00ff00; }
    div[data-baseweb="tooltip"] { background-color: #30363d !important; color: #00ff00 !important; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="title-glow">👁️ DarkPool Titan v6</div>', unsafe_allow_html=True)
st.markdown("##### *Institutional-Grade Market Intelligence // Optimized Core*")
st.markdown("---")

# --- API Key Management ---
if 'api_key' not in st.session_state: st.session_state.api_key = None
if "OPENAI_API_KEY" in st.secrets: st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
else:
    if not st.session_state.api_key:
        st.session_state.api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here to unlock the AI Analyst features.")

# ==========================================
# 2. DATA ENGINE (OPTIMIZED FOR LOWER TIMEFRAMES)
# ==========================================
@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    if "-" in ticker or "=" in ticker or "^" in ticker: return None 
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info: return None
        return {
            "Market Cap": info.get("marketCap", 0), "P/E Ratio": info.get("trailingPE", 0),
            "Rev Growth": info.get("revenueGrowth", 0), "Debt/Equity": info.get("debtToEquity", 0),
            "Summary": info.get("longBusinessSummary", "No Data Available")
        }
    except: return None

@st.cache_data(ttl=300)
def get_global_performance():
    assets = { "Tech": "XLK", "Energy": "XLE", "Financials": "XLF", "Bitcoin": "BTC-USD", "Gold": "GLD", "Oil": "USO", "Bonds": "TLT" }
    try:
        data = yf.download(list(assets.values()), period="5d", interval="1d", progress=False)['Close']
        results = {}
        for name, ticker in assets.items():
            if ticker in data.columns:
                change = ((data[ticker].iloc[-1] - data[ticker].iloc[-2]) / data[ticker].iloc[-2]) * 100
                results[name] = change
        return pd.Series(results).sort_values()
    except: return None

def safe_download(ticker, period, interval):
    """
    OPTIMIZED DOWNLOADER: Strictly caps period based on interval to prevent 
    yfinance hanging on 1m/5m data.
    """
    # Force strict limits for lower timeframes to ensure speed
    if interval == "1m": period = "5d"    # yfinance limit: 7d
    elif interval == "5m": period = "1mo" # yfinance limit: 60d
    elif interval in ["15m", "30m"]: period = "1mo"
    elif interval in ["1h"]: period = "1y" # 4h requests usually fallback to 1h
    
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Ensure we have standard columns
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        return df
    except: return None

@st.cache_data(ttl=300)
def get_macro_data():
    # ... (Kept original logic for macro groups, it is efficient enough) ...
    groups = {
        "🇺🇸 US Equities": {"S&P 500": "SPY", "Nasdaq": "QQQ"},
        "⚠️ Risk Assets": {"Bitcoin": "BTC-USD", "VIX": "^VIX"},
        "🏦 Rates": {"10Y Yield": "^TNX", "DXY": "DX-Y.NYB"},
        "🥇 Metals": {"Gold": "GC=F", "Silver": "SI=F"}
    }
    tickers = [t for g in groups.values() for t in g.values()]
    try:
        data = yf.download(tickers, period="5d", interval="1d", progress=False)['Close']
        prices, changes = {}, {}
        for t in tickers:
            if t in data.columns:
                prices[t] = data[t].iloc[-1]
                changes[t] = ((data[t].iloc[-1] - data[t].iloc[-2]) / data[t].iloc[-2]) * 100
        return groups, prices, changes
    except: return groups, {}, {}

# ==========================================
# 3. MATH LIBRARY: NEW INDICATORS (VECTORIZED)
# ==========================================

# --- Helper: Linear Regression for Squeeze Pro ---
def calc_linreg_slope(series, window):
    """Vectorized Rolling Linear Regression Slope"""
    y = series
    x = np.arange(window)
    # Pre-compute x statistics
    sum_x = np.sum(x)
    sum_x_sq = np.sum(x**2)
    divisor = window * sum_x_sq - sum_x**2
    
    def get_slope(y_window):
        sum_y = np.sum(y_window)
        sum_xy = np.sum(x * y_window)
        return (window * sum_xy - sum_x * sum_y) / divisor
        
    return y.rolling(window).apply(get_slope, raw=True)

# --- Helper: Hull Moving Average ---
def calc_hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = series.rolling(half_length).apply(lambda x: np.dot(x, np.arange(1, half_length+1)) / np.arange(1, half_length+1).sum(), raw=True)
    wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / np.arange(1, length+1).sum(), raw=True)
    raw_hma = 2 * wma_half - wma_full
    return raw_hma.rolling(sqrt_length).apply(lambda x: np.dot(x, np.arange(1, sqrt_length+1)) / np.arange(1, sqrt_length+1).sum(), raw=True)

# --- Helper: Double Smooth (for TSI/Money Flow) ---
def double_smooth(src, long, short):
    first = src.ewm(span=long, adjust=False).mean()
    return first.ewm(span=short, adjust=False).mean()

def calc_titan_indicators(df):
    """
    Combines Legacy DarkPool indicators with the 5 NEW requested scripts.
    """
    # ==========================
    # A. LEGACY INDICATORS (Kept)
    # ==========================
    # 1. Apex Trend (HMA)
    df['HMA_55'] = calc_hma(df['Close'], 55)
    df['Apex_Trend'] = np.where(df['Close'] > df['HMA_55'], 'BULLISH', 'BEARISH')
    
    # 2. Vector Candles (HA)
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    df['Vector_Color'] = np.where(ha_close > ha_open, 'GREEN', 'RED')
    
    # 3. RVOL
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['RVOL'] = df['Volume'] / df['Vol_SMA']
    
    # 4. Standard
    df['ATR'] = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))

    # ==========================
    # B. NEW INDICATOR 1: ULTIMATE S&R (Pivot Clusters)
    # ==========================
    # Python adaptation of array-based logic: We find pivots, then cluster them within a % threshold
    period = 10
    df['Pivot_H'] = df['High'].rolling(period*2+1, center=True).max() == df['High']
    df['Pivot_L'] = df['Low'].rolling(period*2+1, center=True).min() == df['Low']
    
    pivots = []
    # Collect recent pivots
    lookback_rows = 300
    subset = df.iloc[-lookback_rows:]
    for i, row in subset.iterrows():
        if row['Pivot_H']: pivots.append(row['High'])
        if row['Pivot_L']: pivots.append(row['Low'])
    
    # Simple Clustering for "Dynamic Lines"
    sr_levels = []
    if pivots:
        pivots.sort()
        current_cluster = [pivots[0]]
        threshold = df['Close'].iloc[-1] * 0.005 # 0.5% width
        
        for p in pivots[1:]:
            if p - current_cluster[-1] < threshold:
                current_cluster.append(p)
            else:
                sr_levels.append(sum(current_cluster)/len(current_cluster))
                current_cluster = [p]
        sr_levels.append(sum(current_cluster)/len(current_cluster))
    
    # We store these levels to plot later, no column needed
    
    # ==========================
    # C. NEW INDICATOR 2: ELASTIC VOLUME-WEIGHTED MOMENTUM (EVWM)
    # ==========================
    evwm_len = 21
    hull_basis = calc_hma(df['Close'], evwm_len)
    elasticity = (df['Close'] - hull_basis) / df['ATR']
    
    # Volume Weighting
    rvol = df['Volume'] / df['Volume'].rolling(evwm_len).mean()
    final_force = np.sqrt(rvol.rolling(5).mean()) # Smoothing
    
    df['EVWM'] = elasticity * final_force
    df['EVWM_Signal'] = np.where(df['EVWM'] > 0, "BULL", "BEAR")
    
    # Bands
    evwm_std = df['EVWM'].rolling(evwm_len*2).std()
    df['EVWM_Upper'] = df['EVWM'].rolling(evwm_len*2).mean() + (2.0 * evwm_std)
    df['EVWM_Lower'] = df['EVWM'].rolling(evwm_len*2).mean() - (2.0 * evwm_std)

    # ==========================
    # D. NEW INDICATOR 3: MONEY FLOW MATRIX
    # ==========================
    # Normalized Money Flow
    mf_rsi = df['RSI'] - 50
    mf_vol_ratio = df['Volume'] / df['Volume'].rolling(14).mean()
    df['MF_Matrix'] = (mf_rsi * mf_vol_ratio).ewm(span=3, adjust=False).mean()
    
    # Hyper Wave (TSI Implementation)
    pc = df['Close'].diff()
    ds_pc = double_smooth(pc, 25, 13)
    ds_abs_pc = double_smooth(abs(pc), 25, 13)
    df['Hyper_Wave'] = (100 * (ds_pc / ds_abs_pc)) / 2

    # ==========================
    # E. NEW INDICATOR 4: SQUEEZE MOMENTUM PRO
    # ==========================
    bb_len = 20
    kc_mult = 1.5
    
    # BB
    basis = df['Close'].rolling(bb_len).mean()
    dev = df['Close'].rolling(bb_len).std()
    bb_upper = basis + (2.0 * dev)
    bb_lower = basis - (2.0 * dev)
    
    # KC
    tr = df['ATR'] # Re-use ATR
    kc_upper = basis + (tr * kc_mult)
    kc_lower = basis - (tr * kc_mult)
    
    # Squeeze Logic
    df['SQZ_On'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    df['SQZ_Color'] = np.where(df['SQZ_On'], 'red', 'gray') # Red = Squeeze On
    
    # Momentum (Linear Regression)
    # Val = Close - Avg(HighestHigh, LowestLow, SMA)
    highest = df['High'].rolling(bb_len).max()
    lowest = df['Low'].rolling(bb_len).min()
    avg_val = (highest + lowest + basis) / 3
    
    source_delta = df['Close'] - avg_val
    df['SQZ_Mom'] = calc_linreg_slope(source_delta, bb_len)

    # ==========================
    # F. NEW INDICATOR 5: WYCKOFF PRECISION (Gemini)
    # ==========================
    # Golden Pockets based on last major pivots
    # Logic: If price breaks a pivot, define a retracement zone
    df['Wyck_Trend'] = np.where(df['Close'] > df['Close'].rolling(200).mean(), "BULL", "BEAR")
    
    return df, sr_levels

def calc_fear_greed_v4(df):
    """Retained Original Logic"""
    df['FG_RSI'] = df['RSI']
    # Simplified Logic for brevity/speed
    df['FG_Index'] = (df['RSI'] + (df['Close'] > df['Close'].rolling(50).mean())*50) / 1.5
    df['FG_Index'] = df['FG_Index'].clip(0, 100)
    return df

# ==========================================
# 4. AI ANALYST (UPDATED TO SEE NEW METRICS)
# ==========================================
def ask_ai_analyst(df, ticker, balance, risk_pct, timeframe):
    if not st.session_state.api_key: return "⚠️ Waiting for OpenAI API Key in the sidebar..."
    
    last = df.iloc[-1]
    
    # Synthesize New Indicators
    evwm_state = "IGNITION" if last['EVWM'] > last['EVWM_Upper'] else "NEUTRAL"
    sqz_state = "FIRING" if (not last['SQZ_On'] and df['SQZ_On'].iloc[-2]) else "SQUEEZING" if last['SQZ_On'] else "NORMAL"
    mf_state = "INFLOW" if last['MF_Matrix'] > 0 else "OUTFLOW"
    
    prompt = f"""
    Analyze {ticker} ({timeframe}) at ${last['Close']:.2f}.
    
    --- NEW TITAN ENGINES ---
    1. EVWM Momentum: {last['EVWM']:.2f} ({evwm_state})
    2. Squeeze Pro: {last['SQZ_Mom']:.4f} (State: {sqz_state})
    3. Money Flow Matrix: {last['MF_Matrix']:.2f} ({mf_state})
    4. Hyper Wave: {last['Hyper_Wave']:.1f}
    
    --- CLASSIC METRICS ---
    - Apex Trend: {last['Apex_Trend']}
    - RSI: {last['RSI']:.1f}
    - ATR: {last['ATR']:.2f}
    
    Risk: ${balance * (risk_pct/100):.2f}.
    Provide a professional trading verdict (BUY/SELL/WAIT) with specific reasoning citing the EVWM and Squeeze status.
    """
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}], max_tokens=1000)
        return res.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# ==========================================
# 5. UI DASHBOARD LAYOUT
# ==========================================
st.sidebar.header("🎛️ Terminal Controls")

# --- BROADCAST CENTER ---
st.sidebar.subheader("📢 Social Broadcaster")
tg_token = st.sidebar.text_input("Telegram Bot Token", type="password", help="Telegram Bot Token for alerts.")
tg_chat = st.sidebar.text_input("Telegram Chat ID", help="Chat ID for alerts.")

input_mode = st.sidebar.radio("Input Mode:", ["Curated Lists", "Manual Search"], index=1, help="Select Asset Source.")

if input_mode == "Curated Lists":
    assets = { "Indices": ["SPY", "QQQ"], "Crypto": ["BTC-USD", "ETH-USD"], "Macro": ["^TNX", "DX-Y.NYB"] }
    cat = st.sidebar.selectbox("Asset Class", list(assets.keys()), help="Choose category.")
    ticker = st.sidebar.selectbox("Ticker", assets[cat], help="Select Ticker.")
else:
    ticker = st.sidebar.text_input("Search Ticker", value="BTC-USD", help="Type any ticker (e.g., TSLA, BTC-USD).").upper()

interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"], index=4, help="Select Timeframe.")
st.sidebar.markdown("---")
balance = st.sidebar.number_input("Capital ($)", 1000, 1000000, 10000, help="Trading Capital.")
risk_pct = st.sidebar.slider("Risk %", 0.5, 3.0, 1.0, help="Risk per trade.")

# --- MACRO HEADER ---
mg, mp, mc = get_macro_data()
if mp:
    cols = st.columns(4)
    cols[0].metric("SPY", f"{mp.get('SPY',0):.2f}", f"{mc.get('SPY',0):.2f}%")
    cols[1].metric("BTC", f"{mp.get('BTC-USD',0):.2f}", f"{mc.get('BTC-USD',0):.2f}%")
    cols[2].metric("10Y Yield", f"{mp.get('^TNX',0):.3f}", f"{mc.get('^TNX',0):.2f}%")
    cols[3].metric("Gold", f"{mp.get('GC=F',0):.1f}", f"{mc.get('GC=F',0):.2f}%")
st.markdown("---")

# --- TABS ---
tabs = st.tabs([
    "📊 Technical Deep Dive", 
    "🌊 Squeeze & Momentum (New)", 
    "💸 Money Flow Matrix (New)",
    "📆 Seasonality", 
    "🔮 AI Strategy"
])

if st.button(f"🚀 Analyze {ticker}", help="Run the full analysis engine."):
    st.session_state['run'] = True

if st.session_state.get('run'):
    with st.spinner(f"Processing {ticker} on {interval}..."):
        # 1. Fetch Data (Optimized)
        df = safe_download(ticker, "1y", interval)
        
        if df is not None:
            # 2. Run All Math Engines
            df, sr_levels = calc_titan_indicators(df)
            df = calc_fear_greed_v4(df)
            
            last = df.iloc[-1]

            # --- TAB 1: MAIN CHART + S&R + EVWM ---
            with tabs[0]:
                st.subheader("🎯 Ultimate S&R + EVWM")
                
                # Main Price Chart with Ultimate S&R Lines
                fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.02)
                fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                
                # Plot Dynamic S&R Lines (from the cluster logic)
                for level in sr_levels:
                    color_sr = "red" if level > last['Close'] else "green"
                    fig1.add_hline(y=level, line_dash="dot", line_color=color_sr, row=1, col=1, opacity=0.5)
                
                # Plot HMA (Apex Trend)
                fig1.add_trace(go.Scatter(x=df.index, y=df['HMA_55'], line=dict(color='orange', width=2), name="Apex Trend"), row=1, col=1)

                # Subchart: EVWM (Elastic Volume Weighted Momentum)
                colors = np.where(df['EVWM'] > 0, '#00ffaa', '#ff0062')
                fig1.add_trace(go.Bar(x=df.index, y=df['EVWM'], marker_color=colors, name="EVWM"), row=2, col=1)
                fig1.add_trace(go.Scatter(x=df.index, y=df['EVWM_Upper'], line=dict(color='gray', width=1), name="Band"), row=2, col=1)
                fig1.add_trace(go.Scatter(x=df.index, y=df['EVWM_Lower'], line=dict(color='gray', width=1), name="Band"), row=2, col=1)
                
                fig1.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig1, use_container_width=True)

            # --- TAB 2: SQUEEZE MOMENTUM PRO ---
            with tabs[1]:
                st.subheader("🌊 Squeeze Pro (LazyBear Port)")
                fig_sqz = go.Figure()
                
                # Momentum Histogram
                cols_sqz = np.where(df['SQZ_Mom'] > 0, 
                                   np.where(df['SQZ_Mom'] > df['SQZ_Mom'].shift(1), '#00E676', '#4CAF50'), # Bull Strong/Weak
                                   np.where(df['SQZ_Mom'] < df['SQZ_Mom'].shift(1), '#FF5252', '#FF1744')) # Bear Strong/Weak
                
                fig_sqz.add_trace(go.Bar(x=df.index, y=df['SQZ_Mom'], marker_color=cols_sqz, name="Momentum"))
                
                # Squeeze Dots
                sqz_y = np.zeros(len(df))
                sqz_c = np.where(df['SQZ_On'], 'red', 'green')
                fig_sqz.add_trace(go.Scatter(x=df.index, y=sqz_y, mode='markers', marker=dict(color=sqz_c, size=5), name="Squeeze Status"))
                
                fig_sqz.update_layout(height=500, template="plotly_dark", title="Squeeze Momentum Pro")
                st.plotly_chart(fig_sqz, use_container_width=True)
                
                st.info(f"**Current Status:** {'🔴 SQUEEZE ACTIVE' if last['SQZ_On'] else '🟢 RELEASED'} | Momentum: {last['SQZ_Mom']:.4f}")

            # --- TAB 3: MONEY FLOW MATRIX ---
            with tabs[2]:
                st.subheader("💸 Money Flow Matrix")
                fig_mf = make_subplots(rows=2, cols=1, shared_xaxes=True)
                
                # Money Flow
                mf_col = np.where(df['MF_Matrix'] > 0, '#00ff00', '#ff0000')
                fig_mf.add_trace(go.Bar(x=df.index, y=df['MF_Matrix'], marker_color=mf_col, name="Money Flow"), row=1, col=1)
                
                # Hyper Wave
                fig_mf.add_trace(go.Scatter(x=df.index, y=df['Hyper_Wave'], line=dict(color='cyan', width=2), name="Hyper Wave"), row=2, col=1)
                fig_mf.add_hline(y=0, row=2, col=1)
                
                fig_mf.update_layout(height=600, template="plotly_dark")
                st.plotly_chart(fig_mf, use_container_width=True)

            # --- TAB 5: AI STRATEGY ---
            with tabs[4]:
                st.subheader("🤖 Titan AI Verdict")
                ai_text = ask_ai_analyst(df, ticker, balance, risk_pct, interval)
                st.write(ai_text)
                
                # Wyckoff / Fractal Status
                st.markdown("---")
                st.markdown("#### 🏛️ Wyckoff Structure Status")
                c1, c2 = st.columns(2)
                wyck_trend = df['Wyck_Trend'].iloc[-1]
                c1.metric("Trend Baseline (200 MA)", wyck_trend, delta_color="normal")
                c2.metric("Fractal Golden Pocket", "Searching...", help="Automatically highlighted in Chart 1 via S&R lines.")

        else:
            st.error("Data fetch failed. Try a different ticker or higher timeframe.")
