import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import streamlit.components.v1 as components
from scipy.stats import linregress
import requests
import urllib.parse
from datetime import datetime, time

# ==========================================
# 1. PREMIUM UI CONFIGURATION
# ==========================================
# FIX: Changed page_title to "Equity Titan" as requested
st.set_page_config(layout="wide", page_title="📊Equity Titan", page_icon="🌍")

st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
        font-family: 'SF Mono', 'Roboto Mono', monospace;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #2979FF;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #fff;
    }
    
    /* Signal Colors */
    .bull { color: #00E676 !important; text-shadow: 0 0 10px rgba(0, 230, 118, 0.4); }
    .bear { color: #FF1744 !important; text-shadow: 0 0 10px rgba(255, 23, 68, 0.4); }
    .neu  { color: #B0BEC5 !important; }
    
    /* Divergence Alert Style */
    .div-alert {
        background-color: rgba(255, 82, 82, 0.1);
        border: 1px solid #FF5252;
        color: #FF5252;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
        animation: pulse 2s infinite;
    }
    .div-alert-bull {
        background-color: rgba(0, 230, 118, 0.1);
        border: 1px solid #00E676;
        color: #00E676;
    }
    
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    /* Header Glow */
    .titan-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2979FF, #E040FB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: #121212;
        border-radius: 8px;
        color: #888;
        border: 1px solid #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2979FF !important;
        color: #fff !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1.1 WORLD CLOCK WIDGET (FIXED)
# ==========================================
# FIX: Added DOMContentLoaded and robust JS to ensure clocks render immediately
clock_html = """
<!DOCTYPE html>
<html>
<head>
<style>
    body { margin: 0; background: transparent; font-family: 'SF Mono', monospace; }
    .clock-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }
    .clock-box { text-align: center; }
    .label { color: #888; font-size: 0.75rem; letter-spacing: 1px; margin-bottom: 2px; }
    .time { font-weight: bold; font-size: 1.1rem; }
    .ny { color: #2979FF; }
    .uk { color: #E040FB; }
    .jp { color: #00E676; }
    .divider { width: 1px; height: 30px; background: #333; }
</style>
</head>
<body>
<div class="clock-container">
    <div class="clock-box">
        <div class="label">NEW YORK</div>
        <div id="clock-ny" class="time ny">--:--:--</div>
    </div>
    <div class="divider"></div>
    <div class="clock-box">
        <div class="label">LONDON</div>
        <div id="clock-uk" class="time uk">--:--:--</div>
    </div>
    <div class="divider"></div>
    <div class="clock-box">
        <div class="label">TOKYO</div>
        <div id="clock-jp" class="time jp">--:--:--</div>
    </div>
</div>
<script>
function updateTime() {
    const now = new Date();
    const fmt = (tz) => {
        try {
            return new Intl.DateTimeFormat('en-US', { timeZone: tz, hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(now);
        } catch (e) { return "--:--:--"; }
    };
    document.getElementById('clock-ny').innerText = fmt('America/New_York');
    document.getElementById('clock-uk').innerText = fmt('Europe/London');
    document.getElementById('clock-jp').innerText = fmt('Asia/Tokyo');
}
// Ensure script runs after load
document.addEventListener("DOMContentLoaded", function() {
    updateTime();
    setInterval(updateTime, 1000);
});
updateTime(); // Fallback immediate run
</script>
</body>
</html>
"""
components.html(clock_html, height=85)

# --- SIDEBAR CONTROL ---
st.sidebar.header("⚡ System Control")

# API Keys
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI Key", type="password")

st.sidebar.subheader("📡 Broadcast Keys")
if 'tg_token' not in st.session_state: st.session_state.tg_token = ""
if 'tg_chat' not in st.session_state: st.session_state.tg_chat = ""

if "TELEGRAM_TOKEN" in st.secrets: st.session_state.tg_token = st.secrets["TELEGRAM_TOKEN"]
if "TELEGRAM_CHAT_ID" in st.secrets: st.session_state.tg_chat = st.secrets["TELEGRAM_CHAT_ID"]

tg_token = st.sidebar.text_input("Telegram Bot Token", value=st.session_state.tg_token, type="password")
tg_chat = st.sidebar.text_input("Telegram Chat ID", value=st.session_state.tg_chat)

# --- TOP TECH & MINERS ASSETS ---
stock_assets = {
    "NVIDIA (NVDA)": "NVDA", "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN", "Google (GOOGL)": "GOOGL", "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA", "Netflix (NFLX)": "NFLX", "Broadcom (AVGO)": "AVGO",
    "AMD (AMD)": "AMD", "Intel (INTC)": "INTC", "TSMC (TSM)": "TSM",
    "Micron (MU)": "MU", "Qualcomm (QCOM)": "QCOM", "ARM Holdings (ARM)": "ARM",
    "Super Micro (SMCI)": "SMCI", "Dell (DELL)": "DELL",
    "Marathon Digital (MARA)": "MARA", "Riot Platforms (RIOT)": "RIOT",
    "CleanSpark (CLSK)": "CLSK", "Hut 8 (HUT)": "HUT", "Coinbase (COIN)": "COIN",
    "MicroStrategy (MSTR)": "MSTR", "Iris Energy (IREN)": "IREN",
    "Core Scientific (CORZ)": "CORZ",
    "Palantir (PLTR)": "PLTR", "Snowflake (SNOW)": "SNOW", "C3.ai (AI)": "AI",
    "CrowdStrike (CRWD)": "CRWD", "Palo Alto (PANW)": "PANW",
    "Newmont Gold (NEM)": "NEM", "Barrick Gold (GOLD)": "GOLD",
    "Freeport-McMoRan (FCX)": "FCX", "Albemarle (ALB)": "ALB"
}

# --- SEARCH FEATURE ---
search_mode = st.sidebar.radio("Asset Selection", 
                               ["Top List", "Search US/Global", "Search UK (LSE)", "Search Japan (TSE)"], 
                               horizontal=False)

ticker = "NVDA" # Default
ticker_name = "NVIDIA"

if search_mode == "Top List":
    ticker_name_input = st.sidebar.selectbox("Target Asset", list(stock_assets.keys()))
    ticker = stock_assets[ticker_name_input]
    ticker_name = ticker_name_input
elif search_mode == "Search US/Global":
    custom_symbol = st.sidebar.text_input("Enter Ticker", value="NVDA").upper().strip()
    ticker = custom_symbol
    ticker_name = f"{custom_symbol} (Global)"
elif search_mode == "Search UK (LSE)":
    uk_symbol = st.sidebar.text_input("Enter UK Ticker", value="RR").upper().strip()
    ticker = f"{uk_symbol}.L" if not uk_symbol.endswith(".L") else uk_symbol
    ticker_name = f"{ticker} (LSE)"
elif search_mode == "Search Japan (TSE)":
    jp_symbol = st.sidebar.text_input("Enter Japan Code", value="7203").upper().strip()
    ticker = f"{jp_symbol}.T" if not jp_symbol.endswith(".T") else jp_symbol
    ticker_name = f"{ticker} (TSE)"

interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=3)

# ==========================================
# 2. ADVANCED MATH ENGINE
# ==========================================
def calculate_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    diff = 2 * wma_half - wma_full
    return calculate_wma(diff, sqrt_length)

def calculate_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return calculate_rma(tr, length)

def calculate_linreg_mom(series, length=20):
    x = np.arange(length)
    slope = series.rolling(length).apply(lambda y: linregress(x, y)[0], raw=True)
    return slope

@st.cache_data(ttl=600)
def get_institutional_trend(ticker):
    try:
        # FIX: Flatten multi-index columns for Institutional Trend
        df_d = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
        
        df_w = yf.download(ticker, period="2y", interval="1wk", progress=False)
        if isinstance(df_w.columns, pd.MultiIndex): df_w.columns = df_w.columns.get_level_values(0)
        
        ema_d = df_d['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        ema_w = df_w['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        return ema_d, ema_w
    except:
        return 0, 0

def calculate_engine(df, ticker):
    # 1. Apex
    apex_len = 55
    apex_mult = 1.5
    df['Apex_Base'] = calculate_hma(df['Close'], apex_len)
    df['Apex_ATR'] = calculate_atr(df, apex_len)
    df['ATR'] = df['Apex_ATR'] # Safe copy for dashboard
    df['Apex_Upper'] = df['Apex_Base'] + (df['Apex_ATR'] * apex_mult)
    df['Apex_Lower'] = df['Apex_Base'] - (df['Apex_ATR'] * apex_mult)
    df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, np.nan))
    df['Apex_Trend'] = df['Apex_Trend'].ffill().fillna(0)

    # 2. Pivots
    liq_len = 10
    df['Pivot_High'] = df['High'].rolling(liq_len*2+1, center=True).max()
    df['Pivot_Low'] = df['Low'].rolling(liq_len*2+1, center=True).min()

    # 3. Money Flow Matrix
    mfLen = 14
    mfSmooth = 3
    delta_mf = df['Close'].diff()
    gain_mf = delta_mf.where(delta_mf > 0, 0)
    loss_mf = -delta_mf.where(delta_mf < 0, 0)
    avg_gain_mf = gain_mf.ewm(alpha=1/mfLen, adjust=False).mean()
    avg_loss_mf = loss_mf.ewm(alpha=1/mfLen, adjust=False).mean()
    # Handle division by zero
    avg_loss_mf = avg_loss_mf.replace(0, 0.0001)
    rs_mf = avg_gain_mf / avg_loss_mf
    rsi_mf = 100 - (100 / (1 + rs_mf))
    
    mfVolume = df['Volume'] / df['Volume'].rolling(mfLen).mean()
    df['Matrix_MF'] = ((rsi_mf - 50) * mfVolume).ewm(span=mfSmooth, adjust=False).mean()
    
    df['MF_BB_Mid'] = df['Matrix_MF'].rolling(20).mean()
    df['MF_BB_Std'] = df['Matrix_MF'].rolling(20).std()
    df['MF_BB_Up'] = df['MF_BB_Mid'] + (df['MF_BB_Std'] * 2.0)
    df['MF_BB_Low'] = df['MF_BB_Mid'] - (df['MF_BB_Std'] * 2.0)

    # Hyper Wave
    ds_pc = df['Close'].diff().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    ds_abs_pc = df['Close'].diff().abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    df['Matrix_HyperWave'] = (100 * (ds_pc / ds_abs_pc)) / 2

    # Div Signal
    df['Div_Signal'] = 0
    price_hh = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(2))
    mf_lh = (df['Matrix_MF'] < df['Matrix_MF'].shift(1)) & (df['Matrix_MF'] < df['Matrix_MF'].shift(2))
    df.loc[price_hh & mf_lh & (df['Matrix_MF'] > 0), 'Div_Signal'] = -1
    
    price_ll = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(2))
    mf_hl = (df['Matrix_MF'] > df['Matrix_MF'].shift(1)) & (df['Matrix_MF'] > df['Matrix_MF'].shift(2))
    df.loc[price_ll & mf_hl & (df['Matrix_MF'] < 0), 'Div_Signal'] = 1

    # Gann
    df['Gann_High'] = df['High'].rolling(3).mean()
    df['Gann_Low'] = df['Low'].rolling(3).mean()
    
    # Vol Delta
    df['Vol_Delta'] = np.where(df['Close'] >= df['Open'], df['Volume'], -df['Volume'])
    
    # MFI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    pos = np.where(tp > tp.shift(1), rmf, 0)
    neg = np.where(tp < tp.shift(1), rmf, 0)
    mfi_ratio = pd.Series(pos).rolling(14).sum() / pd.Series(neg).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # ADX
    up = df['High'].diff()
    down = -df['Low'].diff()
    tr = calculate_atr(df, 14)
    p_dm = np.where((up > down) & (up > 0), up, 0)
    m_dm = np.where((down > up) & (down > 0), down, 0)
    p_di = 100 * calculate_rma(pd.Series(p_dm), 14) / tr
    m_di = 100 * calculate_rma(pd.Series(m_dm), 14) / tr
    df['ADX'] = calculate_rma((np.abs(p_di - m_di) / (p_di + m_di)) * 100, 14)

    # Ichimoku
    df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['SpanB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    
    # MACD
    df['EMA_200'] = df['Close'].ewm(span=200).mean()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Hist'] = (df['MACD'] - df['MACD'].ewm(span=9).mean()) * (df['Volume'] / df['Volume'].rolling(20).mean())

    # RSI & Stoch
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
    df['Stoch_K'] = ((df['RSI'] - df['RSI'].rolling(14).min()) / (df['RSI'].rolling(14).max() - df['RSI'].rolling(14).min())).rolling(3).mean() * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # DarkPool MAs
    for p in [10, 20, 50, 100, 200]: df[f'DP_MA{p}'] = df['Close'].ewm(span=p).mean()
    df['DP_Score'] = sum(np.where(df['Close'] > df[f'DP_MA{p}'], 1, 0) for p in [10, 20, 50, 100, 200])

    # Inst Trend
    ema_d, ema_w = get_institutional_trend(ticker)
    df['Inst_Trend'] = np.where(ema_d > ema_w, 1, -1)
    
    # RVOL & VWAP
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['VWAP'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def calculate_strategies(df):
    df['Sig_Mom'] = np.where((df['Close'] - df['Close'].shift(12) > 0) & (df['Close'].shift(1) - df['Close'].shift(13) > 0), 1, np.where((df['Close'] - df['Close'].shift(12) < 0), -1, 0))
    box_high = df['High'].rolling(20).max().shift(1)
    df['Sig_ADX'] = np.where((df['Close'] > box_high) & (df['ADX'] < 25), 1, 0)
    sma20 = df['Close'].rolling(20).mean()
    df['Sig_BB'] = np.where(df['Close'] < (sma20 - 2*df['Close'].rolling(20).std()), 1, np.where(df['Close'] > (sma20 + 2*df['Close'].rolling(20).std()), -1, 0))
    df['Sig_RSI'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    return df

@st.cache_data(ttl=300)
def get_data(ticker, interval):
    period = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}[interval]
    df = yf.download(ticker, period=period, interval=("1h" if interval=="4h" else interval), progress=False)
    
    # FIX: Check for and flatten MultiIndex columns which causes Dark Pool/Oscillators to break
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if interval == "4h":
        df = df.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
    return df

# --- RENDER DASHBOARD ---
# FIX: Renamed header to match Title as per user request
st.markdown(f'<div class="titan-header">🏢 EQUITY TITAN: {ticker_name}</div>', unsafe_allow_html=True)
tv_int = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}[interval]
tv_sym = ticker
if ".L" in ticker: tv_sym = "LSE:" + ticker.replace(".L", "")
if ".T" in ticker: tv_sym = "TSE:" + ticker.replace(".T", "")

components.html(f"""<div class="tradingview-widget-container"><div id="tv"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"width": "100%","height": 450,"symbol": "{tv_sym}","interval": "{tv_int}","theme": "dark","style": "1","locale": "en","container_id": "tv"}});</script></div>""", height=460)

df = get_data(ticker, interval)

if df is not None and not df.empty:
    df = calculate_engine(df, ticker)
    df = calculate_strategies(df)
    last = df.iloc[-1]
    
    # Levels
    atr_val = last['ATR']
    stop_loss = last['Close'] - (2 * atr_val) if last['Apex_Trend'] == 1 else last['Close'] + (2 * atr_val)
    take_profit = last['Close'] + (3 * atr_val) if last['Apex_Trend'] == 1 else last['Close'] - (3 * atr_val)
    
    # --- DNA HUD ---
    st.markdown("### 🧬 Market DNA")
    if last['Div_Signal'] == -1: st.markdown(f'<div class="div-alert">⚠️ BEARISH DIVERGENCE DETECTED</div>', unsafe_allow_html=True)
    elif last['Div_Signal'] == 1: st.markdown(f'<div class="div-alert div-alert-bull">⚠️ BULLISH DIVERGENCE DETECTED</div>', unsafe_allow_html=True)
    
    cols = st.columns(6)
    def card(label, value, condition):
        c = "bull" if condition == 1 else "bear" if condition == -1 else "neu"
        return f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {c}">{value}</div></div>"""
    
    cols[0].markdown(card("Apex Trend", "BULL" if last['Apex_Trend']==1 else "BEAR" if last['Apex_Trend']==-1 else "NEU", last['Apex_Trend']), unsafe_allow_html=True)
    cols[1].markdown(card("Momentum", "POS" if last['Sig_Mom']==1 else "NEG", last['Sig_Mom']), unsafe_allow_html=True)
    cols[2].markdown(card("Inst. Trend", "BULL" if last['Inst_Trend']==1 else "BEAR", last['Inst_Trend']), unsafe_allow_html=True)
    cols[3].markdown(card("Money Flow", "O/S" if last['MFI']<20 else "O/B" if last['MFI']>80 else "NEU", 1 if last['MFI']<20 else -1 if last['MFI']>80 else 0), unsafe_allow_html=True)
    cols[4].markdown(card("Bollinger", "BUY" if last['Sig_BB']==1 else "SELL" if last['Sig_BB']==-1 else "NEU", last['Sig_BB']), unsafe_allow_html=True)
    cols[5].markdown(card("Rel. Vol", f"{last['RVOL']:.2f}x", 1 if last['RVOL']>1.5 else -1 if last['RVOL']<0.75 else 0), unsafe_allow_html=True)

    # --- TABS ---
    tab_apex, tab_dpma, tab_matrix, tab_cloud, tab_osc, tab_ai, tab_cast = st.tabs(["🌊 Apex Master", "💀 DarkPool Trends", "🔋 Money Flow Matrix", "☁️ Ichimoku", "📈 Oscillators", "🤖 AI Analyst", "📡 Broadcast"])

    with tab_apex:
        fig = go.Figure()
        if interval in ["15m", "1h"]:
            for date in df.index.normalize().unique()[-5:]:
                fig.add_vrect(x0=date + pd.Timedelta(hours=13, minutes=30), x1=date + pd.Timedelta(hours=16, minutes=30), fillcolor="purple", opacity=0.1, layer="below", line_width=0)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.15)' if last['Apex_Trend']==1 else 'rgba(255, 23, 68, 0.15)', line=dict(width=0)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Base'], line=dict(color='#00E676' if last['Apex_Trend']==1 else '#FF1744', width=2), name="Apex Base"))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FFD700', width=2, dash='dash'), name="VWAP"))
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, title=f"Apex Trend & Sessions ({interval})")
        st.plotly_chart(fig, use_container_width=True)

    with tab_dpma:
        fig_dp = go.Figure()
        fig_dp.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA10'], line=dict(color='#00E5FF', width=1), name="EMA 10"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA20'], line=dict(color='#2979FF', width=1), name="EMA 20"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA50'], line=dict(color='#00E676', width=2), name="EMA 50"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA100'], line=dict(color='#FF9100', width=2), name="EMA 100"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA200'], line=dict(color='#FF1744', width=2), name="EMA 200"))
        fig_dp.update_layout(height=600, template="plotly_dark", title="DarkPool Institutional MAs")
        st.plotly_chart(fig_dp, use_container_width=True)

    with tab_matrix:
        fm = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
        fm.add_trace(go.Bar(x=df.index, y=df['Matrix_MF'], marker_color=['#00E676' if v > 0 else '#FF1744' for v in df['Matrix_MF']], name="Money Flow"), row=1, col=1)
        fm.add_trace(go.Scatter(x=df.index, y=df['Matrix_HyperWave'], line=dict(color='white'), name="Hyper Wave"), row=1, col=1)
        fm.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color='cyan', name="MACD"), row=2, col=1)
        fm.add_trace(go.Bar(x=df.index, y=df['Vol_Delta'], marker_color=['#00E676' if v > 0 else '#FF1744' for v in df['Vol_Delta']], name="Vol Delta"), row=3, col=1)
        fm.update_layout(height=600, template="plotly_dark", title="Money Flow Matrix")
        st.plotly_chart(fm, use_container_width=True)

    with tab_cloud:
        fc = go.Figure()
        fc.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fc.add_trace(go.Scatter(x=df.index, y=df['Tenkan'], line=dict(color='#2962FF', width=1), name="Tenkan"))
        fc.add_trace(go.Scatter(x=df.index, y=df['Kijun'], line=dict(color='#B71C1C', width=1), name="Kijun"))
        fc.add_trace(go.Scatter(x=df.index, y=df['SpanA'], line=dict(width=0), showlegend=False))
        fc.add_trace(go.Scatter(x=df.index, y=df['SpanB'], fill='tonexty', fillcolor='rgba(67, 160, 71, 0.2)', line=dict(width=0), name="Cloud"))
        fc.update_layout(height=500, template="plotly_dark", title="Ichimoku Cloud")
        st.plotly_chart(fc, use_container_width=True)

    with tab_osc:
        fo = go.Figure()
        fo.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line=dict(color='#2962FF', width=2), name="K%"))
        fo.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], line=dict(color='#FF6D00', width=2), name="D%"))
        fo.add_hline(y=80, line_color="gray", line_dash="dot")
        fo.add_hline(y=20, line_color="gray", line_dash="dot")
        fo.update_layout(height=400, template="plotly_dark", title="Stochastic RSI")
        st.plotly_chart(fo, use_container_width=True)

    with tab_ai:
        if st.button("Generate Alpha Report"):
            if api_key:
                client = OpenAI(api_key=api_key)
                prompt = f"""
                Analyze {ticker} ({interval}) at ${last['Close']:.2f}.
                Trend: {'BULL' if last['Apex_Trend']==1 else 'BEAR'}.
                RVOL: {last['RVOL']:.2f}x.
                Divergence: {last['Div_Signal']}.
                Money Flow: {last['Matrix_MF']:.2f}.
                VWAP: ${last['VWAP']:.2f}.
                Plan trades with risk management.
                """
                st.info(client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":prompt}]).choices[0].message.content)
            else: st.error("Key Missing")

    with tab_cast:
        div_msg = ""
        if last['Div_Signal'] == 1: div_msg = "\n⚠️ BULL DIV"
        if last['Div_Signal'] == -1: div_msg = "\n⚠️ BEAR DIV"
        preview = f"TITAN SIGNAL: {ticker} ({interval})\nDIRECTION: {'LONG' if last['Apex_Trend']==1 else 'SHORT'}\nENTRY: ${last['Close']:,.2f}\nSTOP: ${stop_loss:,.2f}\nRVOL: {last['RVOL']:.2f}x\nVWAP: ${last['VWAP']:.2f}{div_msg}"
        st.text_area("Preview", preview, height=150)
        if st.button("Send🚀") and tg_token: requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={"chat_id":tg_chat, "text":preview})

else: st.error("Data Stream Error.")
