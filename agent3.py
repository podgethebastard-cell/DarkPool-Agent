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

# ==========================================
# 1. PREMIUM UI CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Titan Terminal: Global Equity", page_icon="🌍")

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
# 1.1 WORLD CLOCK WIDGET
# ==========================================
st.markdown("""
<div style="
    display: flex;
    justify-content: space-around;
    align-items: center;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 15px;
    font-family: 'SF Mono', monospace;
">
    <div style="text-align: center;">
        <div style="color: #888; font-size: 0.75rem; letter-spacing: 1px;">NEW YORK</div>
        <div id="clock-ny" style="color: #2979FF; font-weight: bold; font-size: 1.1rem;">--:--:--</div>
    </div>
    <div style="width: 1px; height: 30px; background: #333;"></div>
    <div style="text-align: center;">
        <div style="color: #888; font-size: 0.75rem; letter-spacing: 1px;">LONDON</div>
        <div id="clock-uk" style="color: #E040FB; font-weight: bold; font-size: 1.1rem;">--:--:--</div>
    </div>
    <div style="width: 1px; height: 30px; background: #333;"></div>
    <div style="text-align: center;">
        <div style="color: #888; font-size: 0.75rem; letter-spacing: 1px;">TOKYO</div>
        <div id="clock-jp" style="color: #00E676; font-weight: bold; font-size: 1.1rem;">--:--:--</div>
    </div>
</div>

<script>
function updateTime() {
    const now = new Date();
    
    // Formatters
    const fmtNY = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const fmtUK = new Intl.DateTimeFormat('en-GB', { timeZone: 'Europe/London', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const fmtJP = new Intl.DateTimeFormat('ja-JP', { timeZone: 'Asia/Tokyo', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // Direct update attempt
    try {
        const doc = window.parent.document; // Try to reach parent if in iframe
        const elNY = document.getElementById('clock-ny') || doc.getElementById('clock-ny');
        const elUK = document.getElementById('clock-uk') || doc.getElementById('clock-uk');
        const elJP = document.getElementById('clock-jp') || doc.getElementById('clock-jp');
        
        if(elNY) elNY.innerText = fmtNY.format(now);
        if(elUK) elUK.innerText = fmtUK.format(now);
        if(elJP) elJP.innerText = fmtJP.format(now);
    } catch(e) {}
}
setInterval(updateTime, 1000);
updateTime();
</script>
""", unsafe_allow_html=True)

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
    # 👑 THE MAGNIFICENT 7 & LEADERS
    "NVIDIA (NVDA)": "NVDA", "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN", "Google (GOOGL)": "GOOGL", "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA", "Netflix (NFLX)": "NFLX", "Broadcom (AVGO)": "AVGO",

    # 💾 SEMICONDUCTORS & HARDWARE
    "AMD (AMD)": "AMD", "Intel (INTC)": "INTC", "TSMC (TSM)": "TSM",
    "Micron (MU)": "MU", "Qualcomm (QCOM)": "QCOM", "ARM Holdings (ARM)": "ARM",
    "Super Micro (SMCI)": "SMCI", "Dell (DELL)": "DELL",

    # ⛏️ CRYPTO MINERS & PROXIES
    "Marathon Digital (MARA)": "MARA", "Riot Platforms (RIOT)": "RIOT",
    "CleanSpark (CLSK)": "CLSK", "Hut 8 (HUT)": "HUT", "Coinbase (COIN)": "COIN",
    "MicroStrategy (MSTR)": "MSTR", "Iris Energy (IREN)": "IREN",
    "Core Scientific (CORZ)": "CORZ",

    # 🧠 AI & BIG DATA
    "Palantir (PLTR)": "PLTR", "Snowflake (SNOW)": "SNOW", "C3.ai (AI)": "AI",
    "CrowdStrike (CRWD)": "CRWD", "Palo Alto (PANW)": "PANW",

    # 🏭 TRADITIONAL MINING (Metals)
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
    custom_symbol = st.sidebar.text_input("Enter Ticker (e.g. GME, ^GSPC)", value="NVDA").upper().strip()
    ticker = custom_symbol
    ticker_name = f"{custom_symbol} (Global)"

elif search_mode == "Search UK (LSE)":
    uk_symbol = st.sidebar.text_input("Enter UK Ticker (e.g. RR, BARC)", value="RR").upper().strip()
    if not uk_symbol.endswith(".L"):
        ticker = f"{uk_symbol}.L"
    else:
        ticker = uk_symbol
    ticker_name = f"{ticker} (LSE)"

elif search_mode == "Search Japan (TSE)":
    jp_symbol = st.sidebar.text_input("Enter Japan Code (e.g. 7203, 9984)", value="7203").upper().strip()
    if not jp_symbol.endswith(".T"):
        ticker = f"{jp_symbol}.T"
    else:
        ticker = jp_symbol
    ticker_name = f"{ticker} (TSE)"

# Default to Daily (index 3)
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

# --- INSTITUTIONAL DATA FETCHING (1D & 1W) ---
@st.cache_data(ttl=600)
def get_institutional_trend(ticker):
    try:
        # Download 1 Year of Daily Data
        df_d = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
        
        # Download 2 Years of Weekly Data
        df_w = yf.download(ticker, period="2y", interval="1wk", progress=False)
        if isinstance(df_w.columns, pd.MultiIndex): df_w.columns = df_w.columns.get_level_values(0)
        
        # Calculate 50 EMAs
        ema_d = df_d['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        ema_w = df_w['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        
        return ema_d, ema_w
    except:
        return 0, 0

def calculate_engine(df, ticker):
    """Core Calculation Pipeline: 14 Indicators"""
    
    # 1. APEX TREND MASTER
    apex_len = 55
    apex_mult = 1.5
    df['Apex_Base'] = calculate_hma(df['Close'], apex_len)
    df['Apex_ATR'] = calculate_atr(df, apex_len)
    df['Apex_Upper'] = df['Apex_Base'] + (df['Apex_ATR'] * apex_mult)
    df['Apex_Lower'] = df['Apex_Base'] - (df['Apex_ATR'] * apex_mult)
    df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, 
                        np.where(df['Close'] < df['Apex_Lower'], -1, np.nan))
    df['Apex_Trend'] = df['Apex_Trend'].ffill().fillna(0)

    # 2. LIQUIDITY ZONES (Pivots)
    liq_len = 10
    df['Pivot_High'] = df['High'].rolling(liq_len*2+1, center=True).max()
    df['Pivot_Low'] = df['Low'].rolling(liq_len*2+1, center=True).min()
    df['Supply_Zone'] = np.where(df['High'] == df['Pivot_High'], df['High'], np.nan)
    df['Demand_Zone'] = np.where(df['Low'] == df['Pivot_Low'], df['Low'], np.nan)

    # 3. MONEY FLOW MATRIX (Superior Replacement for Squeeze)
    # 3.1 Money Flow (RSI * Vol Ratio)
    mfLen = 14
    mfSmooth = 3
    # RSI Calculation
    delta_mf = df['Close'].diff()
    gain_mf = delta_mf.where(delta_mf > 0, 0)
    loss_mf = -delta_mf.where(delta_mf < 0, 0)
    avg_gain_mf = gain_mf.ewm(alpha=1/mfLen, adjust=False).mean()
    avg_loss_mf = loss_mf.ewm(alpha=1/mfLen, adjust=False).mean()
    rs_mf = avg_gain_mf / avg_loss_mf
    rsi_mf = 100 - (100 / (1 + rs_mf))
    
    rsiSource = rsi_mf - 50
    mfVolume = df['Volume'] / df['Volume'].rolling(mfLen).mean()
    df['Matrix_MF'] = (rsiSource * mfVolume).ewm(span=mfSmooth, adjust=False).mean()
    
    # 3.2 Dynamic Thresholds (Bollinger on Money Flow)
    bbLen = 20
    bbMult = 2.0
    df['MF_BB_Mid'] = df['Matrix_MF'].rolling(bbLen).mean()
    df['MF_BB_Std'] = df['Matrix_MF'].rolling(bbLen).std()
    df['MF_BB_Up'] = df['MF_BB_Mid'] + (df['MF_BB_Std'] * bbMult)
    df['MF_BB_Low'] = df['MF_BB_Mid'] - (df['MF_BB_Std'] * bbMult)
    
    # 3.3 Hyper Wave (TSI)
    tsiLong = 25
    tsiShort = 13
    pc = df['Close'].diff()
    # Double Smooth Function using EWM
    ds_pc = pc.ewm(span=tsiLong, adjust=False).mean().ewm(span=tsiShort, adjust=False).mean()
    ds_abs_pc = pc.abs().ewm(span=tsiLong, adjust=False).mean().ewm(span=tsiShort, adjust=False).mean()
    df['Matrix_HyperWave'] = (100 * (ds_pc / ds_abs_pc)) / 2

    # 4. GANN HIGH LOW
    gann_len = 3
    df['Gann_High'] = df['High'].rolling(gann_len).mean()
    df['Gann_Low'] = df['Low'].rolling(gann_len).mean()
    df['Gann_Trend'] = np.where(df['Close'] > df['Gann_High'].shift(1), 1, 
                        np.where(df['Close'] < df['Gann_Low'].shift(1), -1, np.nan))
    df['Gann_Trend'] = df['Gann_Trend'].ffill().fillna(0)

    # 5. VOLUME DELTA
    df['Vol_Delta'] = np.where(df['Close'] >= df['Open'], df['Volume'], -df['Volume'])

    # 6. MFI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    pos_flow = np.where(tp > tp.shift(1), rmf, 0)
    neg_flow = np.where(tp < tp.shift(1), rmf, 0)
    mfi_ratio = pd.Series(pos_flow).rolling(14).sum() / pd.Series(neg_flow).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # 7. OPTIMIZED ADX
    up = df['High'].diff()
    down = -df['Low'].diff()
    p_dm = np.where((up > down) & (up > 0), up, 0)
    m_dm = np.where((down > up) & (down > 0), down, 0)
    tr = calculate_atr(df, 14)
    p_di = 100 * calculate_rma(pd.Series(p_dm), 14) / tr
    m_di = 100 * calculate_rma(pd.Series(m_dm), 14) / tr
    dx = (np.abs(p_di - m_di) / (p_di + m_di)) * 100
    df['ADX'] = calculate_rma(dx, 14)

    # 8. ICHIMOKU
    nine_high = df['High'].rolling(window=9).max()
    nine_low = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (nine_high + nine_low) / 2
    twenty_six_high = df['High'].rolling(window=26).max()
    twenty_six_low = df['Low'].rolling(window=26).min()
    df['Kijun'] = (twenty_six_high + twenty_six_low) / 2
    df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    fifty_two_high = df['High'].rolling(window=52).max()
    fifty_two_low = df['Low'].rolling(window=52).min()
    df['SpanB'] = ((fifty_two_high + fifty_two_low) / 2).shift(26)

    # 9. EMA
    df['EMA_200'] = df['Close'].ewm(span=200).mean()

    # 10. DARKPOOL MACD
    e12 = df['Close'].ewm(span=12).mean()
    e26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = e12 - e26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Hist'] = (df['MACD'] - df['Signal']) * (df['Volume'] / df['Volume'].rolling(20).mean())

    # 11. STOCHASTIC RSI (FIXED)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stoch Logic
    min_rsi = df['RSI'].rolling(14).min()
    max_rsi = df['RSI'].rolling(14).max()
    stoch = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    df['Stoch_K'] = stoch.rolling(3).mean() * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # 12. ATR (Already calc for ADX/Apex)
    
    # 13. DARKPOOL MOVING AVERAGES (5-Layer System)
    df['DP_MA10'] = df['Close'].ewm(span=10).mean()
    df['DP_MA20'] = df['Close'].ewm(span=20).mean()
    df['DP_MA50'] = df['Close'].ewm(span=50).mean()
    df['DP_MA100'] = df['Close'].ewm(span=100).mean()
    df['DP_MA200'] = df['Close'].ewm(span=200).mean()
    
    df['DP_Score'] = (
        np.where(df['Close'] > df['DP_MA10'], 1, 0) + 
        np.where(df['Close'] > df['DP_MA20'], 1, 0) + 
        np.where(df['Close'] > df['DP_MA50'], 1, 0) + 
        np.where(df['Close'] > df['DP_MA100'], 1, 0) + 
        np.where(df['Close'] > df['DP_MA200'], 1, 0)
    )

    # 14. INSTITUTIONAL TREND (1D & 1W EMA Cloud)
    # We fetch this once and apply it to the whole column for the AI to read
    ema_d, ema_w = get_institutional_trend(ticker)
    df['Inst_EMA_D'] = ema_d
    df['Inst_EMA_W'] = ema_w
    df['Inst_Trend'] = np.where(ema_d > ema_w, 1, -1) # 1 = Bullish, -1 = Bearish

    return df

def calculate_strategies(df):
    """4 Core Strategy Signals"""
    # 1. Momentum (12)
    mom = df['Close'] - df['Close'].shift(12)
    df['Sig_Mom'] = np.where((mom > 0) & (mom.shift(1) > 0), 1, np.where((mom < 0) & (mom.shift(1) < 0), -1, 0))
    
    # 2. ADX Breakout (Close > 20 High + ADX < 25)
    box_high = df['High'].rolling(20).max().shift(1)
    box_low = df['Low'].rolling(20).min().shift(1)
    df['Sig_ADX'] = np.where((df['Close'] > box_high) & (df['ADX'] < 25), 1,
                      np.where((df['Close'] < box_low) & (df['ADX'] < 25), -1, 0))

    # 3. Bollinger Directed
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    # Close < Lower (Dip Buy attempt) or Close > Upper (Breakout/Overbought)
    # Simple crossover logic:
    df['Sig_BB'] = np.where(df['Close'] < (sma20 - 2*std20), 1, np.where(df['Close'] > (sma20 + 2*std20), -1, 0))

    # 4. RSI Strategy (30/70)
    # Uses the fixed RSI from calculate_engine
    df['Sig_RSI'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    
    return df

@st.cache_data(ttl=300)
def get_data(ticker, interval):
    p_map = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
    period = p_map.get(interval, "1y")
    d_int = "1h" if interval == "4h" else interval
    df = yf.download(ticker, period=period, interval=d_int, progress=False)
    
    # Handle MultiIndex (Fix for new yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    if interval == "4h":
        agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
        df = df.resample('4h').agg(agg).dropna()
        
    return df

# ==========================================
# 3. DASHBOARD RENDERER
# ==========================================
st.markdown(f'<div class="titan-header">🏢 TITAN TERMINAL: {ticker_name}</div>', unsafe_allow_html=True)

# 1. TRADINGVIEW WIDGET (Enhanced for Global)
tv_int_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}

# Smart TradingView Symbol Mapper
if search_mode == "Search UK (LSE)":
    # yf uses "RR.L", TV uses "LSE:RR"
    clean_sym = ticker.replace(".L", "")
    tv_sym = f"LSE:{clean_sym}"
elif search_mode == "Search Japan (TSE)":
    # yf uses "7203.T", TV uses "TSE:7203"
    clean_sym = ticker.replace(".T", "")
    tv_sym = f"TSE:{clean_sym}"
else:
    # Default behavior for US/Global
    tv_sym = ticker

components.html(
    f"""<div class="tradingview-widget-container"><div id="tradingview_widget"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{"width": "100%","height": 450,"symbol": "{tv_sym}","interval": "{tv_int_map[interval]}","timezone": "Etc/UTC","theme": "dark","style": "1","locale": "en","toolbar_bg": "#f1f3f6","enable_publishing": false,"hide_side_toolbar": false,"allow_symbol_change": true,"container_id": "tradingview_widget"}});</script></div>""",
    height=460,
)

df = get_data(ticker, interval)

if df is not None and not df.empty:
    df = calculate_engine(df, ticker)
    df = calculate_strategies(df)
    last = df.iloc[-1]
    
    # --- PRE-CALCULATE LEVELS FOR BROADCAST & AI ---
    curr_price = last['Close']
    atr_val = last['ATR']
    trend_dir = "LONG" if last['Apex_Trend'] == 1 else "SHORT" if last['Apex_Trend'] == -1 else "NEUTRAL"
    
    if trend_dir == "LONG":
        stop_loss = curr_price - (2 * atr_val)
        take_profit = curr_price + (3 * atr_val)
        entry_price = curr_price
    else:
        stop_loss = curr_price + (2 * atr_val)
        take_profit = curr_price - (3 * atr_val)
        entry_price = curr_price
        
    # Variables for AI Context
    stop_long = curr_price - (2 * atr_val)
    stop_short = curr_price + (2 * atr_val)
    target_long = curr_price + (3 * atr_val)
    target_short = curr_price - (3 * atr_val)
    cloud_top = last['Apex_Upper']
    cloud_bot = last['Apex_Lower']

    # --- 2. SIGNAL HUD ---
    st.markdown("### 🧬 Market DNA")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    def card(label, value, condition):
        color_class = "bull" if condition == 1 else "bear" if condition == -1 else "neu"
        return f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {color_class}">{value}</div></div>"""
    
    # Apex Trend
    apex_cond = last['Apex_Trend']
    apex_txt = "BULLISH" if apex_cond == 1 else "BEARISH" if apex_cond == -1 else "NEUTRAL"
    c1.markdown(card("Apex Trend", apex_txt, apex_cond), unsafe_allow_html=True)
    
    # Momentum (Strat 1)
    mom_cond = last['Sig_Mom']
    mom_txt = "POSITIVE" if mom_cond == 1 else "NEGATIVE"
    c2.markdown(card("Momentum", mom_txt, mom_cond), unsafe_allow_html=True)
    
    # Institutional Trend (Indicator 14)
    inst_cond = last['Inst_Trend']
    inst_txt = "MACRO BULL" if inst_cond == 1 else "MACRO BEAR"
    c3.markdown(card("Inst. Trend (1D/1W)", inst_txt, inst_cond), unsafe_allow_html=True)
    
    # Money Flow
    mfi_val = last['MFI']
    mfi_cond = 1 if mfi_val < 20 else -1 if mfi_val > 80 else 0
    mfi_txt = "OVERSOLD" if mfi_val < 20 else "OVERBOUGHT" if mfi_val > 80 else "NEUTRAL"
    c4.markdown(card("Money Flow", mfi_txt, mfi_cond), unsafe_allow_html=True)
    
    # Bollinger Strat
    bb_cond = last['Sig_BB']
    bb_txt = "DIP BUY" if bb_cond == 1 else "RIP SELL" if bb_cond == -1 else "WAIT"
    c5.markdown(card("Bollinger", bb_txt, bb_cond), unsafe_allow_html=True)

    # --- 3. TABS ---
    st.markdown("<br>", unsafe_allow_html=True)
    tab_apex, tab_dpma, tab_matrix, tab_cloud, tab_osc, tab_ai, tab_cast = st.tabs(["🌊 Apex Master", "💀 DarkPool Trends", "🔋 Money Flow Matrix", "☁️ Ichimoku", "📈 Oscillators", "🤖 AI Analyst", "📡 Broadcast"])

    with tab_apex:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.15)' if last['Apex_Trend'] == 1 else 'rgba(255, 23, 68, 0.15)', line=dict(width=0), name="Apex Cloud"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Base'], line=dict(color='#00E676' if last['Apex_Trend'] == 1 else '#FF1744', width=2), name="Apex Base"))
        # Gann Trace
        fig.add_trace(go.Scatter(x=df.index, y=df['Gann_High'], line=dict(color='yellow', width=1, dash='dot'), name="Gann Level"))
        fig.update_layout(height=600, template="plotly_dark", title=f"Apex Trend & Liquidity ({interval})", xaxis_rangeslider_visible=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    with tab_dpma:
        # DarkPool MAs (#13)
        fig_dp = go.Figure()
        fig_dp.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA10'], line=dict(color='#00E5FF', width=1), name="EMA 10 (Fast)"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA20'], line=dict(color='#2979FF', width=1), name="EMA 20"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA50'], line=dict(color='#00E676', width=2), name="EMA 50 (Trend)"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA100'], line=dict(color='#FF9100', width=2), name="EMA 100"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA200'], line=dict(color='#FF1744', width=2), name="EMA 200 (Inst)"))
        fig_dp.update_layout(height=600, template="plotly_dark", title=f"DarkPool Institutional MAs ({interval})", xaxis_rangeslider_visible=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig_dp, use_container_width=True)

    with tab_matrix:
        # REPLACED SQUEEZE WITH MONEY FLOW MATRIX
        fig_matrix = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05)
        
        # 1. Money Flow Bars
        mf_colors = ['#00E676' if v > 0 else '#FF1744' for v in df['Matrix_MF']]
        fig_matrix.add_trace(go.Bar(x=df.index, y=df['Matrix_MF'], marker_color=mf_colors, name="Money Flow"), row=1, col=1)
        # Hyper Wave Line Overlay
        fig_matrix.add_trace(go.Scatter(x=df.index, y=df['Matrix_HyperWave'], line=dict(color='white', width=2), name="Hyper Wave"), row=1, col=1)
        # Threshold Bands
        fig_matrix.add_trace(go.Scatter(x=df.index, y=df['MF_BB_Up'], line=dict(color='gray', width=1, dash='dot'), name="Upper Band"), row=1, col=1)
        fig_matrix.add_trace(go.Scatter(x=df.index, y=df['MF_BB_Low'], line=dict(color='gray', width=1, dash='dot'), name="Lower Band"), row=1, col=1)
        
        # 2. MACD (Standard)
        fig_matrix.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color='cyan', name="MACD Hist"), row=2, col=1)
        
        # 3. Vol Delta
        vd_col = ['#00E676' if v > 0 else '#FF1744' for v in df['Vol_Delta']]
        fig_matrix.add_trace(go.Bar(x=df.index, y=df['Vol_Delta'], marker_color=vd_col, name="Vol Delta"), row=3, col=1)
        
        fig_matrix.update_layout(height=600, template="plotly_dark", title="Money Flow Matrix & Cycles", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig_matrix, use_container_width=True)

    with tab_cloud:
        fig_ichi = go.Figure()
        fig_ichi.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['Tenkan'], line=dict(color='#2962FF', width=1), name="Tenkan"))
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['Kijun'], line=dict(color='#B71C1C', width=1), name="Kijun"))
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['SpanA'], line=dict(width=0), showlegend=False))
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['SpanB'], fill='tonexty', fillcolor='rgba(67, 160, 71, 0.2)', line=dict(width=0), name="Kumo Cloud"))
        fig_ichi.update_layout(height=500, template="plotly_dark", title="Ichimoku Cloud", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_ichi, use_container_width=True)

    with tab_osc:
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line=dict(color='#2962FF', width=2), name="K%"))
        fig_stoch.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], line=dict(color='#FF6D00', width=2), name="D%"))
        fig_stoch.add_hline(y=80, line_color="gray", line_dash="dot")
        fig_stoch.add_hline(y=20, line_color="gray", line_dash="dot")
        fig_stoch.update_layout(height=400, template="plotly_dark", title="Stochastic RSI")
        st.plotly_chart(fig_stoch, use_container_width=True)

    with tab_ai:
        if st.button("Generate Alpha Report"):
            if not api_key:
                st.error("Missing OpenAI API Key")
            else:
                with st.spinner("Calculating Precise Levels..."):
                    client = OpenAI(api_key=api_key)
                    
                    # AI Context variables regarding Matrix
                    mf_last = last['Matrix_MF']
                    hw_last = last['Matrix_HyperWave']
                    
                    prompt = f"""
                    Act as a Senior Wall Street Quantitative Analyst. 
                    Analyze {ticker} ({interval}) at Price ${curr_price:.2f}.
                    
                    --- TECHNICAL DATA ---
                    1. ATR (Volatility): ${atr_val:.2f}
                    2. Apex Trend: {apex_txt} (Cloud Top: ${cloud_top:.2f}, Bot: ${cloud_bot:.2f})
                    3. DarkPool MAs: {last['DP_Score']:.0f}/5 (Institutional Trend)
                    4. Institutional Trend (1D/1W): {inst_txt}
                    5. Liquidity: Supply @ ${last['Pivot_High']:.2f}, Demand @ ${last['Pivot_Low']:.2f}
                    6. Money Flow Matrix: Flow={mf_last:.2f}, HyperWave={hw_last:.2f} (Pos=Bull, Neg=Bear)
                    
                    --- STRATEGY SIGNALS ---
                    - ADX Breakout: {'YES' if last['Sig_ADX'] != 0 else 'NO'}
                    - Bollinger: {bb_txt}
                    - RSI: {last['Sig_RSI']}
                    
                    --- CALCULATED LEVELS (Use these if valid) ---
                    - LONG SETUP: Stop < ${stop_long:.2f}, Target > ${target_long:.2f}
                    - SHORT SETUP: Stop > ${stop_short:.2f}, Target < ${target_short:.2f}
                    
                    --- MISSION ---
                    Synthesize a TRADE PLAN in strictly formatted MARKDOWN.
                    Do not print messy raw numbers. Round to 2 decimal places.
                    Focus on Equity/Tech specific dynamics (Sector strength, correlations).
                    
                    OUTPUT FORMAT:
                    ### 📋 Trade Plan: Equity Alpha Report
                    
                    **1. VERDICT:** LONG / SHORT / WAIT (Confidence Level)
                    * **Rationale:** (1 sentence confluence summary)
                    
                    **2. ENTRY ZONE**
                    * **Price Range:** (e.g. $150.00 - $151.50)
                    * **Rationale:** (Technical basis)
                    
                    **3. STOP LOSS**
                    * **Hard Stop:** (Specific Price)
                    * **Rationale:** (e.g. Below Apex Cloud)
                    
                    **4. TAKE PROFIT**
                    * **Conservative:** (Price)
                    * **Aggressive:** (Price)
                    
                    **5. TRAILING STOP**
                    * **Dynamic Rule:** (e.g. Close below ${cloud_bot:.2f})
                    """
                    res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":prompt}])
                    st.info(res.choices[0].message.content)

    with tab_cast:
        st.subheader("📡 Broadcast Center")
        
        # UPGRADED TELEGRAM MESSAGE FORMAT
        sig_emoji = "🟢" if apex_cond == 1 else "🔴" if apex_cond == -1 else "⚪"
        direction = "LONG" if apex_cond == 1 else "SHORT" if apex_cond == -1 else "NEUTRAL"
        
        broadcast_msg = f"""
🔥 TITAN EQUITY SIGNAL: {ticker} ({interval})
{sig_emoji} DIRECTION: {direction}
🚪 ENTRY: ${entry_price:,.2f}
🛑 STOP LOSS: ${stop_loss:,.2f}
🎯 TARGET: ${take_profit:,.2f}
🌊 Trend: {apex_txt}
📊 Momentum: {mom_txt}
💰 Money Flow: {mfi_txt}
💀 Institutional Trend: {inst_txt}
⚠️ *Not financial advice. DYOR.*
#Stocks #Tech #Titan #{ticker}
        """
        
        msg_preview = st.text_area("Message Preview", value=broadcast_msg, height=250)
        
        c_tg, c_x = st.columns(2)
        
        if c_tg.button("Send to Telegram 🚀"):
            if tg_token and tg_chat:
                try:
                    url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                    data = {"chat_id": tg_chat, "text": msg_preview}
                    requests.post(url, data=data)
                    st.success("Signal Broadcasted!")
                except Exception as e:
                    st.error(f"Failed: {e}")
            else:
                st.warning("⚠️ Enter Telegram Keys in Sidebar")
                
        if c_x.button("Post to X (Twitter)"):
            encoded = urllib.parse.quote(msg_preview)
            st.link_button("🐦 Launch Tweet", f"https://twitter.com/intent/tweet?text={encoded}")

else:
    st.error("Initializing Data Stream... Markets may be closed or Ticker invalid.")
