"""
TITAN INTRADAY PRO - Production-Ready Trading Dashboard
Version 17.0: The "Gann Activation" Update (Gann HiLo + 6-Engine Core)
"""
import time
import math
import sqlite3
import atexit
import random
from typing import Dict, Optional, List
from contextlib import contextmanager

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime, timezone

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
ADX_THRESHOLD = 23.0
RVOL_THRESHOLD = 1.15
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
DB_PATH = "titan_signals.db"
MAX_RETRIES = 2
RETRY_DELAY = 0.5

# SAFE DEFAULTS
mf_len = 14
vol_len = 20
amplitude = 10
channel_dev = 3.0
hma_len = 50
use_hma_filter = True
tp1_r = 1.5
tp2_r = 3.0
tp3_r = 5.0
gann_len = 3  # Default Gann Length

# API ENDPOINTS
BINANCE_API_BASE = "https://api.binance.us/api/v3"
BYBIT_API_BASE = "https://api.bybit.com/v5/market/kline"
COINBASE_API_BASE = "https://api.exchange.coinbase.com/products"

# HEADERS
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

# TOP 100 CRYPTO ASSETS
TOP_100_ASSETS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "SHIB", "TON",
    "DOT", "LINK", "BCH", "TRX", "LTC", "NEAR", "MATIC", "UNI", "APT", "ICP",
    "PEPE", "WIF", "FET", "RNDR", "STX", "FIL", "ATOM", "ARB", "IMX", "KAS",
    "ETC", "OP", "INJ", "GRT", "LDO", "TIA", "VET", "FLOKI", "BONK", "MKR",
    "RUNE", "SEI", "ALGO", "ORDI", "EGLD", "FLOW", "QNT", "GALA", "AAVE", "SNX",
    "SAND", "HBAR", "AXS", "BSV", "MINA", "BEAM", "EOS", "BTT", "MANA", "XLM",
    "KCS", "CAKE", "NEO", "CHZ", "JUP", "APE", "IOTA", "XTZ", "LUNC", "BLUR",
    "ZEC", "KLAY", "CFX", "GNO", "ROSE", "DYDX", "CRV", "WOO", "1INCH", "COMP",
    "RPL", "ZIL", "FXS", "BAT", "LRC", "ENJ", "MASK", "QTUM", "TWT", "CELO"
]
TOP_100_SYMBOLS = [f"{asset}USDT" for asset in TOP_100_ASSETS]

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================
st.set_page_config(
    page_title="TITAN TERMINAL",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Background */
    .main { background-color: #0b0c10; }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background: rgba(31, 40, 51, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(102, 252, 241, 0.1);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        border-color: #66fcf1;
        box-shadow: 0 0 15px rgba(102, 252, 241, 0.2);
    }
    
    /* Fonts */
    h1, h2, h3, h4, h5 { 
        font-family: 'Roboto Mono', monospace; 
        color: #c5c6c7; 
        font-weight: 700;
    }
    p, span, div { font-family: 'Inter', sans-serif; color: #c5c6c7; }
    
    /* Buttons */
    .stButton > button {
        border-radius: 4px; 
        font-weight: 700;
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #45a29e;
        color: #66fcf1;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        background: #45a29e;
        color: #0b0c10;
        border-color: #66fcf1;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { 
        background-color: #050608; 
        border-right: 1px solid #1f2833;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 4px 4px 0 0;
        background-color: #1f2833;
        color: #c5c6c7;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0b0c10;
        color: #66fcf1;
        border-top: 2px solid #66fcf1;
    }
</style>
""", unsafe_allow_html=True)

# HEADER
c_head1, c_head2 = st.columns([3, 1])
with c_head1:
    st.title("💠 TITAN TERMINAL v17")
    st.caption("AI-POWERED MULTI-ENGINE EXECUTION SUITE")
with c_head2:
    utc_now = datetime.now(timezone.utc)
    st.metric("UTC TIME", utc_now.strftime("%H:%M:%S"))

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.header("⚙️ SYSTEM CONTROL")
    
    # Session
    hour = utc_now.hour
    session = "🌏 ASIA"
    if 7 <= hour < 15: session = "🇪🇺 LONDON"
    elif 12 <= hour < 21: session = "🇺🇸 NEW YORK"
    st.info(f"ACTIVE SESSION: {session}")

    with st.expander("📚 Engines Guide", expanded=False):
        st.markdown("""
        **1. TITAN (Execution)**
        * ATR Trailing Stops & Trend Structure.
        
        **2. APEX (Structure)**
        * HMA Trend Cloud & Pivot Liquidity.
        
        **3. GANN (Trend)**
        * High/Low Activator Step-Line.
        * Green = Bull, Red = Bear.
        
        **4. MATRIX (Flow)**
        * Money Flow Index + Hyper Wave.
        """)

    st.subheader("📡 FEED")
    symbol_select = st.selectbox("Asset", options=TOP_100_SYMBOLS, index=0)
    symbol = symbol_select.strip().upper().replace("/", "").replace("-", "")
    timeframe = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    limit = st.slider("Depth", 100, 1000, 300, 50)

    st.markdown("---")
    st.subheader("🧠 STRATEGY")
    amplitude = st.number_input("Amplitude", 2, 200, 10)
    channel_dev = st.number_input("ATR Dev", 0.5, 10.0, 3.0, 0.1)
    hma_len = st.number_input("HMA Len", 2, 400, 50)
    gann_len = st.number_input("Gann Len", 1, 50, 3) # NEW
    
    with st.expander("🎯 Targets (R)"):
        tp1_r = st.number_input("TP1", value=1.5, step=0.1)
        tp2_r = st.number_input("TP2", value=3.0, step=0.1)
        tp3_r = st.number_input("TP3", value=5.0, step=0.1)

    st.markdown("---")
    st.subheader("📊 VOL METRICS")
    hero_metric = st.selectbox("Hero Metric", ["CMF", "Volume RSI", "Volume Oscillator", "RVOL"])
    mf_len = st.number_input("MF Len", 2, 200, 14)
    vol_len = st.number_input("Vol Len", 5, 200, 20)

    st.markdown("---")
    st.subheader("🤖 TELEGRAM")
    tg_on = st.checkbox("Auto-Broadcast", False)
    
    # --- FIXED: LOAD FROM SECRETS ---
    # Try loading from secrets.toml first
    try: sec_token = st.secrets["TELEGRAM_TOKEN"]
    except: sec_token = ""
    
    try: sec_chat = st.secrets["TELEGRAM_CHAT_ID"]
    except: sec_chat = ""

    # Use secrets as default value
    tg_token = st.text_input("Token", value=sec_token, type="password")
    tg_chat = st.text_input("Chat ID", value=sec_chat)

    if st.button("Test Link"):
        t_clean = tg_token.strip()
        c_clean = tg_chat.strip()
        
        if not t_clean or not c_clean:
            st.error("Missing Creds (Check secrets.toml or Inputs)")
        else:
            try:
                r = requests.post(f"https://api.telegram.org/bot{t_clean}/sendMessage", json={"chat_id": c_clean, "text": "💠 TITAN: ONLINE"})
                if r.status_code == 200:
                    st.success("Linked! ✅")
                else:
                    st.error(f"Error {r.status_code}: Check Token/Chat ID")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

    persist = st.checkbox("DB Log", True)
    telegram_cooldown_s = st.number_input("Cooldown (s)", 5, 600, 30)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def calculate_hma(series, length):
    half_len = int(length / 2)
    sqrt_len = int(math.sqrt(length))
    wma_f = series.rolling(length).mean()
    wma_h = series.rolling(half_len).mean()
    diff = 2 * wma_h - wma_f
    return diff.rolling(sqrt_len).mean()

def calculate_fibonacci(df, lookback=50):
    recent = df.iloc[-lookback:]
    h, l = recent['high'].max(), recent['low'].min()
    d = h - l
    return {'fib_382': h - (d*0.382), 'fib_500': h - (d*0.5), 'fib_618': h - (d*0.618)}

# =============================================================================
# ANALYST & SENTIMENT
# =============================================================================
def calculate_fear_greed_index(df):
    try:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        vol_score = 50 - ((df['log_ret'].rolling(30).std().iloc[-1] - df['log_ret'].rolling(90).std().iloc[-1]) / df['log_ret'].rolling(90).std().iloc[-1]) * 100
        vol_score = max(0, min(100, vol_score))
        rsi = df['rsi'].iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        dist = (df['close'].iloc[-1] - sma_50) / sma_50
        trend_score = 50 + (dist * 1000)
        fg = (vol_score * 0.3) + (rsi * 0.4) + (max(0, min(100, trend_score)) * 0.3)
        return int(fg)
    except: return 50

def generate_ai_analysis(row, symbol, tf, fibs, fg_index):
    titan_trend = "BULLISH" if row['is_bull'] else "BEARISH"
    apex_trend = "BULLISH" if row['apex_trend'] == 1 else "BEARISH" if row['apex_trend'] == -1 else "NEUTRAL"
    gann_trend = "BULLISH" if row['gann_trend'] == 1 else "BEARISH"
    
    # Confluence Check
    confluence = "MIXED"
    if titan_trend == apex_trend == gann_trend:
        confluence = "⭐⭐⭐ TRIPLE CONFLUENCE"
    elif titan_trend == gann_trend:
        confluence = "⭐⭐ DOUBLE CONFLUENCE"

    sent = "NEUTRAL"
    if fg_index >= 75: sent = "EXTREME GREED 🤑"
    elif fg_index >= 55: sent = "GREED 🟢"
    elif fg_index <= 25: sent = "EXTREME FEAR 😱"
    elif fg_index <= 45: sent = "FEAR 🔴"

    return (
        f"**🤖 TITAN AI Analyst Report**\n\n"
        f"**1. Market Regime:**\n"
        f"• Confluence: **{confluence}**\n"
        f"• TITAN: **{titan_trend}**\n"
        f"• GANN: **{gann_trend}**\n"
        f"• Sentiment: {sent} ({fg_index})\n\n"
        f"**2. Volume & Flow:**\n"
        f"• Money Flow: {'Inflow' if row['money_flow'] > 0 else 'Outflow'}\n"
        f"• Trap Candle: {'YES ⚠️' if row['hidden_liq'] else 'No'}\n"
        f"• Golden Zone: {fibs['fib_500']:.2f}\n\n"
        f"**3. Plan:**\n"
        f"Invalidation at {row['entry_stop']:.2f}."
    )

# =============================================================================
# DATA ENGINE
# =============================================================================
@contextmanager
def get_db_connection(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    try: yield conn
    finally: conn.close()

def init_db(path: str = DB_PATH):
    if not persist: return None
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, symbol TEXT, timeframe TEXT, direction TEXT,
            entry REAL, stop REAL, tp1 REAL, tp2 REAL, tp3 REAL,
            adx REAL, rvol REAL, notes TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

if 'db_conn' not in st.session_state: st.session_state.db_conn = init_db()

def journal_signal(conn, payload):
    if not conn: return False
    try:
        conn.execute("""
            INSERT INTO signals (ts, symbol, timeframe, direction, entry, stop, tp1, tp2, tp3, adx, rvol, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (payload['ts'], payload['symbol'], payload['timeframe'], payload['direction'], 
              payload['entry'], payload['stop'], payload['tp1'], payload['tp2'], payload['tp3'],
              payload['adx'], payload['rvol'], payload['notes']))
        conn.commit()
        return True
    except: return False

def send_telegram_msg(token, chat, msg, cooldown):
    if not token or not chat: return False
    # Strip whitespace
    token = token.strip()
    chat = chat.strip()
    
    last = st.session_state.get("last_tg", 0)
    if time.time() - last < cooldown: return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"}, timeout=5)
        if r.status_code == 200:
            st.session_state["last_tg"] = time.time()
            return True
    except: pass
    return False

@st.cache_data(ttl=5)
def get_klines(symbol_bin: str, interval: str, limit: int) -> pd.DataFrame:
    # 1. Binance US
    try:
        r = requests.get(f"{BINANCE_API_BASE}/klines", params={"symbol": symbol_bin, "interval": interval, "limit": limit}, headers=HEADERS, timeout=4)
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
            return df[['timestamp','open','high','low','close','volume']]
    except: pass
    # 2. Bybit
    try:
        imap = {"1m":"1", "5m":"5", "15m":"15", "1h":"60", "4h":"240", "1d":"D"}
        r = requests.get(BYBIT_API_BASE, params={"category":"spot", "symbol":symbol_bin, "interval":imap.get(interval,"60"), "limit":limit}, headers=HEADERS, timeout=4)
        if r.json().get('retCode') == 0:
            df = pd.DataFrame(r.json()['result']['list'], columns=['t','o','h','l','c','v','to'])
            df['timestamp'] = pd.to_datetime(df['t'].astype(float), unit='ms')
            df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
            return df.iloc[::-1].reset_index(drop=True)[['timestamp','open','high','low','close','volume']]
    except: pass
    # 3. Coinbase
    try:
        gmap = {"1m":60, "5m":300, "15m":900, "1h":3600, "4h":21600, "1d":86400}
        sym_cb = f"{symbol_bin[:-4]}-{symbol_bin[-4:]}" if symbol_bin.endswith("USDT") else "BTC-USD"
        r = requests.get(f"{COINBASE_API_BASE}/{sym_cb}/candles", params={"granularity": gmap.get(interval, 3600)}, headers=HEADERS, timeout=5)
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=['t','l','h','o','c','v'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
            df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
            return df.sort_values('timestamp').reset_index(drop=True).iloc[-limit:]
    except: pass
    return pd.DataFrame()

# =============================================================================
# MULTI-CORE LOGIC ENGINE
# =============================================================================
def run_engines(df, amp, dev, hma_l, hma_on, tp1, tp2, tp3, mf_l, vol_l, gann_l):
    if df.empty: return df
    df = df.copy().reset_index(drop=True)
    
    # INDICATORS
    df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['hma'] = calculate_hma(df['close'], hma_l)
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain/loss)))
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_l).mean()
    
    # MONEY FLOW
    rsi_source = df['rsi'] - 50
    vol_sma = df['volume'].rolling(mf_l).mean()
    mf_volume = df['volume'] / vol_sma
    df['money_flow'] = (rsi_source * mf_volume).ewm(span=3).mean() 
    
    pc = df['close'].diff()
    ds_pc = pc.ewm(span=25).mean().ewm(span=13).mean()
    ds_abs_pc = abs(pc).ewm(span=25).mean().ewm(span=13).mean()
    df['hyper_wave'] = (100 * (ds_pc / ds_abs_pc)) / 2
    
    mf_std = df['money_flow'].rolling(20).std()
    mf_sma = df['money_flow'].rolling(20).mean()
    df['mf_upper'] = mf_sma + (mf_std * 2.0)
    df['mf_lower'] = mf_sma - (mf_std * 2.0)
    df['adx'] = 25.0 
    
    # ADVANCED VOLUME
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.fillna(0)
    mf_vol = mfm * df['volume']
    df['cmf'] = mf_vol.rolling(20).sum() / df['volume'].rolling(20).sum()
    
    v_delta = df['volume'].diff()
    v_gain = (v_delta.where(v_delta > 0, 0)).rolling(14).mean()
    v_loss = (-v_delta.where(v_delta < 0, 0)).rolling(14).mean()
    df['vrsi'] = 100 - (100 / (1 + (v_gain/v_loss)))
    
    v_short = df['volume'].rolling(14).mean()
    v_long = df['volume'].rolling(28).mean()
    df['vol_osc'] = 100 * (v_short - v_long) / v_long
    
    body_size = (df['close'] - df['open']).abs()
    range_size = df['high'] - df['low']
    is_doji = body_size <= (range_size * 0.1)
    df['hidden_liq'] = is_doji & (df['rvol'] > 2.0)

    # --- TITAN ENGINE ---
    df['ll'] = df['low'].rolling(amp).min()
    df['hh'] = df['high'].rolling(amp).max()
    
    trend = np.zeros(len(df)); stop = np.full(len(df), np.nan)
    curr_t = 0; curr_s = np.nan
    for i in range(amp, len(df)):
        c = df.at[i,'close']; d = df.at[i,'atr']*dev
        if curr_t == 0:
            s = df.at[i,'ll'] + d
            curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
            if c < curr_s: curr_t = 1; curr_s = df.at[i,'hh'] - d
        else:
            s = df.at[i,'hh'] - d
            curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
            if c > curr_s: curr_t = 0; curr_s = df.at[i,'ll'] + d
        trend[i] = curr_t; stop[i] = curr_s
    
    df['is_bull'] = trend == 0
    df['entry_stop'] = stop
    
    cond_buy = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False)) & (df['rvol']>RVOL_THRESHOLD) & (df['rsi']<70)
    cond_sell = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True)) & (df['rvol']>RVOL_THRESHOLD) & (df['rsi']>30)
    if hma_on:
        cond_buy &= (df['close'] > df['hma'])
        cond_sell &= (df['close'] < df['hma'])
    
    df['buy'] = cond_buy; df['sell'] = cond_sell
    
    # LADDER
    df['sig_id'] = (df['buy']|df['sell']).cumsum()
    df['entry'] = np.where(df['buy']|df['sell'], df['close'], np.nan)
    df['entry'] = df.groupby('sig_id')['entry'].ffill()
    df['stop_val'] = df.groupby('sig_id')['entry_stop'].ffill()
    risk = abs(df['entry'] - df['stop_val'])
    df['tp1'] = np.where(df['is_bull'], df['entry']+(risk*tp1), df['entry']-(risk*tp1))
    df['tp2'] = np.where(df['is_bull'], df['entry']+(risk*tp2), df['entry']-(risk*tp2))
    df['tp3'] = np.where(df['is_bull'], df['entry']+(risk*tp3), df['entry']-(risk*tp3))

    # --- APEX ENGINE ---
    apex_base = calculate_hma(df['close'], 55)
    apex_atr = df['atr'] * 1.5
    df['apex_upper'] = apex_base + apex_atr
    df['apex_lower'] = apex_base - apex_atr
    
    apex_t = np.zeros(len(df))
    for i in range(1, len(df)):
        if df.at[i, 'close'] > df.at[i, 'apex_upper']: apex_t[i] = 1
        elif df.at[i, 'close'] < df.at[i, 'apex_lower']: apex_t[i] = -1
        else: apex_t[i] = apex_t[i-1]
    df['apex_trend'] = apex_t

    df['pivot_high'] = df['high'].rolling(10*2+1, center=True).max()
    df['pivot_low'] = df['low'].rolling(10*2+1, center=True).min()
    df['is_res'] = (df['high'] == df['pivot_high'])
    df['is_sup'] = (df['low'] == df['pivot_low'])

    # --- GANN HILO ENGINE ---
    sma_high = df['high'].rolling(gann_l).mean()
    sma_low = df['low'].rolling(gann_l).mean()
    
    g_trend = np.zeros(len(df))
    g_act = np.zeros(len(df))
    curr_g_t = 1
    curr_g_a = sma_low.iloc[gann_l] if len(sma_low) > gann_l else 0
    
    for i in range(gann_l, len(df)):
        c = df.at[i,'close']
        h_ma = sma_high.iloc[i]
        l_ma = sma_low.iloc[i]
        prev_a = g_act[i-1] if i > 0 else curr_g_a
        
        if curr_g_t == 1:
            if c < prev_a:
                curr_g_t = -1
                curr_g_a = h_ma
            else:
                curr_g_a = l_ma
        else:
            if c > prev_a:
                curr_g_t = 1
                curr_g_a = l_ma
            else:
                curr_g_a = h_ma
                
        g_trend[i] = curr_g_t
        g_act[i] = curr_g_a
        
    df['gann_trend'] = g_trend
    df['gann_act'] = g_act
    
    return df

# =============================================================================
# MAIN UI
# =============================================================================
with st.spinner("Initializing Terminal..."):
    df = get_klines(symbol, timeframe, limit)

if not df.empty:
    df = df.dropna(subset=['close'])
    with st.spinner("Processing Algorithms..."):
        df = run_engines(df, int(amplitude), channel_dev, int(hma_len), use_hma_filter, tp1_r, tp2_r, tp3_r, int(mf_len), int(vol_len), int(gann_len))
    
    last = df.iloc[-1]
    fibs = calculate_fibonacci(df)
    fg_index = calculate_fear_greed_index(df)
    ai_report = generate_ai_analysis(last, symbol, timeframe, fibs, fg_index)
    
    # --- METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PRICE", f"{last['close']:.2f}", f"{'BULL' if last['is_bull'] else 'BEAR'}")
    m2.metric("GANN TREND", "BULL" if last['gann_trend']==1 else "BEAR")
    m3.metric("STOP LOSS", f"{last['entry_stop']:.2f}")
    m4.metric("TP3 (5R)", f"{last['tp3']:.2f}")

    # --- ACTION CENTER ---
    c_act1, c_act2, c_act3 = st.columns(3)
    signal_txt = f"🔥 *TITAN SIGNAL: {symbol}*\n⏰ TF: {timeframe} | 🧭 Dir: *{'LONG 🟢' if last['is_bull'] else 'SHORT 🔴'}*\n📍 Entry: `{last['close']:.2f}`\n🛑 Stop: `{last['entry_stop']:.2f}`\n🎯 *LADDER:*\n1️⃣ TP1: `{last['tp1']:.2f}`\n2️⃣ TP2: `{last['tp2']:.2f}`\n3️⃣ TP3: `{last['tp3']:.2f}`\n\n{ai_report}\n⚠️ _NFA_"

    with c_act1:
        if st.button("🔥 BROADCAST SIGNAL", use_container_width=True):
            if tg_token and tg_chat:
                if send_telegram_msg(tg_token, tg_chat, signal_txt, 0): st.success("SENT!")
                else: st.error("FAILED")
            else: st.error("NO CREDS (Check secrets.toml or Inputs)")
    
    with c_act2: st.download_button("📥 DOWNLOAD CSV", df.to_csv(), "titan.csv", "text/csv", use_container_width=True)
    with c_act3: st.info("Backtest results placeholder")

    # --- MAIN CHART (TITAN) ---
    st.markdown("### 🏹 EXECUTION")
    fig = go.Figure()
    fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', name='HMA', line=dict(color='#66fcf1', width=1)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['entry_stop'], mode='lines', name='Stop', line=dict(color='#ff9900', width=1)))
    
    buys = df[df['buy']]; sells = df[df['sell']]
    if not buys.empty: fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00'), name='BUY'))
    if not sells.empty: fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000'), name='SELL'))
    
    fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, yaxis=dict(autorange=True))
    st.plotly_chart(fig, use_container_width=True)

    # --- TABBED ANALYSIS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 GANN HILO", "🌊 APEX", "💸 MATRIX", "📉 VOL", "🧠 SENTIMENT"])
    
    with tab1:
        fig6 = go.Figure()
        fig6.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
        
        # Color segment the Gann line
        for i in range(1, len(df)):
            color = '#00ff00' if df['gann_trend'].iloc[i] == 1 else '#ff0000'
            fig6.add_trace(go.Scatter(
                x=df['timestamp'].iloc[i-1:i+1], y=df['gann_act'].iloc[i-1:i+1],
                mode='lines', line=dict(color=color, width=2), showlegend=False
            ))
            
        fig6.update_layout(height=500, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), title="Gann High Low Activator")
        st.plotly_chart(fig6, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['apex_upper'], mode='lines', line=dict(width=0), showlegend=False))
        fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['apex_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(102, 252, 241, 0.1)', name='Cloud'))
        fig2.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
        
        for i, r in df[df['is_sup']].iloc[-5:].iterrows():
            fig2.add_shape(type="rect", x0=r['timestamp'], y0=r['low'], x1=df['timestamp'].iloc[-1], y1=r['low']-(r['atr']*0.5), fillcolor="rgba(0,255,0,0.2)", line_width=0)
        for i, r in df[df['is_res']].iloc[-5:].iterrows():
            fig2.add_shape(type="rect", x0=r['timestamp'], y0=r['high'], x1=df['timestamp'].iloc[-1], y1=r['high']+(r['atr']*0.5), fillcolor="rgba(255,0,0,0.2)", line_width=0)
        fig2.update_layout(height=500, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig4 = go.Figure()
        colors = ['#00e676' if x > 0 else '#ff1744' for x in df['money_flow']]
        fig4.add_trace(go.Bar(x=df['timestamp'], y=df['money_flow'], marker_color=colors, name='Flow'))
        fig4.add_trace(go.Scatter(x=df['timestamp'], y=df['hyper_wave'], mode='lines', name='Hyper Wave', line=dict(color='yellow', width=2)))
        fig4.update_layout(height=500, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        fig5 = go.Figure()
        vol_d = df['cmf'] if hero_metric == "CMF" else df['vrsi'] if hero_metric == "Volume RSI" else df['vol_osc'] if hero_metric == "Volume Oscillator" else df['rvol']
        fig5.add_trace(go.Scatter(x=df['timestamp'], y=vol_d, mode='lines', name=hero_metric, fill='tozeroy', line=dict(color='#b388ff')))
        traps = df[df['hidden_liq']]
        if not traps.empty: fig5.add_trace(go.Scatter(x=traps['timestamp'], y=vol_d[df['hidden_liq']], mode='markers', marker=dict(symbol='diamond', size=10, color='yellow'), name='Trap'))
        fig5.update_layout(height=500, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig5, use_container_width=True)

    with tab5:
        fig3 = go.Figure(go.Indicator(
            mode = "gauge+number", value = fg_index, title = {'text': "MARKET SENTIMENT"},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "white"}, 'steps': [{'range': [0, 25], 'color': '#ff1744'}, {'range': [75, 100], 'color': '#00b0ff'}]}
        ))
        fig3.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)

    # --- FOOTER ---
    st.markdown("---")
    components.html(f"""<div id="tv" style="height:500px;border-radius:10px;overflow:hidden;"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"autosize":true,"symbol":"BINANCE:{symbol}","interval":"{'D' if timeframe=='1d' else '60'}","timezone":"Etc/UTC","theme":"dark","style":"1","container_id":"tv","hide_side_toolbar":false}});</script>""", height=500)
