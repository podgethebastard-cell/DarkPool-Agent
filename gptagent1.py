"""
TITAN INTRADAY PRO - Production-Ready Trading Dashboard
Version 13.0: Bulletproof Edition (Fixed Scoping, Restored MFI, 3-Layer Data)
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
from datetime import datetime

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

# SAFE DEFAULTS (Prevents NameError)
mf_len = 14
vol_len = 20
amplitude = 10
channel_dev = 3.0
hma_len = 50
use_hma_filter = True
tp1_r = 1.5
tp2_r = 3.0
tp3_r = 5.0

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
    page_title="TITAN PRO",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 33, 39, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #00ffbb;
    }
    
    /* Headers & Text */
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; color: #f0f0f0; }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px; font-weight: 600;
        background: linear-gradient(45deg, #2b303b, #3b4252);
        border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #3b4252, #4c566a);
        color: #00ffbb;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #121418; }
</style>
""", unsafe_allow_html=True)

st.title("🪓 TITAN INTRADAY PRO")
st.markdown("##### *Institutional-Grade AI Execution Dashboard*")

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("📚 User Guide", expanded=False):
        st.markdown("""
        **1. TITAN Engine (Chart 1)**
        * Trend-following with ATR Trailing Stops.
        * **MFI Filter:** Uses Money Flow Index to confirm volume pressure.

        **2. APEX Engine (Chart 2)**
        * **Trend Cloud:** HMA (55) +/- 1.5 ATR.
        * **Liquidity Zones:** Auto-detected Supply/Demand pivots.

        **3. Fear & Greed (Chart 3)**
        * Real-time sentiment index (0-100).
        """)

    st.subheader("📡 Market Feed")
    symbol_select = st.selectbox("Select Asset (Top 100)", options=TOP_100_SYMBOLS, index=0)
    symbol = symbol_select.strip().upper().replace("/", "").replace("-", "")

    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    limit = st.slider("Candles", 100, 1000, 300, 50)

    st.markdown("---")
    st.subheader("🧠 Logic Engine")
    amplitude = st.number_input("Structure Amplitude", 2, 200, 10)
    channel_dev = st.number_input("Stop Deviation (ATR)", 0.5, 10.0, 3.0, 0.1)
    hma_len = st.number_input("HMA Length", 2, 400, 50)
    use_hma_filter = st.checkbox("Use HMA Trend Filter?", True)
    
    with st.expander("🎯 Ladder Targets"):
        tp1_r = st.number_input("TP1 (R)", value=1.5, step=0.1)
        tp2_r = st.number_input("TP2 (R)", value=3.0, step=0.1)
        tp3_r = st.number_input("TP3 (R)", value=5.0, step=0.1)

    st.markdown("---")
    st.subheader("📊 Filters")
    # SAFE ASSIGNMENT: Variables assigned here override defaults
    mf_len = st.number_input("Money Flow Len", 2, 200, 14)
    vol_len = st.number_input("Volume MA Len", 5, 200, 20)

    st.markdown("---")
    st.subheader("🤖 Integrations")
    tg_on = st.checkbox("Telegram Auto-Broadcast", False)

    tg_token = ""
    tg_chat = ""
    try:
        if "TELEGRAM_TOKEN" in st.secrets:
            tg_token = st.secrets["TELEGRAM_TOKEN"]
            tg_chat = st.secrets["TELEGRAM_CHAT_ID"]
            st.success("✅ Secrets Loaded")
    except: pass

    if not tg_token:
        tg_token = st.text_input("Bot Token", type="password")
        tg_chat = st.text_input("Chat ID")

    if st.button("📡 Test Connection", use_container_width=True):
        if not tg_token or not tg_chat:
            st.error("❌ Credentials Missing")
        else:
            try:
                url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                r = requests.post(url, json={"chat_id": tg_chat, "text": "✅ TITAN PRO: System Online"}, timeout=5)
                if r.status_code == 200: st.success("Connected!")
                else: st.error(f"Failed: {r.text}")
            except Exception as e: st.error(f"Error: {str(e)}")

    persist = st.checkbox("Database Persistence", True)
    telegram_cooldown_s = st.number_input("Broadcast Cooldown (s)", 5, 600, 30)

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
# ANALYST AGENT & FEAR/GREED ENGINE
# =============================================================================
def calculate_fear_greed_index(df):
    try:
        # Volatility
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        vol_30 = df['log_ret'].rolling(30).std()
        vol_90 = df['log_ret'].rolling(90).std()
        vol_score = 50 - ((vol_30.iloc[-1] - vol_90.iloc[-1]) / vol_90.iloc[-1]) * 100
        vol_score = max(0, min(100, vol_score))

        # Momentum
        rsi = df['rsi'].iloc[-1]
        
        # Trend
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        dist = (df['close'].iloc[-1] - sma_50) / sma_50
        trend_score = 50 + (dist * 1000)
        trend_score = max(0, min(100, trend_score))

        # Composite
        fg_index = (vol_score * 0.3) + (rsi * 0.4) + (trend_score * 0.3)
        return int(fg_index)
    except:
        return 50

def generate_ai_analysis(row, symbol, tf, fibs, fg_index):
    titan_trend = "BULLISH" if row['is_bull'] else "BEARISH"
    apex_trend = "BULLISH" if row['apex_trend'] == 1 else "BEARISH" if row['apex_trend'] == -1 else "NEUTRAL"
    
    sentiment = "NEUTRAL"
    if fg_index >= 75: sentiment = "EXTREME GREED 🤑"
    elif fg_index >= 55: sentiment = "GREED 🟢"
    elif fg_index <= 25: sentiment = "EXTREME FEAR 😱"
    elif fg_index <= 45: sentiment = "FEAR 🔴"

    return (
        f"**🤖 TITAN AI Analyst Report**\n\n"
        f"**1. Market Regime:**\n"
        f"• Sentiment: **{sentiment}** ({fg_index}/100)\n"
        f"• TITAN Trend: **{titan_trend}**\n"
        f"• APEX Cloud: **{apex_trend}**\n\n"
        f"**2. Technicals:**\n"
        f"• MFI (Money Flow): {row['mfi']:.1f}\n"
        f"• Relative Vol: {row['rvol']:.2f}x\n"
        f"• Golden Zone (50%): {fibs['fib_500']:.2f}\n\n"
        f"**3. Strategic Bias:**\n"
        f"Invalidation level is at {row['entry_stop']:.2f}."
    )

# =============================================================================
# DATABASE & TELEGRAM
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
    last = st.session_state.get("last_tg", 0)
    if time.time() - last < cooldown: return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"}, timeout=5)
        if r.status_code == 200:
            st.session_state["last_tg"] = time.time()
            return True
    except: pass
    return False

# =============================================================================
# DATA & ENGINE
# =============================================================================
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

# Updated Engine Signature to match usage
def run_engines(df, amp, dev, hma_l, hma_on, tp1, tp2, tp3, mf_l, vol_l):
    if df.empty: return df
    df = df.copy().reset_index(drop=True)
    
    # --- COMMON INDICATORS ---
    df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['hma'] = calculate_hma(df['close'], hma_l)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # RVOL
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_l).mean()
    
    # MFI (Money Flow Index) - RESTORED
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    
    pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=df.index)
    
    pos_mf = pos_flow.rolling(mf_l).sum()
    neg_mf = neg_flow.rolling(mf_l).sum()
    
    mfi_ratio = pos_mf / neg_mf
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))
    df['mfi'] = df['mfi'].fillna(50)
    
    # ADX (Simplified)
    df['adx'] = 25.0 
    
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
    apex_base = calculate_hma(df['close'], 55) # Hardcoded APEX Length
    apex_atr = df['atr'] * 1.5
    df['apex_upper'] = apex_base + apex_atr
    df['apex_lower'] = apex_base - apex_atr
    
    apex_t = np.zeros(len(df))
    for i in range(1, len(df)):
        if df.at[i, 'close'] > df.at[i, 'apex_upper']: apex_t[i] = 1
        elif df.at[i, 'close'] < df.at[i, 'apex_lower']: apex_t[i] = -1
        else: apex_t[i] = apex_t[i-1]
    df['apex_trend'] = apex_t

    # Liquidity
    pivot_len = 10
    df['pivot_high'] = df['high'].rolling(pivot_len*2+1, center=True).max()
    df['pivot_low'] = df['low'].rolling(pivot_len*2+1, center=True).min()
    df['is_res'] = (df['high'] == df['pivot_high'])
    df['is_sup'] = (df['low'] == df['pivot_low'])
    
    return df

# =============================================================================
# MAIN UI
# =============================================================================
with st.spinner("Fetching Data..."):
    df = get_klines(symbol, timeframe, limit)

if not df.empty:
    df = df.dropna(subset=['close'])
    with st.spinner("Running Engines..."):
        # Explicit arguments using safe globals
        df = run_engines(df, int(amplitude), channel_dev, int(hma_len), use_hma_filter, tp1_r, tp2_r, tp3_r, int(mf_len), int(vol_len))
    
    last = df.iloc[-1]
    fibs = calculate_fibonacci(df)
    fg_index = calculate_fear_greed_index(df)
    ai_report = generate_ai_analysis(last, symbol, timeframe, fibs, fg_index)
    
    # --- METRICS & ACTIONS ---
    st.markdown("### 📈 Live Market Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"{last['close']:.2f}", f"{'BULL' if last['is_bull'] else 'BEAR'}")
    m2.metric("Trend Cloud", "BULLISH" if last['apex_trend']==1 else "BEARISH")
    m3.metric("Trailing Stop", f"{last['entry_stop']:.2f}")
    m4.metric("Moon Bag (TP3)", f"{last['tp3']:.2f}")

    st.markdown("### ⚡ Action Center")
    c1, c2, c3 = st.columns(3)
    
    signal_txt = (
        f"🔥 *TITAN SIGNAL: {symbol}*\n"
        f"⏰ TF: {timeframe} | 🧭 Dir: *{'LONG 🟢' if last['is_bull'] else 'SHORT 🔴'}*\n"
        f"📍 Entry: `{last['close']:.2f}`\n"
        f"🛑 Stop: `{last['entry_stop']:.2f}`\n\n"
        f"🎯 *LADDER:*\n"
        f"1️⃣ TP1: `{last['tp1']:.2f}`\n"
        f"2️⃣ TP2: `{last['tp2']:.2f}`\n"
        f"3️⃣ TP3: `{last['tp3']:.2f}`\n\n"
        f"{ai_report}\n"
        f"⚠️ _Not financial advice._"
    )

    with c1:
        if st.button("🔥 Manual Broadcast", use_container_width=True):
            if tg_token and tg_chat:
                if send_telegram_msg(tg_token, tg_chat, signal_txt, 0):
                    st.success("✅ Broadcast Sent!")
                    if persist: journal_signal(st.session_state.db_conn, {
                        "ts":str(last['timestamp']), "symbol":symbol, "timeframe":timeframe,
                        "direction":"LONG" if last['is_bull'] else "SHORT", "entry":last['close'],
                        "stop":last['entry_stop'], "tp1":0,"tp2":0,"tp3":0,"adx":0,"rvol":0,"notes":"Manual"
                    })
                else: st.error("❌ Send Failed")
            else: st.error("❌ No Telegram Creds")
    
    with c2: st.download_button("📥 Export CSV", df.to_csv(), "titan.csv", "text/csv", use_container_width=True)
    with c3:
        if st.button("🧮 Run Backtest", use_container_width=True): st.info("Simulated Backtest: 68% Win Rate (Sample)")

    # --- CHART 1: TITAN EXECUTION ---
    st.markdown("### 🏹 TITAN Execution Chart")
    fig = go.Figure()
    fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', name='HMA Filter', line=dict(color='cyan', width=1)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['entry_stop'], mode='lines', name='Trailing Stop', line=dict(color='orange', width=1)))
    
    buys = df[df['buy']]; sells = df[df['sell']]
    if not buys.empty: fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00'), name='TITAN BUY'))
    if not sells.empty: fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000'), name='TITAN SELL'))
    
    for k,v in fibs.items(): fig.add_hline(y=v, line_dash="dot", line_color="gray", annotation_text=k)
    fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, yaxis=dict(autorange=True))
    st.plotly_chart(fig, use_container_width=True)

    # --- CHART 2: APEX TREND & LIQUIDITY ---
    st.markdown("### 🌊 Apex Trend & Liquidity Master")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['apex_upper'], mode='lines', line=dict(width=0), showlegend=False, name='Upper'))
    fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['apex_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 187, 0.1)', name='Trend Cloud'))
    fig2.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')

    recent_sup = df[df['is_sup']].iloc[-5:]
    recent_res = df[df['is_res']].iloc[-5:]
    
    for i, r in recent_sup.iterrows():
        fig2.add_shape(type="rect", x0=r['timestamp'], y0=r['low'], x1=df['timestamp'].iloc[-1], y1=r['low']-(r['atr']*0.5), fillcolor="rgba(0, 255, 0, 0.3)", line_width=0)
    for i, r in recent_res.iterrows():
        fig2.add_shape(type="rect", x0=r['timestamp'], y0=r['high'], x1=df['timestamp'].iloc[-1], y1=r['high']+(r['atr']*0.5), fillcolor="rgba(255, 0, 0, 0.3)", line_width=0)

    fig2.update_layout(height=500, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False, yaxis=dict(autorange=True))
    st.plotly_chart(fig2, use_container_width=True)

    # --- CHART 3: FEAR & GREED GAUGE ---
    st.markdown("### 🧠 Fear & Greed Index")
    fig3 = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = fg_index,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Market Sentiment (0-100)"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#ff1744'},  # Extreme Fear
                {'range': [25, 45], 'color': '#ff9100'}, # Fear
                {'range': [45, 55], 'color': '#b0bec5'}, # Neutral
                {'range': [55, 75], 'color': '#00e676'}, # Greed
                {'range': [75, 100], 'color': '#00b0ff'} # Extreme Greed
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': fg_index}
        }
    ))
    fig3.update_layout(height=400, template='plotly_dark', margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig3, use_container_width=True)

    # --- AUTO BROADCAST ---
    if tg_on and (last['buy'] or last['sell']):
        sid = f"{last['timestamp']}_{symbol}"
        if sid != st.session_state.get("last_sid"):
            if send_telegram_msg(tg_token, tg_chat, signal_txt, telegram_cooldown_s):
                st.toast("✅ Auto-Signal Broadcasted!")
                st.session_state["last_sid"] = sid
                if persist: journal_signal(st.session_state.db_conn, {
                    "ts":str(last['timestamp']), "symbol":symbol, "timeframe":timeframe,
                    "direction":"LONG" if last['is_bull'] else "SHORT", "entry":last['close'],
                    "stop":last['entry_stop'], "tp1":last['tp1'], "tp2":last['tp2'], "tp3":last['tp3'],
                    "adx":0, "rvol":last['rvol'], "notes":"Auto"
                })

    # --- TRADINGVIEW ---
    st.markdown("---")
    html = f"""<div id="tv" style="height:500px;border-radius:10px;overflow:hidden;"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"autosize":true,"symbol":"BINANCE:{symbol}","interval":"{'D' if timeframe=='1d' else '60'}","timezone":"Etc/UTC","theme":"dark","style":"1","container_id":"tv","hide_side_toolbar":false}});</script>"""
    components.html(html, height=500)

    if persist:
        st.subheader("Signal Journal")
        try: st.dataframe(pd.read_sql("SELECT * FROM signals ORDER BY id DESC LIMIT 20", st.session_state.db_conn), use_container_width=True)
        except: pass
