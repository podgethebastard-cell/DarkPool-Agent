"""
TITAN INTRADAY PRO - Production-Ready Trading Dashboard
Version 3.2: Triple-Layer Data Engine (Binance -> Bybit -> Coinbase)
"""
import time
import math
import sqlite3
import atexit
from typing import Dict, Optional
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
MAX_RETRIES = 2  # Reduced for faster failover
RETRY_DELAY = 0.5

# API ENDPOINTS
BINANCE_API_BASE = "https://api.binance.us/api/v3"
BYBIT_API_BASE = "https://api.bybit.com/v5/market/kline"
COINBASE_API_BASE = "https://api.exchange.coinbase.com/products"

# HEADERS (Critical to bypass "Expecting Value" WAF blocks)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="TITAN INTRADAY PRO",
    layout="wide",
    page_icon="⚡"
)
st.title("🪓 TITAN INTRADAY PRO — Execution Dashboard")

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
st.sidebar.header("Market Feed")
symbol_input = st.sidebar.text_input("Symbol", value="BTCUSDT")
# Clean symbol for different APIs
symbol_clean = symbol_input.strip().upper().replace("/", "").replace("-", "")
symbol_coinbase = f"{symbol_clean[:-4]}-{symbol_clean[-4:]}" if len(symbol_clean) > 4 else "BTC-USD"

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "5m", "15m", "1h", "4h", "1d"],
    index=3
)
limit = st.sidebar.slider("Candles", 100, 1000, 200, 50)

st.sidebar.markdown("---")
st.sidebar.header("Logic Engine / Ladder")
amplitude = st.sidebar.number_input("Structure lookback", 2, 200, 10)
channel_dev = st.sidebar.number_input("Stop Deviation (ATR x)", 0.5, 10.0, 3.0, 0.1)
hma_len = st.sidebar.number_input("HMA length", 2, 400, 50)
use_hma_filter = st.sidebar.checkbox("Use HMA filter?", True)
tp1_r = st.sidebar.number_input("TP1 (R)", value=1.5, step=0.1)
tp2_r = st.sidebar.number_input("TP2 (R)", value=3.0, step=0.1)
tp3_r = st.sidebar.number_input("TP3 (R)", value=5.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Volume & Momentum")
mf_len = st.sidebar.number_input("Money Flow length", 2, 200, 14)
vol_len = st.sidebar.number_input("Volume rolling length", 5, 200, 20)

st.sidebar.markdown("---")
st.sidebar.header("Integrations")
tg_on = st.sidebar.checkbox("Telegram Broadcast", False)
try:
    tg_token = st.secrets["TELEGRAM_TOKEN"]
    tg_chat = st.secrets["TELEGRAM_CHAT_ID"]
    st.sidebar.success("Telegram secrets loaded")
except:
    tg_token = st.sidebar.text_input("Bot Token", type="password")
    tg_chat = st.sidebar.text_input("Chat ID")

persist = st.sidebar.checkbox("Persist signals to DB", True)
run_backtest = st.sidebar.checkbox("Run backtest", False)
backtest_risk = st.sidebar.number_input("Risk %", 0.1, 5.0, 1.0, 0.1)
telegram_cooldown_s = st.sidebar.number_input("TG Cooldown", 5, 600, 30)

# =============================================================================
# DATABASE
# =============================================================================
@contextmanager
def get_db_connection(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    try:
        yield conn
    finally:
        conn.close()

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

if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_db()

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
    except Exception: return False

# =============================================================================
# TELEGRAM
# =============================================================================
def send_telegram_msg(token, chat, msg, cooldown):
    if not token or not chat: return False
    last = st.session_state.get("last_tg", 0)
    if time.time() - last < cooldown: return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    safe_msg = msg.replace('.', '\\.').replace('-', '\\-').replace('!', '\\!')
    try:
        r = requests.post(url, json={"chat_id": chat, "text": safe_msg, "parse_mode": "Markdown"}, timeout=5)
        if r.status_code == 200:
            st.session_state["last_tg"] = time.time()
            return True
    except: pass
    return False

# =============================================================================
# TRIPLE-LAYER DATA ENGINE
# =============================================================================
@st.cache_data(ttl=5)
def get_klines(symbol_bin: str, interval: str, limit: int) -> pd.DataFrame:
    
    # 1. Try Binance US (Real-time, Standard)
    try:
        url = f"{BINANCE_API_BASE}/klines"
        r = requests.get(url, params={"symbol": symbol_bin, "interval": interval, "limit": limit}, headers=HEADERS, timeout=4)
        if r.status_code == 200:
            data = r.json()
            if data:
                df = pd.DataFrame(data, columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
                return df[['timestamp','open','high','low','close','volume']]
    except Exception: pass

    # 2. Try Bybit V5 (Real-time, Robust)
    try:
        imap = {"1m":"1", "5m":"5", "15m":"15", "1h":"60", "4h":"240", "1d":"D"}
        r = requests.get(BYBIT_API_BASE, 
                         params={"category":"spot", "symbol":symbol_bin, "interval":imap.get(interval,"60"), "limit":limit},
                         headers=HEADERS, timeout=4)
        data = r.json()
        if data.get('retCode') == 0:
            # Bybit returns reversed list [time, open, high, low, close, vol, turnover]
            df = pd.DataFrame(data['result']['list'], columns=['t','o','h','l','c','v','to'])
            df['timestamp'] = pd.to_datetime(df['t'].astype(float), unit='ms')
            df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
            df = df.iloc[::-1].reset_index(drop=True) # Reverse back to chronological
            return df[['timestamp','open','high','low','close','volume']]
    except Exception as e:
        st.warning(f"Bybit Failed: {str(e)}")

    # 3. Try Coinbase (The "Nuclear" Option)
    # Very permissive, rarely blocked.
    try:
        # Map timeframe to seconds
        gmap = {"1m":60, "5m":300, "15m":900, "1h":3600, "4h":21600, "1d":86400} # 4h mapped to 6h on CB
        granularity = gmap.get(interval, 3600)
        
        # Coinbase uses BTC-USD, not BTCUSDT
        sym_cb = f"{symbol_bin[:-4]}-{symbol_bin[-4:]}" if symbol_bin.endswith("USDT") else "BTC-USD"
        
        url = f"{COINBASE_API_BASE}/{sym_cb}/candles"
        r = requests.get(url, params={"granularity": granularity}, headers=HEADERS, timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            # Coinbase: [time, low, high, open, close, volume]
            df = pd.DataFrame(data, columns=['t','l','h','o','c','v'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
            df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df[['timestamp','open','high','low','close','volume']].iloc[-limit:]
    except Exception as e:
        st.error(f"All feeds failed. Last error: {str(e)}")

    return pd.DataFrame()

# =============================================================================
# LOGIC ENGINE
# =============================================================================
def run_titan(df, amp, dev, hma_l, hma_on, tp1, tp2, tp3, mf_l, vol_l):
    if df.empty: return df
    df = df.copy().reset_index(drop=True)
    
    # Indicators
    df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    
    # HMA
    half, sqrt = int(hma_l/2), int(math.sqrt(hma_l))
    wma_f = df['close'].rolling(hma_l).mean()
    wma_h = df['close'].rolling(half).mean()
    df['hma'] = (2*wma_h - wma_f).rolling(sqrt).mean()
    
    # Structure
    df['ll'] = df['low'].rolling(amp).min()
    df['hh'] = df['high'].rolling(amp).max()
    
    # Trend Loop
    trend = np.zeros(len(df)); stop = np.zeros(len(df)); curr_t = 0; curr_s = np.nan
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
    df['stop'] = stop
    
    # Signals
    df['rsi'] = 100 - (100/(1 + (df['close'].diff().clip(lower=0).ewm(alpha=1/14).mean()/(-df['close'].diff().clip(upper=0)).ewm(alpha=1/14).mean())))
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_l).mean()
    df['adx'] = 25.0 # Simplified for perf
    
    cond_buy = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False)) & (df['rvol']>RVOL_THRESHOLD) & (df['rsi']<70)
    cond_sell = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True)) & (df['rvol']>RVOL_THRESHOLD) & (df['rsi']>30)
    
    if hma_on:
        cond_buy &= (df['close'] > df['hma'])
        cond_sell &= (df['close'] < df['hma'])
        
    df['buy'] = cond_buy
    df['sell'] = cond_sell
    
    # Lock Stats
    df['sig_id'] = (df['buy']|df['sell']).cumsum()
    df['entry'] = np.where(df['buy']|df['sell'], df['close'], np.nan)
    df['entry'] = df.groupby('sig_id')['entry'].ffill()
    df['entry_stop'] = np.where(df['buy']|df['sell'], df['stop'], np.nan)
    df['entry_stop'] = df.groupby('sig_id')['entry_stop'].ffill()
    
    risk = abs(df['entry'] - df['entry_stop'])
    df['tp1'] = np.where(df['is_bull'], df['entry']+risk*tp1, df['entry']-risk*tp1)
    df['tp2'] = np.where(df['is_bull'], df['entry']+risk*tp2, df['entry']-risk*tp2)
    df['tp3'] = np.where(df['is_bull'], df['entry']+risk*tp3, df['entry']-risk*tp3)
    
    return df

# =============================================================================
# MAIN UI
# =============================================================================
with st.spinner("Fetching Data (Failover System Active)..."):
    df = get_klines(symbol, timeframe, limit)

if not df.empty:
    with st.spinner("Calculating TITAN Logic..."):
        df = run_titan(df, int(amplitude), channel_dev, int(hma_len), use_hma_filter, tp1_r, tp2_r, tp3_r, mf_len, vol_len)
    
    last = df.iloc[-1]
    
    # Top Chart
    fig = go.Figure()
    fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stop'], mode='lines', name='Stop', line=dict(color='orange', width=1)))
    
    buys = df[df['buy']]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low'], mode='markers', marker=dict(symbol='triangle-up', size=15, color='green'), name='BUY'))
    
    sells = df[df['sell']]
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high'], mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='SELL'))
        
    fig.update_layout(height=600, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{last['close']:.2f}", f"{'BULL' if last['is_bull'] else 'BEAR'}")
    c2.metric("Entry", f"{last['entry']:.2f}")
    c3.metric("Stop", f"{last['entry_stop']:.2f}")
    c4.metric("TP3", f"{last['tp3']:.2f}")
    
    # Broadcast Logic
    if tg_on and (last['buy'] or last['sell']):
        sid = f"{last['timestamp']}_{symbol}"
        if sid != st.session_state.get("last_sid"):
            txt = f"🔥 TITAN: {symbol} {'LONG' if last['is_bull'] else 'SHORT'} @ {last['close']}"
            if send_telegram_msg(tg_token, tg_chat, txt, telegram_cooldown_s):
                st.success("Auto-Broadcast Sent!")
                st.session_state["last_sid"] = sid
                if persist: journal_signal(st.session_state.db_conn, {
                    "ts":str(last['timestamp']), "symbol":symbol, "timeframe":timeframe,
                    "direction":"LONG" if last['is_bull'] else "SHORT", "entry":last['close'],
                    "stop":last['entry_stop'], "tp1":last['tp1'], "tp2":last['tp2'], "tp3":last['tp3'],
                    "adx":0, "rvol":last['rvol'], "notes":"Auto"
                })

    st.markdown("---")
    
    # TradingView
    html = f"""
    <div id="tv" style="height:500px"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.widget({{
        "autosize": true, "symbol": "BINANCE:{symbol}", "interval": "{'D' if timeframe=='1d' else '60'}",
        "timezone": "Etc/UTC", "theme": "dark", "style": "1", "container_id": "tv", "hide_side_toolbar": false
    }});
    </script>
    """
    components.html(html, height=500)

    # Journal
    if persist:
        st.subheader("Signal Journal")
        try:
            st.dataframe(pd.read_sql("SELECT * FROM signals ORDER BY id DESC LIMIT 20", st.session_state.db_conn), use_container_width=True)
        except: pass
