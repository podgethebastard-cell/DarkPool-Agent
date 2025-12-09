"""
TITAN INTRADAY PRO - Production-Ready Trading Dashboard
Version 2.2: UI Fixes & API Hardcoded to Vision
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
MAX_RETRIES = 3
RETRY_DELAY = 0.5

# Hardcoded to Binance Vision as requested
BINANCE_API_BASE = "https://data-api.binance.vision/api/v3"

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
# API Region selector removed; defaulting to Vision.

symbol_input = st.sidebar.text_input("Symbol (Binance format)", value="BTCUSDT")
symbol = symbol_input.strip().upper().replace("/", "").replace("-", "")
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "5m", "15m", "1h", "4h", "1d"],
    index=3
)
limit = st.sidebar.slider(
    "Candles",
    min_value=100,
    max_value=2000,
    value=600,
    step=50
)

st.sidebar.markdown("---")
st.sidebar.header("Logic Engine / Ladder")
amplitude = st.sidebar.number_input(
    "Structure lookback (amplitude)",
    min_value=2,
    max_value=200,
    value=10
)
channel_dev = st.sidebar.number_input(
    "Stop Deviation (ATR x)",
    min_value=0.5,
    max_value=10.0,
    value=3.0,
    step=0.1
)
hma_len = st.sidebar.number_input(
    "HMA length",
    min_value=2,
    max_value=400,
    value=50
)
use_hma_filter = st.sidebar.checkbox("Use HMA filter?", value=True)
tp1_r = st.sidebar.number_input("TP1 (R)", value=1.5, step=0.1)
tp2_r = st.sidebar.number_input("TP2 (R)", value=3.0, step=0.1)
tp3_r = st.sidebar.number_input("TP3 (R)", value=5.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Volume & Momentum")
mf_len = st.sidebar.number_input(
    "Money Flow length",
    min_value=2,
    max_value=200,
    value=14
)
vol_len = st.sidebar.number_input(
    "Volume rolling length",
    min_value=5,
    max_value=200,
    value=20
)

st.sidebar.markdown("---")
st.sidebar.header("Integrations & Persistence")
tg_on = st.sidebar.checkbox("Telegram Broadcast", value=False)

# Try to load secrets, fallback to manual input
try:
    tg_token = st.secrets["TELEGRAM_TOKEN"]
    tg_chat = st.secrets["TELEGRAM_CHAT_ID"]
    st.sidebar.success("Telegram secrets loaded")
except Exception:
    tg_token = st.sidebar.text_input("Telegram Bot Token", type="password")
    tg_chat = st.sidebar.text_input("Telegram Chat ID")

persist = st.sidebar.checkbox("Persist signals to DB", value=True)
run_backtest = st.sidebar.checkbox("Run backtest (slow)", value=False)
backtest_risk = st.sidebar.number_input(
    "Backtest risk % per trade",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

st.sidebar.markdown("---")
st.sidebar.header("Safety")
ai_cooldown_s = st.sidebar.number_input(
    "AI cooldown (s)",
    min_value=5,
    max_value=600,
    value=45
)
telegram_cooldown_s = st.sidebar.number_input(
    "Telegram cooldown (s)",
    min_value=5,
    max_value=600,
    value=30
)

# =============================================================================
# DATABASE MANAGEMENT
# =============================================================================
@contextmanager
def get_db_connection(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    try:
        yield conn
    finally:
        conn.close()

def init_db(path: str = DB_PATH) -> Optional[sqlite3.Connection]:
    if not persist:
        return None
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry REAL,
            stop REAL,
            tp1 REAL,
            tp2 REAL,
            tp3 REAL,
            adx REAL,
            rvol REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts 
        ON signals(symbol, ts DESC)
    """)
    conn.commit()
    return conn

def cleanup_db():
    if 'db_conn' in st.session_state and st.session_state.db_conn:
        try:
            st.session_state.db_conn.close()
        except Exception:
            pass

if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_db()
    atexit.register(cleanup_db)

db_conn = st.session_state.db_conn

def journal_signal(conn: Optional[sqlite3.Connection], payload: Dict) -> bool:
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO signals (
                ts, symbol, timeframe, direction, entry, stop,
                tp1, tp2, tp3, adx, rvol, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload.get("ts"),
            payload.get("symbol"),
            payload.get("timeframe"),
            payload.get("direction"),
            payload.get("entry"),
            payload.get("stop"),
            payload.get("tp1"),
            payload.get("tp2"),
            payload.get("tp3"),
            payload.get("adx"),
            payload.get("rvol"),
            payload.get("notes", "")
        ))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False

# =============================================================================
# TELEGRAM
# =============================================================================
def escape_markdown_v2(text: str) -> str:
    special_chars = r'_*[]()~`>#+-=|{}.!'
    return ''.join('\\' + c if c in special_chars else c for c in text)

def send_telegram_msg(token: str, chat: str, msg: str, cooldown_s: int = 30) -> bool:
    if not token or not chat:
        return False
    last_ts = st.session_state.get("last_telegram_ts", 0)
    if time.time() - last_ts < cooldown_s:
        return False
    payload = {
        "chat_id": chat,
        "text": escape_markdown_v2(msg),
        "parse_mode": "MarkdownV2"
    }
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, json=payload, timeout=8)
            if r.status_code in (200, 201):
                st.session_state["last_telegram_ts"] = time.time()
                return True
            elif r.status_code == 429:
                retry_after = int(r.headers.get('Retry-After', RETRY_DELAY))
                time.sleep(retry_after)
            else:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception:
            time.sleep(RETRY_DELAY * (attempt + 1))
    return False

# =============================================================================
# BINANCE API
# =============================================================================
@st.cache_data(ttl=10)
def get_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    url = f"{BINANCE_API_BASE}/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data or isinstance(data, dict):
                return pd.DataFrame()
            
            cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q', 't', 'tb', 'tq', 'i']
            df = pd.DataFrame(data, columns=cols)
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                st.error(f"API Error: {str(e)}")
            time.sleep(RETRY_DELAY)
    return pd.DataFrame()

# =============================================================================
# INDICATORS & ENGINE
# =============================================================================
def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def calculate_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low = df['high'], df['low']
    up, down = high.diff(), -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = calculate_atr(df, length)
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/length, adjust=False).mean().fillna(0)

def calculate_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_hma(series: pd.Series, length: int = 50) -> pd.Series:
    half_len = int(length / 2)
    sqrt_len = int(math.sqrt(length))
    wma_half = series.rolling(half_len).mean()
    wma_full = series.rolling(length).mean()
    diff = 2 * wma_half - wma_full
    return diff.rolling(sqrt_len).mean()

def calculate_mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    rmf = tp * df['volume']
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(length).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(length).sum()
    return (100 - (100 / (1 + (pos / neg)))).fillna(50)

def run_titan_engine(df, amplitude, channel_dev, hma_len, use_hma_filter, tp1_r, tp2_r, tp3_r, mf_len, vol_len):
    if df.empty: return df
    df = df.copy().reset_index(drop=True)
    N = len(df)
    
    df['atr'] = calculate_atr(df, 14)
    df['dev'] = df['atr'] * channel_dev
    df['hma'] = calculate_hma(df['close'], int(hma_len))
    df['ll'] = df['low'].rolling(amplitude).min()
    df['hh'] = df['high'].rolling(amplitude).max()
    
    trend = np.zeros(N, dtype=int)
    stop = np.full(N, np.nan)
    curr_trend = 0
    curr_stop = np.nan
    
    for i in range(amplitude, N):
        local_low = df.at[i, 'll']
        local_high = df.at[i, 'hh']
        close = df.at[i, 'close']
        dev = df.at[i, 'dev'] if not pd.isna(df.at[i, 'dev']) else 0.0
        
        if curr_trend == 0:
            s = local_low + dev
            curr_stop = s if np.isnan(curr_stop) else max(curr_stop, s)
            if close < curr_stop:
                curr_trend = 1
                curr_stop = local_high - dev
        else:
            s = local_high - dev
            curr_stop = s if np.isnan(curr_stop) else min(curr_stop, s)
            if close > curr_stop:
                curr_trend = 0
                curr_stop = local_low + dev
        trend[i] = curr_trend
        stop[i] = curr_stop
    
    df['trend'] = trend
    df['trend_stop'] = stop
    df['is_bull'] = df['trend'] == 0
    
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['adx'] = calculate_adx(df, 14)
    df['rvol'] = (df['volume'] / df['volume'].rolling(vol_len).mean()).fillna(1.0)
    df['mfi'] = calculate_mfi(df, mf_len)
    
    df['bull_flip'] = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False).astype(bool))
    df['bear_flip'] = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True).astype(bool))
    df['regime'] = (df['adx'] > ADX_THRESHOLD) & (df['rvol'] > RVOL_THRESHOLD)
    df['hma_buy'] = (~use_hma_filter) | (df['close'] > df['hma'])
    df['hma_sell'] = (~use_hma_filter) | (df['close'] < df['hma'])
    
    df['buy_signal'] = df['bull_flip'] & df['regime'] & df['hma_buy'] & (df['rsi'] < RSI_OVERBOUGHT)
    df['sell_signal'] = df['bear_flip'] & df['regime'] & df['hma_sell'] & (df['rsi'] > RSI_OVERSOLD)
    
    df['signal_bar'] = (df['buy_signal'] | df['sell_signal']).astype(int).cumsum()
    df['signal_bar'] = df['signal_bar'].where(df['buy_signal'] | df['sell_signal'])
    df['active_signal'] = df['signal_bar'].ffill()
    
    df['entry_price'] = np.where(df['buy_signal'] | df['sell_signal'], df['close'], np.nan)
    df['entry_stop'] = np.where(df['buy_signal'] | df['sell_signal'], df['trend_stop'], np.nan)
    
    df['entry_price'] = df.groupby('active_signal')['entry_price'].ffill()
    df['entry_stop'] = df.groupby('active_signal')['entry_stop'].ffill()
    df['risk'] = (df['entry_price'] - df['entry_stop']).abs()
    
    df['tp1'] = np.where(df['is_bull'], df['entry_price'] + df['risk']*tp1_r, df['entry_price'] - df['risk']*tp1_r)
    df['tp2'] = np.where(df['is_bull'], df['entry_price'] + df['risk']*tp2_r, df['entry_price'] - df['risk']*tp2_r)
    df['tp3'] = np.where(df['is_bull'], df['entry_price'] + df['risk']*tp3_r, df['entry_price'] - df['risk']*tp3_r)
    
    return df

def backtest_engine(df, risk_pct=1.0, starting_balance=10000.0):
    if df.empty: return {"final_balance": starting_balance, "profit": 0, "num_trades": 0, "win_rate": 0, "trades": []}
    df = df.reset_index(drop=True).copy()
    balance = starting_balance
    pos_open = False
    pos = {}
    trades = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        if (row['buy_signal'] or row['sell_signal']) and not pos_open:
            entry, stop = row['entry_price'], row['entry_stop']
            if pd.isna(entry) or pd.isna(stop) or entry == stop: continue
            size = (balance * (risk_pct/100)) / abs(entry - stop)
            pos_open = True
            pos = {"side": "long" if row['buy_signal'] else "short", "entry": entry, "stop": stop, "size": size, 
                   "tp1": row['tp1'], "tp2": row['tp2'], "tp3": row['tp3'], "remaining": 1.0}
            continue
            
        if pos_open:
            if pos['side'] == 'long':
                if row['low'] <= pos['stop']:
                    pnl = (pos['stop'] - pos['entry']) * pos['size'] * pos['remaining']
                    balance += pnl
                    trades.append({"pnl": pnl, "outcome": "stopped"})
                    pos_open = False
                    continue
                for tp, frac in [('tp1', 0.3), ('tp2', 0.4), ('tp3', 1.0)]:
                    if row['high'] >= pos.get(tp, 0) and pos['remaining'] > 0:
                        pnl = (pos[tp] - pos['entry']) * pos['size'] * (frac if tp != 'tp3' else pos['remaining'])
                        balance += pnl
                        pos['remaining'] -= frac
                        if pos['remaining'] <= 0.01: 
                            trades.append({"pnl": pnl, "outcome": "tp"})
                            pos_open = False
                            break
            else:
                if row['high'] >= pos['stop']:
                    pnl = (pos['entry'] - pos['stop']) * pos['size'] * pos['remaining']
                    balance += pnl
                    trades.append({"pnl": pnl, "outcome": "stopped"})
                    pos_open = False
                    continue
                for tp, frac in [('tp1', 0.3), ('tp2', 0.4), ('tp3', 1.0)]:
                    if row['low'] <= pos.get(tp, 999999) and pos['remaining'] > 0:
                        pnl = (pos['entry'] - pos[tp]) * pos['size'] * (frac if tp != 'tp3' else pos['remaining'])
                        balance += pnl
                        pos['remaining'] -= frac
                        if pos['remaining'] <= 0.01:
                            trades.append({"pnl": pnl, "outcome": "tp"})
                            pos_open = False
                            break
                            
    num = len(trades)
    return {
        "final_balance": balance,
        "profit": balance - starting_balance,
        "num_trades": num,
        "win_rate": (sum(1 for t in trades if t['pnl'] > 0)/num)*100 if num else 0,
        "trades": trades
    }

# =============================================================================
# LAYOUT & UI
# =============================================================================
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Execution Chart — TITAN Engine")
    with st.spinner("Fetching data (Binance Vision)..."):
        df = get_klines(symbol, timeframe, limit)
    
    if not df.empty:
        df = run_titan_engine(df, amplitude, channel_dev, hma_len, use_hma_filter, tp1_r, tp2_r, tp3_r, mf_len, vol_len)
        last = df.iloc[-1]
        
        # --- UI FIX: CLEAN PLOTLY CHART ---
        fig = go.Figure()
        
        # 1. Price Candles
        fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                            name='Price', increasing_line_color='#00ffbb', decreasing_line_color='#ff1155')
        
        # 2. HMA (Subtle)
        if 'hma' in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', name='HMA',
                                     line=dict(color='rgba(0, 179, 255, 0.4)', width=1)))
        
        # 3. Trailing Stops (CRITICAL FIX: connectgaps=False to stop diagonal lines)
        bull_stop = df['trend_stop'].where(df['is_bull'], np.nan)
        bear_stop = df['trend_stop'].where(~df['is_bull'], np.nan)
        
        fig.add_trace(go.Scatter(x=df['timestamp'], y=bull_stop, mode='lines', name='Bull Stop', connectgaps=False,
                                 line=dict(color='#00ffbb', width=1.5)))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=bear_stop, mode='lines', name='Bear Stop', connectgaps=False,
                                 line=dict(color='#ff1155', width=1.5)))
        
        # 4. Clean Markers
        buys = df[df['buy_signal']]
        sells = df[df['sell_signal']]
        
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low']*0.995, mode='markers', name='BUY',
                                     marker=dict(symbol='triangle-up', size=10, color='#00ffbb', line=dict(color='white', width=1))))
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high']*1.005, mode='markers', name='SELL',
                                     marker=dict(symbol='triangle-down', size=10, color='#ff1155', line=dict(color='white', width=1))))

        fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False,
                          margin=dict(l=0, r=0, t=10, b=0), hovermode='x unified',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(gridcolor="#222")
        fig.update_yaxes(gridcolor="#222")
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        st.markdown("### 📈 Live Market Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trend", "🟢 BULL" if last['is_bull'] else "🔴 BEAR")
        m2.metric("Price", f"${last['close']:.2f}")
        m3.metric("Entry", f"${last['entry_price']:.2f}" if not pd.isna(last['entry_price']) else "-")
        m4.metric("Stop", f"${last['trend_stop']:.2f}" if not pd.isna(last['trend_stop']) else "-")
        m5.metric("ADX/RVOL", f"{last['adx']:.0f} / {last['rvol']:.1f}x")
        
        # Signal ID Check for Broadcast
        if tg_on and (last['buy_signal'] or last['sell_signal']):
            sid = f"{last['timestamp']}_{symbol}"
            if sid != st.session_state.get("last_sid", ""):
                txt = f"🔥 TITAN: {symbol} {'LONG' if last['is_bull'] else 'SHORT'} @ {last['close']:.2f}"
                send_telegram_msg(tg_token, tg_chat, txt)
                journal_signal(db_conn, {"ts": str(last['timestamp']), "symbol": symbol, "timeframe": timeframe,
                                         "direction": "LONG" if last['is_bull'] else "SHORT", "entry": last['close'], 
                                         "adx": last['adx'], "rvol": last['rvol']})
                st.session_state["last_sid"] = sid
                st.toast("Signal Broadcasted!")

        # Action Buttons
        b1, b2 = st.columns(2)
        if b1.button("🧮 Run Backtest", use_container_width=True):
            res = backtest_engine(df, backtest_risk)
            st.success(f"PnL: ${res['profit']:.2f} | Win Rate: {res['win_rate']:.1f}%")
        if b2.button("📥 Export Data", use_container_width=True):
            st.download_button("Download CSV", df.to_csv(), "titan_data.csv", "text/csv")

with col_right:
    st.subheader("📊 TradingView")
    html = f"""
    <div id="tv" style="height:650px;"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.widget({{
        "autosize": true, "symbol": "BINANCE:{symbol}", "interval": "{'60' if timeframe=='1h' else 'D'}",
        "timezone": "Etc/UTC", "theme": "dark", "style": "1", "container_id": "tv", "hide_side_toolbar": false
    }});
    </script>
    """
    components.html(html, height=650)
    
    st.markdown("### 📝 Signal Log")
    if persist and db_conn:
        df_log = pd.read_sql("SELECT * FROM signals ORDER BY id DESC LIMIT 20", db_conn)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
