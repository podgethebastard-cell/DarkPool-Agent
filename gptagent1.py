"""
TITAN INTRADAY PRO - Production-Ready Trading Dashboard
Version 2.3: Full Telegram Restoration + Layout Optimization
"""
import time
import math
import sqlite3
import atexit
from typing import Dict, Optional, Tuple
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

# Hardcoded to Binance Vision (Public Data) as requested
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
# DATABASE MANAGEMENT (FULL IMPLEMENTATION)
# =============================================================================
@contextmanager
def get_db_connection(path: str = DB_PATH):
    """Context manager for database connections with timeout safety"""
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    try:
        yield conn
    finally:
        conn.close()


def init_db(path: str = DB_PATH) -> Optional[sqlite3.Connection]:
    """Initialize database with proper schema"""
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
    
    # Create index for faster queries
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts 
        ON signals(symbol, ts DESC)
    """)
    
    conn.commit()
    return conn


def cleanup_db():
    """Cleanup function to close DB on exit"""
    if 'db_conn' in st.session_state and st.session_state.db_conn:
        try:
            st.session_state.db_conn.close()
        except Exception:
            pass


# Initialize DB and register cleanup
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_db()
    atexit.register(cleanup_db)

db_conn = st.session_state.db_conn


def journal_signal(conn: Optional[sqlite3.Connection], payload: Dict) -> bool:
    """Journal signal to database with proper error handling"""
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
# TELEGRAM INTEGRATION (FULL IMPLEMENTATION)
# =============================================================================
def escape_markdown_v2(text: str) -> str:
    """Escape Telegram MarkdownV2 special characters"""
    special_chars = r'_*[]()~`>#+-=|{}.!'
    return ''.join('\\' + c if c in special_chars else c for c in text)


def send_telegram_msg(
    token: str,
    chat: str,
    msg: str,
    cooldown_s: int = 30
) -> bool:
    """
    Send Telegram message with cooldown and retry logic
    """
    if not token or not chat:
        return False
    
    # Check cooldown
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
            elif r.status_code == 429:  # Rate limited
                retry_after = int(r.headers.get('Retry-After', RETRY_DELAY))
                time.sleep(retry_after)
            else:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except requests.exceptions.Timeout:
            time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            st.warning(f"Telegram error (attempt {attempt + 1}): {str(e)}")
            time.sleep(RETRY_DELAY * (attempt + 1))
    
    return False


def make_signal_text(row: pd.Series, symbol: str, timeframe: str) -> str:
    """Generate professional signal text for Telegram"""
    dir_txt = "LONG" if row['is_bull'] else "SHORT"
    
    entry = row['entry_price'] if not pd.isna(row['entry_price']) else 0
    stop = row['entry_stop'] if not pd.isna(row['entry_stop']) else 0
    tp1 = row['tp1'] if not pd.isna(row['tp1']) else 0
    tp2 = row['tp2'] if not pd.isna(row['tp2']) else 0
    tp3 = row['tp3'] if not pd.isna(row['tp3']) else 0
    
    return (
        f"🔥 TITAN SIGNAL — {symbol} {timeframe}\n\n"
        f"Direction: {dir_txt}\n"
        f"Entry: {entry:.2f}\n"
        f"Stop: {stop:.2f}\n\n"
        f"Targets:\n"
        f"TP1 (30%): {tp1:.2f}\n"
        f"TP2 (40%): {tp2:.2f}\n"
        f"TP3 (30%): {tp3:.2f}\n\n"
        f"Indicators:\n"
        f"ADX: {row['adx']:.2f}\n"
        f"RVOL: {row['rvol']:.2f}x\n"
        f"RSI: {row['rsi']:.2f}\n\n"
        f"⚠️ Not financial advice. Trade at your own risk."
    )


def make_signal_id(row: pd.Series) -> str:
    """Generate unique signal ID for deduplication"""
    ts = row['timestamp']
    dir_flag = "L" if row['is_bull'] else "S"
    entry = row['entry_price'] if not pd.isna(row['entry_price']) else 0.0
    return f"{ts.isoformat()}_{dir_flag}_{entry:.8f}"


# =============================================================================
# BINANCE API (VISION DATA)
# =============================================================================
@st.cache_data(ttl=10)
def get_klines(
    symbol: str,
    interval: str,
    limit: int = 500
) -> pd.DataFrame:
    """Fetch klines from Binance Vision (Public Data)"""
    url = f"{BINANCE_API_BASE}/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            if not data or isinstance(data, dict):
                return pd.DataFrame()
            
            cols = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ]
            df = pd.DataFrame(data, columns=cols)
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = \
                df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                st.error(f"API Error: {str(e)}")
            time.sleep(RETRY_DELAY)
    
    return pd.DataFrame()


# =============================================================================
# TECHNICAL INDICATORS & ENGINE
# =============================================================================
def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    
    up = high.diff()
    down = -low.diff()
    
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
    
    wma_half = series.rolling(window=half_len, min_periods=half_len).mean()
    wma_full = series.rolling(window=length, min_periods=length).mean()
    
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(window=sqrt_len, min_periods=sqrt_len).mean()
    
    return hma


def calculate_mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    rmf = tp * df['volume']
    
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(length, min_periods=length).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(length, min_periods=length).sum()
    
    mfi = 100 - (100 / (1 + (pos / neg)))
    
    return mfi.fillna(50)


def run_titan_engine(
    df: pd.DataFrame,
    amplitude: int,
    channel_dev: float,
    hma_len: int,
    use_hma_filter: bool,
    tp1_r: float,
    tp2_r: float,
    tp3_r: float,
    mf_len: int,
    vol_len: int
) -> pd.DataFrame:
    """Core trading engine with non-repainting signal generation"""
    if df.empty:
        return df
    
    df = df.copy().reset_index(drop=True)
    N = len(df)
    
    # Base Indicators
    df['atr'] = calculate_atr(df, 14)
    df['dev'] = df['atr'] * channel_dev
    df['hma'] = calculate_hma(df['close'], int(hma_len))
    df['ll'] = df['low'].rolling(window=amplitude, min_periods=amplitude).min()
    df['hh'] = df['high'].rolling(window=amplitude, min_periods=amplitude).max()
    
    # Logic Engine (Trailing Stop State Machine)
    trend = np.zeros(N, dtype=int)
    stop = np.full(N, np.nan)
    curr_trend = 0
    curr_stop = np.nan
    
    for i in range(amplitude, N):
        local_low = df.at[i, 'll']
        local_high = df.at[i, 'hh']
        close = df.at[i, 'close']
        dev = df.at[i, 'dev'] if not pd.isna(df.at[i, 'dev']) else 0.0
        
        if curr_trend == 0:  # Bull trend
            s = local_low + dev
            curr_stop = s if np.isnan(curr_stop) else max(curr_stop, s)
            if close < curr_stop:
                curr_trend = 1
                curr_stop = local_high - dev
        else:  # Bear trend
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
    
    # Filters
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['adx'] = calculate_adx(df, 14)
    df['rvol'] = (df['volume'] / df['volume'].rolling(vol_len, min_periods=vol_len).mean()).fillna(1.0)
    df['mfi'] = calculate_mfi(df, mf_len)
    
    # Signal Logic
    df['bull_flip'] = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False).astype(bool))
    df['bear_flip'] = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True).astype(bool))
    
    df['regime'] = (df['adx'] > ADX_THRESHOLD) & (df['rvol'] > RVOL_THRESHOLD)
    df['hma_buy'] = (~use_hma_filter) | (df['close'] > df['hma'])
    df['hma_sell'] = (~use_hma_filter) | (df['close'] < df['hma'])
    
    df['buy_signal'] = (
        df['bull_flip'] &
        df['regime'] &
        df['hma_buy'] &
        (df['rsi'] < RSI_OVERBOUGHT)
    )
    df['sell_signal'] = (
        df['bear_flip'] &
        df['regime'] &
        df['hma_sell'] &
        (df['rsi'] > RSI_OVERSOLD)
    )
    
    # State-Locking (Freezing Entry/Stop/TPs at signal bar)
    df['signal_bar'] = (df['buy_signal'] | df['sell_signal']).astype(int).cumsum()
    df['signal_bar'] = df['signal_bar'].where(df['buy_signal'] | df['sell_signal'])
    df['active_signal'] = df['signal_bar'].ffill()
    
    df['entry_price'] = np.where(
        df['buy_signal'] | df['sell_signal'],
        df['close'],
        np.nan
    )
    df['entry_stop'] = np.where(
        df['buy_signal'] | df['sell_signal'],
        df['trend_stop'],
        np.nan
    )
    
    # Forward-fill entry params
    df['entry_price'] = df.groupby('active_signal')['entry_price'].ffill()
    df['entry_stop'] = df.groupby('active_signal')['entry_stop'].ffill()
    
    df['risk'] = np.where(
        (~df['entry_price'].isna()) & (~df['entry_stop'].isna()),
        (df['entry_price'] - df['entry_stop']).abs(),
        np.nan
    )
    
    # Calculate TPs once
    df['tp1'] = np.where(
        df['is_bull'],
        df['entry_price'] + df['risk'] * tp1_r,
        df['entry_price'] - df['risk'] * tp1_r
    )
    df['tp2'] = np.where(
        df['is_bull'],
        df['entry_price'] + df['risk'] * tp2_r,
        df['entry_price'] - df['risk'] * tp2_r
    )
    df['tp3'] = np.where(
        df['is_bull'],
        df['entry_price'] + df['risk'] * tp3_r,
        df['entry_price'] - df['risk'] * tp3_r
    )
    
    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def backtest_engine(
    df: pd.DataFrame,
    risk_pct: float = 1.0,
    starting_balance: float = 10000.0
) -> Dict:
    if df.empty:
        return {
            "final_balance": starting_balance,
            "profit": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "trades": []
        }
    
    df = df.reset_index(drop=True).copy()
    balance = starting_balance
    pos_open = False
    pos = {}
    trades = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Entry Logic
        if (row['buy_signal'] or row['sell_signal']) and not pos_open:
            entry = row['entry_price']
            stop = row['entry_stop']
            
            if pd.isna(entry) or pd.isna(stop) or entry == stop:
                continue
            
            per_unit_risk = abs(entry - stop)
            if per_unit_risk == 0:
                continue
            
            risk_amount = balance * (risk_pct / 100.0)
            size = risk_amount / per_unit_risk
            
            pos_open = True
            pos = {
                "side": "long" if row['buy_signal'] else "short",
                "entry": entry,
                "stop": stop,
                "size": size,
                "tp1": row['tp1'],
                "tp2": row['tp2'],
                "tp3": row['tp3'],
                "remaining": 1.0
            }
            continue
        
        # Position Management
        if pos_open:
            high = row['high']
            low = row['low']
            
            if pos['side'] == 'long':
                # Stop hit check first
                if low <= pos['stop']:
                    loss = (pos['entry'] - pos['stop']) * pos['size'] * pos['remaining']
                    balance -= loss
                    trades.append({
                        "entry": pos['entry'],
                        "exit": pos['stop'],
                        "pnl": -loss,
                        "outcome": "stopped"
                    })
                    pos_open = False
                    pos = {}
                    continue
                
                # TP check
                for tp_label, frac in [('tp1', 0.3), ('tp2', 0.4), ('tp3', 1.0)]:
                    tp_val = pos.get(tp_label)
                    if pd.isna(tp_val) or pos['remaining'] <= 0:
                        continue
                    
                    if high >= tp_val:
                        take_frac = frac if tp_label != 'tp3' else pos['remaining']
                        profit = (tp_val - pos['entry']) * pos['size'] * take_frac
                        balance += profit
                        pos['remaining'] -= take_frac
                        
                        if pos['remaining'] <= 0.01:
                            trades.append({
                                "entry": pos['entry'],
                                "exit": tp_val,
                                "pnl": profit,
                                "outcome": tp_label
                            })
                            pos_open = False
                            pos = {}
                            break
                            
            else:  # Short
                # Stop hit check first
                if high >= pos['stop']:
                    loss = (pos['stop'] - pos['entry']) * pos['size'] * pos['remaining']
                    balance -= loss
                    trades.append({
                        "entry": pos['entry'],
                        "exit": pos['stop'],
                        "pnl": -loss,
                        "outcome": "stopped"
                    })
                    pos_open = False
                    pos = {}
                    continue
                
                # TP check
                for tp_label, frac in [('tp1', 0.3), ('tp2', 0.4), ('tp3', 1.0)]:
                    tp_val = pos.get(tp_label)
                    if pd.isna(tp_val) or pos['remaining'] <= 0:
                        continue
                    
                    if low <= tp_val:
                        take_frac = frac if tp_label != 'tp3' else pos['remaining']
                        profit = (pos['entry'] - tp_val) * pos['size'] * take_frac
                        balance += profit
                        pos['remaining'] -= take_frac
                        
                        if pos['remaining'] <= 0.01:
                            trades.append({
                                "entry": pos['entry'],
                                "exit": tp_val,
                                "pnl": profit,
                                "outcome": tp_label
                            })
                            pos_open = False
                            pos = {}
                            break
    
    num_trades = len(trades)
    win_rate = (sum(1 for t in trades if t['pnl'] > 0) / num_trades * 100) if num_trades > 0 else 0.0
    
    return {
        "final_balance": balance,
        "profit": balance - starting_balance,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "trades": trades
    }


# =============================================================================
# TRADINGVIEW WIDGET
# =============================================================================
def tradingview_widget(symbol_tv: str = "BINANCE:BTCUSDT", interval: str = "60") -> str:
    """Generate TradingView embed HTML"""
    interval_map = {
        "1m": "1", "5m": "5", "15m": "15",
        "1h": "60", "4h": "240", "1d": "D"
    }
    iv = interval_map.get(interval, "60")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin:0;background:#0e0e0e;">
        <div id="tv-widget" style="width:100%;height:640px;"></div>
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>
        new TradingView.widget({{
            "autosize": true,
            "symbol": "{symbol_tv}",
            "interval": "{iv}",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "container_id": "tv-widget",
            "hide_side_toolbar": false,
            "allow_symbol_change": true,
            "save_image": false
        }});
        </script>
    </body>
    </html>
    """
    return html_content


# =============================================================================
# MAIN LAYOUT & EXECUTION
# =============================================================================
# 1. Main Execution Chart (Full Width)
st.subheader("📊 Execution Chart — TITAN Engine")

with st.spinner("Fetching data (Binance Vision)..."):
    df = get_klines(symbol, timeframe, limit)

if not df.empty:
    with st.spinner("Running TITAN engine..."):
        df = run_titan_engine(
            df,
            amplitude=int(amplitude),
            channel_dev=float(channel_dev),
            hma_len=int(hma_len),
            use_hma_filter=bool(use_hma_filter),
            tp1_r=float(tp1_r),
            tp2_r=float(tp2_r),
            tp3_r=float(tp3_r),
            mf_len=int(mf_len),
            vol_len=int(vol_len)
        )
    
    last = df.iloc[-1]
    
    # --- PLOTLY CHART ---
    fig = go.Figure()
    
    # Candles
    fig.add_candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Price',
        increasing_line_color='#00ffbb',
        decreasing_line_color='#ff1155'
    )
    
    # HMA
    if 'hma' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['hma'],
            mode='lines', name='HMA',
            line=dict(color='rgba(0, 179, 255, 0.4)', width=1)
        ))
    
    # Trailing Stops (No gaps)
    bull_stop = df['trend_stop'].where(df['is_bull'], np.nan)
    bear_stop = df['trend_stop'].where(~df['is_bull'], np.nan)
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=bull_stop,
        mode='lines', name='Bull Stop', connectgaps=False,
        line=dict(color='#00ffbb', width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=bear_stop,
        mode='lines', name='Bear Stop', connectgaps=False,
        line=dict(color='#ff1155', width=1.5)
    ))
    
    # Markers
    buys = df[df['buy_signal']]
    sells = df[df['sell_signal']]
    
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys['timestamp'], y=buys['low']*0.995,
            mode='markers', name='BUY',
            marker=dict(symbol='triangle-up', size=12, color='#00ffbb', line=dict(color='white', width=1)),
            text=[f"Entry: {e:.2f}" for e in buys['entry_price']],
            hoverinfo='text+x'
        ))
    
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells['timestamp'], y=sells['high']*1.005,
            mode='markers', name='SELL',
            marker=dict(symbol='triangle-down', size=12, color='#ff1155', line=dict(color='white', width=1)),
            text=[f"Entry: {e:.2f}" for e in sells['entry_price']],
            hoverinfo='text+x'
        ))
    
    fig.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(gridcolor="#222")
    fig.update_yaxes(gridcolor="#222")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- METRICS & TELEGRAM LOGIC ---
    st.markdown("### 📈 Live Market Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    m1.metric("Trend", "🟢 BULL" if last['is_bull'] else "🔴 BEAR")
    m2.metric("Price", f"${last['close']:.2f}")
    m3.metric("Entry", f"${last['entry_price']:.2f}" if not pd.isna(last['entry_price']) else "-")
    m4.metric("Stop", f"${last['trend_stop']:.2f}" if not pd.isna(last['trend_stop']) else "-")
    m5.metric("ADX / RVOL", f"{last['adx']:.0f} / {last['rvol']:.1f}x")
    
    # Auto Broadcast
    if tg_on and tg_token and tg_chat:
        if last['buy_signal'] or last['sell_signal']:
            sid = make_signal_id(last)
            last_sid = st.session_state.get("last_signal_id", "")
            
            if sid != last_sid:
                # 1. Journal
                if db_conn:
                    journal_signal(db_conn, {
                        "ts": str(last['timestamp']),
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "direction": "LONG" if last['is_bull'] else "SHORT",
                        "entry": float(last['entry_price']),
                        "stop": float(last['entry_stop']),
                        "tp1": float(last['tp1']),
                        "tp2": float(last['tp2']),
                        "tp3": float(last['tp3']),
                        "adx": float(last['adx']),
                        "rvol": float(last['rvol']),
                        "notes": "Auto-broadcast"
                    })
                
                # 2. Send
                txt = make_signal_text(last, symbol, timeframe)
                sent = send_telegram_msg(tg_token, tg_chat, txt, int(telegram_cooldown_s))
                
                if sent:
                    st.success("✅ Auto-broadcast sent!")
                    st.session_state["last_signal_id"] = sid
                else:
                    st.info("ℹ️ Telegram cooldown/error")

    # --- ACTION CENTER ---
    st.markdown("---")
    st.markdown("### ⚡ Action Center")
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    
    with c_btn1:
        if st.button("🔥 Manual Broadcast", use_container_width=True):
            if tg_token and tg_chat:
                txt = make_signal_text(last, symbol, timeframe)
                if send_telegram_msg(tg_token, tg_chat, txt, int(telegram_cooldown_s)):
                    st.success("✅ Signal Sent")
                    if db_conn:
                         journal_signal(db_conn, {
                            "ts": str(last['timestamp']),
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "direction": "LONG" if last['is_bull'] else "SHORT",
                            "entry": float(last['entry_price']),
                            "stop": float(last['entry_stop']),
                            "tp1": float(last['tp1']),
                            "tp2": float(last['tp2']),
                            "tp3": float(last['tp3']),
                            "adx": float(last['adx']),
                            "rvol": float(last['rvol']),
                            "notes": "Manual"
                        })
                else:
                    st.error("❌ Send failed (check creds or cooldown)")
            else:
                st.error("❌ Missing Telegram Credentials")
                
    with c_btn2:
        if st.button("📥 Export CSV", use_container_width=True):
            st.download_button(
                "⬇️ Download",
                df.to_csv(),
                f"titan_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
            
    with c_btn3:
        if st.button("🧮 Run Backtest", use_container_width=True):
            res = backtest_engine(df, float(backtest_risk))
            st.success(f"PnL: ${res['profit']:.2f} | Win Rate: {res['win_rate']:.1f}% ({res['num_trades']} trades)")
            if res['trades']:
                st.dataframe(pd.DataFrame(res['trades'][-5:]), use_container_width=True)

    # --- LOWER SECTION: TRADINGVIEW & LOGS ---
    st.markdown("---")
    st.subheader("📉 Manual Charting & Logs")
    
    # TradingView Full Width at Bottom
    tv_symbol = f"BINANCE:{symbol}"
    tv_html = tradingview_widget(tv_symbol, timeframe)
    components.html(tv_html, height=700)
    
    # Signal Log
    if persist and db_conn:
        st.markdown("#### Signal Journal")
        try:
            cur = db_conn.cursor()
            cur.execute("""
                SELECT ts, symbol, timeframe, direction, entry, stop, tp1, tp2, tp3, adx, rvol, notes 
                FROM signals ORDER BY id DESC LIMIT 50
            """)
            rows = cur.fetchall()
            if rows:
                cols = ["TS", "Sym", "TF", "Dir", "Ent", "Stop", "TP1", "TP2", "TP3", "ADX", "RVOL", "Note"]
                st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True, hide_index=True)
            else:
                st.info("No signals recorded yet.")
        except Exception as e:
            st.error(f"DB Read Error: {e}")
