"""
TITAN INTRADAY PRO - Production-Ready Trading Dashboard
Fixed Version: All critical issues resolved
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
# CONSTANTS
# =============================================================================
ADX_THRESHOLD = 23.0
RVOL_THRESHOLD = 1.15
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
DB_PATH = "titan_signals.db"
BINANCE_API_BASE = "https://api.binance.com/api/v3"
MAX_RETRIES = 3
RETRY_DELAY = 0.5

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
vol_metric = st.sidebar.selectbox(
    "Volume metric",
    ["CMF", "MFI", "Volume RSI", "RVOL", "Vol Osc"],
    index=0
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
# DATABASE MANAGEMENT (with proper cleanup)
# =============================================================================
@contextmanager
def get_db_connection(path: str = DB_PATH):
    """Context manager for database connections"""
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def init_db(path: str = DB_PATH) -> Optional[sqlite3.Connection]:
    """Initialize database with proper schema"""
    if not persist:
        return None
    
    conn = sqlite3.connect(path, check_same_thread=False)
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
# TELEGRAM INTEGRATION (with proper escaping and rate limiting)
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
    
    Returns:
        bool: True if message sent successfully
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


# =============================================================================
# BINANCE API (with exponential backoff and proper error handling)
# =============================================================================
@st.cache_data(ttl=10)
def get_klines(
    symbol: str,
    interval: str,
    limit: int = 500
) -> pd.DataFrame:
    """
    Fetch klines from Binance with retry logic and exponential backoff
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles to fetch
    
    Returns:
        DataFrame with OHLCV data or empty DataFrame on failure
    """
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
            
            # Validate response
            if not data or isinstance(data, dict):
                if isinstance(data, dict) and 'msg' in data:
                    st.error(f"Binance API error: {data['msg']}")
                return pd.DataFrame()
            
            # Parse response
            cols = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ]
            df = pd.DataFrame(data, columns=cols)
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = \
                df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            # Validate data quality
            if df['close'].isna().any() or (df['volume'] < 0).any():
                st.warning("Data quality issue detected - some values are invalid")
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                wait_time = int(e.response.headers.get('Retry-After', RETRY_DELAY * (2 ** attempt)))
                st.warning(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error(f"HTTP error: {e}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            wait_time = RETRY_DELAY * (2 ** attempt)
            st.warning(f"Network error (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait_time)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return pd.DataFrame()
    
    return pd.DataFrame()


# =============================================================================
# TECHNICAL INDICATORS (vectorized, non-repainting)
# =============================================================================
def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    
    return atr


def calculate_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Calculate Average Directional Index"""
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
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    
    return adx.fillna(0)


def calculate_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)


def calculate_hma(series: pd.Series, length: int = 50) -> pd.Series:
    """Calculate Hull Moving Average"""
    half_len = int(length / 2)
    sqrt_len = int(math.sqrt(length))
    
    wma_half = series.rolling(window=half_len, min_periods=half_len).mean()
    wma_full = series.rolling(window=length, min_periods=length).mean()
    
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(window=sqrt_len, min_periods=sqrt_len).mean()
    
    return hma


def calculate_mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Calculate Money Flow Index"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    rmf = tp * df['volume']
    
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(length, min_periods=length).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(length, min_periods=length).sum()
    
    mfi = 100 - (100 / (1 + (pos / neg)))
    
    return mfi.fillna(50)


# =============================================================================
# TITAN ENGINE (state-locked, non-repainting)
# =============================================================================
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
    """
    Core trading engine with non-repainting signal generation
    
    Returns:
        DataFrame with all indicators and signals
    """
    if df.empty:
        return df
    
    df = df.copy().reset_index(drop=True)
    N = len(df)
    
    # Calculate base indicators
    df['atr'] = calculate_atr(df, 14)
    df['dev'] = df['atr'] * channel_dev
    df['hma'] = calculate_hma(df['close'], int(hma_len))
    df['ll'] = df['low'].rolling(window=amplitude, min_periods=amplitude).min()
    df['hh'] = df['high'].rolling(window=amplitude, min_periods=amplitude).max()
    
    # Trailing stop engine (deterministic state machine)
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
    
    # Momentum & regime filters
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['adx'] = calculate_adx(df, 14)
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_len, min_periods=vol_len).mean()
    df['rvol'] = df['rvol'].fillna(1.0)
    
    # Volume metrics
    df['mfi'] = calculate_mfi(df, mf_len)
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
          (df['high'] - df['low'])
    df['cmf'] = (mfm.fillna(0) * df['volume']).rolling(vol_len, min_periods=vol_len).sum() / \
                df['volume'].rolling(vol_len, min_periods=vol_len).sum()
    df['vol_rsi'] = calculate_rsi(df['volume'], 14)
    
    v_short = df['volume'].rolling(14, min_periods=14).mean()
    v_long = df['volume'].rolling(28, min_periods=28).mean()
    df['vol_osc'] = np.where(v_long != 0, 100 * (v_short - v_long) / v_long, 0.0)
    
    # Signal detection (trend flips)
    df['bull_flip'] = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False).astype(bool))
    df['bear_flip'] = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True).astype(bool))
    
    # Regime filters
    df['regime'] = (df['adx'] > ADX_THRESHOLD) & (df['rvol'] > RVOL_THRESHOLD)
    df['hma_buy'] = (~use_hma_filter) | (df['close'] > df['hma'])
    df['hma_sell'] = (~use_hma_filter) | (df['close'] < df['hma'])
    
    # Final signals
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
    
    # ==========================================================================
    # CRITICAL FIX: State-locked entry, stop, and ladder
    # These values are frozen at signal bar and never recalculated
    # ==========================================================================
    
    # Create signal ID for each signal bar
    df['signal_bar'] = (df['buy_signal'] | df['sell_signal']).astype(int).cumsum()
    df['signal_bar'] = df['signal_bar'].where(df['buy_signal'] | df['sell_signal'])
    
    # Forward-fill signal bar ID to track which position we're in
    df['active_signal'] = df['signal_bar'].ffill()
    
    # Freeze entry price at signal bar
    df['entry_price'] = np.where(
        df['buy_signal'] | df['sell_signal'],
        df['close'],
        np.nan
    )
    
    # Freeze stop at signal bar
    df['entry_stop'] = np.where(
        df['buy_signal'] | df['sell_signal'],
        df['trend_stop'],
        np.nan
    )
    
    # Forward-fill entry and stop for the duration of the position
    df['entry_price'] = df.groupby('active_signal')['entry_price'].ffill()
    df['entry_stop'] = df.groupby('active_signal')['entry_stop'].ffill()
    
    # Calculate risk ONCE at entry (never changes)
    df['risk'] = np.where(
        (~df['entry_price'].isna()) & (~df['entry_stop'].isna()),
        (df['entry_price'] - df['entry_stop']).abs(),
        np.nan
    )
    
    # Freeze ladder TPs at signal bar (CRITICAL FIX)
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
# BACKTEST ENGINE (with proper stop-before-TP ordering)
# =============================================================================
def backtest_engine(
    df: pd.DataFrame,
    risk_pct: float = 1.0,
    starting_balance: float = 10000.0
) -> Dict:
    """
    Walk-forward backtest with realistic intrabar execution
    
    CRITICAL FIX: Checks if stop hit before TP on same bar
    """
    if df.empty:
        return {
            "final_balance": starting_balance,
            "profit": 0.0,
            "num_trades": 0,
            "win_rate": 0.0
        }
    
    df = df.reset_index(drop=True).copy()
    balance = starting_balance
    pos_open = False
    pos = {}
    trades = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Open new position on signal
        if (row['buy_signal'] or row['sell_signal']) and not pos_open:
            entry = row['entry_price']
            stop = row['entry_stop']
            
            # Validate entry
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
                "remaining": 1.0,
                "entry_bar": i
            }
            continue
        
        # Manage open position
        if pos_open:
            high = row['high']
            low = row['low']
            
            # CRITICAL FIX: Check stop FIRST before TPs
            if pos['side'] == 'long':
                # Check if stopped out (takes priority)
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
                
                # Check TPs (only if not stopped)
                for tp_label, frac in [('tp1', 0.3), ('tp2', 0.4), ('tp3', 1.0)]:
                    tp_val = pos.get(tp_label)
                    if pd.isna(tp_val) or pos['remaining'] <= 0:
                        continue
                    
                    if high >= tp_val:
                        take_frac = frac if tp_label != 'tp3' else pos['remaining']
                        profit = (tp_val - pos['entry']) * pos['size'] * take_frac
                        balance += profit
                        pos['remaining'] -= take_frac
                        
                        if pos['remaining'] <= 0.01:  # Fully closed
                            trades.append({
                                "entry": pos['entry'],
                                "exit": tp_val,
                                "pnl": profit,
                                "outcome": tp_label
                            })
                            pos_open = False
                            pos = {}
                            break
            
            else:  # Short position
                # Check if stopped out (takes priority)
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
                
                # Check TPs (only if not stopped)
                for tp_label, frac in [('tp1', 0.3), ('tp2', 0.4), ('tp3', 1.0)]:
                    tp_val = pos.get(tp_label)
                    if pd.isna(tp_val) or pos['remaining'] <= 0:
                        continue
                    
                    if low <= tp_val:
                        take_frac = frac if tp_label != 'tp3' else pos['remaining']
                        profit = (pos['entry'] - tp_val) * pos['size'] * take_frac
                        balance += profit
                        pos['remaining'] -= take_frac
                        
                        if pos['remaining'] <= 0.01:  # Fully closed
                            trades.append({
                                "entry": pos['entry'],
                                "exit": tp_val,
                                "pnl": profit,
                                "outcome": tp_label
                            })
                            pos_open = False
                            pos = {}
                            break
    
    # Calculate statistics
    num_trades = len(trades)
    if num_trades > 0:
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (wins / num_trades) * 100
    else:
        win_rate = 0.0
    
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
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "1h": "60",
        "4h": "240",
        "1d": "D"
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
# SIGNAL UTILITIES
# =============================================================================
def make_signal_id(row: pd.Series) -> str:
    """Generate unique signal ID"""
    ts = row['timestamp']
    dir_flag = "L" if row['is_bull'] else "S"
    entry = row['entry_price'] if not pd.isna(row['entry_price']) else 0.0
    return f"{ts.isoformat()}_{dir_flag}_{entry:.8f}"


def make_signal_text(row: pd.Series, symbol: str, timeframe: str) -> str:
    """Generate formatted signal text for Telegram"""
    dir_txt = "LONG" if row['is_bull'] else "SHORT"
    
    # Format with proper decimal places
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


# =============================================================================
# MAIN APPLICATION LAYOUT
# =============================================================================
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Execution Chart — TITAN Engine")
    
    # Fetch market data with progress indicator
    with st.spinner("Fetching market data from Binance..."):
        df = get_klines(symbol, timeframe, limit)
    
    if df.empty:
        st.error(
            f"❌ Market data not available for {symbol}.\n\n"
            "Please check:\n"
            "- Symbol format (e.g., BTCUSDT, ETHUSDT)\n"
            "- Network connection\n"
            "- Binance API status"
        )
        st.stop()
    
    # Data quality check
    if len(df) < limit * 0.8:
        st.warning(f"⚠️ Only received {len(df)} candles (requested {limit})")
    
    # Run trading engine
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
    
    # ==========================================================================
    # PLOTLY CHART (with all indicators and signals)
    # ==========================================================================
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#00ffbb',
        decreasing_line_color='#ff1155'
    )
    
    # HMA overlay
    if 'hma' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['hma'],
            mode='lines',
            name='HMA',
            line=dict(color='#00b3ff', width=2)
        ))
    
    # Trailing stops (separate for bull/bear)
    bull_stop = df['trend_stop'].where(df['is_bull'], np.nan)
    bear_stop = df['trend_stop'].where(~df['is_bull'], np.nan)
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=bull_stop,
        mode='lines',
        name='Bull Trail',
        line=dict(color='#00ffbb', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=bear_stop,
        mode='lines',
        name='Bear Trail',
        line=dict(color='#ff1155', width=2, dash='dot')
    ))
    
    # Ladder targets (state-locked)
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['tp1'],
        mode='lines',
        name='TP1 (30%)',
        line=dict(color='#00ff88', width=1, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['tp2'],
        mode='lines',
        name='TP2 (40%)',
        line=dict(color='#00ff88', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['tp3'],
        mode='lines',
        name='TP3 (30%)',
        line=dict(color='#00ff88', width=1, dash='longdash')
    ))
    
    # Signal markers
    buys = df[df['buy_signal']]
    sells = df[df['sell_signal']]
    
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys['timestamp'],
            y=buys['low'] * 0.998,
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=14,
                color='#00ffbb',
                line=dict(color='white', width=1)
            ),
            name='BUY',
            text=[f"Entry: {e:.2f}" for e in buys['entry_price']],
            hoverinfo='text+x'
        ))
    
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells['timestamp'],
            y=sells['high'] * 1.002,
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=14,
                color='#ff1155',
                line=dict(color='white', width=1)
            ),
            name='SELL',
            text=[f"Entry: {e:.2f}" for e in sells['entry_price']],
            hoverinfo='text+x'
        ))
    
    # Layout styling
    fig.update_layout(
        height=720,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig.update_xaxes(gridcolor="#222")
    fig.update_yaxes(gridcolor="#222")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # METRICS DASHBOARD
    # ==========================================================================
    st.markdown("### 📈 Live Market Metrics")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    trend_label = "🟢 BULL" if last['is_bull'] else "🔴 BEAR"
    c1.metric("Trend", trend_label)
    c2.metric("Price", f"${last['close']:.2f}")
    c3.metric(
        "Trailing Stop",
        f"${last['trend_stop']:.2f}" if not pd.isna(last['trend_stop']) else "N/A"
    )
    c4.metric("ADX", f"{last['adx']:.1f}")
    c5.metric("RVOL", f"{last['rvol']:.2f}x")
    
    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("RSI", f"{last['rsi']:.1f}")
    c7.metric("MFI", f"{last['mfi']:.1f}")
    c8.metric(
        "Entry",
        f"${last['entry_price']:.2f}" if not pd.isna(last['entry_price']) else "N/A"
    )
    c9.metric(
        "Risk",
        f"${last['risk']:.2f}" if not pd.isna(last['risk']) else "N/A"
    )
    
    # Signal status indicator
    if last['buy_signal']:
        c10.success("🔥 BUY SIGNAL")
    elif last['sell_signal']:
        c10.error("🔥 SELL SIGNAL")
    else:
        c10.info("⏳ No Signal")
    
    # ==========================================================================
    # ACTION CENTER (broadcast, export, backtest)
    # ==========================================================================
    st.markdown("---")
    st.markdown("### ⚡ Action Center")
    
    # Auto-broadcast logic
    if tg_on and tg_token and tg_chat:
        if last['buy_signal'] or last['sell_signal']:
            sid = make_signal_id(last)
            last_sid = st.session_state.get("last_signal_id", "")
            
            if sid != last_sid:
                # Journal to database
                if db_conn:
                    journal_signal(db_conn, {
                        "ts": str(last['timestamp']),
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "direction": "LONG" if last['is_bull'] else "SHORT",
                        "entry": float(last['entry_price']) if not pd.isna(last['entry_price']) else None,
                        "stop": float(last['entry_stop']) if not pd.isna(last['entry_stop']) else None,
                        "tp1": float(last['tp1']) if not pd.isna(last['tp1']) else None,
                        "tp2": float(last['tp2']) if not pd.isna(last['tp2']) else None,
                        "tp3": float(last['tp3']) if not pd.isna(last['tp3']) else None,
                        "adx": float(last['adx']),
                        "rvol": float(last['rvol']),
                        "notes": "Auto-broadcast"
                    })
                
                # Send to Telegram
                text = make_signal_text(last, symbol, timeframe)
                sent = send_telegram_msg(
                    tg_token,
                    tg_chat,
                    text,
                    cooldown_s=int(telegram_cooldown_s)
                )
                
                if sent:
                    st.success("✅ Signal auto-broadcasted to Telegram")
                    st.session_state["last_signal_id"] = sid
                else:
                    st.info("ℹ️ Telegram not sent (cooldown active or rate limited)")
    
    # Manual action buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🔥 Broadcast Signal (Manual)", use_container_width=True):
            text = make_signal_text(last, symbol, timeframe)
            
            # Journal
            if db_conn:
                journal_signal(db_conn, {
                    "ts": str(last['timestamp']),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "direction": "LONG" if last['is_bull'] else "SHORT",
                    "entry": float(last['entry_price']) if not pd.isna(last['entry_price']) else None,
                    "stop": float(last['entry_stop']) if not pd.isna(last['entry_stop']) else None,
                    "tp1": float(last['tp1']) if not pd.isna(last['tp1']) else None,
                    "tp2": float(last['tp2']) if not pd.isna(last['tp2']) else None,
                    "tp3": float(last['tp3']) if not pd.isna(last['tp3']) else None,
                    "adx": float(last['adx']),
                    "rvol": float(last['rvol']),
                    "notes": "Manual broadcast"
                })
            
            # Send
            if send_telegram_msg(tg_token, tg_chat, text, cooldown_s=int(telegram_cooldown_s)):
                st.success("✅ Manual broadcast sent successfully")
            else:
                st.error("❌ Telegram failed (check credentials or cooldown)")
    
    with col_btn2:
        if st.button("📥 Export Signal CSV", use_container_width=True):
            payload = {
                "timestamp": str(last['timestamp']),
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": "LONG" if last['is_bull'] else "SHORT",
                "entry": float(last['entry_price']) if not pd.isna(last['entry_price']) else None,
                "stop": float(last['entry_stop']) if not pd.isna(last['entry_stop']) else None,
                "tp1": float(last['tp1']) if not pd.isna(last['tp1']) else None,
                "tp2": float(last['tp2']) if not pd.isna(last['tp2']) else None,
                "tp3": float(last['tp3']) if not pd.isna(last['tp3']) else None,
                "adx": float(last['adx']),
                "rvol": float(last['rvol']),
                "rsi": float(last['rsi']),
                "mfi": float(last['mfi'])
            }
            
            csv_data = pd.DataFrame([payload]).to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv_data,
                file_name=f"titan_signal_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_btn3:
        if st.button("🧮 Run Backtest", use_container_width=True):
            with st.spinner("Running backtest simulation..."):
                bt_results = backtest_engine(
                    df,
                    risk_pct=float(backtest_risk),
                    starting_balance=10000.0
                )
                
                st.success("✅ Backtest Complete")
                
                # Display results
                st.markdown("#### Backtest Results")
                res_c1, res_c2, res_c3 = st.columns(3)
                
                res_c1.metric(
                    "Final Balance",
                    f"${bt_results['final_balance']:.2f}"
                )
                res_c2.metric(
                    "Total P/L",
                    f"${bt_results['profit']:.2f}",
                    delta=f"{(bt_results['profit']/10000)*100:.1f}%"
                )
                res_c3.metric(
                    "Win Rate",
                    f"{bt_results['win_rate']:.1f}%",
                    delta=f"{bt_results['num_trades']} trades"
                )
                
                # Show recent trades
                if bt_results['num_trades'] > 0:
                    st.markdown("##### Recent Trades")
                    trades_df = pd.DataFrame(bt_results['trades'][-10:])
                    st.dataframe(trades_df, use_container_width=True)


with col_right:
    st.subheader("📊 Manual Charting — TradingView")
    
    tv_symbol = f"BINANCE:{symbol}"
    tv_html = tradingview_widget(tv_symbol, timeframe)
    components.html(tv_html, height=680)
    
    # ==========================================================================
    # SIGNAL JOURNAL
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 📝 Signal Journal (Last 50)")
    
    if persist and db_conn:
        try:
            cur = db_conn.cursor()
            cur.execute("""
                SELECT 
                    ts, symbol, timeframe, direction, entry, stop,
                    tp1, tp2, tp3, adx, rvol, notes
                FROM signals
                ORDER BY id DESC
                LIMIT 50
            """)
            rows = cur.fetchall()
            
            if rows:
                cols = [
                    "Timestamp", "Symbol", "TF", "Direction", "Entry", "Stop",
                    "TP1", "TP2", "TP3", "ADX", "RVOL", "Notes"
                ]
                journal_df = pd.DataFrame(rows, columns=cols)
                st.dataframe(journal_df, use_container_width=True, hide_index=True)
            else:
                st.info("📭 No signals journaled yet. Signals will appear here when generated.")
        except Exception as e:
            st.error(f"❌ Journal read error: {str(e)}")
    else:
        st.info("💾 Signal persistence is disabled. Enable in sidebar to journal signals.")


# =============================================================================
# FOOTER & DOCUMENTATION
# =============================================================================
st.markdown("---")
st.markdown("""
### 📚 Documentation & Safety Notes

**Key Features:**
- ✅ **Non-repainting signals**: Entry, stop, and TPs are frozen at signal bar
- ✅ **State-locked execution**: No look-ahead bias or recalculation
- ✅ **Production-ready**: Proper error handling, rate limiting, and resource management
- ✅ **Multi-layer filters**: ADX regime + RVOL + HMA + RSI confluence

**Important Warnings:**
- ⚠️ **DO NOT execute live orders from Streamlit** - Use separate execution service
- ⚠️ **Past performance ≠ future results** - Backtest with realistic assumptions
- ⚠️ **Paper trade first** - Verify signals in demo environment before live trading
- ⚠️ **Risk management is critical** - Never risk more than 1-2% per trade

**Architecture Notes:**
- TradingView widget is for manual analysis only
- Execution signals are drawn in Plotly chart (safe, deterministic)
- Database journals all signals for audit trail
- Telegram broadcasts are rate-limited to prevent spam

**For Production Deployment:**
1. Build separate execution microservice (Python/Node.js)
2. Read signals from SQLite database
3. Implement proper position sizing and order management
4. Add exchange API integration (Binance, FTX, etc.)
5. Set up monitoring and alerting (Sentry, DataDog)
6. Use paper trading environment for 30+ days before live

**Support:**
- Report issues on GitHub
- Join community Discord for strategy discussion
- Read full documentation at: [Your Docs URL]

---
**Version:** 2.0.0 (Production-Ready)  
**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}  
**License:** MIT
""")


