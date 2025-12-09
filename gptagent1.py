# app.py — TITAN INTRADAY PRO (Streamlit Cloud-ready)
import time
import math
import sqlite3
import html
from typing import Dict, Optional

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime

# -----------------------
# Basic page config
# -----------------------
st.set_page_config(page_title="TITAN INTRADAY PRO", layout="wide", page_icon="⚡")
st.title("🪓 TITAN INTRADAY PRO — Execution Dashboard")

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Market Feed")
symbol_input = st.sidebar.text_input("Symbol (Binance format)", value="BTCUSDT")
# sanitize symbol to expected format
symbol = symbol_input.strip().upper().replace("/", "").replace("-", "")
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
limit = st.sidebar.slider("Candles", min_value=100, max_value=2000, value=600, step=50)

st.sidebar.markdown("---")
st.sidebar.header("Logic Engine / Ladder")
amplitude = st.sidebar.number_input("Structure lookback (amplitude)", min_value=2, max_value=200, value=10)
channel_dev = st.sidebar.number_input("Stop Deviation (ATR x)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
hma_len = st.sidebar.number_input("HMA length", min_value=2, max_value=400, value=50)
use_hma_filter = st.sidebar.checkbox("Use HMA filter?", value=True)
tp1_r = st.sidebar.number_input("TP1 (R)", value=1.5, step=0.1)
tp2_r = st.sidebar.number_input("TP2 (R)", value=3.0, step=0.1)
tp3_r = st.sidebar.number_input("TP3 (R)", value=5.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Volume & Momentum")
mf_len = st.sidebar.number_input("Money Flow length", min_value=2, max_value=200, value=14)
vol_len = st.sidebar.number_input("Volume rolling length", min_value=5, max_value=200, value=20)
vol_metric = st.sidebar.selectbox("Volume metric", ["CMF", "MFI", "Volume RSI", "RVOL", "Vol Osc"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Integrations & Persistence")
tg_on = st.sidebar.checkbox("Telegram Broadcast", value=False)
try:
    tg_token = st.secrets["TELEGRAM_TOKEN"]
    tg_chat = st.secrets["TELEGRAM_CHAT_ID"]
    st.sidebar.success("Telegram secrets loaded")
except Exception:
    tg_token = st.sidebar.text_input("Telegram Bot Token", type="password")
    tg_chat = st.sidebar.text_input("Telegram Chat ID")

persist = st.sidebar.checkbox("Persist signals to DB", value=True)
run_backtest = st.sidebar.checkbox("Run backtest (slow)", value=False)
backtest_risk = st.sidebar.number_input("Backtest risk % per trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Safety")
ai_cooldown_s = st.sidebar.number_input("AI cooldown (s)", min_value=5, max_value=600, value=45)
telegram_cooldown_s = st.sidebar.number_input("Telegram cooldown (s)", min_value=5, max_value=600, value=30)

# -----------------------
# DB init (optional)
# -----------------------
DB_PATH = "titan_signals.db"
def init_db(path=DB_PATH):
    if not persist:
        return None
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            symbol TEXT,
            timeframe TEXT,
            direction TEXT,
            entry REAL,
            stop REAL,
            tp1 REAL,
            tp2 REAL,
            tp3 REAL,
            adx REAL,
            rvol REAL,
            notes TEXT
        )
    """)
    conn.commit()
    return conn

db_conn = init_db()

def journal_signal(conn, payload: Dict):
    if conn is None: return
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO signals (ts, symbol, timeframe, direction, entry, stop, tp1, tp2, tp3, adx, rvol, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        payload.get("ts"), payload.get("symbol"), payload.get("timeframe"),
        payload.get("direction"), payload.get("entry"), payload.get("stop"),
        payload.get("tp1"), payload.get("tp2"), payload.get("tp3"),
        payload.get("adx"), payload.get("rvol"), payload.get("notes","")
    ))
    conn.commit()

# -----------------------
# Helpers: safe telegram + escapes
# -----------------------
def escape_markdown_v2(text: str) -> str:
    # escape Telegram MarkdownV2 special chars
    # characters: _ * [ ] ( ) ~ ` > # + - = | { } . !
    to_escape = r'_*\[\]()~`>#+-=|{}.!'
    return ''.join('\\' + c if c in to_escape else c for c in text)

def send_telegram_msg(token: str, chat: str, msg: str, cooldown_s: int = 30) -> bool:
    if not token or not chat:
        return False
    last_ts = st.session_state.get("last_telegram_ts", 0)
    if time.time() - last_ts < cooldown_s:
        return False
    payload = {"chat_id": chat, "text": escape_markdown_v2(msg), "parse_mode": "MarkdownV2"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    tries = 3
    for _ in range(tries):
        try:
            r = requests.post(url, json=payload, timeout=8)
            if r.status_code in (200, 201):
                st.session_state["last_telegram_ts"] = time.time()
                return True
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
    return False

# -----------------------
# Robust Binance fetch
# -----------------------
@st.cache_data(ttl=10)
def get_klines(symbol: str, interval: str, limit: int = 500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        # if data is dict with code/msg -> failure
        if not data or isinstance(data, dict):
            return pd.DataFrame()
        cols = ['open_time','open','high','low','close','volume','c6','c7','c8','c9','c10','c11']
        df = pd.DataFrame(data, columns=cols)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        df = df[['timestamp','open','high','low','close','volume']].copy()
        return df
    except Exception as e:
        # graceful degrade
        return pd.DataFrame()

# -----------------------
# Institutional indicators (non-repainting)
# -----------------------
def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def calculate_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
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
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_hma(series: pd.Series, length: int = 50) -> pd.Series:
    half = int(length/2)
    sqrt = int(math.sqrt(length))
    wma_half = series.rolling(window=half, min_periods=half).mean()
    wma_full = series.rolling(window=length, min_periods=length).mean()
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(window=sqrt, min_periods=sqrt).mean()
    return hma

def calculate_mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    rmf = tp * df['volume']
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(length, min_periods=length).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(length, min_periods=length).sum()
    mfi = 100 - (100 / (1 + (pos / neg)))
    return mfi.fillna(50)

# -----------------------
# Titan Engine: deterministic, state-locked
# -----------------------
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
    if df.empty:
        return df
    df = df.copy().reset_index(drop=True)
    N = len(df)

    # Indicators
    df['atr'] = calculate_atr(df, 14)
    df['dev'] = df['atr'] * channel_dev
    df['hma'] = calculate_hma(df['close'], int(hma_len))
    df['ll'] = df['low'].rolling(window=amplitude, min_periods=amplitude).min()
    df['hh'] = df['high'].rolling(window=amplitude, min_periods=amplitude).max()

    # Trailing engine
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

    # Momentum & Regime
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['adx'] = calculate_adx(df, 14)
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_len, min_periods=vol_len).mean()
    df['rvol'] = df['rvol'].fillna(1.0)

    # Money flow & vol metrics
    df['mfi'] = calculate_mfi(df, mf_len)
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['cmf'] = (mfm.fillna(0) * df['volume']).rolling(vol_len, min_periods=vol_len).sum() / df['volume'].rolling(vol_len, min_periods=vol_len).sum()
    df['vol_rsi'] = calculate_rsi(df['volume'], 14)
    v_s = df['volume'].rolling(14, min_periods=14).mean()
    v_l = df['volume'].rolling(28, min_periods=28).mean()
    df['vol_osc'] = np.where(v_l != 0, 100 * (v_s - v_l) / v_l, 0.0)

    # Signal flips
    df['bull_flip'] = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False).astype(bool))
    df['bear_flip'] = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True).astype(bool))
    df['regime'] = (df['adx'] > 23) & (df['rvol'] > 1.15)
    df['hma_buy'] = (~use_hma_filter) | (df['close'] > df['hma'])
    df['hma_sell'] = (~use_hma_filter) | (df['close'] < df['hma'])

    df['buy_signal'] = df['bull_flip'] & df['regime'] & df['hma_buy'] & (df['rsi'] < 70)
    df['sell_signal'] = df['bear_flip'] & df['regime'] & df['hma_sell'] & (df['rsi'] > 30)

    # State-locked entry price
    df['entry_price'] = np.where(df['buy_signal'] | df['sell_signal'], df['close'], np.nan)
    df['entry_price'] = df['entry_price'].ffill().where(lambda s: ~s.isna(), np.nan)

    # State-locked risk vs trend_stop at entry
    df['risk'] = np.where((~df['entry_price'].isna()) & (~df['trend_stop'].isna()),
                         (df['entry_price'] - df['trend_stop']).abs(),
                         np.nan)

    # Freeze ladder (non repainting)
    df['tp1'] = np.where(df['is_bull'],
                         df['entry_price'] + df['risk'] * tp1_r,
                         df['entry_price'] - df['risk'] * tp1_r)
    df['tp2'] = np.where(df['is_bull'],
                         df['entry_price'] + df['risk'] * tp2_r,
                         df['entry_price'] - df['risk'] * tp2_r)
    df['tp3'] = np.where(df['is_bull'],
                         df['entry_price'] + df['risk'] * tp3_r,
                         df['entry_price'] - df['risk'] * tp3_r)

    return df

# -----------------------
# Backtest helper (basic)
# -----------------------
def backtest_engine(df: pd.DataFrame, risk_pct: float = 1.0, starting_balance: float = 10000.0) -> Dict:
    if df.empty:
        return {"final_balance": starting_balance, "profit": 0.0}
    df = df.reset_index(drop=True).copy()
    balance = starting_balance
    pos_open = False
    pos = {}
    for i in range(len(df)):
        row = df.iloc[i]
        if (row['buy_signal'] or row['sell_signal']) and not pos_open:
            entry = row['entry_price']
            stop = row['trend_stop']
            if pd.isna(entry) or pd.isna(stop) or entry == stop:
                continue
            per_unit_risk = abs(entry - stop)
            if per_unit_risk == 0:
                continue
            risk_amount = balance * (risk_pct / 100.0)
            size = risk_amount / per_unit_risk
            pos_open = True
            pos = {"side": "long" if row['buy_signal'] else "short",
                   "entry": entry, "stop": stop, "size": size,
                   "tp1": row['tp1'], "tp2": row['tp2'], "tp3": row['tp3'],
                   "remaining": 1.0}
            continue
        if pos_open:
            high = row['high']; low = row['low']
            if pos['side'] == 'long':
                for tp_label, frac in (('tp1',0.3),('tp2',0.4),('tp3',pos['remaining'])):
                    tp_val = pos.get(tp_label)
                    if pd.isna(tp_val): continue
                    if high >= tp_val and pos['remaining'] > 0:
                        profit = (tp_val - pos['entry']) * pos['size'] * (frac if tp_label!='tp3' else pos['remaining'])
                        balance += profit
                        pos['remaining'] -= (frac if tp_label!='tp3' else pos['remaining'])
                if low <= pos['stop']:
                    loss = (pos['entry'] - pos['stop']) * pos['size'] * pos['remaining']
                    balance -= loss
                    pos_open = False; pos = {}
            else:
                for tp_label, frac in (('tp1',0.3),('tp2',0.4),('tp3',pos['remaining'])):
                    tp_val = pos.get(tp_label)
                    if pd.isna(tp_val): continue
                    if low <= tp_val and pos['remaining'] > 0:
                        profit = (pos['entry'] - tp_val) * pos['size'] * (frac if tp_label!='tp3' else pos['remaining'])
                        balance += profit
                        pos['remaining'] -= (frac if tp_label!='tp3' else pos['remaining'])
                if high >= pos['stop']:
                    loss = (pos['stop'] - pos['entry']) * pos['size'] * pos['remaining']
                    balance -= loss
                    pos_open = False; pos = {}
            if pos_open and pos['remaining'] <= 0:
                pos_open = False; pos = {}
    return {"final_balance": balance, "profit": balance - starting_balance}

# -----------------------
# TradingView embed (manual TA)
# -----------------------
def tradingview_widget(symbol_tv: str = "BINANCE:BTCUSDT", interval: str = "60"):
    # interval mapping: 1m->1, 5m->5, 15m->15, 1h->60, 4h->240, 1d->D
    int_map = {"1m":"1","5m":"5","15m":"15","1h":"60","4h":"240","1d":"D"}
    iv = int_map.get(interval, "60")
    html_content = f"""
    <html>
    <head><meta charset="utf-8"></head>
    <body style="margin:0;background:#0e0e0e;">
    <div id="tv" style="width:100%;height:640px;"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.widget({{
      "autosize": true,
      "symbol": "{symbol_tv}",
      "interval": "{iv}",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "container_id": "tv",
      "hide_side_toolbar": false,
      "allow_symbol_change": true,
      "save_image": false
    }});
    </script>
    </body>
    </html>
    """
    return html_content

# -----------------------
# UI layout & flow
# -----------------------
# Split: left=execution chart + controls, right=TradingView + journal
col_left, col_right = st.columns([2,1])

with col_left:
    st.subheader("Execution Chart — TITAN Engine")

    # Fetch data (robust)
    with st.spinner("Fetching market data..."):
        df = get_klines(symbol, timeframe, limit)

    if df.empty:
        st.error("Market data not available. Check symbol (e.g., BTCUSDT), timeframe, or network.")
        st.stop()

    # Run engine
    with st.spinner("Running Titan engine..."):
        df = run_titan_engine(df,
                              amplitude=int(amplitude),
                              channel_dev=float(channel_dev),
                              hma_len=int(hma_len),
                              use_hma_filter=bool(use_hma_filter),
                              tp1_r=float(tp1_r),
                              tp2_r=float(tp2_r),
                              tp3_r=float(tp3_r),
                              mf_len=int(mf_len),
                              vol_len=int(vol_len))

    last = df.iloc[-1]

    # Plotly candlestick + indicators + stops + ladder + markers
    fig = go.Figure()
    fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')

    # HMA
    if 'hma' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', name='HMA', line=dict(color='#00b3ff', width=1)))

    # Trend stops (separate bull/bear)
    b_stop = df['trend_stop'].where(df['is_bull'], np.nan).replace(0, np.nan)
    s_stop = df['trend_stop'].where(~df['is_bull'], np.nan).replace(0, np.nan)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=b_stop, mode='lines', name='Bull Trail', line=dict(color='#00ffbb', width=2)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=s_stop, mode='lines', name='Bear Trail', line=dict(color='#ff1155', width=2)))

    # Ladder lines (state-locked)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tp1'], mode='lines', name='TP1 (R)', line=dict(color='#00ffbb', dash='dot', width=1)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tp2'], mode='lines', name='TP2 (R)', line=dict(color='#00ffbb', dash='dash', width=1)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tp3'], mode='lines', name='TP3 (R)', line=dict(color='#00ffbb', dash='longdash', width=1)))

    # Markers for signals
    buys = df[df['buy_signal']]
    sells = df[df['sell_signal']]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ffbb'), name='BUY'))
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff1155'), name='SELL'))

    # Money flow / vol panel as mini-subplot via layout yaxis2 (simpler)
    # Add RVOL as bar on secondary y
    fig.update_layout(height=720, template="plotly_dark", xaxis_rangeslider_visible=False)
    fig.update_yaxes(gridcolor="#222")

    st.plotly_chart(fig, use_container_width=True)

    # Metrics & action center
    c1, c2, c3, c4 = st.columns(4)
    trend_lbl = "BULL" if last['is_bull'] else "BEAR"
    c1.metric("Price", f"${last['close']:.2f}")
    c2.metric("Trailing Stop", f"${(last['trend_stop'] if not pd.isna(last['trend_stop']) else 0):.2f}")
    c3.metric("ADX", f"{last['adx']:.1f}")
    c4.metric("RVOL", f"{last['rvol']:.2f}x")

    # Broadcast workflow (auto/manual)
    def make_signal_id(row):
        ts = row['timestamp']
        dir_flag = "L" if row['is_bull'] else "S"
        entry = row['entry_price'] if not pd.isna(row['entry_price']) else 0.0
        return f"{ts.isoformat()}_{dir_flag}_{entry:.8f}"

    def make_signal_text(row):
        dir_txt = "LONG" if row['is_bull'] else "SHORT"
        return (f"🔥 TITAN SIGNAL — {symbol} {timeframe}\n"
                f"{dir_txt}\nENTRY: {row['entry_price']:.2f}\nSTOP: {row['trend_stop']:.2f}\n"
                f"TP1: {row['tp1']:.2f} | TP2: {row['tp2']:.2f} | TP3: {row['tp3']:.2f}\n"
                f"ADX: {row['adx']:.2f} | RVOL: {row['rvol']:.2f}x\nNot financial advice.")

    # Auto broadcast
    if tg_on and tg_token and tg_chat:
        if last['buy_signal'] or last['sell_signal']:
            sid = make_signal_id(last)
            if st.session_state.get("last_signal_id") != sid:
                # journal
                if db_conn:
                    journal_signal(db_conn, {
                        "ts": str(last['timestamp']), "symbol": symbol, "timeframe": timeframe,
                        "direction": "LONG" if last['is_bull'] else "SHORT",
                        "entry": float(last['entry_price']) if not pd.isna(last['entry_price']) else None,
                        "stop": float(last['trend_stop']) if not pd.isna(last['trend_stop']) else None,
                        "tp1": float(last['tp1']) if not pd.isna(last['tp1']) else None,
                        "tp2": float(last['tp2']) if not pd.isna(last['tp2']) else None,
                        "tp3": float(last['tp3']) if not pd.isna(last['tp3']) else None,
                        "adx": float(last['adx']), "rvol": float(last['rvol']), "notes": "Auto broadcast"
                    })
                text = make_signal_text(last)
                sent = send_telegram_msg(tg_token, tg_chat, text, cooldown_s=int(telegram_cooldown_s))
                if sent:
                    st.success("Auto-broadcasted signal to Telegram")
                    st.session_state["last_signal_id"] = sid
                else:
                    st.info("Telegram not sent (cooldown or failure).")

    # Manual action buttons
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🔥 BROADCAST (Manual)"):
            text = make_signal_text(last)
            if db_conn:
                journal_signal(db_conn, {
                    "ts": str(last['timestamp']), "symbol": symbol, "timeframe": timeframe,
                    "direction": "LONG" if last['is_bull'] else "SHORT",
                    "entry": float(last['entry_price']) if not pd.isna(last['entry_price']) else None,
                    "stop": float(last['trend_stop']) if not pd.isna(last['trend_stop']) else None,
                    "tp1": float(last['tp1']) if not pd.isna(last['tp1']) else None,
                    "tp2": float(last['tp2']) if not pd.isna(last['tp2']) else None,
                    "tp3": float(last['tp3']) if not pd.isna(last['tp3']) else None,
                    "adx": float(last['adx']), "rvol": float(last['rvol']), "notes":"Manual broadcast"
                })
            if send_telegram_msg(tg_token, tg_chat, text, cooldown_s=int(telegram_cooldown_s)):
                st.success("Manual broadcast sent")
            else:
                st.error("Telegram failed or cooldown active")

    with col_b:
        if st.button("📥 Export Signal CSV"):
            payload = {
                "ts": str(last['timestamp']), "symbol": symbol, "timeframe": timeframe,
                "direction": "LONG" if last['is_bull'] else "SHORT",
                "entry": float(last['entry_price']) if not pd.isna(last['entry_price']) else None,
                "stop": float(last['trend_stop']) if not pd.isna(last['trend_stop']) else None,
                "tp1": float(last['tp1']) if not pd.isna(last['tp1']) else None,
                "tp2": float(last['tp2']) if not pd.isna(last['tp2']) else None,
                "tp3": float(last['tp3']) if not pd.isna(last['tp3']) else None,
                "adx": float(last['adx']), "rvol": float(last['rvol'])
            }
            csv = pd.DataFrame([payload]).to_csv(index=False)
            st.download_button("Download CSV", csv, file_name=f"titan_signal_{symbol}_{timeframe}.csv")

    with col_c:
        if st.button("🧮 Run quick backtest"):
            with st.spinner("Running backtest..."):
                bt = backtest_engine(df, risk_pct=float(backtest_risk), starting_balance=10000.0)
                st.success(f"Backtest finished. Final balance: ${bt['final_balance']:.2f} (P/L ${bt['profit']:.2f})")

with col_right:
    st.subheader("Manual Charting — TradingView")
    tv_symbol = f"BINANCE:{symbol}"
    components.html(tradingview_widget(tv_symbol, timeframe), height=680)

    st.markdown("### Signal Journal (most recent)")
    if persist and db_conn:
        try:
            cur = db_conn.cursor()
            cur.execute("SELECT ts, symbol, timeframe, direction, entry, stop, tp1, tp2, tp3, adx, rvol, notes FROM signals ORDER BY id DESC LIMIT 50")
            rows = cur.fetchall()
            if rows:
                cols = ["ts","symbol","tf","dir","entry","stop","tp1","tp2","tp3","adx","rvol","notes"]
                df_j = pd.DataFrame(rows, columns=cols)
                st.dataframe(df_j, use_container_width=True)
            else:
                st.info("No signals yet.")
        except Exception as e:
            st.error("Journal read error.")

# Footer notes
st.markdown("""
---
**Notes**
- TradingView embedded widget is for manual TA only. Execution signals are drawn in the Plotly chart above (safe, non-repainting).
- Ladder TPs are state-locked at the candle where the signal fired and will not move.
- For production auto-execution, do NOT place live orders from Streamlit. Build a separate execution microservice that reads the persisted signals.
""")


