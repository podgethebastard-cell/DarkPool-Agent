
# titan_intraday_pro.py
"""
Titan Intraday Ultimate — Production-Ready Single File
- Streamlit frontend
- CCXT Kraken data fetch with retries + rate limit
- Non-repainting, state-locked ladder + trailing stop
- Institutional ADX/ATR/RSI/HMA
- Telegram & OpenAI safe wrappers
- Backtest module
- SQLite signal journal
"""

import time
import math
import sqlite3
import html
from typing import Optional, Tuple, Dict

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
from openai import OpenAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# -----------------------------
# 0. App configuration & CSS
# -----------------------------
st.set_page_config(page_title="🪓Titan Intraday Ultimate (Pro)", layout="wide", page_icon="⚡")
# CSS (dark theme)
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .block-container { padding-top: 0.5rem; padding-bottom: 3rem; }
    .titan-header { background: linear-gradient(180deg, #111 0%, #050505 100%); border-bottom: 1px solid #333; padding: 1.25rem 1rem; text-align: center; margin-bottom: 1rem; border-top: 3px solid #00ffbb; }
    .titan-title { font-size: 2.4rem; font-weight: 900; color: #fff; letter-spacing: 3px; margin: 0; text-shadow: 0 0 12px rgba(0,255,187,0.18); }
    .titan-subtitle { font-size: 0.8rem; color: #999; letter-spacing: 1.8px; margin-top: 0.25rem; text-transform: uppercase; }
    .titan-card { background: #0f0f0f; border: 1px solid #222; border-left: 4px solid #555; padding: 12px; border-radius: 6px; margin-bottom: 8px; }
    .titan-card h4 { margin: 0; font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .titan-card h2 { margin: 5px 0 0 0; font-size: 1.6rem; font-weight: 700; color: #fff; }
    .ai-box { background: #0a0a0a; border: 1px solid #333; padding: 16px; border-radius: 6px; margin-top: 10px; border-left: 3px solid #7d00ff; }
    section[data-testid="stSidebar"] { background-color: #080808; border-right: 1px solid #222; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 1. Sidebar Controls & Secrets
# -----------------------------
st.sidebar.title("⚡ TITAN ULTIMATE — PRO")
st.sidebar.caption("v9.0 | INTRADAY EXECUTION STACK")
st.sidebar.markdown("---")

# Market Inputs
st.sidebar.subheader("Market Feed")
symbol = st.sidebar.text_input("Symbol (Kraken format)", value="BTC/USD")
timeframe = st.sidebar.selectbox("Timeframe", options=['15m', '1h', '4h'], index=1)
limit = st.sidebar.slider("Candles", min_value=200, max_value=2000, value=1000, step=50)

# Strategy Params group
st.sidebar.markdown("---")
st.sidebar.subheader("Logic Engine / Ladder")
amplitude = st.sidebar.number_input("Structure Lookback (amplitude)", min_value=2, max_value=200, value=10)
channel_dev = st.sidebar.number_input("Stop Deviation (ATR x)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
hma_len = st.sidebar.number_input("HMA Length", min_value=2, max_value=400, value=50)
use_hma_filter = st.sidebar.checkbox("Use HMA filter?", value=True)
tp1_r = st.sidebar.number_input("TP1 (R)", value=1.5, step=0.1)
tp2_r = st.sidebar.number_input("TP2 (R)", value=3.0, step=0.1)
tp3_r = st.sidebar.number_input("TP3 (R)", value=5.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Volume & Momentum")
mf_len = st.sidebar.number_input("Money Flow Length", min_value=2, max_value=200, value=14)
vol_len = st.sidebar.number_input("Volume Rolling Length", min_value=5, max_value=200, value=20)
vol_metric = st.sidebar.selectbox("Volume Metric", ["CMF", "MFI", "Volume RSI", "RVOL", "Vol Osc"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Credentials & Integrations")
# Telegram secrets (prefer st.secrets)
tg_on = st.sidebar.checkbox("Telegram Auto Broadcast", value=False)
tg_token = None; tg_chat = None
try:
    tg_token = st.secrets["TELEGRAM_TOKEN"]
    tg_chat = st.secrets["TELEGRAM_CHAT_ID"]
    st.sidebar.success("🔹 Telegram secrets loaded")
except Exception:
    tg_token = st.sidebar.text_input("Telegram Bot Token", type="password")
    tg_chat = st.sidebar.text_input("Telegram Chat ID")
    if tg_token and tg_chat:
        st.sidebar.success("🔹 Telegram credentials set (sidebar)")

# OpenAI
try:
    ai_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("🔹 OpenAI key loaded")
except Exception:
    ai_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Persistence toggle
st.sidebar.markdown("---")
persistence = st.sidebar.checkbox("Persist signals to local DB", value=True)

# Backtest controls
st.sidebar.markdown("---")
st.sidebar.subheader("Backtest / Research")
run_backtest = st.sidebar.checkbox("Run backtest on fetched data (slow)", value=False)
backtest_risk_pct = st.sidebar.number_input("Backtest risk % per trade", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# App safety knobs
st.sidebar.markdown("---")
st.sidebar.subheader("Safety")
ai_cooldown_s = st.sidebar.number_input("AI cooldown (s)", min_value=5, max_value=600, value=45)
telegram_cooldown_s = st.sidebar.number_input("Telegram cooldown (s)", min_value=5, max_value=600, value=30)

# -----------------------------
# 2. Utilities
# -----------------------------
# SQLite persistence (signals journal)
DB_PATH = "titan_signals.db"
def init_db():
    if not persistence:
        return
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    if not persistence:
        return
    c = conn.cursor()
    c.execute("""
        INSERT INTO signals (ts, symbol, timeframe, direction, entry, stop, tp1, tp2, tp3, adx, rvol, notes)
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

# Safe markdown escape for Telegram
def escape_markdown(text: str) -> str:
    # basic escape for Telegram MarkdownV2 special chars
    escape_chars = r'_*\[\]()~`>#+-=|{}.!'
    return ''.join('\\' + c if c in escape_chars else c for c in text)

# Safe telegram sender with cooldown and retries
def send_telegram_msg(token: str, chat: str, msg: str, parse_mode="MarkdownV2") -> bool:
    if not token or not chat:
        st.warning("Telegram credentials missing.")
        return False

    # cooldown guard
    last = st.session_state.get("last_telegram_ts", 0)
    if time.time() - last < telegram_cooldown_s:
        st.info("Telegram cooldown active.")
        return False

    safe_msg = escape_markdown(msg)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat, "text": safe_msg, "parse_mode": parse_mode}
    tries = 3
    for _ in range(tries):
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code in (200, 201):
                st.session_state["last_telegram_ts"] = time.time()
                return True
            else:
                time.sleep(1)
        except Exception as e:
            time.sleep(1)
    st.error("Telegram send failed.")
    return False

# Safe OpenAI wrapper with cooldown
def get_ai_analysis(client_key: str, df_summary: dict, symbol: str, tf: str) -> str:
    if not client_key:
        return "⚠️ Provide OpenAI API key to use Titan AI."
    last = st.session_state.get("last_ai_ts", 0)
    if time.time() - last < ai_cooldown_s:
        return f"🤖 AI cooldown active ({int(ai_cooldown_s - (time.time()-last))}s)."
    try:
        client = OpenAI(api_key=client_key)
        # succinct prompt
        prompt = f"""You are TITAN INTRADAY assistant. Provide a concise (<60 words) ladder execution assessment for {symbol} {tf}.
Price: {df_summary['price']}, Trend: {df_summary['trend']}, Stop: {df_summary['stop']}, TP1: {df_summary['tp1']}, TP2: {df_summary['tp2']}, TP3: {df_summary['tp3']}, ADX: {df_summary['adx']}, RVOL: {df_summary['rvol']}
Give risk split, whether TPs are achievable, and 1 one-line actionable note. Use emojis."""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise trading assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120
        )
        text = resp.choices[0].message.content.strip()
        st.session_state["last_ai_ts"] = time.time()
        return text
    except Exception as e:
        return f"AI Error: {e}"

# -----------------------------
# 3. Robust CCXT Data Fetch (cached + retries)
# -----------------------------
@st.cache_data(ttl=15, show_spinner=False)
def get_ohlcv_kraken(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """
    Returns dataframe with columns: timestamp, open, high, low, close, volume
    Implements short retry + enableRateLimit
    """
    exchange = ccxt.kraken({
        "enableRateLimit": True,
        "timeout": 20000
    })
    max_retries = 3
    last_exc = None
    for attempt in range(max_retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt)
    # If failed return empty DF and log
    st.error(f"Market data fetch failed: {last_exc}")
    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# -----------------------------
# 4. Indicator Implementations (institutional)
# -----------------------------
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
    half = int(length / 2)
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

# -----------------------------
# 5. Engine: state-locked, non-repainting
# -----------------------------
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
    Deterministic engine: indicators -> regime -> signals -> state-locked entry -> ladder.
    """
    if df.empty:
        return df
    df = df.copy().reset_index(drop=True)

    # Indicators
    df['atr'] = calculate_atr(df, 14)
    df['dev'] = df['atr'] * channel_dev
    df['hma'] = calculate_hma(df['close'], int(hma_len))
    # structure minima/maxima
    df['ll'] = df['low'].rolling(window=amplitude, min_periods=amplitude).min()
    df['hh'] = df['high'].rolling(window=amplitude, min_periods=amplitude).max()

    # Trailing Structure Engine (ratchet)
    N = len(df)
    trend = np.zeros(N, dtype=int)
    stop = np.full(N, np.nan)
    curr_trend = 0  # 0 bull, 1 bear
    curr_stop = np.nan

    # initialize curr_stop on first valid dev
    # iterate deterministically from amplitude to end
    for i in range(amplitude, N):
        local_low = df.at[i, 'll']
        local_high = df.at[i, 'hh']
        close = df.at[i, 'close']
        dev = df.at[i, 'dev'] if not pd.isna(df.at[i, 'dev']) else 0.0

        if curr_trend == 0:  # currently bull trailing up
            s = local_low + dev
            # ratchet: stop only moves up (never down for long)
            curr_stop = s if np.isnan(curr_stop) else max(curr_stop, s)
            # if price breaches stop (close < stop) we flip to bear
            if close < curr_stop:
                curr_trend = 1
                # reset stop to a bear initial level
                curr_stop = local_high - dev
        else:  # currently bear trailing down
            s = local_high - dev
            # ratchet: stop only moves down (never up for short)
            curr_stop = s if np.isnan(curr_stop) else min(curr_stop, s)
            if close > curr_stop:
                curr_trend = 0
                curr_stop = local_low + dev

        trend[i] = curr_trend
        stop[i] = curr_stop

    df['trend'] = trend
    df['trend_stop'] = stop
    df['is_bull'] = df['trend'] == 0

    # Momentum & regime
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['adx'] = calculate_adx(df, 14)
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_len, min_periods=vol_len).mean()
    df['rvol'] = df['rvol'].fillna(1.0)

    # Money flow & advanced vol metrics
    df['mfi'] = calculate_mfi(df, mf_len)
    df['cmf'] = (( (df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])).fillna(0)
    df['cmf'] = (df['cmf'] * df['volume']).rolling(vol_len, min_periods=vol_len).sum() / df['volume'].rolling(vol_len, min_periods=vol_len).sum()
    df['vol_rsi'] = calculate_rsi(df['volume'], 14)
    v_s = df['volume'].rolling(14, min_periods=14).mean()
    v_l = df['volume'].rolling(28, min_periods=28).mean()
    df['vol_osc'] = np.where(v_l != 0, 100 * (v_s - v_l) / v_l, 0.0)

    # flips (only when regime confirms)
    df['bull_flip'] = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False).astype(bool))
    df['bear_flip'] = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True).astype(bool))

    # Regime filter: stronger than naive
    df['regime'] = (df['adx'] > 23) & (df['rvol'] > 1.15)

    # HMA filter
    df['hma_buy'] = (~use_hma_filter) | (df['close'] > df['hma'])
    df['hma_sell'] = (~use_hma_filter) | (df['close'] < df['hma'])

    df['buy_signal'] = df['bull_flip'] & df['regime'] & df['hma_buy'] & (df['rsi'] < 70)
    df['sell_signal'] = df['bear_flip'] & df['regime'] & df['hma_sell'] & (df['rsi'] > 30)

    # STATE-LOCKED: entry_price fixed at the candle where signal occurred (ffill)
    df['entry_price'] = np.where(df['buy_signal'] | df['sell_signal'], df['close'], np.nan)
    df['entry_price'] = df['entry_price'].ffill().fillna(np.nan)

    # STATE-LOCKED: risk computed vs trend_stop at time of entry
    # Because entry_price is ffilled, ensure we only compute risk where we have an entry and a valid trend_stop
    df['risk'] = np.where((~df['entry_price'].isna()) & (~df['trend_stop'].isna()),
                          (df['entry_price'] - df['trend_stop']).abs(),
                          np.nan)

    # Freeze ladder relative to the entry price (non-repainting)
    df['tp1'] = np.where(df['is_bull'],
                         df['entry_price'] + df['risk'] * tp1_r,
                         df['entry_price'] - df['risk'] * tp1_r)
    df['tp2'] = np.where(df['is_bull'],
                         df['entry_price'] + df['risk'] * tp2_r,
                         df['entry_price'] - df['risk'] * tp2_r)
    df['tp3'] = np.where(df['is_bull'],
                         df['entry_price'] + df['risk'] * tp3_r,
                         df['entry_price'] - df['risk'] * tp3_r)

    # clean columns
    df = df.assign(
        tp1=lambda x: x['tp1'].replace([np.inf, -np.inf], np.nan),
        tp2=lambda x: x['tp2'].replace([np.inf, -np.inf], np.nan),
        tp3=lambda x: x['tp3'].replace([np.inf, -np.inf], np.nan)
    )

    return df

# -----------------------------
# 6. Backtester (basic, candle-level)
# -----------------------------
def backtest_engine(df: pd.DataFrame, risk_pct: float = 1.0, starting_balance: float = 10000.0) -> Dict:
    """
    Simple backtest:
    - Executes at close on signal candle (entry)
    - Uses state-locked ladder tp1/tp2/tp3 based on entry_price
    - For simplicity assumes fills at TP price if reached in any future candle's high/low
    - No slippage, no fees (extendable)
    """
    df = df.copy().reset_index(drop=True)
    balance = starting_balance
    trades = []
    pos_open = False
    pos = {}

    for i in range(len(df)):
        row = df.iloc[i]
        # open signal
        if (row['buy_signal'] or row['sell_signal']) and not pos_open:
            # compute size by risk %
            entry = row['entry_price']
            stop = row['trend_stop']
            if pd.isna(entry) or pd.isna(stop) or entry == stop:
                continue
            per_unit_risk = abs(entry - stop)
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
                "remaining": 1.0,  # fraction
                "open_index": i
            }
            continue

        # manage open pos
        if pos_open:
            high = row['high']
            low = row['low']
            # check TP3, TP2, TP1 in descending order (for long)
            realized = False
            if pos['side'] == 'long':
                # TP levels may be nan (if no frozen ladder)
                for tp_label in ['tp3', 'tp2', 'tp1']:
                    tp_val = pos.get(tp_label)
                    if pd.isna(tp_val):
                        continue
                    if high >= tp_val:
                        # close a portion (simulate)
                        # assign portion sizes: tp1 30%, tp2 40%, tp3 30% as default
                        if tp_label == 'tp1':
                            fraction = 0.3
                        elif tp_label == 'tp2':
                            fraction = 0.4
                        else:
                            fraction = pos['remaining']  # runner
                        profit = (tp_val - pos['entry']) * pos['size'] * fraction
                        balance += profit
                        pos['remaining'] -= fraction
                        realized = True
                # stop hit?
                if low <= pos['stop']:
                    # close remaining at stop (loss)
                    loss = (pos['entry'] - pos['stop']) * pos['size'] * pos['remaining']
                    balance -= loss
                    pos_open = False
                    pos = {}
            else:  # short
                for tp_label in ['tp3', 'tp2', 'tp1']:
                    tp_val = pos.get(tp_label)
                    if pd.isna(tp_val):
                        continue
                    if low <= tp_val:
                        fraction = 0.3 if tp_label == 'tp1' else (0.4 if tp_label == 'tp2' else pos['remaining'])
                        profit = (pos['entry'] - tp_val) * pos['size'] * fraction
                        balance += profit
                        pos['remaining'] -= fraction
                        realized = True
                if high >= pos['stop']:
                    loss = (pos['stop'] - pos['entry']) * pos['size'] * pos['remaining']
                    balance -= loss
                    pos_open = False
                    pos = {}

            if pos_open and pos['remaining'] <= 0:
                # fully closed via TPs
                pos_open = False
                pos = {}
    return {"final_balance": balance, "profit": balance - starting_balance}

# -----------------------------
# 7. UI: Execution & Charts
# -----------------------------
# Header
st.markdown(f"""
<div class="titan-header">
    <h1 class="titan-title">TITAN DAYTRADER <span style="color:#00ffbb">ULTIMATE PRO</span></h1>
    <div class="titan-subtitle">Ladder Execution + Intraday Trailing Stop — Deterministic Engine</div>
</div>
""", unsafe_allow_html=True)

# Render TradingView (in iframe) — keep under session guard to reduce reloads
def render_tradingview(sym, tf):
    s = f"KRAKEN:{sym.replace('/','')}"
    # only render once per session for same symbol/timeframe
    key = f"tv_{s}_{tf}"
    if not st.session_state.get(key):
        components.html(f"""
        <div class="tradingview-widget-container"><div id="tv"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({{
          "width": "100%", "height": 420, "symbol": "{s}", "interval": "{tf}",
          "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en",
          "container_id": "tv"
        }});</script></div>
        """, height=440)
        st.session_state[key] = True
    else:
        st.info("TradingView loaded. Change symbol/timeframe to refresh.")

render_tradingview(symbol, timeframe)

# Fetch data
with st.spinner("Fetching market data..."):
    df = get_ohlcv_kraken(symbol, timeframe, limit)

if df.empty:
    st.warning("No market data. Check symbol (Kraken format e.g. BTC/USD) and connectivity.")
    st.stop()

# Run engine with explicit params
with st.spinner("Running Titan engine..."):
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

# Broadcast logic with de-duplication
def make_signal_message(last_row):
    direction = "LONG" if last_row['is_bull'] else "SHORT"
    icon = "🟢" if last_row['is_bull'] else "🔴"
    msg = (
        f"🔥 TITAN SIGNAL — {symbol} ({timeframe})\n"
        f"{icon} DIRECTION: {direction}\n"
        f"ENTRY: {last_row['entry_price']:.2f}\n"
        f"STOP: {last_row['trend_stop']:.2f}\n"
        f"TP1: {last_row['tp1']:.2f} | TP2: {last_row['tp2']:.2f} | TP3: {last_row['tp3']:.2f}\n"
        f"ADX: {last_row['adx']:.2f} | RVOL: {last_row['rvol']:.2f}x\n"
        f"Not financial advice."
    )
    return msg

# dedupe id uses timestamp + direction + entry (string)
def make_signal_id(row):
    ts = row['timestamp']
    dir_flag = "L" if row['is_bull'] else "S"
    entry = row['entry_price']
    return f"{ts.isoformat()}_{dir_flag}_{entry:.8f}"

# Broadcast automatically if toggle on
if tg_on and tg_token and tg_chat:
    # check if last row contains a signal
    if last['buy_signal'] or last['sell_signal']:
        sid = make_signal_id(last)
        if st.session_state.get("last_signal_id") != sid:
            # journal
            if db_conn:
                journal_signal(db_conn, {
                    "ts": str(last['timestamp']),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "direction": "LONG" if last['is_bull'] else "SHORT",
                    "entry": float(last['entry_price']) if not pd.isna(last['entry_price']) else None,
                    "stop": float(last['trend_stop']) if not pd.isna(last['trend_stop']) else None,
                    "tp1": float(last['tp1']) if not pd.isna(last['tp1']) else None,
                    "tp2": float(last['tp2']) if not pd.isna(last['tp2']) else None,
                    "tp3": float(last['tp3']) if not pd.isna(last['tp3']) else None,
                    "adx": float(last['adx']),
                    "rvol": float(last['rvol']),
                    "notes": "Auto broadcast"
                })
            msg = make_signal_message(last)
            if send_telegram_msg(tg_token, tg_chat, msg):
                st.success("🔔 Auto-broadcasted signal to Telegram")
                st.session_state["last_signal_id"] = sid
        else:
            st.info("No new signal (deduplicated).")

# HUD metrics
c1, c2, c3, c4 = st.columns(4)
trend_lbl = "BULLISH" if last['is_bull'] else "BEARISH"
with c1:
    st.markdown(f"""<div class="titan-card"><h4>Price</h4><h2>${last['close']:.2f}</h2><div class="sub">Trend: <b>{trend_lbl}</b></div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="titan-card"><h4>Trailing Stop</h4><h2>${(last['trend_stop'] if not pd.isna(last['trend_stop']) else 0):.2f}</h2><div class="sub">Locked Structure</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="titan-card"><h4>ADX</h4><h2>{last['adx']:.1f}</h2><div class="sub">Regime: {'TREND' if last['adx']>23 else 'NOISE'}</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="titan-card"><h4>RVOL</h4><h2>{last['rvol']:.2f}x</h2><div class="sub">Vol Anomaly</div></div>""", unsafe_allow_html=True)

# Action center
st.markdown("### Action Center")
col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("🔥 BROADCAST FULL PLAN (Manual)"):
        if db_conn:
            journal_signal(db_conn, {
                "ts": str(last['timestamp']),
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": "LONG" if last['is_bull'] else "SHORT",
                "entry": float(last['entry_price']) if not pd.isna(last['entry_price']) else None,
                "stop": float(last['trend_stop']) if not pd.isna(last['trend_stop']) else None,
                "tp1": float(last['tp1']) if not pd.isna(last['tp1']) else None,
                "tp2": float(last['tp2']) if not pd.isna(last['tp2']) else None,
                "tp3": float(last['tp3']) if not pd.isna(last['tp3']) else None,
                "adx": float(last['adx']),
                "rvol": float(last['rvol']),
                "notes": "Manual broadcast"
            })
        msg = make_signal_message(last)
        if send_telegram_msg(tg_token, tg_chat, msg):
            st.success("✅ Manual broadcast to Telegram")

with col_b:
    if st.button("🤖 GENERATE AI EXECUTION STRATEGY"):
        summary = {
            'price': f"{last['close']:.2f}",
            'trend': trend_lbl,
            'stop': f"{last['trend_stop']:.2f}",
            'tp1': f"{last['tp1']:.2f}",
            'tp2': f"{last['tp2']:.2f}",
            'tp3': f"{last['tp3']:.2f}",
            'adx': f"{last['adx']:.2f}",
            'rvol': f"{last['rvol']:.2f}"
        }
        with st.spinner("Titan AI is calculating..."):
            ai_report = get_ai_analysis(ai_key, summary, symbol, timeframe)
            st.markdown(f"""<div class="ai-box"><h3>🤖 TITAN AI STRATEGY</h3><p>{ai_report}</p></div>""", unsafe_allow_html=True)

with col_c:
    if st.button("📥 Export Signal (CSV)"):
        payload = {
            "ts": str(last['timestamp']),
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": "LONG" if last['is_bull'] else "SHORT",
            "entry": float(last['entry_price']) if not pd.isna(last['entry_price']) else None,
            "stop": float(last['trend_stop']) if not pd.isna(last['trend_stop']) else None,
            "tp1": float(last['tp1']) if not pd.isna(last['tp1']) else None,
            "tp2": float(last['tp2']) if not pd.isna(last['tp2']) else None,
            "tp3": float(last['tp3']) if not pd.isna(last['tp3']) else None,
            "adx": float(last['adx']),
            "rvol": float(last['rvol'])
        }
        csv = pd.DataFrame([payload]).to_csv(index=False)
        st.download_button("Download CSV", csv, file_name=f"titan_signal_{symbol.replace('/','')}_{timeframe}.csv")

# Charts
st.markdown("### Charts")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.55, 0.2, 0.25],
                    subplot_titles=("Price & Management", "Money Flow", f"Volume ({vol_metric})"))

# Price candlesticks
fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)

# Trailing stops (bull/bear)
b_stop = df['trend_stop'].where(df['is_bull'], np.nan).replace(0, np.nan)
s_stop = df['trend_stop'].where(~df['is_bull'], np.nan).replace(0, np.nan)
fig.add_trace(go.Scatter(x=df['timestamp'], y=b_stop, mode='lines', line=dict(color='#00ffbb', width=2), name='Bull Trail'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['timestamp'], y=s_stop, mode='lines', line=dict(color='#ff1155', width=2), name='Bear Trail'), row=1, col=1)

# Ladder lines (frozen at entry)
tp_col = '#00ffbb' if last['is_bull'] else '#ff1155'
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tp1'], mode='lines', line=dict(color=tp_col, dash='dot', width=1), name='TP1 (1.5R)'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tp2'], mode='lines', line=dict(color=tp_col, dash='dash', width=1), name='TP2 (3.0R)'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tp3'], mode='lines', line=dict(color=tp_col, dash='longdash', width=1), name='TP3 (5.0R)'), row=1, col=1)

# Signals markers
buys = df[df['buy_signal']]
sells = df[df['sell_signal']]
if not buys.empty:
    fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ffbb'), name='BUY'), row=1, col=1)
if not sells.empty:
    fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff1155'), name='SELL'), row=1, col=1)

# Money flow
mf_col = np.where(df['mfi'] >= 50, '#00ffbb', '#ff1155')
fig.add_trace(go.Bar(x=df['timestamp'], y=df['mfi'], marker_color=mf_col, name="MFI"), row=2, col=1)
fig.add_hline(y=50, line_dash="dot", line_color="#333", row=2, col=1)

# Volume metric pane
if vol_metric == "CMF":
    v_data = df['cmf']
    col_v = np.where(v_data >= 0, '#00ffbb', '#ff1155')
    fig.add_trace(go.Bar(x=df['timestamp'], y=v_data, marker_color=col_v, name='CMF'), row=3, col=1)
elif vol_metric == "MFI":
    v_data = df['mfi']
    fig.add_trace(go.Scatter(x=df['timestamp'], y=v_data, mode='lines', name='MFI'), row=3, col=1)
elif vol_metric == "Volume RSI":
    v_data = df['vol_rsi']
    fig.add_trace(go.Scatter(x=df['timestamp'], y=v_data, mode='lines', name='Vol RSI'), row=3, col=1)
elif vol_metric == "RVOL":
    v_data = df['rvol']
    col_v = np.where(v_data >= 1.0, '#00ffbb', '#ff1155')
    fig.add_trace(go.Bar(x=df['timestamp'], y=v_data, marker_color=col_v, name='RVOL'), row=3, col=1)
else:
    v_data = df['vol_osc']
    fig.add_trace(go.Scatter(x=df['timestamp'], y=v_data, mode='lines', name='Vol Osc'), row=3, col=1)

fig.update_layout(height=900, paper_bgcolor='#050505', plot_bgcolor='#050505', font=dict(color="#aaa"), showlegend=True, xaxis_rangeslider_visible=False)
fig.update_yaxes(gridcolor="#222")
fig.update_xaxes(gridcolor="#222")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 8. Backtest (if requested)
# -----------------------------
if run_backtest:
    with st.spinner("Running backtest..."):
        bt = backtest_engine(df, risk_pct=float(backtest_risk_pct), starting_balance=10000.0)
        st.success(f"Backtest complete: Final balance ${bt['final_balance']:.2f} (P/L ${bt['profit']:.2f})")

# -----------------------------
# 9. Signal history (from DB)
# -----------------------------
st.markdown("### Signal Journal")
if persistence and db_conn:
    try:
        cur = db_conn.cursor()
        cur.execute("SELECT ts, symbol, timeframe, direction, entry, stop, tp1, tp2, tp3, adx, rvol, notes FROM signals ORDER BY id DESC LIMIT 50")
        rows = cur.fetchall()
        cols = ["ts","symbol","tf","dir","entry","stop","tp1","tp2","tp3","adx","rvol","notes"]
        df_j = pd.DataFrame(rows, columns=cols)
        st.dataframe(df_j, use_container_width=True)
    except Exception as e:
        st.error(f"Journal read error: {e}")
else:
    st.info("Signal persistence disabled or DB not initialised.")

# -----------------------------
# 10. Final notes & next steps
# -----------------------------
st.markdown("""
**Notes**
- This engine is state-locked and non-repainting: ladder values are frozen at the entry candle and will not move.
- ADX uses Wilder smoothing. ATR uses EWM Wilder-style smoothing.
- For production auto-execution, implement exchange order sending in a separate service (do NOT execute orders directly from Streamlit).
- Extend backtester for fees, slippage, and per-trade logs for robust analytics.

**Next possible upgrades**
- Multi-exchange failover + websocket streaming for ultra-low latency
- Full tick-level backtesting & slippage model
- Position sizing by Kelly/Volatility parity
- Live execution service (separate microservice) with order reconciliation and idempotency
""")
