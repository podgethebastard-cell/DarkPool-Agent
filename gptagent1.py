import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("🔨 TITAN INTRADAY PRO — Execution Dashboard")

# =========================
# SIDEBAR SETTINGS
# =========================
st.sidebar.header("Market Feed")

symbol = st.sidebar.text_input("Symbol (Binance format)", "BTCUSDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","30m","1h","4h"])
candles = st.sidebar.slider("Candles", 100, 1500, 600)

# Removed unused parameters: lookback, use_hma
st.sidebar.header("Logic Engine / Ladder")
atr_mult = st.sidebar.number_input("Stop Deviation (ATR x)", 0.5, 10.0, 3.0)
hma_len = st.sidebar.number_input("HMA Length", 10, 200, 50)

# =========================
# DATA FETCHER (BINANCE)
# =========================
@st.cache_data(ttl=30)
def fetch_data(symbol, tf, limit):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": tf, "limit": limit}
    r = requests.get(url, timeout=10)

    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json()
    if len(data) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "ct","qav","trades","tbav","tqav","ignore"
    ])

    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    return df[["time","open","high","low","close","volume"]]

df = fetch_data(symbol.upper(), timeframe, candles)

if df.empty:
    st.error("Market data not available. Check symbol, timeframe or network.")
    st.stop()

# Data sanity check for HMA calculation
if len(df) < hma_len:
    st.warning(f"Not enough data to calculate HMA. Need at least {hma_len} candles. Current: {len(df)}.")

# =========================
# INDICATORS
# =========================

# --- FIXED: Correct ATR Calculation using True Range ---
# Calculate True Range (TR)
df['tr1'] = df['high'] - df['low']
df['tr2'] = abs(df['high'] - df['close'].shift(1))
df['tr3'] = abs(df['low'] - df['close'].shift(1))
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

# Calculate ATR
df['ATR'] = df['tr'].rolling(14).mean()
# Clean up temporary columns
df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)


def HMA(series, length):
    half = int(length / 2)
    sqrt = int(np.sqrt(length))
    wma1 = series.rolling(half).mean()
    wma2 = series.rolling(length).mean()
    diff = 2 * wma1 - wma2
    return diff.rolling(sqrt).mean()

df["HMA"] = HMA(df["close"], hma_len)

# =========================
# TREND + SIGNAL LOGIC
# =========================
df["trend"] = np.where(df["close"] > df["HMA"], 1, -1)
df["signal"] = df["trend"].diff()

buy_signals = df[df["signal"] == 2]
sell_signals = df[df["signal"] == -2]

# =========================
# TRAILING STOP ENGINE
# =========================
# This logic is for a LONG-ONLY strategy.
# The trailing stop is placed below the price by a multiple of the ATR.
df["trail"] = df["close"] - df["ATR"] * atr_mult
df["trail"] = df["trail"].cummax() # The stop only moves up (higher) over time

# =========================
# DASHBOARD METRICS
# =========================
price = df["close"].iloc[-1]
trail_price = df["trail"].iloc[-1]
trend_state = "BULLISH" if df["trend"].iloc[-1] == 1 else "BEARISH"

col1, col2, col3 = st.columns(3)
col1.metric("Price", f"${price:,.2f}")
col2.metric("Trailing Stop", f"${trail_price:,.2f}")
col3.metric("Trend", trend_state)

# =========================
# PLOTLY CHART
# =========================
fig = go.Figure()

fig.add_candlestick(
    x=df["time"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="Price"
)

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["HMA"],
    line=dict(width=2),
    name="HMA"
))

fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["trail"],
    line=dict(width=2),
    name="Trailing Stop"
))

fig.add_trace(go.Scatter(
    x=buy_signals["time"],
    y=buy_signals["close"],
    mode="markers",
    marker=dict(size=10, symbol="triangle-up"),
    name="BUY"
))

fig.add_trace(go.Scatter(
    x=sell_signals["time"],
    y=sell_signals["close"],
    mode="markers",
    marker=dict(size=10, symbol="triangle-down"),
    name="SELL"
))

fig.update_layout(
    height=700,
    template="plotly_dark",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# LIVE SIGNAL FEED
# =========================
st.subheader("Live Execution Feed")

last_signal = "NONE"
# --- FIXED: Use timestamps for reliable signal detection ---
if not buy_signals.empty and buy_signals['time'].iloc[-1] == df['time'].iloc[-1]:
    last_signal = "BUY"
elif not sell_signals.empty and sell_signals['time'].iloc[-1] == df['time'].iloc[-1]:
    last_signal = "SELL"

st.success(f"Latest Signal: {last_signal}")

st.caption("TITAN Engine Online | REST Feed Active | Cloud Stable")
