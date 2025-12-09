import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime

st.set_page_config(layout="wide")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.title("⚡ TITAN MARKET FEED")

symbol = st.sidebar.text_input("Symbol", "BTCUSDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h"])
limit = st.sidebar.slider("Candles", 100, 1000, 500)

tf_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240"}
tf_binance = tf_map[timeframe]

# =========================
# LIVE PRICE (REST POLLING)
# =========================
@st.cache_data(ttl=10)
def get_ohlc():
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "c1","c2","c3","c4","c5","c6"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
    return df

df = get_ohlc()

price = df["close"].iloc[-1]

# =========================
# SIGNAL ENGINE (TREND + ADX STYLE)
# =========================
df["ema_fast"] = df["close"].ewm(span=20).mean()
df["ema_slow"] = df["close"].ewm(span=50).mean()

df["trend"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
df["signal"] = df["trend"].diff()

buy_signals = df[df["signal"] == 2]
sell_signals = df[df["signal"] == -2]

# =========================
# TRAILING STOP ENGINE (ATR STYLE)
# =========================
atr = (df["high"] - df["low"]).rolling(14).mean()
trail_stop = price - (atr.iloc[-1] * 3)

# =========================
# LADDER TARGETS
# =========================
tp1 = price * 1.005
tp2 = price * 1.01
tp3 = price * 1.02

# =========================
# EXECUTION CHART (FULL CONTROL)
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

fig.add_scatter(x=df["time"], y=df["ema_fast"], name="EMA 20")
fig.add_scatter(x=df["time"], y=df["ema_slow"], name="EMA 50")

fig.add_scatter(
    x=buy_signals["time"],
    y=buy_signals["close"],
    mode="markers",
    marker=dict(size=12, symbol="triangle-up"),
    name="BUY"
)

fig.add_scatter(
    x=sell_signals["time"],
    y=sell_signals["close"],
    mode="markers",
    marker=dict(size=12, symbol="triangle-down"),
    name="SELL"
)

fig.add_hline(y=trail_stop, line_dash="dot", name="Trailing Stop")
fig.add_hline(y=tp1, line_dash="dash", name="TP1")
fig.add_hline(y=tp2, line_dash="dash", name="TP2")
fig.add_hline(y=tp3, line_dash="dash", name="TP3")

fig.update_layout(height=700, template="plotly_dark")

# =========================
# TRADINGVIEW (MANUAL TA)
# =========================
def tradingview(symbol="BINANCE:BTCUSDT", interval="60"):
    return f"""
    <html>
      <head>
        <script src="https://s3.tradingview.com/tv.js"></script>
      </head>
      <body style="margin:0">
        <div id="tv" style="height:700px"></div>
        <script>
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{symbol}",
          "interval": "{interval}",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "container_id": "tv"
        }});
        </script>
      </body>
    </html>
    """

# =========================
# DASHBOARD LAYOUT
# =========================
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("🧠 TITAN EXECUTION ENGINE (LIVE SIGNALS)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 TRADINGVIEW (MANUAL TA)")
    components.html(tradingview(), height=720)

# =========================
# STATS PANEL
# =========================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Price", f"${price:,.2f}")
c2.metric("Trailing Stop", f"${trail_stop:,.2f}")
c3.metric("Regime", "BULLISH" if df["trend"].iloc[-1] == 1 else "BEARISH")
c4.metric("Last Signal", "BUY" if df["signal"].iloc[-1] == 2 else "SELL")


