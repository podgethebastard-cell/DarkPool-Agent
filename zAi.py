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
# --- MODIFIED: Removed 1m, 5m options and set default to 1h ---
timeframe = st.sidebar.selectbox("Timeframe", ["15m", "30m", "1h", "4h"], index=2)
candles = st.sidebar.slider("Candles", 100, 1500, 600)

# Removed unused parameters: lookback, use_hma
st.sidebar.header("Logic Engine / Ladder")
atr_mult = st.sidebar.number_input("Stop Deviation (ATR x)", 0.5, 10.0, 3.0)
hma_len = st.sidebar.number_input("HMA Length", 10, 200, 50)

# --- NEW: Take Profit Settings ---
st.sidebar.header("Take Profit Ladder")
tp_atr_mult = st.sidebar.slider("TP ATR Multiplier", 1.0, 10.0, 4.5, step=0.5)
tp_levels = [1.5, 3.0, 4.5] # Standard laddering levels

# =========================
# DATA FETCHER (BINANCE) - OPTIMIZED FOR RESTRICTED REGIONS
# =========================
@st.cache_data(ttl=30)
def fetch_data(symbol, tf, limit):
    url_primary = "https://api.binance.us/api/v3/klines"
    url_fallback = "https://api.binance.com/api/v3/klines"
    
    params = {"symbol": symbol, "interval": tf, "limit": limit}
    url_to_try = url_primary

    for attempt, current_url in enumerate([url_primary, url_fallback], start=1):
        try:
            r = requests.get(current_url, params=params, timeout=10)
            r.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)

            data = r.json()
            if len(data) == 0:
                st.warning(f"API at {current_url} returned empty data.")
                continue  # Try the next URL

            df = pd.DataFrame(data, columns=[
                "time","open","high","low","close","volume",
                "ct","qav","trades","tbav","tqav","ignore"
            ])

            df["time"] = pd.to_datetime(df["time"], unit="ms")
            for col in ["open","high","low","close","volume"]:
                df[col] = df[col].astype(float)

            st.success(f"Successfully fetched data from {current_url}.")
            return df[["time","open","high","low","close","volume"]]

        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error on attempt {attempt} with {current_url}: {http_err}")
            st.code(r.text) # Show the raw response for debugging
        except requests.exceptions.ConnectionError:
            st.error(f"Connection error on attempt {attempt} with {current_url}. Check your network.")
        except requests.exceptions.Timeout:
            st.error(f"Timeout error on attempt {attempt} with {current_url}.")
        except requests.exceptions.RequestException as err:
            st.error(f"An unexpected error occurred on attempt {attempt} with {current_url}: {err}")

    st.error("All data sources failed. Could not fetch market data.")
    return pd.DataFrame()

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

# --- NEW: DYNAMIC STOP AND TAKE PROFIT CALCULATION ---
price = df["close"].iloc[-1]
atr_val = df["ATR"].iloc[-1]
trend_state = df["trend"].iloc[-1]

if trend_state == 1:  # LONG (BULLISH)
    stop_price = price - (atr_mult * atr_val)
    take_profits = [price + (mult * atr_val) for mult in tp_levels]
elif trend_state == -1: # SHORT (BEARISH)
    stop_price = price + (atr_mult * atr_val)
    take_profits = [price - (mult * atr_val) for mult in tp_levels]
else: # NO TREND
    stop_price = price
    take_profits = [price, price, price]

trail_price = df["trail"].iloc[-1] # Keep the trailing stop for display
tp1, tp2, tp3 = take_profits

# =========================
# DASHBOARD METRICS
# =========================
trend_label = "BULLISH" if trend_state == 1 else "BEARISH" if trend_state == -1 else "NEUTRAL"

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Price", f"${price:,.2f}")
col2.metric("Trailing Stop", f"${trail_price:,.2f}")
col3.metric("Trend", trend_label)
col4.metric("TP1", f"${tp1:,.2f}")
col5.metric("TP2", f"${tp2:,.2f}")
col6.metric("TP3", f"${tp3:,.2f}")

# =========================
# TRADINGVIEW CHART INTEGRATION (FIXED)
# =========================
# --- NEW: Map Binance timeframes to TradingView intervals ---
timeframe_map = {
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240"
}

# --- NEW: Construct the TradingView widget HTML with a delay for script loading ---
tradingview_interval = timeframe_map.get(timeframe, "60") # Default to 1h if not found
tradingview_symbol = symbol.upper() # TradingView uses the full symbol, e.g., "BTCUSDT"

tradingview_html = f"""
<div id="tradingview_chart"></div>
<script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
<script type="text/javascript">
  // Initialize the widget after a short delay to ensure tv.js is loaded
  setTimeout(function() {{
    new TradingView.widget({{
      "width": "100%",
      "height": 610,
      "symbol": "BINANCE:{tradingview_symbol}",
      "interval": "{tradingview_interval}",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#f1f3f6",
      "enable_publishing": false,
      "allow_symbol_change": true,
      "container_id": "tradingview_chart",
      "hide_side_toolbar": false,
      "save_image": true
    }});
  }}, 500); // 500ms delay
</script>
"""

# --- NEW: Render the TradingView widget using st.markdown for better script handling ---
st.markdown(tradingview_html, unsafe_allow_html=True)

# =========================
# LIVE SIGNAL FEED
# =========================
st.subheader("Live Execution Feed")

last_signal = "NONE"
if not buy_signals.empty and buy_signals['time'].iloc[-1] == df['time'].iloc[-1]:
    last_signal = "BUY"
elif not sell_signals.empty and sell_signals['time'].iloc[-1] == df['time'].iloc[-1]:
    last_signal = "SELL"

st.success(f"Latest Signal: {last_signal}")

# --- BROADCASTING LOGIC (ROBUST) ---
st.sidebar.subheader("📡 Broadcast Keys")

# --- AUTOMATIC SECRET LOADING ---
# Try to load from st.secrets first
tg_token = st.secrets.get("TELEGRAM_TOKEN", "")
tg_chat = st.secrets.get("TELEGRAM_CHAT_ID", "")

# If secrets are empty, fall back to sidebar inputs (for local dev)
if not tg_token:
    tg_token = st.sidebar.text_input("Telegram Bot Token", type="password")
if not tg_chat:
    tg_chat = st.sidebar.text_input("Telegram Chat ID")

# --- BROADCASTING LOGIC ---
if tg_token and tg_chat:
    # Only show broadcast section if a signal was generated
    if last_signal != "NONE":
        st.sidebar.subheader("📡 Broadcast")
        # Create a preview based on the signal direction
        if last_signal == "BUY":
            preview = f"🚀 BUY Signal for {symbol} on {timeframe}!\nENTRY: ${price:,.2f}\nSTOP: ${stop_price:,.2f}\nTP1: ${tp1:,.2f}\nTP2: ${tp2:,.2f}\nTP3: ${tp3:,.2f}"
        else: # SELL
            preview = f"🔻 SELL Signal for {symbol} on {timeframe}!\nENTRY: ${price:,.2f}\nSTOP: ${stop_price:,.2f}\nTP1: ${tp1:,.2f}\nTP2: ${tp2:,.2f}\nTP3: ${tp3:,.2f}"
        
        st.text_area("Preview", preview, height=150)
        
        if st.button("Send🚀"):
            try:
                requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={"chat_id":tg_chat, "text":preview})
                st.success("✅ Signal sent to Telegram.")
            except Exception as e:
                st.error(f"❌ Failed to send Telegram message: {e}")
    else:
        st.sidebar.subheader("📡 Broadcast (No Signal to Send)")

st.caption("TITAN Engine Online | REST Feed Active | Cloud Stable")
