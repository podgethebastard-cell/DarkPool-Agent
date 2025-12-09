import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime
import time

# --- NEW: Imports for Broadcasting ---
try:
    from telegram import Bot
    from tweepy import Client, OAuth1UserHandler
    from flask import Flask, request, jsonify
except ImportError:
    st.error("Please install the required libraries: `pip install python-telegram-bot tweepy flask`")
    st.stop()

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

st.sidebar.header("Logic Engine / Ladder")
atr_mult = st.sidebar.number_input("Stop Deviation (ATR x)", 0.5, 10.0, 3.0)
hma_len = st.sidebar.number_input("HMA Length", 10, 200, 50)

# =========================
# SIGNAL BROADCASTING SETTINGS
# =========================
st.sidebar.header("Signal Broadcasting")
broadcast_enabled = st.sidebar.checkbox("Enable Broadcasting", value=False)

if broadcast_enabled:
    # --- TELEGRAM SETTINGS ---
    st.sidebar.subheader("Telegram")
    telegram_bot_token = st.sidebar.text_input("Bot Token", type="password", help="Get from @BotFather")
    telegram_chat_id = st.sidebar.text_input("Chat ID", help="Get from api.telegram.org/bot<token>/getUpdates")

    # --- X (TWITTER) SETTINGS ---
    st.sidebar.subheader("X (Twitter)")
    twitter_api_key = st.sidebar.text_input("API Key", type="password")
    twitter_api_secret = st.sidebar.text_input("API Secret Key", type="password")
    twitter_access_token = st.sidebar.text_input("Access Token", type="password")
    twitter_access_token_secret = st.sidebar.text_input("Access Token Secret", type="password")

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
            r.raise_for_status()

            data = r.json()
            if len(data) == 0:
                st.warning(f"API at {current_url} returned empty data.")
                continue

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
            st.code(r.text)
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
df['tr1'] = df['high'] - df['low']
df['tr2'] = abs(df['high'] - df['close'].shift(1))
df['tr3'] = abs(df['low'] - df['close'].shift(1))
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['ATR'] = df['tr'].rolling(14).mean()
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
df["trail"] = df["close"] - df["ATR"] * atr_mult
df["trail"] = df["trail"].cummax()

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
# LIVE SIGNAL FEED & BROADCASTING
# =========================
st.subheader("Live Execution Feed")

last_signal = "NONE"
if not buy_signals.empty and buy_signals['time'].iloc[-1] == df['time'].iloc[-1]:
    last_signal = "BUY"
    signal_message = f"🚀 BUY Signal for {symbol} on {timeframe}! Price: ${price:,.2f}"
elif not sell_signals.empty and sell_signals['time'].iloc[-1] == df['time'].iloc[-1]:
    last_signal = "SELL"
    signal_message = f"🔻 SELL Signal for {symbol} on {timeframe}! Price: ${price:,.2f}"

st.success(f"Latest Signal: {last_signal}")

# --- BROADCASTING LOGIC ---
if broadcast_enabled and last_signal != "NONE":
    st.info("Broadcasting signal...")
    
    # --- TELEGRAM BROADCAST ---
    if telegram_bot_token and telegram_chat_id:
        try:
            bot = Bot(token=telegram_bot_token)
            bot.send_message(chat_id=telegram_chat_id, text=signal_message)
            st.success("✅ Signal sent to Telegram.")
        except Exception as e:
            st.error(f"❌ Failed to send Telegram message: {e}")

    # --- X (TWITTER) BROADCAST ---
    if all([twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_token_secret]):
        try:
            # Note: This uses Tweepy v2. Rate limits apply.
            handler = OAuth1UserHandler(
                consumer_key=twitter_api_key,
                consumer_secret=twitter_api_secret,
                access_token=twitter_access_token,
                access_token_secret=twitter_access_token_secret
            )
            client = Client(consumer_key=twitter_api_key, consumer_secret=twitter_api_secret, access_token=twitter_access_token, access_token_secret=twitter_access_token_secret)
            
            # Post the tweet
            client.create_tweet(text=signal_message)
            st.success("✅ Signal posted to X (Twitter).")
        except Exception as e:
            st.error(f"❌ Failed to post to X (Twitter): {e}")

st.caption("TITAN Engine Online | REST Feed Active | Cloud Stable")

# =========================
# TRADINGVIEW WEBHOOK INTEGRATION (Flask Server)
# =========================
# This part runs a simple web server to listen for alerts from TradingView.
# It should be run in a separate terminal or as a background process.
# Example Pine Script for a TradingView alert:
# `alertcondition(condition, title="TITAN Signal", message="Signal Fired!")`
# Then, in the alert settings, use a webhook with this URL: http://localhost:5000/tradingview-alert

app = Flask(__name__)

@app.route('/tradingview-alert', methods=['POST'])
def tradingview_alert():
    data = request.json
    st.sidebar.warning(f"Received alert from TradingView: {data}")
    # Here you could trigger the broadcast logic or save the signal to a file
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    # This starts the Flask server. In a real deployment, use a production server like Gunicorn.
    # For local testing, you can run this script and it will start both Streamlit and Flask.
    # However, Streamlit's `st.run()` is not ideal for this. A better approach is to run them separately.
    # For simplicity in this single-file example, we'll just define the route.
    # In a real setup, you would run `flask run` in a separate terminal.
    pass
