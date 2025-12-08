import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import requests
import time

# ==========================================
# 1. TITAN CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(page_title="Titan Scalper [Kraken]", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    /* Success/Error message styling */
    .stAlert { background-color: #1e222b; border: 1px solid #444; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar: Market Config ---
st.sidebar.header("⚡ Titan Scalper Config")
symbol = st.sidebar.text_input("Symbol (Kraken)", value="BTC/USD") 
timeframe = st.sidebar.selectbox("Timeframe", options=['1m', '5m', '15m'], index=1)
limit = st.sidebar.slider("Candles to Load", min_value=100, max_value=1000, value=300)

st.sidebar.markdown("---")

# --- Sidebar: Engine Settings ---
st.sidebar.subheader("Engine Settings")
amplitude = st.sidebar.number_input("Sensitivity (Lookback)", min_value=1, value=5)
channel_dev = st.sidebar.number_input("Volatility Multiplier", min_value=1.0, value=3.0, step=0.1)
hma_len = st.sidebar.number_input("HMA Length", min_value=1, value=50)
use_hma_filter = st.sidebar.checkbox("Use HMA Filter?", value=False)

st.sidebar.markdown("---")

# --- Sidebar: Telegram Bot ---
st.sidebar.subheader("📢 Telegram Broadcaster")
tg_on = st.sidebar.checkbox("Enable Broadcasts", value=False)
bot_token = st.sidebar.text_input("Bot Token", type="password", help="From @BotFather")
chat_id = st.sidebar.text_input("Chat ID", help="Your User or Channel ID")
test_btn = st.sidebar.button("Test Message")

# Initialize Session State for Anti-Spam
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = None

# ==========================================
# 2. HELPER: TRADINGVIEW WIDGET
# ==========================================
def render_tv_widget(symbol):
    clean_sym = symbol.replace("/", "")
    tv_symbol = f"KRAKEN:{clean_sym}"
    
    html_code = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%",
        "height": 500,
        "symbol": "{tv_symbol}",
        "interval": "5",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }}
      );
      </script>
    </div>
    """
    components.html(html_code, height=510)

# ==========================================
# 3. HELPER: TELEGRAM SENDER
# ==========================================
def send_telegram_msg(token, chat, msg):
    if not token or not chat:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": msg,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        st.error(f"Telegram Fail: {e}")

if test_btn:
    send_telegram_msg(bot_token, chat_id, "🔥 **TITAN SYSTEM CONNECTED**\nReady to broadcast signals.")
    st.sidebar.success("Test Sent!")

# ==========================================
# 4. MATH ENGINE (Universal Logic)
# ==========================================
def calculate_hma(series, length):
    def weighted_avg(w):
        def _compute(x):
            return np.dot(x, w) / w.sum()
        return _compute
    
    weights_half = np.arange(1, int(length/2) + 1)
    wma_half = series.rolling(int(length/2)).apply(weighted_avg(weights_half), raw=True)
    
    weights_full = np.arange(1, length + 1)
    wma_full = series.rolling(length).apply(weighted_avg(weights_full), raw=True)
    
    diff = 2 * wma_half - wma_full
    weights_sqrt = np.arange(1, int(np.sqrt(length)) + 1)
    hma = diff.rolling(int(np.sqrt(length))).apply(weighted_avg(weights_sqrt), raw=True)
    return hma

def run_titan_engine(df):
    # 1. Volatility
    df['tr'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr_algo'] = df['tr'].rolling(window=100).mean() / 2
    df['dev'] = df['atr_algo'] * channel_dev
    
    # 2. HMA
    df['hma'] = calculate_hma(df['close'], hma_len)

    # 3. Staircase Trend
    df['lowest_low'] = df['low'].rolling(window=amplitude).min()
    df['highest_high'] = df['high'].rolling(window=amplitude).max()

    trend = np.zeros(len(df))
    trend_stop = np.zeros(len(df))
    curr_trend_stop = df['close'].iloc[0]
    
    curr_trend = 0
    curr_max_low = 0.0
    curr_min_high = 0.0

    for i in range(amplitude, len(df)):
        close = df['close'].iloc[i]
        low_price = df['lowest_low'].iloc[i]
        high_price = df['highest_high'].iloc[i]
        safe_dev = df['dev'].iloc[i] if not np.isnan(df['dev'].iloc[i]) else 0
        
        curr_max_low = max(low_price, curr_max_low)
        curr_min_high = min(high_price, curr_min_high) if curr_min_high != 0 else high_price

        if curr_trend == 0: # Bull
            if close < curr_max_low:
                curr_trend = 1 
                curr_min_high = high_price
            else:
                curr_min_high = min(curr_min_high, high_price)
                if low_price < curr_max_low:
                    curr_max_low = low_price
        else: # Bear
            if close > curr_min_high:
                curr_trend = 0 
                curr_max_low = low_price
            else:
                curr_max_low = max(curr_max_low, low_price)
                if high_price > curr_min_high:
                    curr_min_high = high_price
        
        if curr_trend == 0: # Bull
            up_line = low_price + safe_dev
            if up_line < curr_trend_stop:
                up_line = curr_trend_stop
            curr_trend_stop = up_line
        else: # Bear
            down_line = high_price - safe_dev
            if down_line > curr_trend_stop and curr_trend_stop != 0:
                down_line = curr_trend_stop
            elif curr_trend_stop == 0:
                down_line = high_price - safe_dev
            curr_trend_stop = down_line

        trend[i] = curr_trend
        trend_stop[i] = curr_trend_stop

    df['trend'] = trend
    df['trend_stop'] = trend_stop
    df['is_bull'] = df['trend'] == 0
    df['is_bear'] = df['trend'] == 1
    
    # 4. Signals
    prev_is_bull = df['is_bull'].shift(1).fillna(False).astype(bool)
    prev_is_bear = df['is_bear'].shift(1).fillna(False).astype(bool)

    df['bull_flip'] = (df['is_bull']) & (~prev_is_bull)
    df['bear_flip'] = (df['is_bear']) & (~prev_is_bear)
    
    df['filter_ok_buy'] = ~use_hma_filter | (df['close'] > df['hma'])
    df['filter_ok_sell'] = ~use_hma_filter | (df['close'] < df['hma'])
    
    df['buy_signal'] = df['bull_flip'] & df['filter_ok_buy']
    df['sell_signal'] = df['bear_flip'] & df['filter_ok_sell']

    return df

# ==========================================
# 5. DATA FETCHING
# ==========================================
@st.cache_data(ttl=60) 
def get_data(symbol, timeframe, limit):
    try:
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
st.subheader("Market Overview (TradingView)")
render_tv_widget(symbol)

st.markdown("---")
st.subheader("Titan Vector Analysis")

df = get_data(symbol, timeframe, limit)

if not df.empty:
    df = run_titan_engine(df)
    last_row = df.iloc[-1]
    
    # --- BROADCAST LOGIC ---
    if tg_on and bot_token and chat_id:
        # Check specific columns for signal
        is_buy = last_row['buy_signal']
        is_sell = last_row['sell_signal']
        sig_time = last_row['timestamp']
        
        # Only send if new timestamp
        if (is_buy or is_sell) and (st.session_state.last_signal_time != sig_time):
            
            # Message Variables
            direction = "LONG" if is_buy else "SHORT"
            icon = "🟢" if is_buy else "🔴"
            entry = last_row['close']
            stop = last_row['trend_stop']
            
            # Calc Target (1.5 RR)
            risk = abs(entry - stop)
            target = entry + (risk * 1.5) if is_buy else entry - (risk * 1.5)
            
            trend_str = "BULLISH" if last_row['is_bull'] else "BEARISH"
            
            # Simple derived metrics
            mom_str = "POSITIVE" if last_row['close'] > last_row['open'] else "NEGATIVE"
            inst_str = "MACRO BULL" if last_row['close'] > last_row['hma'] else "MACRO BEAR"

            msg = f"""🔥 *TITAN SIGNAL: {symbol} ({timeframe})*
{icon} DIRECTION: *{direction}*
🚪 ENTRY: `${entry:,.2f}`
🛑 STOP LOSS: `${stop:,.2f}`
🎯 TARGET: `${target:,.2f}`
🌊 Trend: {trend_str}
📊 Momentum: {mom_str}
💀 Institutional Trend: {inst_str}
⚠️ _Not financial advice. DYOR._
#DarkPool #Titan #Crypto"""

            send_telegram_msg(bot_token, chat_id, msg)
            st.session_state.last_signal_time = sig_time
            st.toast(f"Signal Broadcasted: {direction}", icon="🚀")

    # --- METRICS UI ---
    col1, col2, col3, col4 = st.columns(4)
    trend_color = "#00ffbb" if last_row['is_bull'] else "#ff1155"
    trend_text = "BULLISH" if last_row['is_bull'] else "BEARISH"
    
    col1.metric("Price", f"{last_row['close']:.2f}")
    col2.markdown(f"**Trend**<br><span style='color:{trend_color}; font-size:24px; font-weight:bold'>{trend_text}</span>", unsafe_allow_html=True)
    col3.metric("Stop", f"{last_row['trend_stop']:.2f}")
    
    filter_status = "OFF"
    if use_hma_filter:
        filter_status = "PASS" if (last_row['is_bull'] and last_row['close'] > last_row['hma']) or (last_row['is_bear'] and last_row['close'] < last_row['hma']) else "FAIL"
    col4.metric("Filter", filter_status)

    # --- CHART ---
    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price',
        increasing_line_color= '#00ffbb', decreasing_line_color= '#ff1155'
    ))

    # Fix Scaling (Replace 0 with NaN)
    df['trend_stop'] = df['trend_stop'].replace(0, np.nan)

    bull_line = df.copy()
    bull_line.loc[bull_line['trend'] == 1, 'trend_stop'] = None
    bear_line = df.copy()
    bear_line.loc[bear_line['trend'] == 0, 'trend_stop'] = None

    fig.add_trace(go.Scatter(x=bull_line['timestamp'], y=bull_line['trend_stop'], mode='lines', line=dict(color='#00ffbb', width=2), name='Bull Stop'))
    fig.add_trace(go.Scatter(x=bear_line['timestamp'], y=bear_line['trend_stop'], mode='lines', line=dict(color='#ff1155', width=2), name='Bear Stop'))
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', line=dict(color='gray', width=1, dash='dot'), name='HMA', opacity=0.5))

    # Signals
    buys = df[df['buy_signal']]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys['timestamp'], y=buys['low'] * 0.999,
            mode='markers', marker=dict(symbol='diamond', size=10, color='#aaffdd'), name='Buy Signal'
        ))

    sells = df[df['sell_signal']]
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells['timestamp'], y=sells['high'] * 1.001,
            mode='markers', marker=dict(symbol='diamond', size=10, color='#ff99aa'), name='Sell Signal'
        ))

    # Layout
    fig.update_layout(
        height=600,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        xaxis_rangeslider_visible=False,
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(autorange=True, fixedrange=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Waiting for data from Kraken...")
