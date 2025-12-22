"""
TITAN MOBILE PRO - v19.0
ALL-IN-ONE TRADING SUITE: Scanner + Deep Dive + AI Analysis
"""
import time
import math
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime, timezone

# =============================================================================
# 1. PAGE CONFIGURATION & SESSION STATE
# =============================================================================
st.set_page_config(
    page_title="TITAN MOBILE PRO",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'api_region' not in st.session_state:
    st.session_state.api_region = "US"

# =============================================================================
# 2. CUSTOM CSS (MOBILE OPTIMIZED)
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0b0c10; }
    
    /* Mobile-First Card Design */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1f2833, #0b0c10);
        border: 1px solid #45a29e;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        transition: transform 0.1s;
    }
    
    /* Typography */
    h1, h2, h3, h4 { font-family: 'Roboto Mono', monospace; color: #c5c6c7; }
    .stMarkdown { color: #c5c6c7; }
    
    /* Value Styling */
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        color: #66fcf1 !important;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #c5c6c7 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #1f2833;
        border: 1px solid #66fcf1;
        color: #66fcf1;
        height: 3.5em;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
    }
    .stButton > button:active {
        background: #66fcf1;
        color: #0b0c10;
    }

    /* Scanner Grid Cards */
    .scan-card {
        background-color: #1f2833;
        border-left: 5px solid #45a29e;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .scan-bull { border-left-color: #00e676; }
    .scan-bear { border-left-color: #ff1744; }
    
    .scan-symbol { font-size: 20px; font-weight: bold; color: white; }
    .scan-price { float: right; color: #66fcf1; font-family: monospace; }
    .scan-signal { font-size: 14px; margin-top: 5px; }
    
    /* Report Card */
    .report-card {
        background-color: #1f2833;
        border: 1px solid #45a29e;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .highlight { color: #66fcf1; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. UTILITIES & API HANDLER
# =============================================================================

HEADERS = {"User-Agent": "TitanMobile/19.0", "Accept": "application/json"}

def get_base_url():
    """Returns the API URL based on user selection."""
    if st.session_state.api_region == "Global":
        return "https://api.binance.com/api/v3"
    return "https://api.binance.us/api/v3"

@st.cache_data(ttl=5) # Short cache for price data
def fetch_klines(symbol, interval, limit):
    """Fetches raw candle data."""
    base_url = get_base_url()
    symbol_clean = symbol.upper().replace("/", "").replace("-", "")
    if not symbol_clean.endswith("USDT"): symbol_clean += "USDT"
    
    try:
        params = {"symbol": symbol_clean, "interval": interval, "limit": limit}
        r = requests.get(f"{base_url}/klines", params=params, headers=HEADERS, timeout=3)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data, columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            cols = ['open','high','low','close','volume']
            df[cols] = df[cols].astype(float)
            return df[['timestamp'] + cols]
    except Exception as e:
        print(f"API Error: {e}")
    return pd.DataFrame()

def send_telegram(token, chat_id, message):
    if not token or not chat_id: return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=3)
        return True
    except: return False

# =============================================================================
# 4. LOGIC ENGINES (OPTIMIZED)
# =============================================================================

def calculate_hma(series, length):
    """Hull Moving Average"""
    half_len = int(length / 2)
    sqrt_len = int(math.sqrt(length))
    wma_f = series.rolling(length).mean()
    wma_h = series.rolling(half_len).mean()
    diff = 2 * wma_h - wma_f
    return diff.rolling(sqrt_len).mean()

@st.cache_data(show_spinner=False)
def run_titan_strategy(df, amp, dev, hma_len):
    """
    Core Logic Engine.
    Cached to prevent re-calculation on UI interactions.
    """
    if df.empty: return df
    df = df.copy()

    # 1. Volatility & HMA
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
                          abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['hma'] = calculate_hma(df['close'], hma_len)

    # 2. Titan Trend (Supertrend Variant)
    # Using a loop is necessary for recursive trend logic, but it's fast enough for 200 rows.
    # Numba would be overkill for this specific implementation unless rows > 10,000.
    highs = df['high'].rolling(amp).max()
    lows = df['low'].rolling(amp).min()
    
    # Pre-allocate arrays for speed
    closes = df['close'].values
    atrs = df['atr'].values
    h_vals = highs.values
    l_vals = lows.values
    
    trend = np.zeros(len(df), dtype=int)
    stop = np.zeros(len(df))
    
    curr_t = 0 # 0=Bull, 1=Bear
    curr_s = 0.0
    
    for i in range(amp, len(df)):
        c = closes[i]
        d = atrs[i] * dev
        
        if curr_t == 0: # Bull
            s = l_vals[i] + d
            if curr_s < s: curr_s = s # Trail up
            if c < curr_s: 
                curr_t = 1
                curr_s = h_vals[i] - d
        else: # Bear
            s = h_vals[i] - d
            if curr_s == 0 or curr_s > s: curr_s = s # Trail down
            if c > curr_s:
                curr_t = 0
                curr_s = l_vals[i] + d
                
        trend[i] = curr_t
        stop[i] = curr_s
        
    df['trend'] = trend # 0=Bull, 1=Bear
    df['is_bull'] = df['trend'] == 0
    df['stop'] = stop
    
    # 3. Momentum & Squeeze
    df['rvol'] = df['volume'] / df['volume'].rolling(20).mean()
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    kc_range = df['atr'] * 1.5
    df['squeeze'] = ((bb_mid + 2*bb_std) < (bb_mid + kc_range)) & ((bb_mid - 2*bb_std) > (bb_mid - kc_range))
    
    # 4. Sentiment Score (Local Fear & Greed)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Distance from long trend
    sma50 = df['close'].rolling(50).mean()
    dist_score = ((df['close'] - sma50) / sma50) * 100
    
    # Normalize components to 0-100
    rsi_norm = df['rsi']
    vol_norm = df['rvol'].clip(0, 3) * 33
    trend_norm = 50 + (dist_score * 5)
    
    df['sentiment'] = (rsi_norm * 0.4 + vol_norm * 0.3 + trend_norm * 0.3).fillna(50)
    
    return df

# =============================================================================
# 5. UI COMPONENTS
# =============================================================================

def render_deep_dive(symbol, timeframe, settings):
    """Renders the detailed single-asset view."""
    
    # Fetch & Run
    with st.spinner(f"Analyzing {symbol}..."):
        raw_df = fetch_klines(symbol, timeframe, settings['limit'])
        if raw_df.empty:
            st.error("Data fetch failed. Check connection or symbol.")
            return

        df = run_titan_strategy(raw_df, settings['amp'], settings['dev'], settings['hma'])
    
    last = df.iloc[-1]
    
    # --- HEADER ---
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(f"{symbol} ({timeframe})")
    with c2:
        sentiment_color = "#00e676" if last['sentiment'] > 60 else "#ff1744" if last['sentiment'] < 40 else "#ffffff"
        st.markdown(f"<div style='text-align:right; font-size:24px; color:{sentiment_color}; font-weight:bold'>{int(last['sentiment'])}/100</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:right; font-size:12px; color:gray'>SENTIMENT</div>", unsafe_allow_html=True)

    # --- MAIN METRICS ---
    m1, m2, m3 = st.columns(3)
    is_bull = last['is_bull']
    trend_txt = "BULL 🐂" if is_bull else "BEAR 🐻"
    trend_color = "normal" 
    
    m1.metric("TREND", trend_txt)
    m2.metric("PRICE", f"{last['close']:.2f}")
    m3.metric("STOP", f"{last['stop']:.2f}", delta_color="off")

    # --- TARGETS ---
    risk = abs(last['close'] - last['stop'])
    tp1 = last['close'] + (risk * 1.5) if is_bull else last['close'] - (risk * 1.5)
    tp2 = last['close'] + (risk * 3.0) if is_bull else last['close'] - (risk * 3.0)
    
    t1, t2 = st.columns(2)
    t1.metric("TP1 (1.5R)", f"{tp1:.2f}")
    t2.metric("TP2 (3.0R)", f"{tp2:.2f}")

    # --- HTML REPORT ---
    squeeze_status = "⚠️ SQUEEZE" if last['squeeze'] else "⚪ NO SQUEEZE"
    vol_status = "🔥 HIGH" if last['rvol'] > 2.0 else "NORMAL"
    
    html_card = f"""
    <div class="report-card">
        <div><b>SIGNAL DIAGNOSTICS</b></div>
        <hr style="border: 1px solid #45a29e; margin: 5px 0;">
        <div style="display:flex; justify-content:space-between;">
            <span>Volatility:</span> <span class="highlight">{vol_status} ({last['rvol']:.1f})</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span>Market State:</span> <span class="highlight">{squeeze_status}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span>Risk Width:</span> <span class="highlight">{((risk/last['close'])*100):.2f}%</span>
        </div>
    </div>
    """
    st.markdown(html_card, unsafe_allow_html=True)

    # --- CHART ---
    # Simplified chart for mobile performance
    fig = go.Figure()
    fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stop'], mode='lines', line=dict(color='#ff9100', width=1), name='Stop'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', line=dict(color='#66fcf1', width=1, dash='dot'), name='HMA'))
    
    fig.update_layout(
        height=350, 
        margin=dict(l=0,r=0,t=10,b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- ACTIONS ---
    if st.button("📢 SEND ALERT TO TELEGRAM"):
        msg = f"<b>TITAN ALERT</b>\nSymbol: {symbol}\nTrend: {trend_txt}\nPrice: {last['close']}\nStop: {last['stop']:.4f}"
        if send_telegram(settings['tg_token'], settings['tg_chat'], msg):
            st.success("Sent!")
        else:
            st.error("Failed. Check keys.")

def render_scanner(watchlist, timeframe, settings):
    """Multi-Asset Scanner Grid."""
    st.subheader("🔍 MARKET SCANNER")
    st.caption(f"Timeframe: {timeframe}")
    
    if st.button("🔄 SCAN NOW"):
        st.session_state.scanner_results = {} # Clear cache
        
        progress = st.progress(0)
        for i, sym in enumerate(watchlist):
            raw = fetch_klines(sym, timeframe, 100) # Faster limit for scanning
            if not raw.empty:
                res = run_titan_strategy(raw, settings['amp'], settings['dev'], settings['hma'])
                st.session_state.scanner_results[sym] = res.iloc[-1]
            progress.progress((i + 1) / len(watchlist))
        progress.empty()

    # Display Results
    if 'scanner_results' in st.session_state and st.session_state.scanner_results:
        results = st.session_state.scanner_results
        
        # Grid Layout
        for sym, data in results.items():
            trend = "BULL" if data['is_bull'] else "BEAR"
            color_cls = "scan-bull" if data['is_bull'] else "scan-bear"
            emoji = "🐂" if data['is_bull'] else "🐻"
            
            html = f"""
            <div class="scan-card {color_cls}">
                <div class="scan-symbol">{sym} <span class="scan-price">{data['close']:.2f}</span></div>
                <div class="scan-signal">{emoji} {trend} | Sent: {int(data['sentiment'])}</div>
                <div style="font-size:12px; color:gray; margin-top:4px;">Stop: {data['stop']:.4f}</div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("Tap 'SCAN NOW' to analyze the market.")

# =============================================================================
# 6. MAIN APP CONTROLLER
# =============================================================================

def main():
    # --- SIDEBAR SETTINGS ---
    with st.sidebar:
        st.header("⚙️ CONFIG")
        
        # API Toggle
        st.session_state.api_region = st.radio("API Region", ["US", "Global"], horizontal=True)
        
        # Mode Selection
        mode = st.selectbox("APP MODE", ["Deep Dive", "Scanner"])
        
        st.markdown("---")
        st.subheader("Strategy")
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
        amp = st.slider("Amplitude", 5, 50, 10)
        dev = st.slider("Deviation", 1.0, 5.0, 3.0)
        hma = st.slider("HMA Length", 10, 200, 50)
        
        st.markdown("---")
        st.subheader("Telegram Keys")
        tg_token = st.text_input("Token", type="password")
        tg_chat = st.text_input("Chat ID")

    # Settings Dict
    settings = {
        'amp': amp, 'dev': dev, 'hma': hma, 'limit': 300,
        'tg_token': tg_token, 'tg_chat': tg_chat
    }

    # --- HEADER CLOCK ---
    components.html(
        """
        <div style="text-align: center; color: #45a29e; font-family: monospace; font-weight: bold;">
            TITAN OS • <span id="clock"></span>
        </div>
        <script>
        setInterval(() => {
            document.getElementById('clock').innerText = new Date().toLocaleTimeString();
        }, 1000);
        </script>
        """, height=30
    )

    # --- ROUTING ---
    if mode == "Deep Dive":
        symbol = st.text_input("ASSET", value="BTC", placeholder="e.g. BTC, ETH").upper()
        if st.button("🔄 REFRESH DATA"):
             fetch_klines.clear() # Clear cache to force new data
             st.rerun()
        render_deep_dive(symbol, timeframe, settings)
        
    elif mode == "Scanner":
        default_list = ["BTC", "ETH", "SOL", "BNB", "ADA", "XRP", "DOGE", "AVAX"]
        user_list = st.text_area("Watchlist (comma separated)", value=",".join(default_list))
        watchlist = [x.strip().upper() for x in user_list.split(",")]
        render_scanner(watchlist, timeframe, settings)

if __name__ == "__main__":
    main()
