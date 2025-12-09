import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import requests
import time
from openai import OpenAI

# ==========================================
# 1. SETUP PAGE CONFIGURATION (Mobile Template)
# ==========================================
st.set_page_config(
    page_title="Titan Mobile", 
    layout="centered", 
    page_icon="⚡", 
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM CSS (Merged Mobile + Titan Neon)
# ==========================================
st.markdown("""
    <style>
    /* --- MOBILE BASE STYLES --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* Increase button size for touch targets */
    div.stButton > button:first-child {
        width: 100%;
        height: 3.5em;
        font-weight: bold; 
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* --- TITAN SPECIFIC STYLES --- */
    /* TITAN METRIC CARD CSS */
    .titan-card {
        background: #0f0f0f;
        border: 1px solid #222;
        border-left: 4px solid #555;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .titan-card h4 { margin: 0; font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .titan-card h2 { margin: 5px 0 0 0; font-size: 1.4rem; font-weight: 700; color: #fff; } /* Slightly smaller for mobile */
    .titan-card .sub { font-size: 0.7rem; color: #555; margin-top: 5px; }
    
    /* STATUS COLORS */
    .border-bull { border-left-color: #00ffbb !important; }
    .border-bear { border-left-color: #ff1155 !important; }
    .text-bull { color: #00ffbb !important; }
    .text-bear { color: #ff1155 !important; }
    .text-white { color: #fff !important; }
    
    /* AI ANALYSIS BOX */
    .ai-box {
        background: #0a0a0a;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        border-left: 3px solid #7d00ff;
        font-size: 0.9rem;
    }
    
    /* TOASTS */
    div[data-testid="stToast"] { background-color: #1a1a1a; border: 1px solid #333; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. UTILITIES & CALCS
# ==========================================
def send_telegram_msg(token, chat, msg):
    if not token or not chat: return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
        return True
    except: return False

def get_ai_analysis(df_summary, symbol, tf, ai_key):
    if not ai_key: return "⚠️ Missing API Key. Check Config Tab."
    
    try:
        client = OpenAI(api_key=ai_key)
        prompt = f"""
        Act as a Titan Scalper AI. Analyze {symbol} ({tf}):
        Data: {df_summary}
        Output:
        1. Setup Assessment
        2. Confluence Check
        3. Bias (BULL/BEAR/WAIT)
        Keep it under 80 words. Bullet points.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Sniper scalper."}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e: return f"Error: {e}"

# Math Helpers
def weighted_ma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calc_hma_full(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = weighted_ma(series, half_length)
    wma_full = weighted_ma(series, length)
    diff = 2 * wma_half - wma_full
    return weighted_ma(diff, sqrt_length)

def calculate_rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_mfi(high, low, close, volume, length):
    tp = (high + low + close) / 3
    rmf = tp * volume
    pos = rmf.where(tp > tp.shift(1), 0).rolling(length).sum()
    neg = rmf.where(tp < tp.shift(1), 0).rolling(length).sum()
    return 100 - (100 / (1 + (pos / neg)))

# ==========================================
# 4. DATA ENGINE
# ==========================================
@st.cache_data(ttl=10) 
def get_data(symbol, timeframe, limit):
    try:
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except: return pd.DataFrame()

def run_titan_engine(df, amplitude, channel_dev, hma_len, mf_len, vol_len, hyper_long, hyper_short):
    # Dark Vector Logic
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr_algo'] = (df['tr'].ewm(alpha=1/100, adjust=False).mean() / 2)
    df['dev'] = df['atr_algo'] * channel_dev
    df['hma'] = calc_hma_full(df['close'], hma_len)

    # Staircase Logic
    df['ll'] = df['low'].rolling(amplitude).min()
    df['hh'] = df['high'].rolling(amplitude).max()
    
    trend = np.zeros(len(df)); stop = np.zeros(len(df))
    curr_trend = 0; curr_stop = df['close'].iloc[0]
    curr_max_l = 0.0; curr_min_h = 0.0

    # Fast numba-like loop
    for i in range(amplitude, len(df)):
        c = df['close'].iloc[i]; l = df['ll'].iloc[i]; h = df['hh'].iloc[i]
        dev = df['dev'].iloc[i] if not np.isnan(df['dev'].iloc[i]) else 0
        
        curr_max_l = max(l, curr_max_l)
        curr_min_h = min(h, curr_min_h) if curr_min_h != 0 else h

        if curr_trend == 0: # Bull
            if c < curr_max_l: curr_trend = 1; curr_min_h = h
            else: 
                curr_min_h = min(curr_min_h, h)
                if l < curr_max_l: curr_max_l = l
        else: # Bear
            if c > curr_min_h: curr_trend = 0; curr_max_l = l
            else: 
                curr_max_l = max(curr_max_l, l)
                if h > curr_min_h: curr_min_h = h
        
        if curr_trend == 0: # Bull Stop
            s = l + dev
            if s < curr_stop: s = curr_stop
            curr_stop = s
        else: # Bear Stop
            s = h - dev
            if s > curr_stop and curr_stop != 0: s = curr_stop
            elif curr_stop == 0: s = h - dev
            curr_stop = s
            
        trend[i] = curr_trend
        stop[i] = curr_stop

    df['trend'] = trend; df['trend_stop'] = stop
    df['is_bull'] = df['trend'] == 0
    df['bull_flip'] = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False).astype(bool))
    df['bear_flip'] = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True).astype(bool))

    # Money Flow Matrix
    rsi_src = calculate_rsi(df['close'], mf_len) - 50
    mf_vol = df['volume'] / df['volume'].rolling(mf_len).mean()
    df['money_flow'] = (rsi_src * mf_vol).fillna(0).ewm(span=3).mean()
    
    # Hyper Wave
    pc = df['close'].diff()
    ss = pc.ewm(span=hyper_long).mean().ewm(span=hyper_short).mean()
    ss_abs = abs(pc).ewm(span=hyper_long).mean().ewm(span=hyper_short).mean()
    df['hyper_wave'] = np.where(ss_abs != 0, (100 * (ss / ss_abs)) / 2, 0)

    # Metrics
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_len).mean()
    return df

# ==========================================
# 5. MOBILE UI STRUCTURE
# ==========================================

if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = None

# TABS FOR NAVIGATION
tab1, tab2, tab3 = st.tabs(["⚡ Trade", "📊 Deep Dive", "⚙️ Config"])

# --- TAB 3: CONFIG (INPUTS FIRST FOR DATA) ---
with tab3:
    st.header("Titan Configuration")
    
    with st.expander("📡 Market Feed", expanded=True):
        symbol = st.text_input("Symbol (Kraken)", value="BTC/USD") 
        timeframe = st.selectbox("Timeframe", options=['1m', '5m', '15m'], index=1)
        limit = st.slider("Candles", 200, 1500, 500)

    with st.expander("🧠 Logic Engine"):
        amplitude = st.number_input("Sensitivity", min_value=1, value=5)
        channel_dev = st.number_input("Stop Deviation", value=3.0)
        hma_len = st.number_input("HMA Length", value=50)
        mf_len = 14; vol_len = 20
        
    with st.expander("🔑 Keys & Alerts", expanded=True):
        tg_on = st.checkbox("Telegram Active", value=True)
        
        # --- ROBUST KEY HANDLING ---
        # 1. Telegram
        try:
            default_bot = st.secrets["TELEGRAM_TOKEN"]
            default_chat = st.secrets["TELEGRAM_CHAT_ID"]
        except:
            default_bot = ""; default_chat = ""
        
        bot_token = st.text_input("Bot Token", value=default_bot, type="password")
        chat_id = st.text_input("Chat ID", value=default_chat)

        # 2. OpenAI
        try:
            default_ai = st.secrets["OPENAI_API_KEY"]
            st.success("OpenAI Key Found in Secrets! ✅")
        except:
            default_ai = ""
            
        ai_key = st.text_input("OpenAI Key", value=default_ai, type="password", help="Press ENTER after pasting!")

# --- FETCH DATA ---
df = get_data(symbol, timeframe, limit)

# --- TAB 1: TRADE (ACTION CENTER) ---
with tab1:
    if not df.empty:
        df = run_titan_engine(df, amplitude, channel_dev, hma_len, mf_len, vol_len, 25, 13)
        last = df.iloc[-1]
        
        # 1. HEADER STATUS
        st.caption(f"LIVE | {symbol} | {timeframe}")
        
        # 2. METRICS GRID (2x2 for Mobile)
        c1, c2 = st.columns(2)
        trend_lbl = "BULLISH" if last['is_bull'] else "BEARISH"
        trend_cls = "border-bull" if last['is_bull'] else "border-bear"
        trend_txt = "text-bull" if last['is_bull'] else "text-bear"
        
        with c1:
            st.markdown(f"""
            <div class="titan-card {trend_cls}">
                <h4>Price</h4>
                <h2>${last['close']:,.2f}</h2>
                <div class="sub">Trend: <span class="{trend_txt}"><b>{trend_lbl}</b></span></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="titan-card">
                <h4>Money Flow</h4>
                <h2 class="{'text-bull' if last['money_flow']>0 else 'text-bear'}">{last['money_flow']:.2f}</h2>
                <div class="sub">Inst. Pressure</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="titan-card {trend_cls}">
                <h4>Titan Stop</h4>
                <h2>${last['trend_stop']:,.2f}</h2>
                <div class="sub">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="titan-card">
                <h4>RVOL</h4>
                <h2 class="{'text-bull' if last['rvol']>1.5 else 'text-white'}">{last['rvol']:.2f}x</h2>
                <div class="sub">Volume Anomaly</div>
            </div>
            """, unsafe_allow_html=True)

        # 3. ACTION BUTTONS
        st.markdown("---")
        if st.button("🔥 Broadcast Signal", type="primary"):
            is_bull = last['is_bull']
            direction = "LONG" if is_bull else "SHORT"
            icon = "🟢" if is_bull else "🔴"
            msg = f"{icon} *TITAN MOBILE: {symbol}*\nDir: *{direction}*\nPrice: ${last['close']}\nStop: ${last['trend_stop']}"
            if send_telegram_msg(bot_token, chat_id, msg):
                st.success("Sent!")
            else:
                st.error("Check Keys")

        if st.button("🤖 AI Analysis"):
            with st.spinner("Analyzing..."):
                summary = {'price': last['close'], 'trend': trend_lbl, 'stop': last['trend_stop'], 'mf': last['money_flow']}
                # Pass the key explicitly to the function
                rep = get_ai_analysis(summary, symbol, timeframe, ai_key)
                st.markdown(f"""<div class="ai-box">{rep}</div>""", unsafe_allow_html=True)

    else:
        st.info("Please configure symbol in 'Config' tab.")

# --- TAB 2: DEEP DIVE (CHARTS) ---
with tab2:
    if not df.empty:
        # TRADINGVIEW (Hidden behind expander to save load time/space)
        with st.expander("📈 Live TradingView", expanded=False):
             s_tv = f"KRAKEN:{symbol.replace('/','')}"
             components.html(f"""
                <div class="tradingview-widget-container">
                <div id="tv"></div>
                <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                <script>new TradingView.widget({{
                  "width": "100%", "height": 400, "symbol": "{s_tv}", "interval": "5",
                  "theme": "dark", "style": "1", "toolbar_bg": "#f1f3f6", "hide_side_toolbar": false,
                  "container_id": "tv"
                }});</script></div>
                """, height=410)

        # PLOTLY TITAN CHART
        st.subheader("Titan Analysis")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Row 1: Price + Stops
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        b_stop = df['trend_stop'].where(df['is_bull'], np.nan)
        s_stop = df['trend_stop'].where(~df['is_bull'], np.nan)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=b_stop, mode='lines', line=dict(color='#00ffbb', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=s_stop, mode='lines', line=dict(color='#ff1155', width=2)), row=1, col=1)
        
        # Row 2: Money Flow
        col_mf = np.where(df['money_flow'] >= 0, '#00ffbb', '#ff1155')
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['money_flow'], marker_color=col_mf, name="MF"), row=2, col=1)
        
        fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#000', plot_bgcolor='#000', showlegend=False)
        fig.update_xaxes(visible=False) # Clean look for mobile
        st.plotly_chart(fig, use_container_width=True)
