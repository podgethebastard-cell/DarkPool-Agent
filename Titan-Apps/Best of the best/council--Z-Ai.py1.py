import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import requests
import time as time_lib
import tweepy
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PENTAGRAM")

# --- OpenAI Integration ---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ==========================================
# 1. PAGE CONFIGURATION & UI THEME
# ==========================================
st.set_page_config(
    page_title="THE PENTAGRAM | Omni-Terminal",
    layout="wide",
    page_icon="🔯",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
   
    :root {
        --bg: #050505;
        --card: #0a0a0a;
        --border: #1f1f1f;
        --accent: #D500F9; /* Pentagram Purple */
        --accent-dim: rgba(213, 0, 249, 0.1);
        --bull: #00E676;
        --bear: #FF1744;
        --heat: #FFD600;
        --text: #e0e0e0;
        --text-muted: #666;
    }
   
    /* General App Styling */
    .stApp { 
        background-color: var(--bg); 
        font-family: 'Rajdhani', sans-serif; 
        color: var(--text); 
    }
    .stDeployButton { display: none; }
    
    /* Headers & Text */
    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .font-mono { font-family: 'JetBrains Mono', monospace; }

    /* Metrics Container */
    div[data-testid="metric-container"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: var(--accent);
        box-shadow: 0 0 10px var(--accent-dim);
    }
    label[data-testid="stMetricLabel"] { color: var(--text-muted); font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; }

    /* Custom Diagnostics Panel */
    .diag-panel {
        background: #080808;
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        padding: 15px;
        border-radius: 4px;
        height: 100%;
        font-family: 'JetBrains Mono', monospace;
    }
    .diag-header { 
        font-size: 0.75rem; 
        color: var(--accent); 
        text-transform: uppercase; 
        letter-spacing: 2px; 
        font-weight: 700; 
        margin-bottom: 15px; 
        border-bottom: 1px solid #222;
        padding-bottom: 5px;
    }
    .diag-item { 
        display: flex; justify-content: space-between;
        margin-bottom: 8px; 
        font-size: 0.85rem; 
        padding-bottom: 4px;
        border-bottom: 1px dotted #222;
    }
    .diag-label { color: #888; }
    .diag-val { color: #fff; font-weight: 700; text-align: right; }
    .val-bull { color: var(--bull) !important; }
    .val-bear { color: var(--bear) !important; }

    /* Custom Inputs */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > select {
        background-color: #0c0c0c !important; 
        border: 1px solid #333 !important; 
        color: #fff !important;
        font-family: 'JetBrains Mono', monospace;
    }
    .stSelectbox { color: #fff; }

    /* Signal Log */
    .log-container {
        font-size: 0.75rem; 
        color: #aaa; 
        max-height: 250px; 
        overflow-y: auto;
        border: 1px solid #222; 
        padding: 10px; 
        border-radius: 4px; 
        background: #020202;
        font-family: 'JetBrains Mono', monospace;
    }
    .log-entry { 
        border-bottom: 1px solid #111; 
        padding: 6px 0; 
        display: flex; 
        justify-content: space-between; 
    }
    .log-time { color: var(--accent); margin-right: 10px; }
    .log-msg { color: #fff; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #0a0a0a;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        color: #666;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--card);
        color: var(--accent);
        border-bottom: 2px solid var(--accent);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #000; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. GLOBAL CONNECTIVITY HANDLERS
# ==========================================
def send_telegram(token, chat_id, text):
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
            requests.post(url, json=payload, timeout=5)
            return True
        except Exception as e:
            logger.error(f"Telegram Error: {e}")
            return False
    return False

def post_x(key, secret, at, ats, text):
    if key and at:
        try:
            client = tweepy.Client(consumer_key=key, consumer_secret=secret, access_token=at, access_token_secret=ats)
            client.create_tweet(text=text)
            return True
        except Exception as e:
            logger.error(f"X/Twitter Error: {e}")
            return False
    return False

# ==========================================
# 3. PENTAGRAM MATH ENGINE (ALL 5 CORES)
# ==========================================
def rma(series, length): 
    return series.ewm(alpha=1/length, adjust=False).mean()

def wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(series, length): 
    return wma(2 * wma(series, length // 2) - wma(series, length), int(np.sqrt(length)))

def double_smooth(src, l1, l2): 
    return src.ewm(span=l1).mean().ewm(span=l2).mean()

# CORE 1: APEX VECTOR (Physics Engine)
def calc_vector(df, p):
    df = df.copy()
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    # Efficiency Ratio
    eff = np.where(rng==0, 0, body/rng)
    df["eff"] = pd.Series(eff).ewm(span=p["vec_l"]).mean()
    
    # Volume Factor
    vol_ma = df["volume"].rolling(p["vol_n"]).mean()
    vol_fact = np.where(vol_ma==0, 1, df["volume"]/vol_ma)
    
    # Raw Flux Calculation
    raw_v = np.sign(df["close"] - df["open"]) * df["eff"] * vol_fact
    df["flux"] = raw_v.ewm(span=p["vec_sm"]).mean()
    
    # States
    th_s = p["vec_super"] * p["vec_strict"]
    th_r = p["vec_resist"] * p["vec_strict"]
    
    conditions = [
        (df["flux"] > th_s), 
        (df["flux"] < -th_s), 
        (df["flux"].abs() < th_r)
    ]
    df["vec_state"] = np.select(conditions, [2, -2, 0], default=1)
    
    # Divergence Logic
    df["pl"] = (df["flux"].shift(1) < df["flux"].shift(2)) & (df["flux"].shift(1) < df["flux"])
    df["ph"] = (df["flux"].shift(1) > df["flux"].shift(2)) & (df["flux"].shift(1) > df["flux"])
    df["div_bull"] = df["pl"] & (df["close"] < df["close"].shift(5)) & (df["flux"] > df["flux"].shift(5))
    df["div_bear"] = df["ph"] & (df["close"] > df["close"].shift(5)) & (df["flux"] < df["flux"].shift(5))
    
    return df

# CORE 2: APEX BRAIN (Logical Processing)
def calc_brain(df, p):
    df = df.copy()
    base = hma(df["close"], p["br_l"])
    atr = rma(df["high"]-df["low"], p["br_l"])
    
    df["br_u"] = base + (atr * p["br_m"])
    df["br_l_band"] = base - (atr * p["br_m"])
    
    # Trend State Machine
    trend = np.zeros(len(df))
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["br_u"].iloc[i]: trend[i] = 1
        elif df["close"].iloc[i] < df["br_l_band"].iloc[i]: trend[i] = -1
        else: trend[i] = trend[i-1]
    df["br_trend"] = trend
    
    # Entropy Gate
    df["ent"] = df["close"].pct_change().rolling(64).std() * 100
    df["gate"] = df["ent"] < p["br_th"]
    
    # Flow Calculation (Wick + Body pressure)
    rng = df["high"] - df["low"]
    safe_rng = rma(rng, 14) + 1e-10
    wick = np.where(rng==0, 0, ((np.minimum(df["open"], df["close"]) - df["low"]) - (df["high"] - np.maximum(df["open"], df["close"])))/rng)
    df["flow"] = pd.Series(wick + ((df["close"]-df["open"])/safe_rng)).ewm(span=34).mean()
    
    # Brain Signals
    df["br_buy"] = (df["br_trend"]==1) & df["gate"] & (df["flux"] > 0.5) & (df["flow"] > 0)
    df["br_sell"] = (df["br_trend"]==-1) & df["gate"] & (df["flux"] < -0.5) & (df["flow"] < 0)
    
    return df

# CORE 3: RQZO (Quantum Mechanics) - FIXED VECTORIZATION
def calc_rqzo(df, p):
    df = df.copy()
    # Normalize Price 0-1 over rolling window
    mn, mx = df["close"].rolling(100).min(), df["close"].rolling(100).max()
    norm = (df["close"] - mn) / (mx - mn + 1e-10)
    
    # Gamma Calculation (Lorentz-like factor based on velocity of price change)
    delta = np.clip((norm - norm.shift(1)).abs(), 0, 0.049)
    # Avoid division by zero or negative under sqrt
    gamma = 1 / np.sqrt(np.clip(1 - (delta/0.05)**2, 1e-10, 1.0))
    
    # Tau Time Dilation Index
    # Create a rolling index relative to gamma
    index_vals = np.arange(len(df))
    tau = (index_vals % 100) / gamma
    
    # Quantum Wave Function Summation (Vectorized)
    # Summation of wavelets from n=1 to 9
    rqzo_val = np.zeros(len(df))
    for n in range(1, 10):
        rqzo_val += (n ** -0.5) * np.sin(tau * np.log(n))
    
    df["rqzo"] = rqzo_val * 10
    return df

# CORE 4: MATRIX (Momentum Matrix)
def calc_matrix(df, p):
    df = df.copy()
    # RSI Calculation
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = rma(gain, 14) / (rma(loss, 14) + 1e-10)
    rsi = 100 - (100/(1+rs))
    
    # Money Flow Index integration with Volume
    vol_ratio = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
    df["mfi"] = ((rsi - 50) * vol_ratio).ewm(span=3).mean()
    
    # Hawking Wave (Double Smoothed Momentum)
    hw_num = double_smooth(df["close"].diff(), 25, 13)
    hw_den = double_smooth(df["close"].diff().abs(), 25, 13) + 1e-10
    df["hw"] = 100 * (hw_num / hw_den) / 2
    
    # Matrix Signal
    df["mat_sig"] = np.sign(df["mfi"]) + np.sign(df["hw"])
    return df

# CORE 5: APEX SMC (Structural Master)
def calc_smc(df, p):
    df = df.copy()
    df["smc_base"] = hma(df["close"], p["smc_l"])
    
    # Trend Cycle Index
    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ap.ewm(span=10).mean()
    de = (ap - esa).abs()
    ci = (ap - esa) / (0.015 * de.ewm(span=10).mean() + 1e-10)
    tci = ci.ewm(span=21).mean()
    
    df["smc_buy"] = (df["close"] > df["smc_base"]) & (tci < 60) & (tci > tci.shift(1))
    df["smc_sell"] = (df["close"] < df["smc_base"]) & (tci > -60) & (tci < tci.shift(1))
    
    # Fair Value Gaps
    df["fvg_b"] = df["low"] > df["high"].shift(2)
    df["fvg_s"] = df["high"] < df["low"].shift(2)
    
    return df

# ==========================================
# 4. DATA HANDLING
# ==========================================
@st.cache_data(ttl=60) # Cache for 60 seconds to reduce API load
def get_data(exch, sym, tf, lim):
    try:
        # Normalize Exchange ID
        exch_id = exch.lower()
        if exch_id == "coinbase": exch_id = "coinbasepro" # CCXT legacy fix sometimes needed
        
        exchange_class = getattr(ccxt, exch_id)
        ex = exchange_class({"enableRateLimit": True})
        
        # Fetch OHLCV
        ohlcv = ex.fetch_ohlcv(sym, tf, limit=lim)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        st.error(f"Data Feed Error: {str(e)}")
        return pd.DataFrame()

def init():
    defaults = {
        "exch": "binance", "sym": "BTC/USDT", "tf": "15m", "lim": 500,
        "vec_l": 14, "vol_n": 55, "vec_sm": 5, "vec_super": 0.6, "vec_resist": 0.3, "vec_strict": 1.0,
        "br_l": 55, "br_m": 1.5, "br_th": 2.0, "smc_l": 55, "auto": False,
        "ai_k": "", "tg_t": "", "tg_c": "", "x_k": "", "x_s": "", "x_at": "", "x_as": "",
        "signal_log": []
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init()

# ==========================================
# 5. SIDEBAR CONTROL DECK
# ==========================================
with st.sidebar:
    st.markdown("### 🔯 CONTROL DECK")
    st.markdown("---")
   
    with st.expander("🌍 Feed Configuration", expanded=True):
        st.session_state.exch = st.selectbox("Exchange", ["Binance", "Kraken", "Bybit", "Coinbase", "OKX"])
        st.session_state.sym = st.text_input("Asset Ticker", st.session_state.sym).upper()
        st.session_state.tf = st.selectbox("Interval", ["1m","5m","15m","1h","4h","1d"], index=2)
        
        # Safe Auto-Pilot Implementation
        auto_mode = st.checkbox("🔄 Live Mode", st.session_state.auto, help="Automatically refreshes data on interaction. Does not block UI.")
        st.session_state.auto = auto_mode

    with st.expander("📡 Omnichannel APIs"):
        st.markdown("#### Telegram")
        c1, c2 = st.columns(2)
        with c1: st.session_state.tg_t = st.text_input("Bot Token", st.session_state.tg_t, type="password")
        with c2: st.session_state.tg_c = st.text_input("Chat ID", st.session_state.tg_c)
        st.markdown("#### X / Twitter")
        st.session_state.x_k = st.text_input("API Key", st.session_state.x_k, type="password")
        st.session_state.x_s = st.text_input("API Secret", st.session_state.x_s, type="password")
        st.session_state.x_at = st.text_input("Access Token", st.session_state.x_at, type="password")
        st.session_state.x_as = st.text_input("Access Secret", st.session_state.x_as, type="password")
        
        st.markdown("#### OpenAI")
        st.session_state.ai_k = st.text_input("OpenAI API Key", st.session_state.ai_k, type="password")

    st.markdown("---")
    if st.button("🔱 RELOAD SYSTEM", type="primary", use_container_width=True):
        get_data.clear()
        st.rerun()
        
    # Signal Log Area
    st.markdown("### 📜 Transmission Log")
    log_container = st.container()
    with log_container:
        log_html = '<div class="log-container">'
        if not st.session_state.signal_log:
            log_html += '<div class="log-entry"><span class="log-msg">System Ready...</span></div>'
        else:
            for entry in reversed(st.session_state.signal_log[-20:]): # Show last 20
                log_html += f'<div class="log-entry"><span class="log-time">{entry["time"]}</span><span class="log-msg">{entry["msg"]}</span></div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

# ==========================================
# 6. PROCESSING & BROADCASTING
# ==========================================
df = get_data(st.session_state.exch, st.session_state.sym, st.session_state.tf, st.session_state.lim)

if df.empty:
    st.error("VOID DETECTED: Check Exchange or Symbol Format.")
    st.stop()

# Ensure Dataframe has enough rows for calculations
if len(df) < 150:
    st.warning("Insufficient data for full analysis. Gathering historical bars...")
    st.stop()

# Chain computation
df = calc_vector(df, st.session_state)
df = calc_brain(df, st.session_state)
df = calc_rqzo(df, st.session_state)
df = calc_matrix(df, st.session_state)
df = calc_smc(df, st.session_state)

last, prev = df.iloc[-1], df.iloc[-2]
curr_time = datetime.datetime.now().strftime("%H:%M:%S")

# Broadcast events
event = None
if last["br_buy"] and not prev["br_buy"]: 
    event = f"🧠 BRAIN LONG: {st.session_state.sym} @ {last['close']}"
elif last["smc_buy"] and not prev["smc_buy"]: 
    event = f"🏛️ SMC BUY: {st.session_state.sym} @ {last['close']}"
elif last["vec_state"] == 2 and prev["vec_state"] != 2: 
    event = f"⚡ SUPERCONDUCTOR: {st.session_state.sym}"
elif last["div_bull"]: 
    event = f"📈 BULL DIVERGENCE: {st.session_state.sym}"

if event:
    # Update Log
    st.session_state.signal_log.append({"time": curr_time, "msg": event})
    
    # Send Notifications
    tg_status = send_telegram(st.session_state.tg_t, st.session_state.tg_c, event)
    x_status = post_x(st.session_state.x_k, st.session_state.x_s, st.session_state.x_at, st.session_state.x_as, event)
    
    # Force UI Update for log
    if st.session_state.auto:
        st.rerun()

# ==========================================
# 7. ORDERLY UI RENDER
# ==========================================
st.title(f"🔯 THE PENTAGRAM // {st.session_state.sym}")

# Global HUD
h1, h2, h3, h4 = st.columns(4)
with h1:
    color = "var(--bull)" if last['close'] >= last['open'] else "var(--bear)"
    st.metric("Live Price", f"{last['close']:.2f}", delta=f"{(last['close']-last['open']):.2f}", delta_color="normal")
with h2:
    flux_color = "var(--bull)" if last['flux']>0 else "var(--bear)"
    st.markdown(f"<div style='text-align:center; color:{color}'>Apex Flux<br><span style='font-size:1.5rem; font-weight:700'>{last['flux']:.3f}</span></div>", unsafe_allow_html=True)
with h3:
    brain_state = "<span style='color:var(--bull)'>SAFE</span>" if last['gate'] else "<span style='color:var(--bear)'>CHAOS</span>"
    st.markdown(f"<div style='text-align:center'>Brain State<br><span style='font-size:1.2rem; font-weight:700'>{brain_state}</span></div>", unsafe_allow_html=True)
with h4:
    mat_color = "var(--bull)" if last['mat_sig'] > 0 else "var(--bear)"
    st.markdown(f"<div style='text-align:center; color:{mat_color}'>Matrix Sig<br><span style='font-size:1.5rem; font-weight:700'>{int(last['mat_sig'])}</span></div>", unsafe_allow_html=True)

# --- Dashboard Tabs ---
t1, t2, t3, t4, t5, t6 = st.tabs(["🧠 Brain", "🏛️ Structure", "⚡ Vector", "⚛️ Quantum", "📺 TV View", "🤖 AI Council"])

def clean_plot():
    return go.Layout(
        template="plotly_dark", 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        margin=dict(l=0, r=0, t=30, b=0), 
        height=500,
        font=dict(color="#e0e0e0", family="Rajdhani"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a", zeroline=False)
    )

# TAB 1: BRAIN
with t1:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name="Price"
        )])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_u'], line=dict(color='rgba(0,230,118,0.3)', width=1), name="Upper Band"))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_l_band'], line=dict(color='rgba(255,23,68,0.3)', width=1), name="Lower Band", fill='tonexty', fillcolor='rgba(255,255,255,0.02)'))
        
        # Add Buy/Sell Markers
        buys = df[df['br_buy']]
        sells = df[df['br_sell']]
        fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low'], mode='markers', marker=dict(symbol='triangle-up', color='#00E676', size=10), name="Brain Buy"))
        fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high'], mode='markers', marker=dict(symbol='triangle-down', color='#FF1744', size=10), name="Brain Sell"))

        fig.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Brain Diagnostics</div>
        <div class="diag-item"><span class="diag-label">Macro Bias:</span><span class="diag-val {'val-bull' if last['br_trend']==1 else 'val-bear'}">{('BULLISH' if last['br_trend']==1 else 'BEARISH')}</span></div>
        <div class="diag-item"><span class="diag-label">Entropy:</span><span class="diag-val">{last['ent']:.2f}</span></div>
        <div class="diag-item"><span class="diag-label">Flow:</span><span class="diag-val">{last['flow']:.3f}</span></div>
        <div style="margin-top:20px; font-size:0.75rem; color:#555; line-height:1.4">
        Brain Core analyzes neural trend clouds and entropy gates to filter noise. Only triggers when Flow > 0.
        </div></div>""", unsafe_allow_html=True)

# TAB 2: STRUCTURE (SMC)
with t2:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['smc_base'], line=dict(color='cyan', width=1), name="SMC Base"))
        
        # FVGs
        fvg_b = df[df['fvg_b']]
        fvg_s = df[df['fvg_s']]
        
        # Add FVG rectangles (requires loop or complex shapes, simplified here with lines for speed)
        # Visualizing Bull FVGs
        for i, row in fvg_b.iterrows():
            fig.add_hrect(y0=row['low'].shift(-2).iloc[i] if i > 1 else row['low'], y1=row['low'], 
                          x0=row['timestamp'], x1=row['timestamp'] + pd.Timedelta(minutes=15), # rough estimation
                          fillcolor="rgba(0, 230, 118, 0.1)", line_width=0)

        fig.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        sig_txt = "BUY" if last['smc_buy'] else ("SELL" if last['smc_sell'] else "WAIT")
        sig_col = "var(--bull)" if last['smc_buy'] else ("var(--bear)" if last['smc_sell'] else "#888")
        
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Structure Diagnostics</div>
        <div class="diag-item"><span class="diag-label">Signal:</span><span class="diag-val" style="color:{sig_col}">{sig_txt}</span></div>
        <div class="diag-item"><span class="diag-label">Base Price:</span><span class="diag-val">{last['smc_base']:.2f}</span></div>
        <div class="diag-item"><span class="diag-label">Bull FVG:</span><span class="diag-val">{'Active' if last['fvg_b'] else 'None'}</span></div>
        <div class="diag-item"><span class="diag-label">Bear FVG:</span><span class="diag-val">{'Active' if last['fvg_s'] else 'None'}</span></div>
        </div>""", unsafe_allow_html=True)

# TAB 3: VECTOR (PHYSICS)
with t3:
    l, r = st.columns([3, 1])
    with l:
        fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.6, 0.4])
        fig_v.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
        
        colors = ["#00E676" if x==2 else ("#FF1744" if x==-2 else "#333") for x in df["vec_state"]]
        fig_v.add_trace(go.Bar(x=df['timestamp'], y=df['flux'], marker_color=colors, name="Flux"), row=2, col=1)
        fig_v.add_hline(y=0, row=2, col=1, line_color="#666", line_width=1)
        
        fig_v.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_v, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Physics Diagnostics</div>
        <div class="diag-item"><span class="diag-label">Efficiency:</span><span class="diag-val">{last['eff']*100:.1f}%</span></div>
        <div class="diag-item"><span class="diag-label">Flux Force:</span><span class="diag-val">{last['flux']:.3f}</span></div>
        <div class="diag-item"><span class="diag-label">State:</span><span class="diag-val">{last['vec_state']}</span></div>
        <div style="margin-top:20px; font-size:0.75rem; color:#555; line-height:1.4">
        Vector Core calculates the physical force of the market via geometric efficiency and volume vectors.
        </div></div>""", unsafe_allow_html=True)

# TAB 4: QUANTUM (RQZO) - NEW IMPLEMENTATION
with t4:
    l, r = st.columns([3, 1])
    with l:
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=df['timestamp'], y=df['rqzo'], mode='lines', line=dict(color='#D500F9', width=2), name="Quantum Wave"))
        fig_q.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_q.add_hrect(y0=10, y1=20, fillcolor="rgba(213, 0, 249, 0.1)", line_width=0, annotation_text="Superposition")
        fig_q.add_hrect(y0=-20, y1=-10, fillcolor="rgba(213, 0, 249, 0.1)", line_width=0)
        
        fig_q.update_layout(clean_plot(), yaxis_title="Amplitude")
        st.plotly_chart(fig_q, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Quantum Diagnostics</div>
        <div class="diag-item"><span class="diag-label">Amplitude:</span><span class="diag-val">{last['rqzo']:.4f}</span></div>
        <div class="diag-item"><span class="diag-label">Gamma:</span><span class="diag-val">Active</span></div>
        <div style="margin-top:20px; font-size:0.75rem; color:#555; line-height:1.4">
        RQZO Core visualizes the probabilistic wave function of price action using time dilation factors.
        </div></div>""", unsafe_allow_html=True)

# TAB 5: TRADINGVIEW
with t5:
    st.markdown("### Official TradingView Chart Confirmation")
    tv_s = st.session_state.sym.replace("/", "")
    # Map simple names to TV specific symbols if needed
    tv_sym = f"BINANCE:{tv_s}" if st.session_state.exch == "Binance" else f"KRAKEN:{tv_s}"
    
    components_html = f"""
    <div class="tradingview-widget-container" style="height:500px; width: 100%;">
      <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {{
      "autosize": true,
      "symbol": "{tv_sym}",
      "interval": "{st.session_state.tf}",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "enable_publishing": false,
      "hide_side_toolbar": false,
      "allow_symbol_change": true,
      "calendar": false,
      "support_host": "https://www.tradingview.com"
      }}
      </script>
    </div>
    """
    st.components.v1.html(components_html, height=540)

# TAB 6: AI COUNCIL
with t6:
    st.markdown("### 🤖 AI Strategy Council")
    c1, c2 = st.columns(2)
    with c1:
        persona = st.selectbox("Select Council Member", 
            ["Grand Strategist (Macro Bias)", "Physicist (Momentum Analysis)", "Risk Manager (Drawdown Calc)", "The Quant (Math Review)"])
    with c2:
        if st.button("🔱 CONSULT ORACLE", type="primary"):
            if OPENAI_AVAILABLE and st.session_state.ai_k:
                with st.spinner("Consulting the neural net..."):
                    try:
                        c = OpenAI(api_key=st.session_state.ai_k)
                        
                        # Construct Context
                        context = f"""
                        Asset: {st.session_state.sym}, Price: {last['close']}
                        Vector State: {last['vec_state']}, Flux: {last['flux']}
                        Brain Trend: {last['br_trend']}, Entropy Gate: {last['gate']}
                        SMC Signal: {'Buy' if last['smc_buy'] else ('Sell' if last['smc_sell'] else 'Neutral')}
                        Matrix Sig: {last['mat_sig']}
                        """
                        
                        system_prompt = "You are an expert crypto trading analyst. Be concise, technical, and decisive."
                        
                        response = c.chat.completions.create(
                            model="gpt-4o", 
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Acting as {persona}. Analyze these metrics: {context}. Provide Bias (Bullish/Bearish), Entry Price, Stop Loss, and a brief reasoning."}
                            ]
                        )
                        st.success(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Oracle Link Failed: {e}")
            else:
                st.error("OpenAI API Key missing or library not installed.")

# Auto-refresh logic at end of script
if st.session_state.auto:
    time_lib.sleep(2) # Short delay to prevent rate limiting, but non-blocking enough
    st.rerun()
