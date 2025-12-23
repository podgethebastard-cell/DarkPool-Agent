

Here is the fully rebranded version of the dashboard, renamed to **QUINTESSENCE**.

I have updated all UI elements, the scoring system (now the **Quintessence Index**), and the general aesthetic to reflect the concept of the "Fifth Element" or "Essence."

```python
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

# ==========================================
# SYSTEM CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QUINTESSENCE_CORE")

# --- Imports ---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ==========================================
# 1. UI THEME & CSS (The "Essence" Upgrade)
# ==========================================
st.set_page_config(
    page_title="QUINTESSENCE | Omni-Terminal",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@500;600;700&display=swap');
   
    :root {
        --bg-dark: #020202;
        --bg-panel: #0a0a0a;
        --border: #222222;
        --accent-primary: #D500F9; /* Electric Purple - The "Essence" Color */
        --accent-sec: #00E5FF;     /* Cyan */
        --bull: #00E676;
        --bear: #FF1744;
        --text-main: #eeeeee;
        --text-dim: #666666;
    }
   
    /* Global Resets */
    .stApp { background-color: var(--bg-dark); font-family: 'Rajdhani', sans-serif; color: var(--text-main); }
    .stDeployButton { display: none !important; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

    /* Typography */
    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; text-transform: uppercase; letter-spacing: 1.5px; }
    .mono { font-family: 'JetBrains Mono', monospace; }

    /* Sidebar Elements */
    .css-1d391kg { padding-top: 2rem; }
    .stSelectbox > div > div > select, 
    .stTextInput > div > div > input {
        background-color: #111 !important; 
        border: 1px solid #333 !important; 
        color: #fff !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Cards (Metrics) */
    div[data-testid="metric-container"] {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        padding: 1rem;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"]:hover {
        border-color: var(--accent-primary);
        transition: 0.3s;
    }
    label[data-testid="stMetricLabel"] { color: var(--text-dim); font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; }
    
    /* Custom Panel (Diagnostics) */
    .diag-box {
        background: #080808;
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent-sec);
        padding: 1rem;
        border-radius: 4px;
        height: 100%;
    }
    .diag-row { display: flex; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px solid #111; padding-bottom: 4px; font-size: 0.85rem; }
    .diag-label { color: #777; }
    .diag-val { font-family: 'JetBrains Mono', monospace; font-weight: 700; }
    .bull-text { color: var(--bull); }
    .bear-text { color: var(--bear); }

    /* Logs */
    .sys-log {
        background: #000;
        border: 1px solid #333;
        height: 250px;
        overflow-y: auto;
        padding: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #aaa;
        border-radius: 4px;
    }
    .log-item { margin-bottom: 5px; border-bottom: 1px dashed #222; padding-bottom: 2px; }
    .log-ts { color: var(--accent-primary); margin-right: 10px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background: #000; padding: 5px; }
    .stTabs [data-baseweb="tab"] { 
        background: transparent; color: #555; padding: 10px 20px; 
        font-weight: 700; text-transform: uppercase; border-radius: 4px;
    }
    .stTabs [aria-selected="true"] { background: var(--bg-panel); color: #fff; border: 1px solid #333; }

    /* Quintessence Score Gauge Simulation */
    .score-container { text-align: center; padding: 10px; }
    .score-val { font-size: 2.5rem; font-weight: 800; line-height: 1; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS (Comms)
# ==========================================
def broadcast_message(token, chat_id, text, service="telegram", keys=None):
    """
    Broadcasts message to external services.
    For X/Twitter, keys must be a dict: {k: ..., s: ..., at: ..., ats: ...}
    """
    if service == "telegram":
        if not token or not chat_id: return False
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=5)
            return True
        except Exception as e:
            logger.error(f"Telegram failed: {e}")
            return False
            
    elif service == "twitter":
        if not keys: return False
        try:
            client = tweepy.Client(
                consumer_key=keys['k'], 
                consumer_secret=keys['s'], 
                access_token=keys['at'], 
                access_token_secret=keys['ats']
            )
            client.create_tweet(text=text)
            return True
        except Exception as e:
            logger.error(f"X/Twitter failed: {e}")
            return False
    return False

# ==========================================
# 3. THE QUINTESSENCE MATH ENGINE
# ==========================================
def sanitize(arr):
    """Helper to remove NaNs and Infs safely."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def rma(src, length): 
    return src.ewm(alpha=1/length, adjust=False).mean()

def wma(src, length):
    weights = np.arange(1, length + 1)
    return src.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(src, length): 
    return wma(2 * wma(src, length // 2) - wma(src, length), int(np.sqrt(length)))

def double_smooth(src, l1, l2): 
    return src.ewm(span=l1).mean().ewm(span=l2).mean()

# --- CORE 1: VECTOR (Physics) ---
def calc_vector(df, p):
    df = df.copy()
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    
    eff = sanitize(np.where(rng==0, 0, body/rng))
    df["eff"] = pd.Series(eff).ewm(span=p["vec_l"]).mean()
    
    vol_ma = sanitize(df["volume"].rolling(p["vol_n"]).mean())
    vol_fact = sanitize(np.where(vol_ma==0, 1, df["volume"]/vol_ma))
    vol_fact = np.clip(vol_fact, 0, 5)
    
    raw_v = np.sign(df["close"] - df["open"]) * df["eff"] * vol_fact
    df["flux"] = sanitize(raw_v.ewm(span=p["vec_sm"]).mean())
    
    th_s = p["vec_super"] * p["vec_strict"]
    th_r = p["vec_resist"] * p["vec_strict"]
    
    conds = [
        (df["flux"] > th_s), 
        (df["flux"] < -th_s), 
        (df["flux"].abs() < th_r)
    ]
    df["vec_state"] = np.select(conds, [2, -2, 0], default=1)
    
    df["pl"] = (df["flux"].shift(1) < df["flux"].shift(2)) & (df["flux"].shift(1) < df["flux"])
    df["ph"] = (df["flux"].shift(1) > df["flux"].shift(2)) & (df["flux"].shift(1) > df["flux"])
    df["div_bull"] = df["pl"] & (df["close"] < df["close"].shift(5)) & (df["flux"] > df["flux"].shift(5))
    df["div_bear"] = df["ph"] & (df["close"] > df["close"].shift(5)) & (df["flux"] < df["flux"].shift(5))
    return df

# --- CORE 2: BRAIN (Logic) ---
def calc_brain(df, p):
    df = df.copy()
    base = hma(df["close"], p["br_l"])
    atr = rma(df["high"]-df["low"], p["br_l"])
    
    df["br_u"] = sanitize(base + (atr * p["br_m"]))
    df["br_l_band"] = sanitize(base - (atr * p["br_m"]))
    
    trend = np.zeros(len(df))
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["br_u"].iloc[i]: trend[i] = 1
        elif df["close"].iloc[i] < df["br_l_band"].iloc[i]: trend[i] = -1
        else: trend[i] = trend[i-1]
    df["br_trend"] = trend
    
    df["ent"] = sanitize(df["close"].pct_change().rolling(64).std() * 100)
    df["gate"] = df["ent"] < p["br_th"]
    
    rng = df["high"] - df["low"]
    safe_rng = sanitize(rma(rng, 14)) + 1e-10
    wick = sanitize(np.where(rng==0, 0, ((np.minimum(df["open"], df["close"]) - df["low"]) - (df["high"] - np.maximum(df["open"], df["close"])))/rng))
    df["flow"] = sanitize(pd.Series(wick + ((df["close"]-df["open"])/safe_rng)).ewm(span=34).mean())
    
    df["br_buy"] = (df["br_trend"]==1) & df["gate"] & (df["flux"] > 0.5) & (df["flow"] > 0)
    df["br_sell"] = (df["br_trend"]==-1) & df["gate"] & (df["flux"] < -0.5) & (df["flow"] < 0)
    return df

# --- CORE 3: RQZO (Quantum) ---
def calc_rqzo(df, p):
    df = df.copy()
    mn, mx = sanitize(df["close"].rolling(100).min()), sanitize(df["close"].rolling(100).max())
    norm = sanitize((df["close"] - mn) / (mx - mn + 1e-10))
    
    delta = sanitize(np.clip((norm - norm.shift(1)).abs(), 0, 0.049))
    gamma = sanitize(1 / np.sqrt(np.clip(1 - (delta/0.05)**2, 1e-10, 1.0)))
    
    index_vals = np.arange(len(df))
    tau = (index_vals % 100) / gamma
    
    rqzo_val = np.zeros(len(df))
    for n in range(1, 10):
        rqzo_val += sanitize((n ** -0.5) * np.sin(tau * np.log(n)))
    
    df["rqzo"] = sanitize(rqzo_val * 10)
    return df

# --- CORE 4: MATRIX (Momentum) ---
def calc_matrix(df, p):
    df = df.copy()
    delta = sanitize(df["close"].diff())
    gain = sanitize(delta.clip(lower=0))
    loss = sanitize(-delta.clip(upper=0))
    rs = rma(gain, 14) / (rma(loss, 14) + 1e-10)
    rsi = 100 - (100/(1+rs))
    
    vol_ratio = sanitize(df["volume"] / (df["volume"].rolling(20).mean() + 1e-10))
    df["mfi"] = sanitize(((rsi - 50) * vol_ratio).ewm(span=3).mean())
    
    hw_num = sanitize(double_smooth(delta, 25, 13))
    hw_den = sanitize(double_smooth(delta.abs(), 25, 13)) + 1e-10
    df["hw"] = sanitize(100 * (hw_num / hw_den) / 2)
    
    df["mat_sig"] = sanitize(np.sign(df["mfi"]) + np.sign(df["hw"]))
    return df

# --- CORE 5: SMC (Structure) ---
def calc_smc(df, p):
    df = df.copy()
    df["smc_base"] = hma(df["close"], p["smc_l"])
    
    ap = sanitize((df["high"] + df["low"] + df["close"]) / 3)
    esa = ap.ewm(span=10).mean()
    de = sanitize((ap - esa).abs())
    tci = sanitize((ap - esa) / (0.015 * de.ewm(span=10).mean() + 1e-10)).ewm(span=21).mean()
    
    df["smc_buy"] = (df["close"] > df["smc_base"]) & (tci < 60) & (tci > tci.shift(1))
    df["smc_sell"] = (df["close"] < df["smc_base"]) & (tci > -60) & (tci < tci.shift(1))
    
    df["fvg_b"] = df["low"] > df["high"].shift(2)
    df["fvg_s"] = df["high"] < df["low"].shift(2)
    return df

# ==========================================
# 4. DATA LAYER
# ==========================================
@st.cache_data(ttl=50) 
def get_data_feed(exch, sym, tf, lim):
    try:
        ex = getattr(ccxt, exch.lower())({"enableRateLimit": True, "timeout": 10000})
        ohlcv = ex.fetch_ohlcv(sym, tf, limit=lim)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        logger.error(f"Feed Error: {e}")
        return pd.DataFrame()

def init_session():
    defaults = {
        "exch": "binance", "sym": "BTC/USDT", "tf": "15m", "lim": 500,
        "vec_l": 14, "vol_n": 55, "vec_sm": 5, "vec_super": 0.6, "vec_resist": 0.3, "vec_strict": 1.0,
        "br_l": 55, "br_m": 1.5, "br_th": 2.0, "smc_l": 55, "auto": False,
        "tg_t": "", "tg_c": "", "x_k": "", "x_s": "", "x_at": "", "x_as": "", "ai_k": "",
        "logs": []
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session()

# ==========================================
# 5. SIDEBAR CONTROL DECK
# ==========================================
with st.sidebar:
    st.markdown("### 💠 QUINTESSENCE")
    st.markdown("#### SYSTEM CONTROLS")
    st.markdown("---")
    
    with st.expander("⚙️ Feed Config", expanded=True):
        st.session_state.exch = st.selectbox("Exchange", ["Binance", "Kraken", "Bybit", "Coinbase", "OKX"])
        st.session_state.sym = st.text_input("Ticker", st.session_state.sym).upper()
        st.session_state.tf = st.selectbox("Timeframe", ["1m","5m","15m","1h","4h","1d"], index=2)
        
        st.markdown("**Live Data Mode**")
        st.session_state.auto = st.checkbox("Enable Auto-Refresh", st.session_state.auto, help="Refreshes data stream automatically.")
        
    with st.expander("📡 API Keys"):
        st.markdown("##### Telegram")
        st.session_state.tg_t = st.text_input("Bot Token", st.session_state.tg_t, type="password")
        st.session_state.tg_c = st.text_input("Chat ID", st.session_state.tg_c)
        
        st.markdown("##### OpenAI")
        st.session_state.ai_k = st.text_input("API Key", st.session_state.ai_k, type="password")
        
    st.markdown("---")
    if st.button("🔄 SYSTEM RESET", use_container_width=True):
        get_data_feed.clear()
        st.rerun()
        
    st.markdown("### 📜 ESSENCE LOG")
    log_html = '<div class="sys-log">'
    for entry in reversed(st.session_state.logs[-30:]):
        log_html += f'<div class="log-item"><span class="log-ts">[{entry["t"]}]</span><span>{entry["m"]}</span></div>'
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)

# ==========================================
# 6. PROCESSING PIPELINE
# ==========================================
# Fetch Data
df = get_data_feed(st.session_state.exch, st.session_state.sym, st.session_state.tf, st.session_state.lim)

if df.empty:
    st.error("CONNECTION FAILED: Check Network or Symbol.")
    st.stop()

if len(df) < 150:
    st.warning("BUFFERING: Gathering historical data...")
    st.stop()

# Compute Cores
try:
    df = calc_vector(df, st.session_state)
    df = calc_brain(df, st.session_state)
    df = calc_rqzo(df, st.session_state)
    df = calc_matrix(df, st.session_state)
    df = calc_smc(df, st.session_state)
except Exception as e:
    st.error(f"CALCULATION ERROR: {e}")
    st.stop()

last, prev = df.iloc[-1], df.iloc[-2]
ts_now = datetime.datetime.now().strftime("%H:%M:%S")

# --- QUINTESSENCE AGGREGATOR (The "Essence" Score) ---
# Score range 0 to 100. 50 = Neutral.
# Weights: Brain(25), Vector(25), SMC(20), Matrix(20), Quantum(10)
score = 50

# Brain
if last['br_trend'] == 1: score += 10
elif last['br_trend'] == -1: score -= 10
if last['br_buy']: score += 15
elif last['br_sell']: score -= 15

# Vector
if last['flux'] > 0.5: score += 12.5
elif last['flux'] < -0.5: score -= 12.5

# SMC
if last['smc_buy']: score += 10
elif last['smc_sell']: score -= 10

# Matrix
if last['mat_sig'] > 0: score += 10
elif last['mat_sig'] < 0: score -= 10

# Quantum (If RQZO is high, adds conviction)
if abs(last['rqzo']) > 10: 
    score += (5 if score > 50 else -5)

# Clamp score
score = np.clip(score, 0, 100)

# --- SIGNAL GENERATION ---
event = None
if last["br_buy"] and not prev["br_buy"]: event = f"🧠 BRAIN LONG: {st.session_state.sym} @ {last['close']}"
elif last["smc_buy"] and not prev["smc_buy"]: event = f"🏛️ SMC BUY: {st.session_state.sym} @ {last['close']}"
elif last["div_bull"]: event = f"⚡ VECTOR DIVERGENCE: {st.session_state.sym}"
elif (last["vec_state"] == 2 and prev["vec_state"] != 2): event = f"⚡ SUPER FLUX: {st.session_state.sym}"

if event:
    st.session_state.logs.append({"t": ts_now, "m": event})
    if st.session_state.tg_t:
        broadcast_message(st.session_state.tg_t, st.session_state.tg_c, event)

# ==========================================
# 7. UI RENDERER
# ==========================================
st.markdown(f"### 💠 QUINTESSENCE // {st.session_state.sym} <span style='font-size:0.6em; color:#666'>{ts_now}</span>", unsafe_allow_html=True)

# Top HUD
c1, c2, c3, c4 = st.columns(4)
with c1:
    price_col = "var(--bull)" if last['close'] >= last['open'] else "var(--bear)"
    st.markdown(f"<div style='color:var(--text-dim); font-size:0.8rem; text-transform:uppercase'>Price</div><div style='color:{price_col}; font-size:1.8rem; font-weight:700'>{last['close']:.2f}</div>", unsafe_allow_html=True)
with c2:
    score_col = "var(--bull)" if score > 55 else ("var(--bear)" if score < 45 else "#fff")
    st.markdown(f"<div style='color:var(--text-dim); font-size:0.8rem; text-transform:uppercase'>Quintessence Index</div><div class='score-container'><div class='score-val' style='color:{score_col}'>{int(score)}</div></div>", unsafe_allow_html=True)
with c3:
    flux_col = "var(--bull)" if last['flux'] > 0 else "var(--bear)"
    st.markdown(f"<div style='color:var(--text-dim); font-size:0.8rem; text-transform:uppercase'>Apex Flux</div><div style='color:{flux_col}; font-size:1.5rem; font-weight:700'>{last['flux']:.3f}</div>", unsafe_allow_html=True)
with c4:
    gate_txt = "OPEN" if last['gate'] else "LOCKED"
    gate_col = "var(--bull)" if last['gate'] else "var(--bear)"
    st.markdown(f"<div style='color:var(--text-dim); font-size:0.8rem; text-transform:uppercase'>Brain Gate</div><div style='color:{gate_col}; font-size:1.5rem; font-weight:700'>{gate_txt}</div>", unsafe_allow_html=True)

# Layout Definition
t1, t2, t3, t4, t5, t6 = st.tabs(["🧠 BRAIN", "🏛️ STRUCTURE", "⚡ VECTOR", "⚛️ QUANTUM", "📺 LIVE TV", "🤖 AI COUNCIL"])

def base_layout():
    return go.Layout(
        template="plotly_dark", 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        margin=dict(l=0, r=0, t=0, b=0), 
        height=500,
        font=dict(color="#888"),
        xaxis=dict(showgrid=False, zeroline=False, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor="#111", zeroline=False)
    )

# --- TAB 1: BRAIN ---
with t1:
    col_chart, col_diag = st.columns([3, 1])
    with col_chart:
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='#00E676', decreasing_line_color='#FF1744',
            name="Price"
        )])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_u'], line=dict(color='#00E676', width=1, dash='dot'), name="Bull Band"))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_l_band'], line=dict(color='#FF1744', width=1, dash='dot'), name="Bear Band"))
        fig.update_layout(base_layout())
        st.plotly_chart(fig, use_container_width=True)
    
    with col_diag:
        st.markdown(f"""
        <div class="diag-box">
            <div style='color:var(--accent-sec); font-weight:700; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px'>DIAGNOSTICS</div>
            <div class="diag-row"><span class="diag-label">Trend</span><span class="diag-val {'bull-text' if last['br_trend']==1 else 'bear-text'}">{('BULLISH' if last['br_trend']==1 else 'BEARISH')}</span></div>
            <div class="diag-row"><span class="diag-label">Entropy</span><span class="diag-val">{last['ent']:.2f}%</span></div>
            <div class="diag-row"><span class="diag-label">Flow</span><span class="diag-val">{last['flow']:.3f}</span></div>
            <div style='margin-top:20px; font-size:0.75rem; color:#666; line-height:1.4'>
            The BRAIN filters market noise using Entropy Gates. Only enter when Flow aligns with Trend.
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: STRUCTURE (SMC) ---
with t2:
    col_chart, col_diag = st.columns([3, 1])
    with col_chart:
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='#00E676', decreasing_line_color='#FF1744'
        )])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['smc_base'], line=dict(color='#00E5FF', width=2), name="SMC Baseline"))
        
        # Visualize ONLY recent FVGs (Performance Hack)
        recent_df = df.tail(50).reset_index(drop=True)
        for i in range(2, len(recent_df)):
            row = recent_df.iloc[i]
            # Bull FVG
            if row['fvg_b']:
                y0 = recent_df.iloc[i-2]['high']
                y1 = row['low']
                t0 = row['timestamp']
                t1 = t0 + pd.Timedelta(minutes=15) if 'm' in st.session_state.tf else t0 + pd.Timedelta(hours=1)
                fig.add_hrect(y0=y0, y1=y1, x0=t0, x1=t1, fillcolor="rgba(0, 230, 118, 0.15)", line_width=0, layer="below")
            # Bear FVG
            if row['fvg_s']:
                y0 = row['high']
                y1 = recent_df.iloc[i-2]['low']
                t0 = row['timestamp']
                t1 = t0 + pd.Timedelta(minutes=15) if 'm' in st.session_state.tf else t0 + pd.Timedelta(hours=1)
                fig.add_hrect(y0=y0, y1=y1, x0=t0, x1=t1, fillcolor="rgba(255, 23, 68, 0.15)", line_width=0, layer="below")

        fig.update_layout(base_layout())
        st.plotly_chart(fig, use_container_width=True)

    with col_diag:
        sig = "BUY" if last['smc_buy'] else ("SELL" if last['smc_sell'] else "NEUTRAL")
        col = "var(--bull)" if last['smc_buy'] else ("var(--bear)" if last['smc_sell'] else "#fff")
        st.markdown(f"""
        <div class="diag-box">
            <div style='color:var(--accent-sec); font-weight:700; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px'>STRUCTURE</div>
            <div class="diag-row"><span class="diag-label">Signal</span><span class="diag-val" style="color:{col}">{sig}</span></div>
            <div class="diag-row"><span class="diag-label">Base</span><span class="diag-val">{last['smc_base']:.2f}</span></div>
            <div class="diag-row"><span class="diag-label">Gap</span><span class="diag-val">{'ACTIVE' if last['fvg_b'] or last['fvg_s'] else 'NONE'}</span></div>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: VECTOR ---
with t3:
    col_chart, col_diag = st.columns([3, 1])
    with col_chart:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.0, row_heights=[0.65, 0.35])
        
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='#333', decreasing_line_color='#333',
            name="Price"
        ), row=1, col=1)
        
        colors = ["#00E676" if x > 0 else "#FF1744" for x in df['flux']]
        fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['flux'], marker_color=colors, name="Flux", opacity=0.9
        ), row=2, col=1)
        
        fig.update_layout(base_layout())
        fig.update_yaxes(range=[-2, 2], row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_diag:
        st.markdown(f"""
        <div class="diag-box">
            <div style='color:var(--accent-sec); font-weight:700; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px'>PHYSICS</div>
            <div class="diag-row"><span class="diag-label">Efficiency</span><span class="diag-val">{last['eff']*100:.1f}%</span></div>
            <div class="diag-row"><span class="diag-label">Flux Force</span><span class="diag-val">{last['flux']:.3f}</span></div>
            <div style='margin-top:20px; font-size:0.75rem; color:#666; line-height:1.4'>
            Flux measures directional efficiency adjusted for volume. Positive = Bullish Energy.
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: QUANTUM ---
with t4:
    col_chart, col_diag = st.columns([3, 1])
    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['rqzo'], 
            mode='lines', line=dict(color='#D500F9', width=1.5), 
            name="Quantum Wave", fill='tozeroy', fillcolor='rgba(213, 0, 249, 0.1)'
        ))
        fig.add_hline(y=10, line_dash="dash", line_color="#444")
        fig.add_hline(y=-10, line_dash="dash", line_color="#444")
        fig.update_layout(base_layout())
        st.plotly_chart(fig, use_container_width=True)

    with col_diag:
        st.markdown(f"""
        <div class="diag-box">
            <div style='color:var(--accent-sec); font-weight:700; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px'>QUANTUM</div>
            <div class="diag-row"><span class="diag-label">Amplitude</span><span class="diag-val">{last['rqzo']:.2f}</span></div>
            <div class="diag-row"><span class="diag-label">Phase</span><span class="diag-val">CALCULATED</span></div>
            <div style='margin-top:20px; font-size:0.75rem; color:#666; line-height:1.4'>
            RQZO visualizes price as a wave function, highlighting cycles and momentum inversions.
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 5: TRADINGVIEW ---
with t5:
    tv_sym = st.session_state.sym.replace("/", "")
    tv_map = {"Binance": "BINANCE", "Kraken": "KRAKEN", "Bybit": "BYBIT"}
    tv_ex = tv_map.get(st.session_state.exch, "BINANCE")
    
    st.components.v1.html(f"""
    <div class="tradingview-widget-container" style="height:600px; width: 100%;">
      <div id="tradingview_widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script>
      new TradingView.widget({{
      "autosize": true,
      "symbol": "{tv_ex}:{tv_sym}",
      "interval": "{st.session_state.tf}",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#f1f3f6",
      "enable_publishing": false,
      "allow_symbol_change": true,
      "container_id": "tradingview_widget",
      "hide_side_toolbar": false
      }});
      </script>
    </div>
    """, height=610)

# --- TAB 6: AI COUNCIL ---
with t6:
    st.markdown("### 🤖 COUNCIL INTELLIGENCE")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔮 GENERATE STRATEGY"):
            if OPENAI_AVAILABLE and st.session_state.ai_k:
                with st.spinner("Synthesizing Neural Networks..."):
                    try:
                        client = OpenAI(api_key=st.session_state.ai_k)
                        prompt = f"""
                        Analyze {st.session_state.sym} Data:
                        Price: {last['close']}
                        Vector Flux: {last['flux']} (Bull > 0)
                        Brain Trend: {last['br_trend']}
                        Quintessence Index: {int(score)}
                        
                        Provide:
                        1. Direction (LONG/SHORT/FLAT)
                        2. Entry Zone
                        3. Stop Loss
                        4. Reasoning (Max 2 sentences)
                        Format as clean bullet points.
                        """
                        response = client.chat.completions.create(
                            model="gpt-4o", messages=[{"role": "user", "content": prompt}]
                        )
                        st.markdown(f"```markdown\n{response.choices[0].message.content}\n```")
                    except Exception as e:
                        st.error(f"Neural Link Error: {e}")
            else:
                st.warning("OpenAI API Key missing in Sidebar.")
    
    with col2:
        st.markdown("""
        <div class="diag-box">
            <div style='color:var(--accent-primary); font-weight:700'>STATUS</div>
            <p style='font-size:0.8rem; color:#888; margin-top:10px'>
            The AI Council aggregates data from the Quintessence Core to generate high-probability trade setups.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Auto-Pilot Loop
if st.session_state.auto:
    time_lib.sleep(2) 
    st.rerun()
```
