
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests
import time as time_lib
import tweepy

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
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    :root {
        --bg: #000000;
        --card: #080808;
        --border: #1a1a1a;
        --accent: #D500F9; 
        --bull: #00E676;
        --bear: #FF1744;
        --heat: #FFD600;
        --text: #e0e0e0;
    }
    
    .stApp { background-color: var(--bg); font-family: 'JetBrains Mono', monospace; color: var(--text); }
    
    /* Orderly Dashboard Cards */
    div[data-testid="metric-container"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    
    /* Diagnostics Panel */
    .diag-panel {
        background: #050505;
        border-left: 4px solid var(--accent);
        padding: 20px;
        border-radius: 0 8px 8px 0;
        height: 100%;
    }
    .diag-header { font-size: 0.75rem; color: #555; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin-bottom: 15px; }
    .diag-item { margin-bottom: 10px; font-size: 0.9rem; border-bottom: 1px solid #111; padding-bottom: 5px; }
    .diag-label { color: #888; margin-right: 10px; }
    .diag-val { color: #fff; font-weight: 700; }
    
    /* Custom Sidebar Inputs */
    .stTextInput input, .stSelectbox div { background-color: #0c0c0c !important; border: 1px solid #333 !important; }
    
    /* Signal Log */
    .log-container {
        font-size: 0.75rem; color: #888; max-height: 200px; overflow-y: auto;
        border: 1px solid #222; padding: 10px; border-radius: 4px; background: #020202;
    }
    .log-entry { border-bottom: 1px solid #111; padding: 5px 0; display: flex; justify-content: space-between; }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .stPlotlyChart { height: 300px !important; }
        div[data-testid="column"] { width: 100% !important; }
        .stMetric { padding: 10px !important; }
        .diag-panel { margin-top: 20px; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. GLOBAL CONNECTIVITY HANDLERS
# ==========================================
def send_telegram(token, chat_id, text):
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                         json={"chat_id": chat_id, "text": text})
        except Exception as e:
            st.sidebar.error(f"Telegram error: {str(e)}")

def post_x(key, secret, at, ats, text):
    if key and at and secret and ats:
        try:
            client = tweepy.Client(
                consumer_key=key,
                consumer_secret=secret,
                access_token=at,
                access_token_secret=ats
            )
            client.create_tweet(text=text)
        except Exception as e:
            st.sidebar.error(f"X/Twitter error: {str(e)}")

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
    df["eff"] = pd.Series(np.where(rng==0, 0, body/rng)).ewm(span=p["vec_l"]).mean()
    vol_fact = np.where(df["volume"].rolling(p["vol_n"]).mean()==0, 1, 
                       df["volume"]/df["volume"].rolling(p["vol_n"]).mean())
    raw_v = np.sign(df["close"] - df["open"]) * df["eff"] * vol_fact
    df["flux"] = raw_v.ewm(span=p["vec_sm"]).mean()
    th_s, th_r = p["vec_super"] * p["vec_strict"], p["vec_resist"] * p["vec_strict"]
    conditions = [(df["flux"] > th_s), (df["flux"] < -th_s), (df["flux"].abs() < th_r)]
    df["vec_state"] = np.select(conditions, [2, -2, 0], default=1)
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
    df["br_u"], df["br_l_band"] = base + (atr * p["br_m"]), base - (atr * p["br_m"])
    trend = np.zeros(len(df))
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["br_u"].iloc[i]: trend[i] = 1
        elif df["close"].iloc[i] < df["br_l_band"].iloc[i]: trend[i] = -1
        else: trend[i] = trend[i-1]
    df["br_trend"] = trend
    df["ent"] = df["close"].pct_change().rolling(64).std() * 100
    df["gate"] = df["ent"] < p["br_th"]
    rng = df["high"] - df["low"]
    wick = np.where(rng==0, 0, ((np.minimum(df["open"], df["close"]) - df["low"]) - 
                               (df["high"] - np.maximum(df["open"], df["close"])))/rng)
    df["flow"] = pd.Series(wick + ((df["close"]-df["open"])/(rma(rng, 14)+1e-10))).ewm(span=34).mean()
    df["br_buy"] = (df["br_trend"]==1) & df["gate"] & (df["flux"] > 0.5) & (df["flow"] > 0)
    df["br_sell"] = (df["br_trend"]==-1) & df["gate"] & (df["flux"] < -0.5) & (df["flow"] < 0)
    return df

# CORE 3: RQZO (Quantum Mechanics) - FIXED VERSION
def calc_rqzo(df, p):
    df = df.copy()
    # Normalize price to 0-1 range over rolling window
    mn = df["close"].rolling(100, min_periods=1).min()
    mx = df["close"].rolling(100, min_periods=1).max()
    norm = (df["close"] - mn) / (mx - mn + 1e-10)
    
    # Calculate gamma (Lorentz factor approximation)
    velocity = (norm - norm.shift(1)).abs().clip(upper=0.049)
    gamma = 1 / np.sqrt(1 - (velocity / 0.05) ** 2)
    
    # Time dilation calculation
    tau = (np.arange(len(df)) % 100) / gamma
    
    # Quantum oscillator sum (vectorized)
    n_vals = np.arange(1, 10)
    # Create broadcasting for vectorized calculation
    tau_expanded = tau.values[:, np.newaxis]  # Shape: (n_samples, 1)
    log_n = np.log(n_vals)[np.newaxis, :]     # Shape: (1, 9)
    
    # Calculate sin(tau * log(n)) for all n, vectorized
    sin_terms = np.sin(tau_expanded * log_n)
    weights = n_vals ** -0.5
    
    # Sum over n values
    rqzo_vals = np.sum(sin_terms * weights, axis=1) * 10
    
    df["rqzo"] = rqzo_vals
    return df

# CORE 4: MATRIX (Momentum Matrix)
def calc_matrix(df, p):
    df = df.copy()
    rs = rma(df["close"].diff().clip(lower=0), 14) / (rma(-df["close"].diff().clip(upper=0), 14) + 1e-10)
    rsi = 100 - (100/(1+rs))
    df["mfi"] = ((rsi - 50) * (df["volume"] / df["volume"].rolling(20).mean())).ewm(span=3).mean()
    df["hw"] = 100 * (double_smooth(df["close"].diff(), 25, 13) / 
                     (double_smooth(df["close"].diff().abs(), 25, 13) + 1e-10)) / 2
    df["mat_sig"] = np.sign(df["mfi"]) + np.sign(df["hw"])
    return df

# CORE 5: APEX SMC (Structural Master)
def calc_smc(df, p):
    df = df.copy()
    df["smc_base"] = hma(df["close"], p["smc_l"])
    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ap.ewm(span=10).mean()
    tci = ((ap - esa) / (0.015 * (ap - esa).abs().ewm(span=10).mean() + 1e-10)).ewm(span=21).mean()
    df["smc_buy"] = (df["close"] > df["smc_base"]) & (tci < 60) & (tci > tci.shift(1))
    df["smc_sell"] = (df["close"] < df["smc_base"]) & (tci > -60) & (tci < tci.shift(1))
    df["fvg_b"] = (df["low"] > df["high"].shift(2))
    df["fvg_s"] = (df["high"] < df["low"].shift(2))
    return df

# ==========================================
# 4. CONFLUENCE & CONSENSUS SYSTEM
# ==========================================
def calc_confluence(df):
    """Calculate Pentagram consensus score"""
    df = df.copy()
    
    # Count bullish/bearish signals from each core
    df["bull_signals"] = (
        (df["vec_state"] == 2).astype(int) +
        (df["br_buy"]).astype(int) +
        (df["smc_buy"]).astype(int) +
        ((df["mat_sig"] > 0) & (df["mat_sig"].abs() > 1)).astype(int) +
        (df["rqzo"] > 0).astype(int)
    )
    
    df["bear_signals"] = (
        (df["vec_state"] == -2).astype(int) +
        (df["br_sell"]).astype(int) +
        (df["smc_sell"]).astype(int) +
        ((df["mat_sig"] < 0) & (df["mat_sig"].abs() > 1)).astype(int) +
        (df["rqzo"] < 0).astype(int)
    )
    
    # Calculate consensus score (-5 to +5)
    df["consensus"] = df["bull_signals"] - df["bear_signals"]
    
    # Strength classification
    conditions = [
        df["consensus"] >= 4,
        df["consensus"] == 3,
        df["consensus"] == 2,
        df["consensus"] == 1,
        df["consensus"] == 0,
        df["consensus"] == -1,
        df["consensus"] == -2,
        df["consensus"] == -3,
        df["consensus"] <= -4
    ]
    
    choices = [
        "STRONG BUY 🔥",
        "BUY 📈", 
        "WEAK BUY ↗️",
        "NEUTRAL BULL ⚡",
        "NEUTRAL ⚖️",
        "NEUTRAL BEAR ⚡",
        "WEAK SELL ↘️",
        "SELL 📉",
        "STRONG SELL 🧯"
    ]
    
    df["consensus_text"] = np.select(conditions, choices, default="NEUTRAL ⚖️")
    
    return df

# ==========================================
# 5. DATA HANDLING WITH ERROR HANDLING
# ==========================================
@st.cache_data(ttl=5)
def get_data(exch, sym, tf, lim):
    try:
        ex_class = getattr(ccxt, exch.lower())
        ex = ex_class({
            "enableRateLimit": True,
            "timeout": 10000,
            "options": {"defaultType": "spot"}
        })
        
        # Clean symbol format
        sym = sym.replace(" ", "").upper()
        
        # Fetch data
        ohlcv = ex.fetch_ohlcv(sym, tf, limit=lim)
        
        if not ohlcv or len(ohlcv) < 10:
            st.warning(f"Insufficient data from {exch}. Trying with smaller limit...")
            ohlcv = ex.fetch_ohlcv(sym, tf, limit=100)
        
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Validate data
        if df.isnull().any().any():
            st.error("Data contains NaN values")
            return pd.DataFrame()
            
        return df
        
    except ccxt.NetworkError as e:
        st.error(f"Network error: {str(e)}. Check connection.")
        return pd.DataFrame()
    except ccxt.ExchangeError as e:
        st.error(f"Exchange error: {str(e)}. Check symbol format.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return pd.DataFrame()

# Initialize State
def init():
    defaults = {
        "exch": "Kraken", "sym": "BTC/USD", "tf": "15m", "lim": 500,
        "vec_l": 14, "vol_n": 55, "vec_sm": 5, "vec_super": 0.6, "vec_resist": 0.3, "vec_strict": 1.0,
        "br_l": 55, "br_m": 1.5, "br_th": 2.0, "smc_l": 55, "auto": False,
        "ai_k": "", "tg_t": "", "tg_c": "", 
        "x_k": "", "x_s": "", "x_at": "", "x_as": "",
        "last_auto_update": 0,
        "signal_logs": []
    }
    for k, v in defaults.items():
        if k not in st.session_state: 
            st.session_state[k] = v

init()

# ==========================================
# 6. SIDEBAR CONTROL DECK
# ==========================================
with st.sidebar:
    st.markdown("### 🔯 CONTROL DECK")
    
    with st.expander("🌍 Feed Configuration", expanded=True):
        st.session_state.exch = st.selectbox("Exchange", ["Kraken", "Binance", "Bybit", "Coinbase", "OKX"])
        st.session_state.sym = st.text_input("Asset Ticker", st.session_state.sym)
        st.session_state.tf = st.selectbox("Interval", ["1m","5m","15m","1h","4h","1d"], index=2)
        st.session_state.auto = st.checkbox("🔄 Auto-Pilot (60s refresh)", st.session_state.auto)
    
    with st.expander("⚙️ Pentagram Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**APEX VECTOR**")
            st.session_state.vec_l = st.slider("Vector Length", 5, 50, st.session_state.vec_l)
            st.session_state.vol_n = st.slider("Volume Period", 20, 100, st.session_state.vol_n)
            st.session_state.vec_super = st.slider("Superconductor", 0.1, 2.0, st.session_state.vec_super, 0.1)
            
        with col2:
            st.markdown("**APEX BRAIN**")
            st.session_state.br_l = st.slider("Brain Length", 20, 100, st.session_state.br_l)
            st.session_state.br_m = st.slider("Brain Multiplier", 0.5, 3.0, st.session_state.br_m, 0.1)
            st.session_state.br_th = st.slider("Entropy Threshold", 0.5, 5.0, st.session_state.br_th, 0.1)
            
        st.markdown("**SMC STRUCTURE**")
        st.session_state.smc_l = st.slider("SMC Length", 20, 100, st.session_state.smc_l)

    with st.expander("📡 Omnichannel APIs"):
        st.session_state.tg_t = st.text_input("Telegram Bot Token", st.session_state.tg_t, type="password")
        st.session_state.tg_c = st.text_input("Telegram Chat ID", st.session_state.tg_c)
        
        st.markdown("**X/Twitter API v2**")
        st.session_state.x_k = st.text_input("X API Key", st.session_state.x_k, type="password")
        st.session_state.x_s = st.text_input("X API Secret", st.session_state.x_s, type="password")
        st.session_state.x_at = st.text_input("X Access Token", st.session_state.x_at, type="password")
        st.session_state.x_as = st.text_input("X Access Token Secret", st.session_state.x_as, type="password")
        
        st.session_state.ai_k = st.text_input("OpenAI Secret", st.session_state.ai_k, type="password")

    if st.button("🔱 RELOAD THE PENTAGRAM", type="primary", use_container_width=True):
        get_data.clear()
        st.rerun()

# ==========================================
# 7. PROCESSING & BROADCASTING
# ==========================================
df = get_data(st.session_state.exch, st.session_state.sym, st.session_state.tf, st.session_state.lim)
if df.empty:
    st.error("VOID DETECTED: Check Exchange or Symbol Format.")
    st.stop()

# Chain computation
df = calc_vector(df, st.session_state)
df = calc_brain(df, st.session_state)
df = calc_rqzo(df, st.session_state)
df = calc_matrix(df, st.session_state)
df = calc_smc(df, st.session_state)
df = calc_confluence(df)  # Add consensus scoring

last, prev = df.iloc[-1], df.iloc[-2]

# Broadcast events
event = None
if last["br_buy"] and not prev["br_buy"]: 
    event = f"🧠 BRAIN LONG: {st.session_state.sym} @ {last['close']:.2f}"
elif last["smc_buy"]: 
    event = f"🏛️ SMC BUY: {st.session_state.sym} @ {last['close']:.2f}"
elif last["vec_state"] == 2 and prev["vec_state"] != 2: 
    event = f"⚡ SUPERCONDUCTOR: {st.session_state.sym}"
elif last["consensus"] >= 3 and prev["consensus"] < 3:
    event = f"🔯 STRONG BUY CONFLUENCE: {st.session_state.sym}"

if event:
    current_time = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{current_time}] {event}"
    st.session_state.signal_logs.insert(0, log_entry)
    if len(st.session_state.signal_logs) > 10:
        st.session_state.signal_logs = st.session_state.signal_logs[:10]
    
    send_telegram(st.session_state.tg_t, st.session_state.tg_c, event)
    post_x(st.session_state.x_k, st.session_state.x_s, 
           st.session_state.x_at, st.session_state.x_as, event)

# ==========================================
# 8. ORDERLY UI RENDER
# ==========================================
st.title(f"🔯 THE PENTAGRAM // {st.session_state.sym}")

# Global HUD
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Live Price", f"{last['close']:.2f}")
col2.metric("Apex Flux", f"{last['flux']:.3f}", delta=("Bull" if last['flux']>0 else "Bear"))
col3.metric("Brain State", ("SAFE" if last['gate'] else "CHAOS"))
col4.metric("Matrix Sig", int(last['mat_sig']))
col5.metric("Pentagram Consensus", last["consensus_text"], delta=f"{last['consensus']}/5")

# Signal Log Panel
st.markdown("### 📜 Signal Log")
log_html = '<div class="log-container">'
for log in st.session_state.signal_logs:
    log_html += f'<div class="log-entry">{log}</div>'
log_html += '</div>'
st.markdown(log_html, unsafe_allow_html=True)

# Plot layout helper
def clean_plot():
    return go.Layout(
        template="plotly_dark", 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        margin=dict(l=0,r=0,t=10,b=0), 
        height=500,
        font=dict(family="JetBrains Mono", color="#e0e0e0")
    )

# --- Dashboard Tabs ---
t1, t2, t3, t4, t5, t6, t7 = st.tabs(["🧠 Brain", "🏛️ Structure", "⚡ Vector", "⚛️ Quantum", "📊 Matrix", "📺 TV View", "🤖 AI Council"])

with t1:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[
            go.Candlestick(
                x=df['timestamp'], 
                open=df['open'], 
                high=df['high'], 
                low=df['low'], 
                close=df['close'],
                name="Price"
            )
        ])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_u'], 
                                line=dict(color='rgba(0,230,118,0.2)'), 
                                name="Upper Band", fill=None))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_l_band'], 
                                line=dict(color='rgba(255,23,68,0.2)'), 
                                name="Lower Band", fill='tonexty'))
        fig.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Brain Diagnostics</div>
        <div class="diag-item"><span class="diag-label">Macro Bias:</span><span class="diag-val">{('BULLISH' if last['br_trend']==1 else 'BEARISH')}</span></div>
        <div class="diag-item"><span class="diag-label">Entropy:</span><span class="diag-val">{last['ent']:.2f}</span></div>
        <div class="diag-item"><span class="diag-label">Flow:</span><span class="diag-val">{last['flow']:.3f}</span></div>
        <div class="diag-item"><span class="diag-label">Buy Signal:</span><span class="diag-val">{last['br_buy']}</span></div>
        <div class="diag-text" style="margin-top:20px; color:#555">Brain Core analyzes neural trend clouds and entropy gates to filter noise.</div></div>""", unsafe_allow_html=True)

with t2:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[
            go.Candlestick(
                x=df['timestamp'], 
                open=df['open'], 
                high=df['high'], 
                low=df['low'], 
                close=df['close']
            )
        ])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['smc_base'], 
                                line=dict(color='cyan', width=2),
                                name="SMC Base"))
        
        # Add FVG markers
        fvg_bull = df[df['fvg_b']]
        fvg_bear = df[df['fvg_s']]
        
        if not fvg_bull.empty:
            fig.add_trace(go.Scatter(x=fvg_bull['timestamp'], y=fvg_bull['high'],
                                    mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                                    name='Bull FVG'))
        
        if not fvg_bear.empty:
            fig.add_trace(go.Scatter(x=fvg_bear['timestamp'], y=fvg_bear['low'],
                                    mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                                    name='Bear FVG'))
        
        fig.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Structure Diagnostics</div>
        <div class="diag-item"><span class="diag-label">Signal:</span><span class="diag-val">{('BUY' if last['smc_buy'] else ('SELL' if last['smc_sell'] else 'WAIT'))}</span></div>
        <div class="diag-item"><span class="diag-label">Bull FVG:</span><span class="diag-val">{last['fvg_b']}</span></div>
        <div class="diag-item"><span class="diag-label">Bear FVG:</span><span class="diag-val">{last['fvg_s']}</span></div>
        <div class="diag-item"><span class="diag-label">Base Price:</span><span class="diag-val">{last['smc_base']:.2f}</span></div>
        <div class="diag-text" style="margin-top:20px; color:#555">SMC Core monitors Fair Value Gaps and WaveTrend momentum pivots.</div></div>""", unsafe_allow_html=True)

with t3:
    l, r = st.columns([3, 1])
    with l:
        fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_v.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
        
        colors = ["#00E676" if x==2 else ("#FF1744" if x==-2 else "#888") for x in df["vec_state"]]
        fig_v.add_trace(go.Bar(x=df['timestamp'], y=df['flux'], marker_color=colors, name="Flux"), row=2, col=1)
        
        # Add threshold lines
        th_s = st.session_state.vec_super * st.session_state.vec_strict
        th_r = st.session_state.vec_resist * st.session_state.vec_strict
        
        fig_v.add_hline(y=th_s, line_dash="dash", line_color="#00E676", opacity=0.5, row=2, col=1)
        fig_v.add_hline(y=-th_s, line_dash="dash", line_color="#FF1744", opacity=0.5, row=2, col=1)
        fig_v.add_hline(y=th_r, line_dash="dot", line_color="#888", opacity=0.3, row=2, col=1)
        fig_v.add_hline(y=-th_r, line_dash="dot", line_color="#888", opacity=0.3, row=2, col=1)
        
        fig_v.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_v, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Physics Diagnostics</div>
        <div class="diag-item"><span class="diag-label">Efficiency:</span><span class="diag-val">{last['eff']*100:.1f}%</span></div>
        <div class="diag-item"><span class="diag-label">Flux:</span><span class="diag-val">{last['flux']:.3f}</span></div>
        <div class="diag-item"><span class="diag-label">State:</span><span class="diag-val">{last['vec_state']}</span></div>
        <div class="diag-item"><span class="diag-label">Volume Factor:</span><span class="diag-val">{last['volume']/df['volume'].rolling(st.session_state.vol_n).mean().iloc[-1]:.2f}x</span></div>
        <div class="diag-text" style="margin-top:20px; color:#555">Vector Core calculates the physical force of the market via geometric efficiency.</div></div>""", unsafe_allow_html=True)

with t4:
    l, r = st.columns([3, 1])
    with l:
        fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_q.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
        fig_q.add_trace(go.Scatter(x=df['timestamp'], y=df['rqzo'], 
                                  line=dict(color='#D500F9', width=2), 
                                  name="RQZO"), row=2, col=1)
        fig_q.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)
        fig_q.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_q, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Quantum Diagnostics</div>
        <div class="diag-item"><span class="diag-label">RQZO Value:</span><span class="diag-val">{last['rqzo']:.2f}</span></div>
        <div class="diag-item"><span class="diag-label">Quantum Bias:</span><span class="diag-val">{'BULLISH' if last['rqzo'] > 0 else 'BEARISH'}</span></div>
        <div class="diag-item"><span class="diag-label">Signal:</span><span class="diag-val">{'BULL' if last['rqzo'] > 0 else 'BEAR'}</span></div>
        <div class="diag-text" style="margin-top:20px; color:#555">Quantum Core models relativistic time dilation effects on price oscillators.</div></div>""", unsafe_allow_html=True)

with t5:
    l, r = st.columns([3, 1])
    with l:
        fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_m.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
        fig_m.add_trace(go.Scatter(x=df['timestamp'], y=df['mfi'], 
                                  line=dict(color='#00E676', width=1.5), 
                                  name="MFI"), row=2, col=1)
        fig_m.add_trace(go.Scatter(x=df['timestamp'], y=df['hw'], 
                                  line=dict(color='#FFD600', width=1.5), 
                                  name="HW"), row=2, col=1)
        fig_m.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)
        fig_m.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_m, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Matrix Diagnostics</div>
        <div class="diag-item"><span class="diag-label">Matrix Signal:</span><span class="diag-val">{int(last['mat_sig'])}</span></div>
        <div class="diag-item"><span class="diag-label">MFI:</span><span class="diag-val">{last['mfi']:.2f}</span></div>
        <div class="diag-item"><span class="diag-label">HW Osc:</span><span class="diag-val">{last['hw']:.2f}</span></div>
        <div class="diag-item"><span class="diag-label">Direction:</span><span class="diag-val">{'BULLISH' if last['mat_sig'] > 0 else 'BEARISH'}</span></div>
        <div class="diag-text" style="margin-top:20px; color:#555">Matrix Core combines volume-weighted RSI with Hull Wave momentum.</div></div>""", unsafe_allow_html=True)

with t6:
    st.markdown("### Official TradingView Confirmation")
    tv_exchanges = {
        "Binance": "BINANCE",
        "Kraken": "KRAKEN", 
        "Bybit": "BYBIT",
        "Coinbase": "COINBASE",
        "OKX": "OKX"
    }
    tv_exch = tv_exchanges.get(st.session_state.exch, st.session_state.exch.upper())
    tv_sym = st.session_state.sym.replace("/", "")
    
    # Handle common symbols
    symbol_map = {
        "BTC/USD": "BTCUSD",
        "ETH/USD": "ETHUSD",
        "SOL/USD": "SOLUSD",
        "XRP/USD": "XRPUSD"
    }
    tv_sym = symbol_map.get(st.session_state.sym, tv_sym)
    
    st.components.v1.html(f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "width": "100%",
          "height": 500,
          "symbol": "{tv_exch}:{tv_sym}",
          "interval": "{st.session_state.tf}",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#0a0a0a",
          "enable_publishing": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "container_id": "tradingview_chart"
        }});
      </script>
    </div>
    """, height=520)

with t7:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        persona = st.selectbox("Council Member", [
            "Grand Strategist", 
            "Quantum Physicist", 
            "Risk Manager",
            "Market Psychologist",
            "The Quant"
        ])
        
        if st.button("🔱 CONSULT THE COUNCIL", type="primary"):
            if OPENAI_AVAILABLE and st.session_state.ai_k:
                try:
                    client = OpenAI(api_key=st.session_state.ai_k)
                    
                    # Create comprehensive analysis prompt
                    prompt = f"""
                    Analyze {st.session_state.sym} at {last['close']}:
                    
                    PENTAGRAM DIAGNOSTICS:
                    - Vector Flux: {last['flux']:.3f} (State: {last['vec_state']})
                    - Brain Trend: {'BULL' if last['br_trend']==1 else 'BEAR'} (Entropy: {last['ent']:.2f})
                    - SMC Signal: {'BUY' if last['smc_buy'] else 'SELL' if last['smc_sell'] else 'NEUTRAL'}
                    - Matrix Signal: {last['mat_sig']:.1f}
                    - Quantum RQZO: {last['rqzo']:.2f}
                    - Consensus: {last.get('consensus_text', 'N/A')} ({last.get('consensus', 0)}/5)
                    
                    RECENT ACTION:
                    - 1h Change: {((df['close'].iloc[-1] / df['close'].iloc[-4]) - 1)*100:.2f}%
                    - Volume Trend: {'📈' if df['volume'].iloc[-1] > df['volume'].iloc[-5] else '📉'}
                    
                    As a {persona}, provide:
                    1. Bias Assessment (Bull/Neutral/Bear)
                    2. Key Risk Factor
                    3. Suggested Action (Entry/Exit/Wait)
                    4. Confidence Level (0-100%)
                    
                    Be concise, technical, and decisive.
                    """
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": f"You are a {persona} in the Pentagram Trading Council. Provide sharp, actionable insights."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=300
                    )
                    
                    st.success("Council Decision Rendered")
                    st.markdown(f"**{persona}:**")
                    st.info(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Council Unavailable: {str(e)}")
            else:
                st.warning("OpenAI API key required for Council access")
    
    with col2:
        # Display live consensus dashboard
        st.markdown("### ⚖️ Live Consensus Dashboard")
        
        # Create gauge for consensus
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=last.get('consensus', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Pentagram Consensus", 'font': {'size': 20}},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-5, 5], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "darkblue"},
                'bgcolor': "black",
                'borderwidth': 2,
                'bordercolor': "#1a1a1a",
                'steps': [
                    {'range': [-5, -3], 'color': "#FF1744"},
                    {'range': [-3, -1], 'color': "#FF7043"},
                    {'range': [-1, 1], 'color': "#616161"},
                    {'range': [1, 3], 'color': "#66BB6A"},
                    {'range': [3, 5], 'color': "#00E676"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': last.get('consensus', 0)
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=300, 
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#e0e0e0", 'family': "JetBrains Mono"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Core signal breakdown
        st.markdown("### 🔢 Core Signal Breakdown")
        signals = {
            "Vector": last['vec_state'],
            "Brain": 1 if last['br_buy'] else (-1 if last['br_sell'] else 0),
            "SMC": 1 if last['smc_buy'] else (-1 if last['smc_sell'] else 0),
            "Matrix": np.sign(last['mat_sig']),
            "Quantum": np.sign(last['rqzo'])
        }
        
        for core, signal in signals.items():
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_a:
                st.write(f"**{core}**")
            with col_b:
                if signal > 0:
                    st.progress(signal, text="🐂")
                elif signal < 0:
                    st.progress(abs(signal), text="🐻")
                else:
                    st.progress(0.5, text="⚖️")
            with col_c:
                st.write(f"{signal:+d}")

# ==========================================
# 9. AUTO-PILOT HANDLING
# ==========================================
if st.session_state.auto:
    current_time = time_lib.time()
    if current_time - st.session_state.get('last_auto_update', 0) > 60:
        st.session_state.last_auto_update = current_time
        st.rerun()

# Display status
if st.session_state.auto:
    st.sidebar.success("🔄 Auto-Pilot: ACTIVE (60s refresh)")
else:
    st.sidebar.info("⏸️ Auto-Pilot: PAUSED")
