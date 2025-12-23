import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests
import time as time_lib

# ==========================================
# 1. SYSTEM CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Titan Omega: Pentagram",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded",
)

# Dark Matter CSS (High Performance / Low Eye Strain)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    :root {
        --bg: #050505;
        --card: #0e0e0e;
        --border: #222;
        --bull: #00E676;
        --bear: #FF1744;
        --heat: #FFD600;
        --chop: #546E7A;
        --text: #e0e0e0;
        --cyan: #00E5FF;
        --vio: #D500F9;
        --smc: #B9F6CA;
    }
    
    .stApp { background-color: var(--bg); font-family: 'JetBrains Mono', monospace; color: var(--text); }
    
    /* KPI Cards */
    div[data-testid="metric-container"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    label[data-testid="stMetricLabel"] { font-size: 0.7rem; color: #666; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; color: #fff; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
    
    /* Log Feed */
    .log-box {
        height: 200px; overflow-y: auto; background: #080808; 
        border: 1px solid var(--border); border-radius: 4px; padding: 10px;
        font-size: 0.75rem; color: #888;
    }
    .log-row { border-bottom: 1px solid #111; padding: 4px 0; display: flex; gap: 8px; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #666; border: none; }
    .stTabs [aria-selected="true"] { color: var(--cyan); border-bottom: 2px solid var(--cyan); }
    
    /* Custom Brain HUD */
    .brain-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr; gap: 10px; margin-bottom: 15px; }
    .b-card { background: #111; border: 1px solid #333; padding: 10px; border-radius: 6px; text-align: center; }
    .b-head { font-size: 0.65rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .b-val { font-size: 1rem; font-weight: bold; margin-top: 4px; }
    
    .st-bull { color: var(--bull); }
    .st-bear { color: var(--bear); }
    .st-neut { color: #666; }
    
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MATH ENGINE (SHARED UTILS)
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

# ==========================================
# 3. INDICATOR MODULES (5 CORES)
# ==========================================

# --- CORE 1: APEX VECTOR (PHYSICS) ---
def calc_apex_vector(df, p):
    df = df.copy()
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    raw_eff = np.where(rng==0, 0, body/rng)
    df["eff"] = pd.Series(raw_eff).ewm(span=p["vec_len"]).mean()
    
    vol_avg = df["volume"].rolling(p["vol_norm"]).mean()
    vol_fact = np.where(vol_avg==0, 1, df["volume"]/vol_avg)
    raw_vec = np.sign(df["close"] - df["open"]) * df["eff"] * vol_fact
    df["flux"] = raw_vec.ewm(span=p["vec_sm"]).mean()
    
    th_s = p["eff_super"] * p["strict"]
    th_r = p["eff_resist"] * p["strict"]
    
    conditions = [(df["flux"] > th_s), (df["flux"] < -th_s), (df["flux"].abs() < th_r)]
    df["vec_state"] = np.select(conditions, [2, -2, 0], default=1) 

    # Divergence
    src = df["flux"]
    lb = p["div_look"]
    df["pl"] = (src.shift(1) < src.shift(2)) & (src.shift(1) < src)
    df["ph"] = (src.shift(1) > src.shift(2)) & (src.shift(1) > src)
    df["div_bull"] = df["pl"] & (df["close"] < df["close"].shift(lb)) & (df["flux"] > df["flux"].shift(lb))
    df["div_bear"] = df["ph"] & (df["close"] > df["close"].shift(lb)) & (df["flux"] < df["flux"].shift(lb))
    return df

# --- CORE 2: APEX BRAIN (LOGIC) ---
def calc_apex_brain(df, p):
    df = df.copy()
    base = hma(df["close"], p["brain_len"])
    atr = rma(df["high"]-df["low"], p["brain_len"])
    df["cortex_u"] = base + (atr * p["brain_mult"])
    df["cortex_l"] = base - (atr * p["brain_mult"])
    
    trend = np.zeros(len(df))
    c = df["close"].values
    u = df["cortex_u"].values
    l = df["cortex_l"].values
    for i in range(1, len(df)):
        if c[i] > u[i]: trend[i] = 1
        elif c[i] < l[i]: trend[i] = -1
        else: trend[i] = trend[i-1]
    df["brain_trend"] = trend
    
    ret = df["close"].pct_change()
    df["ent_proxy"] = ret.rolling(p["ent_len"]).std() * 100
    df["gate_safe"] = df["ent_proxy"] < p["ent_th"]
    df["motor_bull"] = df["flux"] > 0.5
    df["motor_bear"] = df["flux"] < -0.5
    
    rng = df["high"] - df["low"]
    wick = np.where(rng==0, 0, ((np.minimum(df["open"], df["close"]) - df["low"]) - (df["high"] - np.maximum(df["open"], df["close"])))/rng)
    vz = (df["volume"] - df["volume"].rolling(80).mean()) / (df["volume"].rolling(80).std() + 1e-10)
    raw_flow = wick + (vz * 0.5) + ((df["close"]-df["open"])/(rma(rng, 14)+1e-10))
    df["flow"] = raw_flow.ewm(span=34).mean()
    
    df["brain_buy"] = (df["brain_trend"]==1) & df["gate_safe"] & df["motor_bull"] & (df["flow"] > 0)
    df["brain_sell"] = (df["brain_trend"]==-1) & df["gate_safe"] & df["motor_bear"] & (df["flow"] < 0)
    return df

# --- CORE 3: RQZO (QUANTUM) ---
def calc_rqzo(df, p):
    df = df.copy()
    mn = df["close"].rolling(100).min()
    mx = df["close"].rolling(100).max()
    norm = (df["close"] - mn) / (mx - mn + 1e-10)
    vel = (norm - norm.shift(1)).abs()
    c = 5.0 / 100.0
    gamma = 1 / np.sqrt(1 - (np.clip(vel, 0, c*0.99)/c)**2)
    idx = np.arange(len(df))
    tau = (idx % 100) / gamma
    zeta = np.zeros(len(df))
    for n in range(1, 10):
        zeta += (n ** -0.5) * np.sin(tau * np.log(n))
    gate = np.exp(-2 * (df["ent_proxy"]/10 - 0.6).abs())
    df["rqzo"] = zeta * gate * 10
    return df

# --- CORE 4: MATRIX (MOMENTUM) ---
def calc_matrix(df, p):
    df = df.copy()
    chg = df["close"].diff()
    rs = rma(chg.clip(lower=0), 14) / (rma(-chg.clip(upper=0), 14) + 1e-10)
    rsi = 100 - (100/(1+rs))
    mfi_vol = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
    df["mfi"] = ((rsi - 50) * mfi_vol).ewm(span=3).mean()
    hw_src = df["close"].diff()
    df["hw"] = 100 * (double_smooth(hw_src, 25, 13) / (double_smooth(hw_src.abs(), 25, 13) + 1e-10)) / 2
    df["matrix_sig"] = np.sign(df["mfi"]) + np.sign(df["hw"])
    return df

# --- CORE 5: APEX SMC (TREND & LIQUIDITY) ---
def calc_apex_smc(df, p):
    """Smart Money Concepts, Order Blocks, FVG, WaveTrend"""
    df = df.copy()
    
    # 1. HMA Trend (Redundant with Cortex but kept for signal logic)
    df["smc_base"] = hma(df["close"], p["smc_len"])
    atr = rma(df["high"]-df["low"], 14)
    df["smc_u"] = df["smc_base"] + (atr * p["smc_mult"])
    df["smc_l"] = df["smc_base"] - (atr * p["smc_mult"])
    
    smc_trend = np.zeros(len(df))
    c = df["close"].values
    u = df["smc_u"].values
    l = df["smc_l"].values
    for i in range(1, len(df)):
        if c[i] > u[i]: smc_trend[i] = 1
        elif c[i] < l[i]: smc_trend[i] = -1
        else: smc_trend[i] = smc_trend[i-1]
    df["smc_trend"] = smc_trend

    # 2. WaveTrend Signals
    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ap.ewm(span=10).mean()
    d = (ap - esa).abs().ewm(span=10).mean()
    ci = (ap - esa) / (0.015 * d + 1e-10)
    tci = ci.ewm(span=21).mean()
    
    # Momentum check
    df["mom_buy"] = (tci < 60) & (tci > tci.shift(1))
    df["mom_sell"] = (tci > -60) & (tci < tci.shift(1))
    
    # ADX Check
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr = rma(df["high"]-df["low"], 14)
    pdi = 100 * rma(pd.Series(plus_dm), 14) / (tr+1e-10)
    mdi = 100 * rma(pd.Series(minus_dm), 14) / (tr+1e-10)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    df["adx"] = rma(dx, 14)
    adx_ok = df["adx"] > 20
    
    # Volume Check
    vol_ok = df["volume"] > df["volume"].rolling(20).mean()
    
    # Combined Signals
    df["smc_buy"] = (df["smc_trend"]==1) & (df["smc_trend"].shift(1)!=1) & vol_ok & df["mom_buy"] & adx_ok
    df["smc_sell"] = (df["smc_trend"]==-1) & (df["smc_trend"].shift(1)!=-1) & vol_ok & df["mom_sell"] & adx_ok
    
    # 3. Fair Value Gaps (FVG)
    # Bull FVG: Low[0] > High[2]
    # Bear FVG: High[0] < Low[2]
    df["fvg_bull"] = (df["low"] > df["high"].shift(2))
    df["fvg_bear"] = (df["high"] < df["low"].shift(2))
    
    # Order Block Approximation (Last candle of opposing color before impulse)
    # We will just mark recent swing pivots as POI zones
    df["ph"] = df["high"].rolling(5, center=True).max() == df["high"]
    df["pl"] = df["low"].rolling(5, center=True).min() == df["low"]
    
    return df

# ==========================================
# 4. DATA FEED
# ==========================================
def get_driver(name):
    opts = {"enableRateLimit": True}
    if name == 'Binance': return ccxt.binance(opts)
    if name == 'Bybit': return ccxt.bybit(opts)
    if name == 'Coinbase': return ccxt.coinbase(opts)
    if name == 'OKX': return ccxt.okx(opts)
    return ccxt.kraken(opts)

@st.cache_data(ttl=5)
def fetch_data(exch, sym, tf, lim):
    try:
        ex = get_driver(exch)
        ohlcv = ex.fetch_ohlcv(sym, tf, limit=lim)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except: return pd.DataFrame()

def init_session():
    # Secrets
    try: tg_t = st.secrets["TG_TOKEN"]
    except: tg_t = ""
    try: tg_c = st.secrets["TG_CHAT_ID"]
    except: tg_c = ""
    
    defaults = {
        "exch": "Kraken", "sym": "BTC/USD", "tf": "15m", "lim": 500,
        "vec_len": 14, "vol_norm": 55, "vec_sm": 5, "eff_super": 0.6, "eff_resist": 0.3, "strict": 1.0,
        "div_look": 5, "show_reg": True, "show_hid": False,
        "brain_len": 55, "brain_mult": 1.5, "ent_len": 64, "ent_th": 2.0, "fli_len": 34,
        "smc_len": 55, "smc_mult": 1.5, # New SMC params
        "tg_t": tg_t, "tg_c": tg_c, "auto": False
    }
    if "cfg" not in st.session_state: st.session_state.cfg = defaults
    else:
        for k,v in defaults.items(): 
            if k not in st.session_state.cfg: st.session_state.cfg[k] = v

init_session()
cfg = st.session_state.cfg

# ==========================================
# 5. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### 🌌 TITAN OMEGA")
    
    with st.expander("📡 Data & Auto-Pilot", expanded=True):
        cfg["exch"] = st.selectbox("Exchange", ["Kraken", "Binance", "Bybit", "Coinbase", "OKX"])
        cfg["sym"] = st.text_input("Ticker", cfg["sym"])
        c1, c2 = st.columns(2)
        cfg["tf"] = c1.selectbox("Time", ["1m","5m","15m","1h","4h","1d"], index=2)
        cfg["lim"] = c2.number_input("Lookback", 100, 1000, cfg["lim"])
        if st.checkbox("Auto-Refresh (60s)", cfg["auto"]):
            time_lib.sleep(60)
            st.rerun()

    with st.expander("⚡ Apex Vector"):
        cfg["eff_super"] = st.slider("Super Thresh", 0.1, 1.0, 0.6)
        cfg["eff_resist"] = st.slider("Resist Thresh", 0.0, 0.5, 0.3)
        cfg["div_look"] = st.number_input("Div Lookback", 1, 20, 5)

    with st.expander("🧠 Brain (Quad-Core)"):
        cfg["brain_len"] = st.number_input("Cortex Len", 10, 100, 55)
        cfg["ent_th"] = st.slider("Entropy Gate", 0.5, 5.0, 2.0)

    with st.expander("🏛️ SMC & Trend"):
        cfg["smc_len"] = st.number_input("Trend Length", 10, 200, 55)
        cfg["smc_mult"] = st.slider("ATR Mult", 0.5, 5.0, 1.5)

    with st.expander("📢 Alerts"):
        cfg["tg_t"] = st.text_input("Bot Token", cfg["tg_t"], type="password")
        cfg["tg_c"] = st.text_input("Chat ID", cfg["tg_c"], type="password")

    if st.button("RELOAD SYSTEM", type="primary", use_container_width=True):
        fetch_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📝 Event Log")
    log_cont = st.container()

# ==========================================
# 6. PROCESSING CORE
# ==========================================
df = fetch_data(cfg["exch"], cfg["sym"], cfg["tf"], cfg["lim"])
if df.empty:
    st.error("Data Connection Failed.")
    st.stop()

# CHAIN PROCESSING
df = calc_apex_vector(df, cfg)  # Core 1
df = calc_apex_brain(df, cfg)   # Core 2
df = calc_rqzo(df, cfg)         # Core 3
df = calc_matrix(df, cfg)       # Core 4
df = calc_apex_smc(df, cfg)     # Core 5

last = df.iloc[-1]
prev = df.iloc[-2]

# --- Signal Dispatch ---
alerts = []
# SMC
if last["smc_buy"]: alerts.append("🏛️ SMC: STRONG BUY SIGNAL")
if last["smc_sell"]: alerts.append("🏛️ SMC: STRONG SELL SIGNAL")
if last["fvg_bull"] and not prev["fvg_bull"]: alerts.append("🏛️ SMC: BULL FVG CREATED")
# Brain
if last["brain_buy"] and not prev["brain_buy"]: alerts.append("🧠 BRAIN: SYNAPTIC BUY")
if last["brain_sell"] and not prev["brain_sell"]: alerts.append("🧠 BRAIN: SYNAPTIC SELL")
# Vector
if last["vec_state"] == 2 and prev["vec_state"] != 2: alerts.append("⚡ VECTOR: SUPER BULL")
if last["vec_state"] == -2 and prev["vec_state"] != -2: alerts.append("⚡ VECTOR: SUPER BEAR")

# Telegram
if alerts and cfg["tg_t"]:
    for a in alerts:
        try:
            msg = f"🌌 TITAN OMEGA: {cfg['sym']} [{cfg['tf']}]\n{a}\nPrice: {last['close']}"
            requests.post(f"https://api.telegram.org/bot{cfg['tg_t']}/sendMessage", json={"chat_id": cfg['tg_c'], "text": msg})
        except: pass

# UI Log
with log_cont:
    if alerts:
        for a in alerts: st.markdown(f"<div class='log-row'><span style='color:#00E676'>NEW</span> {a}</div>", unsafe_allow_html=True)
    else: st.caption("Scanning 5-Core Physics...")

# ==========================================
# 7. VISUALIZATION
# ==========================================
st.title(f"🌌 {cfg['sym']} // {cfg['tf']}")

# --- 5-CORE HUD ---
c_trend = "BULL" if last["brain_trend"] == 1 else "BEAR"
s_trend = "st-bull" if last["brain_trend"] == 1 else "st-bear"

c_gate = "SAFE" if last["gate_safe"] else "LOCKED"
s_gate = "st-bull" if last["gate_safe"] else "st-bear"

c_vec = "SUPER" if abs(last["vec_state"])==2 else ("CHOP" if last["vec_state"]==0 else "HEAT")
s_vec = "st-bull" if last["vec_state"]==2 else ("st-bear" if last["vec_state"]==-2 else "st-neut")

c_flow = "INFLOW" if last["flow"] > 0 else "OUTFLOW"
s_flow = "st-bull" if last["flow"] > 0 else "st-bear"

c_smc = "BUY" if last["smc_buy"] else ("SELL" if last["smc_sell"] else "WAIT")
s_smc = "st-bull" if last["smc_buy"] else ("st-bear" if last["smc_sell"] else "st-neut")

st.markdown(f"""
<div class="brain-grid">
    <div class="b-card"><div class="b-head">CORTEX (Trend)</div><div class="b-val {s_trend}">{c_trend}</div></div>
    <div class="b-card"><div class="b-head">AMYGDALA (Gate)</div><div class="b-val {s_gate}">{c_gate}</div></div>
    <div class="b-card"><div class="b-head">MOTOR (Vector)</div><div class="b-val {s_vec}">{c_vec}</div></div>
    <div class="b-card"><div class="b-head">OCCIPITAL (Flow)</div><div class="b-val {s_flow}">{c_flow}</div></div>
    <div class="b-card"><div class="b-head">SMC (Signal)</div><div class="b-val {s_smc}">{c_smc}</div></div>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
t1, t2, t3, t4, t5 = st.tabs(["🧠 Brain & Cortex", "🏛️ SMC & Liquidity", "⚡ Apex Vector", "💠 Matrix & RQZO", "📘 Manual"])

def dark_plot():
    return go.Layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=50,t=10,b=0), height=550,
        xaxis=dict(showgrid=False, color="#444"), yaxis=dict(showgrid=True, gridcolor="#222")
    )

with t1:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    
    # Cortex Cloud
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["cortex_u"], line=dict(width=0), showlegend=False))
    cloud_col = "rgba(0, 230, 118, 0.1)" if last["brain_trend"]==1 else "rgba(255, 23, 68, 0.1)"
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["cortex_l"], fill="tonexty", fillcolor=cloud_col, line=dict(width=0), name="Cortex"))
    
    # Brain Signals
    b = df[df["brain_buy"]]
    s = df[df["brain_sell"]]
    fig.add_trace(go.Scatter(x=b["timestamp"], y=b["low"], mode="markers", marker=dict(symbol="triangle-up", color="#00E676", size=12), name="Brain Buy"))
    fig.add_trace(go.Scatter(x=s["timestamp"], y=s["high"], mode="markers", marker=dict(symbol="triangle-down", color="#FF1744", size=12), name="Brain Sell"))
    
    fig.update_layout(dark_plot())
    st.plotly_chart(fig, use_container_width=True)

with t2: # SMC & Liquidity
    fig_s = go.Figure()
    fig_s.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    
    # FVGs
    fvg_b = df[df["fvg_bull"]]
    fvg_s = df[df["fvg_bear"]]
    
    # Plot recent FVGs as markers (full rects require shapes, using markers for speed)
    fig_s.add_trace(go.Scatter(x=fvg_b["timestamp"], y=fvg_b["low"], mode="markers", marker=dict(symbol="square", color="rgba(0,230,118,0.4)", size=8), name="Bull FVG"))
    fig_s.add_trace(go.Scatter(x=fvg_s["timestamp"], y=fvg_s["high"], mode="markers", marker=dict(symbol="square", color="rgba(255,23,68,0.4)", size=8), name="Bear FVG"))
    
    # SMC Signals
    sb = df[df["smc_buy"]]
    ss = df[df["smc_sell"]]
    fig_s.add_trace(go.Scatter(x=sb["timestamp"], y=sb["low"]*0.999, mode="markers+text", text="BUY", textposition="bottom center", marker=dict(symbol="triangle-up", color="#00E676", size=14), name="SMC Buy"))
    fig_s.add_trace(go.Scatter(x=ss["timestamp"], y=ss["high"]*1.001, mode="markers+text", text="SELL", textposition="top center", marker=dict(symbol="triangle-down", color="#FF1744", size=14), name="SMC Sell"))
    
    # Trend Line
    fig_s.add_trace(go.Scatter(x=df["timestamp"], y=df["smc_base"], line=dict(color="cyan", width=1), name="SMC Base"))
    
    fig_s.update_layout(dark_plot())
    st.plotly_chart(fig_s, use_container_width=True)

with t3: # Vector
    fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.03)
    fig_v.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]), row=1, col=1)
    
    # Flux
    cols = ["#00E676" if x==2 else ("#FF1744" if x==-2 else ("#546E7A" if x==0 else "#FFD600")) for x in df["vec_state"]]
    fig_v.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=cols, name="Flux"), row=2, col=1)
    
    # Divs
    bd = df[df["div_bull"]]
    sd = df[df["div_bear"]]
    fig_v.add_trace(go.Scatter(x=bd["timestamp"], y=bd["flux"], mode="markers", marker=dict(color="cyan", size=6), name="Bull Div"), row=2, col=1)
    fig_v.add_trace(go.Scatter(x=sd["timestamp"], y=sd["flux"], mode="markers", marker=dict(color="magenta", size=6), name="Bear Div"), row=2, col=1)
    
    fig_v.update_layout(dark_plot())
    st.plotly_chart(fig_v, use_container_width=True)

with t4: # Matrix/RQZO
    fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True)
    # RQZO
    fig_m.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], line=dict(color="white"), name="RQZO"), row=1, col=1)
    # Matrix
    fig_m.add_trace(go.Scatter(x=df["timestamp"], y=df["mfi"], fill="tozeroy", line_color="#D500F9", name="MFI"), row=2, col=1)
    fig_m.add_trace(go.Bar(x=df["timestamp"], y=df["hw"], marker_color="#00E5FF", name="HyperWave"), row=2, col=1)
    fig_m.update_layout(dark_plot())
    st.plotly_chart(fig_m, use_container_width=True)

with t5:
    st.markdown("### 📘 Titan Omega Manual")
    with st.expander("🧠 The Brain (Quad-Core)", expanded=True):
        st.write("""
        **1. CORTEX:** Trend bias.
        **2. AMYGDALA:** Entropy/Chaos Gate.
        **3. MOTOR:** Vector Momentum.
        **4. OCCIPITAL:** Liquidity Flow.
        """)
    with st.expander("🏛️ SMC (Smart Money)"):
        st.write("""
        **SMC Trend:** HMA + ATR Volatility Bands.
        **FVG:** Fair Value Gaps (Inefficiency).
        **Signals:** WaveTrend Momentum + Volume + ADX Confirmation.
        """)
