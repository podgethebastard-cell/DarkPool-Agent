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
# --- OpenAI Integration Fix ---
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
</style>
""", unsafe_allow_html=True)
# ==========================================
# 2. GLOBAL CONNECTIVITY HANDLERS
# ==========================================
def send_telegram(token, chat_id, text):
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text})
        except: pass
def post_x(key, secret, at, ats, text):
    if key and secret and at and ats:
        try:
            client = tweepy.Client(consumer_key=key, consumer_secret=secret, access_token=at, access_token_secret=ats)
            client.create_tweet(text=text)
        except: pass
# ==========================================
# 3. PENTAGRAM MATH ENGINE (ALL 5 CORES)
# ==========================================
def rma(series, length): return series.ewm(alpha=1/length, adjust=False).mean()
def wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
def hma(series, length): return wma(2 * wma(series, length // 2) - wma(series, length), int(np.sqrt(length)))
def double_smooth(src, l1, l2): return src.ewm(span=l1).mean().ewm(span=l2).mean()
# CORE 1: APEX VECTOR (Physics Engine)
def calc_vector(df, p):
    df = df.copy()
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    df["eff"] = pd.Series(np.where(rng==0, 0, body/rng)).ewm(span=p["vec_l"]).mean()
    vol_fact = np.where(df["volume"].rolling(p["vol_n"]).mean()==0, 1, df["volume"]/df["volume"].rolling(p["vol_n"]).mean())
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
    wick = np.where(rng==0, 0, ((np.minimum(df["open"], df["close"]) - df["low"]) - (df["high"] - np.maximum(df["open"], df["close"])))/rng)
    df["flow"] = pd.Series(wick + ((df["close"]-df["open"])/(rma(rng, 14)+1e-10))).ewm(span=34).mean()
    df["br_buy"] = (df["br_trend"]==1) & df["gate"] & (df["flux"] > 0.5) & (df["flow"] > 0)
    df["br_sell"] = (df["br_trend"]==-1) & df["gate"] & (df["flux"] < -0.5) & (df["flow"] < 0)
    return df
# CORE 3: RQZO (Quantum Mechanics)
def calc_rqzo(df, p):
    df = df.copy()
    mn, mx = df["close"].rolling(100).min(), df["close"].rolling(100).max()
    norm = (df["close"] - mn) / (mx - mn + 1e-10)
    gamma = 1 / np.sqrt(1 - (np.clip((norm - norm.shift(1)).abs(), 0, 0.049)/0.05)**2)
    tau = (np.arange(len(df)) % 100) / gamma
    # Fixed: Vectorized computation
    rqzo_vals = np.zeros(len(df))
    for n in range(1, 10):
        rqzo_vals += (n ** -0.5) * np.sin(tau * np.log(n))
    df["rqzo"] = rqzo_vals * 10
    return df
# CORE 4: MATRIX (Momentum Matrix)
def calc_matrix(df, p):
    df = df.copy()
    rs = rma(df["close"].diff().clip(lower=0), 14) / (rma(-df["close"].diff().clip(upper=0), 14) + 1e-10)
    rsi = 100 - (100/(1+rs))
    df["mfi"] = ((rsi - 50) * (df["volume"] / df["volume"].rolling(20).mean())).ewm(span=3).mean()
    df["hw"] = 100 * (double_smooth(df["close"].diff(), 25, 13) / (double_smooth(df["close"].diff().abs(), 25, 13) + 1e-10)) / 2
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
# 4. DATA HANDLING
# ==========================================
@st.cache_data(ttl=5)
def get_data(exch, sym, tf, lim):
    try:
        ex = getattr(ccxt, exch.lower())({"enableRateLimit": True})
        ohlcv = ex.fetch_ohlcv(sym, tf, limit=lim)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()
# Initialize State
def init():
    defaults = {
        "exch": "Kraken", "sym": "BTC/USD", "tf": "15m", "lim": 500,
        "vec_l": 14, "vol_n": 55, "vec_sm": 5, "vec_super": 0.6, "vec_resist": 0.3, "vec_strict": 1.0,
        "br_l": 55, "br_m": 1.5, "br_th": 2.0,
        "smc_l": 55,
        "auto": False,
        "ai_k": "", "tg_t": "", "tg_c": "", "x_k": "", "x_s": "", "x_at": "", "x_as": "",
        "signals": []  # For logging
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
init()
# ==========================================
# 5. SIDEBAR CONTROL DECK
# ==========================================
with st.sidebar:
    st.markdown("### 🔯 CONTROL DECK")
   
    with st.expander("🌍 Feed Configuration", expanded=True):
        st.session_state.exch = st.selectbox("Exchange", ["Kraken", "Binance", "Bybit", "Coinbase", "OKX"])
        st.session_state.sym = st.text_input("Asset Ticker", st.session_state.sym)
        st.session_state.tf = st.selectbox("Interval", ["1m","5m","15m","1h","4h","1d"], index=2)
        st.session_state.auto = st.checkbox("🔄 Auto-Pilot (60s)", st.session_state.auto)
    with st.expander("🧠 Brain Parameters"):
        st.session_state.br_l = st.number_input("Brain Length", min_value=1, value=st.session_state.br_l)
        st.session_state.br_m = st.number_input("Brain Multiplier", min_value=0.1, value=st.session_state.br_m, step=0.1)
        st.session_state.br_th = st.number_input("Brain Threshold", min_value=0.1, value=st.session_state.br_th, step=0.1)
    with st.expander("⚡ Vector Parameters"):
        st.session_state.vec_l = st.number_input("Vector Length", min_value=1, value=st.session_state.vec_l)
        st.session_state.vol_n = st.number_input("Volume Period", min_value=1, value=st.session_state.vol_n)
        st.session_state.vec_sm = st.number_input("Vector Smooth", min_value=1, value=st.session_state.vec_sm)
        st.session_state.vec_super = st.number_input("Super Threshold", min_value=0.1, value=st.session_state.vec_super, step=0.1)
        st.session_state.vec_resist = st.number_input("Resist Threshold", min_value=0.1, value=st.session_state.vec_resist, step=0.1)
        st.session_state.vec_strict = st.number_input("Strict Factor", min_value=0.1, value=st.session_state.vec_strict, step=0.1)
    with st.expander("🏛️ SMC Parameters"):
        st.session_state.smc_l = st.number_input("SMC Length", min_value=1, value=st.session_state.smc_l)
    with st.expander("📡 Omnichannel APIs"):
        st.session_state.tg_t = st.text_input("Telegram Bot Token", st.session_state.tg_t, type="password")
        st.session_state.tg_c = st.text_input("Telegram Chat ID", st.session_state.tg_c)
        st.session_state.x_k = st.text_input("X Consumer Key", st.session_state.x_k, type="password")
        st.session_state.x_s = st.text_input("X Consumer Secret", st.session_state.x_s, type="password")
        st.session_state.x_at = st.text_input("X Access Token", st.session_state.x_at, type="password")
        st.session_state.x_as = st.text_input("X Access Token Secret", st.session_state.x_as, type="password")
        st.session_state.ai_k = st.text_input("OpenAI Secret", st.session_state.ai_k, type="password")
    if st.button("🔱 RELOAD THE PENTAGRAM", type="primary", use_container_width=True):
        get_data.clear()
        st.rerun()
# ==========================================
# 6. PROCESSING & BROADCASTING
# ==========================================
df = get_data(st.session_state.exch, st.session_state.sym, st.session_state.tf, st.session_state.lim)
if df.empty:
    st.error("VOID DETECTED: Check Exchange or Symbol Format.")
    st.stop()
# Chain computation
p = st.session_state  # Pass params directly
df = calc_vector(df, p)
df = calc_brain(df, p)
df = calc_rqzo(df, p)
df = calc_matrix(df, p)
df = calc_smc(df, p)
last, prev = df.iloc[-1], df.iloc[-2]
# Consensus Score
bull_cores = sum([last['br_buy'], last['smc_buy'], (last['vec_state'] == 2), (last['mat_sig'] > 0), (last['rqzo'] > 0)])  # Example confluence
bear_cores = sum([last['br_sell'], last['smc_sell'], (last['vec_state'] == -2), (last['mat_sig'] < 0), (last['rqzo'] < 0)])
consensus = "STRONG BUY" if bull_cores >= 3 else ("STRONG SELL" if bear_cores >= 3 else "NEUTRAL")
# Broadcast events
event = None
if last["br_buy"] and not prev["br_buy"]: 
    event = f"🧠 BRAIN LONG: {st.session_state.sym} @ {last['close']}"
elif last["br_sell"] and not prev["br_sell"]: 
    event = f"🧠 BRAIN SHORT: {st.session_state.sym} @ {last['close']}"
elif last["smc_buy"] and not prev["smc_buy"]: 
    event = f"🏛️ SMC BUY: {st.session_state.sym} @ {last['close']}"
elif last["smc_sell"] and not prev["smc_sell"]: 
    event = f"🏛️ SMC SELL: {st.session_state.sym} @ {last['close']}"
elif last["vec_state"] == 2 and prev["vec_state"] != 2: 
    event = f"⚡ SUPERCONDUCTOR: {st.session_state.sym}"
elif last["vec_state"] == -2 and prev["vec_state"] != -2: 
    event = f"⚡ SUPERRESISTOR: {st.session_state.sym}"
if event:
    st.session_state.signals.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event))
    send_telegram(st.session_state.tg_t, st.session_state.tg_c, event)
    post_x(st.session_state.x_k, st.session_state.x_s, st.session_state.x_at, st.session_state.x_as, event)
# Auto-refresh logic (non-blocking)
if st.session_state.auto:
    time_lib.sleep(60)
    st.rerun()
# ==========================================
# 7. ORDERLY UI RENDER
# ==========================================
st.title(f"🔯 THE PENTAGRAM // {st.session_state.sym}")
# Global HUD
h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("Live Price", f"{last['close']:.2f}")
h2.metric("Apex Flux", f"{last['flux']:.3f}", delta=("Bull" if last['flux']>0 else "Bear"))
h3.metric("Brain State", ("SAFE" if last['gate'] else "CHAOS"))
h4.metric("Matrix Sig", int(last['mat_sig']))
h5.metric("Consensus", consensus)
# Signal Log
with st.expander("📜 Signal Log"):
    for ts, msg in reversed(st.session_state.signals[-10:]):
        st.markdown(f"**{ts}**: {msg}")
# --- Dashboard Tabs ---
t1, t2, t3, t4, t5, t6, t7 = st.tabs(["🧠 Brain", "🏛️ Structure", "⚡ Vector", "⚛️ Quantum", "🧮 Matrix", "📺 TV View", "🤖 AI Council"])
def clean_plot():
    return go.Layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=500)
with t1:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_u'], line=dict(color='rgba(0,230,118,0.2)'), fill=None))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_l_band'], line=dict(color='rgba(255,23,68,0.2)'), fill='tonexty'))
        fig.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Brain Diagnostics</div>
        <div class="diag-item"><span class="diag-label">Macro Bias:</span><span class="diag-val">{('BULLISH' if last['br_trend']==1 else 'BEARISH')}</span></div>
        <div class="diag-item"><span class="diag-label">Entropy:</span><span class="diag-val">{last['ent']:.2f}</span></div>
        <div class="diag-item"><span class="diag-label">Flow:</span><span class="diag-val">{last['flow']:.3f}</span></div>
        <div class="diag-text" style="margin-top:20px; color:#555">Brain Core analyzes neural trend clouds and entropy gates to filter noise.</div></div>""", unsafe_allow_html=True)
with t2:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['smc_base'], line=dict(color='cyan', width=1)))
        fig.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Structure Diagnostics</div>
        <div class="diag-item"><span class="diag-label">Signal:</span><span class="diag-val">{('BUY' if last['smc_buy'] else ('SELL' if last['smc_sell'] else 'WAIT'))}</span></div>
        <div class="diag-item"><span class="diag-label">Bull FVG:</span><span class="diag-val">{last['fvg_b']}</span></div>
        <div class="diag-item"><span class="diag-label">Bear FVG:</span><span class="diag-val">{last['fvg_s']}</span></div>
        <div class="diag-text" style="margin-top:20px; color:#555">SMC Core monitors Fair Value Gaps and WaveTrend momentum pivots.</div></div>""", unsafe_allow_html=True)
with t3:
    l, r = st.columns([3, 1])
    with l:
        fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
        fig_v.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
        colors = ["#00E676" if x==2 else ("#FF1744" if x==-2 else "#333") for x in df["vec_state"]]
        fig_v.add_trace(go.Bar(x=df['timestamp'], y=df['flux'], marker_color=colors), row=2, col=1)
        fig_v.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_v, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Physics Diagnostics</div>
        <div class="diag-item"><span class="diag-label">Efficiency:</span><span class="diag-val">{last['eff']*100:.1f}%</span></div>
        <div class="diag-item"><span class="diag-label">Flux:</span><span class="diag-val">{last['flux']:.3f}</span></div>
        <div class="diag-text" style="margin-top:20px; color:#555">Vector Core calculates the physical force of the market via geometric efficiency.</div></div>""", unsafe_allow_html=True)
with t4:
    l, r = st.columns([3, 1])
    with l:
        fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
        fig_q.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
        fig_q.add_trace(go.Scatter(x=df['timestamp'], y=df['rqzo'], mode='lines', name='RQZO'), row=2, col=1)
        fig_q.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_q, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Quantum Diagnostics</div>
        <div class="diag-item"><span class="diag-label">RQZO:</span><span class="diag-val">{last['rqzo']:.3f}</span></div>
        <div class="diag-text" style="margin-top:20px; color:#555">RQZO Core applies quantum-inspired normalization and wave functions.</div></div>""", unsafe_allow_html=True)
with t5:
    l, r = st.columns([3, 1])
    with l:
        fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
        fig_m.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
        fig_m.add_trace(go.Scatter(x=df['timestamp'], y=df['mfi'], mode='lines', name='MFI'), row=2, col=1)
        fig_m.add_trace(go.Scatter(x=df['timestamp'], y=df['hw'], mode='lines', name='HW'), row=2, col=1)
        fig_m.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_m, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Matrix Diagnostics</div>
        <div class="diag-item"><span class="diag-label">MFI:</span><span class="diag-val">{last['mfi']:.3f}</span></div>
        <div class="diag-item"><span class="diag-label">HW:</span><span class="diag-val">{last['hw']:.3f}</span></div>
        <div class="diag-item"><span class="diag-label">Signal:</span><span class="diag-val">{last['mat_sig']}</span></div>
        <div class="diag-text" style="margin-top:20px; color:#555">Matrix Core combines momentum indicators for signal strength.</div></div>""", unsafe_allow_html=True)
with t6:
    st.markdown("### Official TradingView Confirmation")
    tv_s = st.session_state.sym.replace("/", "").upper()
    exch_map = {"Kraken": "KRAKEN", "Binance": "BINANCE", "Bybit": "BYBIT", "Coinbase": "COINBASE", "OKX": "OKX"}
    tv_exch = exch_map.get(st.session_state.exch, "BINANCE")
    st.components.v1.html(f"""<script src="https://s3.tradingview.com/tv.js"></script>
    <script>new TradingView.widget({{"width": "100%", "height": 500, "symbol": "{tv_exch}:{tv_s}", "theme": "dark", "style": "1", "container_id": "tv"}});</script>
    <div id="tv"></div>""", height=520)
with t7:
    persona = st.selectbox("Council Member", ["Grand Strategist", "Physicist", "Banker", "The Quant"])
    if st.button("🔱 CONSULT THE COUNCIL"):
        if OPENAI_AVAILABLE and st.session_state.ai_k:
            c = OpenAI(api_key=st.session_state.ai_k)
            prompt = f"Analyze: Flux {last['flux']}, Brain {last['br_trend']}, SMC {last['smc_buy']}, Matrix {last['mat_sig']}, RQZO {last['rqzo']}. Give Bias and Entry."
            r = c.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": prompt}])
            st.info(r.choices[0].message.content)
        else:
            st.warning("OpenAI not available or key missing.")
