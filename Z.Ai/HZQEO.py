

Here is the rebranded version of the application. I have updated the name from **"The Pentagram"** to **"Strategic Quantum Terminal"** and replaced the informal/occult nomenclature with a professional, formal tone suitable for institutional use.

```python
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

# Scientific imports for HZQEO
import math
from scipy.special import zeta as scipy_zeta
from scipy.stats import entropy as scipy_entropy

# --- OpenAI Integration ---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ==========================================
# 1. UI THEME & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Strategic Quantum Terminal | SQT",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    :root {
        --bg: #000000;
        --card: #080808;
        --border: #1a1a1a;
        --accent: #2196F3; /* Changed to Professional Blue */
        --bull: #00E676;
        --bear: #FF1744;
        --heat: #FFD600;
        --text: #e0e0e0;
        --quantum-primary: #00f5d4;
        --quantum-secondary: #9d4edd;
    }
    
    .stApp { background-color: var(--bg); font-family: 'JetBrains Mono', monospace; color: var(--text); }
    
    div[data-testid="metric-container"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    
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
    
    .stTextInput input, .stSelectbox div { background-color: #0c0c0c !important; border: 1px solid #333 !important; }
    
    .quantum-metric {
        background: rgba(0,0,0,0.3);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #222;
    }
    .quantum-metric .label { font-size: 0.7rem; color: #888; margin-bottom: 5px; }
    .quantum-metric .value { font-size: 1.2rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INTEGRATED HZQEO ENGINE
# ==========================================
class HZQEOEngine:
    """Hyperbolic Zeta-Quantum Entropy Oscillator Engine"""
    
    def __init__(self, params):
        self.lookback = params.get('hz_lookback', 100)
        self.zeta_terms = params.get('hz_terms', 50)
        self.entropy_period = params.get('hz_entropy_period', 20)
        self.reynolds_critical = params.get('hz_reynolds', 2000)
        self.hyperbolic_curvature = params.get('hz_curvature', 0.5)
    
    @staticmethod
    def walsh_hadamard(n: int, x: float) -> float:
        sign = 1.0
        x_int = int(x * 256)
        for i in range(8):
            bit_n = (n >> i) & 1
            bit_x = (x_int >> i) & 1
            if bit_n & bit_x == 1:
                sign = -sign
        return sign
    
    @staticmethod
    def normalize_price(price: float, min_p: float, max_p: float) -> float:
        if max_p - min_p == 0: return 0.5
        return (price - min_p) / (max_p - min_p)
    
    @staticmethod
    def calculate_atr(df_window: pd.DataFrame) -> float:
        if len(df_window) < 2: return 0.0
        high = df_window['high'].values
        low = df_window['low'].values
        close = df_window['close'].values
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        true_ranges = np.maximum(np.maximum(tr1, tr2), tr3)
        return true_ranges.mean() if len(true_ranges) > 0 else 0.0

    def calculate_zeta_component(self, df: pd.DataFrame) -> pd.Series:
        if df.empty or len(df) < self.lookback: return pd.Series(dtype=float)
        result = np.zeros(len(df))
        for idx in range(self.lookback, len(df)):
            window = df.iloc[idx - self.lookback:idx]
            norm_price = self.normalize_price(df.iloc[idx]['close'], window['low'].min(), window['high'].max())
            log_returns = np.log(df.iloc[idx-19:idx+1]['close'] / df.iloc[idx-20:idx]['close'].values)
            s_real = 0.5 + 0.1 * log_returns.std() / 0.01
            zeta_osc = 0.0
            for n in range(1, min(self.zeta_terms, 30)):
                wh = self.walsh_hadamard(n, norm_price)
                zeta_osc += wh * math.cos(10 * norm_price * math.log(n + 1)) / math.pow(n + 1, s_real)
            result[idx] = np.tanh(zeta_osc * 2)
        return pd.Series(result, index=df.index)

    def calculate_entropy_factor(self, df: pd.DataFrame) -> pd.Series:
        if df.empty: return pd.Series(dtype=float)
        entropy_vals = np.zeros(len(df))
        bins = 10
        for idx in range(self.entropy_period, len(df)):
            window = df.iloc[idx - self.entropy_period:idx]['close'].values
            if len(window) < 2 or np.std(window) == 0:
                entropy_vals[idx] = 0.5
                continue
            hist, _ = np.histogram(window, bins=bins)
            hist = hist.astype(float)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum
                ent = 0.0
                for p in hist:
                    if p > 0: ent -= p * math.log(p)
                entropy_vals[idx] = 1 - (ent / math.log(bins))
            else:
                entropy_vals[idx] = 0.5
        return pd.Series(entropy_vals, index=df.index)
    
    def calculate_fluid_factor(self, df: pd.DataFrame) -> pd.Series:
        if df.empty: return pd.Series(dtype=float)
        fluid_vals = np.zeros(len(df))
        for idx in range(1, len(df)):
            price_velocity = abs(df.iloc[idx]['close'] - df.iloc[idx-1]['close']) / df.iloc[idx-1]['close']
            if idx >= 10:
                window = df.iloc[idx-10:idx]
                stdev_price = window['close'].std()
                high_low_range = window['high'].max() - window['low'].min()
                fractal_dim = 1.0 + math.log(max(stdev_price / high_low_range, 0.001)) / math.log(2) if high_low_range > 0 else 1.0
            else:
                fractal_dim = 1.0
            reynolds_num = price_velocity * fractal_dim * self.hyperbolic_curvature
            fluid_vals[idx] = math.exp(-abs(reynolds_num) / self.reynolds_critical)
        return pd.Series(fluid_vals, index=df.index)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()
        results = pd.DataFrame(index=df.index)
        results['zeta_osc'] = self.calculate_zeta_component(df)
        results['entropy_factor'] = self.calculate_entropy_factor(df)
        results['fluid_factor'] = self.calculate_fluid_factor(df)
        results['tunnel_prob'] = 1.0 - (df['close'].rolling(50).max() - df['close']) / (df['close'].rolling(50).max() - df['close'].rolling(50).min() + 1e-10)
        
        results['hzqeo_raw'] = results['zeta_osc'] * results['tunnel_prob'] * results['entropy_factor'] * results['fluid_factor']
        results['hzqeo_norm'] = np.tanh(results['hzqeo_raw'] * 2)
        results['hzqeo_signal'] = 0
        results.loc[results['hzqeo_norm'] > 0.8, 'hzqeo_signal'] = 1
        results.loc[results['hzqeo_norm'] < -0.8, 'hzqeo_signal'] = -1
        return results

# ==========================================
# 3. CORE ANALYTICAL ENGINE
# ==========================================
def rma(series, length): return series.ewm(alpha=1/length, adjust=False).mean()
def wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
def hma(series, length): return wma(2 * wma(series, length // 2) - wma(series, length), int(np.sqrt(length)))
def double_smooth(src, l1, l2): return src.ewm(span=l1).mean().ewm(span=l2).mean()

# CORE 1: MOMENTUM VECTOR
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
    return df

# CORE 2: NEURAL PROCESSOR
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
    return df

# CORE 3: QUANTUM DYNAMICS
def calc_rqzo(df, p):
    df = df.copy()
    mn, mx = df["close"].rolling(100).min(), df["close"].rolling(100).max()
    norm = (df["close"] - mn) / (mx - mn + 1e-10)
    gamma = 1 / np.sqrt(1 - (np.clip((norm - norm.shift(1)).abs(), 0, 0.049)/0.05)**2)
    tau = (np.arange(len(df)) % 100) / gamma
    df["rqzo"] = sum([(n ** -0.5) * np.sin(tau * np.log(n)) for n in range(1, 10)]) * 10
    return df

# CORE 4: TREND MATRIX
def calc_matrix(df, p):
    df = df.copy()
    rs = rma(df["close"].diff().clip(lower=0), 14) / (rma(-df["close"].diff().clip(upper=0), 14) + 1e-10)
    rsi = 100 - (100/(1+rs))
    df["mfi"] = ((rsi - 50) * (df["volume"] / df["volume"].rolling(20).mean())).ewm(span=3).mean()
    df["hw"] = 100 * (double_smooth(df["close"].diff(), 25, 13) / (double_smooth(df["close"].diff().abs(), 25, 13) + 1e-10)) / 2
    df["mat_sig"] = np.sign(df["mfi"]) + np.sign(df["hw"])
    return df

# CORE 5: STRUCTURAL ANALYSIS
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
# 4. DATA & STATE HANDLING
# ==========================================
@st.cache_data(ttl=5)
def get_data(exch, sym, tf, lim):
    try:
        ex = getattr(ccxt, exch.lower())({"enableRateLimit": True})
        ohlcv = ex.fetch_ohlcv(sym, tf, limit=lim)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except: return pd.DataFrame()

def init():
    defaults = {
        "exch": "Kraken", "sym": "BTC/USD", "tf": "15m", "lim": 500,
        # Vector Parameters
        "vec_l": 14, "vol_n": 55, "vec_sm": 5, "vec_super": 0.6, "vec_resist": 0.3, "vec_strict": 1.0,
        # Neural Parameters
        "br_l": 55, "br_m": 1.5, "br_th": 2.0, 
        # Structure Parameters
        "smc_l": 55, "auto": False,
        # HZQEO Parameters
        "hz_lookback": 100, "hz_terms": 50, "hz_entropy_period": 20, "hz_reynolds": 2000, "hz_curvature": 0.5,
        # API
        "ai_k": "", "tg_t": "", "tg_c": "", "x_k": "", "x_s": "", "x_at": "", "x_as": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init()

# ==========================================
# 5. SYSTEM CONTROL PANEL (SIDEBAR)
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ SYSTEM CONFIGURATION")
    
    with st.expander("🌍 Feed Configuration", expanded=True):
        st.session_state.exch = st.selectbox("Exchange", ["Kraken", "Binance", "Bybit", "Coinbase", "OKX"])
        st.session_state.sym = st.text_input("Asset Ticker", st.session_state.sym)
        st.session_state.tf = st.selectbox("Interval", ["1m","5m","15m","1h","4h","1d"], index=2)
        if st.checkbox("🔄 Auto-Refresh (60s)", st.session_state.auto):
            time_lib.sleep(60)
            st.rerun()

    with st.expander("⚛️ HZQEO Parameters"):
        st.session_state.hz_lookback = st.slider("Zeta Lookback", 50, 200, st.session_state.hz_lookback)
        st.session_state.hz_terms = st.slider("Zeta Terms", 10, 100, st.session_state.hz_terms)
        st.session_state.hz_entropy_period = st.slider("Entropy Period", 10, 50, st.session_state.hz_entropy_period)

    with st.expander("📡 Communication APIs"):
        st.session_state.tg_t = st.text_input("Telegram Token", st.session_state.tg_t, type="password")
        st.session_state.tg_c = st.text_input("Telegram Chat ID", st.session_state.tg_c)
        st.session_state.x_k = st.text_input("X API Key", st.session_state.x_k, type="password")
        st.session_state.ai_k = st.text_input("OpenAI Secret", st.session_state.ai_k, type="password")

    if st.button("🔄 REFRESH DATA", type="primary", use_container_width=True):
        get_data.clear()
        st.rerun()

# ==========================================
# 6. SYSTEM EXECUTION
# ==========================================
df = get_data(st.session_state.exch, st.session_state.sym, st.session_state.tf, st.session_state.lim)
if df.empty:
    st.error("CONNECTION ERROR: Unable to retrieve market data. Please verify exchange and ticker.")
    st.stop()

# Initialize Engines
hzqeo_engine = HZQEOEngine(st.session_state)

# Chain Computation
df = calc_vector(df, st.session_state)
df = calc_brain(df, st.session_state)
df = calc_rqzo(df, st.session_state)
df = calc_matrix(df, st.session_state)
df = calc_smc(df, st.session_state)

# Attach HZQEO
hzqeo_results = hzqeo_engine.calculate(df)
df = pd.concat([df, hzqeo_results], axis=1)

last, prev = df.iloc[-1], df.iloc[-2]

# Broadcasting Logic
def send_telegram(token, chat_id, text):
    if token and chat_id:
        try: requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text})
        except: pass

def post_x(key, secret, at, ats, text):
    if key and at:
        try:
            client = tweepy.Client(consumer_key=key, consumer_secret=secret, access_token=at, access_token_secret=ats)
            client.create_tweet(text=text)
        except: pass

event = None
# Core Events
if last["br_trend"] == 1 and prev["br_trend"] != 1: event = f"NEURAL BULL: {st.session_state.sym}"
elif last["smc_buy"] and not prev["smc_buy"]: event = f"STRUCTURE BUY: {st.session_state.sym}"
elif last["vec_state"] == 2 and prev["vec_state"] != 2: event = f"MOMENTUM SURGE: {st.session_state.sym}"

# HZQEO Events
hzqeo_sig = last.get('hzqeo_signal', 0)
if hzqeo_sig == 1 and prev.get('hzqeo_signal', 0) != 1: event = f"ZETA OVERBOUGHT: {st.session_state.sym}"
elif hzqeo_sig == -1 and prev.get('hzqeo_signal', 0) != -1: event = f"ZETA OVERSOLD: {st.session_state.sym}"

if event:
    send_telegram(st.session_state.tg_t, st.session_state.tg_c, event)
    post_x(st.session_state.x_k, st.session_state.x_s, st.session_state.x_at, st.session_state.x_as, event)

# ==========================================
# 7. VISUALIZATION DASHBOARD
# ==========================================
st.title(f"📊 STRATEGIC QUANTUM TERMINAL // {st.session_state.sym}")

# Global HUD
h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("Price", f"{last['close']:.2f}")
h2.metric("Momentum", f"{last['flux']:.3f}", delta=("Bull" if last['flux']>0 else "Bear"))
h3.metric("State", ("Stable" if last['gate'] else "Volatile"))
h4.metric("Trend", int(last['mat_sig']))
h5.metric("HZQEO", f"{last.get('hzqeo_norm', 0):.2f}", delta_color="inverse")

def clean_plot():
    return go.Layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=500)

# --- Tabs ---
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "🧠 Neural Engine", "🏛️ Market Structure", "⚡ Momentum Flow", "⚛️ Quantum Dynamics", 
    "🌀 Zeta-Entropy", "📺 Market Charts", "🤖 Strategic AI"
])

# TAB 1: NEURAL ENGINE
with t1:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_u'], line=dict(color='rgba(0,230,118,0.2)'), fill=None))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_l_band'], line=dict(color='rgba(255,23,68,0.2)'), fill='tonexty'))
        fig.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Neural Analysis</div>
        <div class="diag-item"><span class="diag-label">Bias:</span><span class="diag-val">{('BULLISH' if last['br_trend']==1 else 'BEARISH')}</span></div>
        <div class="diag-item"><span class="diag-label">Entropy:</span><span class="diag-val">{last['ent']:.2f}</span></div>
        <div class="diag-item"><span class="diag-label">Flow:</span><span class="diag-val">{last['flow']:.3f}</span></div></div>""", unsafe_allow_html=True)

# TAB 2: MARKET STRUCTURE
with t2:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['smc_base'], line=dict(color='cyan', width=1)))
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Structure Analysis</div>
        <div class="diag-item"><span class="diag-label">Signal:</span><span class="diag-val">{('BUY' if last['smc_buy'] else ('SELL' if last['smc_sell'] else 'HOLD'))}</span></div>
        <div class="diag-item"><span class="diag-label">Gap Bull:</span><span class="diag-val">{last['fvg_b']}</span></div>
        <div class="diag-item"><span class="diag-label">Gap Bear:</span><span class="diag-val">{last['fvg_s']}</span></div></div>""", unsafe_allow_html=True)

# TAB 3: MOMENTUM FLOW
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
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Momentum Analysis</div>
        <div class="diag-item"><span class="diag-label">Efficiency:</span><span class="diag-val">{last['eff']*100:.1f}%</span></div>
        <div class="diag-item"><span class="diag-label">Flux:</span><span class="diag-val">{last['flux']:.3f}</span></div></div>""", unsafe_allow_html=True)

# TAB 4: QUANTUM DYNAMICS
with t4:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rqzo'], name='Quantum Oscillator', line=dict(color='#D500F9', width=1)))
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.markdown(f"""<div class="diag-panel"><div class="diag-header">Quantum State</div>
        <div class="diag-item"><span class="diag-label">RQZO Val:</span><span class="diag-val">{last['rqzo']:.2f}</span></div>
        <div class="diag-text" style="margin-top:20px; color:#555">Core analysis of quantum resonance patterns.</div></div>""", unsafe_allow_html=True)

# TAB 5: ZETA-ENTROPY
with t5:
    st.markdown("### 🌀 Hyperbolic Zeta-Quantum Entropy Analysis")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    # Metrics
    col_m1.markdown(f"<div class='quantum-metric'><div class='label'>HZQEO VAL</div><div class='value' style='color:var(--quantum-primary)'>{last.get('hzqeo_norm', 0):.4f}</div></div>", unsafe_allow_html=True)
    col_m2.markdown(f"<div class='quantum-metric'><div class='label'>ZETA OSC</div><div class='value' style='color:#9d4edd'>{last.get('zeta_osc', 0):.4f}</div></div>", unsafe_allow_html=True)
    col_m3.markdown(f"<div class='quantum-metric'><div class='label'>ENTROPY</div><div class='value' style='color:#ffd60a'>{last.get('entropy_factor', 0):.4f}</div></div>", unsafe_allow_html=True)
    col_m4.markdown(f"<div class='quantum-metric'><div class='label'>FLUIDITY</div><div class='value' style='color:#00ff9d'>{last.get('fluid_factor', 0):.4f}</div></div>", unsafe_allow_html=True)
    
    # Signal Box
    sig_val = last.get('hzqeo_norm', 0)
    if sig_val > 0.8:
        sig_txt = "OVERBOUGHT"
        sig_col = "#FF1744"
        rec = "Consider reducing long exposure"
    elif sig_val < -0.8:
        sig_txt = "OVERSOLD"
        sig_col = "#00E676"
        rec = "Potential accumulation opportunity"
    else:
        sig_txt = "NEUTRAL"
        sig_col = "#888"
        rec = "Market within equilibrium"
        
    st.markdown(f"<div style='padding:10px; border:1px solid {sig_col}; border-radius:4px; margin-bottom:15px; background: rgba(255,255,255,0.05)'><strong style='color:{sig_col}'>SIGNAL: {sig_txt}</strong> | {rec}</div>", unsafe_allow_html=True)

    # Chart
    fig_h = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
    fig_h.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    
    fig_h.add_trace(go.Scatter(x=df.index, y=df['hzqeo_norm'], name='HZQEO', line=dict(color='var(--quantum-primary)', width=2)), row=2, col=1)
    fig_h.add_hline(y=0.8, line_dash="dash", line_color="red", row=2, col=1)
    fig_h.add_hline(y=-0.8, line_dash="dash", line_color="green", row=2, col=1)
    
    comps = ['zeta_osc', 'entropy_factor', 'fluid_factor']
    colors = ['#9d4edd', '#ffd60a', '#00ff9d']
    for c, col in zip(comps, colors):
        if c in df.columns:
            fig_h.add_trace(go.Scatter(x=df.index, y=df[c], name=c, line=dict(color=col, width=1)), row=3, col=1)
            
    fig_h.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=800, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_h, use_container_width=True)

with t6:
    st.markdown("### External Chart Verification")
    tv_s = st.session_state.sym.replace("/", "")
    st.components.v1.html(f"""<script src="https://s3.tradingview.com/tv.js"></script>
    <script>new TradingView.widget({{"width": "100%", "height": 600, "symbol": "{st.session_state.exch.upper()}:{tv_s}", "theme": "dark", "style": "1", "container_id": "tv"}});</script>
    <div id="tv"></div>""", height=620)

with t7:
    st.markdown("### 🤖 Strategic Analysis AI")
    persona = st.selectbox("Analyst Profile", ["Strategist", "Physicist", "Risk Manager", "Quant"])
    if st.button("GENERATE ANALYSIS"):
        if OPENAI_AVAILABLE and st.session_state.ai_k:
            c = OpenAI(api_key=st.session_state.ai_k)
            
            context = f"""
            Analyze {st.session_state.sym}:
            1. CORE METRICS: Momentum {last['flux']}, Neural Trend {last['br_trend']}, Structure Signal {last['smc_buy']}.
            2. HZQEO METRICS: Oscillator {last.get('hzqeo_norm', 0)}, Zeta Strength {last.get('zeta_osc', 0)}, Entropy {last.get('entropy_factor', 0)}.
            
            Provide a professional assessment including: Market Bias, Risk Level, and Strategic Entry/Exit logic.
            """
            
            r = c.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": context}])
            st.info(r.choices[0].message.content)
        else:
            st.warning("OpenAI API Key required for analysis.")
```
