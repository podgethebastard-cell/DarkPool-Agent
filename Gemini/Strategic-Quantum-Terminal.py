import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time as time_lib
import math
import tweepy

# ==========================================
# 1. SYSTEM CONFIGURATION & DPC CSS
# ==========================================
st.set_page_config(
    page_title="Strategic Quantum Terminal | SQT",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded",
)

# Institutional DPC CSS Architecture
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    :root {
        --bg-color: #0e1117;
        --card-bg: #151925;
        --text-color: #e0e0e0;
        --accent-cyan: #00f5d4;
        --accent-purple: #9d4edd;
        --bull-color: #00E676;
        --bear-color: #FF1744;
        --border-color: #2a2e3a;
    }

    .stApp {
        background-color: var(--bg-color);
        font-family: 'Roboto Mono', monospace;
        color: var(--text-color);
    }
    
    /* Custom Metric Containers */
    .metric-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        padding: 15px;
        border-radius: 4px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: var(--accent-cyan);
        box-shadow: 0 0 15px rgba(0, 245, 212, 0.2);
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 5px;
    }
    .delta-pos { color: var(--bull-color); }
    .delta-neg { color: var(--bear-color); }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #11141d;
        border-right: 1px solid var(--border-color);
    }
    
    /* Input Fields */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #0c0c0c !important;
        border: 1px solid #333 !important;
        color: var(--text-color) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #888;
        font-family: 'Roboto Mono', monospace;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: var(--accent-cyan);
        border-bottom: 2px solid var(--accent-cyan);
    }
</style>
""", unsafe_allow_html=True)

# --- OpenAI Integration ---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ==========================================
# 2. HZQEO ENGINE (Novel Math)
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
    def walsh_hadamard_sign(n: int, x: float) -> float:
        """Bitwise Walsh-Hadamard sign calculation."""
        sign = 1.0
        x_int = int(x * 256)
        # Unrolled loop for slight performance gain
        if (n & 1) & (x_int & 1): sign = -sign
        if ((n >> 1) & 1) & ((x_int >> 1) & 1): sign = -sign
        if ((n >> 2) & 1) & ((x_int >> 2) & 1): sign = -sign
        if ((n >> 3) & 1) & ((x_int >> 3) & 1): sign = -sign
        if ((n >> 4) & 1) & ((x_int >> 4) & 1): sign = -sign
        if ((n >> 5) & 1) & ((x_int >> 5) & 1): sign = -sign
        if ((n >> 6) & 1) & ((x_int >> 6) & 1): sign = -sign
        if ((n >> 7) & 1) & ((x_int >> 7) & 1): sign = -sign
        return sign
    
    def calculate_zeta_component(self, df: pd.DataFrame) -> pd.Series:
        if df.empty or len(df) < self.lookback: return pd.Series(dtype=float)
        
        result = np.zeros(len(df))
        close_vals = df['close'].values
        low_vals = df['low'].values
        high_vals = df['high'].values
        
        # Iterating through time
        for idx in range(self.lookback, len(df)):
            window_low = np.min(low_vals[idx - self.lookback:idx])
            window_high = np.max(high_vals[idx - self.lookback:idx])
            current_price = close_vals[idx]
            
            denom = window_high - window_low
            norm_price = (current_price - window_low) / denom if denom != 0 else 0.5
            
            # s_real (Real component of complex s) based on volatility
            log_returns = np.log(close_vals[idx-19:idx+1] / close_vals[idx-20:idx])
            vol = np.std(log_returns)
            s_real = 0.5 + 0.1 * (vol / 0.01)
            
            zeta_osc = 0.0
            # Inner summation
            for n in range(1, min(self.zeta_terms, 30)):
                wh = self.walsh_hadamard_sign(n, norm_price)
                term = wh * math.cos(10 * norm_price * math.log(n + 1)) / math.pow(n + 1, s_real)
                zeta_osc += term
                
            result[idx] = np.tanh(zeta_osc * 2)
            
        return pd.Series(result, index=df.index)

    def calculate_entropy_factor(self, df: pd.DataFrame) -> pd.Series:
        if df.empty: return pd.Series(dtype=float)
        
        entropy_vals = np.zeros(len(df))
        close_vals = df['close'].values
        bins = 10
        
        # Rolling window entropy
        for idx in range(self.entropy_period, len(df)):
            window = close_vals[idx - self.entropy_period:idx]
            if np.std(window) == 0:
                entropy_vals[idx] = 0.5
                continue
                
            hist, _ = np.histogram(window, bins=bins, density=True)
            # Remove zeros for log calculation
            hist = hist[hist > 0]
            if len(hist) > 0:
                ent = -np.sum(hist * np.log(hist))  # Shannon entropy
                # Normalize by max entropy log(bins)
                max_ent = np.log(bins)
                # Invert: Higher entropy (chaos) -> Lower factor (0), Order -> 1
                entropy_vals[idx] = 1.0 - (ent / max_ent) if max_ent > 0 else 0.5
            else:
                entropy_vals[idx] = 0.5
                
        return pd.Series(entropy_vals, index=df.index)
    
    def calculate_fluid_factor(self, df: pd.DataFrame) -> pd.Series:
        if df.empty: return pd.Series(dtype=float)
        
        fluid_vals = np.zeros(len(df))
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Calculate velocity
        velocity = np.zeros_like(close)
        velocity[1:] = np.abs(close[1:] - close[:-1]) / close[:-1]
        
        for idx in range(10, len(df)):
            window_slice = slice(idx-10, idx)
            std_price = np.std(close[window_slice])
            hl_range = np.max(high[window_slice]) - np.min(low[window_slice])
            
            fractal_dim = 1.0
            if hl_range > 0:
                fractal_dim = 1.0 + math.log(max(std_price / hl_range, 0.001)) / math.log(2)
            
            reynolds_num = velocity[idx] * fractal_dim * self.hyperbolic_curvature * 10000
            fluid_vals[idx] = math.exp(-abs(reynolds_num) / self.reynolds_critical)
            
        return pd.Series(fluid_vals, index=df.index)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()
        results = pd.DataFrame(index=df.index)
        
        with st.spinner("Computing Hyperbolic Zeta Components..."):
            results['zeta_osc'] = self.calculate_zeta_component(df)
            results['entropy_factor'] = self.calculate_entropy_factor(df)
            results['fluid_factor'] = self.calculate_fluid_factor(df)
            
            # Tunnel Probability (Price position within Bollinger-esque channel)
            roll_max = df['close'].rolling(50).max()
            roll_min = df['close'].rolling(50).min()
            results['tunnel_prob'] = 1.0 - (roll_max - df['close']) / (roll_max - roll_min + 1e-10)
            
            # Synthesis
            results['hzqeo_raw'] = results['zeta_osc'] * results['tunnel_prob'] * results['entropy_factor']
            results['hzqeo_norm'] = np.tanh(results['hzqeo_raw'] * 2)
            
            results['hzqeo_signal'] = 0
            results.loc[results['hzqeo_norm'] > 0.8, 'hzqeo_signal'] = 1
            results.loc[results['hzqeo_norm'] < -0.8, 'hzqeo_signal'] = -1
            
        return results

# ==========================================
# 3. CORE ANALYTICAL ENGINE
# ==========================================
# Helper math functions
def rma(series, length): return series.ewm(alpha=1/length, adjust=False).mean()
def wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
def hma(series, length): return wma(2 * wma(series, length // 2) - wma(series, length), int(np.sqrt(length)))
def double_smooth(src, l1, l2): return src.ewm(span=l1).mean().ewm(span=l2).mean()

# MODULE 1: MOMENTUM FLUX VELOCITY
def calc_momentum_flux(df, p):
    df = df.copy()
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    
    # Efficiency Ratio
    df["eff"] = pd.Series(np.where(rng==0, 0, body/rng)).ewm(span=p["vec_l"]).mean()
    
    # Volume Factor
    vol_mean = df["volume"].rolling(p["vol_n"]).mean()
    vol_fact = np.where(vol_mean==0, 1, df["volume"]/vol_mean)
    
    # Flux Calculation
    raw_v = np.sign(df["close"] - df["open"]) * df["eff"] * vol_fact
    df["flux"] = raw_v.ewm(span=p["vec_sm"]).mean()
    
    th_s, th_r = p["vec_super"] * p["vec_strict"], p["vec_resist"] * p["vec_strict"]
    
    # Vectorized State Selection
    conditions = [
        (df["flux"] > th_s),
        (df["flux"] < -th_s),
        (df["flux"].abs() < th_r)
    ]
    # States: 2 (Surge Bull), -2 (Surge Bear), 0 (Neutral), 1 (Normal)
    df["vec_state"] = np.select(conditions, [2, -2, 0], default=1)
    return df

# MODULE 2: ADAPTIVE NEURAL BAND
def calc_neural_band(df, p):
    df = df.copy()
    # Baseline
    base = hma(df["close"], p["br_l"])
    atr = rma(df["high"]-df["low"], p["br_l"])
    
    df["br_u"] = base + (atr * p["br_m"])
    df["br_l_band"] = base - (atr * p["br_m"])
    
    # Stateful Trend Calculation (Hysteresis)
    # We use iteration here as it's a path-dependent latch
    close_arr = df["close"].values
    upper_arr = df["br_u"].values
    lower_arr = df["br_l_band"].values
    trend = np.zeros(len(df))
    
    curr_trend = 1
    for i in range(1, len(df)):
        if close_arr[i] > upper_arr[i]:
            curr_trend = 1
        elif close_arr[i] < lower_arr[i]:
            curr_trend = -1
        trend[i] = curr_trend
        
    df["br_trend"] = trend
    
    # Entropy Gate
    df["ent_vol"] = df["close"].pct_change().rolling(64).std() * 100
    df["gate"] = df["ent_vol"] < p["br_th"]
    
    return df

# MODULE 3: REGIME MATRIX
def calc_regime_matrix(df, p):
    df = df.copy()
    # RSI V6
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = rma(up, 14) / (rma(down, 14) + 1e-10)
    rsi = 100 - (100/(1+rs))
    
    # Money Flow Index
    df["mfi"] = ((rsi - 50) * (df["volume"] / df["volume"].rolling(20).mean())).ewm(span=3).mean()
    
    # Hurst Wave
    df["hw"] = 100 * (double_smooth(delta, 25, 13) / (double_smooth(delta.abs(), 25, 13) + 1e-10)) / 2
    
    df["mat_sig"] = np.sign(df["mfi"]) + np.sign(df["hw"])
    return df

# MODULE 4: INSTITUTIONAL MARKET STRUCTURE (SMC)
def calc_smc_structure(df, p):
    df = df.copy()
    df["smc_base"] = hma(df["close"], p["smc_l"])
    
    # TCI (Trend Catcher Index)
    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ap.ewm(span=10).mean()
    d = (ap - esa).abs().ewm(span=10).mean()
    tci = (ap - esa) / (0.015 * d + 1e-10)
    tci_smooth = tci.ewm(span=21).mean()
    
    # Signals
    df["smc_buy"] = (df["close"] > df["smc_base"]) & (tci_smooth < 60) & (tci_smooth > tci_smooth.shift(1))
    df["smc_sell"] = (df["close"] < df["smc_base"]) & (tci_smooth > -60) & (tci_smooth < tci_smooth.shift(1))
    
    # Fair Value Gaps
    df["fvg_bull"] = (df["low"] > df["high"].shift(2))
    df["fvg_bear"] = (df["high"] < df["low"].shift(2))
    return df

# ==========================================
# 4. DATA INFRASTRUCTURE
# ==========================================
@st.cache_data(ttl=60)
def fetch_market_data(exchange_id, symbol, timeframe, limit):
    try:
        ex_class = getattr(ccxt, exchange_id.lower())
        ex = ex_class({"enableRateLimit": True})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        st.error(f"Data Feed Error: {str(e)}")
        return pd.DataFrame()

def init_session_state():
    defaults = {
        "exch": "Kraken", "sym": "BTC/USD", "tf": "15m", "lim": 500,
        "vec_l": 14, "vol_n": 55, "vec_sm": 5, "vec_super": 0.6, "vec_resist": 0.3, "vec_strict": 1.0,
        "br_l": 55, "br_m": 1.5, "br_th": 2.0, 
        "smc_l": 55, "auto_refresh": False,
        "hz_lookback": 100, "hz_terms": 50, "hz_entropy_period": 20, "hz_reynolds": 2000, "hz_curvature": 0.5,
        "api_openai": "", "api_tg_token": "", "api_tg_chat": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session_state()

# ==========================================
# 5. UI CONTROL PANEL
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ TERMINAL CONFIG")
    
    with st.expander("🌍 Market Feed", expanded=True):
        st.session_state.exch = st.selectbox("Exchange", ["Kraken", "Binance", "Bybit", "Coinbase", "OKX"])
        st.session_state.sym = st.text_input("Asset Class", st.session_state.sym)
        st.session_state.tf = st.selectbox("Resolution", ["1m","5m","15m","1h","4h","1d"], index=2)
        
        if st.checkbox("Enable Auto-Refresh (60s)", st.session_state.auto_refresh):
            time_lib.sleep(60)
            st.rerun()

    with st.expander("⚛️ HZQEO Calibrations"):
        st.session_state.hz_lookback = st.slider("Zeta Horizon", 50, 200, st.session_state.hz_lookback)
        st.session_state.hz_curvature = st.slider("Hyperbolic Curve", 0.1, 1.0, st.session_state.hz_curvature)

    with st.expander("🔑 Institutional Keys"):
        st.session_state.api_openai = st.text_input("OpenAI Sk-Key", st.session_state.api_openai, type="password")
        st.session_state.api_tg_token = st.text_input("Telegram Bot Token", st.session_state.api_tg_token, type="password")

    if st.button("EXECUTE QUERY", type="primary", use_container_width=True):
        fetch_market_data.clear()
        st.rerun()

# ==========================================
# 6. SYSTEM EXECUTION PIPELINE
# ==========================================
df = fetch_market_data(st.session_state.exch, st.session_state.sym, st.session_state.tf, st.session_state.lim)

if df.empty:
    st.warning("⚠️ No Data Stream Available. Verify Ticker/Exchange.")
    st.stop()

# Computation Pipeline
hz_engine = HZQEOEngine(st.session_state)
df = calc_momentum_flux(df, st.session_state)
df = calc_neural_band(df, st.session_state)
df = calc_regime_matrix(df, st.session_state)
df = calc_smc_structure(df, st.session_state)
hz_results = hz_engine.calculate(df)
df = pd.concat([df, hz_results], axis=1)

last = df.iloc[-1]
prev = df.iloc[-2]

# ==========================================
# 7. INSTITUTIONAL DASHBOARD
# ==========================================
st.title(f"STRATEGIC QUANTUM TERMINAL // {st.session_state.sym}")

# Custom Metric HUD
col1, col2, col3, col4, col5 = st.columns(5)

def render_metric(col, label, value, delta=None, delta_color="normal"):
    delta_html = ""
    if delta:
        color_class = "delta-pos" if delta_color == "pos" else "delta-neg"
        delta_html = f"<div class='metric-delta {color_class}'>{delta}</div>"
    
    html = f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div>
        {delta_html}
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)

render_metric(col1, "ASSET PRICE", f"{last['close']:.2f}")
render_metric(col2, "FLUX VELOCITY", f"{last['flux']:.3f}", 
              "BULLISH" if last['flux'] > 0 else "BEARISH", 
              "pos" if last['flux'] > 0 else "neg")
render_metric(col3, "REGIME STATE", "STABLE" if last['gate'] else "VOLATILE", 
              "Low Entropy" if last['gate'] else "High Entropy", 
              "pos" if last['gate'] else "neg")
render_metric(col4, "STRUCTURE", "ACCUM" if last['smc_buy'] else ("DISTRIB" if last['smc_sell'] else "NEUTRAL"))
render_metric(col5, "HZQEO INDEX", f"{last.get('hzqeo_norm', 0):.2f}", 
              "OVERBOUGHT" if last.get('hzqeo_norm', 0) > 0.8 else ("OVERSOLD" if last.get('hzqeo_norm', 0) < -0.8 else "EQ"), 
              "neg" if abs(last.get('hzqeo_norm', 0)) > 0.8 else "pos")

# Visualization Tabs
t1, t2, t3, t4, t5 = st.tabs([
    "⚛️ HZQEO ANALYTICS", "🧠 NEURAL BANDS", "🌊 FLUX MOMENTUM", "🏛 MARKET STRUCTURE", "🤖 STRATEGIC AI"
])

def create_base_chart(height=500):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#151925",
        plot_bgcolor="#151925",
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
        font=dict(family="Roboto Mono", color="#e0e0e0")
    )
    return fig

# TAB 1: HZQEO
with t1:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        fig = create_base_chart(600)
        # Price
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        
        # HZQEO Signal
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hzqeo_norm'], name='HZQEO', line=dict(color='#00f5d4', width=2)), row=2, col=1)
        fig.add_hline(y=0.8, line_dash="dot", line_color="#FF1744", row=2, col=1)
        fig.add_hline(y=-0.8, line_dash="dot", line_color="#00E676", row=2, col=1)
        
        # Components
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['entropy_factor'], name='Entropy', line=dict(color='#9d4edd', width=1), opacity=0.5), row=2, col=1)
        
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) # Optional: Hide weekends for crypto if needed, usually remove for crypto
        st.plotly_chart(fig, use_container_width=True)
        
    with col_r:
        st.markdown("### 🌀 Quantum Status")
        st.info(f"Zeta Oscillation: {last.get('zeta_osc', 0):.4f}")
        st.info(f"Entropy Factor: {last.get('entropy_factor', 0):.4f}")
        st.info(f"Fluid Reynolds: {last.get('fluid_factor', 0):.4f}")
        st.markdown("---")
        st.caption("The HZQEO engine utilizes Walsh-Hadamard transforms and hyperbolic tangents to detect non-linear pivot points in the asset's quantum state.")

# TAB 2: NEURAL BANDS
with t2:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_u'], line=dict(color='rgba(0, 230, 118, 0.5)'), name='Neural Upper'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_l_band'], line=dict(color='rgba(255, 23, 68, 0.5)'), fill='tonexty', fillcolor='rgba(255,255,255,0.02)', name='Neural Lower'))
    
    # Buy/Sell Markers from Trend
    buy_sig = df[(df['br_trend'] == 1) & (df['br_trend'].shift(1) == -1)]
    sell_sig = df[(df['br_trend'] == -1) & (df['br_trend'].shift(1) == 1)]
    
    fig.add_trace(go.Scatter(x=buy_sig['timestamp'], y=buy_sig['low'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00E676'), name='Neural Buy'))
    fig.add_trace(go.Scatter(x=sell_sig['timestamp'], y=sell_sig['high'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#FF1744'), name='Neural Sell'))
    
    fig.update_layout(template="plotly_dark", paper_bgcolor="#151925", plot_bgcolor="#151925", height=600, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: FLUX
with t3:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
    
    colors = ["#00E676" if x==2 else ("#FF1744" if x==-2 else "#555") for x in df["vec_state"]]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['flux'], marker_color=colors, name='Momentum Flux'), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", paper_bgcolor="#151925", plot_bgcolor="#151925", height=600)
    st.plotly_chart(fig, use_container_width=True)

# TAB 5: AI STRATEGY
with t5:
    st.markdown("### 🧠 Generative Strategic Analysis")
    col_ai, col_p = st.columns([4, 1])
    
    with col_p:
        persona = st.selectbox("Analyst Persona", ["Quantitative Researcher", "Risk Manager", "Macro Strategist"])
        
    with col_ai:
        if st.button("INITIALIZE STRATEGIC REPORT"):
            if OPENAI_AVAILABLE and st.session_state.api_openai:
                client = OpenAI(api_key=st.session_state.api_openai)
                
                prompt = f"""
                Act as an elite {persona}. Analyze the following institutional metrics for {st.session_state.sym}:
                
                [QUANTUM METRICS]
                - HZQEO Index: {last.get('hzqeo_norm', 0):.4f} (Thresholds: +/- 0.8)
                - Zeta Oscillation: {last.get('zeta_osc', 0):.4f}
                - Entropy Factor: {last.get('entropy_factor', 0):.4f} (1.0 = Order, 0.0 = Chaos)
                
                [TECHNICAL METRICS]
                - Momentum Flux: {last['flux']:.4f} (State: {last['vec_state']})
                - Neural Trend: {'BULLISH' if last['br_trend']==1 else 'BEARISH'}
                - Market Structure: {'ACCUMULATION' if last['smc_buy'] else ('DISTRIBUTION' if last['smc_sell'] else 'NEUTRAL')}
                
                Produce a concise, bullet-pointed strategic report. Focus on probability, risk-reward ratios, and potential invalidation levels.
                Avoid generic advice. Use institutional terminology.
                """
                
                with st.spinner("Generating Neural Analysis..."):
                    try:
                        response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": prompt}])
                        st.markdown(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI Generation Failed: {e}")
            else:
                st.error("API Key Missing or OpenAI module not installed.")
