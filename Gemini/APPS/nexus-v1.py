import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timezone, timedelta
import requests
import time
import math
import random
import urllib.parse
from scipy.stats import linregress
from openai import OpenAI

# ==========================================
# 1. PAGE CONFIGURATION & ARCHITECT THEME
# ==========================================
st.set_page_config(
    page_title="NEXUS TERMINAL // ARCHITECT",
    layout="wide",
    page_icon="👁️",
    initial_sidebar_state="expanded",
)

# DPC CSS ARCHITECTURE
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Roboto+Mono:wght@400;700&display=swap');
    
    :root {
        --bg: #050505;
        --card-bg: #0a0a0a;
        --border: #1f1f1f;
        --accent: #00ff00;
        --bear: #ff0033;
        --text: #e0e0e0;
        --neon-glow: 0 0 10px rgba(0, 255, 0, 0.3);
    }

    .stApp { background-color: var(--bg); font-family: 'JetBrains Mono', monospace; color: var(--text); }
    
    /* Headers */
    h1, h2, h3 { font-family: 'Roboto Mono', monospace; letter-spacing: -1px; }
    .title-glow {
        font-size: 2.2rem;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 0 15px rgba(0, 255, 0, 0.5);
        margin-bottom: 0px;
    }
    .subtitle { font-size: 0.8rem; color: #666; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 20px; }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--accent);
        box-shadow: var(--neon-glow);
    }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono'; font-size: 1.4rem !important; color: #fff !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #888 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #111;
        border: 1px solid #333;
        color: #888;
        border-radius: 4px 4px 0 0;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--bg);
        color: var(--accent);
        border-bottom: 2px solid var(--accent);
    }

    /* Inputs */
    .stTextInput input, .stSelectbox div, .stNumberInput input {
        background-color: #111 !important;
        color: #fff !important;
        border: 1px solid #333 !important;
        border-radius: 4px;
    }
    
    /* Custom Components */
    .diag-panel {
        background: #080808;
        border-left: 3px solid var(--accent);
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 0 4px 4px 0;
    }
    .diag-header { color: #666; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .diag-value { color: #fff; font-size: 1.1rem; font-weight: bold; font-family: 'JetBrains Mono'; }
    
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. THE MATHEMATICAL CORE
# ==========================================

# --- UTILS ---
def safe_float(x):
    try: return float(x)
    except: return 0.0

def rma(series, length):
    """Running Moving Average (Pine Script equivalent)"""
    return series.ewm(alpha=1/length, adjust=False).mean()

def wma(series, length):
    """Weighted Moving Average"""
    length = int(length)
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(series, length):
    """Hull Moving Average"""
    length = int(length)
    half_length = int(length / 2)
    sqrt_length = int(math.sqrt(length))
    wma_half = wma(series, half_length)
    wma_full = wma(series, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_length)

def atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return rma(tr, length)

# --- GOD MODE INDICATORS ---
def calc_god_mode(df):
    """Calculates Trend, Squeeze, Money Flow, and Confluence"""
    df = df.copy()
    
    # 1. Apex Trend (HMA + ATR Bands)
    df['HMA_55'] = hma(df['Close'], 55)
    df['ATR_55'] = atr(df, 55)
    df['Apex_Upper'] = df['HMA_55'] + (df['ATR_55'] * 1.5)
    df['Apex_Lower'] = df['HMA_55'] - (df['ATR_55'] * 1.5)
    
    # Trend State: 1 = Bull, -1 = Bear
    df['Apex_Trend'] = 0
    df.loc[df['Close'] > df['Apex_Upper'], 'Apex_Trend'] = 1
    df.loc[df['Close'] < df['Apex_Lower'], 'Apex_Trend'] = -1
    df['Apex_Trend'] = df['Apex_Trend'].replace(0, method='ffill')

    # 2. Squeeze Momentum
    # BB
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    # KC
    range_ma = atr(df, 20)
    upper_kc = sma20 + (1.5 * range_ma)
    lower_kc = sma20 - (1.5 * range_ma)
    
    df['Squeeze_On'] = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    # Mom (LinReg Proxy: delta from mean)
    df['Sqz_Mom'] = (df['Close'] - sma20).rolling(20).mean() * 100

    # 3. Money Flow Matrix (RSI * Vol Ratio)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    vol_ratio = df['Volume'] / df['Volume'].rolling(20).mean()
    df['RVOL'] = vol_ratio
    
    # Normalized RSI (-50 to 50) * Vol Ratio
    df['MF_Matrix'] = ((df['RSI'] - 50) * vol_ratio).ewm(span=3).mean()

    # 4. God Mode Score (Confluence)
    # Score range -5 to +5
    score = np.zeros(len(df))
    # Trend (+2/-2)
    score += np.where(df['Apex_Trend'] == 1, 2, -2)
    # Momentum (+1/-1)
    score += np.sign(df['Sqz_Mom'])
    # Money Flow (+1/-1)
    score += np.sign(df['MF_Matrix'])
    # Price vs HMA (+1/-1)
    score += np.where(df['Close'] > df['HMA_55'], 1, -1)
    
    df['GM_Score'] = score
    
    return df

# --- QUANTUM PHYSICS ENGINE (From Pentagram) ---
def calc_quantum(df):
    """Calculates RQZO (Regime Quantum Zero Oscillator) and Vector Flux"""
    df = df.copy()
    
    # 1. RQZO (Synthetic Oscillator)
    win = 100
    mn = df["Close"].rolling(win).min()
    mx = df["Close"].rolling(win).max()
    norm = (df["Close"] - mn) / (mx - mn + 1e-10)
    delta = (norm - norm.shift(1)).abs().fillna(0.0)
    
    # Relativistic Gamma Proxy
    v = np.clip(delta, 0, 0.049) / 0.05
    gamma = 1 / np.sqrt(1 - (v ** 2) + 1e-10)
    tau = (np.arange(len(df)) % win) / gamma.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Sum of Sines approximation
    rq = np.zeros(len(df), dtype=float)
    for n in range(1, 6):
        rq += (n ** -0.5) * np.sin(tau * np.log(n))
    df['RQZO'] = rq * 10

    # 2. Vector Flux (Efficiency * Vol)
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    body = (df["Close"] - df["Open"]).abs()
    efficiency = (body / rng).fillna(0.0).ewm(span=14).mean()
    
    # Directional Flux
    raw_flux = np.sign(df["Close"] - df["Open"]) * efficiency * df['RVOL']
    df['Flux'] = raw_flux.ewm(span=5).mean()
    
    return df

# --- SMC ENGINE (Structure) ---
def calc_smc(df, lookback=5):
    """Identify Pivots, FVG, and Order Blocks"""
    df = df.copy()
    
    # Pivots
    df['PH'] = df['High'].rolling(lookback*2+1, center=True).max() == df['High']
    df['PL'] = df['Low'].rolling(lookback*2+1, center=True).min() == df['Low']
    
    # FVG
    # Bullish FVG: Low > High[2]
    df['FVG_Bull'] = (df['Low'] > df['High'].shift(2))
    df['FVG_Bear'] = (df['High'] < df['Low'].shift(2))
    
    # Python Lists for plotting
    structures = {'highs': [], 'lows': [], 'fvgs': [], 'obs': []}
    
    # Detect recent Swing Highs/Lows
    ph_idxs = df.index[df['PH']]
    pl_idxs = df.index[df['PL']]
    
    # Collect last 5 zones
    for i in ph_idxs[-10:]:
        structures['highs'].append({'idx': i, 'price': df.loc[i, 'High']})
    for i in pl_idxs[-10:]:
        structures['lows'].append({'idx': i, 'price': df.loc[i, 'Low']})
        
    return df, structures

# ==========================================
# 3. DATA LAYER
# ==========================================
@st.cache_data(ttl=300)
def get_data(ticker, interval="1h", period="1y"):
    try:
        # yfinance logic
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Flatten MultiIndex if exists
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Clean
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
            
        df = df.dropna()
        if len(df) < 50: return pd.DataFrame()
        
        # Apply Math Chain
        df = calc_god_mode(df)
        df = calc_quantum(df)
        
        return df
    except Exception as e:
        st.error(f"Data Feed Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    if "-" in ticker or "=" in ticker: return None # Skip crypto/forex fundies for now
    try:
        t = yf.Ticker(ticker)
        return t.info
    except: return None

# ==========================================
# 4. BROADCAST & AI
# ==========================================
def get_ai_analysis(df, ticker, api_key):
    if not api_key: return "⚠️ AI Key Missing"
    
    last = df.iloc[-1]
    prompt = f"""
    Analyze {ticker}. Price: {last['Close']:.2f}.
    God Mode Score: {last['GM_Score']:.0f}/5.
    Trend: {'Bull' if last['Apex_Trend']==1 else 'Bear'}.
    Flux: {last['Flux']:.3f}.
    Money Flow: {last['MF_Matrix']:.2f}.
    
    Provide a concise institutional market outlook. Use 3 emojis. Max 100 words.
    Style: Professional, Quant, Direct.
    """
    try:
        client = OpenAI(api_key=api_key)
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

def send_telegram(token, chat_id, msg):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg}
    try:
        requests.post(url, json=payload)
        return True
    except: return False

# ==========================================
# 5. UI CONSTRUCTION
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 📡 CONTROL DECK")
    
    # Ticker Selection
    ticker_mode = st.radio("Universe", ["Crypto", "Equities", "Forex/Macro", "Custom"])
    if ticker_mode == "Crypto":
        ticker = st.selectbox("Asset", ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "LINK-USD"])
    elif ticker_mode == "Equities":
        ticker = st.selectbox("Asset", ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "MSFT", "COIN", "MSTR"])
    elif ticker_mode == "Forex/Macro":
        ticker = st.selectbox("Asset", ["GC=F", "CL=F", "DX-Y.NYB", "EURUSD=X", "^TNX"])
    else:
        ticker = st.text_input("Symbol", "BTC-USD").upper()
        
    # Timeframe
    tf_map = {"15m": "60d", "1h": "730d", "4h": "730d", "1d": "2y", "1wk": "5y"}
    interval = st.selectbox("Interval", list(tf_map.keys()), index=2)
    period = tf_map[interval]
    
    st.markdown("---")
    st.markdown("## 🔐 KEYS")
    api_key = st.text_input("OpenAI Key", type="password")
    tg_token = st.text_input("TG Token", type="password")
    tg_chat = st.text_input("TG Chat ID")
    
    if st.button("🔄 SYSTEM REFRESH", type="primary"):
        st.cache_data.clear()
        st.rerun()

# --- MAIN APP ---
df = get_data(ticker, interval, period)

if df.empty:
    st.error(f"VOID DETECTED: Unable to fetch data for {ticker}")
    st.stop()

# Real-time Metrics
last = df.iloc[-1]
prev = df.iloc[-2]
chg = (last['Close'] - prev['Close']) / prev['Close'] * 100

st.markdown(f"<div class='title-glow'>{ticker} // {interval}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>PRICE: {last['Close']:.2f} | CHANGE: {chg:+.2f}% | ATR: {last['ATR_55']:.2f}</div>", unsafe_allow_html=True)

# Top Level Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("GOD MODE SCORE", f"{last['GM_Score']:.0f}/5", delta="BULL" if last['GM_Score']>0 else "BEAR")
c2.metric("FLUX VECTOR", f"{last['Flux']:.3f}", delta="INFLOW" if last['Flux']>0 else "OUTFLOW")
c3.metric("QUANTUM OSC", f"{last['RQZO']:.2f}", delta="HIGH ENERGY" if abs(last['RQZO'])>5 else "LOW ENERGY")
c4.metric("VOLATILITY", f"{last['RVOL']:.1f}x", "EXPANSION" if last['RVOL']>1.5 else "CONTRACTION")

# TABS
t1, t2, t3, t4, t5 = st.tabs(["👁️ NEXUS CHART", "🧠 QUANTUM PHYSICS", "🏛️ STRUCTURE", "🧬 DNA", "🤖 AI COUNCIL"])

def plot_config(height=600):
    return dict(
        template="plotly_dark",
        height=height,
        margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False
    )

# --- TAB 1: NEXUS CHART (God Mode) ---
with t1:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    
    # Price & Apex Cloud
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 255, 0, 0.05)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HMA_55'], line=dict(color='#ffd700', width=2), name="HMA Trend"), row=1, col=1)
    
    # Signals
    buys = df[df['GM_Score'] >= 4]
    sells = df[df['GM_Score'] <= -4]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00'), name="STRONG BUY"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000'), name="STRONG SELL"), row=1, col=1)

    # Squeeze
    cols = ['#00ff00' if x > 0 else '#ff0033' for x in df['Sqz_Mom']]
    fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=cols, name="Momentum"), row=2, col=1)
    
    # Money Flow
    fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy', line=dict(color='#00ffff'), name="Money Flow"), row=3, col=1)
    
    fig.update_layout(**plot_config(700))
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: QUANTUM PHYSICS ---
with t2:
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig_q.add_trace(go.Scatter(x=df.index, y=df['RQZO'], line=dict(color='#ff00ff', width=2), name="RQZO"), row=1, col=1)
        fig_q.add_hline(y=0, line_dash="dot", row=1, col=1)
        fig_q.add_trace(go.Bar(x=df.index, y=df['Flux'], marker_color=['#00ff00' if x>0 else '#ff0033' for x in df['Flux']], name="Vector Flux"), row=2, col=1)
        fig_q.update_layout(**plot_config(500))
        st.plotly_chart(fig_q, use_container_width=True)
    with col_q2:
        st.markdown("""
        <div class='diag-panel'>
            <div class='diag-header'>Quantum State</div>
            <div class='diag-value'>{}</div>
        </div>
        """.format("SUPER-POSITION" if abs(last['RQZO']) > 8 else "STABLE"), unsafe_allow_html=True)
        
        st.markdown("""
        <div class='diag-panel'>
            <div class='diag-header'>Flux Efficiency</div>
            <div class='diag-value'>{:.1f}%</div>
        </div>
        """.format(abs(last['Flux'])*100), unsafe_allow_html=True)

# --- TAB 3: STRUCTURE (SMC) ---
with t3:
    df_smc, structs = calc_smc(df)
    fig_smc = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    
    # Plot Recent Highs/Lows as lines
    for h in structs['highs']:
        fig_smc.add_shape(type="line", x0=h['idx'], y0=h['price'], x1=df.index[-1], y1=h['price'], line=dict(color="rgba(255,0,0,0.3)", dash="dot"))
    for l in structs['lows']:
        fig_smc.add_shape(type="line", x0=l['idx'], y0=l['price'], x1=df.index[-1], y1=l['price'], line=dict(color="rgba(0,255,0,0.3)", dash="dot"))
        
    # Highlight FVGs (Simple Visualization)
    fvgs_bull = df[df['FVG_Bull']]
    fig_smc.add_trace(go.Scatter(x=fvgs_bull.index, y=fvgs_bull['Low'], mode='markers', marker=dict(symbol='triangle-up', color='yellow', size=5), name="Bull FVG"))
    
    fig_smc.update_layout(**plot_config(600))
    st.plotly_chart(fig_smc, use_container_width=True)

# --- TAB 4: DNA (Seasonality) ---
with t4:
    if len(df) > 100:
        df['Hour'] = df.index.hour
        df['Ret'] = df['Close'].pct_change()
        hourly = df.groupby('Hour')['Ret'].mean() * 100
        
        fig_dna = px.bar(hourly, x=hourly.index, y=hourly.values, color=hourly.values, color_continuous_scale="RdYlGn")
        fig_dna.update_layout(title="Intraday Returns (Hourly)", **plot_config(400))
        st.plotly_chart(fig_dna, use_container_width=True)
    else:
        st.warning("Insufficient data for DNA analysis")

# --- TAB 5: AI & BROADCAST ---
with t5:
    c_ai, c_bc = st.columns(2)
    
    with c_ai:
        st.markdown("### 🤖 The Council")
        if st.button("Consult the Architect"):
            analysis = get_ai_analysis(df, ticker, api_key)
            st.info(analysis)
            
    with c_bc:
        st.markdown("### 📢 Broadcast")
        msg = st.text_area("Signal Message", f"🔥 {ticker} ({interval}) SIGNAL\nScore: {last['GM_Score']:.0f}/5\nPrice: {last['Close']:.2f}")
        if st.button("Send to Telegram"):
            if tg_token and tg_chat:
                ok = send_telegram(tg_token, tg_chat, msg)
                if ok: st.success("Sent!")
                else: st.error("Failed.")
            else:
                st.warning("Configure TG Keys")

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #333; font-size: 0.8rem;'>THE ARCHITECT // NEXUS TERMINAL v1.0 // {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
