import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from openai import OpenAI
import requests
import time as time_lib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Titan v8.1 Pro Terminal",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. UI THEME: HIGH CONTRAST TERMINAL
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #000000; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    section[data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #222; }
    
    div[data-testid="metric-container"] {
        background-color: #0a0a0a;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 10px;
    }
    label[data-testid="stMetricLabel"] { color: #888; font-size: 0.8rem !important; }
    div[data-testid="stMetricValue"] { color: #fff; font-size: 1.4rem !important; font-weight: 700; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid #222; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: transparent;
        border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; color: #666;
    }
    .stTabs [aria-selected="true"] { background-color: #0a0a0a; color: #00E676; border: 1px solid #333; border-bottom: none; }
    
    .console-log {
        font-family: 'Roboto Mono', monospace; font-size: 0.75rem; background: #080808;
        border: 1px solid #222; padding: 10px; border-radius: 4px; height: 200px; overflow-y: auto;
    }
    .log-line { border-bottom: 1px solid #151515; padding: 4px 0; display: flex; gap: 10px; }
    
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #080808; color: #ddd; border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. MATH ENGINE (FIXED SCALING)
# ==========================================
def rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def double_smooth(src, l1, l2):
    return src.ewm(span=l1).mean().ewm(span=l2).mean()

def calc_indicators(df, p):
    df = df.copy()
    
    # --- 1. APEX VECTOR ---
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    df["eff_raw"] = np.where(rng==0, 0, body/rng)
    df["efficiency"] = df["eff_raw"].ewm(span=p["apex_len"]).mean()
    
    vol_avg = df["volume"].rolling(p["apex_vol"]).mean()
    vol_fact = np.where(vol_avg==0, 1, df["volume"]/vol_avg)
    
    raw_vec = np.sign(df["close"]-df["open"]) * df["efficiency"] * vol_fact
    df["flux"] = raw_vec.ewm(span=p["apex_sm"]).mean()
    
    th_s = p["apex_th_s"]
    th_r = p["apex_th_r"]
    
    conditions = [
        (df["flux"] > th_s),
        (df["flux"] < -th_s),
        (df["flux"].abs() < th_r)
    ]
    df["state"] = np.select(conditions, [2, -2, 0], default=1)

    # --- 2. DARK TREND (FIXED INITIALIZATION) ---
    atr = rma(df["high"]-df["low"], p["trend_len"])
    hl2 = (df["high"]+df["low"])/2
    up = hl2 + (p["trend_mult"] * atr)
    dn = hl2 - (p["trend_mult"] * atr)
    
    # Initialize with NaNs to prevent 0-scaling issue
    trend = np.zeros(len(df))
    stop = np.full(len(df), np.nan)
    
    c = df["close"].values
    
    # Initialize first valid index
    stop[0] = dn.iloc[0]
    trend[0] = 1
    
    for i in range(1, len(df)):
        # Handle NaN in previous stop if data is young
        prev_stop = stop[i-1] if not np.isnan(stop[i-1]) else dn.iloc[i]
        
        curr_up = up.iloc[i] if (up.iloc[i] < prev_stop or c[i-1] > prev_stop) else prev_stop
        curr_dn = dn.iloc[i] if (dn.iloc[i] > prev_stop or c[i-1] < prev_stop) else prev_stop
        
        prev_t = trend[i-1]
        
        if prev_t == -1 and c[i] > curr_up:
            trend[i] = 1
            stop[i] = dn.iloc[i]
        elif prev_t == 1 and c[i] < curr_dn:
            trend[i] = -1
            stop[i] = up.iloc[i]
        else:
            trend[i] = prev_t
            if trend[i] == 1:
                stop[i] = max(prev_stop, dn.iloc[i])
            else:
                stop[i] = min(prev_stop, up.iloc[i])

    df["trend"] = trend
    df["stop"] = stop
    
    # --- 3. MATRIX ---
    chg = df["close"].diff()
    gain = chg.clip(lower=0)
    loss = -chg.clip(upper=0)
    rs = rma(gain, 14) / (rma(loss, 14) + 1e-10)
    rsi = 100 - (100/(1+rs))
    
    mfi_vol = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
    df["mfi"] = ((rsi - 50) * mfi_vol).ewm(span=3).mean()
    
    hw_src = df["close"].diff()
    df["hw"] = 100 * (double_smooth(hw_src, 25, 13) / (double_smooth(hw_src.abs(), 25, 13) + 1e-10)) / 2
    
    df["matrix_sig"] = np.sign(df["mfi"]) + np.sign(df["hw"])

    return df

# ==========================================
# 4. DATA FETCHING
# ==========================================
def get_exchange(name):
    if name == 'Binance': return ccxt.binance()
    if name == 'Bybit': return ccxt.bybit()
    if name == 'Coinbase': return ccxt.coinbase()
    if name == 'OKX': return ccxt.okx()
    return ccxt.kraken()

@st.cache_data(ttl=5)
def fetch_candles(exch, sym, tf, lim):
    try:
        ex = get_exchange(exch)
        ex.enableRateLimit = True
        ohlcv = ex.fetch_ohlcv(sym, tf, limit=lim)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# 5. SIDEBAR: THE CONTROL CENTER
# ==========================================
def sidebar_settings():
    st.sidebar.markdown("### 💠 TITAN v8.1")
    
    with st.sidebar.expander("🌍 Exchange Connection", expanded=True):
        exch = st.selectbox("Exchange", ["Kraken", "Binance", "Bybit", "Coinbase", "OKX"])
        sym = st.text_input("Symbol", "BTC/USD")
        tf = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
        limit = st.slider("Candles", 100, 1000, 500)
        auto_ref = st.checkbox("Auto-Refresh (60s)", False)

    with st.sidebar.expander("⚡ Apex Config", expanded=False):
        apex_len = st.number_input("Efficiency Len", 5, 50, 14)
        apex_vol = st.number_input("Volume Norm", 10, 100, 55)
        apex_sm = st.number_input("Flux Smooth", 1, 20, 5)
        apex_th_s = st.slider("Super Thresh", 0.1, 1.0, 0.60)
        apex_th_r = st.slider("Resist Thresh", 0.0, 0.5, 0.30)

    with st.sidebar.expander("🌊 Trend Config", expanded=False):
        trend_len = st.number_input("ATR Length", 5, 100, 55)
        trend_mult = st.number_input("Factor", 1.0, 10.0, 4.0, step=0.1)

    with st.sidebar.expander("🤖 API & Alerts", expanded=False):
        api_key = st.text_input("OpenAI Key", type="password")
        tg_tok = st.text_input("TG Token", type="password")
        tg_id = st.text_input("TG Chat ID")

    if st.sidebar.button("RELOAD DATA", type="primary", use_container_width=True):
        fetch_candles.clear()
        st.rerun()

    if auto_ref:
        time_lib.sleep(60)
        st.rerun()

    return {
        "exch": exch, "sym": sym, "tf": tf, "limit": limit,
        "apex_len": apex_len, "apex_vol": apex_vol, "apex_sm": apex_sm,
        "apex_th_s": apex_th_s, "apex_th_r": apex_th_r,
        "trend_len": trend_len, "trend_mult": trend_mult,
        "api": api_key, "tg_t": tg_tok, "tg_c": tg_id
    }

cfg = sidebar_settings()

# ==========================================
# 6. MAIN LOGIC
# ==========================================
df = fetch_candles(cfg["exch"], cfg["sym"], cfg["tf"], cfg["limit"])

if df.empty:
    st.error(f"Failed to fetch data for {cfg['sym']} on {cfg['exch']}. Check symbol format (e.g. BTC/USD vs BTC/USDT).")
    st.stop()

# Process Indicators
df = calc_indicators(df, cfg)
last = df.iloc[-1]
prev = df.iloc[-2]

# --- Signals ---
signals = []
if last["trend"] != prev["trend"]:
    signals.append(f"TREND FLIP: {('BULL' if last['trend']==1 else 'BEAR')}")
if abs(last["state"]) == 2 and abs(prev["state"]) != 2:
    signals.append(f"APEX STATE: {('SUPER BULL' if last['state']==2 else 'SUPER BEAR')}")
if last["matrix_sig"] != prev["matrix_sig"] and abs(last["matrix_sig"]) == 2:
    signals.append(f"MATRIX: {('BUY' if last['matrix_sig']>0 else 'SELL')}")

# Broadcast
if signals and cfg["tg_t"] and cfg["tg_c"]:
    for sig in signals:
        try:
            requests.post(f"https://api.telegram.org/bot{cfg['tg_t']}/sendMessage", json={"chat_id": cfg['tg_c'], "text": f"🚨 {cfg['sym']}: {sig}"})
        except: pass

# ==========================================
# 7. DASHBOARD UI
# ==========================================
st.title(f"{cfg['sym']} / {cfg['tf']}")

# KPI Row
k1, k2, k3, k4 = st.columns(4)
col_trend = "#00E676" if last["trend"] == 1 else "#FF1744"
col_flux = "#00E676" if last["flux"] > 0 else "#FF1744"

k1.metric("Close Price", f"{last['close']:.2f}")
k2.metric("Trend State", ("BULLISH" if last["trend"]==1 else "BEARISH"), f"Stop: {last['stop']:.2f}")
k3.metric("Apex Flux", f"{last['flux']:.2f}", f"Eff: {last['efficiency']:.2f}")
k4.metric("Matrix", int(last["matrix_sig"]), ("Buy" if last["matrix_sig"] > 0 else "Sell"))

# Charting
t1, t2, t3, t4 = st.tabs(["📊 Main Chart", "⚡ Apex Scanner", "🌊 Matrix Deep-Dive", "🤖 AI Report"])

def dark_chart():
    layout = go.Layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        margin=dict(l=0, r=50, t=10, b=0),
        height=600,
        xaxis=dict(showgrid=False, color="#444"),
        yaxis=dict(showgrid=True, gridcolor="#222", color="#444")
    )
    return layout

# TAB 1: MAIN
with t1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.02)
    
    # Candle
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 increasing_line_color="#00E676", decreasing_line_color="#FF1744", name="Price"), row=1, col=1)
    
    # Trend Stop
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["stop"], mode="lines", line=dict(color=col_trend, width=2), name="Stop"), row=1, col=1)
    
    # Flux
    colors = ["#00E676" if s==2 else ("#FF1744" if s==-2 else "#333") for s in df["state"]]
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=colors, name="Flux"), row=2, col=1)
    fig.add_hline(y=cfg["apex_th_s"], line_color="#444", line_dash="dot", row=2, col=1)
    fig.add_hline(y=-cfg["apex_th_s"], line_color="#444", line_dash="dot", row=2, col=1)
    
    fig.update_layout(dark_chart())
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: APEX DETAILS
with t2:
    c1, c2 = st.columns([3, 1])
    with c1:
        # Efficiency vs Flux Map
        fig_map = go.Figure()
        fig_map.add_trace(go.Scatter(
            x=df["efficiency"], y=df["flux"], mode="markers",
            marker=dict(color=df["flux"], colorscale="RdYlGn", size=8, showscale=False)
        ))
        fig_map.add_trace(go.Scatter(
            x=[last["efficiency"]], y=[last["flux"]], mode="markers+text", text=["NOW"],
            textposition="top center", marker=dict(color="white", size=15, line=dict(color="black", width=2))
        ))
        fig_map.update_layout(dark_chart())
        fig_map.update_layout(title="Regime Map (Efficiency vs Flux)", xaxis_title="Efficiency", yaxis_title="Flux", height=450)
        st.plotly_chart(fig_map, use_container_width=True)
    
    with c2:
        st.markdown("#### 🔬 State Analysis")
        flux = last["flux"]
        if flux > cfg["apex_th_s"]:
            st.success("**SUPER BULL**\n\nMarket is moving with high efficiency and volume support. Resistance is minimal.")
        elif flux < -cfg["apex_th_s"]:
            st.error("**SUPER BEAR**\n\nMarket is crashing with high efficiency. Support is failing.")
        elif abs(flux) < cfg["apex_th_r"]:
            st.warning("**RESISTIVE (CHOP)**\n\nFlux is near zero. Price is stuck in noise. Avoid trend systems.")
        else:
            st.info("**HIGH HEAT**\n\nVolatility is high, but true directional efficiency is not yet confirmed.")

# TAB 3: MATRIX DETAILS
with t3:
    fig_mx = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig_mx.add_trace(go.Scatter(x=df["timestamp"], y=df["mfi"], fill="tozeroy", line_color="#D500F9", name="MFI"), row=1, col=1)
    fig_mx.add_trace(go.Bar(x=df["timestamp"], y=df["hw"], marker_color="#00E5FF", name="HyperWave"), row=2, col=1)
    fig_mx.update_layout(dark_chart())
    st.plotly_chart(fig_mx, use_container_width=True)

# TAB 4: AI REPORT
with t4:
    if not cfg["api"]:
        st.warning("⚠️ API Key missing in Sidebar > API & Alerts")
    else:
        st.subheader("🤖 GPT-4o Synthesis")
        prompt = f"""
        Market: {cfg['sym']} ({cfg['tf']})
        Price: {last['close']}
        
        1. Apex Flux: {last['flux']:.3f} (Threshold {cfg['apex_th_s']})
        2. Trend Direction: {('UP' if last['trend']==1 else 'DOWN')}
        3. Matrix Score: {last['matrix_sig']}
        
        Task: Write a trader's log entry. Be concise. Define the Bias, the Setup, and the Risk.
        """
        if st.button("GENERATE INTEL"):
            with st.spinner("Analyzing Physics..."):
                try:
                    client = OpenAI(api_key=cfg["api"])
                    resp = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": prompt}])
                    st.markdown(f"**Intel:**\n{resp.choices[0].message.content}")
                except Exception as e:
                    st.error(f"AI Error: {e}")

# Footer Log
st.markdown("---")
st.markdown("### 📟 System Log")
log_box = st.container()
with log_box:
    st.markdown(f"""
    <div class="console-log">
        <div class="log-line"><span style="color:#00E676">SYSTEM</span> Titan v8.1 Initialized...</div>
        <div class="log-line"><span style="color:#00E676">DATA</span> Connected to {cfg['exch']} ({cfg['limit']} candles)</div>
        <div class="log-line"><span style="color:#888">INFO</span> Current Close: {last['close']}</div>
        {''.join([f'<div class="log-line"><span style="color:#FFD600">ALERT</span> {s}</div>' for s in signals])}
    </div>
    """, unsafe_allow_html=True)
