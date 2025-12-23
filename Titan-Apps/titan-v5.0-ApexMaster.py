import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from openai import OpenAI
import requests

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Titan v5.2 Omni-Terminal",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. UI THEME
# ==========================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600;700&display=swap');

:root{
  --bg:#000;
  --panel:#090909;
  --panel2:#0e0e0e;
  --border:#222;
  --text:#e6e6e6;
  --bull:#00E676;
  --bear:#FF1744;
  --cyan:#00E5FF;
  --vio:#D500F9;
}

.stApp { background: var(--bg); color: var(--text); font-family: 'Roboto Mono', monospace; }
.block-container { padding-top: 1rem; }

/* Custom Metric Box */
.metric-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    text-align: center;
}
.metric-lbl { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
.metric-val { font-size: 1.4rem; font-weight: 700; color: var(--text); }
.metric-sub { font-size: 0.75rem; color: #666; margin-top: 4px; }

.c-bull { color: var(--bull) !important; }
.c-bear { color: var(--bear) !important; }
.c-cyan { color: var(--cyan) !important; }
.c-vio  { color: var(--vio) !important; }

/* Analysis Box */
.analysis-box {
    border-left: 3px solid var(--cyan);
    background: #050505;
    padding: 12px;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #ccc;
    margin-top: 10px;
}

/* Sidebar Log */
.log-entry {
    font-size: 0.8rem;
    border-bottom: 1px solid #222;
    padding: 6px 0;
    color: #aaa;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. MATH ENGINE
# ==========================================
def rma(series, length):
    return series.ewm(alpha=1 / length, adjust=False).mean()

def double_smooth(src, long_len, short_len):
    return src.ewm(span=long_len, adjust=False).mean().ewm(span=short_len, adjust=False).mean()

def hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    def wma(s, l):
        weights = np.arange(1, l + 1)
        return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    wma_half = wma(series, half_length)
    wma_full = wma(series, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_length)

# --- APEX VECTOR v4.1 ---
def calc_apex_vector(df, p):
    df = df.copy()
    # Efficiency
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    eff = (body / (rng + 1e-10)).ewm(span=int(p["len_vec"])).mean()
    df["efficiency"] = eff

    # Flux
    vol_fact = df["volume"] / (df["volume"].rolling(int(p["vol_norm"])).mean() + 1e-10)
    raw = np.sign(df["close"] - df["open"]) * eff * vol_fact
    df["flux"] = raw.ewm(span=int(p["len_sm"])).mean()

    # State
    th_s = p["eff_super"] * p["strictness"]
    th_r = p["eff_resist"] * p["strictness"]
    cond = [(df["flux"] > th_s), (df["flux"] < -th_s), (df["flux"].abs() < th_r)]
    df["apex_state"] = np.select(cond, [2, -2, 0], default=1)
    
    # Divergence
    src = df["flux"]
    df["pivot_low"] = (src.shift(1) < src.shift(2)) & (src.shift(1) < src)
    df["pivot_high"] = (src.shift(1) > src.shift(2)) & (src.shift(1) > src)
    
    df["div_bull"] = df["pivot_low"] & (df["close"] < df["close"].shift(5)) & (df["flux"] > df["flux"].shift(5))
    df["div_bear"] = df["pivot_high"] & (df["close"] > df["close"].shift(5)) & (df["flux"] < df["flux"].shift(5))
    
    return df

# --- DARK TREND ---
def calc_dark_trend(df, p):
    df = df.copy()
    atr = rma(df["high"] - df["low"], int(p["len_main"]))
    
    hl2 = (df["high"] + df["low"]) / 2
    up = hl2 + (p["st_mult"] * atr)
    dn = hl2 - (p["st_mult"] * atr)
    
    trend = np.zeros(len(df))
    t_up = np.zeros(len(df))
    t_dn = np.zeros(len(df))
    
    t_up[0] = up.iloc[0]
    t_dn[0] = dn.iloc[0]
    
    close = df["close"].values
    up_val = up.values
    dn_val = dn.values
    
    for i in range(1, len(df)):
        if up_val[i] < t_up[i-1] or close[i-1] > t_up[i-1]:
            t_up[i] = up_val[i]
        else:
            t_up[i] = t_up[i-1]
            
        if dn_val[i] > t_dn[i-1] or close[i-1] < t_dn[i-1]:
            t_dn[i] = dn_val[i]
        else:
            t_dn[i] = t_dn[i-1]
            
        prev = trend[i-1]
        if prev == -1 and close[i] > t_up[i-1]:
            trend[i] = 1
        elif prev == 1 and close[i] < t_dn[i-1]:
            trend[i] = -1
        else:
            trend[i] = prev if prev != 0 else 1

    df["trend"] = trend
    df["stop_line"] = np.where(trend==1, t_dn, t_up)
    df["chop"] = 100 * np.log10(atr.rolling(14).sum() / (df["high"].rolling(14).max() - df["low"].rolling(14).min())) / np.log10(14)
    return df

# --- MATRIX ---
def calc_matrix(df, p):
    df = df.copy()
    chg = df["close"].diff()
    rsi = 100 - (100 / (1 + (rma(chg.clip(lower=0), 14) / rma(-chg.clip(upper=0), 14))))
    df["mfi"] = ((rsi - 50) * (df["volume"]/df["volume"].rolling(20).mean())).ewm(span=3).mean()
    
    src = df["close"].diff()
    hw = 100 * (double_smooth(src, 25, 13) / (double_smooth(src.abs(), 25, 13) + 1e-10))
    df["hyperwave"] = hw / 2 
    
    df["matrix_score"] = np.sign(df["mfi"]) + np.sign(df["hyperwave"])
    return df

# --- QUANTUM ---
def calc_quantum(df, p):
    df = df.copy()
    src = df["close"]
    norm = (src - src.rolling(100).min()) / (src.rolling(100).max() - src.rolling(100).min() + 1e-10)
    df["entropy"] = df["close"].pct_change().rolling(20).std() * 100 
    df["rqzo"] = (norm - 0.5) * 10 * (1 - df["entropy"]/10)
    return df

# ==========================================
# 4. DATA & SETTINGS
# ==========================================
@st.cache_resource
def get_exchange():
    return ccxt.kraken({"enableRateLimit": True})

@st.cache_data(ttl=30)
def get_data(sym, tf, lim):
    return pd.DataFrame(get_exchange().fetch_ohlcv(sym, tf, limit=lim), columns=["timestamp", "open", "high", "low", "close", "volume"])

def init_settings():
    defaults = {
        "symbol": "BTC/USD", "timeframe": "15m", "limit": 500,
        "len_main": 55, "st_mult": 4.0, "eff_super": 0.6, "eff_resist": 0.3,
        "webhook_url": "", "vol_norm": 55, "len_vec": 14, "len_sm": 5, "strictness": 1.0
    }
    if "cfg" not in st.session_state:
        st.session_state.cfg = defaults
    else:
        for k,v in defaults.items():
            if k not in st.session_state.cfg: st.session_state.cfg[k] = v

init_settings()
cfg = st.session_state.cfg

# ==========================================
# 5. SIDEBAR & SETTINGS
# ==========================================
with st.sidebar:
    st.header("Titan Config")
    with st.form("settings"):
        cfg["symbol"] = st.text_input("Symbol", cfg["symbol"])
        c1, c2 = st.columns(2)
        cfg["timeframe"] = c1.selectbox("Timeframe", ["1m","5m","15m","1h","4h"], index=2)
        cfg["limit"] = c2.slider("Lookback", 200, 1000, cfg["limit"])
        
        st.divider()
        st.caption("Strategy Params")
        cfg["eff_super"] = st.slider("Apex Super Thresh", 0.1, 1.0, cfg["eff_super"])
        cfg["st_mult"] = st.number_input("Dark Trend Mult", 1.0, 10.0, cfg["st_mult"])
        
        st.divider()
        st.caption("Broadcasting")
        cfg["webhook_url"] = st.text_input("Discord Webhook URL", cfg["webhook_url"], type="password")
        
        if st.form_submit_button("Update System"):
            st.toast("System Updated", icon="🔄")
            get_data.clear()

# ==========================================
# 6. LOGIC & BROADCAST ENGINE
# ==========================================
try:
    df = get_data(cfg["symbol"], cfg["timeframe"], cfg["limit"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
except Exception as e:
    st.error(f"Data Connection Error: {e}")
    st.stop()

# Calculate All Indicators
params = cfg 
try:
    df = calc_apex_vector(df, params)
    df = calc_dark_trend(df, params)
    df = calc_matrix(df, params)
    df = calc_quantum(df, params)
except Exception as e:
    st.error(f"Calculation Error: {e}")
    st.stop()

last = df.iloc[-1]
prev = df.iloc[-2]

# --- BROADCASTER ---
signals = []
if last["trend"] == 1 and prev["trend"] == -1:
    signals.append(f"🌊 DARK TREND: BULLISH FLIP ({last['close']})")
elif last["trend"] == -1 and prev["trend"] == 1:
    signals.append(f"🌊 DARK TREND: BEARISH FLIP ({last['close']})")

if last["apex_state"] == 2 and prev["apex_state"] != 2:
    signals.append("⚡ APEX: SUPER BULL START")
if last["apex_state"] == -2 and prev["apex_state"] != -2:
    signals.append("⚡ APEX: SUPER BEAR START")

if last["div_bull"]: signals.append("💎 APEX: BULLISH DIVERGENCE")
if last["div_bear"]: signals.append("💎 APEX: BEARISH DIVERGENCE")

# Sidebar Log Display
with st.sidebar:
    st.divider()
    st.subheader("🔔 Signal Log")
    if signals:
        st.markdown(f"**Latest ({datetime.now().strftime('%H:%M')}):**")
        for s in signals:
            st.markdown(f"<div class='log-entry' style='color:#00E676'>• {s}</div>", unsafe_allow_html=True)
            if cfg["webhook_url"]:
                try: pass # requests.post(cfg["webhook_url"], json={"content": s})
                except: pass
    else:
        st.caption("No signals on current candle.")

# ==========================================
# 7. VISUALIZATION TABS
# ==========================================
st.title(f"⚡ Titan v5.2 // {cfg['symbol']}")

t_main, t_apex, t_dark, t_mat, t_quant, t_ai = st.tabs([
    "Overview", "Apex Vector", "Dark Trend", "Matrix", "Quantum", "Analyst"
])

# --- TAB 1: OVERVIEW ---
with t_main:
    # Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    
    trend_c = "c-bull" if last["trend"]==1 else "c-bear"
    flux_c = "c-bull" if last["flux"]>0 else "c-bear"
    
    c1.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Master Trend</div>
        <div class="metric-val {trend_c}">{("BULLISH" if last["trend"]==1 else "BEARISH")}</div>
        <div class="metric-sub">Stop: {last['stop_line']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Apex Vector</div>
        <div class="metric-val {flux_c}">{last['flux']:.2f}</div>
        <div class="metric-sub">Eff: {last['efficiency']*100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    mat_score = int(last["matrix_score"])
    mat_txt = "STRONG BUY" if mat_score==2 else ("STRONG SELL" if mat_score==-2 else "NEUTRAL")
    mat_c = "c-bull" if mat_score==2 else ("c-bear" if mat_score==-2 else "c-cyan")
    
    c3.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Matrix Score</div>
        <div class="metric-val {mat_c}">{mat_txt}</div>
        <div class="metric-sub">Raw: {mat_score}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Active Signals</div>
        <div class="metric-val" style="font-size:1.1rem">{len(signals)}</div>
        <div class="metric-sub">Last Scan: {datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Main Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], mode='lines', line=dict(color='gray', dash='dot'), name="Stop"))
    
    buys = df[(df["trend"]==1) & (df["trend"].shift(1)==-1)]
    sells = df[(df["trend"]==-1) & (df["trend"].shift(1)==1)]
    fig.add_trace(go.Scatter(x=buys["timestamp"], y=buys["low"], mode="markers", marker=dict(symbol="triangle-up", color="#00E676", size=12), name="Trend Buy"))
    fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["high"], mode="markers", marker=dict(symbol="triangle-down", color="#FF1744", size=12), name="Trend Sell"))
    
    fig.update_layout(height=550, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: APEX VECTOR DETAILED ---
with t_apex:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        # Dual Pane: Price + Flux
        fig_av = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_av.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]), row=1, col=1)
        
        colors = ["#00E676" if s == 2 else ("#FF1744" if s == -2 else ("#546E7A" if s == 0 else "#FFD600")) for s in df["apex_state"]]
        fig_av.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=colors, name="Flux"), row=2, col=1)
        
        th = cfg["eff_super"]
        fig_av.add_hline(y=th, line_dash="dot", line_color="gray", row=2, col=1)
        fig_av.add_hline(y=-th, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig_av.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_av, use_container_width=True)
        
    with col_r:
        st.markdown("#### 🔍 Regime Map")
        # Regime Map: Efficiency vs Flux
        fig_map = go.Figure()
        fig_map.add_trace(go.Scatter(
            x=df["efficiency"].tail(100), 
            y=df["flux"].tail(100),
            mode='markers',
            marker=dict(
                size=8,
                color=df["flux"].tail(100),
                colorscale='RdYlGn',
                showscale=False
            )
        ))
        # Add 'current' marker
        fig_map.add_trace(go.Scatter(
            x=[last["efficiency"]], y=[last["flux"]],
            mode='markers+text', text=["NOW"], textposition="top center",
            marker=dict(size=14, color='white', line=dict(width=2, color='black'))
        ))
        fig_map.update_layout(
            title="Eff vs Flux (Last 100)",
            xaxis_title="Efficiency", yaxis_title="Flux",
            height=300, template="plotly_dark",
            margin=dict(l=0,r=0,t=30,b=0),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Rule Based Text
        flux_val = last["flux"]
        state_str = "neutral"
        if flux_val > cfg["eff_super"]: state_str = "super_bull"
        elif flux_val < -cfg["eff_super"]: state_str = "super_bear"
        elif abs(flux_val) < cfg["eff_resist"]: state_str = "resistive"
        
        analysis_text = ""
        if state_str == "super_bull":
            analysis_text = "Market is in **Superconductor State (Bullish)**. Efficiency is high, meaning price is moving with low resistance. Volume supports the move."
        elif state_str == "super_bear":
            analysis_text = "Market is in **Superconductor State (Bearish)**. Sellers are dominating with high efficiency. Expect continuation down."
        elif state_str == "resistive":
            analysis_text = "Market is **Resistive (Choppy)**. Flux is too low to sustain a trend. Avoid trading or use mean-reversion tactics."
        else:
            analysis_text = "Market is in **High Heat**. Volatility is present but direction is not fully efficient yet. Caution advised."
            
        st.markdown(f"""
        <div class="analysis-box">
            {analysis_text}
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: DARK TREND DETAILED ---
with t_dark:
    fig_dk = go.Figure()
    fig_dk.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], line=dict(color="#00E5FF", width=2), name="Trend Line"))
    
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] + (df["stop_line"]*0.005), line=dict(width=0), showlegend=False))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] - (df["stop_line"]*0.005), fill="tonexty", 
                                fillcolor=("rgba(0, 230, 118, 0.15)" if last["trend"]==1 else "rgba(255, 23, 68, 0.15)"), 
                                line=dict(width=0), name="Cloud"))
    
    fig_dk.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_dk, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        chop = last["chop"]
        chop_state = "TRENDING" if chop < 50 else "CHOPPY/RANGING"
        chop_color = "c-bull" if chop < 50 else "c-bear"
        
        st.markdown(f"""
        <div class="analysis-box">
            <b>Chop Index:</b> {chop:.1f} <span class="{chop_color}">({chop_state})</span><br>
            Values below 38 indicate strong trends. Values above 61 indicate intense consolidation.
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: MATRIX DETAILED ---
with t_mat:
    fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig_m.add_trace(go.Scatter(x=df["timestamp"], y=df["mfi"], fill="tozeroy", line=dict(color="#D500F9"), name="Money Flow"), row=1, col=1)
    fig_m.add_trace(go.Bar(x=df["timestamp"], y=df["hyperwave"], marker_color="#00E5FF", name="HyperWave"), row=2, col=1)
    fig_m.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_m, use_container_width=True)
    
    st.info("Matrix combines **Money Flow** (Volume+RSI) and **HyperWave** (Double Smoothed Momentum). When both align, the signal is strongest.")

# --- TAB 5: QUANTUM ---
with t_quant:
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], line=dict(color="white"), name="RQZO"))
    
    chaos_zone = df[df["entropy"] > 2.0]
    fig_q.add_trace(go.Scatter(x=chaos_zone["timestamp"], y=chaos_zone["rqzo"], mode="markers", marker=dict(color="red", size=4), name="High Entropy"))
    
    fig_q.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_q, use_container_width=True)
    
    st.markdown("Red dots indicate **High Entropy (Chaos)** areas where price prediction is statistically unreliable.")

# --- TAB 6: AI ANALYST ---
with t_ai:
    st.subheader("🤖 GPT-4o Quant Synthesis")
    
    prompt_txt = f"""
    ASSET: {cfg['symbol']} ({cfg['timeframe']})
    
    1. APEX VECTOR:
    - Flux: {last['flux']:.3f} (Threshold {cfg['eff_super']})
    - Efficiency: {last['efficiency']:.2f}
    - Divergence: {('BULL' if last['div_bull'] else ('BEAR' if last['div_bear'] else 'NONE'))}
    
    2. DARK TREND:
    - Direction: {('UP' if last['trend']==1 else 'DOWN')}
    - Chop Index: {last['chop']:.1f}
    
    3. MATRIX SCORE: {last['matrix_score']}
    
    TASK: Provide a 3-sentence executive summary on BIAS, ENTRY, and RISK.
    """
    
    st.code(prompt_txt, language="text")
    
    ai_key = st.text_input("OpenAI API Key (Optional)", type="password")
    if st.button("Generate Report"):
        if not ai_key:
            st.error("Please provide an API Key.")
        else:
            with st.spinner("Analyzing market physics..."):
                try:
                    client = OpenAI(api_key=ai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"user", "content": prompt_txt}]
                    )
                    st.success(resp.choices[0].message.content)
                except Exception as e:
                    st.error(str(e))  --cyan:#00E5FF;
  --vio:#D500F9;
}

.stApp { background: var(--bg); color: var(--text); font-family: 'Roboto Mono', monospace; }
.block-container { padding-top: 1rem; }

/* Custom Metric Box */
.metric-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    text-align: center;
}
.metric-lbl { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
.metric-val { font-size: 1.4rem; font-weight: 700; color: var(--text); }
.metric-sub { font-size: 0.75rem; color: #666; margin-top: 4px; }

.c-bull { color: var(--bull) !important; }
.c-bear { color: var(--bear) !important; }
.c-cyan { color: var(--cyan) !important; }
.c-vio  { color: var(--vio) !important; }

/* Analysis Box */
.analysis-box {
    border-left: 3px solid var(--cyan);
    background: #050505;
    padding: 12px;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #ccc;
    margin-top: 10px;
}

/* Sidebar Log */
.log-entry {
    font-size: 0.8rem;
    border-bottom: 1px solid #222;
    padding: 6px 0;
    color: #aaa;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. MATH ENGINE
# ==========================================
def rma(series, length):
    return series.ewm(alpha=1 / length, adjust=False).mean()

def double_smooth(src, long_len, short_len):
    return src.ewm(span=long_len, adjust=False).mean().ewm(span=short_len, adjust=False).mean()

def hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    def wma(s, l):
        weights = np.arange(1, l + 1)
        return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    wma_half = wma(series, half_length)
    wma_full = wma(series, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_length)

# --- APEX VECTOR v4.1 ---
def calc_apex_vector(df, p):
    df = df.copy()
    # Efficiency
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    eff = (body / (rng + 1e-10)).ewm(span=int(p["len_vec"])).mean()
    df["efficiency"] = eff

    # Flux
    vol_fact = df["volume"] / (df["volume"].rolling(int(p["vol_norm"])).mean() + 1e-10)
    raw = np.sign(df["close"] - df["open"]) * eff * vol_fact
    df["flux"] = raw.ewm(span=int(p["len_sm"])).mean()

    # State
    th_s = p["eff_super"] * p["strictness"]
    th_r = p["eff_resist"] * p["strictness"]
    cond = [(df["flux"] > th_s), (df["flux"] < -th_s), (df["flux"].abs() < th_r)]
    df["apex_state"] = np.select(cond, [2, -2, 0], default=1)
    
    # Divergence
    src = df["flux"]
    df["pivot_low"] = (src.shift(1) < src.shift(2)) & (src.shift(1) < src)
    df["pivot_high"] = (src.shift(1) > src.shift(2)) & (src.shift(1) > src)
    
    df["div_bull"] = df["pivot_low"] & (df["close"] < df["close"].shift(5)) & (df["flux"] > df["flux"].shift(5))
    df["div_bear"] = df["pivot_high"] & (df["close"] > df["close"].shift(5)) & (df["flux"] < df["flux"].shift(5))
    
    return df

# --- DARK TREND ---
def calc_dark_trend(df, p):
    df = df.copy()
    atr = rma(df["high"] - df["low"], int(p["len_main"]))
    
    hl2 = (df["high"] + df["low"]) / 2
    up = hl2 + (p["st_mult"] * atr)
    dn = hl2 - (p["st_mult"] * atr)
    
    trend = np.zeros(len(df))
    t_up = np.zeros(len(df))
    t_dn = np.zeros(len(df))
    
    t_up[0] = up.iloc[0]
    t_dn[0] = dn.iloc[0]
    
    close = df["close"].values
    up_val = up.values
    dn_val = dn.values
    
    for i in range(1, len(df)):
        if up_val[i] < t_up[i-1] or close[i-1] > t_up[i-1]:
            t_up[i] = up_val[i]
        else:
            t_up[i] = t_up[i-1]
            
        if dn_val[i] > t_dn[i-1] or close[i-1] < t_dn[i-1]:
            t_dn[i] = dn_val[i]
        else:
            t_dn[i] = t_dn[i-1]
            
        prev = trend[i-1]
        if prev == -1 and close[i] > t_up[i-1]:
            trend[i] = 1
        elif prev == 1 and close[i] < t_dn[i-1]:
            trend[i] = -1
        else:
            trend[i] = prev if prev != 0 else 1

    df["trend"] = trend
    df["stop_line"] = np.where(trend==1, t_dn, t_up)
    df["chop"] = 100 * np.log10(atr.rolling(14).sum() / (df["high"].rolling(14).max() - df["low"].rolling(14).min())) / np.log10(14)
    return df

# --- MATRIX ---
def calc_matrix(df, p):
    df = df.copy()
    chg = df["close"].diff()
    rsi = 100 - (100 / (1 + (rma(chg.clip(lower=0), 14) / rma(-chg.clip(upper=0), 14))))
    df["mfi"] = ((rsi - 50) * (df["volume"]/df["volume"].rolling(20).mean())).ewm(span=3).mean()
    
    src = df["close"].diff()
    hw = 100 * (double_smooth(src, 25, 13) / (double_smooth(src.abs(), 25, 13) + 1e-10))
    df["hyperwave"] = hw / 2 
    
    df["matrix_score"] = np.sign(df["mfi"]) + np.sign(df["hyperwave"])
    return df

# --- QUANTUM ---
def calc_quantum(df, p):
    df = df.copy()
    src = df["close"]
    norm = (src - src.rolling(100).min()) / (src.rolling(100).max() - src.rolling(100).min() + 1e-10)
    df["entropy"] = df["close"].pct_change().rolling(20).std() * 100 
    df["rqzo"] = (norm - 0.5) * 10 * (1 - df["entropy"]/10)
    return df

# ==========================================
# 4. DATA & SETTINGS
# ==========================================
@st.cache_resource
def get_exchange():
    return ccxt.kraken({"enableRateLimit": True})

@st.cache_data(ttl=30)
def get_data(sym, tf, lim):
    return pd.DataFrame(get_exchange().fetch_ohlcv(sym, tf, limit=lim), columns=["timestamp", "open", "high", "low", "close", "volume"])

def init_settings():
    defaults = {
        "symbol": "BTC/USD", "timeframe": "15m", "limit": 500,
        "len_main": 55, "st_mult": 4.0, "eff_super": 0.6, "eff_resist": 0.3,
        "webhook_url": "", "vol_norm": 55, "len_vec": 14, "len_sm": 5, "strictness": 1.0
    }
    if "cfg" not in st.session_state:
        st.session_state.cfg = defaults
    else:
        for k,v in defaults.items():
            if k not in st.session_state.cfg: st.session_state.cfg[k] = v

init_settings()
cfg = st.session_state.cfg

# ==========================================
# 5. SIDEBAR & SETTINGS
# ==========================================
with st.sidebar:
    st.header("Titan Config")
    with st.form("settings"):
        cfg["symbol"] = st.text_input("Symbol", cfg["symbol"])
        c1, c2 = st.columns(2)
        cfg["timeframe"] = c1.selectbox("Timeframe", ["1m","5m","15m","1h","4h"], index=2)
        cfg["limit"] = c2.slider("Lookback", 200, 1000, cfg["limit"])
        
        st.divider()
        st.caption("Strategy Params")
        cfg["eff_super"] = st.slider("Apex Super Thresh", 0.1, 1.0, cfg["eff_super"])
        cfg["st_mult"] = st.number_input("Dark Trend Mult", 1.0, 10.0, cfg["st_mult"])
        
        st.divider()
        st.caption("Broadcasting")
        cfg["webhook_url"] = st.text_input("Discord Webhook URL", cfg["webhook_url"], type="password")
        
        if st.form_submit_button("Update System"):
            st.toast("System Updated", icon="🔄")
            get_data.clear()

# ==========================================
# 6. LOGIC & BROADCAST ENGINE
# ==========================================
try:
    df = get_data(cfg["symbol"], cfg["timeframe"], cfg["limit"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
except Exception as e:
    st.error(f"Data Connection Error: {e}")
    st.stop()

# Calculate All Indicators
params = cfg 
try:
    df = calc_apex_vector(df, params)
    df = calc_dark_trend(df, params)
    df = calc_matrix(df, params)
    df = calc_quantum(df, params)
except Exception as e:
    st.error(f"Calculation Error: {e}")
    st.stop()

last = df.iloc[-1]
prev = df.iloc[-2]

# --- BROADCASTER ---
signals = []
if last["trend"] == 1 and prev["trend"] == -1:
    signals.append(f"🌊 DARK TREND: BULLISH FLIP ({last['close']})")
elif last["trend"] == -1 and prev["trend"] == 1:
    signals.append(f"🌊 DARK TREND: BEARISH FLIP ({last['close']})")

if last["apex_state"] == 2 and prev["apex_state"] != 2:
    signals.append("⚡ APEX: SUPER BULL START")
if last["apex_state"] == -2 and prev["apex_state"] != -2:
    signals.append("⚡ APEX: SUPER BEAR START")

if last["div_bull"]: signals.append("💎 APEX: BULLISH DIVERGENCE")
if last["div_bear"]: signals.append("💎 APEX: BEARISH DIVERGENCE")

# Sidebar Log Display
with st.sidebar:
    st.divider()
    st.subheader("🔔 Signal Log")
    if signals:
        st.markdown(f"**Latest ({datetime.now().strftime('%H:%M')}):**")
        for s in signals:
            st.markdown(f"<div class='log-entry' style='color:#00E676'>• {s}</div>", unsafe_allow_html=True)
            if cfg["webhook_url"]:
                try: pass # requests.post(cfg["webhook_url"], json={"content": s})
                except: pass
    else:
        st.caption("No signals on current candle.")

# ==========================================
# 7. VISUALIZATION TABS
# ==========================================
st.title(f"⚡ Titan v5.2 // {cfg['symbol']}")

t_main, t_apex, t_dark, t_mat, t_quant, t_ai = st.tabs([
    "Overview", "Apex Vector", "Dark Trend", "Matrix", "Quantum", "Analyst"
])

# --- TAB 1: OVERVIEW ---
with t_main:
    # Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    
    trend_c = "c-bull" if last["trend"]==1 else "c-bear"
    flux_c = "c-bull" if last["flux"]>0 else "c-bear"
    
    c1.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Master Trend</div>
        <div class="metric-val {trend_c}">{("BULLISH" if last["trend"]==1 else "BEARISH")}</div>
        <div class="metric-sub">Stop: {last['stop_line']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Apex Vector</div>
        <div class="metric-val {flux_c}">{last['flux']:.2f}</div>
        <div class="metric-sub">Eff: {last['efficiency']*100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    mat_score = int(last["matrix_score"])
    mat_txt = "STRONG BUY" if mat_score==2 else ("STRONG SELL" if mat_score==-2 else "NEUTRAL")
    mat_c = "c-bull" if mat_score==2 else ("c-bear" if mat_score==-2 else "c-cyan")
    
    c3.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Matrix Score</div>
        <div class="metric-val {mat_c}">{mat_txt}</div>
        <div class="metric-sub">Raw: {mat_score}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Active Signals</div>
        <div class="metric-val" style="font-size:1.1rem">{len(signals)}</div>
        <div class="metric-sub">Last Scan: {datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Main Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], mode='lines', line=dict(color='gray', dash='dot'), name="Stop"))
    
    buys = df[(df["trend"]==1) & (df["trend"].shift(1)==-1)]
    sells = df[(df["trend"]==-1) & (df["trend"].shift(1)==1)]
    fig.add_trace(go.Scatter(x=buys["timestamp"], y=buys["low"], mode="markers", marker=dict(symbol="triangle-up", color="#00E676", size=12), name="Trend Buy"))
    fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["high"], mode="markers", marker=dict(symbol="triangle-down", color="#FF1744", size=12), name="Trend Sell"))
    
    fig.update_layout(height=550, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: APEX VECTOR DETAILED ---
with t_apex:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        # Dual Pane: Price + Flux
        fig_av = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_av.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]), row=1, col=1)
        
        colors = ["#00E676" if s == 2 else ("#FF1744" if s == -2 else ("#546E7A" if s == 0 else "#FFD600")) for s in df["apex_state"]]
        fig_av.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=colors, name="Flux"), row=2, col=1)
        
        th = cfg["eff_super"]
        fig_av.add_hline(y=th, line_dash="dot", line_color="gray", row=2, col=1)
        fig_av.add_hline(y=-th, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig_av.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_av, use_container_width=True)
        
    with col_r:
        st.markdown("#### 🔍 Regime Map")
        # Regime Map: Efficiency vs Flux
        fig_map = go.Figure()
        fig_map.add_trace(go.Scatter(
            x=df["efficiency"].tail(100), 
            y=df["flux"].tail(100),
            mode='markers',
            marker=dict(
                size=8,
                color=df["flux"].tail(100),
                colorscale='RdYlGn',
                showscale=False
            )
        ))
        # Add 'current' marker
        fig_map.add_trace(go.Scatter(
            x=[last["efficiency"]], y=[last["flux"]],
            mode='markers+text', text=["NOW"], textposition="top center",
            marker=dict(size=14, color='white', line=dict(width=2, color='black'))
        ))
        fig_map.update_layout(
            title="Eff vs Flux (Last 100)",
            xaxis_title="Efficiency", yaxis_title="Flux",
            height=300, template="plotly_dark",
            margin=dict(l=0,r=0,t=30,b=0),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Rule Based Text
        flux_val = last["flux"]
        state_str = "neutral"
        if flux_val > cfg["eff_super"]: state_str = "super_bull"
        elif flux_val < -cfg["eff_super"]: state_str = "super_bear"
        elif abs(flux_val) < cfg["eff_resist"]: state_str = "resistive"
        
        analysis_text = ""
        if state_str == "super_bull":
            analysis_text = "Market is in **Superconductor State (Bullish)**. Efficiency is high, meaning price is moving with low resistance. Volume supports the move."
        elif state_str == "super_bear":
            analysis_text = "Market is in **Superconductor State (Bearish)**. Sellers are dominating with high efficiency. Expect continuation down."
        elif state_str == "resistive":
            analysis_text = "Market is **Resistive (Choppy)**. Flux is too low to sustain a trend. Avoid trading or use mean-reversion tactics."
        else:
            analysis_text = "Market is in **High Heat**. Volatility is present but direction is not fully efficient yet. Caution advised."
            
        st.markdown(f"""
        <div class="analysis-box">
            {analysis_text}
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: DARK TREND DETAILED ---
with t_dark:
    fig_dk = go.Figure()
    fig_dk.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], line=dict(color="#00E5FF", width=2), name="Trend Line"))
    
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] + (df["stop_line"]*0.005), line=dict(width=0), showlegend=False))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] - (df["stop_line"]*0.005), fill="tonexty", 
                                fillcolor=("rgba(0, 230, 118, 0.15)" if last["trend"]==1 else "rgba(255, 23, 68, 0.15)"), 
                                line=dict(width=0), name="Cloud"))
    
    fig_dk.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_dk, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        chop = last["chop"]
        chop_state = "TRENDING" if chop < 50 else "CHOPPY/RANGING"
        chop_color = "c-bull" if chop < 50 else "c-bear"
        
        st.markdown(f"""
        <div class="analysis-box">
            <b>Chop Index:</b> {chop:.1f} <span class="{chop_color}">({chop_state})</span><br>
            Values below 38 indicate strong trends. Values above 61 indicate intense consolidation.
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: MATRIX DETAILED ---
with t_mat:
    fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig_m.add_trace(go.Scatter(x=df["timestamp"], y=df["mfi"], fill="tozeroy", line=dict(color="#D500F9"), name="Money Flow"), row=1, col=1)
    fig_m.add_trace(go.Bar(x=df["timestamp"], y=df["hyperwave"], marker_color="#00E5FF", name="HyperWave"), row=2, col=1)
    fig_m.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_m, use_container_width=True)
    
    st.info("Matrix combines **Money Flow** (Volume+RSI) and **HyperWave** (Double Smoothed Momentum). When both align, the signal is strongest.")

# --- TAB 5: QUANTUM ---
with t_quant:
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], line=dict(color="white"), name="RQZO"))
    
    chaos_zone = df[df["entropy"] > 2.0]
    fig_q.add_trace(go.Scatter(x=chaos_zone["timestamp"], y=chaos_zone["rqzo"], mode="markers", marker=dict(color="red", size=4), name="High Entropy"))
    
    fig_q.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_q, use_container_width=True)
    
    st.markdown("Red dots indicate **High Entropy (Chaos)** areas where price prediction is statistically unreliable.")

# --- TAB 6: AI ANALYST ---
with t_ai:
    st.subheader("🤖 GPT-4o Quant Synthesis")
    
    prompt_txt = f"""
    ASSET: {cfg['symbol']} ({cfg['timeframe']})
    
    1. APEX VECTOR:
    - Flux: {last['flux']:.3f} (Threshold {cfg['eff_super']})
    - Efficiency: {last['efficiency']:.2f}
    - Divergence: {('BULL' if last['div_bull'] else ('BEAR' if last['div_bear'] else 'NONE'))}
    
    2. DARK TREND:
    - Direction: {('UP' if last['trend']==1 else 'DOWN')}
    - Chop Index: {last['chop']:.1f}
    
    3. MATRIX SCORE: {last['matrix_score']}
    
    TASK: Provide a 3-sentence executive summary on BIAS, ENTRY, and RISK.
    """
    
    st.code(prompt_txt, language="text")
    
    ai_key = st.text_input("OpenAI API Key (Optional)", type="password")
    if st.button("Generate Report"):
        if not ai_key:
            st.error("Please provide an API Key.")
        else:
            with st.spinner("Analyzing market physics..."):
                try:
                    client = OpenAI(api_key=ai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"user", "content": prompt_txt}]
                    )
                    st.success(resp.choices[0].message.content)
                except Exception as e:
                    st.error(str(e))  --cyan:#00E5FF;
  --vio:#D500F9;
}

.stApp { background: var(--bg); color: var(--text); font-family: 'Roboto Mono', monospace; }
.block-container { padding-top: 1rem; }

/* Custom Metric Box */
.metric-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    text-align: center;
}
.metric-lbl { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
.metric-val { font-size: 1.4rem; font-weight: 700; color: var(--text); }
.metric-sub { font-size: 0.75rem; color: #666; margin-top: 4px; }

.c-bull { color: var(--bull) !important; }
.c-bear { color: var(--bear) !important; }
.c-cyan { color: var(--cyan) !important; }
.c-vio  { color: var(--vio) !important; }

/* Analysis Box */
.analysis-box {
    border-left: 3px solid var(--cyan);
    background: #050505;
    padding: 12px;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #ccc;
    margin-top: 10px;
}

/* Sidebar Log */
.log-entry {
    font-size: 0.8rem;
    border-bottom: 1px solid #222;
    padding: 6px 0;
    color: #aaa;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. MATH ENGINE
# ==========================================
def rma(series, length):
    return series.ewm(alpha=1 / length, adjust=False).mean()

def double_smooth(src, long_len, short_len):
    return src.ewm(span=long_len, adjust=False).mean().ewm(span=short_len, adjust=False).mean()

def hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    def wma(s, l):
        weights = np.arange(1, l + 1)
        return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    wma_half = wma(series, half_length)
    wma_full = wma(series, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_length)

# --- APEX VECTOR v4.1 ---
def calc_apex_vector(df, p):
    df = df.copy()
    # Efficiency
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    eff = (body / (rng + 1e-10)).ewm(span=int(p["len_vec"])).mean()
    df["efficiency"] = eff

    # Flux
    vol_fact = df["volume"] / (df["volume"].rolling(int(p["vol_norm"])).mean() + 1e-10)
    raw = np.sign(df["close"] - df["open"]) * eff * vol_fact
    df["flux"] = raw.ewm(span=int(p["len_sm"])).mean()

    # State
    th_s = p["eff_super"] * p["strictness"]
    th_r = p["eff_resist"] * p["strictness"]
    cond = [(df["flux"] > th_s), (df["flux"] < -th_s), (df["flux"].abs() < th_r)]
    df["apex_state"] = np.select(cond, [2, -2, 0], default=1)
    
    # Divergence
    src = df["flux"]
    df["pivot_low"] = (src.shift(1) < src.shift(2)) & (src.shift(1) < src)
    df["pivot_high"] = (src.shift(1) > src.shift(2)) & (src.shift(1) > src)
    
    df["div_bull"] = df["pivot_low"] & (df["close"] < df["close"].shift(5)) & (df["flux"] > df["flux"].shift(5))
    df["div_bear"] = df["pivot_high"] & (df["close"] > df["close"].shift(5)) & (df["flux"] < df["flux"].shift(5))
    
    return df

# --- DARK TREND ---
def calc_dark_trend(df, p):
    df = df.copy()
    atr = rma(df["high"] - df["low"], int(p["len_main"]))
    
    hl2 = (df["high"] + df["low"]) / 2
    up = hl2 + (p["st_mult"] * atr)
    dn = hl2 - (p["st_mult"] * atr)
    
    trend = np.zeros(len(df))
    t_up = np.zeros(len(df))
    t_dn = np.zeros(len(df))
    
    t_up[0] = up.iloc[0]
    t_dn[0] = dn.iloc[0]
    
    close = df["close"].values
    up_val = up.values
    dn_val = dn.values
    
    for i in range(1, len(df)):
        if up_val[i] < t_up[i-1] or close[i-1] > t_up[i-1]:
            t_up[i] = up_val[i]
        else:
            t_up[i] = t_up[i-1]
            
        if dn_val[i] > t_dn[i-1] or close[i-1] < t_dn[i-1]:
            t_dn[i] = dn_val[i]
        else:
            t_dn[i] = t_dn[i-1]
            
        prev = trend[i-1]
        if prev == -1 and close[i] > t_up[i-1]:
            trend[i] = 1
        elif prev == 1 and close[i] < t_dn[i-1]:
            trend[i] = -1
        else:
            trend[i] = prev if prev != 0 else 1

    df["trend"] = trend
    df["stop_line"] = np.where(trend==1, t_dn, t_up)
    df["chop"] = 100 * np.log10(atr.rolling(14).sum() / (df["high"].rolling(14).max() - df["low"].rolling(14).min())) / np.log10(14)
    return df

# --- MATRIX ---
def calc_matrix(df, p):
    df = df.copy()
    chg = df["close"].diff()
    rsi = 100 - (100 / (1 + (rma(chg.clip(lower=0), 14) / rma(-chg.clip(upper=0), 14))))
    df["mfi"] = ((rsi - 50) * (df["volume"]/df["volume"].rolling(20).mean())).ewm(span=3).mean()
    
    src = df["close"].diff()
    hw = 100 * (double_smooth(src, 25, 13) / (double_smooth(src.abs(), 25, 13) + 1e-10))
    df["hyperwave"] = hw / 2 
    
    df["matrix_score"] = np.sign(df["mfi"]) + np.sign(df["hyperwave"])
    return df

# --- QUANTUM ---
def calc_quantum(df, p):
    df = df.copy()
    src = df["close"]
    norm = (src - src.rolling(100).min()) / (src.rolling(100).max() - src.rolling(100).min() + 1e-10)
    df["entropy"] = df["close"].pct_change().rolling(20).std() * 100 
    df["rqzo"] = (norm - 0.5) * 10 * (1 - df["entropy"]/10)
    return df

# ==========================================
# 4. DATA & SETTINGS
# ==========================================
@st.cache_resource
def get_exchange():
    return ccxt.kraken({"enableRateLimit": True})

@st.cache_data(ttl=30)
def get_data(sym, tf, lim):
    return pd.DataFrame(get_exchange().fetch_ohlcv(sym, tf, limit=lim), columns=["timestamp", "open", "high", "low", "close", "volume"])

def init_settings():
    defaults = {
        "symbol": "BTC/USD", "timeframe": "15m", "limit": 500,
        "len_main": 55, "st_mult": 4.0, "eff_super": 0.6, "eff_resist": 0.3,
        "webhook_url": "", "vol_norm": 55, "len_vec": 14, "len_sm": 5, "strictness": 1.0
    }
    if "cfg" not in st.session_state:
        st.session_state.cfg = defaults
    else:
        for k,v in defaults.items():
            if k not in st.session_state.cfg: st.session_state.cfg[k] = v

init_settings()
cfg = st.session_state.cfg

# ==========================================
# 5. SIDEBAR & SETTINGS
# ==========================================
with st.sidebar:
    st.header("Titan Config")
    with st.form("settings"):
        cfg["symbol"] = st.text_input("Symbol", cfg["symbol"])
        c1, c2 = st.columns(2)
        cfg["timeframe"] = c1.selectbox("Timeframe", ["1m","5m","15m","1h","4h"], index=2)
        cfg["limit"] = c2.slider("Lookback", 200, 1000, cfg["limit"])
        
        st.divider()
        st.caption("Strategy Params")
        cfg["eff_super"] = st.slider("Apex Super Thresh", 0.1, 1.0, cfg["eff_super"])
        cfg["st_mult"] = st.number_input("Dark Trend Mult", 1.0, 10.0, cfg["st_mult"])
        
        st.divider()
        st.caption("Broadcasting")
        cfg["webhook_url"] = st.text_input("Discord Webhook URL", cfg["webhook_url"], type="password")
        
        if st.form_submit_button("Update System"):
            st.toast("System Updated", icon="🔄")
            get_data.clear()

# ==========================================
# 6. LOGIC & BROADCAST ENGINE
# ==========================================
try:
    df = get_data(cfg["symbol"], cfg["timeframe"], cfg["limit"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
except Exception as e:
    st.error(f"Data Connection Error: {e}")
    st.stop()

# Calculate All Indicators
params = cfg 
try:
    df = calc_apex_vector(df, params)
    df = calc_dark_trend(df, params)
    df = calc_matrix(df, params)
    df = calc_quantum(df, params)
except Exception as e:
    st.error(f"Calculation Error: {e}")
    st.stop()

last = df.iloc[-1]
prev = df.iloc[-2]

# --- BROADCASTER ---
signals = []
if last["trend"] == 1 and prev["trend"] == -1:
    signals.append(f"🌊 DARK TREND: BULLISH FLIP ({last['close']})")
elif last["trend"] == -1 and prev["trend"] == 1:
    signals.append(f"🌊 DARK TREND: BEARISH FLIP ({last['close']})")

if last["apex_state"] == 2 and prev["apex_state"] != 2:
    signals.append("⚡ APEX: SUPER BULL START")
if last["apex_state"] == -2 and prev["apex_state"] != -2:
    signals.append("⚡ APEX: SUPER BEAR START")

if last["div_bull"]: signals.append("💎 APEX: BULLISH DIVERGENCE")
if last["div_bear"]: signals.append("💎 APEX: BEARISH DIVERGENCE")

# Sidebar Log Display
with st.sidebar:
    st.divider()
    st.subheader("🔔 Signal Log")
    if signals:
        st.markdown(f"**Latest ({datetime.now().strftime('%H:%M')}):**")
        for s in signals:
            st.markdown(f"<div class='log-entry' style='color:#00E676'>• {s}</div>", unsafe_allow_html=True)
            if cfg["webhook_url"]:
                try: pass # requests.post(cfg["webhook_url"], json={"content": s})
                except: pass
    else:
        st.caption("No signals on current candle.")

# ==========================================
# 7. VISUALIZATION TABS
# ==========================================
st.title(f"⚡ Titan v5.2 // {cfg['symbol']}")

t_main, t_apex, t_dark, t_mat, t_quant, t_ai = st.tabs([
    "Overview", "Apex Vector", "Dark Trend", "Matrix", "Quantum", "Analyst"
])

# --- TAB 1: OVERVIEW ---
with t_main:
    # Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    
    trend_c = "c-bull" if last["trend"]==1 else "c-bear"
    flux_c = "c-bull" if last["flux"]>0 else "c-bear"
    
    c1.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Master Trend</div>
        <div class="metric-val {trend_c}">{("BULLISH" if last["trend"]==1 else "BEARISH")}</div>
        <div class="metric-sub">Stop: {last['stop_line']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Apex Vector</div>
        <div class="metric-val {flux_c}">{last['flux']:.2f}</div>
        <div class="metric-sub">Eff: {last['efficiency']*100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    mat_score = int(last["matrix_score"])
    mat_txt = "STRONG BUY" if mat_score==2 else ("STRONG SELL" if mat_score==-2 else "NEUTRAL")
    mat_c = "c-bull" if mat_score==2 else ("c-bear" if mat_score==-2 else "c-cyan")
    
    c3.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Matrix Score</div>
        <div class="metric-val {mat_c}">{mat_txt}</div>
        <div class="metric-sub">Raw: {mat_score}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Active Signals</div>
        <div class="metric-val" style="font-size:1.1rem">{len(signals)}</div>
        <div class="metric-sub">Last Scan: {datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Main Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], mode='lines', line=dict(color='gray', dash='dot'), name="Stop"))
    
    buys = df[(df["trend"]==1) & (df["trend"].shift(1)==-1)]
    sells = df[(df["trend"]==-1) & (df["trend"].shift(1)==1)]
    fig.add_trace(go.Scatter(x=buys["timestamp"], y=buys["low"], mode="markers", marker=dict(symbol="triangle-up", color="#00E676", size=12), name="Trend Buy"))
    fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["high"], mode="markers", marker=dict(symbol="triangle-down", color="#FF1744", size=12), name="Trend Sell"))
    
    fig.update_layout(height=550, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: APEX VECTOR DETAILED ---
with t_apex:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        # Dual Pane: Price + Flux
        fig_av = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_av.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]), row=1, col=1)
        
        colors = ["#00E676" if s == 2 else ("#FF1744" if s == -2 else ("#546E7A" if s == 0 else "#FFD600")) for s in df["apex_state"]]
        fig_av.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=colors, name="Flux"), row=2, col=1)
        
        th = cfg["eff_super"]
        fig_av.add_hline(y=th, line_dash="dot", line_color="gray", row=2, col=1)
        fig_av.add_hline(y=-th, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig_av.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_av, use_container_width=True)
        
    with col_r:
        st.markdown("#### 🔍 Regime Map")
        # Regime Map: Efficiency vs Flux
        fig_map = go.Figure()
        fig_map.add_trace(go.Scatter(
            x=df["efficiency"].tail(100), 
            y=df["flux"].tail(100),
            mode='markers',
            marker=dict(
                size=8,
                color=df["flux"].tail(100),
                colorscale='RdYlGn',
                showscale=False
            )
        ))
        # Add 'current' marker
        fig_map.add_trace(go.Scatter(
            x=[last["efficiency"]], y=[last["flux"]],
            mode='markers+text', text=["NOW"], textposition="top center",
            marker=dict(size=14, color='white', line=dict(width=2, color='black'))
        ))
        fig_map.update_layout(
            title="Eff vs Flux (Last 100)",
            xaxis_title="Efficiency", yaxis_title="Flux",
            height=300, template="plotly_dark",
            margin=dict(l=0,r=0,t=30,b=0),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Rule Based Text
        flux_val = last["flux"]
        state_str = "neutral"
        if flux_val > cfg["eff_super"]: state_str = "super_bull"
        elif flux_val < -cfg["eff_super"]: state_str = "super_bear"
        elif abs(flux_val) < cfg["eff_resist"]: state_str = "resistive"
        
        analysis_text = ""
        if state_str == "super_bull":
            analysis_text = "Market is in **Superconductor State (Bullish)**. Efficiency is high, meaning price is moving with low resistance. Volume supports the move."
        elif state_str == "super_bear":
            analysis_text = "Market is in **Superconductor State (Bearish)**. Sellers are dominating with high efficiency. Expect continuation down."
        elif state_str == "resistive":
            analysis_text = "Market is **Resistive (Choppy)**. Flux is too low to sustain a trend. Avoid trading or use mean-reversion tactics."
        else:
            analysis_text = "Market is in **High Heat**. Volatility is present but direction is not fully efficient yet. Caution advised."
            
        st.markdown(f"""
        <div class="analysis-box">
            {analysis_text}
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: DARK TREND DETAILED ---
with t_dark:
    fig_dk = go.Figure()
    fig_dk.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], line=dict(color="#00E5FF", width=2), name="Trend Line"))
    
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] + (df["stop_line"]*0.005), line=dict(width=0), showlegend=False))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] - (df["stop_line"]*0.005), fill="tonexty", 
                                fillcolor=("rgba(0, 230, 118, 0.15)" if last["trend"]==1 else "rgba(255, 23, 68, 0.15)"), 
                                line=dict(width=0), name="Cloud"))
    
    fig_dk.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_dk, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        chop = last["chop"]
        chop_state = "TRENDING" if chop < 50 else "CHOPPY/RANGING"
        chop_color = "c-bull" if chop < 50 else "c-bear"
        
        st.markdown(f"""
        <div class="analysis-box">
            <b>Chop Index:</b> {chop:.1f} <span class="{chop_color}">({chop_state})</span><br>
            Values below 38 indicate strong trends. Values above 61 indicate intense consolidation.
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: MATRIX DETAILED ---
with t_mat:
    fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig_m.add_trace(go.Scatter(x=df["timestamp"], y=df["mfi"], fill="tozeroy", line=dict(color="#D500F9"), name="Money Flow"), row=1, col=1)
    fig_m.add_trace(go.Bar(x=df["timestamp"], y=df["hyperwave"], marker_color="#00E5FF", name="HyperWave"), row=2, col=1)
    fig_m.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_m, use_container_width=True)
    
    st.info("Matrix combines **Money Flow** (Volume+RSI) and **HyperWave** (Double Smoothed Momentum). When both align, the signal is strongest.")

# --- TAB 5: QUANTUM ---
with t_quant:
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], line=dict(color="white"), name="RQZO"))
    
    chaos_zone = df[df["entropy"] > 2.0]
    fig_q.add_trace(go.Scatter(x=chaos_zone["timestamp"], y=chaos_zone["rqzo"], mode="markers", marker=dict(color="red", size=4), name="High Entropy"))
    
    fig_q.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_q, use_container_width=True)
    
    st.markdown("Red dots indicate **High Entropy (Chaos)** areas where price prediction is statistically unreliable.")

# --- TAB 6: AI ANALYST ---
with t_ai:
    st.subheader("🤖 GPT-4o Quant Synthesis")
    
    prompt_txt = f"""
    ASSET: {cfg['symbol']} ({cfg['timeframe']})
    
    1. APEX VECTOR:
    - Flux: {last['flux']:.3f} (Threshold {cfg['eff_super']})
    - Efficiency: {last['efficiency']:.2f}
    - Divergence: {('BULL' if last['div_bull'] else ('BEAR' if last['div_bear'] else 'NONE'))}
    
    2. DARK TREND:
    - Direction: {('UP' if last['trend']==1 else 'DOWN')}
    - Chop Index: {last['chop']:.1f}
    
    3. MATRIX SCORE: {last['matrix_score']}
    
    TASK: Provide a 3-sentence executive summary on BIAS, ENTRY, and RISK.
    """
    
    st.code(prompt_txt, language="text")
    
    ai_key = st.text_input("OpenAI API Key (Optional)", type="password")
    if st.button("Generate Report"):
        if not ai_key:
            st.error("Please provide an API Key.")
        else:
            with st.spinner("Analyzing market physics..."):
                try:
                    client = OpenAI(api_key=ai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"user", "content": prompt_txt}]
                    )
                    st.success(resp.choices[0].message.content)
                except Exception as e:
                    st.error(str(e))  --cyan:#00E5FF;
  --vio:#D500F9;
}

.stApp { background: var(--bg); color: var(--text); font-family: 'Roboto Mono', monospace; }
.block-container { padding-top: 1rem; }

/* Custom Metric Box */
.metric-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
}
.metric-lbl { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.metric-val { font-size: 1.1rem; font-weight: 700; color: var(--text); }
.metric-sub { font-size: 0.7rem; color: #666; }

.c-bull { color: var(--bull) !important; }
.c-bear { color: var(--bear) !important; }
.c-cyan { color: var(--cyan) !important; }
.c-vio  { color: var(--vio) !important; }

/* Analysis Box */
.analysis-box {
    border-left: 3px solid var(--cyan);
    background: #050505;
    padding: 12px;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #ccc;
    margin-top: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. MATH ENGINE
# ==========================================
def rma(series, length):
    return series.ewm(alpha=1 / length, adjust=False).mean()

def double_smooth(src, long_len, short_len):
    return src.ewm(span=long_len, adjust=False).mean().ewm(span=short_len, adjust=False).mean()

def hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    def wma(s, l):
        weights = np.arange(1, l + 1)
        return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    wma_half = wma(series, half_length)
    wma_full = wma(series, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_length)

# --- APEX VECTOR v4.1 ---
def calc_apex_vector(df, p):
    df = df.copy()
    # Efficiency
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    eff = (body / (rng + 1e-10)).ewm(span=int(p["len_vec"])).mean()
    df["efficiency"] = eff

    # Flux
    vol_fact = df["volume"] / (df["volume"].rolling(int(p["vol_norm"])).mean() + 1e-10)
    raw = np.sign(df["close"] - df["open"]) * eff * vol_fact
    df["flux"] = raw.ewm(span=int(p["len_sm"])).mean()

    # State
    th_s = p["eff_super"] * p["strictness"]
    th_r = p["eff_resist"] * p["strictness"]
    cond = [(df["flux"] > th_s), (df["flux"] < -th_s), (df["flux"].abs() < th_r)]
    df["apex_state"] = np.select(cond, [2, -2, 0], default=1)
    
    # Divergence (Simplified Vectorized)
    src = df["flux"]
    df["pivot_low"] = (src.shift(1) < src.shift(2)) & (src.shift(1) < src)
    df["pivot_high"] = (src.shift(1) > src.shift(2)) & (src.shift(1) > src)
    
    # Very basic Divergence hook (Lookback 5)
    df["div_bull"] = df["pivot_low"] & (df["close"] < df["close"].shift(5)) & (df["flux"] > df["flux"].shift(5))
    df["div_bear"] = df["pivot_high"] & (df["close"] > df["close"].shift(5)) & (df["flux"] < df["flux"].shift(5))
    
    return df

# --- DARK TREND ---
def calc_dark_trend(df, p):
    df = df.copy()
    atr = rma(df["high"] - df["low"], int(p["len_main"]))
    
    # SuperTrend-like Logic
    hl2 = (df["high"] + df["low"]) / 2
    up = hl2 + (p["st_mult"] * atr)
    dn = hl2 - (p["st_mult"] * atr)
    
    # Trend Loop
    trend = np.zeros(len(df))
    t_up = np.zeros(len(df))
    t_dn = np.zeros(len(df))
    
    # Init
    t_up[0] = up.iloc[0]
    t_dn[0] = dn.iloc[0]
    
    close = df["close"].values
    up_val = up.values
    dn_val = dn.values
    
    for i in range(1, len(df)):
        # Upper band
        if up_val[i] < t_up[i-1] or close[i-1] > t_up[i-1]:
            t_up[i] = up_val[i]
        else:
            t_up[i] = t_up[i-1]
            
        # Lower band
        if dn_val[i] > t_dn[i-1] or close[i-1] < t_dn[i-1]:
            t_dn[i] = dn_val[i]
        else:
            t_dn[i] = t_dn[i-1]
            
        # Trend Switch
        prev = trend[i-1]
        if prev == -1 and close[i] > t_up[i-1]:
            trend[i] = 1
        elif prev == 1 and close[i] < t_dn[i-1]:
            trend[i] = -1
        else:
            trend[i] = prev if prev != 0 else 1

    df["trend"] = trend
    df["stop_line"] = np.where(trend==1, t_dn, t_up)
    
    # Chop Index
    df["chop"] = 100 * np.log10(atr.rolling(14).sum() / (df["high"].rolling(14).max() - df["low"].rolling(14).min())) / np.log10(14)
    return df

# --- MATRIX ---
def calc_matrix(df, p):
    df = df.copy()
    # Money Flow
    chg = df["close"].diff()
    rsi = 100 - (100 / (1 + (rma(chg.clip(lower=0), 14) / rma(-chg.clip(upper=0), 14))))
    df["mfi"] = ((rsi - 50) * (df["volume"]/df["volume"].rolling(20).mean())).ewm(span=3).mean()
    
    # HyperWave
    src = df["close"].diff()
    hw = 100 * (double_smooth(src, 25, 13) / (double_smooth(src.abs(), 25, 13) + 1e-10))
    df["hyperwave"] = hw / 2 # Scale
    
    df["matrix_score"] = np.sign(df["mfi"]) + np.sign(df["hyperwave"])
    return df

# --- QUANTUM ---
def calc_quantum(df, p):
    df = df.copy()
    # Simplified RQZO logic for speed
    src = df["close"]
    norm = (src - src.rolling(100).min()) / (src.rolling(100).max() - src.rolling(100).min() + 1e-10)
    
    # Entropy (Market Disorder)
    df["entropy"] = df["close"].pct_change().rolling(20).std() * 100 # Rough proxy for disorder
    
    # RQZO Oscillator
    df["rqzo"] = (norm - 0.5) * 10 * (1 - df["entropy"]/10) # Modulation
    return df

# ==========================================
# 4. DATA & SETTINGS
# ==========================================
@st.cache_resource
def get_exchange():
    return ccxt.kraken({"enableRateLimit": True})

@st.cache_data(ttl=30)
def get_data(sym, tf, lim):
    return pd.DataFrame(get_exchange().fetch_ohlcv(sym, tf, limit=lim), columns=["timestamp", "open", "high", "low", "close", "volume"])

def init_settings():
    defaults = {
        "symbol": "BTC/USD", 
        "timeframe": "15m", 
        "limit": 500,
        "len_main": 55, 
        "st_mult": 4.0, 
        "eff_super": 0.6, 
        "eff_resist": 0.3,
        "webhook_url": "",
        "vol_norm": 55,
        "len_vec": 14,
        "len_sm": 5,
        "strictness": 1.0
    }
    if "cfg" not in st.session_state:
        st.session_state.cfg = defaults
    else:
        for k,v in defaults.items():
            if k not in st.session_state.cfg: 
                st.session_state.cfg[k] = v

init_settings()
cfg = st.session_state.cfg

# ==========================================
# 5. SIDEBAR & SETTINGS
# ==========================================
with st.sidebar:
    st.header("Titan Config")
    with st.form("settings"):
        cfg["symbol"] = st.text_input("Symbol", cfg["symbol"])
        c1, c2 = st.columns(2)
        cfg["timeframe"] = c1.selectbox("Timeframe", ["1m","5m","15m","1h","4h"], index=2)
        cfg["limit"] = c2.slider("Lookback", 200, 1000, cfg["limit"])
        
        st.divider()
        st.caption("Strategy Params")
        cfg["eff_super"] = st.slider("Apex Super Thresh", 0.1, 1.0, cfg["eff_super"])
        cfg["st_mult"] = st.number_input("Dark Trend Mult", 1.0, 10.0, cfg["st_mult"])
        
        st.divider()
        st.caption("Broadcasting")
        cfg["webhook_url"] = st.text_input("Discord Webhook URL", cfg["webhook_url"], type="password")
        
        if st.form_submit_button("Update System"):
            st.toast("System Updated", icon="🔄")
            get_data.clear()

# ==========================================
# 6. LOGIC & BROADCAST ENGINE
# ==========================================
try:
    df = get_data(cfg["symbol"], cfg["timeframe"], cfg["limit"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
except Exception as e:
    st.error(f"Data Connection Error: {e}")
    st.stop()

# Calculate All Indicators
params = cfg # Flattened dict access
try:
    df = calc_apex_vector(df, params)
    df = calc_dark_trend(df, params)
    df = calc_matrix(df, params)
    df = calc_quantum(df, params)
except Exception as e:
    st.error(f"Calculation Error: {e}")
    st.stop()

last = df.iloc[-1]
prev = df.iloc[-2]

# --- BROADCASTER ---
signals = []
if last["trend"] == 1 and prev["trend"] == -1:
    signals.append("🌊 DARK TREND: BULLISH FLIP")
elif last["trend"] == -1 and prev["trend"] == 1:
    signals.append("🌊 DARK TREND: BEARISH FLIP")

if last["apex_state"] == 2 and prev["apex_state"] != 2:
    signals.append("⚡ APEX: SUPER BULL START")
if last["apex_state"] == -2 and prev["apex_state"] != -2:
    signals.append("⚡ APEX: SUPER BEAR START")

if last["div_bull"]: signals.append("💎 APEX: BULLISH DIVERGENCE")
if last["div_bear"]: signals.append("💎 APEX: BEARISH DIVERGENCE")

# Trigger Broadcasts
if signals:
    msg = f"**{cfg['symbol']} ({cfg['timeframe']})**\n" + "\n".join(signals)
    st.toast(msg, icon="🔔")
    if cfg["webhook_url"]:
        try:
            # requests.post(cfg["webhook_url"], json={"content": msg}) 
            pass 
        except: pass

# ==========================================
# 7. VISUALIZATION TABS
# ==========================================
st.title(f"⚡ Titan v5.1 // {cfg['symbol']}")

t_main, t_apex, t_dark, t_mat, t_quant, t_ai = st.tabs([
    "Overview", "Apex Vector", "Dark Trend", "Matrix", "Quantum", "Analyst"
])

# --- TAB 1: OVERVIEW ---
with t_main:
    # Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    
    # Dynamic Color Logic
    trend_c = "c-bull" if last["trend"]==1 else "c-bear"
    flux_c = "c-bull" if last["flux"]>0 else "c-bear"
    
    c1.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Master Trend</div>
        <div class="metric-val {trend_c}">{("BULLISH" if last["trend"]==1 else "BEARISH")}</div>
        <div class="metric-sub">Stop: {last['stop_line']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Apex Vector</div>
        <div class="metric-val {flux_c}">{last['flux']:.2f}</div>
        <div class="metric-sub">Eff: {last['efficiency']*100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    mat_score = int(last["matrix_score"])
    mat_txt = "STRONG BUY" if mat_score==2 else ("STRONG SELL" if mat_score==-2 else "NEUTRAL")
    mat_c = "c-bull" if mat_score==2 else ("c-bear" if mat_score==-2 else "c-cyan")
    
    c3.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Matrix Score</div>
        <div class="metric-val {mat_c}">{mat_txt}</div>
        <div class="metric-sub">Raw: {mat_score}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Recent Alerts</div>
        <div class="metric-val" style="font-size:0.9rem">{signals[0] if signals else "No New Signals"}</div>
        <div class="metric-sub">Scan Time: {datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Main Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], mode='lines', line=dict(color='gray', dash='dot'), name="Stop"))
    
    # Add Buy/Sell Markers from Trend
    buys = df[(df["trend"]==1) & (df["trend"].shift(1)==-1)]
    sells = df[(df["trend"]==-1) & (df["trend"].shift(1)==1)]
    fig.add_trace(go.Scatter(x=buys["timestamp"], y=buys["low"], mode="markers", marker=dict(symbol="triangle-up", color="#00E676", size=12), name="Trend Buy"))
    fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["high"], mode="markers", marker=dict(symbol="triangle-down", color="#FF1744", size=12), name="Trend Sell"))
    
    fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: APEX VECTOR DETAILED ---
with t_apex:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        # Dual Pane: Price + Flux
        fig_av = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_av.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]), row=1, col=1)
        
        # Color Flux Bars
        colors = ["#00E676" if s == 2 else ("#FF1744" if s == -2 else ("#546E7A" if s == 0 else "#FFD600")) for s in df["apex_state"]]
        fig_av.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=colors, name="Flux"), row=2, col=1)
        
        # Add Threshold Lines
        th = cfg["eff_super"]
        fig_av.add_hline(y=th, line_dash="dot", line_color="gray", row=2, col=1)
        fig_av.add_hline(y=-th, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig_av.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_av, use_container_width=True)
        
    with col_r:
        st.markdown("#### 🔍 Vector Analysis")
        
        # Rule Based Text Generation
        flux_val = last["flux"]
        eff_val = last["efficiency"]
        
        state_str = "neutral"
        if flux_val > cfg["eff_super"]: state_str = "super_bull"
        elif flux_val < -cfg["eff_super"]: state_str = "super_bear"
        elif abs(flux_val) < cfg["eff_resist"]: state_str = "resistive"
        
        analysis_text = ""
        if state_str == "super_bull":
            analysis_text = "Market is in **Superconductor State (Bullish)**. Efficiency is high, meaning price is moving with low resistance. Volume supports the move."
        elif state_str == "super_bear":
            analysis_text = "Market is in **Superconductor State (Bearish)**. Sellers are dominating with high efficiency. Expect continuation down."
        elif state_str == "resistive":
            analysis_text = "Market is **Resistive (Choppy)**. Flux is too low to sustain a trend. Avoid trading or use mean-reversion tactics."
        else:
            analysis_text = "Market is in **High Heat**. Volatility is present but direction is not fully efficient yet. Caution advised."
            
        st.markdown(f"""
        <div class="analysis-box">
            <b>Current Flux:</b> {flux_val:.3f}<br>
            <b>Efficiency:</b> {eff_val:.2f}<br>
            <hr style="border-color:#333">
            {analysis_text}
        </div>
        """, unsafe_allow_html=True)
        
        if last["div_bull"]:
            st.warning("⚠️ Bullish Divergence Detected!")

# --- TAB 3: DARK TREND DETAILED ---
with t_dark:
    fig_dk = go.Figure()
    fig_dk.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], line=dict(color="#00E5FF", width=2), name="Trend Line"))
    
    # Cloud Effect
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] + (df["stop_line"]*0.005), line=dict(width=0), showlegend=False))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] - (df["stop_line"]*0.005), fill="tonexty", 
                                fillcolor=("rgba(0, 230, 118, 0.15)" if last["trend"]==1 else "rgba(255, 23, 68, 0.15)"), 
                                line=dict(width=0), name="Cloud"))
    
    fig_dk.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_dk, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        chop = last["chop"]
        chop_state = "TRENDING" if chop < 50 else "CHOPPY/RANGING"
        chop_color = "c-bull" if chop < 50 else "c-bear"
        
        st.markdown(f"""
        <div class="analysis-box">
            <b>Chop Index:</b> {chop:.1f} <span class="{chop_color}">({chop_state})</span><br>
            Values below 38 indicate strong trends. Values above 61 indicate intense consolidation.
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: MATRIX DETAILED ---
with t_mat:
    fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig_m.add_trace(go.Scatter(x=df["timestamp"], y=df["mfi"], fill="tozeroy", line=dict(color="#D500F9"), name="Money Flow"), row=1, col=1)
    fig_m.add_trace(go.Bar(x=df["timestamp"], y=df["hyperwave"], marker_color="#00E5FF", name="HyperWave"), row=2, col=1)
    fig_m.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_m, use_container_width=True)
    
    st.info("Matrix combines **Money Flow** (Volume+RSI) and **HyperWave** (Double Smoothed Momentum). When both align, the signal is strongest.")

# --- TAB 5: QUANTUM ---
with t_quant:
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], line=dict(color="white"), name="RQZO"))
    
    # Highlight Chaos
    chaos_zone = df[df["entropy"] > 2.0] # Arbitrary threshold for this simplified calc
    fig_q.add_trace(go.Scatter(x=chaos_zone["timestamp"], y=chaos_zone["rqzo"], mode="markers", marker=dict(color="red", size=4), name="High Entropy"))
    
    fig_q.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_q, use_container_width=True)
    
    st.markdown("Red dots indicate **High Entropy (Chaos)** areas where price prediction is statistically unreliable.")

# --- TAB 6: AI ANALYST ---
with t_ai:
    st.subheader("🤖 GPT-4o Quant Synthesis")
    
    prompt_txt = f"""
    ASSET: {cfg['symbol']} ({cfg['timeframe']})
    
    1. APEX VECTOR:
    - Flux: {last['flux']:.3f} (Threshold {cfg['eff_super']})
    - Efficiency: {last['efficiency']:.2f}
    - Divergence: {('BULL' if last['div_bull'] else ('BEAR' if last['div_bear'] else 'NONE'))}
    
    2. DARK TREND:
    - Direction: {('UP' if last['trend']==1 else 'DOWN')}
    - Chop Index: {last['chop']:.1f}
    
    3. MATRIX SCORE: {last['matrix_score']}
    
    TASK: Provide a 3-sentence executive summary on BIAS, ENTRY, and RISK.
    """
    
    st.code(prompt_txt, language="text")
    
    ai_key = st.text_input("OpenAI API Key (Optional)", type="password")
    if st.button("Generate Report"):
        if not ai_key:
            st.error("Please provide an API Key.")
        else:
            with st.spinner("Analyzing market physics..."):
                try:
                    client = OpenAI(api_key=ai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"user", "content": prompt_txt}]
                    )
                    st.success(resp.choices[0].message.content)
                except Exception as e:
                    st.error(str(e))    df = df.copy()
    atr = rma(df["high"] - df["low"], int(p["len_main"]))
    
    # SuperTrend-like Logic
    hl2 = (df["high"] + df["low"]) / 2
    up = hl2 + (p["st_mult"] * atr)
    dn = hl2 - (p["st_mult"] * atr)
    
    # Trend Loop
    trend = np.zeros(len(df))
    t_up = np.zeros(len(df))
    t_dn = np.zeros(len(df))
    
    # Init
    t_up[0] = up.iloc[0]
    t_dn[0] = dn.iloc[0]
    
    close = df["close"].values
    up_val = up.values
    dn_val = dn.values
    
    for i in range(1, len(df)):
        # Upper band
        if up_val[i] < t_up[i-1] or close[i-1] > t_up[i-1]:
            t_up[i] = up_val[i]
        else:
            t_up[i] = t_up[i-1]
            
        # Lower band
        if dn_val[i] > t_dn[i-1] or close[i-1] < t_dn[i-1]:
            t_dn[i] = dn_val[i]
        else:
            t_dn[i] = t_dn[i-1]
            
        # Trend Switch
        prev = trend[i-1]
        if prev == -1 and close[i] > t_up[i-1]:
            trend[i] = 1
        elif prev == 1 and close[i] < t_dn[i-1]:
            trend[i] = -1
        else:
            trend[i] = prev if prev != 0 else 1

    df["trend"] = trend
    df["stop_line"] = np.where(trend==1, t_dn, t_up)
    
    # Chop Index
    df["chop"] = 100 * np.log10(atr.rolling(14).sum() / (df["high"].rolling(14).max() - df["low"].rolling(14).min())) / np.log10(14)
    return df

# --- MATRIX ---
def calc_matrix(df, p):
    df = df.copy()
    # Money Flow
    chg = df["close"].diff()
    rsi = 100 - (100 / (1 + (rma(chg.clip(lower=0), 14) / rma(-chg.clip(upper=0), 14))))
    df["mfi"] = ((rsi - 50) * (df["volume"]/df["volume"].rolling(20).mean())).ewm(span=3).mean()
    
    # HyperWave
    src = df["close"].diff()
    hw = 100 * (double_smooth(src, 25, 13) / (double_smooth(src.abs(), 25, 13) + 1e-10))
    df["hyperwave"] = hw / 2 # Scale
    
    df["matrix_score"] = np.sign(df["mfi"]) + np.sign(df["hyperwave"])
    return df

# --- QUANTUM ---
def calc_quantum(df, p):
    df = df.copy()
    # Simplified RQZO logic for speed
    src = df["close"]
    norm = (src - src.rolling(100).min()) / (src.rolling(100).max() - src.rolling(100).min() + 1e-10)
    
    # Entropy (Market Disorder)
    df["entropy"] = df["close"].pct_change().rolling(20).std() * 100 # Rough proxy for disorder
    
    # RQZO Oscillator
    df["rqzo"] = (norm - 0.5) * 10 * (1 - df["entropy"]/10) # Modulation
    return df

# ==========================================
# 4. DATA & SETTINGS
# ==========================================
@st.cache_resource
def get_exchange():
    return ccxt.kraken({"enableRateLimit": True})

@st.cache_data(ttl=30)
def get_data(sym, tf, lim):
    return pd.DataFrame(get_exchange().fetch_ohlcv(sym, tf, limit=lim), columns=["timestamp", "open", "high", "low", "close", "volume"])

def init_settings():
    defaults = {
        "symbol": "BTC/USD", 
        "timeframe": "15m", 
        "limit": 500,
        "len_main": 55, 
        "st_mult": 4.0, 
        "eff_super": 0.6, 
        "eff_resist": 0.3,
        "webhook_url": "",
        "vol_norm": 55,
        "len_vec": 14,
        "len_sm": 5,
        "strictness": 1.0
    }
    if "cfg" not in st.session_state:
        st.session_state.cfg = defaults
    else:
        for k,v in defaults.items():
            if k not in st.session_state.cfg: 
                st.session_state.cfg[k] = v

init_settings()
cfg = st.session_state.cfg

# ==========================================
# 5. SIDEBAR & SETTINGS
# ==========================================
with st.sidebar:
    st.header("Titan Config")
    with st.form("settings"):
        cfg["symbol"] = st.text_input("Symbol", cfg["symbol"])
        c1, c2 = st.columns(2)
        cfg["timeframe"] = c1.selectbox("Timeframe", ["1m","5m","15m","1h","4h"], index=2)
        cfg["limit"] = c2.slider("Lookback", 200, 1000, cfg["limit"])
        
        st.divider()
        st.caption("Strategy Params")
        cfg["eff_super"] = st.slider("Apex Super Thresh", 0.1, 1.0, cfg["eff_super"])
        cfg["st_mult"] = st.number_input("Dark Trend Mult", 1.0, 10.0, cfg["st_mult"])
        
        st.divider()
        st.caption("Broadcasting")
        cfg["webhook_url"] = st.text_input("Discord Webhook URL", cfg["webhook_url"], type="password")
        
        if st.form_submit_button("Update System"):
            st.toast("System Updated", icon="🔄")
            get_data.clear()

# ==========================================
# 6. LOGIC & BROADCAST ENGINE
# ==========================================
try:
    df = get_data(cfg["symbol"], cfg["timeframe"], cfg["limit"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
except Exception as e:
    st.error(f"Data Connection Error: {e}")
    st.stop()

# Calculate All Indicators
params = cfg # Flattened dict access
try:
    df = calc_apex_vector(df, params)
    df = calc_dark_trend(df, params)
    df = calc_matrix(df, params)
    df = calc_quantum(df, params)
except Exception as e:
    st.error(f"Calculation Error: {e}")
    st.stop()

last = df.iloc[-1]
prev = df.iloc[-2]

# --- BROADCASTER ---
signals = []
if last["trend"] == 1 and prev["trend"] == -1:
    signals.append("🌊 DARK TREND: BULLISH FLIP")
elif last["trend"] == -1 and prev["trend"] == 1:
    signals.append("🌊 DARK TREND: BEARISH FLIP")

if last["apex_state"] == 2 and prev["apex_state"] != 2:
    signals.append("⚡ APEX: SUPER BULL START")
if last["apex_state"] == -2 and prev["apex_state"] != -2:
    signals.append("⚡ APEX: SUPER BEAR START")

if last["div_bull"]: signals.append("💎 APEX: BULLISH DIVERGENCE")
if last["div_bear"]: signals.append("💎 APEX: BEARISH DIVERGENCE")

# Trigger Broadcasts
if signals:
    msg = f"**{cfg['symbol']} ({cfg['timeframe']})**\n" + "\n".join(signals)
    st.toast(msg, icon="🔔")
    if cfg["webhook_url"]:
        try:
            # requests.post(cfg["webhook_url"], json={"content": msg}) 
            pass 
        except: pass

# ==========================================
# 7. VISUALIZATION TABS
# ==========================================
st.title(f"⚡ Titan v5.1 // {cfg['symbol']}")

t_main, t_apex, t_dark, t_mat, t_quant, t_ai = st.tabs([
    "Overview", "Apex Vector", "Dark Trend", "Matrix", "Quantum", "Analyst"
])

# --- TAB 1: OVERVIEW ---
with t_main:
    # Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    
    # Dynamic Color Logic
    trend_c = "c-bull" if last["trend"]==1 else "c-bear"
    flux_c = "c-bull" if last["flux"]>0 else "c-bear"
    
    c1.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Master Trend</div>
        <div class="metric-val {trend_c}">{("BULLISH" if last["trend"]==1 else "BEARISH")}</div>
        <div class="metric-sub">Stop: {last['stop_line']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Apex Vector</div>
        <div class="metric-val {flux_c}">{last['flux']:.2f}</div>
        <div class="metric-sub">Eff: {last['efficiency']*100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    mat_score = int(last["matrix_score"])
    mat_txt = "STRONG BUY" if mat_score==2 else ("STRONG SELL" if mat_score==-2 else "NEUTRAL")
    mat_c = "c-bull" if mat_score==2 else ("c-bear" if mat_score==-2 else "c-cyan")
    
    c3.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Matrix Score</div>
        <div class="metric-val {mat_c}">{mat_txt}</div>
        <div class="metric-sub">Raw: {mat_score}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Recent Alerts</div>
        <div class="metric-val" style="font-size:0.9rem">{signals[0] if signals else "No New Signals"}</div>
        <div class="metric-sub">Scan Time: {datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Main Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], mode='lines', line=dict(color='gray', dash='dot'), name="Stop"))
    
    # Add Buy/Sell Markers from Trend
    buys = df[(df["trend"]==1) & (df["trend"].shift(1)==-1)]
    sells = df[(df["trend"]==-1) & (df["trend"].shift(1)==1)]
    fig.add_trace(go.Scatter(x=buys["timestamp"], y=buys["low"], mode="markers", marker=dict(symbol="triangle-up", color="#00E676", size=12), name="Trend Buy"))
    fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["high"], mode="markers", marker=dict(symbol="triangle-down", color="#FF1744", size=12), name="Trend Sell"))
    
    fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: APEX VECTOR DETAILED ---
with t_apex:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        # Dual Pane: Price + Flux
        fig_av = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_av.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]), row=1, col=1)
        
        # Color Flux Bars
        colors = ["#00E676" if s == 2 else ("#FF1744" if s == -2 else ("#546E7A" if s == 0 else "#FFD600")) for s in df["apex_state"]]
        fig_av.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=colors, name="Flux"), row=2, col=1)
        
        # Add Threshold Lines
        th = cfg["eff_super"]
        fig_av.add_hline(y=th, line_dash="dot", line_color="gray", row=2, col=1)
        fig_av.add_hline(y=-th, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig_av.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_av, use_container_width=True)
        
    with col_r:
        st.markdown("#### 🔍 Vector Analysis")
        
        # Rule Based Text Generation
        flux_val = last["flux"]
        eff_val = last["efficiency"]
        
        state_str = "neutral"
        if flux_val > cfg["eff_super"]: state_str = "super_bull"
        elif flux_val < -cfg["eff_super"]: state_str = "super_bear"
        elif abs(flux_val) < cfg["eff_resist"]: state_str = "resistive"
        
        analysis_text = ""
        if state_str == "super_bull":
            analysis_text = "Market is in **Superconductor State (Bullish)**. Efficiency is high, meaning price is moving with low resistance. Volume supports the move."
        elif state_str == "super_bear":
            analysis_text = "Market is in **Superconductor State (Bearish)**. Sellers are dominating with high efficiency. Expect continuation down."
        elif state_str == "resistive":
            analysis_text = "Market is **Resistive (Choppy)**. Flux is too low to sustain a trend. Avoid trading or use mean-reversion tactics."
        else:
            analysis_text = "Market is in **High Heat**. Volatility is present but direction is not fully efficient yet. Caution advised."
            
        st.markdown(f"""
        <div class="analysis-box">
            <b>Current Flux:</b> {flux_val:.3f}<br>
            <b>Efficiency:</b> {eff_val:.2f}<br>
            <hr style="border-color:#333">
            {analysis_text}
        </div>
        """, unsafe_allow_html=True)
        
        if last["div_bull"]:
            st.warning("⚠️ Bullish Divergence Detected!")

# --- TAB 3: DARK TREND DETAILED ---
with t_dark:
    fig_dk = go.Figure()
    fig_dk.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], line=dict(color="#00E5FF", width=2), name="Trend Line"))
    
    # Cloud Effect
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] + (df["stop_line"]*0.005), line=dict(width=0), showlegend=False))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] - (df["stop_line"]*0.005), fill="tonexty", 
                                fillcolor=("rgba(0, 230, 118, 0.15)" if last["trend"]==1 else "rgba(255, 23, 68, 0.15)"), 
                                line=dict(width=0), name="Cloud"))
    
    fig_dk.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_dk, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        chop = last["chop"]
        chop_state = "TRENDING" if chop < 50 else "CHOPPY/RANGING"
        chop_color = "c-bull" if chop < 50 else "c-bear"
        
        st.markdown(f"""
        <div class="analysis-box">
            <b>Chop Index:</b> {chop:.1f} <span class="{chop_color}">({chop_state})</span><br>
            Values below 38 indicate strong trends. Values above 61 indicate intense consolidation.
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: MATRIX DETAILED ---
with t_mat:
    fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig_m.add_trace(go.Scatter(x=df["timestamp"], y=df["mfi"], fill="tozeroy", line=dict(color="#D500F9"), name="Money Flow"), row=1, col=1)
    fig_m.add_trace(go.Bar(x=df["timestamp"], y=df["hyperwave"], marker_color="#00E5FF", name="HyperWave"), row=2, col=1)
    fig_m.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_m, use_container_width=True)
    
    st.info("Matrix combines **Money Flow** (Volume+RSI) and **HyperWave** (Double Smoothed Momentum). When both align, the signal is strongest.")

# --- TAB 5: QUANTUM ---
with t_quant:
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], line=dict(color="white"), name="RQZO"))
    
    # Highlight Chaos
    chaos_zone = df[df["entropy"] > 2.0] # Arbitrary threshold for this simplified calc
    fig_q.add_trace(go.Scatter(x=chaos_zone["timestamp"], y=chaos_zone["rqzo"], mode="markers", marker=dict(color="red", size=4), name="High Entropy"))
    
    fig_q.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_q, use_container_width=True)
    
    st.markdown("Red dots indicate **High Entropy (Chaos)** areas where price prediction is statistically unreliable.")

# --- TAB 6: AI ANALYST ---
with t_ai:
    st.subheader("🤖 GPT-4o Quant Synthesis")
    
    prompt_txt = f"""
    ASSET: {cfg['symbol']} ({cfg['timeframe']})
    
    1. APEX VECTOR:
    - Flux: {last['flux']:.3f} (Threshold {cfg['eff_super']})
    - Efficiency: {last['efficiency']:.2f}
    - Divergence: {('BULL' if last['div_bull'] else ('BEAR' if last['div_bear'] else 'NONE'))}
    
    2. DARK TREND:
    - Direction: {('UP' if last['trend']==1 else 'DOWN')}
    - Chop Index: {last['chop']:.1f}
    
    3. MATRIX SCORE: {last['matrix_score']}
    
    TASK: Provide a 3-sentence executive summary on BIAS, ENTRY, and RISK.
    """
    
    st.code(prompt_txt, language="text")
    
    ai_key = st.text_input("OpenAI API Key (Optional)", type="password")
    if st.button("Generate Report"):
        if not ai_key:
            st.error("Please provide an API Key.")
        else:
            with st.spinner("Analyzing market physics..."):
                try:
                    client = OpenAI(api_key=ai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"user", "content": prompt_txt}]
                    )
                    st.success(resp.choices[0].message.content)
                except Exception as e:
                    st.error(str(e))  --cyan:#00E5FF;
  --vio:#D500F9;
}

.stApp { background: var(--bg); color: var(--text); font-family: 'Roboto Mono', monospace; }
.block-container { padding-top: 1rem; }

/* Custom Metric Box */
.metric-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
}
.metric-lbl { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.metric-val { font-size: 1.1rem; font-weight: 700; color: var(--text); }
.metric-sub { font-size: 0.7rem; color: #666; }

.c-bull { color: var(--bull) !important; }
.c-bear { color: var(--bear) !important; }
.c-cyan { color: var(--cyan) !important; }
.c-vio  { color: var(--vio) !important; }

/* Analysis Box */
.analysis-box {
    border-left: 3px solid var(--cyan);
    background: #050505;
    padding: 12px;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #ccc;
    margin-top: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. MATH ENGINE
# ==========================================
def rma(series, length):
    return series.ewm(alpha=1 / length, adjust=False).mean()

def double_smooth(src, long_len, short_len):
    return src.ewm(span=long_len, adjust=False).mean().ewm(span=short_len, adjust=False).mean()

def hma(series, length):
    # HMA requires integer lengths for window calculations
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    def wma(s, l):
        weights = np.arange(1, l + 1)
        return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    wma_half = wma(series, half_length)
    wma_full = wma(series, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_length)

# --- APEX VECTOR v4.1 ---
def calc_apex_vector(df, p):
    df = df.copy()
    # Efficiency
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    eff = (body / (rng + 1e-10)).ewm(span=int(p["len_vec"])).mean()
    df["efficiency"] = eff

    # Flux
    vol_fact = df["volume"] / (df["volume"].rolling(int(p["vol_norm"])).mean() + 1e-10)
    raw = np.sign(df["close"] - df["open"]) * eff * vol_fact
    df["flux"] = raw.ewm(span=int(p["len_sm"])).mean()

    # State
    th_s = p["eff_super"] * p["strictness"]
    th_r = p["eff_resist"] * p["strictness"]
    cond = [(df["flux"] > th_s), (df["flux"] < -th_s), (df["flux"].abs() < th_r)]
    df["apex_state"] = np.select(cond, [2, -2, 0], default=1)
    
    # Divergence (Simplified Vectorized)
    src = df["flux"]
    df["pivot_low"] = (src.shift(1) < src.shift(2)) & (src.shift(1) < src)
    df["pivot_high"] = (src.shift(1) > src.shift(2)) & (src.shift(1) > src)
    
    # Very basic Divergence hook (Lookback 5)
    df["div_bull"] = df["pivot_low"] & (df["close"] < df["close"].shift(5)) & (df["flux"] > df["flux"].shift(5))
    df["div_bear"] = df["pivot_high"] & (df["close"] > df["close"].shift(5)) & (df["flux"] < df["flux"].shift(5))
    
    return df

# --- DARK TREND ---
def calc_dark_trend(df, p):
    df = df.copy()
    atr = rma(df["high"] - df["low"], int(p["len_main"]))
    
    # SuperTrend-like Logic
    hl2 = (df["high"] + df["low"]) / 2
    up = hl2 + (p["st_mult"] * atr)
    dn = hl2 - (p["st_mult"] * atr)
    
    # Trend Loop
    trend = np.zeros(len(df))
    t_up = np.zeros(len(df))
    t_dn = np.zeros(len(df))
    
    # Init
    t_up[0] = up.iloc[0]
    t_dn[0] = dn.iloc[0]
    
    close = df["close"].values
    up_val = up.values
    dn_val = dn.values
    
    for i in range(1, len(df)):
        # Upper band
        if up_val[i] < t_up[i-1] or close[i-1] > t_up[i-1]:
            t_up[i] = up_val[i]
        else:
            t_up[i] = t_up[i-1]
            
        # Lower band
        if dn_val[i] > t_dn[i-1] or close[i-1] < t_dn[i-1]:
            t_dn[i] = dn_val[i]
        else:
            t_dn[i] = t_dn[i-1]
            
        # Trend Switch
        prev = trend[i-1]
        if prev == -1 and close[i] > t_up[i-1]:
            trend[i] = 1
        elif prev == 1 and close[i] < t_dn[i-1]:
            trend[i] = -1
        else:
            trend[i] = prev if prev != 0 else 1

    df["trend"] = trend
    df["stop_line"] = np.where(trend==1, t_dn, t_up)
    
    # Chop Index
    df["chop"] = 100 * np.log10(atr.rolling(14).sum() / (df["high"].rolling(14).max() - df["low"].rolling(14).min())) / np.log10(14)
    return df

# --- MATRIX ---
def calc_matrix(df, p):
    df = df.copy()
    # Money Flow
    chg = df["close"].diff()
    rsi = 100 - (100 / (1 + (rma(chg.clip(lower=0), 14) / rma(-chg.clip(upper=0), 14))))
    df["mfi"] = ((rsi - 50) * (df["volume"]/df["volume"].rolling(20).mean())).ewm(span=3).mean()
    
    # HyperWave
    src = df["close"].diff()
    hw = 100 * (double_smooth(src, 25, 13) / (double_smooth(src.abs(), 25, 13) + 1e-10))
    df["hyperwave"] = hw / 2 # Scale
    
    df["matrix_score"] = np.sign(df["mfi"]) + np.sign(df["hyperwave"])
    return df

# --- QUANTUM ---
def calc_quantum(df, p):
    df = df.copy()
    # Simplified RQZO logic for speed
    src = df["close"]
    norm = (src - src.rolling(100).min()) / (src.rolling(100).max() - src.rolling(100).min() + 1e-10)
    
    # Entropy (Market Disorder)
    df["entropy"] = df["close"].pct_change().rolling(20).std() * 100 # Rough proxy for disorder
    
    # RQZO Oscillator
    df["rqzo"] = (norm - 0.5) * 10 * (1 - df["entropy"]/10) # Modulation
    return df

# ==========================================
# 4. DATA & SETTINGS (FIXED)
# ==========================================
@st.cache_resource
def get_exchange():
    return ccxt.kraken({"enableRateLimit": True})

@st.cache_data(ttl=30)
def get_data(sym, tf, lim):
    return pd.DataFrame(get_exchange().fetch_ohlcv(sym, tf, limit=lim), columns=["timestamp", "open", "high", "low", "close", "volume"])

def init_settings():
    # ADDED: Missing defaults (vol_norm, len_vec, etc) to prevent KeyError
    defaults = {
        "symbol": "BTC/USD", 
        "timeframe": "15m", 
        "limit": 500,
        "len_main": 55, 
        "st_mult": 4.0, 
        "eff_super": 0.6, 
        "eff_resist": 0.3,
        "webhook_url": "",
        # Missing keys fixed here:
        "vol_norm": 55,
        "len_vec": 14,
        "len_sm": 5,
        "strictness": 1.0
    }
    if "cfg" not in st.session_state:
        st.session_state.cfg = defaults
    else:
        # Merge missing keys into existing session
        for k,v in defaults.items():
            if k not in st.session_state.cfg: 
                st.session_state.cfg[k] = v

init_settings()
cfg = st.session_state.cfg

# ==========================================
# 5. SIDEBAR & SETTINGS
# ==========================================
with st.sidebar:
    st.header("Titan Config")
    with st.form("settings"):
        cfg["symbol"] = st.text_input("Symbol", cfg["symbol"])
        c1, c2 = st.columns(2)
        cfg["timeframe"] = c1.selectbox("Timeframe", ["1m","5m","15m","1h","4h"], index=2)
        cfg["limit"] = c2.slider("Lookback", 200, 1000, cfg["limit"])
        
        st.divider()
        st.caption("Strategy Params")
        cfg["eff_super"] = st.slider("Apex Super Thresh", 0.1, 1.0, cfg["eff_super"])
        cfg["st_mult"] = st.number_input("Dark Trend Mult", 1.0, 10.0, cfg["st_mult"])
        
        st.divider()
        st.caption("Broadcasting")
        cfg["webhook_url"] = st.text_input("Discord Webhook URL", cfg["webhook_url"], type="password")
        
        if st.form_submit_button("Update System"):
            st.toast("System Updated", icon="🔄")
            get_data.clear()

# ==========================================
# 6. LOGIC & BROADCAST ENGINE
# ==========================================
try:
    df = get_data(cfg["symbol"], cfg["timeframe"], cfg["limit"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
except Exception as e:
    st.error(f"Data Connection Error: {e}")
    st.stop()

# Calculate All Indicators
params = cfg # Flattened dict access
try:
    df = calc_apex_vector(df, params)
    df = calc_dark_trend(df, params)
    df = calc_matrix(df, params)
    df = calc_quantum(df, params)
except Exception as e:
    st.error(f"Calculation Error: {e}")
    st.stop()

last = df.iloc[-1]
prev = df.iloc[-2]

# --- BROADCASTER ---
signals = []
if last["trend"] == 1 and prev["trend"] == -1:
    signals.append("🌊 DARK TREND: BULLISH FLIP")
elif last["trend"] == -1 and prev["trend"] == 1:
    signals.append("🌊 DARK TREND: BEARISH FLIP")

if last["apex_state"] == 2 and prev["apex_state"] != 2:
    signals.append("⚡ APEX: SUPER BULL START")
if last["apex_state"] == -2 and prev["apex_state"] != -2:
    signals.append("⚡ APEX: SUPER BEAR START")

if last["div_bull"]: signals.append("💎 APEX: BULLISH DIVERGENCE")
if last["div_bear"]: signals.append("💎 APEX: BEARISH DIVERGENCE")

# Trigger Broadcasts
if signals:
    msg = f"**{cfg['symbol']} ({cfg['timeframe']})**\n" + "\n".join(signals)
    st.toast(msg, icon="🔔")
    if cfg["webhook_url"]:
        try:
            # requests.post(cfg["webhook_url"], json={"content": msg}) 
            pass 
        except: pass

# ==========================================
# 7. VISUALIZATION TABS
# ==========================================
st.title(f"⚡ Titan v5.1 // {cfg['symbol']}")

t_main, t_apex, t_dark, t_mat, t_quant, t_ai = st.tabs([
    "Overview", "Apex Vector", "Dark Trend", "Matrix", "Quantum", "Analyst"
])

# --- TAB 1: OVERVIEW ---
with t_main:
    # Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    
    # Dynamic Color Logic
    trend_c = "c-bull" if last["trend"]==1 else "c-bear"
    flux_c = "c-bull" if last["flux"]>0 else "c-bear"
    
    c1.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Master Trend</div>
        <div class="metric-val {trend_c}">{("BULLISH" if last["trend"]==1 else "BEARISH")}</div>
        <div class="metric-sub">Stop: {last['stop_line']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Apex Vector</div>
        <div class="metric-val {flux_c}">{last['flux']:.2f}</div>
        <div class="metric-sub">Eff: {last['efficiency']*100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    mat_score = int(last["matrix_score"])
    mat_txt = "STRONG BUY" if mat_score==2 else ("STRONG SELL" if mat_score==-2 else "NEUTRAL")
    mat_c = "c-bull" if mat_score==2 else ("c-bear" if mat_score==-2 else "c-cyan")
    
    c3.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Matrix Score</div>
        <div class="metric-val {mat_c}">{mat_txt}</div>
        <div class="metric-sub">Raw: {mat_score}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
    <div class="metric-box">
        <div class="metric-lbl">Recent Alerts</div>
        <div class="metric-val" style="font-size:0.9rem">{signals[0] if signals else "No New Signals"}</div>
        <div class="metric-sub">Scan Time: {datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Main Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], mode='lines', line=dict(color='gray', dash='dot'), name="Stop"))
    
    # Add Buy/Sell Markers from Trend
    buys = df[(df["trend"]==1) & (df["trend"].shift(1)==-1)]
    sells = df[(df["trend"]==-1) & (df["trend"].shift(1)==1)]
    fig.add_trace(go.Scatter(x=buys["timestamp"], y=buys["low"], mode="markers", marker=dict(symbol="triangle-up", color="#00E676", size=12), name="Trend Buy"))
    fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["high"], mode="markers", marker=dict(symbol="triangle-down", color="#FF1744", size=12), name="Trend Sell"))
    
    fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: APEX VECTOR DETAILED ---
with t_apex:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        # Dual Pane: Price + Flux
        fig_av = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_av.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]), row=1, col=1)
        
        # Color Flux Bars
        colors = ["#00E676" if s == 2 else ("#FF1744" if s == -2 else ("#546E7A" if s == 0 else "#FFD600")) for s in df["apex_state"]]
        fig_av.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=colors, name="Flux"), row=2, col=1)
        
        # Add Threshold Lines
        th = cfg["eff_super"]
        fig_av.add_hline(y=th, line_dash="dot", line_color="gray", row=2, col=1)
        fig_av.add_hline(y=-th, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig_av.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_av, use_container_width=True)
        
    with col_r:
        st.markdown("#### 🔍 Vector Analysis")
        
        # Rule Based Text Generation
        flux_val = last["flux"]
        eff_val = last["efficiency"]
        
        state_str = "neutral"
        if flux_val > cfg["eff_super"]: state_str = "super_bull"
        elif flux_val < -cfg["eff_super"]: state_str = "super_bear"
        elif abs(flux_val) < cfg["eff_resist"]: state_str = "resistive"
        
        analysis_text = ""
        if state_str == "super_bull":
            analysis_text = "Market is in **Superconductor State (Bullish)**. Efficiency is high, meaning price is moving with low resistance. Volume supports the move."
        elif state_str == "super_bear":
            analysis_text = "Market is in **Superconductor State (Bearish)**. Sellers are dominating with high efficiency. Expect continuation down."
        elif state_str == "resistive":
            analysis_text = "Market is **Resistive (Choppy)**. Flux is too low to sustain a trend. Avoid trading or use mean-reversion tactics."
        else:
            analysis_text = "Market is in **High Heat**. Volatility is present but direction is not fully efficient yet. Caution advised."
            
        st.markdown(f"""
        <div class="analysis-box">
            <b>Current Flux:</b> {flux_val:.3f}<br>
            <b>Efficiency:</b> {eff_val:.2f}<br>
            <hr style="border-color:#333">
            {analysis_text}
        </div>
        """, unsafe_allow_html=True)
        
        if last["div_bull"]:
            st.warning("⚠️ Bullish Divergence Detected!")

# --- TAB 3: DARK TREND DETAILED ---
with t_dark:
    fig_dk = go.Figure()
    fig_dk.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], line=dict(color="#00E5FF", width=2), name="Trend Line"))
    
    # Cloud Effect
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] + (df["stop_line"]*0.005), line=dict(width=0), showlegend=False))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] - (df["stop_line"]*0.005), fill="tonexty", 
                                fillcolor=("rgba(0, 230, 118, 0.15)" if last["trend"]==1 else "rgba(255, 23, 68, 0.15)"), 
                                line=dict(width=0), name="Cloud"))
    
    fig_dk.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_dk, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        chop = last["chop"]
        chop_state = "TRENDING" if chop < 50 else "CHOPPY/RANGING"
        chop_color = "c-bull" if chop < 50 else "c-bear"
        
        st.markdown(f"""
        <div class="analysis-box">
            <b>Chop Index:</b> {chop:.1f} <span class="{chop_color}">({chop_state})</span><br>
            Values below 38 indicate strong trends. Values above 61 indicate intense consolidation.
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: MATRIX DETAILED ---
with t_mat:
    fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig_m.add_trace(go.Scatter(x=df["timestamp"], y=df["mfi"], fill="tozeroy", line=dict(color="#D500F9"), name="Money Flow"), row=1, col=1)
    fig_m.add_trace(go.Bar(x=df["timestamp"], y=df["hyperwave"], marker_color="#00E5FF", name="HyperWave"), row=2, col=1)
    fig_m.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_m, use_container_width=True)
    
    st.info("Matrix combines **Money Flow** (Volume+RSI) and **HyperWave** (Double Smoothed Momentum). When both align, the signal is strongest.")

# --- TAB 5: QUANTUM ---
with t_quant:
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], line=dict(color="white"), name="RQZO"))
    
    # Highlight Chaos
    chaos_zone = df[df["entropy"] > 2.0] # Arbitrary threshold for this simplified calc
    fig_q.add_trace(go.Scatter(x=chaos_zone["timestamp"], y=chaos_zone["rqzo"], mode="markers", marker=dict(color="red", size=4), name="High Entropy"))
    
    fig_q.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_q, use_container_width=True)
    
    st.markdown("Red dots indicate **High Entropy (Chaos)** areas where price prediction is statistically unreliable.")

# --- TAB 6: AI ANALYST ---
with t_ai:
    st.subheader("🤖 GPT-4o Quant Synthesis")
    
    prompt_txt = f"""
    ASSET: {cfg['symbol']} ({cfg['timeframe']})
    
    1. APEX VECTOR:
    - Flux: {last['flux']:.3f} (Threshold {cfg['eff_super']})
    - Efficiency: {last['efficiency']:.2f}
    - Divergence: {('BULL' if last['div_bull'] else ('BEAR' if last['div_bear'] else 'NONE'))}
    
    2. DARK TREND:
    - Direction: {('UP' if last['trend']==1 else 'DOWN')}
    - Chop Index: {last['chop']:.1f}
    
    3. MATRIX SCORE: {last['matrix_score']}
    
    TASK: Provide a 3-sentence executive summary on BIAS, ENTRY, and RISK.
    """
    
    st.code(prompt_txt, language="text")
    
    ai_key = st.text_input("OpenAI API Key (Optional)", type="password")
    if st.button("Generate Report"):
        if not ai_key:
            st.error("Please provide an API Key.")
        else:
            with st.spinner("Analyzing market physics..."):
                try:
                    client = OpenAI(api_key=ai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"user", "content": prompt_txt}]
                    )
                    st.success(resp.choices[0].message.content)
                except Exception as e:
                    st.error(str(e))with t_apex:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        # Dual Pane: Price + Flux
        fig_av = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
        fig_av.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]), row=1, col=1)
        
        # Color Flux Bars
        colors = ["#00E676" if s == 2 else ("#FF1744" if s == -2 else ("#546E7A" if s == 0 else "#FFD600")) for s in df["apex_state"]]
        fig_av.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=colors, name="Flux"), row=2, col=1)
        
        # Add Threshold Lines
        th = cfg["eff_super"]
        fig_av.add_hline(y=th, line_dash="dot", line_color="gray", row=2, col=1)
        fig_av.add_hline(y=-th, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig_av.update_layout(height=600, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_av, use_container_width=True)
        
    with col_r:
        st.markdown("#### 🔍 Vector Analysis")
        
        # Rule Based Text Generation
        flux_val = last["flux"]
        eff_val = last["efficiency"]
        
        state_str = "neutral"
        if flux_val > cfg["eff_super"]: state_str = "super_bull"
        elif flux_val < -cfg["eff_super"]: state_str = "super_bear"
        elif abs(flux_val) < cfg["eff_resist"]: state_str = "resistive"
        
        analysis_text = ""
        if state_str == "super_bull":
            analysis_text = "Market is in **Superconductor State (Bullish)**. Efficiency is high, meaning price is moving with low resistance. Volume supports the move."
        elif state_str == "super_bear":
            analysis_text = "Market is in **Superconductor State (Bearish)**. Sellers are dominating with high efficiency. Expect continuation down."
        elif state_str == "resistive":
            analysis_text = "Market is **Resistive (Choppy)**. Flux is too low to sustain a trend. Avoid trading or use mean-reversion tactics."
        else:
            analysis_text = "Market is in **High Heat**. Volatility is present but direction is not fully efficient yet. Caution advised."
            
        st.markdown(f"""
        <div class="analysis-box">
            <b>Current Flux:</b> {flux_val:.3f}<br>
            <b>Efficiency:</b> {eff_val:.2f}<br>
            <hr style="border-color:#333">
            {analysis_text}
        </div>
        """, unsafe_allow_html=True)
        
        if last["div_bull"]:
            st.warning("⚠️ Bullish Divergence Detected!")

# --- TAB 3: DARK TREND DETAILED ---
with t_dark:
    fig_dk = go.Figure()
    fig_dk.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"], line=dict(color="#00E5FF", width=2), name="Trend Line"))
    
    # Cloud Effect
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] + (df["stop_line"]*0.005), line=dict(width=0), showlegend=False))
    fig_dk.add_trace(go.Scatter(x=df["timestamp"], y=df["stop_line"] - (df["stop_line"]*0.005), fill="tonexty", 
                                fillcolor=("rgba(0, 230, 118, 0.15)" if last["trend"]==1 else "rgba(255, 23, 68, 0.15)"), 
                                line=dict(width=0), name="Cloud"))
    
    fig_dk.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_dk, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        chop = last["chop"]
        chop_state = "TRENDING" if chop < 50 else "CHOPPY/RANGING"
        chop_color = "c-bull" if chop < 50 else "c-bear"
        
        st.markdown(f"""
        <div class="analysis-box">
            <b>Chop Index:</b> {chop:.1f} <span class="{chop_color}">({chop_state})</span><br>
            Values below 38 indicate strong trends. Values above 61 indicate intense consolidation.
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: MATRIX DETAILED ---
with t_mat:
    fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig_m.add_trace(go.Scatter(x=df["timestamp"], y=df["mfi"], fill="tozeroy", line=dict(color="#D500F9"), name="Money Flow"), row=1, col=1)
    fig_m.add_trace(go.Bar(x=df["timestamp"], y=df["hyperwave"], marker_color="#00E5FF", name="HyperWave"), row=2, col=1)
    fig_m.update_layout(height=550, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_m, use_container_width=True)
    
    st.info("Matrix combines **Money Flow** (Volume+RSI) and **HyperWave** (Double Smoothed Momentum). When both align, the signal is strongest.")

# --- TAB 5: QUANTUM ---
with t_quant:
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], line=dict(color="white"), name="RQZO"))
    
    # Highlight Chaos
    chaos_zone = df[df["entropy"] > 2.0] # Arbitrary threshold for this simplified calc
    fig_q.add_trace(go.Scatter(x=chaos_zone["timestamp"], y=chaos_zone["rqzo"], mode="markers", marker=dict(color="red", size=4), name="High Entropy"))
    
    fig_q.update_layout(height=500, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_q, use_container_width=True)
    
    st.markdown("Red dots indicate **High Entropy (Chaos)** areas where price prediction is statistically unreliable.")

# --- TAB 6: AI ANALYST ---
with t_ai:
    st.subheader("🤖 GPT-4o Quant Synthesis")
    
    # Compile a prompt
    prompt_txt = f"""
    ASSET: {cfg['symbol']} ({cfg['timeframe']})
    
    1. APEX VECTOR:
    - Flux: {last['flux']:.3f} (Threshold {cfg['eff_super']})
    - Efficiency: {last['efficiency']:.2f}
    - Divergence: {('BULL' if last['div_bull'] else ('BEAR' if last['div_bear'] else 'NONE'))}
    
    2. DARK TREND:
    - Direction: {('UP' if last['trend']==1 else 'DOWN')}
    - Chop Index: {last['chop']:.1f}
    
    3. MATRIX SCORE: {last['matrix_score']}
    
    TASK: Provide a 3-sentence executive summary on BIAS, ENTRY, and RISK.
    """
    
    st.code(prompt_txt, language="text")
    
    ai_key = st.text_input("OpenAI API Key (Optional)", type="password")
    if st.button("Generate Report"):
        if not ai_key:
            st.error("Please provide an API Key.")
        else:
            with st.spinner("Analyzing market physics..."):
                try:
                    client = OpenAI(api_key=ai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"user", "content": prompt_txt}]
                    )
                    st.success(resp.choices[0].message.content)
                except Exception as e:
                    st.error(str(e))
