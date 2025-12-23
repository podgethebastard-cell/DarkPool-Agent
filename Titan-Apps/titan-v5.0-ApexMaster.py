import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time
from openai import OpenAI
import requests
import time as time_lib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Titan v7.0 Omni-Terminal",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. UI THEME & CSS (GLASSMORPHISM)
# ==========================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

:root{
  --bg: #000000;
  --card: #090909;
  --border: #1f1f1f;
  --text: #e0e0e0;
  --bull: #00E676;
  --bear: #FF1744;
  --cyan: #00E5FF;
  --vio: #D500F9;
  --glass: rgba(20, 20, 20, 0.6);
}

.stApp { background-color: var(--bg); font-family: 'JetBrains Mono', monospace; }

/* Metrics & Cards */
div[data-testid="metric-container"] {
    background-color: var(--glass);
    border: 1px solid var(--border);
    backdrop-filter: blur(10px);
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}

/* Custom Analysis Panel */
.titan-panel {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}
.titan-h { 
    color: #666; 
    font-size: 0.7rem; 
    text-transform: uppercase; 
    letter-spacing: 2px; 
    font-weight: 700; 
    margin-bottom: 10px; 
    border-bottom: 1px solid #222;
    padding-bottom: 5px;
}
.titan-txt { color: #ccc; font-size: 0.95rem; line-height: 1.6; }

/* Status Pills */
.pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-right: 8px;
    letter-spacing: 1px;
}
.p-bull { background: rgba(0, 230, 118, 0.15); color: var(--bull); border: 1px solid var(--bull); box-shadow: 0 0 10px rgba(0,230,118,0.1); }
.p-bear { background: rgba(255, 23, 68, 0.15); color: var(--bear); border: 1px solid var(--bear); box-shadow: 0 0 10px rgba(255,23,68,0.1); }
.p-neut { background: rgba(85, 85, 85, 0.15); color: #888; border: 1px solid #444; }

/* Log Feed */
.log-item {
    padding: 8px 0;
    border-bottom: 1px solid #111;
    color: #aaa;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.8rem;
}
.log-ts { font-family: monospace; color: #555; font-size: 0.7rem; }

/* Plotly Fix */
.js-plotly-plot .plotly .modebar { opacity: 0.5 !important; }
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

# --- INDICATORS ---
def calc_all_indicators(df, p):
    df = df.copy()
    
    # 1. APEX VECTOR
    rng = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    eff = (body / (rng + 1e-10)).ewm(span=int(p["len_vec"])).mean()
    df["eff"] = eff
    vol_fact = df["volume"] / (df["volume"].rolling(int(p["vol_norm"])).mean() + 1e-10)
    raw = np.sign(df["close"] - df["open"]) * eff * vol_fact
    df["flux"] = raw.ewm(span=int(p["len_sm"])).mean()
    
    th_s = p["eff_super"] * p["strictness"]
    df["apex_state"] = np.select(
        [(df["flux"] > th_s), (df["flux"] < -th_s), (df["flux"].abs() < (p["eff_resist"]*p["strictness"]))], 
        [2, -2, 0], default=1
    )

    # 2. DARK TREND
    atr = rma(df["high"] - df["low"], int(p["len_main"]))
    hl2 = (df["high"] + df["low"]) / 2
    up = hl2 + (p["st_mult"] * atr)
    dn = hl2 - (p["st_mult"] * atr)
    
    trend = np.zeros(len(df))
    stop = np.zeros(len(df))
    close = df["close"].values
    
    t_curr = 1
    s_curr = dn.iloc[0]
    
    for i in range(1, len(df)):
        if close[i] > stop[i-1] and trend[i-1] == -1:
            t_curr = 1
            s_curr = dn.iloc[i]
        elif close[i] < stop[i-1] and trend[i-1] == 1:
            t_curr = -1
            s_curr = up.iloc[i]
        else:
            t_curr = trend[i-1]
            if t_curr == 1:
                s_curr = max(stop[i-1], dn.iloc[i])
            else:
                s_curr = min(stop[i-1], up.iloc[i])
        
        trend[i] = t_curr
        stop[i] = s_curr
        
    df["trend"] = trend
    df["stop"] = stop
    df["chop"] = 100 * np.log10(atr.rolling(14).sum() / (df["high"].rolling(14).max() - df["low"].rolling(14).min())) / np.log10(14)

    # 3. MATRIX
    chg = df["close"].diff()
    rsi_src = 100 - (100 / (1 + (rma(chg.clip(lower=0), 14) / rma(-chg.clip(upper=0), 14))))
    df["mfi"] = ((rsi_src - 50) * (df["volume"]/df["volume"].rolling(20).mean())).ewm(span=3).mean()
    hw_src = df["close"].diff()
    df["hw"] = (100 * (double_smooth(hw_src, 25, 13) / (double_smooth(hw_src.abs(), 25, 13) + 1e-10))) / 2
    df["matrix"] = np.sign(df["mfi"]) + np.sign(df["hw"])

    # 4. QUANTUM
    norm = (df["close"] - df["close"].rolling(100).min()) / (df["close"].rolling(100).max() - df["close"].rolling(100).min() + 1e-10)
    df["entropy"] = df["close"].pct_change().rolling(20).std() * 100
    df["rqzo"] = (norm - 0.5) * 10 * (1 - df["entropy"]/10)

    return df

# ==========================================
# 4. DATA HANDLING (MULTI-EXCHANGE)
# ==========================================
def get_exchange(name):
    if name == 'Binance': return ccxt.binance({"enableRateLimit": True})
    if name == 'Coinbase': return ccxt.coinbase({"enableRateLimit": True})
    if name == 'Bybit': return ccxt.bybit({"enableRateLimit": True})
    if name == 'OKX': return ccxt.okx({"enableRateLimit": True})
    return ccxt.kraken({"enableRateLimit": True})

@st.cache_data(ttl=10) # Short TTL for live feel
def fetch_data(exch_name, sym, tf, lim):
    ex = get_exchange(exch_name)
    return pd.DataFrame(ex.fetch_ohlcv(sym, tf, limit=lim), columns=["timestamp", "open", "high", "low", "close", "volume"])

def init_session():
    try: ai_key = st.secrets["OPENAI_API_KEY"]
    except: ai_key = ""
    try: tg_tok = st.secrets["TG_TOKEN"]
    except: tg_tok = ""
    try: tg_id = st.secrets["TG_CHAT_ID"]
    except: tg_id = ""

    defaults = {
        "exchange": "Kraken", "symbol": "BTC/USD", "timeframe": "15m", "limit": 500,
        "len_main": 55, "st_mult": 4.0, "eff_super": 0.6, "eff_resist": 0.3,
        "vol_norm": 55, "len_vec": 14, "len_sm": 5, "strictness": 1.0,
        "api_key": ai_key, "tg_token": tg_tok, "tg_chat": tg_id, "auto_refresh": False
    }
    if "cfg" not in st.session_state:
        st.session_state.cfg = defaults
    else:
        for k, v in defaults.items():
            if k not in st.session_state.cfg: st.session_state.cfg[k] = v

init_session()
cfg = st.session_state.cfg

# ==========================================
# 5. SIDEBAR CONFIG
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ Titan v7.0")
    
    # Auto Refresh Logic
    if st.checkbox("🔄 Auto-Refresh (60s)", value=cfg["auto_refresh"]):
        st.caption("Live Monitoring Active")
        time_lib.sleep(60)
        st.rerun()

    with st.form("main_cfg"):
        c_ex, c_tf = st.columns(2)
        cfg["exchange"] = c_ex.selectbox("Exchange", ["Kraken", "Binance", "Bybit", "Coinbase", "OKX"], index=["Kraken", "Binance", "Bybit", "Coinbase", "OKX"].index(cfg["exchange"]))
        cfg["timeframe"] = c_tf.selectbox("Time", ["1m","5m","15m","1h","4h","1d"], index=2)
        
        cfg["symbol"] = st.text_input("Asset Pair", cfg["symbol"], help="Format depends on exchange (e.g. BTC/USDT for Binance)")
        cfg["limit"] = st.number_input("Lookback", 200, 1000, cfg["limit"])
        
        st.markdown("---")
        st.caption("Strategy Fine-Tuning")
        cfg["eff_super"] = st.slider("Apex Thresh", 0.1, 1.0, cfg["eff_super"])
        cfg["st_mult"] = st.slider("Trend Width", 1.0, 6.0, cfg["st_mult"])
        
        st.markdown("---")
        cfg["api_key"] = st.text_input("OpenAI Key", cfg["api_key"], type="password")
        cfg["tg_token"] = st.text_input("TG Token", cfg["tg_token"], type="password")
        cfg["tg_chat"] = st.text_input("TG Chat ID", cfg["tg_chat"], type="password")
        
        if st.form_submit_button("APPLY SETTINGS"):
            st.toast("Settings Updated", icon="✅")
            fetch_data.clear()

    st.markdown("---")
    st.markdown("### 📡 Live Feed")
    log_container = st.container()

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
try:
    df = fetch_data(cfg["exchange"], cfg["symbol"], cfg["timeframe"], cfg["limit"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = calc_all_indicators(df, cfg)
    last = df.iloc[-1]
    prev = df.iloc[-2]
except Exception as e:
    st.error(f"Data Error: {e} - Check Symbol Format for {cfg['exchange']}")
    st.stop()

# ==========================================
# 7. TELEGRAM BROADCAST LOGIC
# ==========================================
signals = []
# Trend Flip
if last["trend"] == 1 and prev["trend"] == -1:
    signals.append(("🚀", f"DARK TREND: BULLISH FLIP ({last['close']})"))
elif last["trend"] == -1 and prev["trend"] == 1:
    signals.append(("🔻", f"DARK TREND: BEARISH FLIP ({last['close']})"))

# Apex State Change
if last["apex_state"] == 2 and prev["apex_state"] != 2:
    signals.append(("⚡", "APEX: SUPER BULL START"))
if last["apex_state"] == -2 and prev["apex_state"] != -2:
    signals.append(("⚡", "APEX: SUPER BEAR START"))

# Matrix Flip
if last["matrix"] > 0 and prev["matrix"] <= 0:
    signals.append(("💠", "MATRIX: BUY SIGNAL"))
if last["matrix"] < 0 and prev["matrix"] >= 0:
    signals.append(("💠", "MATRIX: SELL SIGNAL"))

with log_container:
    if signals:
        for icon, msg in signals:
            st.markdown(f"""
            <div class="log-item">
                <span class="log-ts">{datetime.now().strftime('%H:%M')}</span>
                <span style="color:#e0e0e0;">{icon} {msg}</span>
            </div>
            """, unsafe_allow_html=True)
            
            if cfg["tg_token"] and cfg["tg_chat"]:
                try:
                    requests.post(f"https://api.telegram.org/bot{cfg['tg_token']}/sendMessage", json={"chat_id": cfg["tg_chat"], "text": f"{icon} {msg}"})
                except: pass
    else:
        st.caption(f"Monitoring {cfg['exchange']} stream...")

# ==========================================
# 8. VISUALIZATION HELPERS
# ==========================================
def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
        margin=dict(l=0, r=0, t=20, b=0),
        height=500,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        font=dict(family="JetBrains Mono", size=10, color="#888"),
    )
    # Remove grid lines for cleaner look
    fig.update_xaxes(showgrid=False, zeroline=False, color="#444")
    fig.update_yaxes(showgrid=True, gridcolor="#222", zeroline=False, color="#444")
    return fig

# ==========================================
# 9. TABS & DASHBOARD
# ==========================================
def get_apex_analysis(row):
    flux = row["flux"]
    if flux > cfg["eff_super"]: 
        return "super_bull", f"Flux is extremely high ({flux:.2f}). Market is in a **Superconductor State** (Low Resistance)."
    elif flux < -cfg["eff_super"]: 
        return "super_bear", f"Flux is extremely negative ({flux:.2f}). Sellers dominate efficiently."
    elif abs(flux) < cfg["eff_resist"]:
        return "chop", "Flux is near zero. The market is **Resistive/Choppy**."
    else:
        return "heat", "Flux is moderate (**High Heat**). Volatility is present without direction."

def get_dark_analysis(row):
    chop = row["chop"]
    trend = "Bullish" if row["trend"] == 1 else "Bearish"
    state = "Trending" if chop < 45 else ("Consolidating" if chop < 60 else "Choppy")
    return (1 if row["trend"]==1 else -1), f"Trend is **{trend}**. Chop Index: {chop:.1f} ({state})."

def get_combo_analysis(row):
    score = 0
    if row["flux"] > cfg["eff_super"]: score += 1
    if row["trend"] == 1: score += 1
    if row["matrix"] > 0: score += 1
    
    if row["flux"] < -cfg["eff_super"]: score -= 1
    if row["trend"] == -1: score -= 1
    if row["matrix"] < 0: score -= 1
    
    if score >= 3: return "bull", "💎 TRIPLE CONFLUENCE BULL"
    if score <= -3: return "bear", "💎 TRIPLE CONFLUENCE BEAR"
    return "neut", "No major confluence."

apex_state, apex_txt = get_apex_analysis(last)
dark_state, dark_txt = get_dark_analysis(last)
combo_state, combo_txt = get_combo_analysis(last)

st.title(f"⚡ {cfg['symbol']} // {cfg['timeframe']}")

t1, t2, t3, t4, t5 = st.tabs(["Overview", "Apex Vector", "Dark Trend", "Matrix", "AI Analyst"])

# --- TAB 1: OVERVIEW ---
with t1:
    if combo_state != "neut":
        c_color = "#00E676" if combo_state == "bull" else "#FF1744"
        st.markdown(f"""<div style="background:{c_color}10; border:1px solid {c_color}; padding:15px; border-radius:8px; text-align:center; color:{c_color}; font-weight:bold; margin-bottom:15px; box-shadow:0 0 15px {c_color}20;">{combo_txt}</div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Apex Flux", f"{last['flux']:.2f}", delta=("Bull" if last['flux']>0 else "Bear"))
    c2.metric("Trend", ("UP" if last['trend']==1 else "DOWN"), f"Stop: {last['stop']:.2f}")
    c3.metric("Matrix", int(last['matrix']), ("Buy" if last['matrix']>0 else "Sell"))
    c4.metric("Entropy", f"{last['entropy']:.2f}", ("Chaos" if last['entropy']>2.0 else "Stable"))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price",
                                 increasing_line_color="#00E676", decreasing_line_color="#FF1744"))
    color_trend = "#00E676" if last["trend"] == 1 else "#FF1744"
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["stop"], mode="lines", line=dict(color=color_trend, width=2), name="Stop"))
    st.plotly_chart(style_chart(fig), use_container_width=True)

# --- TAB 2: APEX VECTOR ---
with t2:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        fig_av = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.03)
        fig_av.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                        increasing_line_color="#00E676", decreasing_line_color="#FF1744"), row=1, col=1)
        colors = ["#00E676" if x == 2 else ("#FF1744" if x == -2 else ("#546E7A" if x == 0 else "#FFD600")) for x in df["apex_state"]]
        fig_av.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=colors), row=2, col=1)
        fig_av.add_hline(y=cfg["eff_super"], row=2, col=1, line=dict(color="#333", dash="dot"))
        fig_av.add_hline(y=-cfg["eff_super"], row=2, col=1, line=dict(color="#333", dash="dot"))
        st.plotly_chart(style_chart(fig_av), use_container_width=True)
    
    with col_r:
        pill_class = "p-bull" if apex_state=="super_bull" else ("p-bear" if apex_state=="super_bear" else "p-neut")
        pill_text = apex_state.replace("_", " ").upper()
        st.markdown(f"""
        <div class="titan-panel">
            <div class="titan-h">Status</div>
            <div class="pill {pill_class}">{pill_text}</div>
            <br><br>
            <div class="titan-h">Analysis</div>
            <div class="titan-txt">{apex_txt}</div>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: DARK TREND ---
with t3:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        fig_dt = go.Figure()
        fig_dt.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                        increasing_line_color="#00E676", decreasing_line_color="#FF1744"))
        fig_dt.add_trace(go.Scatter(x=df["timestamp"], y=df["stop"], line=dict(color=color_trend, width=2)))
        st.plotly_chart(style_chart(fig_dt), use_container_width=True)
    with col_r:
        pill_class = "p-bull" if dark_state == 1 else "p-bear"
        st.markdown(f"""
        <div class="titan-panel">
            <div class="titan-h">Trend State</div>
            <div class="pill {pill_class}">{'BULLISH' if dark_state==1 else 'BEARISH'}</div>
            <br><br>
            <div class="titan-h">Context</div>
            <div class="titan-txt">{dark_txt}</div>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: MATRIX ---
with t4:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        fig_mx = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig_mx.add_trace(go.Scatter(x=df["timestamp"], y=df["mfi"], fill="tozeroy", line=dict(color="#D500F9", width=2), name="MFI"), row=1, col=1)
        fig_mx.add_trace(go.Bar(x=df["timestamp"], y=df["hw"], marker_color="#00E5FF", name="HyperWave"), row=2, col=1)
        st.plotly_chart(style_chart(fig_mx), use_container_width=True)
    with col_r:
        st.markdown("""<div class="titan-panel"><div class="titan-h">Matrix Logic</div><div class="titan-txt"><span style='color:#D500F9'>Purple</span>: Money Flow (Volume + RSI).<br><span style='color:#00E5FF'>Cyan</span>: HyperWave (Momentum).<br><br>When both align, moves expand.</div></div>""", unsafe_allow_html=True)

# --- TAB 5: AI ANALYST ---
with t5:
    st.subheader("🤖 GPT-4o Quant Synthesis")
    user_prompt = f"""
    Analyze this Crypto Market State:
    Asset: {cfg['symbol']} ({cfg['timeframe']})
    1. Apex Flux: {last['flux']:.2f} (State: {apex_state})
    2. Trend: {('Bull' if last['trend']==1 else 'Bear')} (Stop: {last['stop']:.2f})
    3. Matrix Score: {last['matrix']}
    4. Confluence: {combo_txt}
    
    Output: Executive Summary (Bias, Entry Tactic, Risk Level).
    """
    st.code(user_prompt, language="text")
    if st.button("Generate Strategy"):
        if not cfg["api_key"]:
            st.error("API Key required. Check secrets or sidebar.")
        else:
            with st.spinner("Synthesizing..."):
                try:
                    client = OpenAI(api_key=cfg["api_key"])
                    resp = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": user_prompt}])
                    st.success(resp.choices[0].message.content)
                except Exception as e:
                    st.error(str(e))
