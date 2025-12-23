import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from openai import OpenAI

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Titan v5.0 Apex Master",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed",
)

# ==========================================
# 2. UI THEME (CLEAN DARK TERMINAL)
# ==========================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600;700&display=swap');

:root{
  --bg:#000;
  --panel:#070707;
  --panel2:#0b0b0b;
  --border:#1f1f1f;
  --muted:#8a8a8a;
  --text:#e6e6e6;
  --bull:#00E676;
  --bear:#FF1744;
  --heat:#FFD600;
  --cyan:#00E5FF;
  --vio:#B388FF;
}

.stApp { background: var(--bg); color: var(--text); font-family: 'Roboto Mono', monospace; }
#MainMenu, footer, header {visibility: hidden;}

.block-container { padding-top: 0.9rem; padding-bottom: 1.25rem; }

a, a:visited { color: var(--cyan); }

hr { border: none; border-top: 1px solid var(--border); margin: 0.8rem 0; }

 /* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
  background: var(--panel2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

/* Buttons */
.stButton>button {
  width: 100%;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: linear-gradient(180deg, #101010, #070707);
  color: var(--text);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 0.55rem 0.75rem;
}
.stButton>button:hover { border-color: var(--cyan); color: var(--cyan); }

/* Top bar */
.titan-topbar{
  display:flex; align-items:center; justify-content:space-between;
  background: rgba(5,5,5,0.75);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 0.7rem 0.9rem;
  margin-bottom: 0.8rem;
  backdrop-filter: blur(6px);
}
.titan-title{
  display:flex; flex-direction:column; gap:2px;
}
.titan-title .h{
  font-size: 1.05rem; font-weight: 800; letter-spacing: 0.4px;
}
.titan-title .s{
  font-size: 0.78rem; color: var(--muted);
}

/* Status pill */
.pill{
  display:inline-flex; align-items:center; gap:8px;
  border-radius: 999px;
  padding: 0.25rem 0.55rem;
  border: 1px solid var(--border);
  background: rgba(15,15,15,0.8);
  font-size: 0.75rem;
  color: var(--text);
}
.dot{
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--cyan);
  box-shadow: 0 0 12px rgba(0,229,255,0.35);
}

/* Cards */
.titan-card{
  background: linear-gradient(180deg, rgba(10,10,10,0.9), rgba(6,6,6,0.9));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 0.85rem 0.9rem;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.titan-card .k { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 800; }
.titan-card .v { margin-top: 0.25rem; font-size: 1.35rem; font-weight: 900; }
.titan-card .sub { margin-top: 0.25rem; font-size: 0.78rem; color: #b0b0b0; }

.accent-bull { border-left: 3px solid var(--bull); }
.accent-bear { border-left: 3px solid var(--bear); }
.accent-heat { border-left: 3px solid var(--heat); }
.accent-chop { border-left: 3px solid #546E7A; }

.c-bull{ color: var(--bull); }
.c-bear{ color: var(--bear); }
.c-heat{ color: var(--heat); }
.c-cyan{ color: var(--cyan); }

/* AI box */
.ai-box{
  background: rgba(6,6,6,0.9);
  border: 1px solid var(--border);
  border-left: 3px solid var(--vio);
  border-radius: 16px;
  padding: 0.9rem;
  font-size: 0.88rem;
  color: #cfcfcf;
  line-height: 1.45;
}

/* Mobile-ish */
@media (max-width: 900px){
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 3. MATH ENGINE (YOUR ORIGINAL FUNCTIONS)
# ==========================================
def rma(series, length):
    return series.ewm(alpha=1 / length, adjust=False).mean()

def double_smooth(src, long_len, short_len):
    smooth1 = src.ewm(span=long_len, adjust=False).mean()
    smooth2 = smooth1.ewm(span=short_len, adjust=False).mean()
    return smooth2

def wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = wma(series, half_length)
    wma_full = wma(series, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_length)

def calc_apex_vector(df, p):
    df = df.copy()
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["raw_eff"] = np.where(df["range"] == 0, 0.0, df["body"] / df["range"])
    df["efficiency"] = df["raw_eff"].ewm(span=p["vec_len"], adjust=False).mean()

    df["vol_avg"] = df["volume"].rolling(p["vol_norm"]).mean()
    df["vol_fact"] = np.where(df["vol_avg"] == 0, 1.0, df["volume"] / df["vol_avg"])

    df["direction"] = np.sign(df["close"] - df["open"])
    th_super = p["eff_super"] * p["strictness"]
    th_resist = p["eff_resist"] * p["strictness"]

    df["vector_raw"] = df["direction"] * df["efficiency"] * df["vol_fact"]
    df["apex_flux"] = df["vector_raw"].ewm(span=p["flux_sm"], adjust=False).mean()

    conditions = [
        (df["apex_flux"] > th_super),
        (df["apex_flux"] < -th_super),
        (df["apex_flux"].abs() < th_resist),
    ]
    choices = [2, -2, 0]
    df["apex_state"] = np.select(conditions, choices, default=1)
    return df

def calc_dark_vector(df, p):
    df = df.copy()
    atr1 = rma(df["high"] - df["low"], 1)
    sum_atr = atr1.rolling(p["chop_len"]).sum()
    max_hi = df["high"].rolling(p["chop_len"]).max()
    min_lo = df["low"].rolling(p["chop_len"]).min()
    df["chop_idx"] = 100 * np.log10(sum_atr / (max_hi - min_lo)) / np.log10(p["chop_len"])
    df["is_choppy"] = df["chop_idx"] > p["chop_thresh"]

    atr = rma(df["high"] - df["low"], p["st_len"])
    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + (p["st_mult"] * atr)
    basic_lower = hl2 - (p["st_mult"] * atr)

    final_upper = np.zeros(len(df))
    final_lower = np.zeros(len(df))
    trend = np.zeros(len(df))
    close = df["close"].values

    for i in range(1, len(df)):
        if basic_upper.iloc[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper.iloc[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower.iloc[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower.iloc[i]
        else:
            final_lower[i] = final_lower[i - 1]

        prev_trend = trend[i - 1] if i > 1 else 1
        if prev_trend == -1 and close[i] > final_upper[i - 1]:
            trend[i] = 1
        elif prev_trend == 1 and close[i] < final_lower[i - 1]:
            trend[i] = -1
        else:
            trend[i] = prev_trend

    df["st_trend"] = trend
    df["st_val"] = np.where(trend == 1, final_lower, final_upper)
    df["sig_buy"] = (df["st_trend"] == 1) & (df["st_trend"].shift(1) == -1) & (~df["is_choppy"])
    df["sig_sell"] = (df["st_trend"] == -1) & (df["st_trend"].shift(1) == 1) & (~df["is_choppy"])
    return df

def calc_matrix(df, p):
    # (Leaving your logic shape intact; only making it safe & readable.)
    df = df.copy()
    change = df["close"].diff()
    up, down = change.clip(lower=0), -change.clip(upper=0)

    rs = (rma(up, p["mf_len"]) / (rma(down, p["mf_len"]) + 1e-12))
    rsi = 100 - (100 / (1 + rs))
    rsi_src = rsi - 50  # centered RSI

    mf_vol = df["volume"] / (df["volume"].rolling(p["mf_len"]).mean() + 1e-12)
    df["money_flow"] = (rsi_src * mf_vol).ewm(span=p["mf_smooth"], adjust=False).mean()

    pc = df["close"].diff()
    ds_pc = double_smooth(pc, p["hw_long"], p["hw_short"])
    ds_abs = double_smooth(pc.abs(), p["hw_long"], p["hw_short"])
    df["hyper_wave"] = (100 * (ds_pc / (ds_abs + 1e-12)) / 2).fillna(0)

    s1 = np.sign(df["money_flow"])
    s2 = np.sign(df["hyper_wave"])
    s3 = np.sign(df["hyper_wave"].diff())
    df["matrix_score"] = s1 + s2 + s3
    return df

def calc_rqzo(df, p):
    df = df.copy()
    src = df["close"]
    norm_p = (src - src.rolling(100).min()) / (src.rolling(100).max() - src.rolling(100).min() + 1e-10)
    vel = np.abs(norm_p.diff()).fillna(0)
    c = p["term_vol"] / 100.0
    gamma = 1 / np.sqrt(1 - (np.clip(vel, 0, c * 0.99) / c) ** 2)

    fdi = np.full(len(df), 1.5)
    vals = src.values
    for i in range(p["fdi_len"], len(df)):
        w = vals[i - p["fdi_len"] : i]
        path = np.sum(np.abs(np.diff(w)))
        rng = np.max(w) - np.min(w)
        if path > 0 and rng > 0:
            fdi[i] = 1 + (np.log10(rng / p["fdi_len"]) / np.log10(path / p["fdi_len"]))
    df["fdi"] = fdi

    ret = src.pct_change().fillna(0)
    ent = np.zeros(len(df))
    for i in range(p["ent_len"], len(df)):
        hist, _ = np.histogram(ret[i - p["ent_len"] : i], bins=5)
        probs = hist[hist > 0] / p["ent_len"]
        ent[i] = -np.sum(probs * np.log(probs)) / np.log(5)
    df["entropy"] = ent

    n_eff = np.clip(np.floor(p["zeta_n"] / fdi), 1, 50).astype(int)
    tau = (np.arange(len(df)) % 100) / gamma
    zeta = np.zeros(len(df))
    max_n = int(n_eff.max())

    for n in range(1, max_n + 1):
        term = (n ** -0.5) * np.sin(tau * np.log(n))
        zeta += np.where(n <= n_eff, term, 0)

    df["rqzo"] = zeta * np.exp(-2 * np.abs(df["entropy"] - 0.6)) * 10

    ma = df["rqzo"].rolling(20).mean()
    std = df["rqzo"].rolling(20).std()
    w = (2.5 - df["fdi"]) * std
    df["rqzo_u"] = ma + w
    df["rqzo_l"] = ma - w
    return df

def calc_apex_master(df, p):
    df = df.copy()
    if p["ma_type"] == "HMA":
        df["baseline"] = hma(df["close"], p["len_main"])
    else:
        df["baseline"] = df["close"].rolling(p["len_main"]).mean()

    df["atr"] = rma(df["high"] - df["low"], p["len_main"])
    df["cloud_u"] = df["baseline"] + (df["atr"] * p["mult"])
    df["cloud_l"] = df["baseline"] - (df["atr"] * p["mult"])

    trend = np.zeros(len(df))
    close = df["close"].values
    upper = df["cloud_u"].values
    lower = df["cloud_l"].values
    for i in range(1, len(df)):
        if close[i] > upper[i]:
            trend[i] = 1
        elif close[i] < lower[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    df["apex_trend"] = trend

    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ap.ewm(span=10, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=10, adjust=False).mean()
    ci = (ap - esa) / (0.015 * (d + 1e-12))
    tci = ci.ewm(span=21, adjust=False).mean()

    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = rma(df["high"] - df["low"], 14)
    plus_di = 100 * rma(pd.Series(plus_dm), 14) / (tr + 1e-12)
    minus_di = 100 * rma(pd.Series(minus_dm), 14) / (tr + 1e-12)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
    adx = rma(dx, 14)

    vol_avg = df["volume"].rolling(20).mean()
    vol_ok = df["volume"] > vol_avg

    df["sig_apex_buy"] = (df["apex_trend"] == 1) & (df["apex_trend"].shift(1) != 1) & vol_ok & (tci < 60) & (adx > 20)
    df["sig_apex_sell"] = (df["apex_trend"] == -1) & (df["apex_trend"].shift(1) != -1) & vol_ok & (tci > -60) & (adx > 20)

    df["ph"] = df["high"].rolling(p["liq_len"] * 2 + 1, center=True).max() == df["high"]
    df["pl"] = df["low"].rolling(p["liq_len"] * 2 + 1, center=True).min() == df["low"]

    df["fvg_bull"] = (df["low"] > df["high"].shift(2)) & ((df["low"] - df["high"].shift(2)) > df["atr"] * 0.5)
    df["fvg_bear"] = (df["high"] < df["low"].shift(2)) & ((df["low"].shift(2) - df["high"]) > df["atr"] * 0.5)
    return df

# ==========================================
# 4. DATA ACCESS (CACHE + HEALTH)
# ==========================================
@st.cache_resource
def get_exchange():
    # enableRateLimit improves stability UX (fewer random fetch fails)
    return ccxt.kraken({"enableRateLimit": True})

@st.cache_data(ttl=15)
def get_data(sym, tf, lim):
    ex = get_exchange()
    ohlcv = ex.fetch_ohlcv(sym, tf, limit=lim)
    return pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

# ==========================================
# 5. SETTINGS MODEL (FORM + PRESETS)
# ==========================================
def preset_params(name: str):
    if name == "Scalp":
        return dict(timeframe="1m", limit=400, len_main=34, eff_super=0.72, st_mult=3.2, zeta_n=18)
    if name == "Swing":
        return dict(timeframe="15m", limit=650, len_main=55, eff_super=0.60, st_mult=4.0, zeta_n=25)
    if name == "Long":
        return dict(timeframe="4h", limit=800, len_main=89, eff_super=0.52, st_mult=4.8, zeta_n=34)
    return {}

if "cfg" not in st.session_state:
    st.session_state.cfg = {
        "symbol": "BTC/USD",
        "preset": "Swing",
        "timeframe": "15m",
        "limit": 500,
        "len_main": 55,
        "eff_super": 0.6,
        "st_mult": 4.0,
        "zeta_n": 25,
    }

def settings_panel():
    cfg = st.session_state.cfg
    with st.form("settings_form", border=False):
        st.caption("Terminal Config")

        colA, colB = st.columns([1.1, 0.9])
        with colA:
            symbol = st.text_input("Symbol", cfg["symbol"], help="Exchange symbol (Kraken). Example: BTC/USD")
        with colB:
            preset = st.selectbox("Preset", ["Scalp", "Swing", "Long", "Custom"], index=["Scalp","Swing","Long","Custom"].index(cfg["preset"]))

        if preset != "Custom":
            p = preset_params(preset)
            timeframe = st.selectbox("Timeframe", ['1m','5m','15m','1h','4h','1d'], index=['1m','5m','15m','1h','4h','1d'].index(p["timeframe"]))
            limit = st.slider("Lookback", 200, 1000, p["limit"], help="More candles = slower, but more stable indicators.")
            len_main = st.number_input("Apex Trend Length", 10, 200, p["len_main"])
            eff_super = st.number_input("Vector: Super Thresh", 0.1, 1.0, float(p["eff_super"]))
            st_mult = st.number_input("Dark: Trend Factor", 1.0, 10.0, float(p["st_mult"]))
            zeta_n = st.number_input("Quantum: Harmonics", 5, 100, int(p["zeta_n"]))
        else:
            timeframe = st.selectbox("Timeframe", ['1m','5m','15m','1h','4h','1d'], index=['1m','5m','15m','1h','4h','1d'].index(cfg["timeframe"]))
            limit = st.slider("Lookback", 200, 1000, cfg["limit"])
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                len_main = st.number_input("Apex Trend Length", 10, 200, cfg["len_main"])
                eff_super = st.number_input("Vector: Super Thresh", 0.1, 1.0, float(cfg["eff_super"]))
            with col2:
                st_mult = st.number_input("Dark: Trend Factor", 1.0, 10.0, float(cfg["st_mult"]))
                zeta_n = st.number_input("Quantum: Harmonics", 5, 100, int(cfg["zeta_n"]))

        apply = st.form_submit_button("Apply Settings")

    if apply:
        st.session_state.cfg = {
            "symbol": symbol.strip(),
            "preset": preset,
            "timeframe": timeframe,
            "limit": int(limit),
            "len_main": int(len_main),
            "eff_super": float(eff_super),
            "st_mult": float(st_mult),
            "zeta_n": int(zeta_n),
        }
        st.toast("Settings applied", icon="⚙️")

# Prefer popover; fallback sidebar (clean UX either way)
try:
    with st.popover("⚙️ Settings", use_container_width=True):
        settings_panel()
except Exception:
    with st.sidebar:
        settings_panel()

cfg = st.session_state.cfg

# ==========================================
# 6. TOP COMMAND BAR
# ==========================================
left, right = st.columns([1.2, 0.8], vertical_alignment="center")
with left:
    st.markdown(
        f"""
        <div class="titan-topbar">
          <div class="titan-title">
            <div class="h">⚡ Titan v5.0 — Apex Master Terminal</div>
            <div class="s">{cfg["symbol"]} • {cfg["timeframe"]} • lookback {cfg["limit"]}</div>
          </div>
          <div style="display:flex; gap:10px; align-items:center;">
            <span class="pill"><span class="dot"></span> Live</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    colr1, colr2 = st.columns([1, 1])
    with colr1:
        if st.button("↻ Refresh"):
            get_data.clear()  # clears cached fetches for immediate refresh
            st.rerun()
    with colr2:
        st.download_button(
            "⬇ Snapshot CSV",
            data=b"",  # replaced later once df exists
            file_name="titan_snapshot.csv",
            disabled=True,
            use_container_width=True,
        )

# ==========================================
# 7. RUN PIPELINE (WITH GOOD UX STATES)
# ==========================================
params = {
    "vec_len": 14, "flux_sm": 5, "eff_super": cfg["eff_super"], "eff_resist": 0.3, "strictness": 1.0, "vol_norm": 55,
    "chop_len": 14, "chop_thresh": 60, "st_len": 10, "st_mult": cfg["st_mult"],
    "mf_len": 14, "mf_smooth": 3, "hw_long": 25, "hw_short": 13,
    "term_vol": 5.0, "fdi_len": 20, "ent_len": 20, "zeta_n": cfg["zeta_n"],
    "ma_type": "HMA", "len_main": cfg["len_main"], "mult": 1.5, "liq_len": 10
}

with st.spinner("Connecting to Kraken + compiling signals…"):
    try:
        df = get_data(cfg["symbol"], cfg["timeframe"], cfg["limit"])
    except Exception as e:
        df = pd.DataFrame()
        st.error(f"Data fetch failed: {e}")

if df.empty:
    st.info("System Initializing… Try Refresh. If symbol/timeframe is invalid, update settings.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df = calc_apex_vector(df, params)
df = calc_dark_vector(df, params)
df = calc_matrix(df, params)
df = calc_rqzo(df, params)
df = calc_apex_master(df, params)

last = df.iloc[-1]
last_update = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# Activate snapshot download now that df exists
st.download_button(
    "⬇ Snapshot CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"titan_{cfg['symbol'].replace('/','-')}_{cfg['timeframe']}.csv",
    use_container_width=True,
)

st.caption(f"Last update: {last_update}")

# ==========================================
# 8. READABLE STATE LABELS
# ==========================================
apex_code = last["apex_state"]
if apex_code == 2:
    s_txt, s_col, accent = "SUPER (BULL)", "c-bull", "accent-bull"
elif apex_code == -2:
    s_txt, s_col, accent = "SUPER (BEAR)", "c-bear", "accent-bear"
elif apex_code == 0:
    s_txt, s_col, accent = "RESISTIVE", "c-cyan", "accent-chop"
else:
    s_txt, s_col, accent = "HIGH HEAT", "c-heat", "accent-heat"

master_trend = "BULLISH" if last["apex_trend"] == 1 else "BEARISH"
master_col = "c-bull" if last["apex_trend"] == 1 else "c-bear"
mat_score = last["matrix_score"]
mat_txt = "STRONG BUY" if mat_score > 1 else ("STRONG SELL" if mat_score < -1 else "NEUTRAL")
mat_col = "c-bull" if mat_score > 1 else ("c-bear" if mat_score < -1 else "c-cyan")

# ==========================================
# 9. TABS
# ==========================================
t1, t2, t3, t4, t5 = st.tabs(["⚡ Terminal", "🌊 Liquidity/SMC", "💠 Matrix", "⚛️ Quantum", "🤖 AI Analyst"])

# --- TAB 1
with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="titan-card {accent}"><div class="k">Apex Vector</div><div class="v {s_col}">{s_txt}</div><div class="sub">Flux: {last['apex_flux']:.2f}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="titan-card {accent}"><div class="k">Apex Master Trend</div><div class="v {master_col}">{master_trend}</div><div class="sub">Baseline: {last['baseline']:.2f}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="titan-card"><div class="k">Matrix</div><div class="v {mat_col}">{mat_txt}</div><div class="sub">Score: {int(mat_score)}</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="titan-card"><div class="k">Quantum</div><div class="v">{last['rqzo']:.2f}</div><div class="sub">Entropy: {last['entropy']:.2f}</div></div>""", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
    ))

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["cloud_u"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["cloud_l"], fill="tonexty",
        fillcolor="rgba(0, 230, 118, 0.12)" if last["apex_trend"] == 1 else "rgba(255, 23, 68, 0.12)",
        line=dict(width=0), name="Trend Cloud"
    ))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["baseline"], line=dict(color="gray", width=1, dash="dot"), name="Baseline"))

    buys = df[df["sig_apex_buy"]]
    sells = df[df["sig_apex_sell"]]
    fig.add_trace(go.Scatter(
        x=buys["timestamp"], y=buys["low"] * 0.999, mode="markers+text",
        text="BUY", textposition="bottom center",
        marker=dict(symbol="triangle-up", size=12, color="#00E676"),
        name="Apex Buy"
    ))
    fig.add_trace(go.Scatter(
        x=sells["timestamp"], y=sells["high"] * 1.001, mode="markers+text",
        text="SELL", textposition="top center",
        marker=dict(symbol="triangle-down", size=12, color="#FF1744"),
        name="Apex Sell"
    ))

    fig.update_layout(
        height=620,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(rangeslider=dict(visible=False)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # X-Factor: Signal Timeline
    st.markdown("#### 🧾 Signal Timeline")
    events = []
    for _, r in df.tail(220).iterrows():
        if r.get("sig_apex_buy", False):
            events.append((r["timestamp"], "Apex BUY", f"Flux {r['apex_flux']:.2f} • rqzo {r['rqzo']:.2f}"))
        if r.get("sig_apex_sell", False):
            events.append((r["timestamp"], "Apex SELL", f"Flux {r['apex_flux']:.2f} • rqzo {r['rqzo']:.2f}"))
        if r.get("sig_buy", False):
            events.append((r["timestamp"], "Dark BUY", f"Chop {r['chop_idx']:.1f}"))
        if r.get("sig_sell", False):
            events.append((r["timestamp"], "Dark SELL", f"Chop {r['chop_idx']:.1f}"))
    events = sorted(events, key=lambda x: x[0], reverse=True)[:14]

    if events:
        ev_df = pd.DataFrame(events, columns=["Time", "Event", "Context"])
        st.dataframe(ev_df, use_container_width=True, height=280)
    else:
        st.caption("No recent signals in the last window.")

# --- TAB 2
with t2:
    fig_smc = go.Figure()
    fig_smc.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
    ))

    bull_fvg = df[df["fvg_bull"]]
    bear_fvg = df[df["fvg_bear"]]
    fig_smc.add_trace(go.Scatter(
        x=bull_fvg["timestamp"], y=bull_fvg["low"], mode="markers",
        marker=dict(symbol="square", color="rgba(0,230,118,0.35)", size=8),
        name="Bull FVG"
    ))
    fig_smc.add_trace(go.Scatter(
        x=bear_fvg["timestamp"], y=bear_fvg["high"], mode="markers",
        marker=dict(symbol="square", color="rgba(255,23,68,0.35)", size=8),
        name="Bear FVG"
    ))

    p_highs = df[df["ph"]]
    p_lows = df[df["pl"]]
    fig_smc.add_trace(go.Scatter(x=p_highs["timestamp"], y=p_highs["high"], mode="markers",
                                 marker=dict(symbol="circle-open", color="#FF1744", size=7), name="Supply Liq"))
    fig_smc.add_trace(go.Scatter(x=p_lows["timestamp"], y=p_lows["low"], mode="markers",
                                 marker=dict(symbol="circle-open", color="#00E676", size=7), name="Demand Liq"))

    fig_smc.update_layout(
        height=620,
        template="plotly_dark",
        title="SMC: FVGs & Pivot Liquidity",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False)),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_smc, use_container_width=True)

# --- TAB 3
with t3:
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["money_flow"], fill="tozeroy", name="Money Flow"), row=1, col=1)
    fig2.add_hline(y=0, line_dash="dot", row=1, col=1)
    fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["hyper_wave"], name="Hyper Wave"), row=2, col=1)
    fig2.update_layout(
        height=620,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(rangeslider=dict(visible=False)),
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- TAB 4
with t4:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo_u"], line=dict(width=0), showlegend=False))
    fig3.add_trace(go.Scatter(
        x=df["timestamp"], y=df["rqzo_l"], fill="tonexty",
        fillcolor="rgba(179, 136, 255, 0.16)", line=dict(width=0), name="Vol Bands"
    ))
    fig3.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], name="RQZO"))

    chaos = df[df["entropy"] > 0.8]
    fig3.add_trace(go.Scatter(x=chaos["timestamp"], y=chaos["rqzo"], mode="markers", name="Chaos Zone"))

    fig3.update_layout(
        height=560,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(rangeslider=dict(visible=False)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- TAB 5
with t5:
    st.markdown("#### 🤖 AI Analyst")
    st.caption("Generates a compact trade read: **BIAS / SETUP / RISK / TACTIC**")

    try:
        ai_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        ai_key = ""

    with st.expander("🔐 API Key", expanded=False):
        user_key = st.text_input("OpenAI Key", value=ai_key, type="password")
        st.caption("Tip: Put it in Streamlit secrets for cleaner UX.")

    summ = f"""
Sym: {cfg['symbol']}
Apex Vector: {s_txt} (Flux {last['apex_flux']:.2f})
Master Trend: {master_trend} (Baseline {last['baseline']:.2f})
Matrix: {mat_txt} (Score {mat_score})
RQZO: {last['rqzo']:.2f}
Entropy: {last['entropy']:.2f}
SMC: FVG Bull {bool(last['fvg_bull'])}, FVG Bear {bool(last['fvg_bear'])}
"""

    st.text_area("Context sent to model", value=summ.strip(), height=160)

    if st.button("Generate Quant Report"):
        if not user_key:
            st.error("No API key provided.")
        else:
            with st.spinner("Synthesizing…"):
                client = OpenAI(api_key=user_key)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": f"Analyze this quant data for a crypto trade:\n{summ}\nOutput: BIAS, SETUP, RISK, TACTIC. <50 words."
                    }],
                )
            st.markdown(f"<div class='ai-box'>{resp.choices[0].message.content}</div>", unsafe_allow_html=True)
