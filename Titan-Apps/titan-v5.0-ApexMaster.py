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
.c-div-bull { color: #00B0FF; }
.c-div-bear { color: #FF4081; }

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
# 3. MATH ENGINE & HELPERS
# ==========================================
def rma(series, length):
    return series.ewm(alpha=1 / length, adjust=False).mean()

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

def vwma(series, volume, length):
    return (series * volume).rolling(window=length).mean() / volume.rolling(window=length).mean()

def apply_smoothing(series, type_str, length, vol_series=None):
    if type_str == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    elif type_str == "SMA":
        return series.rolling(window=length).mean()
    elif type_str == "RMA":
        return rma(series, length)
    elif type_str == "WMA":
        return wma(series, length)
    elif type_str == "VWMA" and vol_series is not None:
        return vwma(series, vol_series, length)
    else:
        return series.ewm(span=length, adjust=False).mean() # Fallback EMA

def double_smooth(src, long_len, short_len):
    smooth1 = src.ewm(span=long_len, adjust=False).mean()
    smooth2 = smooth1.ewm(span=short_len, adjust=False).mean()
    return smooth2

# ==========================================
# 4. APEX VECTOR v4.1 LOGIC
# ==========================================
def calc_apex_vector_v4(df, p):
    df = df.copy()
    
    # --- 1. Geometric Efficiency ---
    df["range_abs"] = df["high"] - df["low"]
    df["body_abs"] = (df["close"] - df["open"]).abs()
    # Avoid div by zero
    df["raw_eff"] = np.where(df["range_abs"] == 0, 0.0, df["body_abs"] / df["range_abs"])
    df["efficiency"] = df["raw_eff"].ewm(span=p["len_vec"], adjust=False).mean()

    # --- 2. Volume Flux ---
    df["vol_avg"] = df["volume"].rolling(p["vol_norm"]).mean()
    if p["use_vol"]:
        df["vol_fact"] = np.where(df["vol_avg"] == 0, 1.0, df["volume"] / df["vol_avg"])
    else:
        df["vol_fact"] = 1.0

    # --- 3. The Apex Vector ---
    df["direction"] = np.sign(df["close"] - df["open"])
    df["vector_raw"] = df["direction"] * df["efficiency"] * df["vol_fact"]

    # --- 4. Smoothing Kernel ---
    df["flux"] = apply_smoothing(df["vector_raw"], p["sm_type"], p["len_sm"], df["volume"])

    # --- 5. Logic & State Machine ---
    th_super = p["eff_super"] * p["strictness"]
    th_resist = p["eff_resist"] * p["strictness"]

    conditions = [
        (df["flux"] > th_super),           # Super Bull
        (df["flux"] < -th_super),          # Super Bear
        (df["flux"].abs() < th_resist)     # Resistive
    ]
    # 2=Bull, -2=Bear, 0=Resist, 1=Heat (Default/Else)
    choices = [2, -2, 0] 
    df["apex_state"] = np.select(conditions, choices, default=1)

    # --- 6. Divergence Engine (Fixed Loop) ---
    look = p["div_look"]
    
    # Initialize output columns
    df["div_bull_reg"] = False
    df["div_bull_hid"] = False
    df["div_bear_reg"] = False
    df["div_bear_hid"] = False
    
    flux = df["flux"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    
    prev_pl_flux = np.nan
    prev_pl_price = np.nan
    prev_ph_flux = np.nan
    prev_ph_price = np.nan
    
    div_bull_reg_arr = np.zeros(n, dtype=bool)
    div_bull_hid_arr = np.zeros(n, dtype=bool)
    div_bear_reg_arr = np.zeros(n, dtype=bool)
    div_bear_hid_arr = np.zeros(n, dtype=bool)

    # Simple rolling window check for pivots (center of window is max/min)
    for i in range(look, n - look):
        # 1. Pivot Low Detection
        window_fl = flux[i-look : i+look+1]
        if flux[i] == np.min(window_fl):
            # Pivot Low found at 'i'
            pl = flux[i]
            price_at_pivot = lows[i] 
            
            if not np.isnan(prev_pl_flux):
                # Regular Bull
                if p["show_reg"] and (price_at_pivot < prev_pl_price) and (pl > prev_pl_flux):
                    div_bull_reg_arr[i] = True
                # Hidden Bull
                if p["show_hid"] and (price_at_pivot > prev_pl_price) and (pl < prev_pl_flux):
                    div_bull_hid_arr[i] = True
            
            prev_pl_flux = pl
            prev_pl_price = price_at_pivot

        # 2. Pivot High Detection
        if flux[i] == np.max(window_fl):
            ph = flux[i]
            price_at_pivot = highs[i]
            
            if not np.isnan(prev_ph_flux):
                # Regular Bear
                if p["show_reg"] and (price_at_pivot > prev_ph_price) and (ph < prev_ph_flux):
                    div_bear_reg_arr[i] = True
                # Hidden Bear
                if p["show_hid"] and (price_at_pivot < prev_ph_price) and (ph > prev_ph_flux):
                    div_bear_hid_arr[i] = True
            
            prev_ph_flux = ph
            prev_ph_price = price_at_pivot

    df["div_bull_reg"] = div_bull_reg_arr
    df["div_bull_hid"] = div_bull_hid_arr
    df["div_bear_reg"] = div_bear_reg_arr
    df["div_bear_hid"] = div_bear_hid_arr

    # Helper for Alert Text
    df["div_status"] = np.select(
        [df["div_bull_reg"], df["div_bear_reg"], df["div_bull_hid"], df["div_bear_hid"]],
        ["Bull Reg", "Bear Reg", "Bull Hid", "Bear Hid"],
        default="None"
    )

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
    df = df.copy()
    change = df["close"].diff()
    up, down = change.clip(lower=0), -change.clip(upper=0)
    rs = (rma(up, p["mf_len"]) / (rma(down, p["mf_len"]) + 1e-12))
    rsi = 100 - (100 / (1 + rs))
    rsi_src = rsi - 50 
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
# 5. DATA ACCESS
# ==========================================
@st.cache_resource
def get_exchange():
    return ccxt.kraken({"enableRateLimit": True})

@st.cache_data(ttl=15)
def get_data(sym, tf, lim):
    ex = get_exchange()
    ohlcv = ex.fetch_ohlcv(sym, tf, limit=lim)
    return pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

# ==========================================
# 6. SETTINGS MODEL (WITH FIX FOR KEYERROR)
# ==========================================
# Define ALL required keys and their defaults here:
default_cfg = {
    "symbol": "BTC/USD", 
    "preset": "Swing", 
    "timeframe": "15m", 
    "limit": 600,
    "len_main": 55, 
    "st_mult": 4.0, 
    "zeta_n": 25,
    # APEX VECTOR v4.1 New Keys
    "eff_super": 0.60, 
    "eff_resist": 0.30, 
    "len_vec": 14, 
    "sm_type": "EMA", 
    "len_sm": 5, 
    "strictness": 1.0, 
    "div_look": 5,
    "show_reg": True, 
    "show_hid": False
}

# 1. Initialize if not present
if "cfg" not in st.session_state:
    st.session_state.cfg = default_cfg.copy()
else:
    # 2. MIGRATION/MERGE: If 'cfg' exists but lacks new keys (e.g. from prev run), add them.
    for k, v in default_cfg.items():
        if k not in st.session_state.cfg:
            st.session_state.cfg[k] = v

def settings_panel():
    cfg = st.session_state.cfg
    with st.form("settings_form", border=False):
        st.caption("Terminal Config")
        colA, colB = st.columns([1.1, 0.9])
        with colA:
            symbol = st.text_input("Symbol", cfg["symbol"])
        with colB:
            preset = st.selectbox("Preset", ["Scalp", "Swing", "Custom"], index=["Scalp", "Swing", "Custom"].index(cfg["preset"]))
        
        # Default Logic
        if preset == "Scalp":
            p_def = {"timeframe": "1m", "limit": 400, "len_main": 34, "eff_super": 0.72}
        elif preset == "Swing":
            p_def = {"timeframe": "15m", "limit": 650, "len_main": 55, "eff_super": 0.60}
        else:
            p_def = cfg

        if preset != "Custom":
            timeframe = st.selectbox("Timeframe", ['1m','5m','15m','1h','4h','1d'], index=['1m','5m','15m','1h','4h','1d'].index(p_def["timeframe"]))
            limit = st.slider("Lookback", 200, 1000, p_def["limit"])
        else:
            timeframe = st.selectbox("Timeframe", ['1m','5m','15m','1h','4h','1d'], index=['1m','5m','15m','1h','4h','1d'].index(cfg["timeframe"]))
            limit = st.slider("Lookback", 200, 1000, cfg["limit"])
        
        st.divider()
        st.markdown("**Apex Vector Engine**")
        c1, c2 = st.columns(2)
        with c1:
            eff_super = st.number_input("Super Thresh", 0.1, 1.0, float(p_def["eff_super"] if preset != "Custom" else cfg["eff_super"]))
            eff_resist = st.number_input("Resist Thresh", 0.0, 0.5, float(cfg["eff_resist"]))
            sm_type = st.selectbox("Smoothing", ["EMA","SMA","RMA","WMA","VWMA"], index=["EMA","SMA","RMA","WMA","VWMA"].index(cfg["sm_type"]))
        with c2:
            len_vec = st.number_input("Vector Len", 2, 50, int(cfg["len_vec"]))
            len_sm = st.number_input("Smooth Len", 1, 20, int(cfg["len_sm"]))
            div_look = st.number_input("Div Lookback", 1, 20, int(cfg["div_look"]))
        
        strictness = st.slider("Strictness", 0.5, 2.0, float(cfg["strictness"]))
        
        st.divider()
        with st.expander("Other Params"):
            len_main = st.number_input("Master Trend Len", 10, 200, int(p_def["len_main"] if preset!="Custom" else cfg["len_main"]))
            st_mult = st.number_input("Dark Trend Mult", 1.0, 10.0, float(cfg["st_mult"]))
            zeta_n = st.number_input("Quantum N", 5, 100, int(cfg["zeta_n"]))
            show_reg = st.checkbox("Show Reg Divs", cfg["show_reg"])
            show_hid = st.checkbox("Show Hidden Divs", cfg["show_hid"])

        apply = st.form_submit_button("Apply Settings")

    if apply:
        st.session_state.cfg.update({
            "symbol": symbol.strip(), "preset": preset, "timeframe": timeframe, "limit": limit,
            "len_main": len_main, "st_mult": st_mult, "zeta_n": zeta_n,
            "eff_super": eff_super, "eff_resist": eff_resist, "len_vec": len_vec,
            "sm_type": sm_type, "len_sm": len_sm, "strictness": strictness,
            "div_look": div_look, "show_reg": show_reg, "show_hid": show_hid
        })
        st.toast("Settings updated", icon="⚡")

# Safe Check for Popover (Avoids Streamlit Form Duplicate Key Error)
if hasattr(st, "popover"):
    with st.popover("⚙️ Settings", use_container_width=True):
        settings_panel()
else:
    with st.sidebar:
        settings_panel()

cfg = st.session_state.cfg

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
params = {
    "vol_norm": 55, "chop_len": 14, "chop_thresh": 60, "st_len": 10, 
    "mf_len": 14, "mf_smooth": 3, "hw_long": 25, "hw_short": 13,
    "term_vol": 5.0, "fdi_len": 20, "ent_len": 20, "ma_type": "HMA", "mult": 1.5, "liq_len": 10,
    "use_vol": True,
    # User Controlled
    "eff_super": cfg["eff_super"], "eff_resist": cfg["eff_resist"], 
    "len_vec": cfg["len_vec"], "sm_type": cfg["sm_type"], "len_sm": cfg["len_sm"],
    "strictness": cfg["strictness"], "div_look": cfg["div_look"],
    "show_reg": cfg["show_reg"], "show_hid": cfg["show_hid"],
    "st_mult": cfg["st_mult"], "zeta_n": cfg["zeta_n"], "len_main": cfg["len_main"]
}

# Top Bar
left, right = st.columns([1.2, 0.8], vertical_alignment="center")
with left:
    st.markdown(f"""
        <div class="titan-topbar">
          <div class="titan-title"><div class="h">⚡ Titan v5.0 — Apex Master Terminal</div><div class="s">{cfg["symbol"]} • {cfg["timeframe"]}</div></div>
          <div><span class="pill"><span class="dot"></span> Live</span></div>
        </div>
        """, unsafe_allow_html=True)
with right:
    c_r1, c_r2 = st.columns(2)
    with c_r1:
        if st.button("↻ Refresh"):
            get_data.clear()
            st.rerun()

# Fetch & Calc
with st.spinner("Processing Signals..."):
    try:
        df = get_data(cfg["symbol"], cfg["timeframe"], cfg["limit"])
    except Exception as e:
        df = pd.DataFrame()
        st.error(f"Error: {e}")

if df.empty:
    st.warning("No data. Check symbol.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df = calc_apex_vector_v4(df, params)
df = calc_dark_vector(df, params)
df = calc_matrix(df, params)
df = calc_rqzo(df, params)
df = calc_apex_master(df, params)

last = df.iloc[-1]

# Snapshot Button
with right:
    with c_r2:
        st.download_button("⬇ CSV", df.to_csv(index=False).encode("utf-8"), "titan_data.csv", use_container_width=True)

# ==========================================
# 8. VISUALIZATION
# ==========================================
# Determine Colors & Text for HUD
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

# Tabs
t1, t2, t3, t4, t5 = st.tabs(["⚡ Terminal", "🌊 Liquidity/SMC", "💠 Matrix", "⚛️ Quantum", "🤖 AI Analyst"])

with t1:
    # 1. Metric Cards (HUD)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="titan-card {accent}"><div class="k">Apex Vector State</div><div class="v {s_col}">{s_txt}</div><div class="sub">Flux: {last['flux']:.2f}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="titan-card {accent}"><div class="k">Apex Efficiency</div><div class="v">{last['efficiency']*100:.0f}%</div><div class="sub">Div Status: <span class="{('c-div-bull' if 'Bull' in last['div_status'] else ('c-div-bear' if 'Bear' in last['div_status'] else ''))}">{last['div_status']}</span></div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="titan-card"><div class="k">Master Trend</div><div class="v {master_col}">{master_trend}</div><div class="sub">Baseline: {last['baseline']:.2f}</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="titan-card"><div class="k">Quantum</div><div class="v">{last['rqzo']:.2f}</div><div class="sub">Entropy: {last['entropy']:.2f}</div></div>""", unsafe_allow_html=True)

    # 2. Dual-Pane Chart (Price + Vector)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # -- Pane 1: Price & Master Trend --
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["cloud_u"], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["cloud_l"], fill="tonexty", 
                             fillcolor="rgba(0, 230, 118, 0.1)" if last["apex_trend"]==1 else "rgba(255, 23, 68, 0.1)", 
                             line=dict(width=0), name="Cloud"), row=1, col=1)
    
    # -- Pane 2: Apex Vector Flux --
    colors = []
    for s in df["apex_state"]:
        if s == 2: colors.append("#00E676")   # Bull
        elif s == -2: colors.append("#FF1744") # Bear
        elif s == 0: colors.append("#546E7A")  # Resist
        else: colors.append("#FFD600")         # Heat
    
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=colors, name="Flux Vector"), row=2, col=1)
    
    # Threshold Lines
    th_s = cfg["eff_super"] * cfg["strictness"]
    fig.add_hline(y=th_s, line_dash="dot", line_color="#00E676", row=2, col=1, opacity=0.5)
    fig.add_hline(y=-th_s, line_dash="dot", line_color="#FF1744", row=2, col=1, opacity=0.5)

    # Divergence Markers
    bull_reg = df[df["div_bull_reg"]]
    bear_reg = df[df["div_bear_reg"]]
    fig.add_trace(go.Scatter(x=bull_reg["timestamp"], y=bull_reg["flux"], mode="markers", marker=dict(symbol="circle", size=8, color="#00B0FF"), name="Bull Div"), row=2, col=1)
    fig.add_trace(go.Scatter(x=bear_reg["timestamp"], y=bear_reg["flux"], mode="markers", marker=dict(symbol="circle", size=8, color="#FF4081"), name="Bear Div"), row=2, col=1)

    fig.update_layout(height=700, template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0), 
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                      hovermode="x unified", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 3. Signal Event Log
    st.markdown("#### 🧾 Signal Timeline")
    events = []
    for _, r in df.tail(300).iterrows():
        if r["div_bull_reg"]: events.append((r["timestamp"], "Apex Bull Div", f"Flux {r['flux']:.2f}"))
        if r["div_bear_reg"]: events.append((r["timestamp"], "Apex Bear Div", f"Flux {r['flux']:.2f}"))
        if r["sig_apex_buy"]: events.append((r["timestamp"], "Master BUY", f"Baseline {r['baseline']:.2f}"))
        if r["sig_apex_sell"]: events.append((r["timestamp"], "Master SELL", f"Baseline {r['baseline']:.2f}"))
    
    if events:
        ev_df = pd.DataFrame(events, columns=["Time", "Event", "Context"]).sort_values("Time", ascending=False).head(10)
        st.dataframe(ev_df, use_container_width=True, hide_index=True)

with t2:
    # Liquidity / SMC View
    fig_smc = go.Figure()
    fig_smc.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]))
    # FVGs
    bf = df[df["fvg_bull"]]
    brf = df[df["fvg_bear"]]
    fig_smc.add_trace(go.Scatter(x=bf["timestamp"], y=bf["low"], mode="markers", marker=dict(symbol="square", color="rgba(0,230,118,0.5)", size=6), name="Bull FVG"))
    fig_smc.add_trace(go.Scatter(x=brf["timestamp"], y=brf["high"], mode="markers", marker=dict(symbol="square", color="rgba(255,23,68,0.5)", size=6), name="Bear FVG"))
    
    fig_smc.update_layout(height=600, template="plotly_dark", margin=dict(t=30,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_smc, use_container_width=True)

with t3:
    # Matrix View
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["money_flow"], fill="tozeroy", name="Money Flow"), row=1, col=1)
    fig2.add_hline(y=0, line_dash="dot", row=1, col=1)
    fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["hyper_wave"], name="Hyper Wave"), row=2, col=1)
    fig2.update_layout(height=600, template="plotly_dark", margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig2, use_container_width=True)

with t4:
    # Quantum View
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo_u"], line=dict(width=0), showlegend=False))
    fig3.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo_l"], fill="tonexty", fillcolor="rgba(179, 136, 255, 0.16)", line=dict(width=0), name="Vol Bands"))
    fig3.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], name="RQZO", line=dict(color="#B388FF")))
    fig3.update_layout(height=600, template="plotly_dark", margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig3, use_container_width=True)

with t5:
    st.markdown("#### 🤖 AI Analyst")
    ai_key = st.secrets.get("OPENAI_API_KEY", "")
    with st.expander("API Key"):
        user_key = st.text_input("OpenAI Key", value=ai_key, type="password")
    
    prompt_txt = f"""
    Symbol: {cfg['symbol']}
    Apex State: {s_txt} (Flux {last['flux']:.2f})
    Divergence: {last['div_status']}
    Master Trend: {master_trend}
    RQZO: {last['rqzo']:.2f} / Entropy: {last['entropy']:.2f}
    """
    st.text_area("Data Context", prompt_txt, height=140)
    
    if st.button("Generate Strategy"):
        if not user_key:
            st.error("Missing Key")
        else:
            with st.spinner("Analyzing Physics..."):
                try:
                    client = OpenAI(api_key=user_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"user", "content":f"Act as a Quant Trader. Analyze:\n{prompt_txt}\nProvide concise: BIAS, SETUP, RISK, TACTIC."}]
                    )
                    st.markdown(f"<div class='ai-box'>{resp.choices[0].message.content}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(str(e))
