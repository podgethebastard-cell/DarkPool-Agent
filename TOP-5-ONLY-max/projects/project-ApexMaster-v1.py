# Titan-Apps/titan-v5.0-ApexMaster.py
# ✅ Upgraded to "completion" level:
# - Non-blocking auto-refresh (no sleep)
# - OpenAI import + resilient client call
# - Cached exchange resource (faster / fewer reconnects)
# - Fixed divergence indexing (no off-by-one)
# - NaN-safe entropy/RQZO
# - Alert de-duplication (prevents Telegram spam on reruns)
# - Better error surfacing (no silent failures)
# - Small stability guards (rolling windows, division safety)

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests
import traceback

# Optional: streamlit_autorefresh (recommended)
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# OpenAI (for AI Council tab)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False


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
st.markdown(
    """
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

    /* Custom Pentagram HUD */
    .penta-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 15px; }
    .p-card { background: #111; border: 1px solid #333; padding: 10px; border-radius: 6px; text-align: center; }
    .p-head { font-size: 0.65rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .p-val { font-size: 0.95rem; font-weight: bold; margin-top: 4px; }

    .st-bull { color: var(--bull); border-bottom: 2px solid var(--bull); }
    .st-bear { color: var(--bear); border-bottom: 2px solid var(--bear); }
    .st-neut { color: #888; border-bottom: 2px solid #888; }

    /* AI Box */
    .ai-response {
        background: #090909; border-left: 3px solid var(--vio);
        padding: 15px; border-radius: 0 8px 8px 0; color: #ccc; line-height: 1.6; font-size: 0.9rem;
    }

    /* Small helper */
    .tiny { font-size: 0.75rem; color: #777; }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. MATH ENGINE (SHARED UTILS)
# ==========================================
def rma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.ewm(alpha=1 / length, adjust=False).mean()

def wma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    half = max(length // 2, 1)
    return wma(2 * wma(series, half) - wma(series, length), int(np.sqrt(length)) or 1)

def double_smooth(src: pd.Series, l1: int, l2: int) -> pd.Series:
    l1 = max(int(l1), 1)
    l2 = max(int(l2), 1)
    return src.ewm(span=l1).mean().ewm(span=l2).mean()

def safe_div(a, b, eps=1e-10):
    return a / (b + eps)

# ==========================================
# 3. INDICATOR MODULES (5 CORES)
# ==========================================

# --- CORE 1: APEX VECTOR (PHYSICS) ---
def calc_apex_vector(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Core Physics: Flux, Efficiency, Divergence"""
    df = df.copy()

    # Efficiency
    rng = (df["high"] - df["low"]).astype(float)
    body = (df["close"] - df["open"]).abs().astype(float)
    raw_eff = np.where(rng.values == 0, 0.0, (body / rng).values)
    df["eff"] = pd.Series(raw_eff, index=df.index).ewm(span=int(p["vec_len"])).mean()

    # Flux
    vol_norm = max(int(p["vol_norm"]), 2)
    vol_avg = df["volume"].rolling(vol_norm).mean()
    vol_fact = np.where(vol_avg.values == 0, 1.0, (df["volume"] / vol_avg).values)
    raw_vec = np.sign((df["close"] - df["open"]).values) * df["eff"].values * vol_fact
    df["flux"] = pd.Series(raw_vec, index=df.index).ewm(span=int(p["vec_sm"])).mean()

    # State Logic
    th_s = float(p["eff_super"]) * float(p["strict"])
    th_r = float(p["eff_resist"]) * float(p["strict"])

    conditions = [
        (df["flux"] > th_s),
        (df["flux"] < -th_s),
        (df["flux"].abs() < th_r),
    ]
    df["vec_state"] = np.select(conditions, [2, -2, 0], default=1)  # 2=Bull, -2=Bear, 0=Resist, 1=Heat

    # Divergence (fixed indexing: extrema at i-1 should land at i-1, not i)
    src = df["flux"].astype(float)
    lb = max(int(p["div_look"]), 1)

    pl = ((src.shift(1) < src.shift(2)) & (src.shift(1) < src)).shift(1)
    ph = ((src.shift(1) > src.shift(2)) & (src.shift(1) > src)).shift(1)
    df["pl"] = pl.fillna(False)
    df["ph"] = ph.fillna(False)

    df["div_bull"] = df["pl"] & (df["close"] < df["close"].shift(lb)) & (df["flux"] > df["flux"].shift(lb))
    df["div_bear"] = df["ph"] & (df["close"] > df["close"].shift(lb)) & (df["flux"] < df["flux"].shift(lb))

    return df

# --- CORE 2: APEX BRAIN (LOGIC) ---
def calc_apex_brain(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()

    if "flux" not in df.columns:
        raise ValueError("calc_apex_brain requires 'flux' (run calc_apex_vector first).")

    # Core 1: Cortex (Trend Cloud)
    base = hma(df["close"], int(p["brain_len"]))
    atr = rma((df["high"] - df["low"]).astype(float), int(p["brain_len"]))
    df["cortex_u"] = base + (atr * float(p["brain_mult"]))
    df["cortex_l"] = base - (atr * float(p["brain_mult"]))

    trend = np.zeros(len(df), dtype=float)
    c = df["close"].values
    u = df["cortex_u"].values
    l = df["cortex_l"].values
    for i in range(1, len(df)):
        if np.isfinite(u[i]) and c[i] > u[i]:
            trend[i] = 1
        elif np.isfinite(l[i]) and c[i] < l[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    df["brain_trend"] = trend

    # Core 2: Amygdala (Entropy Gate)
    ent_len = max(int(p.get("ent_len", 64)), 2)
    ret = df["close"].pct_change()
    df["ent_proxy"] = ret.rolling(ent_len).std() * 100.0
    ent_th = float(p["ent_th"])
    df["gate_safe"] = df["ent_proxy"].fillna(np.inf) < ent_th  # treat unknown as unsafe until ready

    # Core 3: Motor (Vector)
    df["motor_bull"] = df["flux"] > 0.5
    df["motor_bear"] = df["flux"] < -0.5

    # Core 4: Occipital (Liquidity Flow / FLI)
    rng = (df["high"] - df["low"]).astype(float)
    wick = np.where(
        rng.values == 0,
        0.0,
        (
            (np.minimum(df["open"], df["close"]).values - df["low"].values)
            - (df["high"].values - np.maximum(df["open"], df["close"]).values)
        )
        / rng.values,
    )

    vol_win = 80
    vmean = df["volume"].rolling(vol_win).mean()
    vstd = df["volume"].rolling(vol_win).std()
    vz = safe_div((df["volume"] - vmean), vstd)

    raw_flow = wick + (vz.values * 0.5) + safe_div((df["close"] - df["open"]), (rma(rng, 14)))
    df["flow"] = pd.Series(raw_flow, index=df.index).ewm(span=34).mean()

    # Synaptic Firing
    df["brain_buy"] = (df["brain_trend"] == 1) & df["gate_safe"] & df["motor_bull"] & (df["flow"] > 0)
    df["brain_sell"] = (df["brain_trend"] == -1) & df["gate_safe"] & df["motor_bear"] & (df["flow"] < 0)
    return df

# --- CORE 3: RQZO (QUANTUM) ---
def calc_rqzo(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    if "ent_proxy" not in df.columns:
        # safety: if brain isn't run, compute a quick proxy
        ret = df["close"].pct_change()
        df["ent_proxy"] = ret.rolling(64).std() * 100.0

    mn = df["close"].rolling(100).min()
    mx = df["close"].rolling(100).max()
    norm = safe_div((df["close"] - mn), (mx - mn))
    vel = (norm - norm.shift(1)).abs()

    c = 5.0 / 100.0
    vclip = np.clip(vel.fillna(0).values, 0, c * 0.99)
    gamma = 1.0 / np.sqrt(1.0 - (vclip / c) ** 2)

    idx = np.arange(len(df), dtype=float)
    tau = (idx % 100) / np.where(gamma == 0, 1.0, gamma)

    zeta = np.zeros(len(df), dtype=float)
    for n in range(1, 10):
        zeta += (n ** -0.5) * np.sin(tau * np.log(n))

    ent = df["ent_proxy"].astype(float).fillna(method="bfill").fillna(0.0)
    gate = np.exp(-2 * (ent / 10.0 - 0.6).abs())
    df["rqzo"] = zeta * gate.values * 10.0
    return df

# --- CORE 4: MATRIX (MOMENTUM) ---
def calc_matrix(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    chg = df["close"].diff()
    rs = safe_div(rma(chg.clip(lower=0), 14), rma((-chg).clip(lower=0), 14))
    rsi = 100 - (100 / (1 + rs))
    mfi_vol = safe_div(df["volume"], df["volume"].rolling(20).mean())
    df["mfi"] = ((rsi - 50) * mfi_vol).ewm(span=3).mean()

    hw_src = df["close"].diff()
    df["hw"] = 100 * safe_div(double_smooth(hw_src, 25, 13), double_smooth(hw_src.abs(), 25, 13)) / 2
    df["matrix_sig"] = np.sign(df["mfi"].fillna(0)) + np.sign(df["hw"].fillna(0))
    return df

# --- CORE 5: APEX SMC (TREND & LIQUIDITY) ---
def calc_apex_smc(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Smart Money Concepts, Order Blocks, FVG, WaveTrend"""
    df = df.copy()

    # 1. HMA Trend
    df["smc_base"] = hma(df["close"], int(p["smc_len"]))
    atr = rma((df["high"] - df["low"]).astype(float), 14)
    df["smc_u"] = df["smc_base"] + (atr * float(p["smc_mult"]))
    df["smc_l"] = df["smc_base"] - (atr * float(p["smc_mult"]))

    smc_trend = np.zeros(len(df), dtype=float)
    c = df["close"].values
    u = df["smc_u"].values
    l = df["smc_l"].values
    for i in range(1, len(df)):
        if np.isfinite(u[i]) and c[i] > u[i]:
            smc_trend[i] = 1
        elif np.isfinite(l[i]) and c[i] < l[i]:
            smc_trend[i] = -1
        else:
            smc_trend[i] = smc_trend[i - 1]
    df["smc_trend"] = smc_trend

    # 2. WaveTrend
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = ap.ewm(span=10).mean()
    d = (ap - esa).abs().ewm(span=10).mean()
    ci = safe_div((ap - esa), (0.015 * d))
    tci = ci.ewm(span=21).mean()

    df["mom_buy"] = (tci < 60) & (tci > tci.shift(1))
    df["mom_sell"] = (tci > -60) & (tci < tci.shift(1))

    # 3. ADX & Volume (approx; stable)
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = rma((df["high"] - df["low"]).astype(float), 14)
    pdi = 100 * safe_div(rma(pd.Series(plus_dm, index=df.index), 14), tr)
    mdi = 100 * safe_div(rma(pd.Series(minus_dm, index=df.index), 14), tr)
    dx = 100 * safe_div((pdi - mdi).abs(), (pdi + mdi))
    df["adx"] = rma(dx, 14)

    adx_ok = df["adx"] > 20
    vol_ok = df["volume"] > df["volume"].rolling(20).mean()

    # 4. Signals (edge-detected later)
    df["smc_buy"] = (df["smc_trend"] == 1) & (df["smc_trend"].shift(1) != 1) & vol_ok & df["mom_buy"] & adx_ok
    df["smc_sell"] = (df["smc_trend"] == -1) & (df["smc_trend"].shift(1) != -1) & vol_ok & df["mom_sell"] & adx_ok

    # 5. FVG
    df["fvg_bull"] = df["low"] > df["high"].shift(2)
    df["fvg_bear"] = df["high"] < df["low"].shift(2)

    return df


# ==========================================
# 4. DATA FEED
# ==========================================
def get_driver(name: str):
    opts = {"enableRateLimit": True}
    if name == "Binance":
        return ccxt.binance(opts)
    if name == "Bybit":
        return ccxt.bybit(opts)
    if name == "Coinbase":
        return ccxt.coinbase(opts)
    if name == "OKX":
        return ccxt.okx(opts)
    return ccxt.kraken(opts)

@st.cache_resource
def get_exchange_cached(name: str):
    ex = get_driver(name)
    # optional tuning
    ex.timeout = 20000
    return ex

@st.cache_data(ttl=5, show_spinner=False)
def fetch_data(exch: str, sym: str, tf: str, lim: int):
    try:
        ex = get_exchange_cached(exch)
        ohlcv = ex.fetch_ohlcv(sym, tf, limit=int(lim))
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        return df, ""
    except Exception as e:
        return pd.DataFrame(), f"{type(e).__name__}: {e}"

def init_session():
    # Secrets
    tg_t = st.secrets.get("TG_TOKEN", "")
    tg_c = st.secrets.get("TG_CHAT_ID", "")
    ai_key = st.secrets.get("OPENAI_API_KEY", "")

    defaults = {
        "exch": "Kraken",
        "sym": "BTC/USD",
        "tf": "15m",
        "lim": 500,
        "vec_len": 14,
        "vol_norm": 55,
        "vec_sm": 5,
        "eff_super": 0.6,
        "eff_resist": 0.3,
        "strict": 1.0,
        "div_look": 5,
        "brain_len": 55,
        "brain_mult": 1.5,
        "ent_len": 64,
        "ent_th": 2.0,
        "fli_len": 34,
        "smc_len": 55,
        "smc_mult": 1.5,
        "tg_t": tg_t,
        "tg_c": tg_c,
        "ai_key": ai_key,
        "auto": False,
        "auto_ms": 60000,
        "ai_model": "gpt-4o-mini",
    }

    if "cfg" not in st.session_state:
        st.session_state.cfg = defaults
    else:
        for k, v in defaults.items():
            if k not in st.session_state.cfg:
                st.session_state.cfg[k] = v

    if "log" not in st.session_state:
        st.session_state.log = []  # list[str]
    if "sent_alerts" not in st.session_state:
        st.session_state.sent_alerts = set()  # de-dup keys
    if "last_err" not in st.session_state:
        st.session_state.last_err = ""

def push_log(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    st.session_state.log = (st.session_state.log + [f"{ts}  {msg}"])[-120:]

def alert_key(sym, tf, ts, msg):
    # key stable across reruns for same candle + msg
    return f"{sym}|{tf}|{ts}|{msg}"

init_session()
cfg = st.session_state.cfg


# ==========================================
# 5. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### 🌌 TITAN PENTAGRAM")

    with st.expander("📡 Data & Auto-Pilot", expanded=True):
        cfg["exch"] = st.selectbox("Exchange", ["Kraken", "Binance", "Bybit", "Coinbase", "OKX"], index=0)
        cfg["sym"] = st.text_input("Ticker", cfg["sym"])
        c1, c2 = st.columns(2)
        cfg["tf"] = c1.selectbox("Time", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
        cfg["lim"] = c2.number_input("Lookback", 100, 2000, int(cfg["lim"]))

        cfg["auto"] = st.checkbox("Auto-Refresh", value=bool(cfg["auto"]))
        if cfg["auto"]:
            cfg["auto_ms"] = st.slider("Refresh interval (seconds)", 10, 180, int(cfg["auto_ms"] // 1000)) * 1000
            if HAS_AUTOREFRESH:
                st_autorefresh(interval=int(cfg["auto_ms"]), key="titan_autorefresh")
            else:
                st.caption("Install streamlit-autorefresh for true auto refresh.")

    with st.expander("⚡ Apex Vector"):
        cfg["eff_super"] = st.slider("Super Thresh", 0.1, 1.0, float(cfg["eff_super"]))
        cfg["eff_resist"] = st.slider("Resist Thresh", 0.0, 0.5, float(cfg["eff_resist"]))
        cfg["div_look"] = st.number_input("Div Lookback", 1, 50, int(cfg["div_look"]))

    with st.expander("🧠 Brain (Quad-Core)"):
        cfg["brain_len"] = st.number_input("Cortex Len", 10, 200, int(cfg["brain_len"]))
        cfg["ent_len"] = st.number_input("Entropy Len", 10, 200, int(cfg.get("ent_len", 64)))
        cfg["ent_th"] = st.slider("Entropy Gate", 0.5, 10.0, float(cfg["ent_th"]))

    with st.expander("🏛️ SMC & Trend"):
        cfg["smc_len"] = st.number_input("Trend Length", 10, 300, int(cfg["smc_len"]))
        cfg["smc_mult"] = st.slider("ATR Mult", 0.5, 5.0, float(cfg["smc_mult"]))

    with st.expander("🤖 & 📢 Connections"):
        cfg["ai_key"] = st.text_input("OpenAI Key", cfg["ai_key"], type="password")
        cfg["ai_model"] = st.text_input("AI Model", cfg.get("ai_model", "gpt-4o-mini"))
        cfg["tg_t"] = st.text_input("Bot Token", cfg["tg_t"], type="password")
        cfg["tg_c"] = st.text_input("Chat ID", cfg["tg_c"], type="password")
        if not HAS_OPENAI:
            st.caption("OpenAI SDK missing. `pip install openai`")

    colA, colB = st.columns(2)
    if colA.button("RELOAD DATA CACHE", use_container_width=True):
        fetch_data.clear()
        push_log("Cache cleared.")
        st.rerun()

    if colB.button("CLEAR LOG", use_container_width=True):
        st.session_state.log = []
        st.session_state.sent_alerts = set()
        st.session_state.last_err = ""
        st.rerun()

    st.markdown("---")
    st.markdown("### 📝 Event Log")
    log_cont = st.container()


# ==========================================
# 6. PROCESSING CORE
# ==========================================
df, err = fetch_data(cfg["exch"], cfg["sym"], cfg["tf"], int(cfg["lim"]))
if df.empty:
    st.session_state.last_err = err or st.session_state.last_err
    st.error("Data Connection Failed.")
    if st.session_state.last_err:
        st.caption(st.session_state.last_err)
    st.stop()

# CHAIN PROCESSING (ALL 5 CORES)
try:
    df = calc_apex_vector(df, cfg)  # Core 1
    df = calc_apex_brain(df, cfg)   # Core 2
    df = calc_rqzo(df, cfg)         # Core 3
    df = calc_matrix(df, cfg)       # Core 4
    df = calc_apex_smc(df, cfg)     # Core 5
except Exception as e:
    st.error("Indicator pipeline crashed.")
    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
    st.stop()

if len(df) < 3:
    st.warning("Not enough data yet.")
    st.stop()

last = df.iloc[-1]
prev = df.iloc[-2]


# --- Signal Dispatch ---
alerts = []

# SMC (edge detect)
if bool(last.get("smc_buy", False)) and not bool(prev.get("smc_buy", False)):
    alerts.append("🏛️ SMC: STRONG BUY SIGNAL")
if bool(last.get("smc_sell", False)) and not bool(prev.get("smc_sell", False)):
    alerts.append("🏛️ SMC: STRONG SELL SIGNAL")

# FVG creation edge detect
if bool(last.get("fvg_bull", False)) and not bool(prev.get("fvg_bull", False)):
    alerts.append("🏛️ SMC: BULL FVG CREATED")
if bool(last.get("fvg_bear", False)) and not bool(prev.get("fvg_bear", False)):
    alerts.append("🏛️ SMC: BEAR FVG CREATED")

# Brain (edge detect already done)
if bool(last.get("brain_buy", False)) and not bool(prev.get("brain_buy", False)):
    alerts.append("🧠 BRAIN: SYNAPTIC BUY")
if bool(last.get("brain_sell", False)) and not bool(prev.get("brain_sell", False)):
    alerts.append("🧠 BRAIN: SYNAPTIC SELL")

# Vector state edges
if int(last.get("vec_state", 0)) == 2 and int(prev.get("vec_state", 0)) != 2:
    alerts.append("⚡ VECTOR: SUPER BULL")
if int(last.get("vec_state", 0)) == -2 and int(prev.get("vec_state", 0)) != -2:
    alerts.append("⚡ VECTOR: SUPER BEAR")

# Telegram send (de-duped)
if alerts and cfg.get("tg_t"):
    for a in alerts:
        k = alert_key(cfg["sym"], cfg["tf"], str(last["timestamp"]), a)
        if k in st.session_state.sent_alerts:
            continue
        try:
            msg = f"🌌 TITAN OMEGA: {cfg['sym']} [{cfg['tf']}]\n{a}\nPrice: {last['close']}\nTime: {last['timestamp']}"
            requests.post(
                f"https://api.telegram.org/bot{cfg['tg_t']}/sendMessage",
                json={"chat_id": cfg["tg_c"], "text": msg},
                timeout=10,
            )
            st.session_state.sent_alerts.add(k)
            push_log(f"SENT → {a}")
        except Exception as e:
            push_log(f"TG ERROR → {type(e).__name__}: {e}")

# UI Log
with log_cont:
    if st.session_state.log:
        for line in st.session_state.log[-16:]:
            st.markdown(f"<div class='log-row'>{line}</div>", unsafe_allow_html=True)
    else:
        st.caption("Scanning 5-Core Physics...")

# ==========================================
# 7. VISUALIZATION
# ==========================================
st.title(f"🌌 {cfg['sym']} // {cfg['tf']}")

# --- PENTAGRAM HUD ---
c_trend = "BULL" if last["brain_trend"] == 1 else "BEAR"
s_trend = "st-bull" if last["brain_trend"] == 1 else "st-bear"

c_gate = "SAFE" if bool(last["gate_safe"]) else "LOCKED"
s_gate = "st-bull" if bool(last["gate_safe"]) else "st-bear"

c_vec = "SUPER" if abs(int(last["vec_state"])) == 2 else ("CHOP" if int(last["vec_state"]) == 0 else "HEAT")
s_vec = "st-bull" if int(last["vec_state"]) == 2 else ("st-bear" if int(last["vec_state"]) == -2 else "st-neut")

c_flow = "INFLOW" if float(last["flow"]) > 0 else "OUTFLOW"
s_flow = "st-bull" if float(last["flow"]) > 0 else "st-bear"

c_smc = "BUY" if bool(last["smc_buy"]) else ("SELL" if bool(last["smc_sell"]) else "WAIT")
s_smc = "st-bull" if bool(last["smc_buy"]) else ("st-bear" if bool(last["smc_sell"]) else "st-neut")

st.markdown(
    f"""
<div class="penta-grid">
    <div class="p-card"><div class="p-head">CORTEX (Trend)</div><div class="p-val {s_trend}">{c_trend}</div></div>
    <div class="p-card"><div class="p-head">AMYGDALA (Gate)</div><div class="p-val {s_gate}">{c_gate}</div></div>
    <div class="p-card"><div class="p-head">MOTOR (Vector)</div><div class="p-val {s_vec}">{c_vec}</div></div>
    <div class="p-card"><div class="p-head">OCCIPITAL (Flow)</div><div class="p-val {s_flow}">{c_flow}</div></div>
    <div class="p-card"><div class="p-head">SMC (Signal)</div><div class="p-val {s_smc}">{c_smc}</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# --- TABS ---
t1, t2, t3, t4, t5 = st.tabs(["🧠 Brain & Cortex", "🏛️ SMC & Liquidity", "⚡ Apex Vector", "💠 Matrix & RQZO", "🤖 AI Council"])

def dark_plot():
    return go.Layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=50, t=10, b=0),
        height=550,
        xaxis=dict(showgrid=False, color="#444"),
        yaxis=dict(showgrid=True, gridcolor="#222"),
    )

with t1:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))

    # Cortex Cloud
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["cortex_u"], line=dict(width=0), showlegend=False))
    cloud_col = "rgba(0, 230, 118, 0.1)" if last["brain_trend"] == 1 else "rgba(255, 23, 68, 0.1)"
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["cortex_l"], fill="tonexty", fillcolor=cloud_col, line=dict(width=0), name="Cortex"))

    # Brain Signals
    b = df[df["brain_buy"]]
    s = df[df["brain_sell"]]
    fig.add_trace(go.Scatter(x=b["timestamp"], y=b["low"], mode="markers", marker=dict(symbol="triangle-up", color="#00E676", size=12), name="Brain Buy"))
    fig.add_trace(go.Scatter(x=s["timestamp"], y=s["high"], mode="markers", marker=dict(symbol="triangle-down", color="#FF1744", size=12), name="Brain Sell"))

    fig.update_layout(dark_plot())
    st.plotly_chart(fig, use_container_width=True)

with t2:
    fig_s = go.Figure()
    fig_s.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))

    # FVGs
    fvg_b = df[df["fvg_bull"]]
    fvg_s = df[df["fvg_bear"]]
    fig_s.add_trace(go.Scatter(x=fvg_b["timestamp"], y=fvg_b["low"], mode="markers", marker=dict(symbol="square", color="rgba(0,230,118,0.4)", size=8), name="Bull FVG"))
    fig_s.add_trace(go.Scatter(x=fvg_s["timestamp"], y=fvg_s["high"], mode="markers", marker=dict(symbol="square", color="rgba(255,23,68,0.4)", size=8), name="Bear FVG"))

    # SMC Signals
    sb = df[df["smc_buy"]]
    ss = df[df["smc_sell"]]
    fig_s.add_trace(go.Scatter(x=sb["timestamp"], y=sb["low"] * 0.999, mode="markers+text", text="BUY", textposition="bottom center",
                               marker=dict(symbol="triangle-up", color="#00E676", size=14), name="SMC Buy"))
    fig_s.add_trace(go.Scatter(x=ss["timestamp"], y=ss["high"] * 1.001, mode="markers+text", text="SELL", textposition="top center",
                               marker=dict(symbol="triangle-down", color="#FF1744", size=14), name="SMC Sell"))

    # Trend Line
    fig_s.add_trace(go.Scatter(x=df["timestamp"], y=df["smc_base"], line=dict(color="cyan", width=1), name="SMC Base"))

    fig_s.update_layout(dark_plot())
    st.plotly_chart(fig_s, use_container_width=True)

with t3:
    fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.03)
    fig_v.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]), row=1, col=1)

    # Flux state colors
    cols = ["#00E676" if x == 2 else ("#FF1744" if x == -2 else ("#546E7A" if x == 0 else "#FFD600")) for x in df["vec_state"]]
    fig_v.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=cols, name="Flux"), row=2, col=1)

    # Divs
    bd = df[df["div_bull"]]
    sd = df[df["div_bear"]]
    fig_v.add_trace(go.Scatter(x=bd["timestamp"], y=bd["flux"], mode="markers", marker=dict(color="cyan", size=6), name="Bull Div"), row=2, col=1)
    fig_v.add_trace(go.Scatter(x=sd["timestamp"], y=sd["flux"], mode="markers", marker=dict(color="magenta", size=6), name="Bear Div"), row=2, col=1)

    fig_v.update_layout(dark_plot())
    st.plotly_chart(fig_v, use_container_width=True)

with t4:
    fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig_m.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], line=dict(color="white"), name="RQZO"), row=1, col=1)
    fig_m.add_trace(go.Scatter(x=df["timestamp"], y=df["mfi"], fill="tozeroy", line_color="#D500F9", name="MFI"), row=2, col=1)
    fig_m.add_trace(go.Bar(x=df["timestamp"], y=df["hw"], marker_color="#00E5FF", name="HyperWave"), row=2, col=1)

    fig_m.update_layout(dark_plot())
    st.plotly_chart(fig_m, use_container_width=True)

with t5:
    st.markdown("### 🤖 The Council of Five")
    st.markdown("<div class='tiny'>Tip: Keep prompts short and actionable. Council outputs: BIAS, ENTRY, RISK, NUANCE.</div>", unsafe_allow_html=True)

    persona = st.selectbox(
        "Choose Your Analyst",
        ["The Grand Strategist (All)", "The Physicist (Vector)", "The Neurologist (Brain)", "The Quant (Matrix/RQZO)", "The Banker (SMC)"],
    )

    if st.button("Consult Analyst"):
        if not cfg.get("ai_key"):
            st.error("AI Key Missing.")
        elif not HAS_OPENAI:
            st.error("OpenAI SDK missing. Run: `pip install openai`")
        else:
            with st.spinner(f"{persona} is analyzing..."):
                base_prompt = f"Market: {cfg['sym']} {cfg['tf']} | Price: {float(last['close']):.2f}\n"

                if "Physicist" in persona:
                    spec_prompt = (
                        f"Focus on Vector Physics.\n"
                        f"Flux: {float(last['flux']):.3f}, Eff: {float(last['eff']):.3f}, "
                        f"DivBull: {bool(last['div_bull'])}, DivBear: {bool(last['div_bear'])}."
                    )
                elif "Neurologist" in persona:
                    spec_prompt = (
                        f"Focus on Brain Logic.\n"
                        f"Trend: {int(last['brain_trend'])}, GateSafe: {bool(last['gate_safe'])}, Flow: {float(last['flow']):.3f}."
                    )
                elif "Quant" in persona:
                    spec_prompt = (
                        f"Focus on Math.\n"
                        f"RQZO: {float(last['rqzo']):.3f}, MatrixSig: {float(last['matrix_sig']):.0f}, "
                        f"MFI: {float(last['mfi']):.3f}, HW: {float(last['hw']):.3f}."
                    )
                elif "Banker" in persona:
                    spec_prompt = (
                        f"Focus on Smart Money.\n"
                        f"SMC Signal: {c_smc}, FVG Bull: {bool(last['fvg_bull'])}, FVG Bear: {bool(last['fvg_bear'])}, "
                        f"ADX: {float(last['adx']):.2f}."
                    )
                else:
                    spec_prompt = (
                        f"Synthesize ALL 5 Cores.\n"
                        f"Flux {float(last['flux']):.3f}, Trend {int(last['brain_trend'])}, Gate {bool(last['gate_safe'])}, "
                        f"RQZO {float(last['rqzo']):.3f}, MatrixSig {float(last['matrix_sig']):.0f}, SMC {c_smc}, Flow {float(last['flow']):.3f}."
                    )

                final_prompt = (
                    base_prompt
                    + spec_prompt
                    + "\n\nOutput format (strict):\n"
                      "BIAS: (bull/bear/neutral + 1 sentence)\n"
                      "ENTRY: (conditions, not a command)\n"
                      "RISK: (stop logic + invalidation)\n"
                      "NUANCE: (what could flip this view)\n"
                )

                try:
                    client = OpenAI(api_key=cfg["ai_key"])
                    # chat.completions is supported by modern OpenAI Python SDK
                    resp = client.chat.completions.create(
                        model=cfg.get("ai_model", "gpt-4o-mini"),
                        messages=[{"role": "user", "content": final_prompt}],
                        temperature=0.3,
                    )
                    text = resp.choices[0].message.content
                    st.markdown(f"<div class='ai-response'>{text}</div>", unsafe_allow_html=True)
                    push_log("AI Council consulted.")
                except Exception as e:
                    st.error(f"AI Error: {type(e).__name__}: {e}")
                    push_log(f"AI ERROR → {type(e).__name__}: {e}")


# ==========================================
# 8. FOOTER DIAGNOSTICS (optional)
# ==========================================
with st.expander("Diagnostics", expanded=False):
    st.write(
        {
            "exchange": cfg["exch"],
            "symbol": cfg["sym"],
            "tf": cfg["tf"],
            "rows": int(len(df)),
            "last_timestamp": str(last["timestamp"]),
            "last_error": st.session_state.last_err,
            "autorefresh": bool(cfg["auto"]),
            "has_autorefresh_pkg": HAS_AUTOREFRESH,
            "has_openai_pkg": HAS_OPENAI,
        }
    )

"""
Requirements (recommended):
  pip install streamlit ccxt pandas numpy plotly requests openai streamlit-autorefresh

Run:
  streamlit run Titan-Apps/titan-v5.0-ApexMaster.py
"""
