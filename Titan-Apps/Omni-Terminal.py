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
import logging

# Optional: auto refresh (non-blocking)
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOR_AVAILABLE = True
except ImportError:
    AUTOR_AVAILABLE = False

# Optional: X/Twitter
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

# Optional: OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =========================
# LOGGING (visible + useful)
# =========================
logger = logging.getLogger("pentagram")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)


# ==========================================
# 1. PAGE CONFIGURATION & UI THEME
# ==========================================
st.set_page_config(
    page_title="Omni-Terminal",
    layout="wide",
    page_icon="🔥",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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

    .log-container {
        font-size: 0.75rem; color: #888; max-height: 240px; overflow-y: auto;
        border: 1px solid #222; padding: 10px; border-radius: 4px; background: #020202;
    }
    .log-entry { border-bottom: 1px solid #111; padding: 6px 0; display: flex; justify-content: space-between; gap: 10px; }
    .log-ts { color: #555; white-space: nowrap; }
    .log-msg { color: #bbb; }

    .pane-btn > button {
        width: 100%;
        text-align: left !important;
        background: #050505 !important;
        border: 1px solid #1a1a1a !important;
        border-left: 4px solid var(--accent) !important;
        padding: 16px 14px !important;
        border-radius: 6px !important;
        color: #e0e0e0 !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .pane-btn > button:hover {
        border-color: #2a2a2a !important;
        transform: translateY(-1px);
    }

    .pill {
        display:inline-block;
        padding: 2px 8px;
        border: 1px solid #222;
        border-radius: 999px;
        font-size: 0.75rem;
        color: #bbb;
        background: #070707;
        margin-right: 6px;
    }
    .pill-bull { border-color: #0b3; color: #9f9; }
    .pill-bear { border-color: #b03; color: #f99; }
    .pill-warn { border-color: #aa0; color: #ff9; }

</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. GLOBAL CONNECTIVITY HANDLERS
# ==========================================
def send_telegram(token, chat_id, text):
    if not token or not chat_id:
        return False, "Telegram token/chat_id missing"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        ok = r.status_code == 200
        return ok, ("" if ok else f"Telegram HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        return False, f"Telegram error: {e}"


def post_x(consumer_key, consumer_secret, access_token, access_secret, text):
    if not TWEEPY_AVAILABLE:
        return False, "tweepy not installed"
    if not (consumer_key and consumer_secret and access_token and access_secret):
        return False, "X credentials incomplete"
    try:
        client = tweepy.Client(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_secret,
        )
        client.create_tweet(text=text)
        return True, ""
    except Exception as e:
        return False, f"X error: {e}"


# ==========================================
# 3. PENTAGRAM MATH ENGINE (ALL 5 CORES)
# ==========================================
def rma(series, length):
    length = int(length)
    if length <= 1:
        return series
    return series.ewm(alpha=1 / length, adjust=False).mean()


def wma(series, length):
    length = int(length)
    if length <= 1:
        return series
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hma(series, length):
    length = int(length)
    if length <= 2:
        return series
    return wma(2 * wma(series, length // 2) - wma(series, length), int(np.sqrt(length)))


def double_smooth(src, l1, l2):
    return src.ewm(span=int(l1), adjust=False).mean().ewm(span=int(l2), adjust=False).mean()


# CORE 1: APEX VECTOR (Physics Engine)
def calc_vector(df, p):
    df = df.copy()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()

    eff_raw = (body / rng).fillna(0.0)
    df["eff"] = eff_raw.ewm(span=int(p["vec_l"]), adjust=False).mean()

    vol_ma = df["volume"].rolling(int(p["vol_n"])).mean().replace(0, np.nan)
    vol_fact = (df["volume"] / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    raw_v = np.sign(df["close"] - df["open"]) * df["eff"] * vol_fact
    df["flux"] = pd.Series(raw_v).ewm(span=int(p["vec_sm"]), adjust=False).mean()

    th_s = float(p["vec_super"]) * float(p["vec_strict"])
    th_r = float(p["vec_resist"]) * float(p["vec_strict"])

    conditions = [
        (df["flux"] > th_s),
        (df["flux"] < -th_s),
        (df["flux"].abs() < th_r),
    ]
    # 2 = super bull, -2 = super bear, 0 = neutral, 1 = mild state (default)
    df["vec_state"] = np.select(conditions, [2, -2, 0], default=1)

    df["pl"] = (df["flux"].shift(1) < df["flux"].shift(2)) & (df["flux"].shift(1) < df["flux"])
    df["ph"] = (df["flux"].shift(1) > df["flux"].shift(2)) & (df["flux"].shift(1) > df["flux"])

    lookback = 5
    df["div_bull"] = df["pl"] & (df["close"] < df["close"].shift(lookback)) & (df["flux"] > df["flux"].shift(lookback))
    df["div_bear"] = df["ph"] & (df["close"] > df["close"].shift(lookback)) & (df["flux"] < df["flux"].shift(lookback))
    return df


# CORE 2: APEX BRAIN (Logical Processing)
def calc_brain(df, p):
    df = df.copy()
    base = hma(df["close"], int(p["br_l"]))

    # NOTE: This is not true ATR; keeping your original intent, but hardened.
    atr_like = rma((df["high"] - df["low"]).abs(), int(p["br_l"])).fillna(0.0)

    df["br_u"] = base + (atr_like * float(p["br_m"]))
    df["br_l_band"] = base - (atr_like * float(p["br_m"]))

    # Trend state machine
    trend = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        c = df["close"].iloc[i]
        if c > df["br_u"].iloc[i]:
            trend[i] = 1
        elif c < df["br_l_band"].iloc[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    df["br_trend"] = trend

    df["ent"] = df["close"].pct_change().rolling(64).std() * 100
    df["gate"] = df["ent"] < float(p["br_th"])

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    wick_balance = (
        (np.minimum(df["open"], df["close"]) - df["low"]) - (df["high"] - np.maximum(df["open"], df["close"]))
    ) / rng
    wick_balance = wick_balance.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["flow"] = pd.Series(
        wick_balance + ((df["close"] - df["open"]) / (rma(rng.fillna(0.0), 14) + 1e-10))
    ).ewm(span=34, adjust=False).mean()

    # Requires df["flux"] already present (vector first)
    df["br_buy"] = (df["br_trend"] == 1) & df["gate"] & (df["flux"] > 0.5) & (df["flow"] > 0)
    df["br_sell"] = (df["br_trend"] == -1) & df["gate"] & (df["flux"] < -0.5) & (df["flow"] < 0)
    return df


# CORE 3: RQZO (Quantum Mechanics)
def calc_rqzo(df, p):
    df = df.copy()
    win = 100
    mn = df["close"].rolling(win).min()
    mx = df["close"].rolling(win).max()
    norm = (df["close"] - mn) / (mx - mn + 1e-10)

    delta = (norm - norm.shift(1)).abs().fillna(0.0)
    v = np.clip(delta, 0, 0.049) / 0.05
    gamma = 1 / np.sqrt(1 - (v**2) + 1e-10)

    tau = (np.arange(len(df)) % win) / gamma.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    rq = np.zeros(len(df), dtype=float)
    for n in range(1, 10):
        rq += (n**-0.5) * np.sin(tau * np.log(n))
    df["rqzo"] = rq * 10
    return df


# CORE 4: MATRIX (Momentum Matrix)
def calc_matrix(df, p):
    df = df.copy()
    diff = df["close"].diff()
    up = diff.clip(lower=0)
    dn = (-diff).clip(lower=0)

    rs = rma(up, 14) / (rma(dn, 14) + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    vol_ma = df["volume"].rolling(20).mean().replace(0, np.nan)
    vol_factor = (df["volume"] / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    df["mfi"] = ((rsi - 50) * vol_factor).ewm(span=3, adjust=False).mean()

    num = double_smooth(df["close"].diff(), 25, 13)
    den = double_smooth(df["close"].diff().abs(), 25, 13) + 1e-10
    df["hw"] = 100 * (num / den) / 2

    df["mat_sig"] = np.sign(df["mfi"]) + np.sign(df["hw"])
    return df


# CORE 5: APEX SMC (Structural Master)
def calc_smc(df, p):
    df = df.copy()
    df["smc_base"] = hma(df["close"], int(p["smc_l"]))

    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ap.ewm(span=10, adjust=False).mean()
    dev = (ap - esa).abs().ewm(span=10, adjust=False).mean()

    tci = ((ap - esa) / (0.015 * dev + 1e-10)).ewm(span=21, adjust=False).mean()

    df["smc_buy"] = (df["close"] > df["smc_base"]) & (tci < 60) & (tci > tci.shift(1))
    df["smc_sell"] = (df["close"] < df["smc_base"]) & (tci > -60) & (tci < tci.shift(1))

    df["fvg_b"] = df["low"] > df["high"].shift(2)
    df["fvg_s"] = df["high"] < df["low"].shift(2)
    df["tci"] = tci
    return df


# ==========================================
# 4. DATA HANDLING (cached exchange + symbol resolution + retries)
# ==========================================
TF_TTL = {"1m": 10, "5m": 20, "15m": 30, "1h": 60, "4h": 120, "1d": 300}

@st.cache_resource
def get_exchange(exch_name: str):
    ex_class = getattr(ccxt, exch_name.lower())
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()
    return ex

def resolve_symbol(ex, sym: str) -> str:
    """
    Try to resolve common user formats into an exchange-supported symbol.
    This avoids constant "VOID DETECTED" for normal usage.
    """
    sym = sym.strip().upper()

    # Direct hit
    if sym in ex.symbols:
        return sym

    # Common substitutions
    candidates = [sym]

    if sym.endswith("/USD"):
        candidates.append(sym.replace("/USD", "/USDT"))
        candidates.append(sym.replace("/USD", "/USD:USD"))  # some derivatives formats
    if sym.endswith("/USDT"):
        candidates.append(sym.replace("/USDT", "/USD"))

    # Kraken XBT alias attempt
    candidates.append(sym.replace("BTC/", "XBT/"))

    for c in candidates:
        if c in ex.symbols:
            return c

    # Last resort: if user typed without slash
    if "/" not in sym:
        for guess in [sym[:-4] + "/" + sym[-4:], sym[:-3] + "/" + sym[-3:]]:
            if guess in ex.symbols:
                return guess

    raise ValueError(f"Unsupported symbol on {ex.id}: {sym}")

@st.cache_data(show_spinner=False)
def fetch_ohlcv(exch_name: str, sym: str, tf: str, lim: int, _ttl_bucket: int):
    ex = get_exchange(exch_name)
    sym2 = resolve_symbol(ex, sym)

    # Simple retry (ccxt will rate-limit internally, but network glitches happen)
    last_err = None
    for _ in range(3):
        try:
            ohlcv = ex.fetch_ohlcv(sym2, tf, limit=lim)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
            return df, sym2, ""
        except Exception as e:
            last_err = str(e)
            time_lib.sleep(0.4)

    return pd.DataFrame(), sym, last_err or "unknown error"


# ==========================================
# 5. STATE INIT
# ==========================================
def init():
    defaults = {
        "exch": "Kraken",
        "sym": "BTC/USD",
        "tf": "15m",
        "lim": 500,
        "auto": False,
        "debug": False,

        "vec_l": 14, "vol_n": 55, "vec_sm": 5, "vec_super": 0.6, "vec_resist": 0.3, "vec_strict": 1.0,
        "br_l": 55, "br_m": 1.5, "br_th": 2.0,
        "smc_l": 55,

        "tg_t": "", "tg_c": "",
        "x_k": "", "x_s": "", "x_at": "", "x_as": "",
        "ai_k": "",

        "signal_log": [],
        "last_alert_ts": {},  # {signal_name: last_timestamp_iso}
        "tv_symbol_override": "",

        # --- Scanner / Radar ---
        "scanner_on": False,
        "scanner_tf": "15m",
        "scanner_lim": 240,
        "scanner_topn": 25,
        "scanner_abs_flux_min": 0.9,
        "scanner_symbols_text": "BTC/USD\nETH/USD\nXRP/USD",
        "scanner_autopilot": False,
        "scanner_interval_sec": 120,

        "radar_events": [],          # list of dicts
        "radar_last_seen": set(),    # dedupe keys stored as set
        "replay_selected_key": "",

        # --- Backtest-lite ---
        "bt_n_signals": 50,
        "bt_horizon": 48,            # candles forward
        "bt_target_pct": 1.0,
        "bt_stop_pct": 0.7,
        "bt_signal_family": "ALL",

        # --- MTF consensus ---
        "mtf_tfs": ["15m", "1h", "4h"],
        "mtf_lim": 300,

        # --- Replay panel ---
        "replay_window": 120,
        "replay_center_pad": 30,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log_event(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    st.session_state.signal_log = ([{"ts": ts, "msg": msg}] + st.session_state.signal_log)[:200]

def radar_log_event(ev: dict):
    # deterministic dedupe key
    k = f'{ev.get("exchange","")}::{ev.get("symbol","")}::{ev.get("tf","")}::{str(ev.get("candle_ts",""))}::{ev.get("kind","")}'
    if k in st.session_state.radar_last_seen:
        return
    st.session_state.radar_last_seen.add(k)
    st.session_state.radar_events = ([ev] + st.session_state.radar_events)[:600]

init()


# ==========================================
# 6. SIDEBAR CONTROL DECK
# ==========================================
with st.sidebar:
    st.markdown("### 🔯 CONTROL DECK")

    with st.expander("🌍 Feed Configuration", expanded=True):
        st.session_state.exch = st.selectbox(
            "Exchange",
            ["Kraken", "Binance", "Bybit", "Coinbase", "OKX"],
            index=["Kraken", "Binance", "Bybit", "Coinbase", "OKX"].index(st.session_state.exch)
            if st.session_state.exch in ["Kraken", "Binance", "Bybit", "Coinbase", "OKX"] else 0
        )
        st.session_state.sym = st.text_input("Asset Ticker (CCXT format)", st.session_state.sym)
        tf_list = ["1m", "5m", "15m", "1h", "4h", "1d"]
        st.session_state.tf = st.selectbox(
            "Interval",
            tf_list,
            index=tf_list.index(st.session_state.tf) if st.session_state.tf in tf_list else 2
        )
        st.session_state.lim = st.slider("Candles", 200, 2000, int(st.session_state.lim), step=50)

        st.session_state.auto = st.toggle("🔄 Auto-Pilot (60s)", value=bool(st.session_state.auto))
        st.session_state.debug = st.toggle("🧪 Debug Diagnostics", value=bool(st.session_state.debug))

        if st.session_state.auto:
            if AUTOR_AVAILABLE:
                st_autorefresh(interval=60 * 1000, key="pentagram_autorefresh")
            else:
                st.warning("Auto-Pilot requires `streamlit-autorefresh`. Install it or use manual reload.")

    with st.expander("🧠 Core Parameters", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.vec_l = st.slider("Vector Length", 3, 60, int(st.session_state.vec_l))
            st.session_state.vol_n = st.slider("Volume Norm", 5, 200, int(st.session_state.vol_n))
            st.session_state.vec_sm = st.slider("Flux Smooth", 1, 30, int(st.session_state.vec_sm))
        with c2:
            st.session_state.vec_super = st.slider("Super Threshold", 0.1, 3.0, float(st.session_state.vec_super), 0.1)
            st.session_state.vec_resist = st.slider("Neutral Threshold", 0.05, 2.0, float(st.session_state.vec_resist), 0.05)
            st.session_state.vec_strict = st.slider("Strictness", 0.5, 2.5, float(st.session_state.vec_strict), 0.1)

        c3, c4 = st.columns(2)
        with c3:
            st.session_state.br_l = st.slider("Brain Length", 10, 150, int(st.session_state.br_l))
            st.session_state.br_m = st.slider("Brain Mult", 0.5, 5.0, float(st.session_state.br_m), 0.1)
        with c4:
            st.session_state.br_th = st.slider("Entropy Gate", 0.1, 10.0, float(st.session_state.br_th), 0.1)
            st.session_state.smc_l = st.slider("SMC Base Length", 10, 150, int(st.session_state.smc_l))

    with st.expander("📡 Scanner / Radar (20–100 symbols)", expanded=False):
        st.session_state.scanner_on = st.toggle("Enable Scanner Mode", value=bool(st.session_state.scanner_on))
        st.session_state.scanner_tf = st.selectbox(
            "Scanner TF",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            index=["1m", "5m", "15m", "1h", "4h", "1d"].index(st.session_state.scanner_tf)
            if st.session_state.scanner_tf in ["1m", "5m", "15m", "1h", "4h", "1d"] else 2
        )
        st.session_state.scanner_lim = st.slider("Scanner candles", 120, 800, int(st.session_state.scanner_lim), step=20)
        st.session_state.scanner_topn = st.slider("Leaderboard size", 5, 100, int(st.session_state.scanner_topn), step=5)
        st.session_state.scanner_abs_flux_min = st.slider("Record events if |flux| ≥", 0.1, 5.0, float(st.session_state.scanner_abs_flux_min), 0.1)
        st.session_state.scanner_interval_sec = st.slider("Scanner interval (sec)", 30, 600, int(st.session_state.scanner_interval_sec), step=10)

        st.session_state.scanner_symbols_text = st.text_area(
            "Symbols (one per line, CCXT format)",
            st.session_state.scanner_symbols_text,
            height=160
        )

        st.session_state.scanner_autopilot = st.toggle(
            "Scanner Auto-Pilot",
            value=bool(st.session_state.scanner_autopilot)
        )
        if st.session_state.scanner_autopilot and AUTOR_AVAILABLE:
            st_autorefresh(interval=int(st.session_state.scanner_interval_sec) * 1000, key="scanner_autorefresh")
        elif st.session_state.scanner_autopilot and not AUTOR_AVAILABLE:
            st.warning("Scanner Auto-Pilot requires `streamlit-autorefresh`.")

        if st.button("Run Scanner Now", use_container_width=True):
            st.session_state["_scanner_run_now"] = True

    with st.expander("🧩 MTF Consensus (15m/1h/4h)", expanded=False):
        st.session_state.mtf_lim = st.slider("MTF candles per TF", 120, 800, int(st.session_state.mtf_lim), step=20)

    with st.expander("🧪 Backtest-lite", expanded=False):
        st.session_state.bt_signal_family = st.selectbox(
            "Signals",
            ["ALL", "BRAIN", "SMC", "VECTOR"],
            index=["ALL", "BRAIN", "SMC", "VECTOR"].index(st.session_state.bt_signal_family)
        )
        st.session_state.bt_n_signals = st.slider("Last N signals", 10, 400, int(st.session_state.bt_n_signals), step=10)
        st.session_state.bt_horizon = st.slider("Forward horizon (candles)", 5, 400, int(st.session_state.bt_horizon), step=5)
        st.session_state.bt_target_pct = st.slider("Target %", 0.1, 20.0, float(st.session_state.bt_target_pct), 0.1)
        st.session_state.bt_stop_pct = st.slider("Stop %", 0.1, 20.0, float(st.session_state.bt_stop_pct), 0.1)

    with st.expander("📡 Omnichannel APIs", expanded=False):
        st.session_state.tg_t = st.text_input("Telegram Bot Token", st.session_state.tg_t, type="password")
        st.session_state.tg_c = st.text_input("Telegram Chat ID", st.session_state.tg_c)

        st.markdown("**X / Twitter (optional)**")
        st.session_state.x_k = st.text_input("X Consumer Key", st.session_state.x_k, type="password")
        st.session_state.x_s = st.text_input("X Consumer Secret", st.session_state.x_s, type="password")
        st.session_state.x_at = st.text_input("X Access Token", st.session_state.x_at, type="password")
        st.session_state.x_as = st.text_input("X Access Secret", st.session_state.x_as, type="password")

        st.markdown("**OpenAI (optional)**")
        st.session_state.ai_k = st.text_input("OpenAI Secret", st.session_state.ai_k, type="password")

    with st.expander("📺 TradingView", expanded=False):
        st.session_state.tv_symbol_override = st.text_input(
            "TradingView Symbol Override (e.g. BINANCE:BTCUSDT)",
            st.session_state.tv_symbol_override
        )

    if st.button("🔱 RELOAD THE PENTAGRAM", type="primary", use_container_width=True):
        fetch_ohlcv.clear()
        st.rerun()


# ==========================================
# 7. PROCESSING (main symbol)
# ==========================================
ttl_bucket = TF_TTL.get(st.session_state.tf, 30)
df, resolved_sym, fetch_err = fetch_ohlcv(
    st.session_state.exch,
    st.session_state.sym,
    st.session_state.tf,
    int(st.session_state.lim),
    ttl_bucket
)

if df.empty:
    st.error("VOID DETECTED: Data fetch failed.")
    st.caption(f"Exchange={st.session_state.exch}  Symbol={st.session_state.sym}  TF={st.session_state.tf}")
    if fetch_err:
        st.code(fetch_err)
    st.stop()

# Chain computation
df = calc_vector(df, st.session_state)
df = calc_brain(df, st.session_state)
df = calc_rqzo(df, st.session_state)
df = calc_matrix(df, st.session_state)
df = calc_smc(df, st.session_state)

last, prev = df.iloc[-1], df.iloc[-2]


# ==========================================
# 8. ALERT ENGINE (deduped)
# ==========================================
def should_alert(signal_name: str, candle_ts: pd.Timestamp) -> bool:
    key = signal_name
    ts = str(candle_ts)
    prev_ts = st.session_state.last_alert_ts.get(key)
    if prev_ts == ts:
        return False
    st.session_state.last_alert_ts[key] = ts
    return True

events = []

# Edge-detected signals
if bool(last["br_buy"]) and not bool(prev["br_buy"]):
    events.append(("br_buy", f"🧠 BRAIN LONG: {resolved_sym} @ {last['close']:.2f} ({st.session_state.tf})"))
if bool(last["br_sell"]) and not bool(prev["br_sell"]):
    events.append(("br_sell", f"🧠 BRAIN SHORT: {resolved_sym} @ {last['close']:.2f} ({st.session_state.tf})"))

if bool(last["smc_buy"]) and not bool(prev["smc_buy"]):
    events.append(("smc_buy", f"🏛️ SMC BUY: {resolved_sym} @ {last['close']:.2f} ({st.session_state.tf})"))
if bool(last["smc_sell"]) and not bool(prev["smc_sell"]):
    events.append(("smc_sell", f"🏛️ SMC SELL: {resolved_sym} @ {last['close']:.2f} ({st.session_state.tf})"))

if int(last["vec_state"]) == 2 and int(prev["vec_state"]) != 2:
    events.append(("vec_super", f"⚡ SUPERCONDUCTOR: {resolved_sym} flux={last['flux']:.3f} ({st.session_state.tf})"))
if int(last["vec_state"]) == -2 and int(prev["vec_state"]) != -2:
    events.append(("vec_crash", f"⚡ CRASH STATE: {resolved_sym} flux={last['flux']:.3f} ({st.session_state.tf})"))

# Dispatch (deduped per candle)
for sig, msg in events:
    if should_alert(sig, last["timestamp"]):
        log_event(msg)

        ok_tg, err_tg = send_telegram(st.session_state.tg_t, st.session_state.tg_c, msg)
        if st.session_state.debug and (st.session_state.tg_t or st.session_state.tg_c):
            log_event("TG ✅" if ok_tg else f"TG ❌ {err_tg}")

        ok_x, err_x = post_x(st.session_state.x_k, st.session_state.x_s, st.session_state.x_at, st.session_state.x_as, msg)
        if st.session_state.debug and (st.session_state.x_k or st.session_state.x_at):
            log_event("X ✅" if ok_x else f"X ❌ {err_x}")


# ==========================================
# 8B. SCANNER MODE (watchlist + flux spike leaderboard + auto-record radar events)
# ==========================================
def parse_symbols(text: str) -> list:
    lines = [x.strip() for x in (text or "").splitlines()]
    lines = [x for x in lines if x]
    # deterministic de-dupe preserving order
    out = []
    seen = set()
    for s in lines:
        su = s.upper()
        if su not in seen:
            seen.add(su)
            out.append(su)
    return out

def fast_flux_for_symbol(exch: str, sym: str, tf: str, lim: int, ttl: int):
    dfi, sym2, err = fetch_ohlcv(exch, sym, tf, lim, ttl)
    if dfi.empty:
        return None, sym2, err
    dfi = calc_vector(dfi, st.session_state)
    return float(dfi["flux"].iloc[-1]), sym2, ""

def run_scanner():
    symbols = parse_symbols(st.session_state.scanner_symbols_text)
    if not symbols:
        return pd.DataFrame(columns=["symbol", "resolved", "flux", "abs_flux", "close", "ts", "err"]), []

    ttl = TF_TTL.get(st.session_state.scanner_tf, 30)
    rows = []
    recorded = []

    # sequential scan (ccxt rate-limit handled by exchange; cache reduces load)
    for s in symbols[: max(1, int(st.session_state.scanner_topn) * 4)]:
        flux, resolved, err = fast_flux_for_symbol(
            st.session_state.exch,
            s,
            st.session_state.scanner_tf,
            int(st.session_state.scanner_lim),
            ttl
        )
        if flux is None:
            rows.append({"symbol": s, "resolved": resolved, "flux": np.nan, "abs_flux": np.nan, "close": np.nan, "ts": "", "err": err})
            continue

        # fetch last candle close/ts deterministically by using same cached OHLCV call again
        dfi, _, _ = fetch_ohlcv(st.session_state.exch, s, st.session_state.scanner_tf, int(st.session_state.scanner_lim), ttl)
        close = float(dfi["close"].iloc[-1])
        ts = dfi["timestamp"].iloc[-1]

        rows.append({"symbol": s, "resolved": resolved, "flux": float(flux), "abs_flux": float(abs(flux)), "close": close, "ts": ts, "err": ""})

        if abs(float(flux)) >= float(st.session_state.scanner_abs_flux_min):
            kind = "RADAR_BULL" if float(flux) > 0 else "RADAR_BEAR"
            ev = {
                "exchange": st.session_state.exch,
                "symbol": resolved,
                "tf": st.session_state.scanner_tf,
                "candle_ts": ts,
                "kind": kind,
                "flux": float(flux),
                "close": close,
                "msg": f"📡 RADAR {kind.replace('RADAR_','')}: {resolved} flux={float(flux):.3f} @ {close:.2f} ({st.session_state.scanner_tf})",
            }
            radar_log_event(ev)
            recorded.append(ev)

    out = pd.DataFrame(rows)
    out = out.sort_values("abs_flux", ascending=False, na_position="last")
    out = out.head(int(st.session_state.scanner_topn))
    return out, recorded


scanner_df = None
scanner_recorded = []
scanner_should_run = bool(st.session_state.scanner_on) and (
    bool(st.session_state.get("_scanner_run_now", False)) or bool(st.session_state.scanner_autopilot)
)

if scanner_should_run:
    st.session_state["_scanner_run_now"] = False
    try:
        scanner_df, scanner_recorded = run_scanner()
        if len(scanner_recorded) > 0:
            log_event(f"📡 Scanner recorded {len(scanner_recorded)} radar event(s).")
    except Exception as e:
        scanner_df = pd.DataFrame(columns=["symbol", "resolved", "flux", "abs_flux", "close", "ts", "err"])
        log_event(f"📡 Scanner error: {e}")


# ==========================================
# 8C. MTF CONSENSUS (15m/1h/4h alignment + score)
# ==========================================
def tf_signals_snapshot(exch: str, sym: str, tf: str, lim: int, ttl: int):
    dfi, rsym, err = fetch_ohlcv(exch, sym, tf, lim, ttl)
    if dfi.empty:
        return {"tf": tf, "symbol": rsym, "err": err}

    dfi = calc_vector(dfi, st.session_state)
    dfi = calc_brain(dfi, st.session_state)
    dfi = calc_matrix(dfi, st.session_state)
    dfi = calc_smc(dfi, st.session_state)

    lx = dfi.iloc[-1]

    # deterministic sign extraction (no heuristics)
    br = int(lx["br_trend"])
    vec = int(lx["vec_state"])
    # map vec_state to direction: 2 -> +1, -2 -> -1, 0 -> 0, 1 -> sign(flux) (still deterministic from flux)
    if vec == 2:
        vec_dir = 1
    elif vec == -2:
        vec_dir = -1
    elif vec == 0:
        vec_dir = 0
    else:
        vec_dir = 1 if float(lx["flux"]) > 0 else (-1 if float(lx["flux"]) < 0 else 0)

    mat = int(np.sign(int(lx["mat_sig"])))  # -1, 0, +1

    smc_dir = 1 if bool(lx["smc_buy"]) else (-1 if bool(lx["smc_sell"]) else 0)

    return {
        "tf": tf,
        "symbol": rsym,
        "ts": lx["timestamp"],
        "close": float(lx["close"]),
        "flux": float(lx["flux"]),
        "br_trend": br,
        "vec_dir": vec_dir,
        "mat_dir": mat,
        "smc_dir": smc_dir,
        "gate": bool(lx["gate"]),
        "tci": float(lx["tci"]),
        "err": "",
    }

def mtf_consensus(exch: str, sym: str, tfs: list, lim: int):
    snaps = []
    for tf in tfs:
        ttl = TF_TTL.get(tf, 30)
        snaps.append(tf_signals_snapshot(exch, sym, tf, lim, ttl))

    # compute alignment score per TF and global
    # score per TF: sum of (br_trend, vec_dir, mat_dir, smc_dir) with br_trend already -1/0/1
    # normalized to [-1, +1] by /4
    rows = []
    for s in snaps:
        if s.get("err"):
            rows.append({"TF": s["tf"], "Close": np.nan, "Flux": np.nan, "Gate": "", "BR": "", "VEC": "", "MAT": "", "SMC": "", "Score": np.nan, "TS": "", "Err": s["err"]})
            continue
        br = int(s["br_trend"])
        vecd = int(s["vec_dir"])
        matd = int(s["mat_dir"])
        smcd = int(s["smc_dir"])
        score = (br + vecd + matd + smcd) / 4.0
        rows.append({
            "TF": s["tf"],
            "Close": float(s["close"]),
            "Flux": float(s["flux"]),
            "Gate": "SAFE" if bool(s["gate"]) else "CHAOS",
            "BR": br,
            "VEC": vecd,
            "MAT": matd,
            "SMC": smcd,
            "Score": float(score),
            "TS": str(s["ts"]),
            "Err": "",
        })

    out = pd.DataFrame(rows)
    valid = out[~out["Score"].isna()].copy()
    global_score = float(valid["Score"].mean()) if len(valid) else np.nan

    # Direction label from global_score deterministically
    if np.isnan(global_score):
        label = "VOID"
    elif global_score > 0.25:
        label = "BULL"
    elif global_score < -0.25:
        label = "BEAR"
    else:
        label = "NEUTRAL"

    return out, global_score, label

mtf_df, mtf_score, mtf_label = mtf_consensus(
    st.session_state.exch,
    resolved_sym,
    st.session_state.mtf_tfs,
    int(st.session_state.mtf_lim)
)


# ==========================================
# 8D. REPLAY + BACKTEST-LITE
# ==========================================
def build_signal_events_from_df(dfi: pd.DataFrame, family: str):
    # deterministic “event” extraction from existing signals (edge detect)
    evs = []

    def add(kind, i, direction):
        row = dfi.iloc[i]
        evs.append({
            "kind": kind,
            "dir": direction,
            "candle_ts": row["timestamp"],
            "close": float(row["close"]),
            "flux": float(row.get("flux", np.nan)),
            "tf": st.session_state.tf,
            "symbol": resolved_sym,
            "exchange": st.session_state.exch,
        })

    for i in range(1, len(dfi)):
        a = dfi.iloc[i - 1]
        b = dfi.iloc[i]

        if family in ["ALL", "BRAIN"]:
            if bool(b.get("br_buy", False)) and not bool(a.get("br_buy", False)):
                add("BRAIN_BUY", i, "bull")
            if bool(b.get("br_sell", False)) and not bool(a.get("br_sell", False)):
                add("BRAIN_SELL", i, "bear")

        if family in ["ALL", "SMC"]:
            if bool(b.get("smc_buy", False)) and not bool(a.get("smc_buy", False)):
                add("SMC_BUY", i, "bull")
            if bool(b.get("smc_sell", False)) and not bool(a.get("smc_sell", False)):
                add("SMC_SELL", i, "bear")

        if family in ["ALL", "VECTOR"]:
            if int(b.get("vec_state", 0)) == 2 and int(a.get("vec_state", 0)) != 2:
                add("VECTOR_SUPER", i, "bull")
            if int(b.get("vec_state", 0)) == -2 and int(a.get("vec_state", 0)) != -2:
                add("VECTOR_CRASH", i, "bear")

    return evs

def backtest_lite(dfi: pd.DataFrame):
    family = st.session_state.bt_signal_family
    horizon = int(st.session_state.bt_horizon)
    target = float(st.session_state.bt_target_pct) / 100.0
    stop = float(st.session_state.bt_stop_pct) / 100.0
    nmax = int(st.session_state.bt_n_signals)

    evs = build_signal_events_from_df(dfi, family)
    evs = evs[-nmax:]  # last N (chronologically)
    if not evs:
        return pd.DataFrame(), {"n": 0}

    rows = []
    for ev in evs:
        # locate index deterministically by timestamp equality
        idx = dfi.index[dfi["timestamp"] == ev["candle_ts"]]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        entry = float(dfi["close"].iloc[i])

        end = min(len(dfi) - 1, i + horizon)
        fw = dfi.iloc[i + 1 : end + 1]
        if fw.empty:
            continue

        if ev["dir"] == "bull":
            max_fav = float((fw["high"].max() - entry) / entry)
            max_adv = float((fw["low"].min() - entry) / entry)  # negative when adverse
            hit_target = max_fav >= target
            hit_stop = max_adv <= -stop
        else:
            # short: favorable is price down
            max_fav = float((entry - fw["low"].min()) / entry)
            max_adv = float((entry - fw["high"].max()) / entry)  # negative when adverse (since entry-high is negative)
            hit_target = max_fav >= target
            hit_stop = max_adv <= -stop

        # Determine outcome deterministically:
        # If both touched within horizon, we cannot infer order without intrabar path.
        # Mark as "BOTH" rather than guessing.
        if hit_target and hit_stop:
            outcome = "BOTH"
        elif hit_target:
            outcome = "TARGET"
        elif hit_stop:
            outcome = "STOP"
        else:
            outcome = "NONE"

        rows.append({
            "TS": str(ev["candle_ts"]),
            "Kind": ev["kind"],
            "Dir": ev["dir"],
            "Entry": entry,
            "MaxFav%": round(max_fav * 100.0, 3),
            "MaxAdv%": round(max_adv * 100.0, 3),
            "Target%": st.session_state.bt_target_pct,
            "Stop%": st.session_state.bt_stop_pct,
            "Outcome": outcome,
            "Horizon": horizon,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"n": 0}

    n = len(out)
    hit_t = int((out["Outcome"] == "TARGET").sum())
    hit_s = int((out["Outcome"] == "STOP").sum())
    both = int((out["Outcome"] == "BOTH").sum())
    none = int((out["Outcome"] == "NONE").sum())

    stats = {
        "n": n,
        "target_hits": hit_t,
        "stop_hits": hit_s,
        "both": both,
        "none": none,
        "target_hit_rate_excl_both": round((hit_t / max(1, (n - both))) * 100.0, 2),
        "avg_max_fav_pct": round(float(out["MaxFav%"].mean()), 3),
        "avg_max_adv_pct": round(float(out["MaxAdv%"].mean()), 3),
    }
    return out, stats


def replay_window_df(dfi: pd.DataFrame, center_ts, window: int):
    # Find nearest index deterministically:
    # 1) exact match preferred
    idx = dfi.index[dfi["timestamp"] == center_ts]
    if len(idx):
        i = int(idx[0])
    else:
        # nearest by absolute timediff
        diffs = (dfi["timestamp"] - pd.to_datetime(center_ts)).abs()
        i = int(diffs.idxmin())
    half = max(10, int(window) // 2)
    a = max(0, i - half)
    b = min(len(dfi), i + half)
    return dfi.iloc[a:b].copy(), i


# ==========================================
# 9. UI RENDER
# ==========================================
st.title(f"🔯 THE PENTAGRAM // {resolved_sym}  ·  {st.session_state.exch}  ·  {st.session_state.tf}")

h1, h2, h3, h4 = st.columns(4)
h1.metric("Live Price", f"{last['close']:.2f}")
h2.metric("Apex Flux", f"{last['flux']:.3f}", delta=("Bull" if last['flux'] > 0 else "Bear"))
h3.metric("Brain Gate", ("SAFE" if bool(last["gate"]) else "CHAOS"))
h4.metric("Matrix Sig", int(last["mat_sig"]))

# Quick MTF badge row (deterministic)
badge = f'<span class="pill pill-warn">MTF {mtf_label}</span><span class="pill">Score {mtf_score if not np.isnan(mtf_score) else "NaN"}</span>'
st.markdown(badge, unsafe_allow_html=True)

tabs = st.tabs([
    "🧠 Brain",
    "🏛️ Structure",
    "⚡ Vector",
    "⚛️ Quantum",
    "🧩 MTF Consensus",
    "📡 Scanner",
    "🛰️ Replay",
    "🧪 Backtest-lite",
    "📺 TV View",
    "🤖 AI Council",
])

def clean_plot(height=520):
    return go.Layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=height
    )

# ---- Brain
with tabs[0]:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[
            go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"])
        ])
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["br_u"], name="Upper", opacity=0.25))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["br_l_band"], name="Lower", opacity=0.25, fill="tonexty"))
        fig.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with r:
        st.markdown(
            f"""<div class="diag-panel">
            <div class="diag-header">Brain Diagnostics</div>
            <div class="diag-item"><span class="diag-label">Macro Bias:</span><span class="diag-val">{('BULLISH' if last['br_trend']==1 else ('BEARISH' if last['br_trend']==-1 else 'NEUTRAL'))}</span></div>
            <div class="diag-item"><span class="diag-label">Entropy:</span><span class="diag-val">{float(last['ent']):.2f}</span></div>
            <div class="diag-item"><span class="diag-label">Flow:</span><span class="diag-val">{float(last['flow']):.3f}</span></div>
            <div class="diag-item"><span class="diag-label">Signal:</span><span class="diag-val">{('BUY' if bool(last['br_buy']) else ('SELL' if bool(last['br_sell']) else 'WAIT'))}</span></div>
            <div style="margin-top:18px; color:#555; font-size:0.85rem;">
                Brain filters trend via bands + entropy gate. SAFE = low noise regime.
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

# ---- Structure
with tabs[1]:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[
            go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"])
        ])
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["smc_base"], name="SMC Base", line=dict(width=1)))
        fig.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with r:
        st.markdown(
            f"""<div class="diag-panel">
            <div class="diag-header">Structure Diagnostics</div>
            <div class="diag-item"><span class="diag-label">Signal:</span><span class="diag-val">{('BUY' if bool(last['smc_buy']) else ('SELL' if bool(last['smc_sell']) else 'WAIT'))}</span></div>
            <div class="diag-item"><span class="diag-label">TCI:</span><span class="diag-val">{float(last['tci']):.2f}</span></div>
            <div class="diag-item"><span class="diag-label">Bull FVG:</span><span class="diag-val">{bool(last['fvg_b'])}</span></div>
            <div class="diag-item"><span class="diag-label">Bear FVG:</span><span class="diag-val">{bool(last['fvg_s'])}</span></div>
            <div style="margin-top:18px; color:#555; font-size:0.85rem;">
                SMC watches base + WaveTrend turns + FVG presence.
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

# ---- Vector
with tabs[2]:
    l, r = st.columns([3, 1])
    with l:
        fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.02)
        fig_v.add_trace(
            go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]),
            row=1, col=1
        )
        # Keep your colors, but distinguish mild(1) vs neutral(0)
        bar_colors = []
        for s in df["vec_state"].astype(int).tolist():
            if s == 2:
                bar_colors.append("#00E676")
            elif s == -2:
                bar_colors.append("#FF1744")
            elif s == 0:
                bar_colors.append("#444")
            else:
                bar_colors.append("#222")

        fig_v.add_trace(go.Bar(x=df["timestamp"], y=df["flux"], marker_color=bar_colors, name="Flux"), row=2, col=1)
        fig_v.update_layout(clean_plot(height=560), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_v, use_container_width=True)

    with r:
        st.markdown(
            f"""<div class="diag-panel">
            <div class="diag-header">Physics Diagnostics</div>
            <div class="diag-item"><span class="diag-label">Efficiency:</span><span class="diag-val">{float(last['eff'])*100:.1f}%</span></div>
            <div class="diag-item"><span class="diag-label">Flux:</span><span class="diag-val">{float(last['flux']):.3f}</span></div>
            <div class="diag-item"><span class="diag-label">State:</span><span class="diag-val">{int(last['vec_state'])}</span></div>
            <div class="diag-item"><span class="diag-label">Div Bull:</span><span class="diag-val">{bool(last['div_bull'])}</span></div>
            <div class="diag-item"><span class="diag-label">Div Bear:</span><span class="diag-val">{bool(last['div_bear'])}</span></div>
            <div style="margin-top:18px; color:#555; font-size:0.85rem;">
                Vector = candle efficiency × relative volume → smoothed flux + state.
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

# ---- Quantum
with tabs[3]:
    l, r = st.columns([3, 1])
    with l:
        fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.02)
        fig_q.add_trace(
            go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"]),
            row=1, col=1
        )
        fig_q.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], name="RQZO"), row=2, col=1)
        fig_q.update_layout(clean_plot(height=560), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_q, use_container_width=True)

    with r:
        rq = float(last["rqzo"])
        rq_bias = "POSITIVE" if rq > 1 else ("NEGATIVE" if rq < -1 else "NEUTRAL")
        st.markdown(
            f"""<div class="diag-panel">
            <div class="diag-header">Quantum Diagnostics</div>
            <div class="diag-item"><span class="diag-label">RQZO:</span><span class="diag-val">{rq:.2f}</span></div>
            <div class="diag-item"><span class="diag-label">Bias:</span><span class="diag-val">{rq_bias}</span></div>
            <div style="margin-top:18px; color:#555; font-size:0.85rem;">
                RQZO is a synthetic oscillator derived from normalized price regime changes.
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

# ---- MTF Consensus
with tabs[4]:
    st.markdown("### 🧩 Multi-Timeframe Consensus (15m / 1h / 4h)")
    st.dataframe(mtf_df, use_container_width=True, hide_index=True)
    if not np.isnan(mtf_score):
        st.markdown(
            f"""
<div class="diag-panel" style="border-left-color:#FFD600; border-radius:8px;">
  <div class="diag-header">Consensus</div>
  <div class="diag-item"><span class="diag-label">Label:</span><span class="diag-val">{mtf_label}</span></div>
  <div class="diag-item"><span class="diag-label">Score:</span><span class="diag-val">{mtf_score:.3f}</span></div>
  <div style="margin-top:18px; color:#555; font-size:0.85rem;">
    Score per TF = (BR + VEC + MAT + SMC) / 4. Global score = mean across TFs.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.warning("MTF consensus is VOID (one or more TF fetches failed). Check exchange/symbol support.")

# ---- Scanner
with tabs[5]:
    st.markdown("### 📡 Scanner Mode (Flux Spike Leaderboard + Radar Timeline)")
    if not st.session_state.scanner_on:
        st.info("Scanner is OFF. Enable it in the sidebar.")
    else:
        if scanner_df is None:
            st.caption("Scanner has not run yet (toggle Auto-Pilot or click Run Scanner Now).")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("#### Leaderboard (sorted by |flux|)")
                st.dataframe(scanner_df, use_container_width=True, hide_index=True)
            with c2:
                st.markdown("#### Recorded This Run")
                st.write(len(scanner_recorded))
                for ev in scanner_recorded[:12]:
                    st.markdown(f"- {ev['msg']}")

        st.markdown("#### Radar Timeline (auto-recorded spikes)")
        if not st.session_state.radar_events:
            st.caption("No radar events recorded yet.")
        else:
            # show last 40
            radar_view = []
            for ev in st.session_state.radar_events[:40]:
                radar_view.append({
                    "TS": str(ev.get("candle_ts", "")),
                    "Symbol": ev.get("symbol", ""),
                    "TF": ev.get("tf", ""),
                    "Kind": ev.get("kind", ""),
                    "Flux": ev.get("flux", np.nan),
                    "Close": ev.get("close", np.nan),
                })
            st.dataframe(pd.DataFrame(radar_view), use_container_width=True, hide_index=True)

# ---- Replay
with tabs[6]:
    st.markdown("### 🛰️ Signal Replay (click → re-render candle context + optional AI “why”)")

    # Build unified replay catalog: radar events + main symbol events
    main_evs = build_signal_events_from_df(df, "ALL")
    # Convert radar events into same schema
    radar_evs = []
    for ev in st.session_state.radar_events[:250]:
        radar_evs.append({
            "kind": ev.get("kind", ""),
            "dir": "bull" if "BULL" in ev.get("kind", "") else ("bear" if "BEAR" in ev.get("kind", "") else ""),
            "candle_ts": ev.get("candle_ts", ""),
            "close": ev.get("close", np.nan),
            "flux": ev.get("flux", np.nan),
            "tf": ev.get("tf", ""),
            "symbol": ev.get("symbol", ""),
            "exchange": ev.get("exchange", ""),
        })

    catalog = []
    for ev in (radar_evs + main_evs[::-1]):  # radar first, then recent main
        key = f'{ev.get("exchange","")}::{ev.get("symbol","")}::{ev.get("tf","")}::{str(ev.get("candle_ts",""))}::{ev.get("kind","")}'
        label = f'{ev.get("kind","")} · {ev.get("symbol","")} · {ev.get("tf","")} · {str(ev.get("candle_ts",""))}'
        catalog.append((label, key, ev))

    if not catalog:
        st.caption("No events to replay yet.")
    else:
        labels = [x[0] for x in catalog]
        label = st.selectbox("Select Event", labels, index=0)
        chosen = next(x for x in catalog if x[0] == label)
        ev = chosen[2]

        # Use the event symbol/tf if radar event; else main
        replay_ex = ev.get("exchange", st.session_state.exch) or st.session_state.exch
        replay_sym = ev.get("symbol", resolved_sym) or resolved_sym
        replay_tf = ev.get("tf", st.session_state.tf) or st.session_state.tf

        ttl = TF_TTL.get(replay_tf, 30)
        replay_df, replay_resolved, replay_err = fetch_ohlcv(
            replay_ex,
            replay_sym,
            replay_tf,
            max(240, int(st.session_state.replay_window) + 60),
            ttl
        )
        if replay_df.empty:
            st.error("Replay fetch failed.")
            st.code(replay_err)
        else:
            # compute full cores for contextual “why”
            replay_df = calc_vector(replay_df, st.session_state)
            replay_df = calc_brain(replay_df, st.session_state)
            replay_df = calc_rqzo(replay_df, st.session_state)
            replay_df = calc_matrix(replay_df, st.session_state)
            replay_df = calc_smc(replay_df, st.session_state)

            wdf, center_i = replay_window_df(replay_df, ev["candle_ts"], int(st.session_state.replay_window))

            st.markdown(f"**Replay:** {replay_resolved} · {replay_ex} · {replay_tf}")
            fig = go.Figure(data=[
                go.Candlestick(x=wdf["timestamp"], open=wdf["open"], high=wdf["high"], low=wdf["low"], close=wdf["close"])
            ])
            # add key overlays
            fig.add_trace(go.Scatter(x=wdf["timestamp"], y=wdf["smc_base"], name="SMC Base", opacity=0.35))
            fig.add_trace(go.Scatter(x=wdf["timestamp"], y=wdf["br_u"], name="Brain Upper", opacity=0.18))
            fig.add_trace(go.Scatter(x=wdf["timestamp"], y=wdf["br_l_band"], name="Brain Lower", opacity=0.18))
            fig.update_layout(clean_plot(height=560), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # Snapshot at center candle (nearest)
            diffs = (replay_df["timestamp"] - pd.to_datetime(ev["candle_ts"])).abs()
            i = int(diffs.idxmin())
            row = replay_df.loc[i]

            st.markdown("#### Deterministic Context Snapshot")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Close", f"{float(row['close']):.2f}")
            c2.metric("Flux", f"{float(row['flux']):.3f}")
            c3.metric("BR Trend", int(row["br_trend"]))
            c4.metric("Gate", "SAFE" if bool(row["gate"]) else "CHAOS")

            st.markdown(
                f"""
<div class="diag-panel" style="border-left-color:#00d4ff; border-radius:8px;">
  <div class="diag-header">Signal State (at replay candle)</div>
  <div class="diag-item"><span class="diag-label">BRAIN buy/sell:</span><span class="diag-val">{bool(row.get("br_buy", False))}/{bool(row.get("br_sell", False))}</span></div>
  <div class="diag-item"><span class="diag-label">SMC buy/sell:</span><span class="diag-val">{bool(row.get("smc_buy", False))}/{bool(row.get("smc_sell", False))}</span></div>
  <div class="diag-item"><span class="diag-label">VEC state:</span><span class="diag-val">{int(row.get("vec_state", 0))}</span></div>
  <div class="diag-item"><span class="diag-label">MAT sig:</span><span class="diag-val">{int(row.get("mat_sig", 0))}</span></div>
  <div class="diag-item"><span class="diag-label">TCI:</span><span class="diag-val">{float(row.get("tci", 0.0)):.2f}</span></div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("#### AI “Why” (optional)")
            if st.button("Explain this replay candle", use_container_width=True):
                if not (OPENAI_AVAILABLE and st.session_state.ai_k):
                    st.error("OpenAI client not available or missing key.")
                else:
                    try:
                        c = OpenAI(api_key=st.session_state.ai_k)
                        prompt = f"""
You are a trading analyst. Explain *deterministically* what the dashboard shows at the replay candle.

Event:
- Kind: {ev.get("kind","")}
- Exchange: {replay_ex}
- Symbol: {replay_resolved}
- TF: {replay_tf}
- Candle TS: {str(pd.to_datetime(ev["candle_ts"]))}
- Close: {float(row["close"]):.6f}
- Flux: {float(row["flux"]):.6f}
- vec_state: {int(row.get("vec_state",0))}
- br_trend: {int(row.get("br_trend",0))}
- gate: {bool(row.get("gate",False))}
- mat_sig: {int(row.get("mat_sig",0))}
- smc_buy/smc_sell: {bool(row.get("smc_buy",False))}/{bool(row.get("smc_sell",False))}
- tci: {float(row.get("tci",0.0)):.6f}

Rules:
- No trade advice. Only interpret the computed signals and what conditions are satisfied.
- Output:
  1) One-line state summary (bull/bear/neutral wording allowed)
  2) Bullet list of which conditions are TRUE and which are FALSE (Brain/SMC/Vector/Matrix)
  3) One sentence on what would need to change for the opposite state.
""".strip()
                        r = c.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt}],
                        )
                        st.info(r.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI error: {e}")

# ---- Backtest-lite
with tabs[7]:
    st.markdown("### 🧪 Backtest-lite (last N signals, hit-rate + excursion)")
    bt_df, bt_stats = backtest_lite(df)
    if bt_df.empty:
        st.caption("No backtest rows (no signals or insufficient forward candles).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signals", bt_stats["n"])
        c2.metric("Target hits", bt_stats["target_hits"])
        c3.metric("Stop hits", bt_stats["stop_hits"])
        c4.metric("Hit-rate% (excl BOTH)", bt_stats["target_hit_rate_excl_both"])

        st.markdown(
            f"""
<div class="diag-panel" style="border-left-color:#FFD600; border-radius:8px;">
  <div class="diag-header">Excursion</div>
  <div class="diag-item"><span class="diag-label">Avg MaxFav%:</span><span class="diag-val">{bt_stats["avg_max_fav_pct"]}</span></div>
  <div class="diag-item"><span class="diag-label">Avg MaxAdv%:</span><span class="diag-val">{bt_stats["avg_max_adv_pct"]}</span></div>
  <div class="diag-item"><span class="diag-label">BOTH (target+stop):</span><span class="diag-val">{bt_stats["both"]}</span></div>
  <div class="diag-item"><span class="diag-label">NONE:</span><span class="diag-val">{bt_stats["none"]}</span></div>
  <div style="margin-top:18px; color:#555; font-size:0.85rem;">
    If BOTH occurs, intrabar order is unknowable from OHLCV; it is not classified as win/loss.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.dataframe(bt_df.sort_values("TS", ascending=False), use_container_width=True, hide_index=True)

# ---- TV View
with tabs[8]:
    st.markdown("### Official TradingView Confirmation")

    # Prefer explicit override
    if st.session_state.tv_symbol_override.strip():
        tv_symbol = st.session_state.tv_symbol_override.strip()
    else:
        # Minimal default guess (still imperfect; override recommended)
        # Binance often uses USDT; Kraken may need XBT.
        base_guess = resolved_sym.replace("/", "")
        tv_symbol = f"{st.session_state.exch.upper()}:{base_guess}"

    # Put div BEFORE script, and use unique container id
    container_id = "tv_widget"
    st.components.v1.html(
        f"""
    <div id="{container_id}"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
      new TradingView.widget({{
        "width": "100%",
        "height": 540,
        "symbol": "{tv_symbol}",
        "theme": "dark",
        "style": "1",
        "container_id": "{container_id}"
      }});
    </script>
    """,
        height=560,
    )

    st.caption(f"TradingView symbol used: {tv_symbol} (Override in sidebar if wrong.)")

# ---- AI Council
with tabs[9]:
    persona = st.selectbox("Council Member", ["Grand Strategist", "Physicist", "Banker", "The Quant"])
    prompt = f"""
You are {persona}. Given:
- Symbol: {resolved_sym} ({st.session_state.exch}, {st.session_state.tf})
- Price: {last['close']:.2f}
- Flux: {last['flux']:.3f}
- Brain trend: {int(last['br_trend'])}
- Gate SAFE?: {bool(last['gate'])}
- Matrix sig: {int(last['mat_sig'])}
- SMC buy/sell: {bool(last['smc_buy'])}/{bool(last['smc_sell'])}
Return:
1) Bias (bull/bear/neutral) + confidence (0-100)
2) 1 potential entry idea + invalidation
3) 1 risk note (what could go wrong)
Keep it short and concrete.
""".strip()

    if st.button("🔱 CONSULT THE COUNCIL", use_container_width=True):
        if not (OPENAI_AVAILABLE and st.session_state.ai_k):
            st.error("OpenAI client not available or missing key.")
        else:
            try:
                c = OpenAI(api_key=st.session_state.ai_k)
                r = c.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                )
                st.info(r.choices[0].message.content)
            except Exception as e:
                st.error(f"AI error: {e}")

    st.markdown("### Signal Log (Main + Scanner notes)")
    st.markdown('<div class="log-container">', unsafe_allow_html=True)
    for row in st.session_state.signal_log[:50]:
        st.markdown(
            f'<div class="log-entry"><span class="log-ts">{row["ts"]}</span><span class="log-msg">{row["msg"]}</span></div>',
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# Optional debug footer
if st.session_state.debug:
    st.caption(f"Resolved Symbol: {resolved_sym} · Candles: {len(df)} · Last candle: {last['timestamp']}")
```
