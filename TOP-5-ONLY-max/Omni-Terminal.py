
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
logger = logging.getLogger("Terminal")
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

st.markdown("""
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
</style>
""", unsafe_allow_html=True)


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
    return series.ewm(alpha=1/length, adjust=False).mean()

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
        (np.minimum(df["open"], df["close"]) - df["low"]) -
        (df["high"] - np.maximum(df["open"], df["close"]))
    ) / rng
    wick_balance = wick_balance.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["flow"] = pd.Series(
        wick_balance + ((df["close"] - df["open"]) / (rma(rng.fillna(0.0), 14) + 1e-10))
    ).ewm(span=34, adjust=False).mean()

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
    gamma = 1 / np.sqrt(1 - (v ** 2) + 1e-10)

    tau = (np.arange(len(df)) % win) / gamma.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    rq = np.zeros(len(df), dtype=float)
    for n in range(1, 10):
        rq += (n ** -0.5) * np.sin(tau * np.log(n))
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

    df["fvg_b"] = (df["low"] > df["high"].shift(2))
    df["fvg_s"] = (df["high"] < df["low"].shift(2))
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log_event(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    st.session_state.signal_log = ([{"ts": ts, "msg": msg}] + st.session_state.signal_log)[:200]

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
# 7. PROCESSING
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
# 9. UI RENDER
# ==========================================
st.title(f"🔯 THE PENTAGRAM // {resolved_sym}  ·  {st.session_state.exch}  ·  {st.session_state.tf}")

h1, h2, h3, h4 = st.columns(4)
h1.metric("Live Price", f"{last['close']:.2f}")
h2.metric("Apex Flux", f"{last['flux']:.3f}", delta=("Bull" if last['flux'] > 0 else "Bear"))
h3.metric("Brain Gate", ("SAFE" if bool(last["gate"]) else "CHAOS"))
h4.metric("Matrix Sig", int(last["mat_sig"]))

t1, t2, t3, t4, t5, t6 = st.tabs(["🧠 Brain", "🏛️ Structure", "⚡ Vector", "⚛️ Quantum", "📺 TV View", "🤖 AI Council"])

def clean_plot(height=520):
    return go.Layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=height
    )

# ---- Brain
with t1:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[
            go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
        ])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_u'], name="Upper", opacity=0.25))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['br_l_band'], name="Lower", opacity=0.25, fill='tonexty'))
        fig.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with r:
        st.markdown(f"""<div class="diag-panel">
            <div class="diag-header">Brain Diagnostics</div>
            <div class="diag-item"><span class="diag-label">Macro Bias:</span><span class="diag-val">{('BULLISH' if last['br_trend']==1 else ('BEARISH' if last['br_trend']==-1 else 'NEUTRAL'))}</span></div>
            <div class="diag-item"><span class="diag-label">Entropy:</span><span class="diag-val">{float(last['ent']):.2f}</span></div>
            <div class="diag-item"><span class="diag-label">Flow:</span><span class="diag-val">{float(last['flow']):.3f}</span></div>
            <div class="diag-item"><span class="diag-label">Signal:</span><span class="diag-val">{('BUY' if bool(last['br_buy']) else ('SELL' if bool(last['br_sell']) else 'WAIT'))}</span></div>
            <div style="margin-top:18px; color:#555; font-size:0.85rem;">
                Brain filters trend via bands + entropy gate. SAFE = low noise regime.
            </div>
        </div>""", unsafe_allow_html=True)

# ---- Structure
with t2:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[
            go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
        ])
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['smc_base'], name="SMC Base", line=dict(width=1)))
        fig.update_layout(clean_plot(), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with r:
        st.markdown(f"""<div class="diag-panel">
            <div class="diag-header">Structure Diagnostics</div>
            <div class="diag-item"><span class="diag-label">Signal:</span><span class="diag-val">{('BUY' if bool(last['smc_buy']) else ('SELL' if bool(last['smc_sell']) else 'WAIT'))}</span></div>
            <div class="diag-item"><span class="diag-label">TCI:</span><span class="diag-val">{float(last['tci']):.2f}</span></div>
            <div class="diag-item"><span class="diag-label">Bull FVG:</span><span class="diag-val">{bool(last['fvg_b'])}</span></div>
            <div class="diag-item"><span class="diag-label">Bear FVG:</span><span class="diag-val">{bool(last['fvg_s'])}</span></div>
            <div style="margin-top:18px; color:#555; font-size:0.85rem;">
                SMC watches base + WaveTrend turns + FVG presence.
            </div>
        </div>""", unsafe_allow_html=True)

# ---- Vector
with t3:
    l, r = st.columns([3, 1])
    with l:
        fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.02)
        fig_v.add_trace(
            go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']),
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

        fig_v.add_trace(go.Bar(x=df['timestamp'], y=df['flux'], marker_color=bar_colors, name="Flux"), row=2, col=1)
        fig_v.update_layout(clean_plot(height=560), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_v, use_container_width=True)

    with r:
        st.markdown(f"""<div class="diag-panel">
            <div class="diag-header">Physics Diagnostics</div>
            <div class="diag-item"><span class="diag-label">Efficiency:</span><span class="diag-val">{float(last['eff'])*100:.1f}%</span></div>
            <div class="diag-item"><span class="diag-label">Flux:</span><span class="diag-val">{float(last['flux']):.3f}</span></div>
            <div class="diag-item"><span class="diag-label">State:</span><span class="diag-val">{int(last['vec_state'])}</span></div>
            <div class="diag-item"><span class="diag-label">Div Bull:</span><span class="diag-val">{bool(last['div_bull'])}</span></div>
            <div class="diag-item"><span class="diag-label">Div Bear:</span><span class="diag-val">{bool(last['div_bear'])}</span></div>
            <div style="margin-top:18px; color:#555; font-size:0.85rem;">
                Vector = candle efficiency × relative volume → smoothed flux + state.
            </div>
        </div>""", unsafe_allow_html=True)

# ---- Quantum (was missing)
with t4:
    l, r = st.columns([3, 1])
    with l:
        fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.02)
        fig_q.add_trace(
            go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']),
            row=1, col=1
        )
        fig_q.add_trace(go.Scatter(x=df["timestamp"], y=df["rqzo"], name="RQZO"), row=2, col=1)
        fig_q.update_layout(clean_plot(height=560), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_q, use_container_width=True)

    with r:
        rq = float(last["rqzo"])
        rq_bias = "POSITIVE" if rq > 1 else ("NEGATIVE" if rq < -1 else "NEUTRAL")
        st.markdown(f"""<div class="diag-panel">
            <div class="diag-header">Quantum Diagnostics</div>
            <div class="diag-item"><span class="diag-label">RQZO:</span><span class="diag-val">{rq:.2f}</span></div>
            <div class="diag-item"><span class="diag-label">Bias:</span><span class="diag-val">{rq_bias}</span></div>
            <div style="margin-top:18px; color:#555; font-size:0.85rem;">
                RQZO is a synthetic oscillator derived from normalized price regime changes.
            </div>
        </div>""", unsafe_allow_html=True)

# ---- TV View
with t5:
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
    st.components.v1.html(f"""
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
    """, height=560)

    st.caption(f"TradingView symbol used: {tv_symbol} (Override in sidebar if wrong.)")

# ---- AI Council
with t6:
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

    st.markdown("### Signal Log")
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
