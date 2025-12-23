# app.py — THE PENTAGRAM | Omni-Terminal (Upgraded, Complete, Copy/Paste)
# Includes:
# - Scanner Mode (watchlist import + incremental batch scanning + flux spike leaderboard)
# - Auto-record top spikes as signal events -> Replay becomes "market radar timeline"
# - MTF Consensus Panel (15m/1h/4h alignment score)
# - Signal Replay (click event -> rebuild candle context + optional AI “why”)
# - Backtest-lite (last N signals -> hit-rate + MFE/MAE + TP/SL first-touch logic)
#
# Optional installs:
#   pip install streamlit-autorefresh tweepy openai
# Required:
#   pip install streamlit ccxt pandas numpy plotly requests

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests
import time as time_lib
import re
from typing import List, Dict, Tuple

# Optional: non-blocking refresh
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOR_AVAILABLE = True
except Exception:
    AUTOR_AVAILABLE = False

# Optional: X/Twitter
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except Exception:
    TWEEPY_AVAILABLE = False

# Optional: OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ============================================================
# 1) PAGE CONFIGURATION & UI THEME
# ============================================================
st.set_page_config(
    page_title="THE PENTAGRAM | Omni-Terminal",
    layout="wide",
    page_icon="🔯",
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
        --text: #e0e0e0;
    }

    .stApp { background-color: var(--bg); font-family: 'JetBrains Mono', monospace; color: var(--text); }

    div[data-testid="metric-container"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 14px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }

    .diag-panel {
        background: #050505;
        border-left: 4px solid var(--accent);
        padding: 18px;
        border-radius: 0 8px 8px 0;
        height: 100%;
    }
    .diag-header { font-size: 0.75rem; color: #555; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin-bottom: 12px; }
    .diag-item { margin-bottom: 10px; font-size: 0.9rem; border-bottom: 1px solid #111; padding-bottom: 6px; }
    .diag-label { color: #888; margin-right: 10px; }
    .diag-val { color: #fff; font-weight: 700; }

    .log-container {
        font-size: 0.75rem; color: #888; max-height: 280px; overflow-y: auto;
        border: 1px solid #222; padding: 10px; border-radius: 4px; background: #020202;
    }
    .log-entry { border-bottom: 1px solid #111; padding: 6px 0; display: flex; justify-content: space-between; gap: 12px; }
    .log-ts { color: #555; white-space: nowrap; }
    .log-msg { color: #bbb; }
    .chip {
        display:inline-block; padding:2px 10px; border:1px solid #222; border-radius:999px;
        font-size:0.75rem; color:#bbb; margin-right:6px; background:#050505;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2) CONNECTIVITY HANDLERS
# ============================================================
def send_telegram(token: str, chat_id: str, text: str) -> Tuple[bool, str]:
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

def post_x(consumer_key: str, consumer_secret: str, access_token: str, access_secret: str, text: str) -> Tuple[bool, str]:
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


# ============================================================
# 3) PENTAGRAM MATH ENGINE (ALL 5 CORES)
# ============================================================
def rma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(alpha=1/length, adjust=False).mean()

def wma(series: pd.Series, length: int) -> pd.Series:
    length = int(length)
    if length <= 1:
        return series
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(series: pd.Series, length: int) -> pd.Series:
    length = int(length)
    if length <= 2:
        return series
    return wma(2 * wma(series, length // 2) - wma(series, length), int(np.sqrt(length)))

def double_smooth(src: pd.Series, l1: int, l2: int) -> pd.Series:
    return src.ewm(span=max(1, int(l1)), adjust=False).mean().ewm(span=max(1, int(l2)), adjust=False).mean()

def calc_vector(df: pd.DataFrame, p: Dict) -> pd.DataFrame:
    df = df.copy()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()

    eff_raw = (body / rng).replace([np.inf, -np.inf], np.nan).fillna(0.0)
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
    df["vec_state"] = np.select(conditions, [2, -2, 0], default=1)

    df["pl"] = (df["flux"].shift(1) < df["flux"].shift(2)) & (df["flux"].shift(1) < df["flux"])
    df["ph"] = (df["flux"].shift(1) > df["flux"].shift(2)) & (df["flux"].shift(1) > df["flux"])

    lookback = 5
    df["div_bull"] = df["pl"] & (df["close"] < df["close"].shift(lookback)) & (df["flux"] > df["flux"].shift(lookback))
    df["div_bear"] = df["ph"] & (df["close"] > df["close"].shift(lookback)) & (df["flux"] < df["flux"].shift(lookback))
    return df

def calc_brain(df: pd.DataFrame, p: Dict) -> pd.DataFrame:
    df = df.copy()
    base = hma(df["close"], int(p["br_l"]))
    atr_like = rma((df["high"] - df["low"]).abs(), int(p["br_l"])).fillna(0.0)

    df["br_u"] = base + (atr_like * float(p["br_m"]))
    df["br_l_band"] = base - (atr_like * float(p["br_m"]))

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

def calc_rqzo(df: pd.DataFrame, p: Dict) -> pd.DataFrame:
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

def calc_matrix(df: pd.DataFrame, p: Dict) -> pd.DataFrame:
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

def calc_smc(df: pd.DataFrame, p: Dict) -> pd.DataFrame:
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

def compute_cores(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    df = calc_vector(df, params)
    df = calc_brain(df, params)
    df = calc_rqzo(df, params)
    df = calc_matrix(df, params)
    df = calc_smc(df, params)
    return df


# ============================================================
# 4) DATA HANDLING (cached exchange + symbol resolution + retries)
# ============================================================
@st.cache_resource
def get_exchange(exch_name: str):
    ex_class = getattr(ccxt, exch_name.lower())
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()
    return ex

def resolve_symbol(ex, sym: str) -> str:
    sym = sym.strip().upper()
    if sym in ex.symbols:
        return sym

    candidates = [sym]
    if sym.endswith("/USD"):
        candidates.append(sym.replace("/USD", "/USDT"))
        candidates.append(sym.replace("/USD", "/USD:USD"))
    if sym.endswith("/USDT"):
        candidates.append(sym.replace("/USDT", "/USD"))

    candidates.append(sym.replace("BTC/", "XBT/"))  # Kraken alias attempt

    for c in candidates:
        if c in ex.symbols:
            return c

    if "/" not in sym:
        for guess in [sym[:-4] + "/" + sym[-4:], sym[:-3] + "/" + sym[-3:]]:
            if guess in ex.symbols:
                return guess

    raise ValueError(f"Unsupported symbol on {ex.id}: {sym}")

@st.cache_data(show_spinner=False, ttl=20)
def fetch_ohlcv(exch_name: str, sym: str, tf: str, lim: int, _cache_buster: int):
    ex = get_exchange(exch_name)
    sym2 = resolve_symbol(ex, sym)

    last_err = None
    for _ in range(3):
        try:
            ohlcv = ex.fetch_ohlcv(sym2, tf, limit=int(lim))
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
            return df, sym2, ""
        except Exception as e:
            last_err = str(e)
            time_lib.sleep(0.35)
    return pd.DataFrame(), sym, last_err or "unknown error"

def latest_core_state(exch: str, symbol: str, tf: str, lim: int, params: Dict):
    df, sym2, err = fetch_ohlcv(exch, symbol, tf, lim, int(time_lib.time() // 20))
    if df.empty:
        raise RuntimeError(err or "fetch failed")
    df = compute_cores(df, params)
    last = df.iloc[-1]
    return df, sym2, last


# ============================================================
# 5) WATCHLIST / SCANNER / EVENTS / BACKTEST HELPERS
# ============================================================
COMMON_QUOTES = ["USDT", "USD", "USDC", "BTC", "ETH", "EUR", "GBP"]

def parse_watchlist_text(raw: str) -> List[str]:
    if not raw:
        return []
    tokens = re.split(r"[,\n\r\t]+", raw)
    out = []
    for t in tokens:
        t = t.strip().upper()
        if t:
            out.append(t)
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

def tv_token_to_ccxt_symbol(token: str, quote_priority: List[str]) -> str:
    t = token.strip().upper()
    if ":" in t:
        _, t = t.split(":", 1)
        t = t.strip().upper()

    if "/" in t:
        return t

    for q in quote_priority:
        if t.endswith(q) and len(t) > len(q):
            base = t[:-len(q)]
            return f"{base}/{q}"
    return t

def normalize_watchlist_for_exchange(ex, tokens: List[str], quote_priority: List[str]) -> Tuple[List[str], List[str]]:
    resolved, rejected = [], []
    for tok in tokens:
        ccxt_sym = tv_token_to_ccxt_symbol(tok, quote_priority=quote_priority)
        try:
            sym2 = resolve_symbol(ex, ccxt_sym)
            if sym2 not in resolved:
                resolved.append(sym2)
        except Exception:
            rejected.append(tok)
    return resolved, rejected

def flux_spike_score(df: pd.DataFrame, window: int = 120) -> Tuple[float, float]:
    flux = df["flux"].astype(float)
    tail = flux.tail(int(window))
    mu = float(tail.mean())
    sd = float(tail.std(ddof=0)) + 1e-10
    last = float(flux.iloc[-1])
    z = (last - mu) / sd
    return last, float(z)

def extract_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = df["timestamp"]
    out["br_buy_evt"] = df["br_buy"] & (~df["br_buy"].shift(1).fillna(False))
    out["br_sell_evt"] = df["br_sell"] & (~df["br_sell"].shift(1).fillna(False))
    out["smc_buy_evt"] = df["smc_buy"] & (~df["smc_buy"].shift(1).fillna(False))
    out["smc_sell_evt"] = df["smc_sell"] & (~df["smc_sell"].shift(1).fillna(False))
    out["vec_super_evt"] = (df["vec_state"].astype(int) == 2) & (df["vec_state"].shift(1).astype(float) != 2)
    out["vec_crash_evt"] = (df["vec_state"].astype(int) == -2) & (df["vec_state"].shift(1).astype(float) != -2)
    return out

def backtest_lite(df: pd.DataFrame, signal_col: str, direction: int,
                  last_n: int = 50, horizon: int = 30, tp_pct: float = 0.8, sl_pct: float = 0.5) -> pd.DataFrame:
    sigs = extract_signals(df)
    idxs = sigs.index[sigs[signal_col].fillna(False)].tolist()
    if not idxs:
        return pd.DataFrame()

    idxs = idxs[-int(last_n):]
    rows = []
    for ix in idxs:
        i = int(df.index.get_loc(ix))
        if i >= len(df) - 2:
            continue

        entry = float(df.loc[ix, "close"])
        fwd_end = min(len(df) - 1, i + int(horizon))

        fwd_high = df["high"].iloc[i+1:fwd_end+1].astype(float)
        fwd_low = df["low"].iloc[i+1:fwd_end+1].astype(float)
        fwd_close = float(df["close"].iloc[fwd_end])

        if direction == 1:
            mfe = ((float(fwd_high.max()) - entry) / entry) * 100.0 if len(fwd_high) else 0.0
            mae = ((float(fwd_low.min()) - entry) / entry) * 100.0 if len(fwd_low) else 0.0
            ret = ((fwd_close - entry) / entry) * 100.0
        else:
            mfe = ((entry - float(fwd_low.min())) / entry) * 100.0 if len(fwd_low) else 0.0
            mae = ((entry - float(fwd_high.max())) / entry) * 100.0 if len(fwd_high) else 0.0
            ret = ((entry - fwd_close) / entry) * 100.0

        hit = None
        if direction == 1:
            tp = entry * (1 + tp_pct / 100.0)
            sl = entry * (1 - sl_pct / 100.0)
            for j in range(i+1, fwd_end+1):
                hi = float(df["high"].iloc[j]); lo = float(df["low"].iloc[j])
                if hi >= tp:
                    hit = "TP"; break
                if lo <= sl:
                    hit = "SL"; break
        else:
            tp = entry * (1 - tp_pct / 100.0)
            sl = entry * (1 + sl_pct / 100.0)
            for j in range(i+1, fwd_end+1):
                hi = float(df["high"].iloc[j]); lo = float(df["low"].iloc[j])
                if lo <= tp:
                    hit = "TP"; break
                if hi >= sl:
                    hit = "SL"; break

        rows.append({
            "ts": str(df["timestamp"].iloc[i]),
            "entry": entry,
            "horizon_close": fwd_close,
            "ret_%": ret,
            "mfe_%": mfe,
            "mae_%": mae,
            "hit": hit or "NONE",
        })

    return pd.DataFrame(rows)

def direction_from_row(r: pd.Series) -> int:
    vec = int(r["vec_state"])
    flux = float(r["flux"])
    brain = int(r["brain_trend"])
    smc_buy = bool(r.get("smc_buy", False))
    smc_sell = bool(r.get("smc_sell", False))
    mat = int(r.get("mat_sig", 0))

    if vec == 2 or flux > 0.25:
        vec_dir = 1
    elif vec == -2 or flux < -0.25:
        vec_dir = -1
    else:
        vec_dir = 0

    smc_dir = 1 if smc_buy else (-1 if smc_sell else 0)
    mat_dir = 1 if mat > 0 else (-1 if mat < 0 else 0)

    raw = 0.45 * brain + 0.30 * vec_dir + 0.15 * smc_dir + 0.10 * mat_dir
    gate_safe = bool(r.get("gate_safe", True))
    if not gate_safe:
        raw *= 0.55

    if raw > 0.20:
        return 1
    if raw < -0.20:
        return -1
    return 0

def consensus_alignment(exch: str, symbol: str, tfs: List[str], lim: int, params: Dict):
    per = []
    for tf in tfs:
        df, sym2, last = latest_core_state(exch, symbol, tf, lim, params)
        per.append({
            "tf": tf,
            "ts": str(last["timestamp"]),
            "close": float(last["close"]),
            "flux": float(last["flux"]),
            "vec_state": int(last["vec_state"]),
            "brain_trend": int(last["br_trend"]),
            "gate_safe": bool(last["gate"]),
            "mat_sig": int(last["mat_sig"]),
            "smc_buy": bool(last["smc_buy"]),
            "smc_sell": bool(last["smc_sell"]),
        })
    dfp = pd.DataFrame(per)
    dfp["dir"] = dfp.apply(lambda rr: direction_from_row(rr), axis=1)
    weights = {"15m": 0.25, "1h": 0.35, "4h": 0.40}
    w = np.array([weights.get(tf, 0.33) for tf in dfp["tf"]], dtype=float)
    w = w / (w.sum() + 1e-10)
    align = float((dfp["dir"].astype(float).values * w).sum()) * 100.0
    return dfp, align


# ============================================================
# 6) STATE / EVENTS / DEDUPE
# ============================================================
def init_state():
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

        "tv_symbol_override": "",

        "signal_log": [],
        "last_alert_ts": {},

        # Watchlist + scanner
        "watchlist_text": "",
        "watchlist_symbols": [],
        "watchlist_rejected": [],
        "scan_results": None,         # DataFrame
        "scan_queue": [],             # list[str]
        "scan_index": 0,              # progress pointer
        "scan_tf": "15m",
        "scan_lim": 260,
        "scan_spike_win": 120,
        "scan_batch_size": 20,
        "scan_running": False,
        "scan_autopilot": False,

        # Radar auto-record controls
        "radar_on": True,
        "radar_topk": 15,
        "radar_minz": 2.0,
        "radar_safe_only": False,
        "radar_last_record_ts": {},   # key=EXCH|SYM|TF -> last ts

        # Replay timeline events (list of dicts)
        "signal_history": [],         # newest first
        "replay_search": "",
        "replay_filter_radar": False,
        "replay_filter_core": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log_event(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    st.session_state.signal_log = ([{"ts": ts, "msg": msg}] + st.session_state.signal_log)[:300]

def should_alert(signal_name: str, candle_ts: pd.Timestamp) -> bool:
    ts = str(candle_ts)
    prev_ts = st.session_state.last_alert_ts.get(signal_name)
    if prev_ts == ts:
        return False
    st.session_state.last_alert_ts[signal_name] = ts
    return True

def record_event(ev: Dict):
    # Strong dedupe by id
    eid = ev.get("id")
    if not eid:
        return
    if any(x.get("id") == eid for x in st.session_state.signal_history):
        return
    st.session_state.signal_history = [ev] + st.session_state.signal_history
    st.session_state.signal_history = st.session_state.signal_history[:1200]

def record_core_signal(exch: str, symbol: str, tf: str, sig: str, last_row: pd.Series):
    ts = str(last_row["timestamp"])
    eid = f"{exch}|{symbol}|{tf}|{sig}|{ts}"
    ev = {
        "id": eid,
        "ts": ts,
        "exch": exch,
        "symbol": symbol,
        "tf": tf,
        "signal": sig,
        "price": float(last_row["close"]),
        "flux": float(last_row["flux"]),
        "flux_z": 0.0,
        "brain_trend": int(last_row["br_trend"]),
        "gate_safe": bool(last_row["gate"]),
        "mat_sig": int(last_row["mat_sig"]),
        "smc_buy": bool(last_row["smc_buy"]),
        "smc_sell": bool(last_row["smc_sell"]),
    }
    record_event(ev)

def record_radar_spike(exch: str, symbol: str, tf: str, row: Dict):
    ts = str(row.get("ts", "")).strip()
    if not ts:
        return

    dedupe_key = f"{exch}|{symbol}|{tf}"
    if st.session_state.radar_last_record_ts.get(dedupe_key) == ts:
        return
    st.session_state.radar_last_record_ts[dedupe_key] = ts

    z = float(row.get("flux_z", 0.0))
    direction = "UP" if z >= 0 else "DOWN"
    sig = f"radar_spike_{direction}"
    eid = f"{exch}|{symbol}|{tf}|{sig}|{ts}"

    ev = {
        "id": eid,
        "ts": ts,
        "exch": exch,
        "symbol": symbol,
        "tf": tf,
        "signal": sig,
        "price": float(row.get("close", np.nan)),
        "flux": float(row.get("flux", 0.0)),
        "flux_z": float(z),
        "brain_trend": int(row.get("brain_trend", 0)) if pd.notna(row.get("brain_trend", 0)) else 0,
        "gate_safe": bool(row.get("gate_safe", True)) if pd.notna(row.get("gate_safe", True)) else True,
        "mat_sig": int(row.get("mat_sig", 0)) if pd.notna(row.get("mat_sig", 0)) else 0,
        "smc_buy": bool(row.get("smc_buy", False)) if pd.notna(row.get("smc_buy", False)) else False,
        "smc_sell": bool(row.get("smc_sell", False)) if pd.notna(row.get("smc_sell", False)) else False,
    }
    record_event(ev)

def auto_record_top_spikes(scan_df: pd.DataFrame, exch: str, tf: str,
                           top_k: int, min_abs_z: float, only_gate_safe: bool):
    if scan_df is None or scan_df.empty:
        return
    df = scan_df.copy()
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df.dropna(subset=["flux_z", "symbol", "close", "ts"])
    df["abs_z"] = df["flux_z"].abs()
    df = df[df["abs_z"] >= float(min_abs_z)]
    if only_gate_safe and "gate_safe" in df.columns:
        df = df[df["gate_safe"] == True]
    df = df.sort_values("abs_z", ascending=False).head(int(top_k))
    for _, r in df.iterrows():
        record_radar_spike(exch, str(r["symbol"]), tf, r.to_dict())


init_state()


# ============================================================
# 7) SIDEBAR CONTROL DECK
# ============================================================
with st.sidebar:
    st.markdown("### 🔯 CONTROL DECK")

    with st.expander("🌍 Feed Configuration", expanded=True):
        exch_list = ["Kraken", "Binance", "Bybit", "Coinbase", "OKX"]
        st.session_state.exch = st.selectbox(
            "Exchange",
            exch_list,
            index=exch_list.index(st.session_state.exch) if st.session_state.exch in exch_list else 0
        )
        st.session_state.sym = st.text_input("Asset Ticker (CCXT format)", st.session_state.sym).strip().upper()

        tf_list = ["1m", "5m", "15m", "1h", "4h", "1d"]
        st.session_state.tf = st.selectbox(
            "Interval",
            tf_list,
            index=tf_list.index(st.session_state.tf) if st.session_state.tf in tf_list else 2
        )
        st.session_state.lim = st.slider("Candles (main)", 200, 2500, int(st.session_state.lim), step=50)

        st.session_state.auto = st.toggle("🔄 Auto-Pilot (60s)", value=bool(st.session_state.auto))
        st.session_state.debug = st.toggle("🧪 Debug Diagnostics", value=bool(st.session_state.debug))

        if st.session_state.auto and AUTOR_AVAILABLE:
            st_autorefresh(interval=60 * 1000, key="pentagram_autorefresh_main")
        elif st.session_state.auto and not AUTOR_AVAILABLE:
            st.warning("Auto-Pilot requires `streamlit-autorefresh`. Install it or disable Auto-Pilot.")

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

    if st.button("🔱 RELOAD", type="primary", use_container_width=True):
        fetch_ohlcv.clear()
        st.rerun()


# ============================================================
# 8) MAIN SYMBOL PROCESSING + ALERTS
# ============================================================
df_main, resolved_sym, fetch_err = fetch_ohlcv(
    st.session_state.exch,
    st.session_state.sym,
    st.session_state.tf,
    int(st.session_state.lim),
    int(time_lib.time() // 20),
)

if df_main.empty:
    st.error("VOID DETECTED: Data fetch failed.")
    st.caption(f"Exchange={st.session_state.exch}  Symbol={st.session_state.sym}  TF={st.session_state.tf}")
    if fetch_err:
        st.code(fetch_err)
    st.stop()

df_main = compute_cores(df_main, st.session_state)
last, prev = df_main.iloc[-1], df_main.iloc[-2]

# Alerts (edge-detected + per-candle dedupe)
events = []
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

for sig, msg in events:
    if should_alert(sig, last["timestamp"]):
        record_core_signal(st.session_state.exch, resolved_sym, st.session_state.tf, sig, last)
        log_event(msg)

        ok_tg, err_tg = send_telegram(st.session_state.tg_t, st.session_state.tg_c, msg)
        if st.session_state.debug and (st.session_state.tg_t or st.session_state.tg_c):
            log_event("TG ✅" if ok_tg else f"TG ❌ {err_tg}")

        ok_x, err_x = post_x(st.session_state.x_k, st.session_state.x_s, st.session_state.x_at, st.session_state.x_as, msg)
        if st.session_state.debug and (st.session_state.x_k or st.session_state.x_at):
            log_event("X ✅" if ok_x else f"X ❌ {err_x}")


# ============================================================
# 9) TOP HUD
# ============================================================
st.title(f"🔯 THE PENTAGRAM // {resolved_sym}  ·  {st.session_state.exch}  ·  {st.session_state.tf}")

h1, h2, h3, h4 = st.columns(4)
h1.metric("Live Price", f"{last['close']:.2f}")
h2.metric("Apex Flux", f"{last['flux']:.3f}", delta=("Bull" if last['flux'] > 0 else "Bear"))
h3.metric("Brain Gate", ("SAFE" if bool(last["gate"]) else "CHAOS"))
h4.metric("Matrix Sig", int(last["mat_sig"]))


# ============================================================
# 10) UI TABS
# ============================================================
t_scanner, t_consensus, t_replay, t_backtest, t_brain, t_struct, t_vector, t_quant, t_tv, t_ai = st.tabs(
    ["📡 Scanner", "🧭 MTF Consensus", "🎞 Signal Replay", "🧪 Backtest-lite",
     "🧠 Brain", "🏛️ Structure", "⚡ Vector", "⚛️ Quantum", "📺 TV View", "🤖 AI Council"]
)

def clean_plot(height=560):
    return go.Layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=height
    )

def plot_replay_context(df: pd.DataFrame, event_ts: str, title: str):
    ts = pd.to_datetime(event_ts)
    idx = (df["timestamp"] - ts).abs().idxmin()
    i = int(df.index.get_loc(idx)) if idx in df.index else len(df) - 1

    left = max(0, i - 80)
    right = min(len(df), i + 20)
    d = df.iloc[left:right].copy()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.02)
    fig.add_trace(
        go.Candlestick(x=d["timestamp"], open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="Price"),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["br_u"], name="Brain Upper", opacity=0.25), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["br_l_band"], name="Brain Lower", opacity=0.25, fill="tonexty"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["smc_base"], name="SMC Base"), row=1, col=1)

    colors = []
    for s in d["vec_state"].astype(int).tolist():
        if s == 2:
            colors.append("#00E676")
        elif s == -2:
            colors.append("#FF1744")
        elif s == 0:
            colors.append("#444")
        else:
            colors.append("#222")
    fig.add_trace(go.Bar(x=d["timestamp"], y=d["flux"], marker_color=colors, name="Flux"), row=2, col=1)

    event_x = df.loc[idx, "timestamp"]
    fig.add_vline(x=event_x, line_width=2, line_dash="dash", line_color="#D500F9")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=620,
        title=title,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 11) SCANNER TAB (Incremental batch scan + radar auto-record)
# ============================================================
def scanner_scan_batch(exch_name: str, symbols: List[str], tf: str, lim: int, params: Dict, spike_window: int, start_idx: int, batch_size: int) -> pd.DataFrame:
    ex = get_exchange(exch_name)
    sleep_s = max(0.05, float(getattr(ex, "rateLimit", 200)) / 1000.0)

    rows = []
    end = min(len(symbols), start_idx + batch_size)
    prog = st.progress(0, text=f"Scanning batch {start_idx+1}-{end}/{len(symbols)}…")
    denom = max(1, (end - start_idx))

    for k, sym in enumerate(symbols[start_idx:end], start=0):
        try:
            df, sym2, err = fetch_ohlcv(exch_name, sym, tf, lim, int(time_lib.time() // 20))
            if df.empty:
                raise RuntimeError(err or "fetch failed")

            df = compute_cores(df, params)
            last_row = df.iloc[-1]
            flux_last, z = flux_spike_score(df, window=spike_window)

            rows.append({
                "symbol": sym2,
                "close": float(last_row["close"]),
                "flux": float(flux_last),
                "flux_z": float(z),
                "vec_state": int(last_row["vec_state"]),
                "brain_trend": int(last_row["br_trend"]),
                "gate_safe": bool(last_row["gate"]),
                "mat_sig": int(last_row["mat_sig"]),
                "smc_buy": bool(last_row["smc_buy"]),
                "smc_sell": bool(last_row["smc_sell"]),
                "ts": str(last_row["timestamp"]),
                "error": None,
            })
        except Exception as e:
            rows.append({
                "symbol": sym,
                "close": np.nan,
                "flux": np.nan,
                "flux_z": np.nan,
                "vec_state": np.nan,
                "brain_trend": np.nan,
                "gate_safe": np.nan,
                "mat_sig": np.nan,
                "smc_buy": np.nan,
                "smc_sell": np.nan,
                "ts": "",
                "error": str(e)[:220],
            })

        prog.progress((k + 1) / denom, text=f"Scanning {k+1}/{denom}: {sym}")
        time_lib.sleep(sleep_s)

    prog.empty()
    out = pd.DataFrame(rows)
    out["abs_z"] = out["flux_z"].abs()
    out = out.sort_values("abs_z", ascending=False, na_position="last").reset_index(drop=True)
    return out


with t_scanner:
    st.markdown("## 📡 Scanner Mode — Flux Spike Leaderboard")
    st.caption("Paste/upload watchlist, resolve to the selected CCXT exchange, then scan in batches (fast, stable).")

    ex = get_exchange(st.session_state.exch)

    left, right = st.columns([2, 1])

    with left:
        uploaded = st.file_uploader("Upload TradingView watchlist export (TXT)", type=["txt"])
        if uploaded is not None:
            try:
                st.session_state.watchlist_text = uploaded.read().decode("utf-8", errors="ignore")
            except Exception:
                st.session_state.watchlist_text = ""
        st.session_state.watchlist_text = st.text_area(
            "Or paste symbols here (comma/newline separated)",
            value=st.session_state.watchlist_text,
            height=120,
            placeholder="BINANCE:BTCUSDT, COINBASE:BTCUSD\nBTC/USDT\nETH/USDT"
        )

    with right:
        quote_priority = st.multiselect("Quote split priority", COMMON_QUOTES, default=COMMON_QUOTES)
        st.session_state.scan_tf = st.selectbox("Scan timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"],
                                                index=["1m","5m","15m","1h","4h","1d"].index(st.session_state.scan_tf) if st.session_state.scan_tf in ["1m","5m","15m","1h","4h","1d"] else 2)
        st.session_state.scan_lim = st.slider("Candles per symbol", 150, 800, int(st.session_state.scan_lim), step=10)
        st.session_state.scan_spike_win = st.slider("Spike window (flux)", 60, 240, int(st.session_state.scan_spike_win), step=10)
        st.session_state.scan_batch_size = st.slider("Batch size", 5, 60, int(st.session_state.scan_batch_size), step=5)

    tokens = parse_watchlist_text(st.session_state.watchlist_text)
    if tokens:
        resolved, rejected = normalize_watchlist_for_exchange(ex, tokens, quote_priority=quote_priority)
        st.session_state.watchlist_symbols = resolved
        st.session_state.watchlist_rejected = rejected

        st.caption(f"Parsed: {len(tokens)} · Resolved on {st.session_state.exch}: {len(resolved)} · Rejected: {len(rejected)}")
        if rejected:
            with st.expander("Rejected tokens (could not resolve on this exchange)"):
                st.write(rejected)
    else:
        st.info("Paste or upload a watchlist to enable scanning.")
        st.stop()

    if len(st.session_state.watchlist_symbols) < 5:
        st.warning("Resolved watchlist is too small. Use CCXT-style symbols (e.g., BTC/USDT) or pick a different exchange.")
        st.stop()

    st.markdown("### Radar Timeline (auto-record spikes)")
    cA, cB, cC, cD, cE = st.columns([1,1,1,1,1])
    with cA:
        st.session_state.radar_on = st.toggle("📌 Auto-record", value=bool(st.session_state.radar_on))
    with cB:
        st.session_state.radar_topk = st.slider("Top K", 5, 50, int(st.session_state.radar_topk), step=5)
    with cC:
        st.session_state.radar_minz = st.slider("Min |z|", 0.5, 6.0, float(st.session_state.radar_minz), step=0.1)
    with cD:
        st.session_state.radar_safe_only = st.toggle("SAFE only", value=bool(st.session_state.radar_safe_only))
    with cE:
        st.session_state.scan_autopilot = st.toggle("🤖 Scan Autopilot", value=bool(st.session_state.scan_autopilot))

    st.markdown("### Scanner Controls")
    b1, b2, b3, b4 = st.columns(4)

    def reset_scan():
        st.session_state.scan_results = None
        st.session_state.scan_queue = st.session_state.watchlist_symbols[:]
        st.session_state.scan_index = 0
        st.session_state.scan_running = True

    with b1:
        if st.button("▶️ Start/Restart Scan", use_container_width=True, type="primary"):
            reset_scan()
            log_event(f"📡 Scan started: {len(st.session_state.scan_queue)} symbols  tf={st.session_state.scan_tf}")

    with b2:
        if st.button("⏭ Scan Next Batch", use_container_width=True):
            if not st.session_state.scan_queue:
                reset_scan()
            else:
                st.session_state.scan_running = True

    with b3:
        if st.button("⏹ Stop Scan", use_container_width=True):
            st.session_state.scan_running = False
            st.session_state.scan_autopilot = False

    with b4:
        if st.button("🧹 Clear Leaderboard", use_container_width=True):
            st.session_state.scan_results = None
            st.session_state.scan_index = 0
            st.session_state.scan_running = False

    # Autopilot refresh (only while scanning)
    if st.session_state.scan_autopilot and st.session_state.scan_running and AUTOR_AVAILABLE:
        st_autorefresh(interval=15 * 1000, key="scanner_autorefresh")
    elif st.session_state.scan_autopilot and st.session_state.scan_running and not AUTOR_AVAILABLE:
        st.warning("Scanner Autopilot requires `streamlit-autorefresh`.")

    # Run one batch per rerun when scan_running is True
    if st.session_state.scan_running:
        queue = st.session_state.watchlist_symbols[:]  # always use resolved list
        if st.session_state.scan_index >= len(queue):
            st.session_state.scan_running = False
            st.session_state.scan_autopilot = False
            log_event("📡 Scan complete.")
        else:
            batch_df = scanner_scan_batch(
                st.session_state.exch,
                queue,
                st.session_state.scan_tf,
                int(st.session_state.scan_lim),
                st.session_state,
                int(st.session_state.scan_spike_win),
                int(st.session_state.scan_index),
                int(st.session_state.scan_batch_size),
            )
            st.session_state.scan_index += int(st.session_state.scan_batch_size)

            # Merge into leaderboard
            if st.session_state.scan_results is None or not isinstance(st.session_state.scan_results, pd.DataFrame):
                merged = batch_df
            else:
                merged = pd.concat([st.session_state.scan_results, batch_df], ignore_index=True)

            # Deduplicate by symbol (keep best abs_z latest)
            merged["abs_z"] = merged["flux_z"].abs()
            merged = merged.sort_values(["symbol", "abs_z"], ascending=[True, False], na_position="last")
            merged = merged.drop_duplicates(subset=["symbol"], keep="first")
            merged = merged.sort_values("abs_z", ascending=False, na_position="last").reset_index(drop=True)
            st.session_state.scan_results = merged

            # Auto-record radar events from current merged leaderboard (top_k etc.)
            if st.session_state.radar_on:
                auto_record_top_spikes(
                    st.session_state.scan_results,
                    exch=st.session_state.exch,
                    tf=st.session_state.scan_tf,
                    top_k=int(st.session_state.radar_topk),
                    min_abs_z=float(st.session_state.radar_minz),
                    only_gate_safe=bool(st.session_state.radar_safe_only),
                )

    # Show leaderboard
    if st.session_state.scan_results is not None and isinstance(st.session_state.scan_results, pd.DataFrame):
        df_scan = st.session_state.scan_results.copy()
        topn = st.slider("Show Top N", 10, 100, 30, step=5)
        st.markdown(f"### Leaderboard (Top {topn})")
        show = df_scan.head(int(topn))
        st.dataframe(
            show[["symbol", "close", "flux", "flux_z", "vec_state", "brain_trend", "gate_safe", "mat_sig", "smc_buy", "smc_sell", "ts", "error"]],
            use_container_width=True,
            hide_index=True
        )

        st.caption(f"Progress: {min(st.session_state.scan_index, len(st.session_state.watchlist_symbols))}/{len(st.session_state.watchlist_symbols)} scanned")

        valid_syms = show["symbol"].dropna().tolist()
        if valid_syms:
            pick = st.selectbox("Quick: Set main chart symbol", valid_syms)
            if st.button("Set symbol", use_container_width=True):
                st.session_state.sym = pick
                st.success(f"Main symbol set: {pick}")


# ============================================================
# 12) MTF CONSENSUS TAB
# ============================================================
with t_consensus:
    st.markdown("## 🧭 MTF Consensus — 15m / 1h / 4h Alignment")
    symbol = st.text_input("Symbol (CCXT format)", value=st.session_state.sym).strip().upper()
    tfs = st.multiselect("Timeframes", ["15m", "1h", "4h", "1d"], default=["15m", "1h", "4h"])
    lim = st.slider("Candles per timeframe", 150, 800, 260, step=10)

    if st.button("🧭 COMPUTE CONSENSUS", type="primary", use_container_width=True):
        try:
            dfp, align = consensus_alignment(st.session_state.exch, symbol, tfs, lim, st.session_state)
            st.session_state._consensus_df = dfp
            st.session_state._consensus_align = align
        except Exception as e:
            st.error(str(e))

    if "_consensus_df" in st.session_state:
        dfp = st.session_state._consensus_df.copy()
        align = float(st.session_state._consensus_align)
        dir_label = "BULLISH" if align > 15 else ("BEARISH" if align < -15 else "NEUTRAL")
        st.metric("Alignment Score", f"{align:.1f}", delta=dir_label)
        st.dataframe(dfp[["tf","ts","close","flux","vec_state","brain_trend","gate_safe","mat_sig","smc_buy","smc_sell","dir"]],
                     use_container_width=True, hide_index=True)


# ============================================================
# 13) SIGNAL REPLAY TAB (Radar Timeline)
# ============================================================
with t_replay:
    st.markdown("## 🎞 Signal Replay — Market Radar Timeline")

    if not st.session_state.signal_history:
        st.info("No events yet. Run Scanner with Auto-record enabled, or wait for core alerts.")
        st.stop()

    # Filters
    f1, f2, f3 = st.columns([2,1,1])
    with f1:
        st.session_state.replay_search = st.text_input("Search (symbol/signal)", value=st.session_state.replay_search)
    with f2:
        st.session_state.replay_filter_radar = st.toggle("Radar only", value=bool(st.session_state.replay_filter_radar))
    with f3:
        st.session_state.replay_filter_core = st.toggle("Core only", value=bool(st.session_state.replay_filter_core))

    evs = st.session_state.signal_history[:]
    q = st.session_state.replay_search.strip().lower()
    if q:
        evs = [e for e in evs if (q in str(e.get("symbol","")).lower()) or (q in str(e.get("signal","")).lower())]

    if st.session_state.replay_filter_radar and not st.session_state.replay_filter_core:
        evs = [e for e in evs if str(e.get("signal","")).startswith("radar_spike")]
    if st.session_state.replay_filter_core and not st.session_state.replay_filter_radar:
        evs = [e for e in evs if not str(e.get("signal","")).startswith("radar_spike")]

    # Sort radar spikes first by |z|
    evs_sorted = sorted(
        evs,
        key=lambda e: (0 if str(e.get("signal","")).startswith("radar_spike") else 1, -(abs(float(e.get("flux_z", 0.0))) if pd.notna(e.get("flux_z", 0.0)) else 0.0))
    )

    if not evs_sorted:
        st.info("No events match your filters.")
        st.stop()

    labels = [
        f"{e.get('ts','')} | {e.get('symbol','')} | {e.get('tf','')} | {e.get('signal','')} | z={float(e.get('flux_z',0.0)):.2f}"
        for e in evs_sorted
    ]
    idx = st.selectbox("Pick an event", list(range(len(labels))), format_func=lambda i: labels[i])
    ev = evs_sorted[idx]

    st.markdown(f"### Selected: `{ev.get('symbol')}` · `{ev.get('tf')}` · `{ev.get('signal')}` · `{ev.get('ts')}`")

    chips = []
    if str(ev.get("signal","")).startswith("radar_spike"):
        chips.append(f"Radar z={float(ev.get('flux_z',0.0)):.2f}")
    chips.append(f"Flux={float(ev.get('flux',0.0)):.3f}")
    chips.append(f"Brain={int(ev.get('brain_trend',0))}")
    chips.append("SAFE" if bool(ev.get("gate_safe", True)) else "CHAOS")
    chips.append(f"Matrix={int(ev.get('mat_sig',0))}")
    if bool(ev.get("smc_buy", False)): chips.append("SMC BUY")
    if bool(ev.get("smc_sell", False)): chips.append("SMC SELL")
    st.markdown(" ".join([f'<span class="chip">{c}</span>' for c in chips]), unsafe_allow_html=True)

    lim = st.slider("Replay candles", 200, 1200, 500, step=50)

    if st.button("▶️ REPLAY CONTEXT", type="primary", use_container_width=True):
        try:
            df, sym2, _ = latest_core_state(ev["exch"], ev["symbol"], ev["tf"], lim, st.session_state)
            plot_replay_context(df, ev["ts"], title=f"Replay — {ev['symbol']} {ev['tf']} {ev['signal']} @ {ev['ts']}")

            why = []
            why.append(f"Flux={float(ev.get('flux',0.0)):.3f}")
            if str(ev.get("signal","")).startswith("radar_spike"):
                why.append(f"SpikeZ={float(ev.get('flux_z',0.0)):.2f}")
            why.append(f"BrainTrend={int(ev.get('brain_trend',0))}")
            why.append(f"GateSafe={bool(ev.get('gate_safe',True))}")
            why.append(f"MatrixSig={int(ev.get('mat_sig',0))}")
            if bool(ev.get("smc_buy", False)): why.append("SMC=BUY")
            if bool(ev.get("smc_sell", False)): why.append("SMC=SELL")

            st.markdown("### Why it fired (deterministic)")
            st.write(" · ".join(why))

            st.markdown("### AI “Why” (optional)")
            if OPENAI_AVAILABLE and st.session_state.ai_k:
                if st.button("🤖 Generate AI Narrative", use_container_width=True):
                    c = OpenAI(api_key=st.session_state.ai_k)
                    prompt = f"""
Explain why this signal fired and what it implies.

Signal:
- exch={ev['exch']}
- symbol={ev['symbol']}
- tf={ev['tf']}
- ts={ev['ts']}
- signal={ev['signal']}
- entry_price={float(ev.get('price', 0.0)):.6f}

State:
- flux={float(ev.get('flux', 0.0)):.6f}
- flux_z={float(ev.get('flux_z', 0.0)):.4f}
- brain_trend={int(ev.get('brain_trend', 0))}
- gate_safe={bool(ev.get('gate_safe', True))}
- mat_sig={int(ev.get('mat_sig', 0))}
- smc_buy={bool(ev.get('smc_buy', False))}
- smc_sell={bool(ev.get('smc_sell', False))}

Rules:
- Keep it concrete.
- Give 1 bull case + 1 bear case.
- Suggest 1 invalidation idea.
- If it’s a radar spike, explain what a spike means and what to watch next.
""".strip()
                    r = c.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"user","content":prompt}]
                    )
                    st.info(r.choices[0].message.content)
            else:
                st.caption("Add an OpenAI key in the sidebar to enable AI narrative.")
        except Exception as e:
            st.error(str(e))

    st.markdown("### System Log")
    st.markdown('<div class="log-container">', unsafe_allow_html=True)
    for row in st.session_state.signal_log[:90]:
        st.markdown(
            f'<div class="log-entry"><span class="log-ts">{row["ts"]}</span><span class="log-msg">{row["msg"]}</span></div>',
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 14) BACKTEST TAB
# ============================================================
with t_backtest:
    st.markdown("## 🧪 Backtest-lite — last N signals, hit-rate + MFE/MAE")
    symbol = st.text_input("Symbol", value=st.session_state.sym).strip().upper()
    tf = st.selectbox("Timeframe", ["1m","5m","15m","1h","4h","1d"], index=2)
    lim = st.slider("History candles", 300, 2500, 1200, step=50)

    sig = st.selectbox("Signal type", [
        ("br_buy_evt", "Brain Buy (edge)"),
        ("br_sell_evt", "Brain Sell (edge)"),
        ("smc_buy_evt", "SMC Buy (edge)"),
        ("smc_sell_evt", "SMC Sell (edge)"),
        ("vec_super_evt", "Vector Super (edge)"),
        ("vec_crash_evt", "Vector Crash (edge)"),
    ], format_func=lambda x: x[1])

    direction = st.selectbox("Direction", [("LONG", 1), ("SHORT", -1)], format_func=lambda x: x[0])[1]
    last_n = st.slider("Last N signals", 10, 200, 60, step=10)
    horizon = st.slider("Forward horizon (bars)", 5, 120, 30, step=5)

    tp = st.slider("TP threshold (%)", 0.1, 5.0, 0.8, step=0.1)
    sl = st.slider("SL threshold (%)", 0.1, 5.0, 0.5, step=0.1)

    if st.button("🧪 RUN BACKTEST-LITE", type="primary", use_container_width=True):
        try:
            df, sym2, _ = latest_core_state(st.session_state.exch, symbol, tf, lim, st.session_state)
            bt = backtest_lite(df, signal_col=sig[0], direction=direction, last_n=last_n, horizon=horizon, tp_pct=tp, sl_pct=sl)
            if bt.empty:
                st.warning("No signals found in the selected window.")
            else:
                hit_rate = (bt["hit"] == "TP").mean() * 100.0
                avg_ret = bt["ret_%"].mean()
                avg_mfe = bt["mfe_%"].mean()
                avg_mae = bt["mae_%"].mean()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Hit-rate (TP first)", f"{hit_rate:.1f}%")
                c2.metric("Avg Return (%, horizon)", f"{avg_ret:.2f}")
                c3.metric("Avg MFE (%)", f"{avg_mfe:.2f}")
                c4.metric("Avg MAE (%)", f"{avg_mae:.2f}")

                st.dataframe(bt, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(str(e))


# ============================================================
# 15) CORE TABS (Brain/Structure/Vector/Quantum)
# ============================================================
with t_brain:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[
            go.Candlestick(x=df_main['timestamp'], open=df_main['open'], high=df_main['high'], low=df_main['low'], close=df_main['close'])
        ])
        fig.add_trace(go.Scatter(x=df_main['timestamp'], y=df_main['br_u'], name="Upper", opacity=0.25))
        fig.add_trace(go.Scatter(x=df_main['timestamp'], y=df_main['br_l_band'], name="Lower", opacity=0.25, fill='tonexty'))
        fig.update_layout(clean_plot(height=560), xaxis_rangeslider_visible=False)
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

with t_struct:
    l, r = st.columns([3, 1])
    with l:
        fig = go.Figure(data=[
            go.Candlestick(x=df_main['timestamp'], open=df_main['open'], high=df_main['high'], low=df_main['low'], close=df_main['close'])
        ])
        fig.add_trace(go.Scatter(x=df_main['timestamp'], y=df_main['smc_base'], name="SMC Base", line=dict(width=1)))
        fig.update_layout(clean_plot(height=560), xaxis_rangeslider_visible=False)
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

with t_vector:
    l, r = st.columns([3, 1])
    with l:
        fig_v = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.02)
        fig_v.add_trace(
            go.Candlestick(x=df_main['timestamp'], open=df_main['open'], high=df_main['high'], low=df_main['low'], close=df_main['close']),
            row=1, col=1
        )
        bar_colors = []
        for s in df_main["vec_state"].astype(int).tolist():
            if s == 2:
                bar_colors.append("#00E676")
            elif s == -2:
                bar_colors.append("#FF1744")
            elif s == 0:
                bar_colors.append("#444")
            else:
                bar_colors.append("#222")
        fig_v.add_trace(go.Bar(x=df_main['timestamp'], y=df_main['flux'], marker_color=bar_colors, name="Flux"), row=2, col=1)
        fig_v.update_layout(clean_plot(height=600), xaxis_rangeslider_visible=False)
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

with t_quant:
    l, r = st.columns([3, 1])
    with l:
        fig_q = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.02)
        fig_q.add_trace(
            go.Candlestick(x=df_main['timestamp'], open=df_main['open'], high=df_main['high'], low=df_main['low'], close=df_main['close']),
            row=1, col=1
        )
        fig_q.add_trace(go.Scatter(x=df_main["timestamp"], y=df_main["rqzo"], name="RQZO"), row=2, col=1)
        fig_q.update_layout(clean_plot(height=600), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_q, use_container_width=True)

    with r:
        rq = float(last["rqzo"])
        rq_bias = "POSITIVE" if rq > 1 else ("NEGATIVE" if rq < -1 else "NEUTRAL")
        st.markdown(f"""<div class="diag-panel">
            <div class="diag-header">Quantum Diagnostics</div>
            <div class="diag-item"><span class="diag-label">RQZO:</span><span class="diag-val">{rq:.2f}</span></div>
            <div class="diag-item"><span class="diag-label">Bias:</span><span class="diag-val">{rq_bias}</span></div>
            <div style="margin-top:18px; color:#555; font-size:0.85rem;">
                RQZO is a synthetic oscillator derived from normalized regime changes.
            </div>
        </div>""", unsafe_allow_html=True)


# ============================================================
# 16) TV TAB
# ============================================================
with t_tv:
    st.markdown("### TradingView Confirmation")
    if st.session_state.tv_symbol_override.strip():
        tv_symbol = st.session_state.tv_symbol_override.strip()
    else:
        base_guess = resolved_sym.replace("/", "")
        tv_symbol = f"{st.session_state.exch.upper()}:{base_guess}"

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
    st.caption(f"TradingView symbol: {tv_symbol} (Override in sidebar if needed.)")


# ============================================================
# 17) AI COUNCIL TAB
# ============================================================
with t_ai:
    st.markdown("## 🤖 AI Council")
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

    if st.session_state.debug:
        st.caption(f"Debug: resolved={resolved_sym} candles={len(df_main)} last_ts={last['timestamp']}")
        st.caption(f"OpenAI={OPENAI_AVAILABLE} Tweepy={TWEEPY_AVAILABLE} Autorefresh={AUTOR_AVAILABLE}")
