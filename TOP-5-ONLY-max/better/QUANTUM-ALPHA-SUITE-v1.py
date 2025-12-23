# ============================================================
# QUANTUM ALPHA SUITE — HZQEO (Upgraded UX + Plotly Pro + AI + TG)
# ============================================================
# ✅ Expert Plotly: unified hover, regime bands, signal markers, divergence markers
# ✅ AI Analysis: GPT-5.2 via OpenAI API (auto from st.secrets / env)
# ✅ Telegram: token + chat id integration (auto from st.secrets / env)
# ✅ Intent Echo UX: app adapts to user interaction rhythm
# ✅ Mission Lock: prevents accidental runs / ensures reproducible snapshots
# ✅ Data Health: staleness + flat-feed detection for trust
#
# Secrets expected (Streamlit):
#   [secrets]
#   OPENAI_API_KEY="..."
#   TELEGRAM_TOKEN="..."
#   TELEGRAM_CHAT_ID="..."
# Optional:
#   OPENAI_MODEL="gpt-5.2"  # default
# ============================================================

import os
import time
import math
import warnings
from typing import Optional, Tuple, List, Dict

import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# --------------------------
# Optional SciPy (nice-to-have)
# --------------------------
try:
    from scipy.special import zeta as scipy_zeta
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# --------------------------
# Optional OpenAI client (AI tab)
# --------------------------
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False


# ============================================================
# 0) THEME / CSS (SELF-CONTAINED)
# ============================================================
def inject_quantum_css():
    st.markdown(
        """
<style>
:root{
  --quantum-primary:#00f5d4;
  --quantum-secondary:#9d4edd;
  --quantum-danger:#ff006e;
  --quantum-success:#00ff9d;
  --quantum-warning:#ffd60a;
  --quantum-dim:rgba(226,226,255,0.65);
  --card-bg:rgba(255,255,255,0.04);
  --card-border:rgba(255,255,255,0.08);
}

html, body, [class*="css"] { background:#000 !important; color:#e2e2ff !important; }
section.main > div { padding-top: 0.5rem; }

.quantum-card{
  background:var(--card-bg);
  border:1px solid var(--card-border);
  border-radius:14px;
  padding:1rem;
  box-shadow:0 10px 30px rgba(0,0,0,0.35);
}
.quantum-metric{
  background:var(--card-bg);
  border:1px solid var(--card-border);
  border-radius:14px;
  padding:0.9rem 1rem;
}
.quantum-metric .label{
  font-size:0.75rem;
  letter-spacing:0.08em;
  color:var(--quantum-dim);
}
.quantum-metric .value{
  margin-top:0.35rem;
  font-size:1.35rem;
  font-weight:800;
}
.quantum-header{ padding: 0.3rem 0.2rem; }

.small-dim{ color:var(--quantum-dim); font-size:0.85rem; }
.badge{
  display:inline-block; padding:0.18rem 0.55rem; border-radius:999px;
  border:1px solid rgba(255,255,255,0.14); font-size:0.75rem; color:rgba(255,255,255,0.85);
  margin-left:0.4rem;
}
hr{ border:none; border-top:1px solid rgba(255,255,255,0.08); margin:0.75rem 0; }

/* Buttons */
.stButton>button{
  border-radius:12px !important;
  border:1px solid rgba(255,255,255,0.12) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UX UPGRADE 1: INTENT ECHO MODE (Cognitive Weather)
# ============================================================
class IntentEngine:
    def __init__(self, idle_hesitant=7.5, fast_decisive=1.15, streak_len=3):
        self.idle_hesitant = float(idle_hesitant)
        self.fast_decisive = float(fast_decisive)
        self.streak_len = int(streak_len)

    def tick(self) -> Tuple[str, float]:
        ss = st.session_state
        now = time.time()
        if "ux_last" not in ss:
            ss.ux_last = now
            ss.ux_streak = 0
            ss.ux_intent = "exploratory"
            return ss.ux_intent, 999.0

        dt = now - ss.ux_last
        ss.ux_last = now

        if dt < self.fast_decisive:
            ss.ux_streak += 1
        else:
            ss.ux_streak = max(0, ss.ux_streak - 1)

        if dt > self.idle_hesitant:
            ss.ux_intent = "hesitant"
        elif ss.ux_streak >= self.streak_len:
            ss.ux_intent = "decisive"
        else:
            ss.ux_intent = "exploratory"

        return ss.ux_intent, dt


def render_intent_banner(intent: str, dt: float):
    if intent == "hesitant":
        tag, accent, desc = "🌫️ CALM MODE", "var(--quantum-primary)", f"Idle {dt:.1f}s — surfacing guidance."
    elif intent == "decisive":
        tag, accent, desc = "⚡ MOMENTUM", "var(--quantum-secondary)", "Fast actions — compact UI + pro density."
    else:
        tag, accent, desc = "🧭 SCANNING", "var(--quantum-dim)", "Balanced flow — standard layout."

    st.markdown(
        f"""
<div class="quantum-card" style="border-left:4px solid {accent}; padding:0.75rem 0.9rem;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div style="font-weight:900; letter-spacing:0.06em;">{tag}</div>
    <div class="badge">Intent Echo</div>
  </div>
  <div style="margin-top:0.25rem;" class="small-dim">{desc}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 1) QUANTUM HZQEO ENGINE (Deterministic)
# ============================================================
class HZQEOEngine:
    """Hyperbolic Zeta-Quantum Entropy Oscillator Engine (deterministic)"""

    def __init__(
        self,
        lookback: int = 100,
        zeta_terms: int = 50,
        quantum_barrier: float = 0.618,
        hyperbolic_curvature: float = 0.5,
        entropy_period: int = 20,
        reynolds_critical: float = 2000.0,
    ):
        self.parameters = {
            "lookback": int(lookback),
            "zeta_terms": int(zeta_terms),
            "quantum_barrier": float(quantum_barrier),
            "hyperbolic_curvature": float(hyperbolic_curvature),
            "entropy_period": int(entropy_period),
            "reynolds_critical": float(reynolds_critical),
        }

    @staticmethod
    def walsh_hadamard(n: int, x: float) -> float:
        sign = 1.0
        x_int = int(float(x) * 256)
        for i in range(8):
            bit_n = (n >> i) & 1
            bit_x = (x_int >> i) & 1
            if (bit_n & bit_x) == 1:
                sign = -sign
        return sign

    @staticmethod
    def normalize_price(price: float, min_p: float, max_p: float) -> float:
        denom = (max_p - min_p)
        if denom == 0 or not np.isfinite(denom):
            return 0.5
        v = (price - min_p) / denom
        return float(np.clip(v, 0.0, 1.0))

    @staticmethod
    def calculate_atr(df_window: pd.DataFrame) -> float:
        if df_window is None or len(df_window) < 2:
            return 0.0
        high = df_window["high"].to_numpy(dtype=float)
        low = df_window["low"].to_numpy(dtype=float)
        close = df_window["close"].to_numpy(dtype=float)

        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])

        true_ranges = np.maximum(np.maximum(tr1, tr2), tr3)
        if true_ranges.size == 0:
            return 0.0
        v = float(np.nanmean(true_ranges))
        return 0.0 if (not np.isfinite(v)) else v

    def calculate_zeta_component(self, df: pd.DataFrame) -> pd.Series:
        if df.empty or len(df) < self.parameters["lookback"] + 25:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

        lb = self.parameters["lookback"]
        terms = int(min(self.parameters["zeta_terms"], 30))  # deterministic cap
        result = np.zeros(len(df), dtype=float)

        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)

        for idx in range(lb, len(df)):
            w0 = idx - lb
            w1 = idx
            min_p = float(np.nanmin(lows[w0:w1]))
            max_p = float(np.nanmax(highs[w0:w1]))
            norm_price = self.normalize_price(closes[idx], min_p, max_p)

            r0 = idx - 20
            r1 = idx
            if r0 < 1:
                s_real = 0.5
            else:
                prev = closes[r0:r1]
                curr = closes[r0 + 1 : r1 + 1]
                lr = np.log(np.maximum(curr, 1e-12) / np.maximum(prev, 1e-12))
                lr_std = float(np.nanstd(lr))
                s_real = float(np.clip(0.5 + 0.1 * (lr_std / 0.01), 0.5, 3.0))

            zeta_osc = 0.0
            for n in range(1, terms):
                wh = self.walsh_hadamard(n, norm_price)
                denom = math.pow(n + 1, s_real)
                zeta_osc += wh * math.cos(10.0 * norm_price * math.log(n + 1)) / denom

            result[idx] = math.tanh(zeta_osc * 2.0)

        return pd.Series(result, index=df.index, dtype=float)

    def calculate_tunneling_probability(self, df: pd.DataFrame) -> pd.Series:
        if df.empty or len(df) < 60:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

        tunnel_prob = np.zeros(len(df), dtype=float)
        lookback = 50

        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)

        for idx in range(lookback, len(df)):
            window_high = float(np.nanmax(highs[idx - lookback : idx]))
            window_low = float(np.nanmin(lows[idx - lookback : idx]))
            barrier_width = max(window_high - window_low, 0.001)
            current_energy = float(closes[idx])

            atr = self.calculate_atr(df.iloc[max(idx - 14, 0) : idx + 1])
            denom_close = max(current_energy, 1e-12)
            hbar_eff = max(0.1 * (atr / denom_close) * 100.0, 0.001)

            if current_energy >= window_high:
                tunnel_prob[idx] = 1.0
            else:
                k = math.sqrt(max(2.0 * (window_high - current_energy), 0.0))
                exponent = -2.0 * k * barrier_width / hbar_eff
                exponent = float(np.clip(exponent, -80.0, 0.0))
                tunnel_prob[idx] = math.exp(exponent)

        return pd.Series(tunnel_prob, index=df.index, dtype=float)

    def calculate_entropy_factor(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)

        entropy_vals = np.zeros(len(df), dtype=float)
        period = self.parameters["entropy_period"]
        bins = 10
        closes = df["close"].to_numpy(dtype=float)

        for idx in range(period, len(df)):
            window = closes[idx - period : idx]
            if window.size < 2:
                entropy_vals[idx] = 0.5
                continue

            if float(np.nanstd(window)) == 0.0:
                entropy_vals[idx] = 0.5
                continue

            hist, _ = np.histogram(window, bins=bins)
            hist = hist.astype(float)
            s = float(hist.sum())
            if s <= 0.0:
                entropy_vals[idx] = 0.5
                continue

            p = hist / s
            ent = 0.0
            for pi in p:
                if pi > 0:
                    ent -= float(pi) * math.log(float(pi))

            entropy_vals[idx] = 1.0 - (ent / math.log(bins))

        entropy_vals = np.clip(entropy_vals, 0.0, 1.0)
        return pd.Series(entropy_vals, index=df.index, dtype=float)

    def calculate_fluid_factor(self, df: pd.DataFrame) -> pd.Series:
        if df.empty or len(df) < 2:
            return pd.Series(np.ones(len(df)), index=df.index, dtype=float)

        fluid_vals = np.zeros(len(df), dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)

        for idx in range(1, len(df)):
            prev = max(closes[idx - 1], 1e-12)
            price_velocity = abs(closes[idx] - closes[idx - 1]) / prev

            if idx >= 10:
                w0 = idx - 10
                w1 = idx
                stdev_price = float(np.nanstd(closes[w0:w1]))
                high_low_range = float(np.nanmax(highs[w0:w1]) - np.nanmin(lows[w0:w1]))
                if high_low_range > 0:
                    ratio = max(stdev_price / high_low_range, 0.001)
                    fractal_dim = 1.0 + (math.log(ratio) / math.log(2.0))
                else:
                    fractal_dim = 1.0
            else:
                fractal_dim = 1.0

            reynolds_num = price_velocity * fractal_dim * self.parameters["hyperbolic_curvature"]
            fluid_vals[idx] = math.exp(-abs(reynolds_num) / max(self.parameters["reynolds_critical"], 1e-12))

        fluid_vals = np.clip(fluid_vals, 0.0, 1.0)
        return pd.Series(fluid_vals, index=df.index, dtype=float)

    def calculate_hzqeo(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        results = pd.DataFrame(index=df.index)
        results["zeta_osc"] = self.calculate_zeta_component(df)
        results["tunnel_prob"] = self.calculate_tunneling_probability(df)
        results["entropy_factor"] = self.calculate_entropy_factor(df)
        results["fluid_factor"] = self.calculate_fluid_factor(df)

        results["hzqeo_raw"] = (
            results["zeta_osc"]
            * results["tunnel_prob"]
            * results["entropy_factor"]
            * results["fluid_factor"]
        )

        results["hzqeo_normalized"] = np.tanh(results["hzqeo_raw"].to_numpy(dtype=float) * 2.0)

        results["signal"] = 0
        results.loc[results["hzqeo_normalized"] > 0.8, "signal"] = 1
        results.loc[results["hzqeo_normalized"] < -0.8, "signal"] = -1

        # divergence helpers
        results["price_high"] = (df["close"].rolling(20).max() == df["close"])
        results["osc_high"] = (results["hzqeo_normalized"].rolling(20).max() == results["hzqeo_normalized"])
        results["price_low"] = (df["close"].rolling(20).min() == df["close"])
        results["osc_low"] = (results["hzqeo_normalized"].rolling(20).min() == results["hzqeo_normalized"])

        return results


# ============================================================
# 2) DATA ENGINE (CRYPTO + YFINANCE)
# ============================================================
class EnhancedQuantumDataEngine:
    def __init__(self):
        self.crypto_exchange = ccxt.kraken()
        self.hzqeo_engine: Optional[HZQEOEngine] = None

    def set_engine(self, engine: HZQEOEngine) -> None:
        self.hzqeo_engine = engine

    @st.cache_data(ttl=30, show_spinner=False)
    def fetch_crypto_ohlcv(_self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        try:
            markets = _self.crypto_exchange.load_markets()
            if symbol not in markets:
                return pd.DataFrame()
            if timeframe not in _self.crypto_exchange.timeframes:
                return pd.DataFrame()

            ohlcv = _self.crypto_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float, errors="ignore")
            return df
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=120, show_spinner=False)
    def fetch_yf_ohlcv(_self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        try:
            data = yf.download(ticker, interval=interval, period=period, auto_adjust=False, progress=False)
            if data is None or data.empty:
                return pd.DataFrame()
            data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            data = data[["open", "high", "low", "close", "volume"]].copy()
            data.index = pd.to_datetime(data.index, utc=True)
            return data
        except Exception:
            return pd.DataFrame()

    def calculate_hzqeo_for_symbol(
        self,
        market_type: str,
        symbol: str,
        timeframe: str,
        yf_interval: str,
        yf_period: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        if self.hzqeo_engine is None:
            return pd.DataFrame(), pd.DataFrame(), "HZQEO engine not initialized."

        if market_type == "Crypto":
            df = self.fetch_crypto_ohlcv(symbol, timeframe, 500)
            if df.empty:
                return pd.DataFrame(), pd.DataFrame(), "Crypto fetch failed (symbol/timeframe unsupported or unavailable)."
            hz = self.hzqeo_engine.calculate_hzqeo(df)
            return df, hz, ""
        else:
            df = self.fetch_yf_ohlcv(symbol, yf_interval, yf_period)
            if df.empty:
                return pd.DataFrame(), pd.DataFrame(), "yfinance fetch failed (ticker/interval/period unavailable)."
            hz = self.hzqeo_engine.calculate_hzqeo(df)
            return df, hz, ""


# ============================================================
# 3) TRUST LAYER: Data Health + Causality Thread
# ============================================================
def expected_seconds(timeframe: str) -> int:
    return {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(timeframe, 3600)

def data_health(df: pd.DataFrame, timeframe: str) -> List[str]:
    issues: List[str] = []
    if df is None or df.empty:
        return ["No data returned"]

    last_ts = pd.to_datetime(df.index[-1]).to_pydatetime()
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    age = (datetime.now(timezone.utc) - last_ts).total_seconds()
    if age > expected_seconds(timeframe) * 3:
        issues.append(f"Stale feed: last candle ~{int(age//60)} min old")

    if "volume" in df.columns and (df["volume"].tail(30) == 0).all():
        issues.append("Volume flatlined (symbol/market mismatch?)")

    if df["close"].tail(60).nunique() <= 2:
        issues.append("Price barely moving (bad feed / market closed / low liquidity)")

    if np.isfinite(df["close"].tail(200)).all():
        ret = df["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        if float(ret.tail(200).std() or 0) == 0:
            issues.append("Returns variance ~0 (suspicious feed)")

    return issues

def causality_thread(df: pd.DataFrame, hz: pd.DataFrame) -> List[str]:
    if df.empty or hz.empty:
        return []

    latest = hz.iloc[-1]
    prev = hz.iloc[-2] if len(hz) > 1 else latest

    h = float(latest.get("hzqeo_normalized", 0))
    h_prev = float(prev.get("hzqeo_normalized", 0))
    z = float(latest.get("zeta_osc", 0))
    t = float(latest.get("tunnel_prob", 0))
    e = float(latest.get("entropy_factor", 0))
    f = float(latest.get("fluid_factor", 0))

    bullets: List[str] = []
    bullets.append(f"HZQEO {h:+.3f} (prev {h_prev:+.3f}) → {'EXTREME' if abs(h) > 0.8 else 'REGIME'}")
    bullets.append(f"Zeta {z:+.3f} → {'strong resonance' if abs(z) > 0.6 else 'mild resonance'}")
    bullets.append(f"Tunneling {t:.3f} → {'barrier weak / breakout likely' if t > 0.6 else 'barrier strong'}")
    bullets.append(f"Entropy {e:.3f} → {'ordered trend' if e < 0.4 else 'noisy / disorder'}")
    bullets.append(f"Fluid {f:.3f} → {'laminar (clean moves)' if f > 0.6 else 'turbulent (chop risk)'}")

    # divergence hints
    if bool(latest.get("price_high", False)) and not bool(latest.get("osc_high", False)) and h > 0:
        bullets.append("⚠️ Divergence: price high without osc high (bull weakening)")
    if bool(latest.get("price_low", False)) and not bool(latest.get("osc_low", False)) and h < 0:
        bullets.append("⚠️ Divergence: price low without osc low (bear weakening)")

    return bullets[:7]


# ============================================================
# 4) TELEGRAM INTEGRATION (Do not omit)
# ============================================================
def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return os.getenv(name, default)

def telegram_send(token: str, chat_id: str, text: str) -> Tuple[bool, str]:
    if not token or not chat_id:
        return False, "Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID"
    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=8)
        if r.status_code == 200:
            return True, "Sent"
        return False, f"Telegram error {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"Telegram exception: {e}"


# ============================================================
# 5) OPENAI GPT-5.2 AI ANALYSIS (Do not omit)
# ============================================================
def openai_client_from_secrets() -> Tuple[Optional["OpenAI"], str]:
    api_key = get_secret("OPENAI_API_KEY", "")
    if not api_key:
        return None, "Missing OPENAI_API_KEY"
    if not OPENAI_OK:
        return None, "OpenAI package not installed (pip install openai)"
    try:
        client = OpenAI(api_key=api_key)
        return client, ""
    except Exception as e:
        return None, f"OpenAI init error: {e}"

def ai_analyze_hzqeo(
    client: "OpenAI",
    model: str,
    symbol: str,
    market_type: str,
    timeframe: str,
    df: pd.DataFrame,
    hz: pd.DataFrame,
    user_question: str = "",
) -> str:
    # compact deterministic summary (keeps tokens reasonable)
    last_n = min(120, len(df))
    df_tail = df.tail(last_n).copy()
    hz_tail = hz.tail(last_n).copy()

    latest = hz_tail.iloc[-1].to_dict()
    last_price = float(df_tail["close"].iloc[-1])
    last_ts = str(df_tail.index[-1])

    # quick stats
    returns = df_tail["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    vol = float(returns.std() or 0)
    drift = float((df_tail["close"].iloc[-1] / df_tail["close"].iloc[0] - 1) if len(df_tail) > 1 else 0)

    bullets = "\n".join([f"- {b}" for b in causality_thread(df, hz)])

    prompt = f"""
You are an expert quantitative analyst and UX-friendly explainer.
You must be precise, avoid hallucinations, and clearly separate facts from interpretation.
No financial advice; provide educational risk-aware analysis.

Context:
- Market: {market_type}
- Symbol: {symbol}
- Timeframe: {timeframe}
- Last timestamp (UTC): {last_ts}
- Last close: {last_price:.6f}
- Volatility (std of returns, last {len(returns)} bars): {vol:.6f}
- Drift over last {last_n} bars: {drift*100:.2f}%

HZQEO latest:
- hzqeo_normalized: {float(latest.get("hzqeo_normalized", 0)):.6f}
- zeta_osc: {float(latest.get("zeta_osc", 0)):.6f}
- tunnel_prob: {float(latest.get("tunnel_prob", 0)):.6f}
- entropy_factor: {float(latest.get("entropy_factor", 0)):.6f}
- fluid_factor: {float(latest.get("fluid_factor", 0)):.6f}
- signal (1 overbought / -1 oversold / 0 neutral): {int(latest.get("signal", 0))}

Causality thread:
{bullets}

User question (optional):
{user_question.strip() or "(none)"}

Output format:
1) Snapshot (3 bullets)
2) Regime read (ordered vs noisy, laminar vs turbulent)
3) What would invalidate this read? (2-4 bullets)
4) If/Then playbook (educational, not advice) (3-6 bullets)
5) One-line caution about risk and data quality
""".strip()

    # Prefer Responses API if available; fall back to chat.completions for compatibility.
    try:
        if hasattr(client, "responses"):
            resp = client.responses.create(
                model=model,
                input=prompt,
            )
            # Responses API returns output_text
            return getattr(resp, "output_text", "").strip() or str(resp)
    except Exception:
        pass

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"AI request failed: {e}"


# ============================================================
# 6) VISUALS (Expert Plotly)
# ============================================================
def create_hzqeo_chart(df: pd.DataFrame, hz: pd.DataFrame, symbol: str):
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.42, 0.22, 0.2, 0.16],
        subplot_titles=(
            f"{symbol} Price Action",
            "HZQEO Oscillator (Regime Bands + Signals)",
            "Component Breakdown",
            "Entropy (Order/Disorder)",
        ),
    )

    # Price candlestick (clean, pro hover)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_width=1,
            decreasing_line_width=1,
        ),
        row=1,
        col=1,
    )

    # Oscillator
    fig.add_trace(
        go.Scatter(
            x=hz.index,
            y=hz["hzqeo_normalized"],
            name="HZQEO",
            mode="lines",
            line=dict(width=2),
            hovertemplate="HZQEO: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Regime bands (visual trust)
    # Oversold band
    fig.add_hrect(y0=-1.2, y1=-0.8, fillcolor="rgba(0,255,157,0.10)", line_width=0, row=2, col=1)
    # Neutral band
    fig.add_hrect(y0=-0.2, y1=0.2, fillcolor="rgba(226,226,255,0.05)", line_width=0, row=2, col=1)
    # Overbought band
    fig.add_hrect(y0=0.8, y1=1.2, fillcolor="rgba(255,0,110,0.10)", line_width=0, row=2, col=1)

    # Reference lines
    fig.add_hline(y=0.8, line_dash="dash", line_color="rgba(255,0,110,0.75)", row=2, col=1)
    fig.add_hline(y=-0.8, line_dash="dash", line_color="rgba(0,255,157,0.75)", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(180,180,220,0.35)", row=2, col=1)

    # Signal markers on oscillator
    if "signal" in hz.columns:
        sig_up = hz[hz["signal"] == -1]  # oversold -> potential up
        sig_dn = hz[hz["signal"] == 1]   # overbought -> potential down
        if not sig_up.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig_up.index,
                    y=sig_up["hzqeo_normalized"],
                    mode="markers",
                    name="Oversold Signal",
                    marker=dict(symbol="triangle-up", size=10, line=dict(width=1)),
                    hovertemplate="Oversold signal<br>%{x}<br>HZQEO: %{y:.4f}<extra></extra>",
                ),
                row=2, col=1
            )
        if not sig_dn.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig_dn.index,
                    y=sig_dn["hzqeo_normalized"],
                    mode="markers",
                    name="Overbought Signal",
                    marker=dict(symbol="triangle-down", size=10, line=dict(width=1)),
                    hovertemplate="Overbought signal<br>%{x}<br>HZQEO: %{y:.4f}<extra></extra>",
                ),
                row=2, col=1
            )

    # Divergence markers (subtle, but useful)
    div_bear = hz[(hz.get("price_high", False)) & (~hz.get("osc_high", False))]
    div_bull = hz[(hz.get("price_low", False)) & (~hz.get("osc_low", False))]

    if not div_bear.empty:
        fig.add_trace(
            go.Scatter(
                x=div_bear.index,
                y=div_bear["hzqeo_normalized"],
                mode="markers",
                name="Divergence Hint (Bear)",
                marker=dict(symbol="x", size=8),
                hovertemplate="Divergence hint (bear)<br>%{x}<extra></extra>",
            ),
            row=2, col=1
        )

    if not div_bull.empty:
        fig.add_trace(
            go.Scatter(
                x=div_bull.index,
                y=div_bull["hzqeo_normalized"],
                mode="markers",
                name="Divergence Hint (Bull)",
                marker=dict(symbol="circle-open", size=8),
                hovertemplate="Divergence hint (bull)<br>%{x}<extra></extra>",
            ),
            row=2, col=1
        )

    # Components
    comps = ["zeta_osc", "tunnel_prob", "entropy_factor", "fluid_factor"]
    for c in comps:
        if c in hz.columns:
            fig.add_trace(
                go.Scatter(
                    x=hz.index,
                    y=hz[c],
                    name=c.replace("_", " ").title(),
                    mode="lines",
                    line=dict(width=1.5),
                    hovertemplate=f"{c}: "+"%{y:.4f}<extra></extra>",
                ),
                row=3,
                col=1,
            )

    # Entropy area (legibility)
    if "entropy_factor" in hz.columns:
        fig.add_trace(
            go.Scatter(
                x=hz.index,
                y=hz["entropy_factor"],
                name="Entropy Factor",
                fill="tozeroy",
                mode="lines",
                line=dict(width=1.5),
                hovertemplate="Entropy: %{y:.4f}<extra></extra>",
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        height=1050,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e2ff",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="HZQEO", row=2, col=1, range=[-1.2, 1.2])
    fig.update_yaxes(title_text="Components", row=3, col=1)
    fig.update_yaxes(title_text="Entropy", row=4, col=1, range=[0, 1])

    # Better x-axis
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


def render_hzqeo_metrics(hz: pd.DataFrame):
    if hz.empty:
        return
    latest = hz.iloc[-1]
    metrics = [
        ("HZQEO", f"{float(latest['hzqeo_normalized']):+.4f}"),
        ("ZETA", f"{float(latest.get('zeta_osc', 0)):+.4f}"),
        ("TUNNEL", f"{float(latest.get('tunnel_prob', 0)):.4f}"),
        ("ENTROPY", f"{float(latest.get('entropy_factor', 0)):.4f}"),
        ("FLUID", f"{float(latest.get('fluid_factor', 0)):.4f}"),
    ]
    cols = st.columns(len(metrics))
    for i, (k, v) in enumerate(metrics):
        with cols[i]:
            st.markdown(
                f"""
<div class="quantum-metric">
  <div class="label">{k}</div>
  <div class="value">{v}</div>
</div>
                """,
                unsafe_allow_html=True,
            )


def render_hzqeo_signal_card(hz: pd.DataFrame):
    if hz.empty:
        return

    latest = hz.iloc[-1]
    prev = hz.iloc[-2] if len(hz) > 1 else latest

    h = float(latest["hzqeo_normalized"])
    h_prev = float(prev["hzqeo_normalized"])

    if h > 0.8:
        signal = "OVERBOUGHT 🔴"
        color = "var(--quantum-danger)"
        rec = "Extreme zone. Watch for mean reversion and divergence hints."
    elif h < -0.8:
        signal = "OVERSOLD 🟢"
        color = "var(--quantum-success)"
        rec = "Extreme zone. Watch for bounce conditions and confirmation."
    elif h > 0 and h > h_prev:
        signal = "BULLISH MOMENTUM ↗"
        color = "var(--quantum-primary)"
        rec = "Positive oscillator and rising vs prior — momentum strengthening."
    elif h < 0 and h < h_prev:
        signal = "BEARISH MOMENTUM ↘"
        color = "var(--quantum-warning)"
        rec = "Negative oscillator and falling vs prior — downside pressure increasing."
    else:
        signal = "NEUTRAL ⚪"
        color = "var(--quantum-dim)"
        rec = "Equilibrium band. Wait for clarity or use additional filters."

    st.markdown(
        f"""
<div class="quantum-card" style="border-left:4px solid {color};">
  <h3 style="margin:0; color:{color};">HZQEO SIGNAL: {signal}</h3>
  <p style="margin:0.6rem 0 0 0;">{rec}</p>
  <div style="display:flex; gap:1rem; margin-top:0.8rem; color:var(--quantum-dim);">
    <div>
      <small>Current: {h:+.4f}</small><br/>
      <small>Previous: {h_prev:+.4f}</small>
    </div>
    <div>
      <small>Zeta: {float(latest.get('zeta_osc', 0)):+.4f}</small><br/>
      <small>Tunnel: {float(latest.get('tunnel_prob', 0)):.4f}</small>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    col1, col2, col3 = st.columns([2, 3, 2])

    with col1:
        st.markdown(
            """
<div class="quantum-header">
  <h1 style="margin:0; font-size:2.2rem; font-weight:900;
    background:linear-gradient(135deg,var(--quantum-primary),var(--quantum-secondary));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    QUANTUM HZQEO
  </h1>
  <p style="margin:0.4rem 0 0 0; color:var(--quantum-dim); font-size:0.9rem;">
    Hyperbolic Zeta-Quantum Entropy Oscillator
  </p>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
<div style="text-align:center; padding:1rem;">
  <div style="font-size:0.85rem; color:var(--quantum-dim); margin-bottom:0.5rem;">OPERATOR TERMINAL</div>
  <div style="font-size:1.5rem; font-weight:800; color:var(--quantum-primary);">ZETA × ENTROPY × TUNNELING</div>
  <div style="font-size:0.85rem; color:var(--quantum-dim); margin-top:0.5rem;">
    Deterministic engine with AI explainability + Telegram dispatch
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        now = datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f"""
<div style="text-align:right; padding:1rem;">
  <div style="font-size:0.85rem; color:var(--quantum-dim);">QUANTUM TIME</div>
  <div style="font-size:1.8rem; font-weight:900; font-family:monospace; color:var(--quantum-primary);">{now}</div>
  <div style="font-size:0.85rem; color:var(--quantum-dim); margin-top:0.5rem;">Plotly Pro / GPT-5.2 / TG</div>
</div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 7) MAIN APP
# ============================================================
st.set_page_config(
    page_title="QUANTUM ALPHA SUITE — HZQEO (Pro)",
    layout="wide",
    page_icon="⚛️",
    initial_sidebar_state="expanded",
)

inject_quantum_css()
render_header()

intent, dt = IntentEngine().tick()
render_intent_banner(intent, dt)

st.caption("Educational tool only. Not financial advice. Validate data quality before acting.")

engine = EnhancedQuantumDataEngine()

# Secrets + config
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-5.2")
TELEGRAM_TOKEN = get_secret("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID", "")

# Persist config
if "hzqeo_config" not in st.session_state:
    st.session_state.hzqeo_config = {
        "lookback": 100,
        "zeta_terms": 50,
        "quantum_barrier": 0.618,
        "hyperbolic_curvature": 0.5,
        "entropy_period": 20,
        "reynolds_critical": 2000,
    }

# Persist last run results
if "last_run" not in st.session_state:
    st.session_state.last_run = {
        "df": pd.DataFrame(),
        "hz": pd.DataFrame(),
        "market_type": "",
        "symbol": "",
        "timeframe": "",
        "yf_interval": "",
        "yf_period": "",
        "run_fingerprint": "",
        "ran_at_utc": "",
    }

tab1, tab2, tab3, tab4 = st.tabs(
    ["⚛️ HZQEO OSCILLATOR", "📊 QUANTUM ANALYSIS", "⚙️ ADVANCED PARAMS", "🧠 AI + TELEGRAM"]
)

# --------------------------
# TAB 3 FIRST: parameters
# --------------------------
with tab3:
    st.markdown("### HZQEO PARAMETER CONFIGURATION")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### MATHEMATICAL PARAMETERS")
        st.session_state.hzqeo_config["lookback"] = st.slider("Lookback Period", 50, 500, st.session_state.hzqeo_config["lookback"])
        st.session_state.hzqeo_config["zeta_terms"] = st.slider("Zeta Terms", 10, 200, st.session_state.hzqeo_config["zeta_terms"])
        st.session_state.hzqeo_config["quantum_barrier"] = st.slider(
            "Quantum Barrier Level", 0.1, 0.9, float(st.session_state.hzqeo_config["quantum_barrier"]), 0.01
        )

    with c2:
        st.markdown("#### PHYSICS PARAMETERS")
        st.session_state.hzqeo_config["hyperbolic_curvature"] = st.slider(
            "Hyperbolic Curvature", 0.1, 2.0, float(st.session_state.hzqeo_config["hyperbolic_curvature"]), 0.1
        )
        st.session_state.hzqeo_config["entropy_period"] = st.slider("Entropy Period", 10, 100, st.session_state.hzqeo_config["entropy_period"])
        st.session_state.hzqeo_config["reynolds_critical"] = st.slider(
            "Critical Reynolds Number", 500, 5000, int(st.session_state.hzqeo_config["reynolds_critical"]), 100
        )

    st.markdown("---")
    pro_cols = st.columns([2, 1, 1])
    with pro_cols[0]:
        st.info("Tip: larger lookback = smoother but slower. Higher entropy_period = slower entropy response.")
    with pro_cols[1]:
        if st.button("💾 SAVE", use_container_width=True):
            st.success("Saved.")
    with pro_cols[2]:
        if st.button("♻️ CLEAR CACHE", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared. Run again.")

# apply engine params deterministically every render
hz_engine = HZQEOEngine(**st.session_state.hzqeo_config)
engine.set_engine(hz_engine)

# --------------------------
# TAB 1: main run + visuals
# --------------------------
with tab1:
    st.markdown("### HYPERBOLIC ZETA-QUANTUM ENTROPY OSCILLATOR")

    col1, col2, col3 = st.columns(3)
    with col1:
        market_type = st.selectbox("Market Type", ["Crypto", "Stocks", "Forex", "Commodities"], key="hzqeo_market_type")

    with col2:
        if market_type == "Crypto":
            symbol = st.selectbox(
                "Symbol (Kraken)",
                ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", "MATIC/USD"],
                index=0,
            )
        else:
            hint = "Examples: AAPL, MSFT, TSLA | Forex: EURUSD=X | Commodities: GC=F (Gold), CL=F (WTI)"
            symbol = st.text_input("Ticker (yfinance)", value="AAPL", help=hint)

    with col3:
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)

    # yfinance interval/period mapping
    yf_map = {
        "1m": ("1m", "7d"),
        "5m": ("5m", "60d"),
        "15m": ("15m", "60d"),
        "1h": ("60m", "730d"),
        "4h": ("60m", "730d"),
        "1d": ("1d", "10y"),
    }
    yf_interval, yf_period = yf_map[timeframe]

    if market_type != "Crypto" and timeframe == "1m":
        st.info("yfinance 1m is limited to short history; period fixed to 7d deterministically.")

    # ---- Mission Lock (prevents accidental runs) ----
    run_fingerprint = (
        f"{market_type}|{symbol}|{timeframe}|{yf_interval}|{yf_period}|"
        f"{st.session_state.hzqeo_config['lookback']}|{st.session_state.hzqeo_config['zeta_terms']}|"
        f"{st.session_state.hzqeo_config['entropy_period']}|{st.session_state.hzqeo_config['hyperbolic_curvature']}|"
        f"{st.session_state.hzqeo_config['reynolds_critical']}"
    )

    if "armed_calc" not in st.session_state:
        st.session_state.armed_calc = False
        st.session_state.calc_fingerprint = None

    st.markdown("#### 🧷 Mission Lock")
    a, b, c = st.columns([1, 1, 1])
    with a:
        st.session_state.armed_calc = st.checkbox("Arm calculate", value=st.session_state.armed_calc)
    with b:
        confirm = st.checkbox("Confirm snapshot", value=False)
    with c:
        if st.button("Disarm", use_container_width=True):
            st.session_state.armed_calc = False
            st.session_state.calc_fingerprint = None
            st.rerun()

    if st.session_state.armed_calc and confirm:
        st.session_state.calc_fingerprint = run_fingerprint

    ready = st.session_state.armed_calc and (st.session_state.calc_fingerprint == run_fingerprint)

    clicked = st.button("🚀 CALCULATE HZQEO", use_container_width=True, disabled=not ready)
    if not ready:
        st.caption("Arm → Confirm snapshot to run. Any config or symbol change auto-disarms.")

    if clicked:
        with st.spinner("Calculating HZQEO…"):
            df, hz, err = engine.calculate_hzqeo_for_symbol(
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
                yf_interval=yf_interval,
                yf_period=yf_period,
            )

        if err:
            st.error(err)
        elif df.empty or hz.empty:
            st.error("Failed to calculate HZQEO (empty data/results).")
        else:
            # persist last run
            st.session_state.last_run = {
                "df": df,
                "hz": hz,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "yf_interval": yf_interval,
                "yf_period": yf_period,
                "run_fingerprint": run_fingerprint,
                "ran_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }

            # Trust layer: health
            issues = data_health(df, timeframe)
            if issues:
                st.warning(" / ".join(issues))

            st.markdown("### QUANTUM METRICS")
            render_hzqeo_metrics(hz)

            st.markdown("### QUANTUM SIGNAL")
            render_hzqeo_signal_card(hz)

            st.markdown("### 🧵 CAUSALITY THREAD")
            for b in causality_thread(df, hz):
                st.markdown(f"- {b}")

            st.markdown("### ADVANCED VISUALIZATION")
            fig = create_hzqeo_chart(df, hz, symbol)
            st.plotly_chart(fig, use_container_width=True)

            raw_expanded = (st.session_state.get("ux_intent", "exploratory") == "hesitant")
            with st.expander("📈 View Raw HZQEO Data (last 50)", expanded=raw_expanded):
                display_df = hz.tail(50).copy()
                display_df.index = display_df.index.strftime("%Y-%m-%d %H:%M")
                st.dataframe(display_df, use_container_width=True)

# --------------------------
# TAB 2: component explanations + optional zeta viz
# --------------------------
with tab2:
    st.markdown("### QUANTUM COMPONENT ANALYSIS")

    components_exp = {
        "Zeta Oscillator": {
            "description": "Truncated Riemann-Zeta-like series with Walsh–Hadamard basis",
            "interpretation": "Harmonic resonance proxy in normalized price movements",
            "color": "#9d4edd",
        },
        "Quantum Tunneling": {
            "description": "Barrier escape probability from rolling high/low band",
            "interpretation": "Likelihood of overcoming recent resistance ceiling",
            "color": "#ff006e",
        },
        "Entropy Factor": {
            "description": "Shannon entropy of rolling close-price histogram",
            "interpretation": "Order/disorder proxy (scaled to [0,1])",
            "color": "#ffd60a",
        },
        "Fluid Factor": {
            "description": "Reynolds analogy from velocity × fractal proxy",
            "interpretation": "Laminar vs turbulent regime proxy (scaled to [0,1])",
            "color": "#00ff9d",
        },
    }

    cols = st.columns(2)
    for i, (name, data) in enumerate(components_exp.items()):
        with cols[i % 2]:
            st.markdown(
                f"""
<div class="quantum-card" style="border-left:4px solid {data['color']};">
  <h4 style="margin:0; color:{data['color']};">{name}</h4>
  <p style="margin:0.6rem 0 0 0; color:var(--quantum-dim); font-size:0.9rem;">{data['description']}</p>
  <div style="margin-top:0.7rem; padding:0.6rem; border-radius:10px; background:rgba(255,255,255,0.04);">
    <small><b>Interpretation:</b> {data['interpretation']}</small>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### MATHEMATICAL VISUALIZATION")
    if not SCIPY_OK:
        st.warning("SciPy not available; zeta visualization disabled.")
    else:
        x_vals = np.linspace(0.1, 5, 220)
        zeta_real = np.array([scipy_zeta(x) for x in x_vals], dtype=float)

        fig_math = go.Figure()
        fig_math.add_trace(go.Scatter(x=x_vals, y=zeta_real, mode="lines", name="ζ(s)"))
        fig_math.update_layout(
            title="Riemann Zeta Function ζ(s) (Real Axis)",
            height=420,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e2ff",
            xaxis_title="s",
            yaxis_title="ζ(s)",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(fig_math, use_container_width=True)

# --------------------------
# TAB 4: AI (GPT-5.2) + Telegram dispatch
# --------------------------
with tab4:
    st.markdown("### 🧠 AI + TELEGRAM CONTROL CENTER")

    # Show secrets status (without leaking)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f"""
<div class="quantum-card">
  <b>OpenAI</b><br/>
  <span class="small-dim">Key:</span> {"✅ loaded" if bool(get_secret("OPENAI_API_KEY","")) else "❌ missing"}<br/>
  <span class="small-dim">Model:</span> <span class="badge">{OPENAI_MODEL}</span>
</div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""
<div class="quantum-card">
  <b>Telegram</b><br/>
  <span class="small-dim">Token:</span> {"✅ loaded" if bool(TELEGRAM_TOKEN) else "❌ missing"}<br/>
  <span class="small-dim">Chat ID:</span> {"✅ loaded" if bool(TELEGRAM_CHAT_ID) else "❌ missing"}
</div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"""
<div class="quantum-card">
  <b>Last Run</b><br/>
  <span class="small-dim">Ran:</span> {st.session_state.last_run.get("ran_at_utc","(none)")}<br/>
  <span class="small-dim">Symbol:</span> {st.session_state.last_run.get("symbol","(none)")}
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    df = st.session_state.last_run.get("df", pd.DataFrame())
    hz = st.session_state.last_run.get("hz", pd.DataFrame())
    symbol = st.session_state.last_run.get("symbol", "")
    market_type = st.session_state.last_run.get("market_type", "")
    timeframe = st.session_state.last_run.get("timeframe", "")

    if df.empty or hz.empty:
        st.warning("Run a calculation in Tab 1 to enable AI analysis + Telegram dispatch.")
    else:
        # AI analysis
        st.markdown("#### AI Interpretation (GPT-5.2)")
        user_q = st.text_area(
            "Optional question for the AI",
            value="Explain the current regime and what would invalidate it. Mention risks and data quality.",
            height=90,
        )

        ai_col1, ai_col2 = st.columns([2, 1])
        with ai_col1:
            if st.button("🧠 GENERATE AI ANALYSIS", use_container_width=True):
                client, err = openai_client_from_secrets()
                if err:
                    st.error(err)
                else:
                    with st.spinner(f"Querying {OPENAI_MODEL}…"):
                        analysis = ai_analyze_hzqeo(
                            client=client,
                            model=OPENAI_MODEL,
                            symbol=symbol,
                            market_type=market_type,
                            timeframe=timeframe,
                            df=df,
                            hz=hz,
                            user_question=user_q,
                        )
                    st.session_state["latest_ai_analysis"] = analysis

        with ai_col2:
            st.markdown("**Telegram Dispatch**")
            st.caption("Sends the latest AI + snapshot.")

        analysis_text = st.session_state.get("latest_ai_analysis", "")
        if analysis_text:
            st.markdown("##### AI Output")
            st.markdown(
                f"""
<div class="quantum-card">
<pre style="white-space:pre-wrap; margin:0; color:#e2e2ff; background:rgba(255,255,255,0.03); padding:0.9rem; border-radius:12px;">
{analysis_text}
</pre>
</div>
                """,
                unsafe_allow_html=True,
            )

        # Telegram: send snapshot + AI
        st.markdown("#### 📡 Telegram Actions")
        latest = hz.iloc[-1]
        health = data_health(df, timeframe)
        thread = causality_thread(df, hz)

        snapshot_md = (
            f"*HZQEO SNAPSHOT*\n"
            f"- Market: `{market_type}`\n"
            f"- Symbol: `{symbol}`\n"
            f"- TF: `{timeframe}`\n"
            f"- Time (UTC): `{df.index[-1]}`\n"
            f"- Close: `{float(df['close'].iloc[-1]):.6f}`\n"
            f"- HZQEO: `{float(latest['hzqeo_normalized']):+.4f}`\n"
            f"- Zeta: `{float(latest.get('zeta_osc',0)):+.4f}` | Tunnel: `{float(latest.get('tunnel_prob',0)):.4f}`\n"
            f"- Entropy: `{float(latest.get('entropy_factor',0)):.4f}` | Fluid: `{float(latest.get('fluid_factor',0)):.4f}`\n"
        )

        if thread:
            snapshot_md += "\n*Causality*\n" + "\n".join([f"- {x}" for x in thread])

        if health:
            snapshot_md += "\n\n*Data Health*\n" + "\n".join([f"- ⚠️ {x}" for x in health])

        # Mission lock for sending (prevents fat-finger)
        send_fp = st.session_state.last_run.get("run_fingerprint", "") + f"|{float(latest['hzqeo_normalized']):.4f}"
        if "armed_send" not in st.session_state:
            st.session_state.armed_send = False
            st.session_state.send_fingerprint = None

        sc1, sc2, sc3 = st.columns([1, 1, 1])
        with sc1:
            st.session_state.armed_send = st.checkbox("Arm send", value=st.session_state.armed_send)
        with sc2:
            send_confirm = st.checkbox("Confirm snapshot", value=False, help="Locks to last-run + latest oscillator value")
        with sc3:
            if st.button("Disarm send", use_container_width=True):
                st.session_state.armed_send = False
                st.session_state.send_fingerprint = None
                st.rerun()

        if st.session_state.armed_send and send_confirm:
            st.session_state.send_fingerprint = send_fp

        send_ready = st.session_state.armed_send and (st.session_state.send_fingerprint == send_fp)

        tgc1, tgc2 = st.columns(2)
        with tgc1:
            if st.button("📨 SEND SNAPSHOT", use_container_width=True, disabled=not send_ready):
                ok, msg = telegram_send(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, snapshot_md)
                st.success("Sent ✅" if ok else "Failed ❌")
                if not ok:
                    st.error(msg)

        with tgc2:
            if st.button("🧠 SEND AI + SNAPSHOT", use_container_width=True, disabled=not (send_ready and bool(analysis_text))):
                payload = snapshot_md + "\n\n*AI Interpretation*\n" + analysis_text
                ok, msg = telegram_send(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, payload)
                st.success("Sent ✅" if ok else "Failed ❌")
                if not ok:
                    st.error(msg)

        if not send_ready:
            st.caption("To send: Arm send → Confirm snapshot. Any new run disarms automatically.")


# ============================================================
# End
# ============================================================
