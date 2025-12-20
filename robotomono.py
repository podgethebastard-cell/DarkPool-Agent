# app.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Optional deps (handled safely)
try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


# ============================================================
# DarkPool UI (DPC CSS)
# ============================================================
DPC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Roboto Mono', monospace !important;
    background-color: #0e1117 !important;
    color: #e0e0e0 !important;
}

.stApp {
    background: radial-gradient(1200px 600px at 20% 10%, rgba(0, 255, 160, 0.06), transparent 60%),
                radial-gradient(900px 500px at 80% 20%, rgba(0, 180, 255, 0.05), transparent 55%),
                #0e1117 !important;
}

h1, h2, h3, h4 {
    color: #e0e0e0 !important;
    text-shadow: 0 0 10px rgba(0, 255, 180, 0.12);
    letter-spacing: 0.5px;
}

[data-testid="stSidebar"] {
    background-color: rgba(14, 17, 23, 0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 14px 14px 10px 14px;
    box-shadow: 0 0 24px rgba(0, 255, 180, 0.05);
}

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(0,255,180,0.06);
    color: #aafbe6;
    font-size: 12px;
    margin-right: 8px;
    margin-bottom: 6px;
    text-shadow: 0 0 10px rgba(0,255,180,0.10);
}

.hr {
    height: 1px;
    width: 100%;
    background: rgba(255,255,255,0.07);
    margin: 10px 0 14px 0;
}

.small {
    opacity: 0.85;
    font-size: 12px;
}

.stButton button {
    border-radius: 14px !important;
    border: 1px solid rgba(0,255,180,0.25) !important;
    background: rgba(0,255,180,0.06) !important;
    color: #e0e0e0 !important;
    box-shadow: 0 0 18px rgba(0,255,180,0.10);
}
.stButton button:hover {
    border: 1px solid rgba(0,255,180,0.45) !important;
    background: rgba(0,255,180,0.10) !important;
}

a { color: #7ee7ff !important; }
</style>
"""


# ============================================================
# Colors
# ============================================================
COL_BULL_NEON = "#00f2ff"
COL_BEAR_NEON = "#ff0055"
COL_ACC_GOLD = "#ffe600"

APX_BULL = "#00E676"
APX_BEAR = "#FF1744"
NEUT = "#78909C"


# ============================================================
# Secrets / Ops helpers
# ============================================================
def _get_secret(*keys: str, default: Optional[str] = None) -> Optional[str]:
    """
    Looks for a secret by trying multiple key candidates across:
      - st.secrets (flat or nested)
      - environment variables
    Returns first match.
    """
    # 1) Streamlit secrets flat keys
    try:
        for k in keys:
            if k in st.secrets:
                v = st.secrets.get(k)
                if v is not None and str(v).strip():
                    return str(v).strip()
    except Exception:
        pass

    # 2) Streamlit secrets common nested groups
    nested_groups = ["openai", "OPENAI", "telegram", "TELEGRAM", "bot", "BOT", "secrets", "SECRETS"]
    try:
        for group in nested_groups:
            if group in st.secrets and isinstance(st.secrets[group], dict):
                g = st.secrets[group]
                for k in keys:
                    if k in g and str(g[k]).strip():
                        return str(g[k]).strip()
    except Exception:
        pass

    # 3) Env vars
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip():
            return str(v).strip()

    return default


def _mask(s: Optional[str]) -> str:
    if not s:
        return "—"
    s = str(s)
    if len(s) <= 8:
        return "*" * len(s)
    return s[:3] + "*" * (len(s) - 6) + s[-3:]


def telegram_send_message(token: str, chat_id: str, text: str, parse_mode: str = "Markdown") -> Tuple[bool, str]:
    """
    Sends a message via Telegram Bot API.
    """
    if not token or not chat_id:
        return False, "Missing token/chat_id"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}

    try:
        if requests is not None:
            r = requests.post(url, json=payload, timeout=10)
            ok = (r.status_code == 200)
            if not ok:
                return False, f"HTTP {r.status_code}: {r.text[:200]}"
            return True, "sent"
        else:
            # fallback minimal urllib
            import urllib.request
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
            return True, body[:120]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def openai_analyze(openai_api_key: str, model: str, system: str, user: str) -> Tuple[bool, str]:
    """
    Uses OpenAI Responses API via official SDK (if installed).
    """
    if OpenAI is None:
        return False, "OpenAI SDK not installed. Add: pip install openai"
    try:
        client = OpenAI(api_key=openai_api_key)
        # Responses API (recommended by OpenAI docs)
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        # SDK returns a structured object; simplest: output_text
        text = getattr(resp, "output_text", None)
        if text:
            return True, str(text).strip()
        # fallback: try to serialize
        return True, json.dumps(resp.model_dump(), indent=2)[:4000]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ============================================================
# Core utilities
# ============================================================
def _annualize_factor(interval: str) -> float:
    if interval.endswith("d"):
        return 252.0
    if interval.endswith("wk"):
        return 52.0
    if interval.endswith("mo"):
        return 12.0
    if interval.endswith("h"):
        hours = float(interval.replace("h", ""))
        return 252.0 * (6.5 / max(hours, 1e-9))
    if interval.endswith("m"):
        mins = float(interval.replace("m", ""))
        return 252.0 * (6.5 * 60.0 / max(mins, 1e-9))
    return 252.0


def _to_utc_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_convert("UTC").tz_localize(None)
        except Exception:
            pass
    return df


def _flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        return df
    df.columns = [str(c).strip().title() for c in df.columns]
    return df


def base_layout(title: str, height: int) -> Dict[str, Any]:
    return dict(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        title=dict(text=title, x=0.01),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
    )


def ensure_ohlcv(px: pd.DataFrame) -> pd.DataFrame:
    need = {"Open", "High", "Low", "Close", "Volume"}
    missing = need.difference(set(px.columns))
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}")
    px = px.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px = px.dropna(subset=["Open", "High", "Low", "Close"])
    px["Volume"] = px["Volume"].fillna(0.0)
    return px


def resample_ohlcv(px: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = px["Open"].resample(rule).first()
    h = px["High"].resample(rule).max()
    l = px["Low"].resample(rule).min()
    c = px["Close"].resample(rule).last()
    v = px["Volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def heikin_ashi(px: pd.DataFrame) -> pd.DataFrame:
    df = px.copy()
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["Low"], ha_open, ha_close], axis=1).min(axis=1)
    out = df.copy()
    out["Open"], out["High"], out["Low"], out["Close"] = ha_open, ha_high, ha_low, ha_close
    return out


# ============================================================
# TA primitives
# ============================================================
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0.0)
    return (sign * volume).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    direction = np.sign(tp.diff()).fillna(0.0)
    pos = rmf.where(direction > 0, 0.0)
    neg = rmf.where(direction < 0, 0.0).abs()
    pos_sum = pos.rolling(length, min_periods=length).sum()
    neg_sum = neg.rolling(length, min_periods=length).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + ratio))).clip(0, 100)


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    w = np.arange(1, length + 1, dtype=float)
    denom = float(w.sum())

    def _wma(x: np.ndarray) -> float:
        return float(np.dot(x, w) / denom)

    return series.rolling(length, min_periods=length).apply(_wma, raw=True)


def hma(series: pd.Series, length: int) -> pd.Series:
    n = int(length)
    n2 = max(int(n / 2), 1)
    ns = max(int(math.sqrt(n)), 1)
    return wma(2.0 * wma(series, n2) - wma(series, n), ns)


def get_ma(ma_type: str, series: pd.Series, length: int) -> pd.Series:
    t = (ma_type or "EMA").upper()
    if t == "SMA":
        return series.rolling(length, min_periods=length).mean()
    if t == "EMA":
        return series.ewm(span=length, adjust=False, min_periods=length).mean()
    if t == "RMA":
        return rma(series, length)
    if t == "HMA":
        return hma(series, length)
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).clip(0, 100)


def dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)

    tr = true_range(high, low, close)
    atr = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean() / atr.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean() / atr.replace(0, np.nan))

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return plus_di, minus_di, adx


# ============================================================
# Engine contracts
# ============================================================
@dataclass
class EngineContext:
    symbol: str
    interval: str
    ann_factor: float
    px: pd.DataFrame
    params: Dict[str, Any] = field(default_factory=dict)
    shared: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FigureBundle:
    title: str
    fig: go.Figure


@dataclass
class ModuleOutput:
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    figures: List[FigureBundle] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)  # human-readable alert lines


class IndicatorModule(Protocol):
    name: str
    def compute(self, ctx: EngineContext) -> ModuleOutput: ...


class IndicatorEngine:
    def __init__(self) -> None:
        self._modules: List[IndicatorModule] = []

    def register(self, module: IndicatorModule) -> None:
        self._modules.append(module)

    def run(self, ctx: EngineContext, enabled: Dict[str, bool]) -> Dict[str, ModuleOutput]:
        outputs: Dict[str, ModuleOutput] = {}
        for m in self._modules:
            if enabled.get(m.name, True):
                outputs[m.name] = m.compute(ctx)
        return outputs


# ============================================================
# DataHub
# ============================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbols: Tuple[str, ...], start: date, end: date, interval: str) -> pd.DataFrame:
    _end = end + timedelta(days=1) if interval in ("1d", "1wk", "1mo") else end
    df = yf.download(
        tickers=list(symbols),
        start=start,
        end=_end,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError("No data returned. Check tickers / interval / date range.")
    df = _to_utc_naive_index(df)
    df = _flatten_ohlcv(df)
    return df


def extract_single_ticker(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        tops = list(map(str, df.columns.get_level_values(0).unique()))
        lut = {t.upper(): t for t in tops}
        if symbol.upper() not in lut:
            raise KeyError(f"Ticker '{symbol}' not found in returned dataset: {tops}")
        real = lut[symbol.upper()]
        out = df[real].copy()
        out.columns = [str(c).strip().title() for c in out.columns]
        return out.dropna(how="all")
    out = df.copy()
    out.columns = [str(c).strip().title() for c in out.columns]
    return out.dropna(how="all")


# ============================================================
# Module: Price Master
# ============================================================
class PriceMasterModule:
    name = "PriceMaster"

    def compute(self, ctx: EngineContext) -> ModuleOutput:
        px = ctx.px
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=px.index,
            open=px["Open"], high=px["High"], low=px["Low"], close=px["Close"],
            name=f"{ctx.symbol} OHLC",
            increasing_line_color="#00FF9A",
            decreasing_line_color="#FF2E63",
            increasing_fillcolor="rgba(0,255,154,0.15)",
            decreasing_fillcolor="rgba(255,46,99,0.15)",
            line_width=1,
        ))
        fig.update_layout(**base_layout(f"{ctx.symbol} — Price Master", 520))
        fig.update_xaxes(rangeslider_visible=False)

        last = float(px["Close"].dropna().iloc[-1]) if px["Close"].notna().any() else float("nan")
        logret = np.log(px["Close"]).diff()
        ann_vol = float(logret.dropna().std(ddof=1) * math.sqrt(ctx.ann_factor)) if logret.dropna().shape[0] > 10 else float("nan")

        return ModuleOutput(
            name=self.name,
            figures=[FigureBundle("Price Master", fig)],
            metrics={"Last": last, "AnnVol": ann_vol},
        )


# ============================================================
# Module: DPLR Mk II
# ============================================================
class DPLRModule:
    name = "DPLR"

    def compute(self, ctx: EngineContext) -> ModuleOutput:
        p = ctx.params["dplr"]
        px = ctx.px

        len_reactor = int(p["len_reactor"])
        len_mfi = int(p["len_mfi"])
        vol_mult = float(p["vol_mult"])
        show_wyckoff = bool(p["show_wyckoff"])

        out = px.copy()
        vol_avg = out["Volume"].rolling(50, min_periods=50).mean()
        vol_std = out["Volume"].rolling(50, min_periods=50).std(ddof=1)

        out["obv"] = obv(out["Close"], out["Volume"])
        out["mfi"] = mfi(out["High"], out["Low"], out["Close"], out["Volume"], len_mfi)

        out["slope_price"] = (out["Close"] - out["Close"].shift(len_reactor)) / float(len_reactor)
        out["slope_obv"] = (out["obv"] - out["obv"].shift(len_reactor)) / float(len_reactor)

        is_block = out["Volume"] > (vol_avg + (vol_std * vol_mult))
        is_accum = (out["slope_price"] < 0.0) & (out["slope_obv"] > 0.0)
        is_dist = (out["slope_price"] > 0.0) & (out["slope_obv"] < 0.0)

        lowest_low = out["Low"].rolling(20, min_periods=20).min()
        highest_high = out["High"].rolling(20, min_periods=20).max()
        mid = (out["High"] + out["Low"]) / 2.0

        is_spring = show_wyckoff & (out["Low"] <= lowest_low.shift(1)) & (out["Close"] > mid) & (out["Volume"] > vol_avg * 1.1)
        is_upthrust = show_wyckoff & (out["High"] >= highest_high.shift(1)) & (out["Close"] < mid) & (out["Volume"] > vol_avg * 1.1)

        up_pressure = (out["High"] - out["Open"]) + (out["Close"] - out["Low"])
        down_pressure = (out["Open"] - out["Low"]) + (out["High"] - out["Close"])
        buy_vol_ratio = up_pressure / (up_pressure + down_pressure + 1e-5)
        delta_vol = (out["Volume"] * (buy_vol_ratio - 0.5) * 2.0).fillna(0.0)

        spring = is_spring.fillna(False)
        upth = is_upthrust.fillna(False)
        block = (is_block & (~spring) & (~upth)).fillna(False)

        # Alerts: last bar only
        alerts: List[str] = []
        if len(out) > 0:
