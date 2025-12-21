# app.py
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# =============================================================================
# Streamlit Foundry — Neon Market Analyst (single-file app)
# Educational simulations only. Not investment advice.
# =============================================================================

# -----------------------------
# Product Manager (compact notes)
# -----------------------------
# Top trading questions + KPIs:
# 1) Which names have the best risk-adjusted trend/mean-reversion characteristics? (Sharpe, Sortino, MaxDD, Omega)
# 2) What signals would have worked historically, and what are the trade stats? (Win rate, Profit factor, Avg trade, Exposure)
# 3) How does a basket allocation compare under different weighting/rebalancing schemes? (CAGR, Vol, Sharpe, MaxDD, Ulcer)


# =============================================================================
# PAGE CONFIG + NEON CSS
# =============================================================================
st.set_page_config(page_title="Neon Market Analyst", layout="wide")

NEON_CSS = """
<style>
:root{
  --bg:#070A12;
  --panel:#0B1020;
  --panel2:#0E1630;
  --text:#DDE7FF;
  --muted:#92A4D6;
  --neon:#37F6FF;
  --neon2:#B14CFF;
  --good:#37ff8b;
  --bad:#ff3b6d;
  --warn:#ffd23f;
  --border: rgba(55,246,255,0.18);
  --shadow: 0 0 18px rgba(55,246,255,0.12);
}

.stApp{
  background: radial-gradient(1200px 700px at 20% 0%, rgba(177,76,255,0.14), transparent 60%),
              radial-gradient(900px 600px at 95% 12%, rgba(55,246,255,0.16), transparent 55%),
              linear-gradient(180deg, var(--bg), #050610 80%);
  color: var(--text);
}

[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(11,16,32,0.95), rgba(8,10,18,0.95));
  border-right: 1px solid var(--border);
}

.block-container{ padding-top: 1.1rem; }

div[data-testid="stVerticalBlockBorderWrapper"]{
  background: rgba(11,16,32,0.78);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: var(--shadow);
}

.stButton button{
  background: linear-gradient(90deg, rgba(55,246,255,0.18), rgba(177,76,255,0.18));
  border: 1px solid rgba(55,246,255,0.35);
  color: var(--text);
  border-radius: 14px;
  box-shadow: 0 0 16px rgba(55,246,255,0.10);
}
.stButton button:hover{
  border: 1px solid rgba(55,246,255,0.65);
  box-shadow: 0 0 22px rgba(55,246,255,0.18);
  transform: translateY(-1px);
}

.stTextInput input, .stNumberInput input, .stTextArea textarea{
  background: rgba(14,22,48,0.55) !important;
  border: 1px solid rgba(55,246,255,0.18) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
}

.stSelectbox div, .stMultiSelect div{
  background: rgba(14,22,48,0.55) !important;
  border-radius: 12px !important;
}

[data-testid="stDataFrame"]{
  border: 1px solid rgba(55,246,255,0.20);
  border-radius: 16px;
  box-shadow: var(--shadow);
}

hr{
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(55,246,255,0.35), transparent);
}
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

st.title("⚡ Neon Market Analyst")
st.markdown(
    """
**Institutional-style screening + portfolio construction** (educational).  
Includes **regime-aware scoring**, **Plotly charts**, **signals + trades**, and **broadcast hooks**.
"""
)
st.caption("Educational tool. Not investment advice. Data quality varies by ticker/provider.")


# =============================================================================
# SECRETS / BROADCAST (Telegram + TradingView webhook)
# =============================================================================
def _get_secret(path: List[str], default: str = "") -> str:
    cur = st.secrets if hasattr(st, "secrets") else {}
    try:
        for k in path:
            cur = cur[k]
        return str(cur)
    except Exception:
        # environment fallback
        env_key = "_".join([p.upper() for p in path])
        return str(os.environ.get(env_key, default) or default)

def _mask(s: str) -> str:
    if not s:
        return ""
    if len(s) <= 6:
        return "***"
    return s[:2] + "***" + s[-2:]

def escape_html(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

@dataclass
class TelegramSender:
    token: str = ""
    chat_id: str = ""

    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send_message(self, text_html: str) -> bool:
        if not self.enabled():
            return False
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                data={"chat_id": self.chat_id, "text": text_html, "parse_mode": "HTML"},
                timeout=25,
            )
            return True
        except Exception as e:
            st.error(f"Telegram sendMessage error: {e}")
            return False

    def send_file(self, file_bytes: bytes, filename: str) -> bool:
        if not self.enabled():
            return False
        try:
            bio = io.BytesIO(file_bytes)
            bio.seek(0)
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendDocument",
                data={"chat_id": self.chat_id, "caption": "📎 Report attached"},
                files={"document": (filename, bio, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                timeout=60,
            )
            return True
        except Exception as e:
            st.error(f"Telegram sendDocument error: {e}")
            return False

@dataclass
class TradingViewWebhook:
    url: str = ""
    passphrase: str = ""

    def enabled(self) -> bool:
        return bool(self.url)

    def post_alert(self, payload: dict) -> bool:
        if not self.enabled():
            return False
        body = dict(payload)
        if self.passphrase:
            body["passphrase"] = self.passphrase
        try:
            r = requests.post(self.url, json=body, timeout=20)
            return 200 <= r.status_code < 300
        except Exception as e:
            st.error(f"TradingView webhook error: {e}")
            return False

telegram = TelegramSender(
    token=_get_secret(["telegram", "bot_token"], default=_get_secret(["TELEGRAM_TOKEN"], "")),
    chat_id=_get_secret(["telegram", "chat_id"], default=_get_secret(["TELEGRAM_CHAT_ID"], "")),
)
tv_webhook = TradingViewWebhook(
    url=_get_secret(["tradingview", "webhook_url"], default=_get_secret(["TRADINGVIEW_WEBHOOK_URL"], "")),
    passphrase=_get_secret(["tradingview", "passphrase"], default=_get_secret(["TRADINGVIEW_PASSPHRASE"], "")),
)

# =============================================================================
# OPENAI (optional) — NO-OP if missing key
# =============================================================================
def make_openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)

OPENAI_API_KEY = _get_secret(["OPENAI_API_KEY"], "")
# (We do not echo secrets — only masked status)
OPENAI_STATUS = bool(OPENAI_API_KEY)


# =============================================================================
# DATA: UNIVERSE SOURCES
# =============================================================================
UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (NeonAnalyst/1.0; +https://streamlit.io)",
    "Accept": "*/*",
}

ISHARES_PRODUCTS = {
    "ACWI (Global All-Country)": {
        "ticker": "ACWI",
        "holdings_csv": "https://www.ishares.com/us/products/239600/ishares-msci-acwi-etf/1467271812596.ajax?dataType=fund&fileName=ACWI_holdings&fileType=csv",
    },
    "ITOT (US Total Market)": {
        "ticker": "ITOT",
        "holdings_csv": "https://www.ishares.com/us/products/239724/ishares-core-sp-total-us-stock-market-etf/1467271812596.ajax?dataType=fund&fileName=ITOT_holdings&fileType=csv",
    },
    "IWM (Russell 2000)": {
        "ticker": "IWM",
        "holdings_csv": "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?dataType=fund&fileName=IWM_holdings&fileType=csv",
    },
    "EWU (UK Large+Mid)": {
        "ticker": "EWU",
        "holdings_csv": "https://www.ishares.com/us/products/239690/ishares-msci-united-kingdom-etf/1467271812596.ajax?dataType=fund&fileName=EWU_holdings&fileType=csv",
    },
    "EZU (Eurozone Large+Mid)": {
        "ticker": "EZU",
        "holdings_csv": "https://www.ishares.com/us/products/239644/ishares-msci-emu-etf/1467271812596.ajax?dataType=fund&fileName=EZU_holdings&fileType=csv",
    },
    "IEUR (Europe Broad)": {
        "ticker": "IEUR",
        "holdings_csv": "https://www.ishares.com/us/products/264617/ishares-core-msci-europe-etf/1467271812596.ajax?dataType=fund&fileName=IEUR_holdings&fileType=csv",
    },
}

VANECK_GDXJ_HOLDINGS = "https://www.vaneck.com/us/en/investments/junior-gold-miners-etf-gdxj/downloads/holdings/"

EXCHANGE_SUFFIX = {
    "London Stock Exchange": ".L", "LSE": ".L",
    "Euronext Amsterdam": ".AS",
    "XETRA": ".DE", "Deutsche Börse": ".DE",
    "Euronext Paris": ".PA",
    "SIX Swiss Exchange": ".SW",
    "Borsa Italiana": ".MI",
    "Bolsa de Madrid": ".MC",
    "Nasdaq Stockholm": ".ST",
    "Oslo Stock Exchange": ".OL",
    "Copenhagen Stock Exchange": ".CO",
    "Helsinki Stock Exchange": ".HE",
    "Euronext Brussels": ".BR",
}

def _safe_read_csv_bytes(b: bytes) -> pd.DataFrame:
    text = b.decode("utf-8", errors="ignore").splitlines()
    header_idx = 0
    for i, line in enumerate(text[:80]):
        if "Ticker" in line and "Name" in line:
            header_idx = i
            break
    return pd.read_csv(io.StringIO("\n".join(text[header_idx:])))

@st.cache_data(ttl=6 * 3600)
def fetch_ishares_holdings_tickers(preset_name: str, max_names: int = 2500) -> List[str]:
    cfg = ISHARES_PRODUCTS[preset_name]
    url = cfg["holdings_csv"]
    try:
        r = requests.get(url, headers=UA_HEADERS, timeout=40)
        df = _safe_read_csv_bytes(r.content)
        if "Ticker" not in df.columns:
            return []
        df = df.dropna(subset=["Ticker"]).copy()
        df["Ticker"] = df["Ticker"].astype(str).str.strip()
        if "Exchange" in df.columns:
            df["Exchange"] = df["Exchange"].astype(str).str.strip()

        tickers = []
        for _, row in df.iterrows():
            t = str(row["Ticker"]).strip()
            if t in ("-", "nan", "None", ""):
                continue
            if "." in t:
                tickers.append(t)
                continue
            ex = str(row.get("Exchange", "")).strip()
            suf = EXCHANGE_SUFFIX.get(ex, "")
            tickers.append(t + suf if suf else t)
            if len(tickers) >= max_names:
                break
        return sorted(list(dict.fromkeys(tickers)))
    except Exception:
        return []

@st.cache_data(ttl=6 * 3600)
def fetch_ftse100_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    try:
        tables = pd.read_html(url)
        target = None
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if "ticker" in cols:
                target = t
                break
        if target is None:
            return []
        col_map = {c: "Ticker" for c in target.columns if str(c).lower() == "ticker"}
        target = target.rename(columns=col_map)

        tickers = []
        for x in target["Ticker"].dropna().astype(str).tolist():
            t = x.strip().upper()
            if not t:
                continue
            if "." not in t:
                t = t + ".L"
            tickers.append(t)
        return sorted(list(dict.fromkeys(tickers)))
    except Exception:
        return []

@st.cache_data(ttl=6 * 3600)
def fetch_eurostoxx50_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
    try:
        tables = pd.read_html(url)
        target = None
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if "ticker" in cols:
                target = t
                break
        if target is None:
            return []
        col_map = {c: "Ticker" for c in target.columns if str(c).lower() == "ticker"}
        target = target.rename(columns=col_map)
        tickers = target["Ticker"].dropna().astype(str).str.strip().tolist()
        tickers = [t for t in tickers if t and t.lower() != "nan"]
        return sorted(list(dict.fromkeys(tickers)))
    except Exception:
        return []

@st.cache_data(ttl=6 * 3600)
def fetch_vaneck_gdxj_holdings(max_names: int = 300) -> List[str]:
    fallback = ["AEM", "NEM", "GOLD", "WPM", "FNV", "RGLD", "AU", "KGC", "HL", "AG", "SSRM", "IAG", "HMY"]
    try:
        r = requests.get(VANECK_GDXJ_HOLDINGS, headers=UA_HEADERS, timeout=45)
        text = r.content.decode("utf-8", errors="ignore").splitlines()
        header_idx = None
        for i, line in enumerate(text[:120]):
            if "Ticker" in line and ("Holding" in line or "Name" in line):
                header_idx = i
                break
        if header_idx is None:
            return sorted(list(dict.fromkeys(fallback)))

        df = pd.read_csv(io.StringIO("\n".join(text[header_idx:])))
        if "Ticker" not in df.columns:
            return sorted(list(dict.fromkeys(fallback)))

        tickers = []
        for x in df["Ticker"].dropna().astype(str).tolist():
            t = x.strip()
            if not t or t.lower() == "nan":
                continue
            t = re.sub(r"\s+", " ", t)
            if " " in t:
                base, market = t.split(" ", 1)
                market = market.strip().upper()
                suf = {"LN": ".L", "CN": ".TO", "AU": ".AX", "HK": ".HK"}.get(market, "")
                tickers.append(base + suf)
            else:
                tickers.append(t)
            if len(tickers) >= max_names:
                break
        return sorted(list(dict.fromkeys(tickers)))
    except Exception:
        return sorted(list(dict.fromkeys(fallback)))


# =============================================================================
# DATA: OHLCV download + cleaning + resampling
# =============================================================================
def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out.sort_index()
    return out

def _cast_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    cols = list(out.columns)
    # handle yfinance variants
    rename_map = {}
    for c in cols:
        lc = str(c).lower()
        if lc == "adj close":
            rename_map[c] = "Adj Close"
        elif lc == "close":
            rename_map[c] = "Close"
        elif lc == "open":
            rename_map[c] = "Open"
        elif lc == "high":
            rename_map[c] = "High"
        elif lc == "low":
            rename_map[c] = "Low"
        elif lc == "volume":
            rename_map[c] = "Volume"
    if rename_map:
        out = out.rename(columns=rename_map)

    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # basic OHLC sanity: drop rows with all OHLC missing
    ohlc_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in out.columns]
    if ohlc_cols:
        out = out.dropna(how="all", subset=ohlc_cols)
    return out

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Rule examples: '1D', '1W', '4H' (yfinance data may not support intraday for many tickers)
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if rule in ("1D", "1W") or rule.endswith(("D", "W", "M")):
        agg = {}
        if "Open" in df.columns: agg["Open"] = "first"
        if "High" in df.columns: agg["High"] = "max"
        if "Low" in df.columns: agg["Low"] = "min"
        if "Close" in df.columns: agg["Close"] = "last"
        if "Adj Close" in df.columns: agg["Adj Close"] = "last"
        if "Volume" in df.columns: agg["Volume"] = "sum"
        out = df.resample(rule).agg(agg)
        out = out.dropna(how="all")
        return out
    return df

def _to_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Adj Close" in df.columns:
        return df["Adj Close"].dropna()
    if "Close" in df.columns:
        return df["Close"].dropna()
    return pd.Series(dtype=float)

def generate_dummy_ohlcv(
    symbols: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = "1D",
    seed: int = 7,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    out = {}
    for i, s in enumerate(symbols):
        n = len(idx)
        if n < 50:
            continue
        drift = 0.0002 + 0.0001 * (i % 5)
        vol = 0.012 + 0.003 * (i % 7)
        rets = drift + vol * rng.standard_normal(n)
        px = 100 * np.exp(np.cumsum(rets))
        close = pd.Series(px, index=idx)
        open_ = close.shift(1).fillna(close.iloc[0]) * (1 + 0.001 * rng.standard_normal(n))
        high = pd.concat([open_, close], axis=1).max(axis=1) * (1 + np.abs(0.002 * rng.standard_normal(n)))
        low = pd.concat([open_, close], axis=1).min(axis=1) * (1 - np.abs(0.002 * rng.standard_normal(n)))
        volu = (1e6 * (1 + 0.2 * rng.standard_normal(n))).clip(1e5, 5e6)
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volu}, index=idx)
        out[s] = df
    return out

@st.cache_data(ttl=2 * 3600)
def download_prices_chunk(tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        df = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def download_prices_all(tickers: List[str], period: str, interval: str, chunk_size: int = 80) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    tickers = [t for t in tickers if t]
    if not tickers:
        return out
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        df = download_prices_chunk(chunk, period, interval)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            for t in chunk:
                if t in df.columns.get_level_values(0):
                    sub = df[t].dropna(how="all")
                    sub = _ensure_utc_index(_cast_ohlcv(sub))
                    if not sub.empty:
                        out[t] = sub
        else:
            t = chunk[0]
            sub = _ensure_utc_index(_cast_ohlcv(df.dropna(how="all")))
            if not sub.empty:
                out[t] = sub
    return out


# =============================================================================
# INDICATORS (vectorized pandas) + Pine-ish adapter
# =============================================================================
def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ag = up.ewm(alpha=1 / period, adjust=False).mean()
    al = dn.ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = ema(close, fast) - ema(close, slow)
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    if df.empty or not all(c in df.columns for c in ["High", "Low", "Close"]):
        return pd.Series(dtype=float)
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()

def bollinger(close: pd.Series, length: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return ma, upper, lower

class Pine:
    """Tiny Pine-style helpers: nz, na, crossover/crossunder, ta.ema/ta.rsi/etc."""
    @staticmethod
    def na(x) -> pd.Series:
        return x.isna()

    @staticmethod
    def nz(x: pd.Series, repl: float = 0.0) -> pd.Series:
        return x.fillna(repl)

    @staticmethod
    def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a > b) & (a.shift(1) <= b.shift(1))

    @staticmethod
    def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a < b) & (a.shift(1) >= b.shift(1))

    class ta:
        @staticmethod
        def ema(x: pd.Series, length: int) -> pd.Series:
            return ema(x, length)

        @staticmethod
        def rsi(x: pd.Series, length: int) -> pd.Series:
            return rsi_wilder(x, length)

        @staticmethod
        def atr(df: pd.DataFrame, length: int) -> pd.Series:
            return atr(df, length)


# =============================================================================
# QUANT METRICS
# =============================================================================
def annualized_vol(ret: pd.Series) -> float:
    if len(ret) < 30:
        return float("nan")
    return float(ret.std() * math.sqrt(252))

def annualized_return(close: pd.Series) -> float:
    if len(close) < 2:
        return float("nan")
    days = (close.index[-1] - close.index[0]).days
    if days <= 0:
        return float("nan")
    total = close.iloc[-1] / close.iloc[0]
    return float(total ** (365.25 / days) - 1)

def max_drawdown(close: pd.Series) -> float:
    peak = close.cummax()
    dd = (close / peak) - 1.0
    return float(dd.min())

def ulcer_index(close: pd.Series) -> float:
    peak = close.cummax()
    dd = (close / peak - 1.0) * 100.0
    return float(np.sqrt(np.mean(dd.values ** 2))) if len(dd) else float("nan")

def omega_ratio(ret: pd.Series, threshold: float = 0.0) -> float:
    if len(ret) < 60:
        return float("nan")
    gains = (ret - threshold).clip(lower=0).sum()
    losses = (threshold - ret).clip(lower=0).sum()
    if losses == 0:
        return float("nan")
    return float(gains / losses)

def historical_var_cvar(ret: pd.Series, level: float = 0.05) -> Tuple[float, float]:
    if len(ret) < 60:
        return (float("nan"), float("nan"))
    q = ret.quantile(level)
    cvar = ret[ret <= q].mean()
    return (float(q), float(cvar))

def sharpe_ratio(ret: pd.Series, rf_annual: float) -> float:
    if len(ret) < 60:
        return float("nan")
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    ex = ret - rf_daily
    sd = ex.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(ex.mean() / sd * math.sqrt(252))

def sortino_ratio(ret: pd.Series, rf_annual: float) -> float:
    if len(ret) < 60:
        return float("nan")
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    ex = ret - rf_daily
    dn = ex[ex < 0]
    sd = dn.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(ex.mean() / sd * math.sqrt(252))

def calmar_ratio(cagr: float, mdd: float) -> float:
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return float("nan")
    return float(cagr / abs(mdd))

def hurst_exponent(close: pd.Series) -> float:
    if len(close) < 220:
        return float("nan")
    x = close.values
    lags = [2, 5, 10, 20, 50, 100]
    tau, valid = [], []
    for lag in lags:
        if lag >= len(x):
            continue
        diff = x[lag:] - x[:-lag]
        v = np.std(diff)
        if v <= 0:
            continue
        tau.append(np.sqrt(v))
        valid.append(lag)
    if len(tau) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(valid), np.log(tau), 1)
    return float(slope * 2.0)

def alpha_beta(asset_ret: pd.Series, bench_ret: pd.Series, rf_annual: float) -> Tuple[float, float]:
    df = pd.DataFrame({"a": asset_ret, "b": bench_ret}).dropna()
    if len(df) < 90:
        return (float("nan"), float("nan"))
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    y = df["a"] - rf_daily
    x = df["b"] - rf_daily
    xv = np.var(x)
    if xv == 0 or np.isnan(xv):
        return (float("nan"), float("nan"))
    beta = float(np.cov(x, y)[0, 1] / xv)
    alpha_d = float(y.mean() - beta * x.mean())
    alpha_a = float((1 + alpha_d) ** 252 - 1)
    return (alpha_a, beta)

def breakout_strength(close: pd.Series, lookback: int = 126) -> float:
    if len(close) < lookback + 5:
        return float("nan")
    win = close.iloc[-lookback:]
    lo, hi = float(win.min()), float(win.max())
    if hi == lo:
        return float("nan")
    return float((close.iloc[-1] - lo) / (hi - lo))

def zscore_20(close: pd.Series) -> float:
    if len(close) < 25:
        return float("nan")
    w = close.iloc[-20:]
    mu = w.mean()
    sd = w.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((close.iloc[-1] - mu) / sd)

def expectancy_signal(close: pd.Series) -> float:
    # educational heuristic only
    if len(close) < 260:
        return float("nan")
    fwd = close.pct_change(20).shift(-20)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi14 = rsi_wilder(close, 14)
    z20 = (close - close.rolling(20).mean()) / close.rolling(20).std()

    trend_cond = (close > sma200) & (rsi14 >= 45) & (rsi14 <= 70)
    mr_cond = (close < sma50) & (z20 < -1.0)

    mask = (trend_cond | mr_cond).fillna(False)
    sample = fwd[mask].dropna()
    if len(sample) < 20:
        return float("nan")
    mu, sd = float(sample.mean()), float(sample.std())
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(mu / sd)

def detect_regime(close: pd.Series) -> str:
    if len(close) < 220:
        return "Unknown"
    sma200 = close.rolling(200).mean()
    slope = (sma200.iloc[-1] - sma200.iloc[-20]) / (abs(sma200.iloc[-20]) + 1e-9)
    h = hurst_exponent(close)
    h = 0.5 if (h is None or np.isnan(h)) else h
    if slope > 0 and h > 0.55:
        return "Trend"
    if slope < 0 and h < 0.50:
        return "MeanReversion"
    return "Mixed"


# =============================================================================
# SCORING
# =============================================================================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def safe_float(x) -> Optional[float]:
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except Exception:
        return None

def score_row(row: dict, weights: dict, regime: str) -> float:
    sharpe = safe_float(row.get("Sharpe"))
    sortino = safe_float(row.get("Sortino"))
    calmar = safe_float(row.get("Calmar"))
    omega = safe_float(row.get("Omega"))
    ulcer = safe_float(row.get("Ulcer"))
    mdd = safe_float(row.get("MaxDD"))
    vol = safe_float(row.get("Vol"))
    alpha = safe_float(row.get("Alpha"))
    mom = safe_float(row.get("CAGR"))
    brk = safe_float(row.get("Breakout"))
    hurst = safe_float(row.get("Hurst"))
    rsi14 = safe_float(row.get("RSI14"))
    z20 = safe_float(row.get("Z20"))
    rsi2 = safe_float(row.get("RSI2"))
    exp = safe_float(row.get("Expectancy"))

    def mean_or0(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else 0.0

    risk = mean_or0([
        clamp01((sharpe + 0.5) / 2.5) if sharpe is not None else None,
        clamp01((sortino + 0.5) / 3.5) if sortino is not None else None,
        clamp01((calmar + 0.2) / 2.2) if calmar is not None else None,
        clamp01((omega - 0.8) / 1.2) if omega is not None else None,
        clamp01((alpha + 0.10) / 0.40) if alpha is not None else None,
        clamp01((0.0 - mdd) / 0.60) if mdd is not None else None,
        clamp01((0.90 - vol) / 0.90) if vol is not None else None,
        clamp01((10.0 - ulcer) / 10.0) if ulcer is not None else None,
    ])

    trend = mean_or0([
        clamp01((mom + 0.10) / 0.50) if mom is not None else None,
        clamp01(brk) if brk is not None else None,
        clamp01(1 - abs((hurst or 0.5) - 0.62) / 0.25) if hurst is not None else None,
        clamp01(1 - abs((rsi14 or 50) - 55) / 35) if rsi14 is not None else None,
    ])

    meanrev = mean_or0([
        clamp01((-z20) / 2.5) if z20 is not None else None,
        clamp01((30 - rsi2) / 30) if rsi2 is not None else None,
    ])

    edge = mean_or0([
        clamp01((exp + 0.10) / 0.30) if exp is not None else None,
    ])

    rw = dict(weights)
    if regime == "Trend":
        rw["trend"] *= 1.20
        rw["meanrev"] *= 0.80
        rw["edge"] *= 1.10
    elif regime == "MeanReversion":
        rw["trend"] *= 0.85
        rw["meanrev"] *= 1.25
        rw["edge"] *= 1.10

    denom = sum(rw.values()) if sum(rw.values()) > 0 else 1.0
    score = (rw["risk"] * risk + rw["trend"] * trend + rw["meanrev"] * meanrev + rw["edge"] * edge) / denom
    return round(score * 100, 1)


# =============================================================================
# SIGNALS + BACKTEST (single-name, educational)
# Avoid look-ahead: entries/exits executed on NEXT bar OPEN (if available), else next Close.
# =============================================================================
@dataclass
class BacktestConfig:
    fee_bps: float = 5.0        # per side
    slippage_bps: float = 2.0   # per side
    position_size: float = 1.0  # fraction of equity allocated per trade (single asset)
    allow_short: bool = False

def _bps_cost(bps: float) -> float:
    return bps / 10000.0

def build_signals(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Returns df with columns: ema_fast, ema_slow, rsi14, bb_mid, bb_up, bb_dn, macd, macd_sig, atr, signal
    signal: +1 enter long, -1 exit long (or enter short if allow_short; we keep it long-only by default)
    """
    if df.empty:
        return df
    out = df.copy()
    close = out["Close"]

    out["ema_fast"] = ema(close, 20)
    out["ema_slow"] = ema(close, 50)
    out["rsi14"] = rsi_wilder(close, 14)
    out["macd"], out["macd_sig"], out["macd_hist"] = macd(close, 12, 26, 9)
    out["atr14"] = atr(out, 14)
    out["bb_mid"], out["bb_up"], out["bb_dn"] = bollinger(close, 20, 2.0)

    # Pine-ish cross logic
    if mode == "Trend (EMA cross)":
        enter = Pine.crossover(out["ema_fast"], out["ema_slow"])
        exit_ = Pine.crossunder(out["ema_fast"], out["ema_slow"])
    elif mode == "Mean Reversion (BB + RSI)":
        enter = (close < out["bb_dn"]) & (out["rsi14"] < 30)
        exit_ = (close > out["bb_mid"]) | (out["rsi14"] > 55)
    else:  # MACD momentum
        enter = Pine.crossover(out["macd"], out["macd_sig"])
        exit_ = Pine.crossunder(out["macd"], out["macd_sig"])

    out["signal"] = 0
    out.loc[enter.fillna(False), "signal"] = 1
    out.loc[exit_.fillna(False), "signal"] = -1
    return out

def run_backtest_single(df: pd.DataFrame, cfg: BacktestConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    df must have Open/Close and signal column.
    Returns trades table and equity curve (normalized to 1.0).
    """
    if df.empty or "signal" not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float)

    d = df.copy()
    # Execution price = next open if exists, else next close; shift(-1)
    exec_px = d["Open"].shift(-1)
    exec_px = exec_px.fillna(d["Close"].shift(-1)).fillna(d["Close"])
    d["exec_px"] = exec_px

    fee = _bps_cost(cfg.fee_bps)
    slip = _bps_cost(cfg.slippage_bps)
    cost = fee + slip

    in_pos = False
    entry_time = None
    entry_px = None

    equity = 1.0
    eq_curve = []
    eq_index = []

    trades = []
    # mark-to-market daily using close-to-close while in position; costs on entry/exit
    close = d["Close"]

    # build position series without look-ahead: position toggled on bar of signal, applied from next bar
    pos = pd.Series(0, index=d.index, dtype=float)
    for i in range(len(d) - 1):  # last bar can't execute next
        sig = int(d["signal"].iloc[i]) if pd.notna(d["signal"].iloc[i]) else 0
        if not in_pos and sig == 1:
            in_pos = True
            entry_time = d.index[i + 1]  # execution time next bar
            entry_px = float(d["exec_px"].iloc[i])
            equity *= (1 - cost)  # entry cost
        elif in_pos and sig == -1:
            # exit next bar
            exit_time = d.index[i + 1]
            exit_px = float(d["exec_px"].iloc[i])
            pnl = (exit_px / entry_px - 1.0) * cfg.position_size
            equity *= (1 + pnl)
            equity *= (1 - cost)  # exit cost
            trades.append({
                "entry_time": entry_time,
                "entry_px": entry_px,
                "exit_time": exit_time,
                "exit_px": exit_px,
                "return": pnl,
                "win": pnl > 0,
                "bars": int((exit_time - entry_time).days) if isinstance(exit_time, pd.Timestamp) else None,
            })
            in_pos = False
            entry_time, entry_px = None, None

        pos.iloc[i + 1] = 1.0 if in_pos else 0.0

    # equity curve: apply daily returns when in position
    ret = close.pct_change().fillna(0.0)
    strat_ret = ret * pos.shift(1).fillna(0.0) * cfg.position_size
    eq = (1.0 + strat_ret).cumprod()
    # apply costs already applied multiplicatively to equity scalar above? Keep it consistent by applying scalar to curve end:
    # We'll approximate by scaling the curve to match final equity after discrete cost events.
    if len(eq) > 0 and eq.iloc[-1] != 0:
        scale = equity / float(eq.iloc[-1])
        eq = eq * scale
    return pd.DataFrame(trades), eq.rename("equity")

def trade_kpis(trades: pd.DataFrame, equity: pd.Series) -> dict:
    if equity is None or equity.empty:
        return {}
    ret = equity.pct_change().dropna()
    cagr = annualized_return(equity)
    vol = annualized_vol(ret)
    mdd = max_drawdown(equity)
    sharpe = sharpe_ratio(ret, 0.0)
    win_rate = float(trades["win"].mean()) if trades is not None and len(trades) else float("nan")
    avg_trade = float(trades["return"].mean()) if trades is not None and len(trades) else float("nan")
    if trades is not None and len(trades):
        gains = trades.loc[trades["return"] > 0, "return"].sum()
        losses = -trades.loc[trades["return"] < 0, "return"].sum()
        profit_factor = float(gains / losses) if losses > 0 else float("nan")
    else:
        profit_factor = float("nan")
    exposure = float((equity.pct_change().fillna(0.0) != 0).mean())
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "WinRate": win_rate,
        "ProfitFactor": profit_factor,
        "AvgTrade": avg_trade,
        "Trades": int(len(trades)) if trades is not None else 0,
        "Exposure": exposure,
    }


# =============================================================================
# PORTFOLIO BUILDER (Equal / InvVol / RiskParity)
# =============================================================================
def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s

def weights_equal(n: int) -> np.ndarray:
    return np.ones(n) / n

def weights_inverse_vol(returns: pd.DataFrame) -> np.ndarray:
    vol = returns.std().replace(0, np.nan)
    inv = 1.0 / vol
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _normalize_weights(inv.values)

def weights_risk_parity(cov: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return np.ones(n) / n
        mrc = cov @ w
        rc = w * mrc / port_var
        target = np.ones(n) / n
        w_new = w * (target / (rc + 1e-12))
        w_new = _normalize_weights(w_new)
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    return w

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if index is None or len(index) == 0:
        return pd.DatetimeIndex([])
    if freq == "None":
        return pd.DatetimeIndex([index[0]])
    if freq == "Monthly":
        g = pd.Series(index=index, data=1).groupby(index.to_period("M")).head(1).index
        return pd.DatetimeIndex(sorted(set(g)))
    if freq == "Weekly":
        g = pd.Series(index=index, data=1).groupby(index.to_period("W")).head(1).index
        return pd.DatetimeIndex(sorted(set(g)))
    return pd.DatetimeIndex([index[0]])

def backtest_portfolio(
    prices: pd.DataFrame,
    method: str,
    rf_annual: float,
    rebalance: str,
    lookback_days: int,
    max_weight: float,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    prices = prices.dropna(how="all").ffill().dropna()
    if prices.empty or prices.shape[1] < 2:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    rets = prices.pct_change().dropna()
    idx = rets.index
    rdates = rebalance_dates(idx, rebalance)

    w_hist, date_hist = [], []
    port_ret = pd.Series(0.0, index=idx)
    last_rc = None

    for i, d in enumerate(rdates):
        d_loc = idx.get_loc(d)
        start = max(0, d_loc - lookback_days)
        window = rets.iloc[start:d_loc].copy()

        if len(window) < min(40, lookback_days // 3):
            w = weights_equal(rets.shape[1])
        else:
            if method == "Equal Weight":
                w = weights_equal(rets.shape[1])
            elif method == "Inverse Vol":
                w = weights_inverse_vol(window)
            else:
                cov = window.cov().values
                w = weights_risk_parity(cov)

        w = np.minimum(w, max_weight)
        w = _normalize_weights(w)

        next_loc = idx.get_loc(rdates[i + 1]) if i + 1 < len(rdates) else len(idx)
        seg = rets.iloc[d_loc:next_loc]
        port_ret.iloc[d_loc:next_loc] = seg.fillna(0.0).values @ w

        w_hist.append(w)
        date_hist.append(d)

        if len(window) >= 40:
            cov = window.cov().values
            pv = float(w @ cov @ w)
            mrc = cov @ w
            last_rc = (w * mrc) / (pv + 1e-12)

    equity = (1.0 + port_ret).cumprod()
    w_df = pd.DataFrame(w_hist, index=pd.DatetimeIndex(date_hist), columns=rets.columns)

    rc_df = pd.DataFrame({
        "ticker": rets.columns,
        "risk_contribution": last_rc if last_rc is not None else np.nan,
    }).sort_values("risk_contribution", ascending=False)

    return equity.rename("equity"), w_df, rc_df

def portfolio_stats(equity: pd.Series, rf_annual: float) -> dict:
    if equity is None or equity.empty:
        return {}
    ret = equity.pct_change().dropna()
    return {
        "CAGR": annualized_return(equity),
        "Vol": annualized_vol(ret),
        "Sharpe": sharpe_ratio(ret, rf_annual),
        "Sortino": sortino_ratio(ret, rf_annual),
        "Calmar": calmar_ratio(annualized_return(equity), max_drawdown(equity)),
        "Omega": omega_ratio(ret, 0.0),
        "MaxDD": max_drawdown(equity),
        "Ulcer": ulcer_index(equity),
        "VaR_95": historical_var_cvar(ret, 0.05)[0],
        "CVaR_95": historical_var_cvar(ret, 0.05)[1],
    }


# =============================================================================
# EXPORT
# =============================================================================
def build_excel(df_ranked: pd.DataFrame, settings: dict, trades: Optional[pd.DataFrame] = None, portfolio: Optional[pd.DataFrame] = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        if df_ranked is not None and not df_ranked.empty:
            df_ranked.to_excel(w, sheet_name="Ranked", index=False)
        pd.DataFrame([settings]).to_excel(w, sheet_name="Settings", index=False)
        if trades is not None and not trades.empty:
            trades.to_excel(w, sheet_name="Trades", index=False)
        if portfolio is not None and not portfolio.empty:
            portfolio.to_excel(w, sheet_name="Portfolio", index=False)
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# SCAN: compute metrics for ticker
# =============================================================================
def compute_metrics_for_ticker(t: str, df: pd.DataFrame, rf_annual: float, bench_ret: Optional[pd.Series]) -> Optional[dict]:
    close = _to_close(df)
    if close.empty or len(close) < 120:
        return None
    ret = close.pct_change().dropna()

    cagr = annualized_return(close)
    vol = annualized_vol(ret)
    mdd = max_drawdown(close)
    ui = ulcer_index(close)
    rsi14 = float(rsi_wilder(close, 14).iloc[-1])
    rsi2 = float(rsi_wilder(close, 2).iloc[-1]) if len(close) > 10 else float("nan")
    z20 = zscore_20(close)
    brk = breakout_strength(close, 126)
    h = hurst_exponent(close)
    exp = expectancy_signal(close)
    var95, cvar95 = historical_var_cvar(ret, 0.05)
    sharpe = sharpe_ratio(ret, rf_annual)
    sortino = sortino_ratio(ret, rf_annual)
    calmar = calmar_ratio(cagr, mdd)
    omega = omega_ratio(ret, 0.0)
    alpha, beta = (float("nan"), float("nan"))
    if bench_ret is not None:
        alpha, beta = alpha_beta(ret, bench_ret, rf_annual)

    regime = detect_regime(close)

    return {
        "ticker": t,
        "Price": float(close.iloc[-1]),
        "CAGR": cagr,
        "Vol": vol,
        "MaxDD": mdd,
        "Ulcer": ui,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Omega": omega,
        "VaR_95": var95,
        "CVaR_95": cvar95,
        "Alpha": alpha,
        "Beta": beta,
        "RSI14": rsi14,
        "RSI2": rsi2,
        "Z20": z20,
        "Breakout": brk,
        "Hurst": h,
        "Expectancy": exp,
        "Regime": regime,
    }

def run_scan(
    tickers: List[str],
    period: str,
    interval: str,
    rf_annual: float,
    benchmark: str,
    max_tickers: int,
    chunk_size: int,
    weights: dict,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    tickers = tickers[:max_tickers]

    # benchmark
    bench_ret = None
    if benchmark:
        bmap = download_prices_all([benchmark], period, interval, chunk_size=1)
        if benchmark in bmap:
            bclose = _to_close(bmap[benchmark])
            if len(bclose) > 120:
                bench_ret = bclose.pct_change().dropna()

    # download in chunks
    price_map = download_prices_all(tickers, period, interval, chunk_size=chunk_size)

    # fallback dummy data if provider fails
    if not price_map:
        start = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=365 * 2)
        end = pd.Timestamp.utcnow().tz_localize("UTC")
        price_map = generate_dummy_ohlcv(tickers[: min(len(tickers), 60)], start, end, freq="1D")

    rows = []
    prog = st.progress(0)
    status = st.empty()
    items = list(price_map.items())
    total = len(items)

    for i, (t, df) in enumerate(items, start=1):
        status.text(f"Quant scan: {t} ({i}/{total})")
        try:
            r = compute_metrics_for_ticker(t, df, rf_annual, bench_ret)
            if r:
                rows.append(r)
        except Exception:
            pass
        prog.progress(i / max(total, 1))

    prog.empty()
    status.empty()

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["Score"] = out.apply(lambda r: score_row(r.to_dict(), weights, r.get("Regime", "Mixed")), axis=1)
    out = out.sort_values("Score", ascending=False).reset_index(drop=True)
    return out


# =============================================================================
# UI / SIDEBAR
# =============================================================================
st.sidebar.header("⚙️ Control Room")

# AI settings
if OPENAI_STATUS:
    st.sidebar.success("OpenAI key loaded (masked)")
else:
    OPENAI_API_KEY = st.sidebar.text_input("OpenAI API key", type="password")

model_name = st.sidebar.text_input("Model", value="gpt-4o-mini")
temperature = st.sidebar.slider("AI temperature", 0.0, 1.0, 0.35, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Broadcast")
st.sidebar.caption(f"Telegram: {'ON' if telegram.enabled() else 'OFF'}  ({_mask(telegram.token)})")
st.sidebar.caption(f"TradingView webhook: {'ON' if tv_webhook.enabled() else 'OFF'}")

st.sidebar.markdown("---")
st.sidebar.subheader("Market Universe")
preset = st.sidebar.selectbox(
    "Preset",
    options=[
        "ACWI (Global All-Country)",
        "ITOT (US Total Market)",
        "IWM (Russell 2000)",
        "FTSE 100 (Wikipedia)",
        "Euro Stoxx 50 (Wikipedia)",
        "EWU (UK Large+Mid)",
        "EZU (Eurozone Large+Mid)",
        "IEUR (Europe Broad)",
        "GDXJ (Junior Miners holdings)",
        "Custom",
    ],
    index=0,
)
custom_tickers = st.sidebar.text_area("Custom tickers (comma/space separated)", value="AAPL MSFT NVDA")
append_custom = st.sidebar.checkbox("Add custom tickers on top of preset", value=False)
max_universe = st.sidebar.slider("Max tickers to scan", 25, 2500, 300, 25)

st.sidebar.markdown("---")
st.sidebar.subheader("Time & Benchmark")
period = st.sidebar.selectbox("Lookback", ["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
benchmark = st.sidebar.text_input("Benchmark (alpha/beta)", value="SPY")
rf_rate = st.sidebar.slider("Risk-free (annual %)", 0.0, 10.0, 2.0, 0.25) / 100.0
chunk_size = st.sidebar.slider("Download chunk size", 20, 200, 80, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("Factor Studio (weights)")
preset_style = st.sidebar.selectbox(
    "Style preset",
    ["Balanced", "Trend Hunter", "Mean Reversion", "Defensive"],
    index=0,
)
STYLE_PRESETS = {
    "Balanced":       {"risk": 0.45, "trend": 0.30, "meanrev": 0.15, "edge": 0.10},
    "Trend Hunter":   {"risk": 0.35, "trend": 0.45, "meanrev": 0.05, "edge": 0.15},
    "Mean Reversion": {"risk": 0.35, "trend": 0.10, "meanrev": 0.40, "edge": 0.15},
    "Defensive":      {"risk": 0.60, "trend": 0.15, "meanrev": 0.10, "edge": 0.15},
}
w0 = STYLE_PRESETS[preset_style]
with st.sidebar.expander("Fine-tune weights", expanded=False):
    w_risk = st.slider("Risk-adjusted", 0.0, 1.0, float(w0["risk"]), 0.01)
    w_trend = st.slider("Trend / Momentum", 0.0, 1.0, float(w0["trend"]), 0.01)
    w_mr = st.slider("Mean reversion", 0.0, 1.0, float(w0["meanrev"]), 0.01)
    w_edge = st.slider("Edge proxy", 0.0, 1.0, float(w0["edge"]), 0.01)
weights = {"risk": w_risk, "trend": w_trend, "meanrev": w_mr, "edge": w_edge}

st.sidebar.markdown("---")
top_n = st.sidebar.slider("Top N to display", 5, 50, 15, 1)


# =============================================================================
# Build universe
# =============================================================================
def parse_tickers(s: str) -> List[str]:
    if not s.strip():
        return []
    raw = s.replace("\n", " ").replace(";", " ").replace(",", " ").split()
    return [t.strip().upper() for t in raw if t.strip()]

def get_universe(preset_name: str) -> List[str]:
    if preset_name == "Custom":
        return sorted(list(dict.fromkeys(parse_tickers(custom_tickers))))
    if preset_name == "FTSE 100 (Wikipedia)":
        return fetch_ftse100_tickers()
    if preset_name == "Euro Stoxx 50 (Wikipedia)":
        return fetch_eurostoxx50_tickers()
    if preset_name == "GDXJ (Junior Miners holdings)":
        return fetch_vaneck_gdxj_holdings(max_names=max_universe)
    if preset_name in ("EWU (UK Large+Mid)", "EZU (Eurozone Large+Mid)", "IEUR (Europe Broad)"):
        key = preset_name.split(" ")[0].strip()
        # map to ishares dict keys by ticker label
        mapping = {"EWU": "EWU (UK Large+Mid)", "EZU": "EZU (Eurozone Large+Mid)", "IEUR": "IEUR (Europe Broad)"}
        return fetch_ishares_holdings_tickers(mapping[key], max_names=max_universe)
    if preset_name in ISHARES_PRODUCTS:
        return fetch_ishares_holdings_tickers(preset_name, max_names=max_universe)
    return []

universe = get_universe(preset)
if append_custom and custom_tickers.strip():
    universe = sorted(list(dict.fromkeys(universe + parse_tickers(custom_tickers))))
universe = universe[:max_universe]


# =============================================================================
# Session state
# =============================================================================
if "ranked" not in st.session_state:
    st.session_state.ranked = pd.DataFrame()
if "excel_bytes" not in st.session_state:
    st.session_state.excel_bytes = None
if "last_settings" not in st.session_state:
    st.session_state.last_settings = {}
if "deep_data" not in st.session_state:
    st.session_state.deep_data = {}  # ticker -> ohlcv df
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame()
if "portfolio_excel" not in st.session_state:
    st.session_state.portfolio_excel = None


# =============================================================================
# Plotly helpers (Lead UI/UX Designer)
# =============================================================================
def plot_candles(df: pd.DataFrame, title: str, add_indicators: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="#37ff8b",
        decreasing_line_color="#ff3b6d",
        showlegend=False,
    ))
    if add_indicators:
        for col, color, width in [
            ("ema_fast", "#37F6FF", 1.6),
            ("ema_slow", "#B14CFF", 1.6),
            ("bb_mid", "#92A4D6", 1.0),
            ("bb_up", "rgba(55,246,255,0.55)", 1.0),
            ("bb_dn", "rgba(55,246,255,0.55)", 1.0),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", line=dict(color=color, width=width), name=col))
    fig.update_layout(
        title=title,
        height=520,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

def add_signal_markers(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    if "signal" not in df.columns:
        return fig
    ent = df[df["signal"] == 1]
    ex = df[df["signal"] == -1]
    if len(ent):
        fig.add_trace(go.Scatter(
            x=ent.index, y=ent["Low"] * 0.995,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="#37ff8b"),
            name="Enter",
        ))
    if len(ex):
        fig.add_trace(go.Scatter(
            x=ex.index, y=ex["High"] * 1.005,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#ff3b6d"),
            name="Exit",
        ))
    return fig


# =============================================================================
# Tabs
# =============================================================================
tab_screen, tab_deep, tab_port, tab_broadcast, tab_settings = st.tabs(
    ["📡 Screen", "🔍 Deep Dive", "🧺 Portfolio", "📣 Broadcast", "🧪 Theme"]
)

# -----------------------------
# Screen
# -----------------------------
with tab_screen:
    a, b, c, d = st.columns([1.4, 1, 1, 1])
    with a:
        st.markdown("### Market Scan")
        st.caption("Quant scan + regime-aware score. If data provider fails, demo uses synthetic OHLCV.")
    with b:
        st.metric("Universe", f"{len(universe)}")
    with c:
        st.metric("Lookback", period)
    with d:
        st.metric("Benchmark", benchmark if benchmark else "—")

    st.markdown("---")
    run = st.button("🚀 Run Neon Screen", use_container_width=True)

    if run:
        if not universe:
            st.error("Universe is empty. Choose a preset or add tickers.")
        else:
            df_ranked = run_scan(
                tickers=universe,
                period=period,
                interval=interval,
                rf_annual=rf_rate,
                benchmark=benchmark.strip().upper(),
                max_tickers=max_universe,
                chunk_size=chunk_size,
                weights=weights,
            )
            if df_ranked.empty:
                st.warning("No results. Try different tickers or longer lookback.")
            else:
                df_ranked = df_ranked.head(top_n).copy()

                trend_pct = float((df_ranked["Regime"] == "Trend").mean() * 100.0)
                mr_pct = float((df_ranked["Regime"] == "MeanReversion").mean() * 100.0)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Top regime: Trend", f"{trend_pct:.0f}%")
                m2.metric("Top regime: MeanRev", f"{mr_pct:.0f}%")
                m3.metric("Median Sharpe", f"{np.nanmedian(df_ranked['Sharpe']):.2f}")
                m4.metric("Median MaxDD", f"{np.nanmedian(df_ranked['MaxDD']):.2f}")

                settings = {
                    "date": date.today().isoformat(),
                    "preset": preset,
                    "period": period,
                    "interval": interval,
                    "benchmark": benchmark,
                    "rf_rate": rf_rate,
                    "max_universe": max_universe,
                    "chunk_size": chunk_size,
                    "weights": weights,
                    "style": preset_style,
                    "top_n": top_n,
                }

                st.session_state.ranked = df_ranked
                st.session_state.last_settings = settings
                st.session_state.excel_bytes = build_excel(df_ranked, settings)

    df_ranked = st.session_state.ranked
    if df_ranked is not None and not df_ranked.empty:
        st.markdown("---")
        st.markdown("### 🏆 Ranked Picks")
        show_cols = [
            "ticker", "Score", "Price", "Regime",
            "Sharpe", "Sortino", "Calmar", "Omega",
            "MaxDD", "Ulcer", "Vol",
            "Alpha", "Beta",
            "RSI14", "RSI2", "Z20", "Breakout", "Hurst", "Expectancy",
        ]
        show_cols = [c for c in show_cols if c in df_ranked.columns]
        st.dataframe(df_ranked[show_cols], use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            fname = f"Neon_Screen_{date.today().isoformat()}.xlsx"
            st.download_button("📥 Download Excel", data=st.session_state.excel_bytes, file_name=fname, use_container_width=True)
        with c2:
            if telegram.enabled():
                if st.button("📡 Send Telegram Signal (Top 5)", use_container_width=True):
                    topk = df_ranked.head(min(5, len(df_ranked)))
                    header = (
                        f"⚡ <b>NEON SIGNAL</b>\n"
                        f"Preset: <code>{escape_html(st.session_state.last_settings.get('preset',''))}</code> · "
                        f"Lookback: <code>{escape_html(period)}</code> · "
                        f"Benchmark: <code>{escape_html(benchmark)}</code>\n"
                        f"Date: <code>{escape_html(date.today().isoformat())}</code>\n\n"
                    )
                    lines = []
                    for i, r in topk.iterrows():
                        lines.append(
                            f"<b>{i+1})</b> <code>{escape_html(r['ticker'])}</code> · "
                            f"Score <b>{escape_html(r['Score'])}</b> · "
                            f"Sharpe {escape_html(round(r['Sharpe'],2) if pd.notna(r['Sharpe']) else 'NA')} · "
                            f"MDD {escape_html(round(r['MaxDD'],2) if pd.notna(r['MaxDD']) else 'NA')} · "
                            f"Regime <i>{escape_html(r['Regime'])}</i>"
                        )
                    footer = "\n\n<i>Educational only. Validate data independently.</i>"
                    ok = telegram.send_message(header + "\n".join(lines) + footer)
                    st.success("✅ Sent!") if ok else st.error("❌ Failed.")
            else:
                st.info("Telegram not configured (secrets.telegram.bot_token/chat_id or env).")
        with c3:
            if tv_webhook.enabled():
                if st.button("📣 Post TradingView Webhook (Top 1)", use_container_width=True):
                    top1 = df_ranked.iloc[0].to_dict()
                    payload = {
                        "event": "neon_screen_top1",
                        "ticker": top1.get("ticker"),
                        "score": float(top1.get("Score", np.nan)) if pd.notna(top1.get("Score", np.nan)) else None,
                        "regime": top1.get("Regime"),
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "note": "Educational screen signal",
                    }
                    ok = tv_webhook.post_alert(payload)
                    st.success("✅ Posted!") if ok else st.error("❌ Failed.")
            else:
                st.info("TradingView webhook not configured (secrets.tradingview.webhook_url or env).")


# -----------------------------
# Deep Dive
# -----------------------------
with tab_deep:
    df_ranked = st.session_state.ranked
    if df_ranked is None or df_ranked.empty:
        st.info("Run a screen first.")
    else:
        st.markdown("### 🔍 Deep Dive")
        pick = st.selectbox("Select ticker", df_ranked["ticker"].tolist(), index=0)

        # download single ticker OHLCV for chart + signals
        dd_period = st.selectbox("Deep-dive lookback", ["6mo", "1y", "2y", "5y"], index=2, key="dd_period")
        dd_interval = st.selectbox("Deep-dive interval", ["1d", "1wk"], index=0, key="dd_interval")
        signal_mode = st.selectbox("Signal model", ["Trend (EMA cross)", "Mean Reversion (BB + RSI)", "MACD Momentum"], index=0)
        fee_bps = st.slider("Fee (bps) per side", 0.0, 50.0, 5.0, 1.0)
        slippage_bps = st.slider("Slippage (bps) per side", 0.0, 50.0, 2.0, 1.0)

        load_btn = st.button("📈 Load Chart + Backtest", use_container_width=True)

        if load_btn:
            pmap = download_prices_all([pick], dd_period, dd_interval, chunk_size=1)
            if pick in pmap:
                df = pmap[pick]
            else:
                start = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=365 * 2)
                end = pd.Timestamp.utcnow().tz_localize("UTC")
                df = generate_dummy_ohlcv([pick], start, end, freq="1D").get(pick, pd.DataFrame())

            df = _ensure_utc_index(_cast_ohlcv(df))
            if df.empty or not all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
                st.error("No OHLCV available for deep dive.")
            else:
                df_sig = build_signals(df, signal_mode)
                trades, equity = run_backtest_single(
                    df_sig,
                    BacktestConfig(
                        fee_bps=float(fee_bps),
                        slippage_bps=float(slippage_bps),
                        position_size=1.0,
                        allow_short=False,
                    ),
                )
                k = trade_kpis(trades, equity)

                # KPI row
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric("Trades", k.get("Trades", 0))
                k2.metric("Win rate", f"{k.get('WinRate', np.nan)*100:.1f}%" if pd.notna(k.get("WinRate")) else "NA")
                k3.metric("Profit factor", f"{k.get('ProfitFactor', np.nan):.2f}" if pd.notna(k.get("ProfitFactor")) else "NA")
                k4.metric("CAGR", f"{k.get('CAGR', np.nan)*100:.1f}%" if pd.notna(k.get("CAGR")) else "NA")
                k5.metric("Sharpe", f"{k.get('Sharpe', np.nan):.2f}" if pd.notna(k.get("Sharpe")) else "NA")
                k6.metric("MaxDD", f"{k.get('MaxDD', np.nan):.2f}" if pd.notna(k.get("MaxDD")) else "NA")

                st.markdown("---")
                fig = plot_candles(df_sig, f"{pick} — Candles + Indicators ({signal_mode})", add_indicators=True)
                fig = add_signal_markers(fig, df_sig)
                st.plotly_chart(fig, use_container_width=True)

                # Equity
                if equity is not None and not equity.empty:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity", line=dict(color="#37F6FF", width=2)))
                    dd = equity / equity.cummax() - 1.0
                    fig2.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown", line=dict(color="#ff3b6d", width=1)))
                    fig2.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=35, b=20), title="Strategy Equity + Drawdown (educational)")
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("---")
                st.markdown("#### Trades")
                if trades is None or trades.empty:
                    st.info("No completed trades in this window.")
                else:
                    st.dataframe(trades.sort_values("entry_time", ascending=False), use_container_width=True, hide_index=True)

                # Export
                settings = dict(st.session_state.last_settings)
                settings.update({
                    "deep_ticker": pick,
                    "deep_period": dd_period,
                    "deep_interval": dd_interval,
                    "signal_mode": signal_mode,
                    "fee_bps": fee_bps,
                    "slippage_bps": slippage_bps,
                })
                xbytes = build_excel(st.session_state.ranked, settings, trades=trades)
                st.download_button(
                    "📥 Download Deep Dive Excel (Ranked + Trades)",
                    data=xbytes,
                    file_name=f"Neon_DeepDive_{pick}_{date.today().isoformat()}.xlsx",
                    use_container_width=True,
                )


# -----------------------------
# Portfolio
# -----------------------------
with tab_port:
    st.markdown("### 🧺 Portfolio Builder")
    st.caption("Build a basket from screened names; allocate (Equal / InvVol / Risk Parity) + rebalance backtest.")

    df_ranked = st.session_state.ranked
    if df_ranked is None or df_ranked.empty:
        st.info("Run a screen first — then come back here.")
    else:
        left, right = st.columns([1.05, 1])

        with left:
            st.markdown("#### 1) Choose constituents")
            default_names = df_ranked.head(min(12, len(df_ranked)))["ticker"].tolist()
            basket = st.multiselect("Portfolio tickers", df_ranked["ticker"].tolist(), default=default_names)

            st.markdown("#### 2) Allocation method")
            method = st.selectbox("Weighting method", ["Equal Weight", "Inverse Vol", "Risk Parity"], index=2)
            reb = st.selectbox("Rebalance frequency", ["Monthly", "Weekly", "None"], index=0)
            lookback_days = st.slider("Lookback window (trading days) for weights", 60, 504, 252, 21)
            max_weight = st.slider("Max weight cap per name", 0.05, 1.0, 0.20, 0.01)

            st.markdown("#### 3) Backtest settings")
            bt_period = st.selectbox("Backtest lookback", ["6mo", "1y", "2y", "5y"], index=2)
            bt_interval = st.selectbox("Backtest interval", ["1d", "1wk"], index=0)
            bench_for_bt = st.text_input("Benchmark for comparison (optional)", value=benchmark)

            run_port = st.button("✨ Build + Backtest Portfolio", use_container_width=True)

        with right:
            st.markdown("#### What you’re optimizing for")
            st.write(
                """
- **Equal Weight**: simple and robust; turnover depends on rebalance frequency.
- **Inverse Vol**: reduces single-name risk concentration; often improves drawdowns.
- **Risk Parity**: targets **equal risk contribution** using the covariance matrix.
                """
            )
            st.markdown("---")
            st.markdown("#### Implementation note")
            st.write("Weights are computed on a trailing window ending at each rebalance date (no forward data).")

        if run_port:
            if len(basket) < 2:
                st.error("Pick at least 2 tickers.")
            else:
                with st.spinner("Downloading portfolio prices..."):
                    pmap = download_prices_all(basket, bt_period, bt_interval, chunk_size=min(chunk_size, 80))

                    if not pmap:
                        # fallback demo
                        start = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=365 * 2)
                        end = pd.Timestamp.utcnow().tz_localize("UTC")
                        pmap = generate_dummy_ohlcv(basket, start, end, freq="1D")

                    closes = {}
                    for t, dff in pmap.items():
                        dff = _ensure_utc_index(_cast_ohlcv(dff))
                        s = _to_close(dff)
                        if not s.empty:
                            closes[t] = s
                    prices = pd.DataFrame(closes).dropna(how="all").ffill().dropna()

                if prices.shape[1] < 2 or prices.shape[0] < 60:
                    st.error("Not enough data for this basket. Try longer lookback or different names.")
                else:
                    equity, w_hist, rc_df = backtest_portfolio(
                        prices=prices,
                        method=method,
                        rf_annual=rf_rate,
                        rebalance=reb,
                        lookback_days=lookback_days,
                        max_weight=max_weight,
                    )
                    if equity.empty:
                        st.error("Portfolio backtest failed.")
                    else:
                        # benchmark compare
                        bench_eq = None
                        if bench_for_bt.strip():
                            b = bench_for_bt.strip().upper()
                            bmap = download_prices_all([b], bt_period, bt_interval, chunk_size=1)
                            if b not in bmap:
                                bmap = generate_dummy_ohlcv([b], prices.index.min(), prices.index.max(), freq="1D")
                            if b in bmap:
                                bclose = _to_close(_ensure_utc_index(_cast_ohlcv(bmap[b])))
                                if not bclose.empty:
                                    bclose = bclose.reindex(equity.index).ffill().dropna()
                                    if not bclose.empty:
                                        bench_eq = (bclose / bclose.iloc[0]).rename("Benchmark")

                        st.markdown("---")
                        st.markdown("#### Equity Curve")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Portfolio", line=dict(color="#37F6FF", width=2)))
                        if bench_eq is not None:
                            fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq.values, mode="lines", name=str(bench_for_bt).upper(), line=dict(color="#B14CFF", width=1.6)))
                        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=35, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("#### Drawdown")
                        dd = equity / equity.cummax() - 1.0
                        figd = go.Figure()
                        figd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown", line=dict(color="#ff3b6d", width=2)))
                        figd.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=35, b=20))
                        st.plotly_chart(figd, use_container_width=True)

                        stats = portfolio_stats(equity, rf_rate)
                        s1, s2, s3, s4, s5 = st.columns(5)
                        s1.metric("CAGR", f"{stats.get('CAGR', np.nan)*100:.1f}%" if pd.notna(stats.get("CAGR")) else "NA")
                        s2.metric("Vol", f"{stats.get('Vol', np.nan)*100:.1f}%" if pd.notna(stats.get("Vol")) else "NA")
                        s3.metric("Sharpe", f"{stats.get('Sharpe', np.nan):.2f}" if pd.notna(stats.get("Sharpe")) else "NA")
                        s4.metric("MaxDD", f"{stats.get('MaxDD', np.nan):.2f}" if pd.notna(stats.get("MaxDD")) else "NA")
                        s5.metric("Omega", f"{stats.get('Omega', np.nan):.2f}" if pd.notna(stats.get("Omega")) else "NA")

                        st.markdown("---")
                        st.markdown("#### Correlation Matrix (returns)")
                        rets = prices.pct_change().dropna()
                        corr = rets.corr()
                        try:
                            st.dataframe(corr.style.background_gradient(axis=None), use_container_width=True)
                        except Exception:
                            st.dataframe(corr, use_container_width=True)

                        st.markdown("#### Current Weights (last rebalance)")
                        if not w_hist.empty:
                            last_w = w_hist.iloc[-1].sort_values(ascending=False)
                            port_df = pd.DataFrame({"ticker": last_w.index, "weight": last_w.values})
                            st.dataframe(port_df, use_container_width=True, hide_index=True)

                            st.markdown("#### Risk Contributions (last rebalance)")
                            st.dataframe(rc_df, use_container_width=True, hide_index=True)

                            st.session_state.portfolio_df = port_df.copy()
                            settings = dict(st.session_state.last_settings)
                            settings.update({
                                "portfolio_method": method,
                                "rebalance": reb,
                                "lookback_days": lookback_days,
                                "max_weight": max_weight,
                                "bt_period": bt_period,
                                "bt_interval": bt_interval,
                            })
                            st.session_state.portfolio_excel = build_excel(st.session_state.ranked, settings, portfolio=port_df)

                            p1, p2 = st.columns([1, 1])
                            with p1:
                                st.download_button(
                                    "📥 Download Portfolio Weights (CSV)",
                                    data=port_df.to_csv(index=False).encode("utf-8"),
                                    file_name=f"portfolio_weights_{date.today().isoformat()}.csv",
                                    use_container_width=True,
                                )
                            with p2:
                                st.download_button(
                                    "📥 Download Full Excel (Screen + Portfolio)",
                                    data=st.session_state.portfolio_excel,
                                    file_name=f"Neon_Screen_Portfolio_{date.today().isoformat()}.xlsx",
                                    use_container_width=True,
                                )


# -----------------------------
# Broadcast tab
# -----------------------------
with tab_broadcast:
    st.markdown("### 📣 Broadcast")
    st.caption("Outbound alerts only. No tokens are displayed. Configure via `.streamlit/secrets.toml` or env vars.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Telegram")
        st.write(f"Status: **{'Enabled' if telegram.enabled() else 'Disabled'}**")
        st.write("Secrets: `telegram.bot_token`, `telegram.chat_id` (or `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`).")
        if telegram.enabled():
            demo = st.text_area("Message (HTML enabled)", value="⚡ <b>NEON</b> test message", height=90)
            if st.button("Send test Telegram message"):
                ok = telegram.send_message(demo)
                st.success("✅ Sent!") if ok else st.error("❌ Failed.")
    with c2:
        st.markdown("#### TradingView webhook")
        st.write(f"Status: **{'Enabled' if tv_webhook.enabled() else 'Disabled'}**")
        st.write("Secrets: `tradingview.webhook_url`, optional `tradingview.passphrase`.")
        if tv_webhook.enabled():
            payload = {
                "event": "neon_test",
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "note": "Educational test alert",
            }
            st.code(json.dumps(payload, indent=2))
            if st.button("Post test webhook"):
                ok = tv_webhook.post_alert(payload)
                st.success("✅ Posted!") if ok else st.error("❌ Failed.")


# -----------------------------
# Theme tab
# -----------------------------
with tab_settings:
    st.markdown("### 🧪 Theme")
    st.write(
        """
This app uses **in-app neon CSS**. Optional Streamlit-wide theming:

Create `.streamlit/config.toml`:

```toml
[theme]
base="dark"
primaryColor="#37F6FF"
backgroundColor="#070A12"
secondaryBackgroundColor="#0B1020"
textColor="#DDE7FF"
font="sans serif"
