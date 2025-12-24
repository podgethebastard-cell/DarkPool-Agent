
import io
import json
import math
import time
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import openai
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# PAGE CONFIG + NEON CSS
# =============================================================================
st.set_page_config(page_title="Neon Institutional Analyst", layout="wide", page_icon="⚡")

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

/* app background */
.stApp{
  background: radial-gradient(1200px 700px at 20% 0%, rgba(177,76,255,0.14), transparent 60%),
              radial-gradient(900px 600px at 95% 12%, rgba(55,246,255,0.16), transparent 55%),
              linear-gradient(180deg, var(--bg), #050610 80%);
  color: var(--text);
}

/* sidebar */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(11,16,32,0.95), rgba(8,10,18,0.95));
  border-right: 1px solid var(--border);
}

.block-container{ padding-top: 1.1rem; }

/* cards / blocks */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background: rgba(11,16,32,0.78);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: var(--shadow);
}

/* buttons neon */
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

/* inputs */
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

/* dataframe */
[data-testid="stDataFrame"]{
  border: 1px solid rgba(55,246,255,0.20);
  border-radius: 16px;
  box-shadow: var(--shadow);
}

/* separators */
hr{
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(55,246,255,0.35), transparent);
}
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

st.title("⚡ Neon Institutional Analyst")
st.markdown(
    """
**Institutional-grade screening** combining **Quant Scanning**, **Deep Fundamentals**, **AI Memos**, and **Portfolio Construction**.
"""
)

# =============================================================================
# OPENAI CLIENT
# =============================================================================
def make_openai_client(api_key: str):
    return openai.OpenAI(api_key=api_key)

# =============================================================================
# TELEGRAM
# =============================================================================
def escape_html(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def tg_send_message(token: str, chat_id: str, text_html: str) -> bool:
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text_html, "parse_mode": "HTML"},
            timeout=25,
        )
        return True
    except Exception as e:
        st.error(f"Telegram sendMessage error: {e}")
        return False

def tg_send_file(token: str, chat_id: str, file_bytes: bytes, filename: str) -> bool:
    try:
        bio = io.BytesIO(file_bytes)
        bio.seek(0)
        requests.post(
            f"https://api.telegram.org/bot{token}/sendDocument",
            data={"chat_id": chat_id, "caption": "📎 Report attached"},
            files={"document": (filename, bio, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            timeout=60,
        )
        return True
    except Exception as e:
        st.error(f"Telegram sendDocument error: {e}")
        return False

# =============================================================================
# UNIVERSE SOURCES (ETF holdings + index constituents)
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
    for i, line in enumerate(text[:50]):
        if "Ticker" in line and "Name" in line:
            header_idx = i
            break
    df = pd.read_csv(io.StringIO("\n".join(text[header_idx:])))
    return df

@st.cache_data(ttl=6 * 3600)
def fetch_ishares_holdings_tickers(preset_name: str, max_names: int = 2500) -> list[str]:
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
def fetch_ftse100_tickers() -> list[str]:
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
def fetch_eurostoxx50_tickers() -> list[str]:
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
def fetch_vaneck_gdxj_holdings(max_names: int = 300) -> list[str]:
    fallback = [
        "AEM", "NEM", "GOLD", "WPM", "FNV", "RGLD",
        "AU", "KGC", "HL", "AG", "SSRM", "IAG",
        "HMY",
    ]
    try:
        r = requests.get(VANECK_GDXJ_HOLDINGS, headers=UA_HEADERS, timeout=45)
        text = r.content.decode("utf-8", errors="ignore").splitlines()
        header_idx = None
        for i, line in enumerate(text[:80]):
            if "Ticker" in line and "Holding" in line:
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
                suf = {
                    "LN": ".L",
                    "CN": ".TO",
                    "AU": ".AX",
                    "HK": ".HK",
                }.get(market, "")
                tickers.append(base + suf)
            else:
                tickers.append(t)
            if len(tickers) >= max_names:
                break
        return sorted(list(dict.fromkeys(tickers)))
    except Exception:
        return sorted(list(dict.fromkeys(fallback)))


# =============================================================================
# PRICE DOWNLOAD (BATCH + CHUNKED)
# =============================================================================
@st.cache_data(ttl=2 * 3600)
def download_prices_chunk(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
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

def download_prices_all(tickers: list[str], period: str, interval: str, chunk_size: int = 80) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        df = download_prices_chunk(chunk, period, interval)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            for t in chunk:
                if t in df.columns.get_level_values(0):
                    sub = df[t].dropna(how="all")
                    if not sub.empty:
                        out[t] = sub
        else:
            t = chunk[0]
            out[t] = df.dropna(how="all")
    return out

def _to_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Adj Close" in df.columns:
        return df["Adj Close"].dropna()
    if "Close" in df.columns:
        return df["Close"].dropna()
    return pd.Series(dtype=float)


# =============================================================================
# QUANT MATH
# =============================================================================
def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ag = up.ewm(alpha=1/period, adjust=False).mean()
    al = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def annualized_vol(ret: pd.Series) -> float:
    if len(ret) < 30:
        return np.nan
    return float(ret.std() * math.sqrt(252))

def annualized_return(close: pd.Series) -> float:
    if len(close) < 2:
        return np.nan
    days = (close.index[-1] - close.index[0]).days
    if days <= 0:
        return np.nan
    total = close.iloc[-1] / close.iloc[0]
    return float(total ** (365.25 / days) - 1)

def max_drawdown(close: pd.Series) -> float:
    peak = close.cummax()
    dd = (close / peak) - 1.0
    return float(dd.min())

def ulcer_index(close: pd.Series) -> float:
    peak = close.cummax()
    dd = (close / peak - 1.0) * 100.0
    return float(np.sqrt(np.mean(dd.values ** 2))) if len(dd) else np.nan

def omega_ratio(ret: pd.Series, threshold: float = 0.0) -> float:
    if len(ret) < 60:
        return np.nan
    gains = (ret - threshold).clip(lower=0).sum()
    losses = (threshold - ret).clip(lower=0).sum()
    if losses == 0:
        return np.nan
    return float(gains / losses)

def historical_var_cvar(ret: pd.Series, level: float = 0.05) -> tuple[float, float]:
    if len(ret) < 60:
        return (np.nan, np.nan)
    q = ret.quantile(level)
    cvar = ret[ret <= q].mean()
    return (float(q), float(cvar))

def sharpe_ratio(ret: pd.Series, rf_annual: float) -> float:
    if len(ret) < 60:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    ex = ret - rf_daily
    sd = ex.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(ex.mean() / sd * math.sqrt(252))

def rolling_sharpe(ret: pd.Series, rf_annual: float, window: int = 63) -> float:
    if len(ret) < window + 5:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    ex = ret - rf_daily
    w = ex.iloc[-window:]
    sd = w.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(w.mean() / sd * math.sqrt(252))

def sortino_ratio(ret: pd.Series, rf_annual: float) -> float:
    if len(ret) < 60:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    ex = ret - rf_daily
    dn = ex[ex < 0]
    sd = dn.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(ex.mean() / sd * math.sqrt(252))

def calmar_ratio(cagr: float, mdd: float) -> float:
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return np.nan
    return float(cagr / abs(mdd))

def hurst_exponent(close: pd.Series) -> float:
    if len(close) < 220:
        return np.nan
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
        return np.nan
    slope, _ = np.polyfit(np.log(valid), np.log(tau), 1)
    return float(slope * 2.0)

def alpha_beta(asset_ret: pd.Series, bench_ret: pd.Series, rf_annual: float) -> tuple[float, float]:
    df = pd.DataFrame({"a": asset_ret, "b": bench_ret}).dropna()
    if len(df) < 90:
        return (np.nan, np.nan)
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    y = df["a"] - rf_daily
    x = df["b"] - rf_daily
    xv = np.var(x)
    if xv == 0 or np.isnan(xv):
        return (np.nan, np.nan)
    beta = float(np.cov(x, y)[0, 1] / xv)
    alpha_d = float(y.mean() - beta * x.mean())
    alpha_a = float((1 + alpha_d) ** 252 - 1)
    return (alpha_a, beta)

def breakout_strength(close: pd.Series, lookback: int = 126) -> float:
    if len(close) < lookback + 5:
        return np.nan
    win = close.iloc[-lookback:]
    lo, hi = float(win.min()), float(win.max())
    if hi == lo:
        return np.nan
    return float((close.iloc[-1] - lo) / (hi - lo))

def zscore_20(close: pd.Series) -> float:
    if len(close) < 25:
        return np.nan
    w = close.iloc[-20:]
    mu = w.mean()
    sd = w.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float((close.iloc[-1] - mu) / sd)

def expectancy_signal(close: pd.Series) -> float:
    if len(close) < 260:
        return np.nan
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
        return np.nan
    mu, sd = float(sample.mean()), float(sample.std())
    if sd == 0 or np.isnan(sd):
        return np.nan
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
# FUNDAMENTALS (Deep Dive)
# =============================================================================
@st.cache_data(ttl=6 * 3600)
def fetch_deep_fundamentals(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        market_cap = float(info.get("marketCap") or np.nan)
        
        # Safe raw statement fetch
        try:
            fin = t.financials
            cash = t.cashflow
            has_stmts = not fin.empty
        except:
            fin, cash = pd.DataFrame(), pd.DataFrame()
            has_stmts = False

        # CAGR Calculation
        sales_cagr_3y = info.get('revenueGrowth', np.nan)
        if has_stmts and "Total Revenue" in fin.index:
            try:
                cols = sorted(fin.columns)
                if len(cols) >= 4:
                    rev_now = fin[cols[-1]].loc["Total Revenue"]
                    rev_3yr = fin[cols[-4]].loc["Total Revenue"]
                    sales_cagr_3y = (rev_now / rev_3yr)**(1/3) - 1
                elif len(cols) >= 2:
                    rev_now = fin[cols[-1]].loc["Total Revenue"]
                    rev_prev = fin[cols[0]].loc["Total Revenue"]
                    sales_cagr_3y = (rev_now / rev_prev) - 1
            except:
                pass

        # FCF Calculation
        p_fcf = np.nan
        fcf_share = 0.0
        try:
            fcf = info.get('freeCashflow', None)
            if fcf is None and has_stmts and "Free Cash Flow" in cash.index:
                 fcf = cash.iloc[0]["Free Cash Flow"] 
            
            if fcf and fcf > 0:
                p_fcf = market_cap / fcf
                fcf_share = fcf / info.get('sharesOutstanding', 1)
        except:
            pass

        return {
            "market_cap": market_cap,
            "forward_pe": float(info.get("forwardPE") or np.nan),
            "price_to_sales": float(info.get("priceToSalesTrailing12Months") or np.nan),
            "price_to_book": float(info.get("priceToBook") or np.nan),
            "peg_ratio": float(info.get("pegRatio") or np.nan),
            "dividend_yield": float(info.get("dividendYield") or 0.0),
            "profit_margin": float(info.get("profitMargins") or np.nan),
            "eps_growth": float(info.get("earningsGrowth") or np.nan),
            "revenue_growth": float(sales_cagr_3y),
            "debt_to_equity": float(info.get("debtToEquity") or np.nan),
            "p_fcf": float(p_fcf),
            "fcf_share": float(fcf_share),
            "insider_percent": float(info.get('heldPercentInsiders', 0)),
            "industry": info.get("industry", "Unknown"),
            "sector": info.get("sector", "Unknown"),
            "name": info.get("longName", ticker),
            "country": info.get("country", "Unknown"),
        }
    except Exception:
        return {
            "market_cap": np.nan, "forward_pe": np.nan, "price_to_sales": np.nan, "price_to_book": np.nan,
            "peg_ratio": np.nan, "dividend_yield": 0.0, "profit_margin": np.nan, "eps_growth": np.nan,
            "revenue_growth": np.nan, "debt_to_equity": np.nan, "p_fcf": np.nan, "fcf_share": 0.0, "insider_percent": 0.0,
            "industry": "Unknown", "sector": "Unknown", "name": ticker, "country": "Unknown"
        }


# =============================================================================
# REGIME-ADAPTIVE SCORE (customizable weights)
# =============================================================================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def safe(x):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except Exception:
        return None

def score_row(row: dict, weights: dict, regime: str) -> float:
    sharpe = safe(row.get("Sharpe"))
    rsharpe = safe(row.get("RollSharpe"))
    sortino = safe(row.get("Sortino"))
    calmar = safe(row.get("Calmar"))
    omega = safe(row.get("Omega"))
    ulcer = safe(row.get("Ulcer"))
    mdd = safe(row.get("MaxDD"))
    vol = safe(row.get("Vol"))
    alpha = safe(row.get("Alpha"))
    mom = safe(row.get("CAGR"))
    brk = safe(row.get("Breakout"))
    hurst = safe(row.get("Hurst"))
    rsi14 = safe(row.get("RSI14"))
    z20 = safe(row.get("Z20"))
    rsi2 = safe(row.get("RSI2"))
    exp = safe(row.get("Expectancy"))

    pe = safe(row.get("forward_pe"))
    pb = safe(row.get("price_to_book"))
    ps = safe(row.get("price_to_sales"))
    pfcf = safe(row.get("p_fcf"))
    margin = safe(row.get("profit_margin"))
    de = safe(row.get("debt_to_equity"))
    revg = safe(row.get("revenue_growth"))
    epsg = safe(row.get("eps_growth"))

    def mean_or0(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else 0.0

    risk = mean_or0([
        clamp01((sharpe + 0.5) / 2.5) if sharpe is not None else None,
        clamp01((rsharpe + 0.5) / 2.5) if rsharpe is not None else None,
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

    value = mean_or0([
        clamp01((45 - pe) / 45) if pe is not None else None,
        clamp01((10 - pb) / 10) if pb is not None else None,
        clamp01((15 - ps) / 15) if ps is not None else None,
        clamp01((35 - pfcf) / 35) if pfcf is not None else None,
    ])
    quality = mean_or0([
        clamp01((margin + 0.05) / 0.40) if margin is not None else None,
        clamp01((250 - de) / 250) if de is not None else None,
        clamp01((revg + 0.10) / 0.60) if revg is not None else None,
        clamp01((epsg + 0.10) / 0.80) if epsg is not None else None,
    ])

    rw = dict(weights)
    if regime == "Trend":
        rw["trend"] *= 1.20
        rw["meanrev"] *= 0.80
        rw["edge"] *= 1.15
    elif regime == "MeanReversion":
        rw["trend"] *= 0.85
        rw["meanrev"] *= 1.25
        rw["edge"] *= 1.15

    denom = sum(rw.values()) if sum(rw.values()) > 0 else 1.0
    score = (
        rw["risk"] * risk +
        rw["trend"] * trend +
        rw["meanrev"] * meanrev +
        rw["edge"] * edge +
        rw["value"] * value +
        rw["quality"] * quality
    ) / denom

    return round(score * 100, 1)


# =============================================================================
# MAIN SCAN
# =============================================================================
def compute_metrics_for_ticker(t: str, df: pd.DataFrame, rf_annual: float, bench_ret: pd.Series | None) -> dict | None:
    close = _to_close(df)
    if close.empty or len(close) < 120:
        return None

    ret = close.pct_change().dropna()
    vol = annualized_vol(ret)
    cagr = annualized_return(close)
    mdd = max_drawdown(close)
    ui = ulcer_index(close)
    rsi14 = float(rsi_wilder(close, 14).iloc[-1])
    rsi2 = float(rsi_wilder(close, 2).iloc[-1]) if len(close) > 10 else np.nan
    z20 = zscore_20(close)
    brk = breakout_strength(close, 126)
    h = hurst_exponent(close)
    exp = expectancy_signal(close)
    var95, cvar95 = historical_var_cvar(ret, 0.05)
    sharpe = sharpe_ratio(ret, rf_annual)
    rsh = rolling_sharpe(ret, rf_annual, 63)
    sortino = sortino_ratio(ret, rf_annual)
    calmar = calmar_ratio(cagr, mdd)
    omega = omega_ratio(ret, 0.0)
    alpha, beta = (np.nan, np.nan)
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
        "RollSharpe": rsh,
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
    tickers: list[str],
    period: str,
    interval: str,
    rf_annual: float,
    benchmark: str,
    max_tickers: int,
    chunk_size: int,
    workers: int,
    weights: dict,
    fundamentals_mode: str,
    fundamentals_top_k: int,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    tickers = tickers[:max_tickers]

    bench_ret = None
    if benchmark:
        bench_map = download_prices_all([benchmark], period, interval, chunk_size=1)
        if benchmark in bench_map:
            bclose = _to_close(bench_map[benchmark])
            if len(bclose) > 120:
                bench_ret = bclose.pct_change().dropna()

    price_map = download_prices_all(tickers, period, interval, chunk_size=chunk_size)
    if not price_map:
        return pd.DataFrame()

    rows = []
    prog = st.progress(0)
    status = st.empty()

    items = list(price_map.items())
    total = len(items)
    done = 0

    def work(item):
        t, df = item
        return compute_metrics_for_ticker(t, df, rf_annual, bench_ret)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, it): it[0] for it in items}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
                if r:
                    rows.append(r)
            except Exception:
                pass
            done += 1
            prog.progress(done / max(total, 1))
            status.text(f"Quant scan: {t} ({done}/{total})")

    prog.empty()
    status.empty()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Score"] = df.apply(lambda r: score_row(r.to_dict(), weights, r.get("Regime", "Mixed")), axis=1)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)

    if fundamentals_mode != "Off":
        topK = min(fundamentals_top_k, len(df))
        status = st.empty()
        prog = st.progress(0)
        frows = []
        for i in range(topK):
            t = df.loc[i, "ticker"]
            status.text(f"Fundamentals ({fundamentals_mode}): {t} ({i+1}/{topK})")
            prog.progress((i+1) / max(topK, 1))
            f = fetch_deep_fundamentals(t)
            f["ticker"] = t
            frows.append(f)
        prog.empty()
        status.empty()

        fdf = pd.DataFrame(frows).set_index("ticker")
        df = df.set_index("ticker").join(fdf, how="left").reset_index()

        if "p_fcf" not in df.columns:
            df["p_fcf"] = np.nan

        df["Score"] = df.apply(lambda r: score_row(r.to_dict(), weights, r.get("Regime", "Mixed")), axis=1)
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)

    return df


# =============================================================================
# EXPORT
# =============================================================================
def build_excel(df: pd.DataFrame, settings: dict, portfolio: pd.DataFrame | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, sheet_name="Ranked", index=False)
        pd.DataFrame([settings]).to_excel(w, sheet_name="Settings", index=False)
        if portfolio is not None and not portfolio.empty:
            portfolio.to_excel(w, sheet_name="Portfolio", index=False)
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# AI MEMOS + CHATBOT (asks preferences if not set)
# =============================================================================
def ai_memos(api_key: str, model: str, temperature: float, rows: list[dict]) -> dict:
    if not api_key or not rows:
        return {}
    client = make_openai_client(api_key)
    prompt = {
        "instructions": (
            "Return ONLY valid JSON keyed by ticker. "
            "For each ticker include: verdict (Buy/Hold/Sell), thesis (2 sentences), "
            "technical (1 sentence), risks (1 sentence), catalysts (1 sentence). "
            "Use only provided data; if missing say 'data limited'. No markdown."
        ),
        "rows": rows,
    }
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a senior buy-side PM. Return only JSON."},
                {"role": "user", "content": json.dumps(prompt)[:15000]},
            ],
            temperature=temperature,
        )
        text = resp.choices[0].message.content.strip()
        return json.loads(text) if text else {}
    except Exception as e:
        st.warning(f"AI memo failed: {e}")
        return {}

def analyst_chat(api_key: str, model: str, temperature: float, context: dict):
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "analyst_onboarding_done" not in st.session_state:
        st.session_state.analyst_onboarding_done = False

    if not st.session_state.analyst_onboarding_done:
        st.session_state.chat.append({
            "role": "assistant",
            "content": (
                "👋 Quick setup—what do you want to analyze?\n\n"
                "**Market:** `global`, `us total`, `russell`, `uk`, `euro`, `jnr miners`\n"
                "**Style:** `trend`, `mean reversion`, `balanced`, `defensive`, `value/quality`\n\n"
                "Reply like: **`russell + trend`** (or tell me your objective)."
            )
        })
        st.session_state.analyst_onboarding_done = True

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Ask the AI analyst… (it can also configure your screen/portfolio)")
    if not user_msg:
        return

    st.session_state.chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    msg = user_msg.lower()
    if "pref_market" not in st.session_state:
        st.session_state.pref_market = None
    if "pref_style" not in st.session_state:
        st.session_state.pref_style = None

    if any(k in msg for k in ["global", "acwi", "whole market", "world"]):
        st.session_state.pref_market = "ACWI (Global All-Country)"
    elif any(k in msg for k in ["us total", "itot", "total market"]):
        st.session_state.pref_market = "ITOT (US Total Market)"
    elif any(k in msg for k in ["russell", "iwm", "r2k", "2000"]):
        st.session_state.pref_market = "IWM (Russell 2000)"
    elif any(k in msg for k in ["uk", "ftse", "britain", "london"]):
        st.session_state.pref_market = "FTSE 100 (Wikipedia)"
    elif any(k in msg for k in ["euro", "stoxx", "europe"]):
        st.session_state.pref_market = "Euro Stoxx 50 (Wikipedia)"
    elif any(k in msg for k in ["jnr", "junior", "miners", "gdxj"]):
        st.session_state.pref_market = "GDXJ (Junior Miners holdings)"

    if any(k in msg for k in ["trend", "momentum", "breakout"]):
        st.session_state.pref_style = "Trend Hunter"
    elif any(k in msg for k in ["mean reversion", "reversion", "oversold", "snapback"]):
        st.session_state.pref_style = "Mean Reversion"
    elif any(k in msg for k in ["defensive", "low risk", "low vol"]):
        st.session_state.pref_style = "Defensive"
    elif any(k in msg for k in ["value", "quality"]):
        st.session_state.pref_style = "Value & Quality"
    elif "balanced" in msg:
        st.session_state.pref_style = "Balanced"

    config_note = []
    if st.session_state.pref_market:
        config_note.append(f"Market set to **{st.session_state.pref_market}**")
    if st.session_state.pref_style:
        config_note.append(f"Style set to **{st.session_state.pref_style}**")

    if config_note:
        with st.chat_message("assistant"):
            st.markdown("✅ " + " · ".join(config_note) + "\n\nRun the screen when you’re ready.")
        st.session_state.chat.append({"role": "assistant", "content": "✅ " + " · ".join(config_note)})
        return

    if not api_key:
        with st.chat_message("assistant"):
            st.warning("Add your OpenAI API key in the sidebar to use the analyst.")
        return

    client = make_openai_client(api_key)
    system = (
        "You are a senior institutional PM + quant analyst. "
        "Be concise, numerate, skeptical. "
        "Use ONLY context numbers; if missing say data limited. "
        "No personalized investment advice; talk generally and include risks. "
        "When helpful: propose next steps and what to validate."
    )
    ctx = json.dumps(context)[:16000]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Context JSON:\n{ctx}\n\nUser:\n{user_msg}"},
                    ],
                    temperature=temperature,
                )
                ans = resp.choices[0].message.content.strip()
                st.markdown(ans)
                st.session_state.chat.append({"role": "assistant", "content": ans})
            except Exception as e:
                st.error(f"Chat error: {e}")


# =============================================================================
# PORTFOLIO BUILDER (Equal / InvVol / RiskParity + Backtest + Heatmap)
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
    w = inv.values
    return _normalize_weights(w)

def weights_risk_parity(cov: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """
    Simple multiplicative update risk parity:
    want equal risk contributions rc_i = w_i*(Σw)_i / (w'Σw)
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return np.ones(n) / n
        mrc = cov @ w
        rc = w * mrc / port_var
        target = np.ones(n) / n
        # multiplicative update
        w_new = w * (target / (rc + 1e-12))
        w_new = _normalize_weights(w_new)
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    return w

def compute_portfolio_equity(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    pr = returns.fillna(0.0).values @ weights
    eq = pd.Series(pr, index=returns.index, name="portfolio_return")
    equity = (1.0 + eq).cumprod()
    equity.name = "equity"
    return equity

def portfolio_stats(equity: pd.Series, rf_annual: float) -> dict:
    if equity is None or equity.empty:
        return {}
    ret = equity.pct_change().dropna()
    cagr = annualized_return(equity)
    vol = annualized_vol(ret)
    mdd = max_drawdown(equity)
    ui = ulcer_index(equity)
    sharpe = sharpe_ratio(ret, rf_annual)
    sortino = sortino_ratio(ret, rf_annual)
    calmar = calmar_ratio(cagr, mdd)
    omega = omega_ratio(ret, 0.0)
    var95, cvar95 = historical_var_cvar(ret, 0.05)
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Omega": omega,
        "MaxDD": mdd,
        "Ulcer": ui,
        "VaR_95": var95,
        "CVaR_95": cvar95,
    }

def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "None":
        return pd.DatetimeIndex([index[0]])
    # monthly / weekly using pandas period grouping
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
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    prices: wide close prices (index date, columns tickers)
    Returns: equity curve, weights over time, risk contributions last rebalance.
    """
    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()
    if prices.empty or prices.shape[1] < 2:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    rets = prices.pct_change().dropna()
    idx = rets.index
    rdates = rebalance_dates(idx, rebalance)

    w_hist = []
    date_hist = []

    port_ret = pd.Series(0.0, index=idx)

    last_rc = None

    for i, d in enumerate(rdates):
        # determine window for estimation
        start = max(0, idx.get_loc(d) - lookback_days)
        end = idx.get_loc(d)
        window = rets.iloc[start:end].copy()
        if len(window) < min(40, lookback_days // 3):
            # not enough data: fallback equal weights
            w = weights_equal(rets.shape[1])
        else:
            if method == "Equal Weight":
                w = weights_equal(rets.shape[1])
            elif method == "Inverse Vol":
                w = weights_inverse_vol(window)
            else:  # Risk Parity
                cov = window.cov().values
                w = weights_risk_parity(cov)

        # weight cap + renormalize
        w = np.minimum(w, max_weight)
        w = _normalize_weights(w)

        # apply weights from d to next rebalance date (or end)
        d_loc = idx.get_loc(d)
        next_loc = idx.get_loc(rdates[i+1]) if i + 1 < len(rdates) else len(idx)
        seg = rets.iloc[d_loc:next_loc]
        port_ret.iloc[d_loc:next_loc] = seg.fillna(0.0).values @ w

        w_hist.append(w)
        date_hist.append(d)

        # last risk contributions snapshot
        if len(window) >= 40:
            cov = window.cov().values
            pv = float(w @ cov @ w)
            mrc = cov @ w
            rc = (w * mrc) / (pv + 1e-12)
            last_rc = rc

    equity = (1.0 + port_ret).cumprod()
    w_df = pd.DataFrame(w_hist, index=pd.DatetimeIndex(date_hist), columns=rets.columns)

    rc_df = pd.DataFrame({
        "ticker": rets.columns,
        "risk_contribution": last_rc if last_rc is not None else np.nan
    }).sort_values("risk_contribution", ascending=False)

    return equity, w_df, rc_df


# =============================================================================
# SIDEBAR: FULL CUSTOMIZATION
# =============================================================================
st.sidebar.header("⚙️ Control Room")

api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
if api_key:
    st.sidebar.success("OpenAI key loaded")
else:
    api_key = st.sidebar.text_input("OpenAI API key", type="password")

model_name = st.sidebar.text_input("Model", value="gpt-4o-mini")
temperature = st.sidebar.slider("AI temperature", 0.0, 1.0, 0.35, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Telegram")
tele_token = st.secrets.get("TELEGRAM_TOKEN", "") if hasattr(st, "secrets") else ""
tele_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "") if hasattr(st, "secrets") else ""
if not (tele_token and tele_chat_id):
    tele_token = st.sidebar.text_input("Bot token", type="password")
    tele_chat_id = st.sidebar.text_input("Chat ID")
use_telegram = bool(tele_token and tele_chat_id)

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
        "EWU holdings (UK Large+Mid)",
        "EZU holdings (Eurozone Large+Mid)",
        "IEUR holdings (Europe Broad)",
        "GDXJ (Junior Miners holdings)",
        "Custom",
    ],
    index=0,
)

custom_tickers = st.sidebar.text_area("Custom tickers (comma/space separated)", value="")
max_universe = st.sidebar.slider("Max tickers to scan", 50, 2500, 600, 50)

st.sidebar.markdown("---")
st.sidebar.subheader("Time & Benchmark")
period = st.sidebar.selectbox("Lookback", ["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk"], index=0)
benchmark = st.sidebar.text_input("Benchmark for alpha/beta", value="SPY")
rf_rate = st.sidebar.slider("Risk-free (annual %)", 0.0, 10.0, 2.0, 0.25) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("Performance / Optimization")
chunk_size = st.sidebar.slider("Download chunk size", 20, 200, 80, 10)
workers = st.sidebar.slider("CPU workers", 2, 24, 10, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Depth")
fundamentals_mode = st.sidebar.selectbox("Fundamentals", ["Off", "Light"], index=1)
fundamentals_top_k = st.sidebar.slider("Fundamentals top-K", 10, 250, 60, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("Factor Studio (weights)")
preset_style = st.sidebar.selectbox(
    "Style preset",
    ["Balanced", "Trend Hunter", "Mean Reversion", "Defensive", "Value & Quality"],
    index=0
)

STYLE_PRESETS = {
    "Balanced":       {"risk": 0.28, "trend": 0.20, "meanrev": 0.12, "edge": 0.18, "value": 0.12, "quality": 0.10},
    "Trend Hunter":   {"risk": 0.24, "trend": 0.32, "meanrev": 0.06, "edge": 0.22, "value": 0.08, "quality": 0.08},
    "Mean Reversion": {"risk": 0.26, "trend": 0.10, "meanrev": 0.28, "edge": 0.22, "value": 0.08, "quality": 0.06},
    "Defensive":      {"risk": 0.40, "trend": 0.10, "meanrev": 0.10, "edge": 0.20, "value": 0.10, "quality": 0.10},
    "Value & Quality":{"risk": 0.26, "trend": 0.12, "meanrev": 0.10, "edge": 0.16, "value": 0.20, "quality": 0.16},
}
w0 = STYLE_PRESETS[preset_style]
with st.sidebar.expander("Fine-tune weights", expanded=False):
    w_risk = st.slider("Risk-adjusted", 0.0, 1.0, float(w0["risk"]), 0.01)
    w_trend = st.slider("Trend / Momentum", 0.0, 1.0, float(w0["trend"]), 0.01)
    w_mr = st.slider("Mean reversion", 0.0, 1.0, float(w0["meanrev"]), 0.01)
    w_edge = st.slider("Expectancy edge", 0.0, 1.0, float(w0["edge"]), 0.01)
    w_value = st.slider("Value", 0.0, 1.0, float(w0["value"]), 0.01)
    w_quality = st.slider("Quality / Growth", 0.0, 1.0, float(w0["quality"]), 0.01)

weights = {"risk": w_risk, "trend": w_trend, "meanrev": w_mr, "edge": w_edge, "value": w_value, "quality": w_quality}

st.sidebar.markdown("---")
st.sidebar.subheader("AI output")
top_n = st.sidebar.slider("Top N to display / memo", 5, 50, 15, 1)
make_memos = st.sidebar.checkbox("Generate AI memos for Top N", value=True)


# =============================================================================
# BUILD UNIVERSE
# =============================================================================
def parse_tickers(s: str) -> list[str]:
    if not s.strip():
        return []
    raw = s.replace("\n", " ").replace(";", " ").replace(",", " ").split()
    return [t.strip().upper() for t in raw if t.strip()]

def get_universe(preset_name: str) -> list[str]:
    if preset_name == "Custom":
        return sorted(list(dict.fromkeys(parse_tickers(custom_tickers))))
    if preset_name == "FTSE 100 (Wikipedia)":
        return fetch_ftse100_tickers()
    if preset_name == "Euro Stoxx 50 (Wikipedia)":
        return fetch_eurostoxx50_tickers()
    if preset_name == "GDXJ (Junior Miners holdings)":
        return fetch_vaneck_gdxj_holdings(max_names=max_universe)
    if preset_name in ("EWU holdings (UK Large+Mid)", "EZU holdings (Eurozone Large+Mid)", "IEUR holdings (Europe Broad)"):
        key = preset_name.split(" holdings")[0].strip()
        for k in ISHARES_PRODUCTS:
            if k.startswith(key):
                return fetch_ishares_holdings_tickers(k, max_names=max_universe)
        return []
    if preset_name in ISHARES_PRODUCTS:
        return fetch_ishares_holdings_tickers(preset_name, max_names=max_universe)
    return []

universe = get_universe(preset)
append_custom = st.sidebar.checkbox("Add custom tickers on top of preset", value=False)
if append_custom and custom_tickers.strip():
    universe = sorted(list(dict.fromkeys(universe + parse_tickers(custom_tickers))))
universe = universe[:max_universe]


# =============================================================================
# STATE
# =============================================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "excel_bytes" not in st.session_state:
    st.session_state.excel_bytes = None
if "memos" not in st.session_state:
    st.session_state.memos = {}
if "last_settings" not in st.session_state:
    st.session_state.last_settings = {}
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame()
if "portfolio_excel" not in st.session_state:
    st.session_state.portfolio_excel = None


# =============================================================================
# MAIN UI: TABS
# =============================================================================
tab_screen, tab_deep, tab_port, tab_chat, tab_settings = st.tabs(
    ["📡 Screen", "🔍 Deep Dive", "🧺 Portfolio", "🧠 AI Analyst Chat", "🧪 Settings & Theme"]
)

with tab_screen:
    a, b, c, d = st.columns([1.4, 1, 1, 1])
    with a:
        st.markdown("### Market Scan")
        st.caption("Fast quant scan first. Fundamentals optional for top-K only.")
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
            if len(universe) > 1200 and fundamentals_mode != "Off":
                st.warning("Huge universe detected. Consider Fundamentals=Off (or reduce top-K) for speed.")

            df = run_scan(
                tickers=universe,
                period=period,
                interval=interval,
                rf_annual=rf_rate,
                benchmark=benchmark,
                max_tickers=max_universe,
                chunk_size=chunk_size,
                workers=workers,
                weights=weights,
                fundamentals_mode=fundamentals_mode,
                fundamentals_top_k=fundamentals_top_k,
            )

            if df.empty:
                st.warning("No results. Try a different preset, increase lookback, or check tickers.")
            else:
                df = df.head(top_n).copy()

                trend_pct = float((df["Regime"] == "Trend").mean() * 100.0)
                mr_pct = float((df["Regime"] == "MeanReversion").mean() * 100.0)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Top regime: Trend", f"{trend_pct:.0f}%")
                m2.metric("Top regime: MeanRev", f"{mr_pct:.0f}%")
                m3.metric("Median Sharpe", f"{np.nanmedian(df['Sharpe']):.2f}")
                m4.metric("Median MaxDD", f"{np.nanmedian(df['MaxDD']):.2f}")

                memos = {}
                if make_memos and api_key:
                    with st.spinner("Writing AI memos..."):
                        payload = df.to_dict(orient="records")
                        memos = ai_memos(api_key, model_name, temperature, payload)
                    df["AI_Verdict"] = df["ticker"].map(lambda t: memos.get(t, {}).get("verdict", ""))
                    df["AI_Thesis"] = df["ticker"].map(lambda t: memos.get(t, {}).get("thesis", ""))
                    df["AI_Risks"] = df["ticker"].map(lambda t: memos.get(t, {}).get("risks", ""))
                    df["AI_Catalysts"] = df["ticker"].map(lambda t: memos.get(t, {}).get("catalysts", ""))

                settings = {
                    "date": date.today().isoformat(),
                    "preset": preset,
                    "period": period,
                    "interval": interval,
                    "benchmark": benchmark,
                    "rf_rate": rf_rate,
                    "max_universe": max_universe,
                    "chunk_size": chunk_size,
                    "workers": workers,
                    "fundamentals": fundamentals_mode,
                    "fundamentals_top_k": fundamentals_top_k,
                    "weights": weights,
                    "style": preset_style,
                    "top_n": top_n,
                    "model": model_name,
                }

                st.session_state.df = df
                st.session_state.memos = memos
                st.session_state.last_settings = settings
                st.session_state.excel_bytes = build_excel(df, settings, portfolio=st.session_state.portfolio_df)

    df = st.session_state.df
    if df is not None and not df.empty:
        st.markdown("---")
        st.markdown("### 🏆 Ranked Picks")

        show_cols = [
            "ticker", "Score", "Price", "Regime",
            "Sharpe", "RollSharpe", "Sortino", "Calmar", "Omega",
            "MaxDD", "Ulcer", "Vol",
            "Alpha", "Beta",
            "RSI14", "RSI2", "Z20", "Breakout", "Hurst", "Expectancy",
            "sector", "industry", "country",
            "forward_pe", "price_to_book", "price_to_sales", "profit_margin", "dividend_yield",
            "AI_Verdict"
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            fname = f"Neon_Screen_{date.today().isoformat()}.xlsx"
            st.download_button("📥 Download Excel", data=st.session_state.excel_bytes, file_name=fname, use_container_width=True)

        with c2:
            if use_telegram:
                if st.button("📡 Send SHORT Telegram Signal", use_container_width=True):
                    topk = df.head(min(5, len(df)))
                    header = (
                        f"⚡ <b>NEON SIGNAL</b>\n"
                        f"Preset: <code>{escape_html(st.session_state.last_settings.get('preset',''))}</code> · "
                        f"Lookback: <code>{escape_html(period)}</code> · "
                        f"Benchmark: <code>{escape_html(benchmark)}</code>\n"
                        f"Date: <code>{escape_html(date.today().isoformat())}</code>\n\n"
                    )
                    lines = []
                    for i, r in topk.iterrows():
                        verdict = r.get("AI_Verdict", "")
                        lines.append(
                            f"<b>{i+1})</b> <code>{escape_html(r['ticker'])}</code> "
                            f"{escape_html(verdict)} · "
                            f"Score <b>{escape_html(r['Score'])}</b> · "
                            f"Sharpe {escape_html(round(r['Sharpe'],2) if pd.notna(r['Sharpe']) else 'NA')} · "
                            f"MDD {escape_html(round(r['MaxDD'],2) if pd.notna(r['MaxDD']) else 'NA')} · "
                            f"Regime <i>{escape_html(r['Regime'])}</i>"
                        )
                    footer = "\n\n<i>Educational only. Validate data independently.</i>"
                    ok = tg_send_message(tele_token, tele_chat_id, header + "\n".join(lines) + footer)
                    st.success("✅ Sent!") if ok else st.error("❌ Failed.")
            else:
                st.info("Telegram not configured.")

        with c3:
            if use_telegram and st.session_state.excel_bytes:
                if st.button("📎 Send Excel to Telegram", use_container_width=True):
                    fname = f"Neon_Screen_{date.today().isoformat()}.xlsx"
                    ok = tg_send_file(tele_token, tele_chat_id, st.session_state.excel_bytes, fname)
                    st.success("✅ Sent!") if ok else st.error("❌ Failed.")
            else:
                st.info("Telegram not configured / no report yet.")


with tab_deep:
    df = st.session_state.df
    if df is None or df.empty:
        st.info("Run a screen first.")
    else:
        st.markdown("### 🔍 Deep Dive")
        pick = st.selectbox("Select ticker", df["ticker"].tolist(), index=0)
        row = df[df["ticker"] == pick].iloc[0].to_dict()

        a, b, c, d = st.columns(4)
        a.metric("Score", row.get("Score", "—"))
        b.metric("Regime", row.get("Regime", "—"))
        c.metric("Sharpe", f"{row.get('Sharpe', np.nan):.2f}" if pd.notna(row.get("Sharpe")) else "NA")
        d.metric("MaxDD", f"{row.get('MaxDD', np.nan):.2f}" if pd.notna(row.get("MaxDD")) else "NA")

        st.markdown("---")
        st.markdown("#### Quant Snapshot")
        snap_cols = ["Price","CAGR","Vol","Sharpe","RollSharpe","Sortino","Calmar","Omega","Ulcer","VaR_95","CVaR_95","Alpha","Beta","RSI14","RSI2","Z20","Breakout","Hurst","Expectancy"]
        snap = {k: row.get(k) for k in snap_cols if k in row}
        st.json(snap)

        if row.get("AI_Thesis"):
            st.markdown("#### AI Memo")
            st.write(f"**Verdict:** {row.get('AI_Verdict','')}")
            st.write(row.get("AI_Thesis",""))
            st.write(f"**Risks:** {row.get('AI_Risks','')}")
            st.write(f"**Catalysts:** {row.get('AI_Catalysts','')}")


with tab_port:
    st.markdown("### 🧺 Portfolio Builder")
    st.caption("Build a basket from the screened names, allocate (Equal / InvVol / Risk Parity), backtest with rebalancing, and export.")

    df = st.session_state.df
    if df is None or df.empty:
        st.info("Run a screen first — then come back here.")
    else:
        left, right = st.columns([1.05, 1])

        with left:
            st.markdown("#### 1) Choose constituents")
            default_names = df.head(min(12, len(df)))["ticker"].tolist()
            basket = st.multiselect("Portfolio tickers", df["ticker"].tolist(), default=default_names)

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
- **Equal Weight**: simple, robust, higher turnover sensitivity to rebalance.
- **Inverse Vol**: reduces single-name risk; often improves drawdowns vs equal weight.
- **Risk Parity**: targets **equal risk contribution**, tends to produce smoother equity curves.
                """
            )
            st.markdown("---")
            st.markdown("#### Pro tip")
            st.write("Try **Risk Parity + Monthly rebalance** as a default, then compare to Equal Weight.")

        if run_port:
            if len(basket) < 2:
                st.error("Pick at least 2 tickers.")
            else:
                with st.spinner("Downloading portfolio prices..."):
                    pmap = download_prices_all(basket, bt_period, bt_interval, chunk_size=min(chunk_size, 80))
                    if not pmap:
                        st.error("Failed to download prices for basket.")
                    else:
                        # wide prices
                        closes = {}
                        for t, dff in pmap.items():
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
                                    bmap = download_prices_all([bench_for_bt.strip().upper()], bt_period, bt_interval, chunk_size=1)
                                    if bench_for_bt.strip().upper() in bmap:
                                        bclose = _to_close(bmap[bench_for_bt.strip().upper()])
                                        if not bclose.empty:
                                            bclose = bclose.reindex(equity.index).ffill().dropna()
                                            if not bclose.empty:
                                                bench_eq = (bclose / bclose.iloc[0])

                                # charts
                                st.markdown("---")
                                st.markdown("#### Equity Curve")
                                chart_df = pd.DataFrame({"Portfolio": equity})
                                if bench_eq is not None:
                                    chart_df["Benchmark"] = bench_eq.reindex(equity.index).ffill()
                                st.line_chart(chart_df, use_container_width=True)

                                st.markdown("#### Drawdown Curve")
                                dd = equity / equity.cummax() - 1.0
                                st.line_chart(dd.rename("Drawdown"), use_container_width=True)

                                # stats
                                stats = portfolio_stats(equity, rf_rate)
                                s1, s2, s3, s4, s5 = st.columns(5)
                                s1.metric("CAGR", f"{stats.get('CAGR', np.nan)*100:.1f}%" if pd.notna(stats.get("CAGR")) else "NA")
                                s2.metric("Vol", f"{stats.get('Vol', np.nan)*100:.1f}%" if pd.notna(stats.get("Vol")) else "NA")
                                s3.metric("Sharpe", f"{stats.get('Sharpe', np.nan):.2f}" if pd.notna(stats.get("Sharpe")) else "NA")
                                s4.metric("MaxDD", f"{stats.get('MaxDD', np.nan):.2f}" if pd.notna(stats.get("MaxDD")) else "NA")
                                s5.metric("Omega", f"{stats.get('Omega', np.nan):.2f}" if pd.notna(stats.get("Omega")) else "NA")

                                # correlation heatmap as styled dataframe
                                st.markdown("---")
                                st.markdown("#### Correlation Matrix (returns)")
                                rets = prices.pct_change().dropna()
                                corr = rets.corr()

                                # styled gradient
                                try:
                                    st.dataframe(
                                        corr.style.background_gradient(axis=None),
                                        use_container_width=True
                                    )
                                except Exception:
                                    st.dataframe(corr, use_container_width=True)

                                st.markdown("#### Current Weights (last rebalance)")
                                if not w_hist.empty:
                                    last_w = w_hist.iloc[-1].sort_values(ascending=False)
                                    port_df = pd.DataFrame({"ticker": last_w.index, "weight": last_w.values})
                                    st.dataframe(port_df, use_container_width=True, hide_index=True)

                                    st.markdown("#### Risk Contributions (last rebalance)")
                                    st.dataframe(rc_df, use_container_width=True, hide_index=True)

                                    # store for export + telegram
                                    st.session_state.portfolio_df = port_df.copy()
                                    st.session_state.portfolio_excel = build_excel(
                                        df=st.session_state.df,
                                        settings={**st.session_state.last_settings, "portfolio_method": method, "rebalance": reb, "lookback_days": lookback_days},
                                        portfolio=port_df
                                    )

                                    p1, p2, p3 = st.columns([1, 1, 1])
                                    with p1:
                                        st.download_button(
                                            "📥 Download Portfolio Weights (CSV)",
                                            data=port_df.to_csv(index=False).encode("utf-8"),
                                            file_name=f"portfolio_weights_{date.today().isoformat()}.csv",
                                            use_container_width=True,
                                        )
                                    with p2:
                                        if st.session_state.portfolio_excel is not None:
                                            st.download_button(
                                                "📥 Download Full Excel (Screen + Portfolio)",
                                                data=st.session_state.portfolio_excel,
                                                file_name=f"Neon_Screen_Portfolio_{date.today().isoformat()}.xlsx",
                                                use_container_width=True,
                                            )
                                    with p3:
                                        if use_telegram and st.session_state.portfolio_excel is not None:
                                            if st.button("📡 Send Portfolio Summary to Telegram", use_container_width=True):
                                                topw = port_df.sort_values("weight", ascending=False).head(8)
                                                lines = []
                                                for i, r in topw.iterrows():
                                                    lines.append(f"<b>{i+1})</b> <code>{escape_html(r['ticker'])}</code> · {escape_html(round(r['weight']*100,2))}%")

                                                msg = (
                                                    f"🧺 <b>NEON PORTFOLIO</b>\n"
                                                    f"Method: <code>{escape_html(method)}</code> · Rebalance: <code>{escape_html(reb)}</code>\n"
                                                    f"Lookback: <code>{escape_html(lookback_days)}</code> days · Period: <code>{escape_html(bt_period)}</code>\n"
                                                    f"Stats: CAGR <b>{escape_html(round(stats.get('CAGR',0)*100,1))}%</b> · "
                                                    f"Sharpe <b>{escape_html(round(stats.get('Sharpe',0),2))}</b> · "
                                                    f"MaxDD <b>{escape_html(round(stats.get('MaxDD',0),2))}</b>\n\n"
                                                    + "\n".join(lines) +
                                                    "\n\n<i>Educational only. Validate before acting.</i>"
                                                )
                                                ok = tg_send_message(tele_token, tele_chat_id, msg)
                                                st.success("✅ Sent!") if ok else st.error("❌ Failed.")
                                        else:
                                            st.info("Telegram not configured / build portfolio first.")


with tab_chat:
    st.markdown("### 🧠 AI Analyst Chat")
    st.caption("The analyst can configure market/style if you haven’t specified them yet, and can comment on the portfolio too.")

    df = st.session_state.df
    context = {
        "settings": st.session_state.last_settings,
        "top_table": df.head(min(15, len(df))).to_dict(orient="records") if df is not None and not df.empty else [],
        "portfolio": st.session_state.portfolio_df.to_dict(orient="records") if st.session_state.portfolio_df is not None and not st.session_state.portfolio_df.empty else [],
        "notes": "Quant signals are heuristic. No guaranteed edge. Validate liquidity, spreads, and news before acting.",
        "pref_market": getattr(st.session_state, "pref_market", None),
        "pref_style": getattr(st.session_state, "pref_style", None),
    }
    analyst_chat(api_key, model_name, temperature, context)


with tab_settings:
    st.markdown("### 🧪 Settings & Theme")
    st.write(
        """
This app uses **in-app neon CSS**. If you want Streamlit-wide theming too, create:

`.streamlit/config.toml`

```toml
[theme]
base="dark"
primaryColor="#37F6FF"
backgroundColor="#070A12"
secondaryBackgroundColor="#0B1020"
textColor="#DDE7FF"
font="sans serif"
