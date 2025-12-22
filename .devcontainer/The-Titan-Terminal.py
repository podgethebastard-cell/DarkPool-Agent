```python
# ============================================================
# 👁️ DarkPool Titan Terminal (Fully Optimized, No Omissions)
# - Performance: batched downloads, aggressive caching, fewer reruns
# - Correctness: safer indicator math (RSI/ADX/MFI), fewer NaN traps
# - UX: form-based “Analyze” to prevent expensive reruns
# - Robustness: consistent yfinance column handling, resample-safe 4H
# ============================================================

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import datetime
import requests
import urllib.parse
import math

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM UI
# ==========================================
st.set_page_config(layout="wide", page_title="🏦Titan Terminal", page_icon="👁️")

st.markdown(
    """
<style>
.stApp {
    background-color: #0e1117;
    color: #e0e0e0;
    font-family: 'Roboto Mono', monospace;
}
.title-glow {
    font-size: 3em;
    font-weight: bold;
    color: #ffffff;
    text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 40px #00ff00;
    margin-bottom: 20px;
}
div[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 10px;
    border-radius: 8px;
    transition: transform 0.2s;
}
div[data-testid="stMetric"]:hover {
    transform: scale(1.02);
    border-color: #00ff00;
}
div[data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
    font-weight: 700;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #161b22;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
    border: 1px solid #30363d;
    color: #8b949e;
}
.stTabs [aria-selected="true"] {
    background-color: #0e1117;
    color: #00ff00;
    border-bottom: 2px solid #00ff00;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #30363d !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="title-glow">👁️ DarkPool Titan Terminal</div>', unsafe_allow_html=True)
st.markdown("##### *Institutional-Grade Market Intelligence*")
st.markdown("---")

# ==========================================
# 1.1 API KEY MANAGEMENT (OpenAI)
# ==========================================
if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "OPENAI_API_KEY" in st.secrets:
    st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
else:
    if not st.session_state.api_key:
        st.session_state.api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key here to unlock the AI Analyst features.",
        )

# ==========================================
# 2. FAST DATA ENGINE (ROBUST + CACHED)
# ==========================================

def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle MultiIndex columns and ensure standard OHLCV names."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _ensure_close(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee df['Close'] exists (fallback to Adj Close)."""
    if df is None or df.empty:
        return df
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.copy()
            df["Close"] = df["Adj Close"]
        else:
            return None
    return df

def _bad_ticker_for_fundamentals(ticker: str) -> bool:
    # yfinance fundamentals are often unreliable/invalid for indices/futures/forex
    # Keep your original filters, plus a few common non-equity patterns.
    t = ticker.upper().strip()
    if "-" in t or "=" in t or "^" in t:
        return True
    if t.endswith("=X") or t.endswith("-USD") or t.endswith(".NS") or t.endswith(".L") or t.endswith(".TO"):
        # NOTE: still may work, but fundamentals often missing; we don't block entirely here.
        return False
    return False

@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(ticker: str):
    """Fetches key financial metrics safely (cached)."""
    if _bad_ticker_for_fundamentals(ticker) and ("^" in ticker or "=" in ticker):
        return None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info:
            return None
        return {
            "Market Cap": info.get("marketCap", 0),
            "P/E Ratio": info.get("trailingPE", 0),
            "Rev Growth": info.get("revenueGrowth", 0),
            "Debt/Equity": info.get("debtToEquity", 0),
            "Summary": info.get("longBusinessSummary", "No Data Available"),
        }
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def _download_batch(tickers: list[str], period: str, interval: str, group_by: str = "ticker") -> pd.DataFrame | None:
    """Batched yfinance download (cached)."""
    try:
        df = yf.download(
            tickers,
            period=period,
            interval=interval,
            progress=False,
            group_by=group_by,
            auto_adjust=False,
            threads=True,
        )
        df = _normalize_yf_columns(df)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def safe_download(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    """Robust price downloader (cached) with 4H support via 1H resample."""
    try:
        # Yahoo does not natively support 4h -> pull 1h then resample.
        dl_interval = "1h" if interval == "4h" else interval

        df = yf.download(
            ticker,
            period=period,
            interval=dl_interval,
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        df = _normalize_yf_columns(df)
        if df is None or df.empty:
            return None
        df = _ensure_close(df)
        if df is None:
            return None

        # Normalize index to DatetimeIndex (yfinance already does, but be safe)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Resample to 4h if requested
        if interval == "4h":
            agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            if "Adj Close" in df.columns:
                agg["Adj Close"] = "last"
            df = df.resample("4h").agg(agg).dropna()

        # Drop rows that are fully NA (rare but happens)
        df = df.dropna(how="all")
        if df.empty:
            return None

        return df
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def get_global_performance():
    """Fetches performance of a Global Multi-Asset Basket (batched)."""
    assets = {
        "Tech (XLK)": "XLK",
        "Energy (XLE)": "XLE",
        "Financials (XLF)": "XLF",
        "Bitcoin (BTC)": "BTC-USD",
        "Gold (GLD)": "GLD",
        "Oil (USO)": "USO",
        "Treasuries (TLT)": "TLT",
    }
    tickers_list = list(assets.values())
    data = _download_batch(tickers_list, period="5d", interval="1d")
    if data is None:
        return None

    results = {}
    for name, sym in assets.items():
        try:
            if len(tickers_list) > 1 and isinstance(data.columns, pd.MultiIndex):
                # Not typical here due to normalize, but handle anyway
                df = data[sym]
            else:
                # yfinance group_by=ticker returns a top-level column per ticker
                df = data[sym] if (len(tickers_list) > 1 and sym in data.columns) else data

            if df is None or df.empty or len(df) < 2:
                continue

            df = _normalize_yf_columns(df)
            col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
            if col is None:
                continue
            price = float(df[col].iloc[-1])
            prev = float(df[col].iloc[-2])
            if prev == 0:
                continue
            change = ((price - prev) / prev) * 100.0
            results[name] = change
        except Exception:
            continue

    if not results:
        return None
    return pd.Series(results).sort_values(ascending=True)

@st.cache_data(ttl=300, show_spinner=False)
def get_macro_data():
    """Fetches 40 global macro indicators grouped by sector using BATCH DOWNLOAD (FAST)."""
    groups = {
        "🇺🇸 US Equities": {"S&P 500": "SPY", "Nasdaq 100": "QQQ", "Dow Jones": "^DJI", "Russell 2000": "^RUT"},
        "🌍 Global Indices": {"FTSE 100": "^FTSE", "DAX": "^GDAXI", "Nikkei 225": "^N225", "Hang Seng": "^HSI"},
        "🏦 Rates & Bonds": {"10Y Yield": "^TNX", "2Y Yield": "^IRX", "30Y Yield": "^TYX", "T-Bond (TLT)": "TLT"},
        "💱 Forex & Volatility": {"DXY Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "VIX (Fear)": "^VIX"},
        "⚠️ Risk Assets": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Semis (SMH)": "SMH", "Junk Bonds": "HYG"},
        "⚡ Energy": {"WTI Crude": "CL=F", "Brent Crude": "BZ=F", "Natural Gas": "NG=F", "Uranium": "URA"},
        "🥇 Precious Metals": {"Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F"},
        "🏗️ Industrial & Ag": {"Copper": "HG=F", "Rare Earths": "REMX", "Corn": "ZC=F", "Wheat": "ZW=F"},
        "🇬🇧 UK Desk": {"GBP/USD": "GBPUSD=X", "GBP/JPY": "GBPJPY=X", "EUR/GBP": "EURGBP=X", "UK Gilts": "IGLT.L"},
        "📈 Growth & Real Assets": {"Emerging Mkts": "EEM", "China (FXI)": "FXI", "Real Estate": "VNQ", "Soybeans": "ZS=F"},
    }

    all_syms = []
    sym_to_name = {}
    for _, g_dict in groups.items():
        for name, sym in g_dict.items():
            all_syms.append(sym)
            sym_to_name[sym] = name

    data_batch = _download_batch(all_syms, period="5d", interval="1d")
    if data_batch is None:
        return groups, {}, {}

    prices = {}
    changes = {}

    # group_by='ticker' -> columns are tickers at top-level (normalized)
    # So access by data_batch[sym] where possible.
    for sym in all_syms:
        try:
            if sym in data_batch.columns and isinstance(data_batch[sym], pd.DataFrame):
                df = data_batch[sym]
            else:
                # some single-ticker edge or yfinance formatting oddities
                df = data_batch

            df = _normalize_yf_columns(df)
            if df is None or df.empty or len(df) < 2:
                continue
            col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
            if col is None:
                continue
            curr = float(df[col].iloc[-1])
            prev = float(df[col].iloc[-2])
            if prev == 0:
                continue
            chg = ((curr - prev) / prev) * 100.0
            name = sym_to_name.get(sym, sym)
            prices[name] = curr
            changes[name] = chg
        except Exception:
            continue

    return groups, prices, changes

# ==========================================
# 3. MATH LIBRARY & INDICATORS (OPTIMIZED)
# ==========================================

def calculate_wma(series: pd.Series, length: int) -> pd.Series:
    length = int(length)
    if length <= 1:
        return series.copy()
    weights = np.arange(1, length + 1, dtype=float)
    denom = weights.sum()

    def _wma(x):
        return np.dot(x, weights) / denom

    return series.rolling(length).apply(_wma, raw=True)

def calculate_hma(series: pd.Series, length: int) -> pd.Series:
    length = int(length)
    if length <= 1:
        return series.copy()
    half = max(1, length // 2)
    sqrt_len = max(1, int(math.sqrt(length)))
    wma_half = calculate_wma(series, half)
    wma_full = calculate_wma(series, length)
    diff = 2 * wma_half - wma_full
    return calculate_wma(diff, sqrt_len)

def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    length = int(length)
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()

def calculate_rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    length = int(length)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Standard ADX (Wilder) – faster, safer, more correct than sign-diff shortcuts."""
    length = int(length)
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr.replace(0, np.nan)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()
    return adx

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    atr = calculate_atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    close = df["Close"].to_numpy(dtype=float)
    upper = upperband.to_numpy(dtype=float)
    lower = lowerband.to_numpy(dtype=float)

    st = np.zeros(len(df), dtype=float)
    trend = np.zeros(len(df), dtype=float)  # 1 = up, -1 = down

    # init
    st[0] = lower[0] if np.isfinite(lower[0]) else close[0]
    trend[0] = 1

    for i in range(1, len(df)):
        prev_st = st[i - 1]
        prev_trend_up = close[i - 1] > prev_st

        if prev_trend_up:
            st[i] = max(lower[i], prev_st) if close[i] > prev_st else upper[i]
            trend[i] = 1 if close[i] > prev_st else -1
            if close[i] < lower[i] and trend[i - 1] == 1:
                st[i] = upper[i]
                trend[i] = -1
        else:
            st[i] = min(upper[i], prev_st) if close[i] < prev_st else lower[i]
            trend[i] = -1 if close[i] < prev_st else 1
            if close[i] > upper[i] and trend[i - 1] == -1:
                st[i] = lower[i]
                trend[i] = 1

    return pd.Series(st, index=df.index), pd.Series(trend, index=df.index)

def calculate_mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Real Money Flow Index (safer than the prior diff*vol proxy)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    rmf = tp * df["Volume"]
    direction = tp.diff()
    pos = rmf.where(direction > 0, 0.0)
    neg = rmf.where(direction < 0, 0.0).abs()
    pos_sum = pos.rolling(length).sum()
    neg_sum = neg.rolling(length).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    return mfi.fillna(50)

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Base Indicators + 10 GOD MODE INDICATORS (optimized & safer)."""
    df = df.copy()

    # --- Base calcs ---
    df["HMA"] = calculate_hma(df["Close"], 55)
    df["ATR"] = calculate_atr(df, 14)
    df["Pivot_Resist"] = df["High"].rolling(20).max()
    df["Pivot_Support"] = df["Low"].rolling(20).min()
    df["MFI"] = calculate_mfi(df, 14)

    # --- RSI (single source of truth, reused) ---
    df["RSI"] = calculate_rsi_wilder(df["Close"], 14)

    # ==========================================
    # GOD MODE INDICATORS
    # ==========================================

    # 1. Apex Trend & Liquidity Master (HMA + ATR Bands)
    apex_mult = 1.5
    df["Apex_Base"] = df["HMA"]
    df["Apex_ATR"] = calculate_atr(df, 55)
    df["Apex_Upper"] = df["Apex_Base"] + (df["Apex_ATR"] * apex_mult)
    df["Apex_Lower"] = df["Apex_Base"] - (df["Apex_ATR"] * apex_mult)

    apex_raw = np.where(
        df["Close"] > df["Apex_Upper"],
        1,
        np.where(df["Close"] < df["Apex_Lower"], -1, np.nan),
    )
    df["Apex_Trend"] = pd.Series(apex_raw, index=df.index).ffill().fillna(0).astype(int)

    # 2. DarkPool Squeeze Momentum
    df["Sqz_Basis"] = df["Close"].rolling(20).mean()
    df["Sqz_Dev"] = df["Close"].rolling(20).std() * 2.0
    df["Sqz_Upper_BB"] = df["Sqz_Basis"] + df["Sqz_Dev"]
    df["Sqz_Lower_BB"] = df["Sqz_Basis"] - df["Sqz_Dev"]

    df["Sqz_Ma_KC"] = df["Close"].rolling(20).mean()
    df["Sqz_Range_MA"] = calculate_atr(df, 20)
    df["Sqz_Upper_KC"] = df["Sqz_Ma_KC"] + (df["Sqz_Range_MA"] * 1.5)
    df["Sqz_Lower_KC"] = df["Sqz_Ma_KC"] - (df["Sqz_Range_MA"] * 1.5)

    df["Squeeze_On"] = (df["Sqz_Lower_BB"] > df["Sqz_Lower_KC"]) & (df["Sqz_Upper_BB"] < df["Sqz_Upper_KC"])

    highest = df["High"].rolling(20).max()
    lowest = df["Low"].rolling(20).min()
    avg_val = (highest + lowest + df["Sqz_Ma_KC"]) / 3.0
    df["Sqz_Mom"] = (df["Close"] - avg_val).rolling(20).mean() * 100.0

    # 3. Money Flow Matrix (Normalized RSI * Vol)
    rsi_src = df["RSI"] - 50
    vol_ma14 = df["Volume"].rolling(14).mean().replace(0, np.nan)
    mf_vol = (df["Volume"] / vol_ma14).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["MF_Matrix"] = (rsi_src * mf_vol).ewm(span=3, adjust=False).mean()

    # 4. Dark Vector Scalping (Staircase / Donchian proxy)
    amp = 5
    df["VS_Low"] = df["Low"].rolling(amp).min()
    df["VS_High"] = df["High"].rolling(amp).max()
    vs_raw = np.where(
        df["Close"] > df["VS_High"].shift(1),
        1,
        np.where(df["Close"] < df["VS_Low"].shift(1), -1, np.nan),
    )
    df["VS_Trend"] = pd.Series(vs_raw, index=df.index).ffill().fillna(0).astype(int)

    # 5. Advanced Volume (RVOL)
    df["RVOL"] = (df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["RVOL"] = df["RVOL"].fillna(1.0)

    # 6. Elastic Volume Weighted Momentum (EVWM)
    ev_len = 21
    ev_base = calculate_hma(df["Close"], ev_len)
    ev_atr = calculate_atr(df, ev_len).replace(0, np.nan)
    ev_elast = (df["Close"] - ev_base) / ev_atr
    ev_force = np.sqrt(df["RVOL"].ewm(span=5, adjust=False).mean())
    df["EVWM"] = (ev_elast * ev_force).replace([np.inf, -np.inf], np.nan).fillna(0)

    # 7. Ultimate S&R (Pivot Breaks) already present: Pivot_Resist / Pivot_Support

    # 8. Gann High Low Activator
    gann_len = 3
    df["Gann_High"] = df["High"].rolling(gann_len).mean()
    df["Gann_Low"] = df["Low"].rolling(gann_len).mean()
    gann_raw = np.where(
        df["Close"] > df["Gann_High"].shift(1),
        1,
        np.where(df["Close"] < df["Gann_Low"].shift(1), -1, np.nan),
    )
    df["Gann_Trend"] = pd.Series(gann_raw, index=df.index).ffill().fillna(0).astype(int)

    # 9. Dark Vector (SuperTrend)
    _, st_dir = calculate_supertrend(df, 10, 4.0)
    df["DarkVector_Trend"] = st_dir.fillna(0).astype(int)

    # 10. Wyckoff VSA (Trend Shield)
    df["Trend_Shield_Bull"] = df["Close"] > df["Close"].rolling(200).mean()

    # --- GOD MODE CONFLUENCE SIGNAL ---
    df["GM_Score"] = (
        df["Apex_Trend"]
        + df["Gann_Trend"]
        + df["DarkVector_Trend"]
        + df["VS_Trend"]
        + np.sign(df["Sqz_Mom"].fillna(0))
    )

    # --- Classic dashboard indicators (kept, computed once) ---
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]

    low_min = df["Low"].rolling(14).min()
    high_max = df["High"].rolling(14).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df["Stoch_K"] = (100 * (df["Close"] - low_min) / denom).fillna(50)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    df["ROC"] = df["Close"].pct_change(14) * 100
    df["EMA_Fast"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_Slow"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    df["OBV"] = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (df["Volume"] * tp).cumsum() / df["Volume"].cumsum().replace(0, np.nan)

    df["ADX"] = calculate_adx(df, 14)

    # Momentum composite score (unchanged idea, safer math)
    rsi_norm = (df["RSI"] - 50) * 2
    macd_norm = np.where(df["Hist"] > 0, np.minimum(df["Hist"] * 10, 100), np.maximum(df["Hist"] * 10, -100))
    stoch_norm = (df["Stoch_K"] - 50) * 2
    roc_norm = np.where(df["ROC"] > 0, np.minimum(df["ROC"] * 10, 100), np.maximum(df["ROC"] * 10, -100))
    df["Mom_Score"] = np.round((rsi_norm + macd_norm + stoch_norm + roc_norm) / 4).fillna(0)

    return df

def calc_fear_greed_v4(df: pd.DataFrame) -> pd.DataFrame:
    """🔥 DarkPool's Fear & Greed v4 Port (optimized, reuses RSI/MACD/BB/MA)."""
    df = df.copy()

    # 1. RSI Component
    df["FG_RSI"] = calculate_rsi_wilder(df["Close"], 14)

    # 2. MACD Component
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    df["FG_MACD"] = (50 + (hist * 10)).clip(0, 100)

    # 3. Bollinger Band Component
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)
    bb_denom = (upper - lower).replace(0, np.nan)
    df["FG_BB"] = ((df["Close"] - lower) / bb_denom * 100).clip(0, 100).fillna(50)

    # 4. Moving Average Trend Component
    sma50 = df["Close"].rolling(50).mean()
    sma200 = df["Close"].rolling(200).mean()

    conditions = [
        (df["Close"] > sma50) & (sma50 > sma200),
        (df["Close"] > sma50),
        (df["Close"] < sma50) & (sma50 < sma200),
    ]
    choices = [75, 60, 25]
    df["FG_MA"] = np.select(conditions, choices, default=40)

    # Composite Index
    df["FG_Raw"] = (df["FG_RSI"] * 0.30) + (df["FG_MACD"] * 0.25) + (df["FG_BB"] * 0.25) + (df["FG_MA"] * 0.20)
    df["FG_Index"] = df["FG_Raw"].rolling(5).mean().fillna(df["FG_Raw"])

    # FOMO
    vol_ma = df["Volume"].rolling(20).mean().replace(0, np.nan)
    high_vol = df["Volume"] > (vol_ma * 2.5)
    high_rsi = df["FG_RSI"] > 70
    momentum = df["Close"] > (df["Close"].shift(3) * 1.02)
    above_bb = df["Close"] > upper
    df["IS_FOMO"] = (high_vol & high_rsi & momentum & above_bb).fillna(False)

    # PANIC
    daily_drop = df["Close"].pct_change() * 100
    sharp_drop = daily_drop < -3.0
    panic_vol = df["Volume"] > (vol_ma * 3.0)
    low_rsi = df["FG_RSI"] < 30
    df["IS_PANIC"] = (sharp_drop & panic_vol & (low_rsi | (daily_drop < -5.0))).fillna(False)

    return df

def run_monte_carlo(df: pd.DataFrame, days: int = 30, simulations: int = 1000) -> np.ndarray:
    """🔮 Monte Carlo Simulation (vectorized)."""
    close = df["Close"].dropna()
    last_price = float(close.iloc[-1])
    returns = close.pct_change().dropna()
    if returns.empty:
        return np.full((days, simulations), last_price, dtype=float)

    mu = float(returns.mean())
    sigma = float(returns.std(ddof=0))

    daily = np.random.normal(mu, sigma, (days, simulations))
    paths = np.empty((days, simulations), dtype=float)
    paths[0, :] = last_price
    for t in range(1, days):
        paths[t, :] = paths[t - 1, :] * (1.0 + daily[t, :])
    return paths

def calc_volume_profile(df: pd.DataFrame, bins: int = 50):
    """📊 Institutional Volume Profile (VPVR) without mutating input df."""
    low = df["Low"].min()
    high = df["High"].max()
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        vp = pd.DataFrame({"Price": [], "Volume": []})
        return vp, np.nan

    price_bins = np.linspace(low, high, bins)
    mid = (df["Close"] + df["Open"]) / 2.0

    cats = pd.cut(mid, bins=price_bins, labels=price_bins[:-1], include_lowest=True)
    vp = df.groupby(cats, observed=False)["Volume"].sum().reset_index()
    vp.columns = ["Bin", "Volume"]
    vp["Price"] = vp["Bin"].astype(float)

    if vp["Volume"].empty:
        return vp[["Price", "Volume"]], np.nan

    poc_idx = int(vp["Volume"].idxmax())
    poc_price = float(vp.loc[poc_idx, "Price"])
    return vp[["Price", "Volume"]], poc_price

def get_sr_channels(df: pd.DataFrame, pivot_period: int = 10, loopback: int = 290, max_width_pct: float = 5, min_strength: int = 1):
    """Python implementation of 'Support Resistance Channels' logic (optimized but same output intent)."""
    if df is None or df.empty:
        return []
    if len(df) < loopback:
        loopback = len(df)

    window = df.iloc[-loopback:].copy()

    span = pivot_period * 2 + 1
    window["Is_Pivot_H"] = window["High"] == window["High"].rolling(span, center=True).max()
    window["Is_Pivot_L"] = window["Low"] == window["Low"].rolling(span, center=True).min()

    pivots = np.concatenate([window.loc[window["Is_Pivot_H"], "High"].to_numpy(), window.loc[window["Is_Pivot_L"], "Low"].to_numpy()])
    if pivots.size == 0:
        return []

    pivots.sort()

    price_range = float(window["High"].max() - window["Low"].min())
    max_width = price_range * (max_width_pct / 100.0)

    potential = []
    n = len(pivots)
    for i in range(n):
        seed = pivots[i]
        cluster_min = seed
        cluster_max = seed

        j = i
        while j + 1 < n and (pivots[j + 1] - seed) <= max_width:
            j += 1
            cluster_max = pivots[j]

        touches = int(((window["High"] >= cluster_min) & (window["Low"] <= cluster_max)).sum())
        pivot_count = (j - i + 1)
        score = (pivot_count * 20) + touches

        if score >= min_strength:
            potential.append({"min": float(cluster_min), "max": float(cluster_max), "score": int(score)})

    potential.sort(key=lambda x: x["score"], reverse=True)

    final = []
    for zone in potential:
        overlap = any((zone["min"] < z["max"]) and (zone["max"] > z["min"]) for z in final)
        if not overlap:
            final.append(zone)
            if len(final) >= 6:
                break
    return final

def calculate_smc(df: pd.DataFrame, swing_length: int = 5):
    """🏦 LuxAlgo Smart Money Concepts (kept; micro-optimized)."""
    smc_data = {"structures": [], "order_blocks": [], "fvgs": []}
    if df is None or df.empty or len(df) < 10:
        return smc_data

    # FVGs
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    idx = df.index

    for i in range(2, len(df)):
        if lows[i] > highs[i - 2]:
            smc_data["fvgs"].append(
                {"x0": idx[i - 2], "x1": idx[i], "y0": float(highs[i - 2]), "y1": float(lows[i]), "color": "rgba(0, 255, 104, 0.3)"}
            )
        if highs[i] < lows[i - 2]:
            smc_data["fvgs"].append(
                {"x0": idx[i - 2], "x1": idx[i], "y0": float(lows[i - 2]), "y1": float(highs[i]), "color": "rgba(255, 0, 8, 0.3)"}
            )

    span = swing_length * 2 + 1
    pivH = df["High"].rolling(span, center=True).max() == df["High"]
    pivL = df["Low"].rolling(span, center=True).min() == df["Low"]

    last_high = None
    last_low = None
    trend = 0  # 1 bull, -1 bear

    close = df["Close"].to_numpy()

    for i in range(swing_length, len(df)):
        curr_idx = idx[i]
        curr_close = float(close[i])

        pivot_i = i - swing_length
        if pivH.iloc[pivot_i]:
            last_high = {"price": float(highs[pivot_i]), "idx": idx[pivot_i], "i": pivot_i}
        if pivL.iloc[pivot_i]:
            last_low = {"price": float(lows[pivot_i]), "idx": idx[pivot_i], "i": pivot_i}

        if last_high and curr_close > last_high["price"]:
            label = "CHoCH" if trend != 1 else "BOS"
            trend = 1
            smc_data["structures"].append({"x0": last_high["idx"], "x1": curr_idx, "y": last_high["price"], "color": "green", "label": label})

            if last_low:
                subset = df.iloc[last_low["i"] : i]
                if not subset.empty:
                    ob_idx = subset["Low"].idxmin()
                    ob_row = df.loc[ob_idx]
                    smc_data["order_blocks"].append(
                        {"x0": ob_idx, "x1": df.index[-1], "y0": float(ob_row["Low"]), "y1": float(ob_row["High"]), "color": "rgba(33, 87, 243, 0.4)"}
                    )
            last_high = None

        elif last_low and curr_close < last_low["price"]:
            label = "CHoCH" if trend != -1 else "BOS"
            trend = -1
            smc_data["structures"].append({"x0": last_low["idx"], "x1": curr_idx, "y": last_low["price"], "color": "red", "label": label})

            if last_high:
                subset = df.iloc[last_high["i"] : i]
                if not subset.empty:
                    ob_idx = subset["High"].idxmax()
                    ob_row = df.loc[ob_idx]
                    smc_data["order_blocks"].append(
                        {"x0": ob_idx, "x1": df.index[-1], "y0": float(ob_row["Low"]), "y1": float(ob_row["High"]), "color": "rgba(255, 0, 0, 0.4)"}
                    )
            last_low = None

    return smc_data

@st.cache_data(ttl=1800, show_spinner=False)
def calc_correlations(ticker: str, lookback_days: int = 180):
    """🧩 Cross-Asset Correlation Matrix (cached)."""
    macro_tickers = {
        "S&P 500": "SPY",
        "Bitcoin": "BTC-USD",
        "10Y Yield": "^TNX",
        "Dollar (DXY)": "DX-Y.NYB",
        "Gold": "GC=F",
        "Oil": "CL=F",
    }

    df_main = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)["Close"]
    df_macro = yf.download(list(macro_tickers.values()), period="1y", interval="1d", progress=False, auto_adjust=False)

    df_macro = _normalize_yf_columns(df_macro)
    if isinstance(df_macro, pd.DataFrame) and "Close" in df_macro.columns:
        # if single-column (rare), attempt reshape
        macro_close = df_macro["Close"]
    else:
        macro_close = df_macro["Close"] if isinstance(df_macro, pd.DataFrame) and "Close" in df_macro else df_macro

    combined = pd.DataFrame(macro_close).copy()
    combined[ticker] = df_main
    combined = combined.dropna()

    if combined.empty:
        return pd.Series(dtype=float)

    corr = combined.iloc[-lookback_days:].corr()
    if ticker not in corr.columns:
        return pd.Series(dtype=float)

    target = corr[ticker].drop(labels=[ticker], errors="ignore").sort_values(ascending=False)

    inv_map = {v: k for k, v in macro_tickers.items()}
    target.index = [inv_map.get(x, x) for x in target.index]
    return target

@st.cache_data(ttl=1800, show_spinner=False)
def calc_mtf_trend(ticker: str):
    """📡 Multi-Timeframe Trend Radar (cached)."""
    timeframes = {"1H": "1h", "4H": "1h", "Daily": "1d", "Weekly": "1wk"}
    trends = {}

    for tf_name, tf_code in timeframes.items():
        try:
            if tf_name in {"1H", "4H"}:
                period = "1y"
            else:
                period = "2y"

            df = yf.download(ticker, period=period, interval=tf_code, progress=False, auto_adjust=False)
            df = _normalize_yf_columns(df)
            df = _ensure_close(df)

            if df is None or df.empty or len(df) < 50:
                trends[tf_name] = {"Trend": "N/A", "RSI": "N/A", "EMA Spread": "N/A"}
                continue

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            if tf_name == "4H":
                agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                if "Adj Close" in df.columns:
                    agg["Adj Close"] = "last"
                df = df.resample("4h").agg(agg).dropna()

            df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
            df["RSI"] = calculate_rsi_wilder(df["Close"], 14)

            last = df.iloc[-1]
            if last["Close"] > last["EMA20"] and last["EMA20"] > last["EMA50"]:
                trend = "BULLISH"
            elif last["Close"] < last["EMA20"] and last["EMA20"] < last["EMA50"]:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            trends[tf_name] = {"Trend": trend, "RSI": f"{last['RSI']:.1f}", "EMA Spread": f"{(last['EMA20'] - last['EMA50']):.2f}"}

        except Exception:
            trends[tf_name] = {"Trend": "N/A", "RSI": "N/A", "EMA Spread": "N/A"}

    return pd.DataFrame(trends).T

@st.cache_data(ttl=900, show_spinner=False)
def calc_intraday_dna(ticker: str):
    """⏱️ Intraday Seasonality (Hour of Day) (cached)."""
    try:
        df = yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=False)
        df = _normalize_yf_columns(df)
        df = _ensure_close(df)
        if df is None or df.empty:
            return None

        df = df.copy()
        df["Return"] = df["Close"].pct_change() * 100.0
        df["Hour"] = df.index.hour

        win_rate = df["Return"].gt(0).groupby(df["Hour"]).mean() * 100.0
        hourly = df.groupby("Hour")["Return"].agg(["mean", "sum", "count"]).rename(columns={"mean": "Avg Return", "sum": "Total Return", "count": "Count"})
        hourly["Win Rate"] = win_rate
        return hourly
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_seasonality_stats(ticker: str):
    """Calculates Monthly Seasonality and Probability Stats (cached)."""
    try:
        df = yf.download(ticker, period="20y", interval="1mo", progress=False, auto_adjust=False)
        df = _normalize_yf_columns(df)
        df = _ensure_close(df)
        if df is None or df.empty or len(df) < 12:
            return None

        df = df.dropna().copy()
        df["Return"] = df["Close"].pct_change() * 100.0
        df["Year"] = df.index.year
        df["Month"] = df.index.month

        heatmap_data = df.pivot_table(index="Year", columns="Month", values="Return")

        periods = [1, 3, 6, 12]
        hold_stats = {}
        for p in periods:
            rolling_ret = df["Close"].pct_change(periods=p) * 100.0
            rolling_ret = rolling_ret.dropna()
            if rolling_ret.empty:
                hold_stats[p] = {"Win Rate": 0.0, "Avg Return": 0.0}
                continue
            win_rate = float((rolling_ret > 0).mean() * 100.0)
            avg_ret = float(rolling_ret.mean())
            hold_stats[p] = {"Win Rate": win_rate, "Avg Return": avg_ret}

        month_stats = df.groupby("Month")["Return"].agg(["mean", "count"])
        month_stats["Win Rate"] = df.groupby("Month")["Return"].apply(lambda x: (x > 0).mean() * 100.0)
        month_stats = month_stats.rename(columns={"mean": "Avg Return"})
        month_stats = month_stats[["Avg Return", "Win Rate", "count"]].rename(columns={"count": "Count"})

        return heatmap_data, hold_stats, month_stats
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def calc_day_of_week_dna(ticker: str, lookback: int, calc_mode: str):
    """DarkPool's Day of Week Seasonality DNA Port (cached)."""
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=False)
        df = _normalize_yf_columns(df)
        df = _ensure_close(df)
        if df is None or df.empty:
            return None

        df = df.iloc[-lookback:].copy()

        if calc_mode == "Close to Close (Total)":
            df["Day_Return"] = df["Close"].pct_change() * 100.0
        else:
            df["Day_Return"] = ((df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)) * 100.0

        df = df.dropna()
        df["Day_Name"] = df.index.day_name()

        pivot_ret = df.pivot(columns="Day_Name", values="Day_Return").fillna(0.0)
        cum_ret = pivot_ret.cumsum()

        stats = df.groupby("Day_Name")["Day_Return"].agg(["count", "sum", "mean"])
        stats["Win Rate"] = df.groupby("Day_Name")["Day_Return"].apply(lambda x: (x > 0).mean() * 100.0)
        stats = stats.rename(columns={"count": "Count", "sum": "Total Return", "mean": "Avg Return"})

        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        stats = stats.reindex([d for d in order if d in stats.index])
        return cum_ret, stats
    except Exception:
        return None

# ==========================================
# 4. AI ANALYST (TIMEFRAME-AWARE + SESSION CACHED)
# ==========================================
def ask_ai_analyst(df: pd.DataFrame, ticker: str, fundamentals: dict | None, balance: float, risk_pct: float, timeframe: str) -> str:
    if not st.session_state.api_key:
        return "⚠️ Waiting for OpenAI API Key in the sidebar..."

    last = df.iloc[-1]
    trend = "BULLISH" if last["Close"] > last["HMA"] else "BEARISH"
    risk_dollars = balance * (risk_pct / 100.0)

    gm_score = float(last.get("GM_Score", 0))
    gm_verdict = "STRONG BUY" if gm_score >= 3 else "STRONG SELL" if gm_score <= -3 else "NEUTRAL"

    # kept (but not used for explicit levels)
    if trend == "BULLISH":
        stop_level = last.get("Pivot_Support", np.nan)
        direction = "LONG"
    else:
        stop_level = last.get("Pivot_Resist", np.nan)
        direction = "SHORT"

    if pd.isna(stop_level) or abs(last["Close"] - stop_level) < (last["ATR"] * 0.5):
        stop_level = last["Close"] - (last["ATR"] * 2) if direction == "LONG" else last["Close"] + (last["ATR"] * 2)

    dist = abs(last["Close"] - stop_level)
    if dist == 0 or not np.isfinite(dist):
        dist = float(last["ATR"]) if np.isfinite(last["ATR"]) else 1.0

    _shares = risk_dollars / dist  # computed but not disclosed

    fund_text = "N/A"
    if fundamentals:
        pe = fundamentals.get("P/E Ratio", "N/A")
        growth = fundamentals.get("Rev Growth", 0) or 0
        fund_text = f"P/E: {pe}. Growth: {growth * 100:.1f}%."

    fg_val = float(last.get("FG_Index", 50))
    fg_state = (
        "EXTREME GREED"
        if fg_val >= 80
        else "GREED"
        if fg_val >= 60
        else "NEUTRAL"
        if fg_val >= 40
        else "FEAR"
        if fg_val >= 20
        else "EXTREME FEAR"
    )
    psych_alert = ""
    if bool(last.get("IS_FOMO", False)):
        psych_alert = "WARNING: ALGORITHMIC FOMO DETECTED."
    if bool(last.get("IS_PANIC", False)):
        psych_alert = "WARNING: PANIC SELLING DETECTED."

    prompt = f"""
Act as a Senior Market Analyst. Analyze {ticker} on the **{timeframe} timeframe** at price ${last['Close']:.2f}.

--- DATA FEED ---
Technicals: Trend is {trend}. Volatility (ATR) is {last['ATR']:.2f}.
RSI: {last['RSI']:.1f}.
Volume (RVOL): {last['RVOL']:.1f}x.
Titan Score: {gm_score:.0f} ({gm_verdict}).
Momentum: {'Rising' if last['Sqz_Mom'] > 0 else 'Falling'}.
Sentiment: {fg_state} ({fg_val:.1f}/100).
{psych_alert}
Fundamentals: {fund_text}

--- MISSION ---
Provide a concise, high-level overview of what is happening with this asset.
1. Analyze the current market structure (Trend vs Chop).
2. Explain the correlation between the technicals and sentiment.
3. Provide a general outlook on potential direction.

IMPORTANT:
- Do NOT provide specific Entry prices, Exit prices, or Stop Loss numbers.
- Do NOT give specific financial advice.
- Keep it to a market situation overview only.
- USE EMOJIS liberally to make the report engaging and visually appealing (e.g., 🚀, 📉, 🐂, 🐻, 🧠, ⚠️).
"""

    try:
        client = OpenAI(api_key=st.session_state.api_key)
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"

def _ai_cache_key(ticker: str, interval: str, last_ts) -> str:
    return f"{ticker}|{interval}|{str(last_ts)}"

# ==========================================
# 5. UI DASHBOARD LAYOUT (FORM-BASED)
# ==========================================
st.sidebar.header("🎛️ Terminal Controls")
st.sidebar.subheader("📢 Social Broadcaster")

# Telegram State
if "tg_token" not in st.session_state:
    st.session_state.tg_token = ""
if "tg_chat" not in st.session_state:
    st.session_state.tg_chat = ""

if "TELEGRAM_TOKEN" in st.secrets:
    st.session_state.tg_token = st.secrets["TELEGRAM_TOKEN"]
if "TELEGRAM_CHAT_ID" in st.secrets:
    st.session_state.tg_chat = st.secrets["TELEGRAM_CHAT_ID"]

tg_token = st.sidebar.text_input(
    "Telegram Bot Token",
    value=st.session_state.tg_token,
    type="password",
    help="Enter your Telegram Bot Token",
)
tg_chat = st.sidebar.text_input(
    "Telegram Chat ID",
    value=st.session_state.tg_chat,
    help="Enter your Telegram Chat ID",
)

input_mode = st.sidebar.radio(
    "Input Mode:",
    ["Curated Lists", "Manual Search (Global)"],
    index=1,
    help="Select input mode",
)

assets = {
    "Indices": ["SPY", "QQQ", "DIA", "IWM", "VTI"],
    "Crypto (Top 20)": [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "TRX-USD",
        "LINK-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "BCH-USD",
        "XLM-USD", "ALGO-USD", "ATOM-USD", "UNI-USD", "FIL-USD",
    ],
    "Tech Giants (Top 10)": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "NFLX", "INTC"],
    "Macro & Commodities": ["^TNX", "DX-Y.NYB", "GC=F", "SI=F", "CL=F", "NG=F", "^VIX", "TLT"],
}

if input_mode == "Curated Lists":
    cat = st.sidebar.selectbox("Asset Class", list(assets.keys()), help="Select asset class")
    ticker = st.sidebar.selectbox("Ticker", assets[cat], help="Select ticker")
else:
    st.sidebar.info("Type any ticker (e.g. SSLN.L, BTC-USD)")
    ticker = st.sidebar.text_input("Search Ticker Symbol", value="BTC-USD", help="Enter ticker symbol").upper().strip()

interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=3, help="Select time interval")
st.sidebar.markdown("---")

balance = st.sidebar.number_input("Capital ($)", 1000, 1000000, 10000, help="Enter your capital")
risk_pct = st.sidebar.slider("Risk %", 0.5, 3.0, 1.0, help="Select risk percentage")

# --- GLOBAL MACRO HEADER (cached) ---
macro_groups, m_price, m_chg = get_macro_data()

if m_price:
    group_names = list(macro_groups.keys())
    for i in range(0, len(group_names), 2):
        cols = st.columns(2)
        g1 = group_names[i]
        with cols[0].container(border=True):
            st.markdown(f"#### {g1}")
            sc = st.columns(4)
            for x, (n, _) in enumerate(macro_groups[g1].items()):
                fmt = "{:.3f}" if any(k in n for k in ["Yield", "GBP", "EUR", "JPY"]) else "{:,.2f}"
                sc[x].metric(n.split("(")[0], fmt.format(m_price.get(n, 0)), f"{m_chg.get(n, 0):.2f}%")

        if i + 1 < len(group_names):
            g2 = group_names[i + 1]
            with cols[1].container(border=True):
                st.markdown(f"#### {g2}")
                sc = st.columns(4)
                for x, (n, _) in enumerate(macro_groups[g2].items()):
                    fmt = "{:.3f}" if any(k in n for k in ["Yield", "GBP", "EUR", "JPY"]) else "{:,.2f}"
                    sc[x].metric(n.split("(")[0], fmt.format(m_price.get(n, 0)), f"{m_chg.get(n, 0):.2f}%")
    st.markdown("---")

# --- MAIN ANALYSIS TABS ---
tab1, tab2, tab3, tab4, tab9, tab5, tab6, tab7, tab8, tab10 = st.tabs(
    [
        "📊 God Mode Technicals",
        "🌍 Sector & Fundamentals",
        "📅 Monthly Seasonality",
        "📆 Day of Week DNA",
        "🧩 Correlation & MTF",
        "📟 DarkPool Dashboard",
        "🏦 Smart Money Concepts",
        "🔮 Quantitative Forecasting",
        "📊 Volume Profile",
        "📡 Broadcast & TradingView",
    ]
)

# Use a FORM to prevent every widget change from re-running expensive analysis
with st.sidebar.form("analyze_form", border=True):
    st.write("⚡ Execute Analysis")
    run = st.form_submit_button(f"Analyze {ticker}")

# Persist “last analysis”
if run:
    st.session_state["run_analysis"] = True
    st.session_state["last_ticker"] = ticker
    st.session_state["last_interval"] = interval

if st.session_state.get("run_analysis"):
    with st.spinner(f"Analyzing {ticker} in God Mode..."):
        # Dynamic fetch period
        if interval in ["1m", "2m", "5m", "15m", "30m"]:
            fetch_period = "59d"
        elif interval in ["1h", "4h"]:
            fetch_period = "1y"
        else:
            fetch_period = "2y"

        df = safe_download(ticker, fetch_period, interval)

        if df is None or df.empty:
            st.error("Data connection failed. Try another ticker.")
        else:
            # Indicators
            df = calc_indicators(df)
            df = calc_fear_greed_v4(df)

            fund = get_fundamentals(ticker)
            sr_zones = get_sr_channels(df)

            # --- TAB 1: TECHNICALS (GOD MODE) ---
            with tab1:
                st.subheader(f"🎯 Apex God Mode: {ticker}")
                col_chart, col_gauge = st.columns([0.75, 0.25])

                with col_chart:
                    fig = make_subplots(
                        rows=3,
                        cols=1,
                        shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.02,
                    )

                    # Price
                    fig.add_trace(
                        go.Candlestick(
                            x=df.index,
                            open=df["Open"],
                            high=df["High"],
                            low=df["Low"],
                            close=df["Close"],
                            name="Price",
                        ),
                        row=1,
                        col=1,
                    )

                    # Apex Cloud
                    fig.add_trace(go.Scatter(x=df.index, y=df["Apex_Upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["Apex_Lower"],
                            fill="tonexty",
                            fillcolor="rgba(0, 230, 118, 0.1)",
                            line=dict(width=0),
                            name="Apex Cloud",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(go.Scatter(x=df.index, y=df["HMA"], line=dict(color="yellow", width=2), name="HMA Trend"), row=1, col=1)

                    # GM Signals
                    buy_signals = df[df["GM_Score"] >= 3]
                    sell_signals = df[df["GM_Score"] <= -3]
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals.index,
                            y=buy_signals["Low"] * 0.98,
                            mode="markers",
                            marker=dict(symbol="triangle-up", color="#00ff00", size=10),
                            name="GM Buy",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals.index,
                            y=sell_signals["High"] * 1.02,
                            mode="markers",
                            marker=dict(symbol="triangle-down", color="#ff0000", size=10),
                            name="GM Sell",
                        ),
                        row=1,
                        col=1,
                    )

                    # SR zones
                    for z in sr_zones:
                        col = "rgba(0, 255, 0, 0.15)" if df["Close"].iloc[-1] > z["max"] else "rgba(255, 0, 0, 0.15)"
                        fig.add_shape(
                            type="rect",
                            x0=df.index[0],
                            x1=df.index[-1],
                            xref="x",
                            yref="y",
                            y0=z["min"],
                            y1=z["max"],
                            fillcolor=col,
                            line=dict(width=0),
                            row=1,
                            col=1,
                        )

                    # Squeeze Momentum
                    colors = ["#00E676" if v > 0 else "#FF5252" for v in df["Sqz_Mom"].fillna(0)]
                    fig.add_trace(go.Bar(x=df.index, y=df["Sqz_Mom"], marker_color=colors, name="Squeeze Mom"), row=2, col=1)

                    # Money Flow Matrix
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df["MF_Matrix"], fill="tozeroy", line=dict(color="cyan", width=1), name="Money Flow"),
                        row=3,
                        col=1,
                    )

                    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, title_text="God Mode Technical Stack")
                    st.plotly_chart(fig, use_container_width=True)

                with col_gauge:
                    fg_val = float(df["FG_Index"].iloc[-1])
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=fg_val,
                            title={"text": "Fear & Greed"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "white"},
                                "steps": [{"range": [0, 20], "color": "#FF0000"}, {"range": [80, 100], "color": "#00FF00"}],
                            },
                        )
                    )
                    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.markdown("### 🧬 Indicator DNA")
                    last_row = df.iloc[-1]
                    st.metric("God Mode Score", f"{last_row['GM_Score']:.0f} / 5", delta="Bullish" if last_row["GM_Score"] > 0 else "Bearish")
                    st.metric("Apex Trend", "BULL" if last_row["Apex_Trend"] == 1 else "BEAR")
                    st.metric("Squeeze", "ON" if bool(last_row["Squeeze_On"]) else "OFF")
                    st.metric("Money Flow", f"{last_row['MF_Matrix']:.2f}")

                st.markdown("### 🤖 Strategy Briefing")

                # AI verdict cached in session_state by (ticker, interval, last timestamp) to avoid re-calling on reruns
                ai_key = _ai_cache_key(ticker, interval, df.index[-1])
                if "ai_verdicts" not in st.session_state:
                    st.session_state.ai_verdicts = {}

                if ai_key not in st.session_state.ai_verdicts:
                    st.session_state.ai_verdicts[ai_key] = ask_ai_analyst(df, ticker, fund, balance, risk_pct, interval)

                ai_verdict = st.session_state.ai_verdicts[ai_key]
                st.info(ai_verdict)

            # --- TAB 2: FUNDAMENTALS ---
            with tab2:
                if fund:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("P/E Ratio", f"{fund.get('P/E Ratio', 'N/A')}")
                    c2.metric("Rev Growth", f"{(fund.get('Rev Growth', 0) or 0) * 100:.1f}%")
                    c3.metric("Debt/Equity", f"{fund.get('Debt/Equity', 'N/A')}")
                    st.write(f"**Summary:** {fund.get('Summary', 'No Data')}")
                else:
                    st.info("Fundamentals not available for this ticker.")

                st.subheader("🔥 Global Market Heatmap")
                s_data = get_global_performance()
                if s_data is not None and not s_data.empty:
                    fig_sector = go.Figure(
                        go.Bar(
                            x=s_data.values,
                            y=s_data.index,
                            orientation="h",
                            marker_color=["#00ff00" if v >= 0 else "#ff0000" for v in s_data.values],
                        )
                    )
                    fig_sector.update_layout(height=400, template="plotly_dark")
                    st.plotly_chart(fig_sector, use_container_width=True)

            # --- TAB 3: SEASONALITY ---
            with tab3:
                seas = get_seasonality_stats(ticker)
                if seas:
                    hm, hold, month = seas
                    fig_hm = px.imshow(hm, color_continuous_scale="RdYlGn", text_auto=".1f")
                    fig_hm.update_layout(template="plotly_dark", height=500)
                    st.plotly_chart(fig_hm, use_container_width=True)

                    c1, c2 = st.columns(2)
                    c1.dataframe(pd.DataFrame(hold).T.style.format("{:.1f}%").background_gradient(cmap="RdYlGn"))

                    curr_m = datetime.datetime.now().month
                    if curr_m in month.index:
                        c2.metric("Current Month Win Rate", f"{month.loc[curr_m, 'Win Rate']:.1f}%")
                    else:
                        c2.metric("Current Month Win Rate", "N/A")
                else:
                    st.warning("Seasonality unavailable for this ticker/time horizon.")

            # --- TAB 4: DNA & HOURLY ---
            with tab4:
                st.subheader("📆 Day & Hour DNA")
                c1, c2 = st.columns(2)

                dna_res = calc_day_of_week_dna(ticker, 250, "Close to Close (Total)")
                if dna_res:
                    cum, stats = dna_res
                    with c1:
                        st.markdown("**Day of Week Performance**")
                        fig_dna = go.Figure()
                        for c in cum.columns:
                            fig_dna.add_trace(go.Scatter(x=cum.index, y=cum[c], name=c))
                        fig_dna.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig_dna, use_container_width=True)
                        st.dataframe(stats.style.background_gradient(subset=["Win Rate"], cmap="RdYlGn"), use_container_width=True)
                else:
                    with c1:
                        st.info("Day-of-week DNA unavailable.")

                hourly_res = calc_intraday_dna(ticker)
                if hourly_res is not None and not hourly_res.empty:
                    with c2:
                        st.markdown("**Intraday (Hourly) Performance**")
                        fig_hr = px.bar(hourly_res, x=hourly_res.index, y="Avg Return", color="Win Rate", color_continuous_scale="RdYlGn")
                        fig_hr.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig_hr, use_container_width=True)
                        st.dataframe(hourly_res.style.format("{:.2f}"), use_container_width=True)
                else:
                    with c2:
                        st.info("Hourly DNA unavailable.")

            # --- TAB 9: CORRELATION & MTF ---
            with tab9:
                st.subheader("🧩 Cross-Asset Intelligence")
                c1, c2 = st.columns([0.4, 0.6])

                with c1:
                    st.markdown("**📡 Multi-Timeframe Radar**")
                    mtf_df = calc_mtf_trend(ticker)

                    def color_trend(val):
                        color = "#00ff00" if val == "BULLISH" else "#ff0000" if val == "BEARISH" else "white"
                        return f"color: {color}; font-weight: bold"

                    if not mtf_df.empty:
                        st.dataframe(mtf_df.style.map(color_trend, subset=["Trend"]), use_container_width=True)
                    else:
                        st.info("MTF data unavailable.")

                with c2:
                    st.markdown("**🔗 Macro Correlation Matrix (180 Days)**")
                    corr_data = calc_correlations(ticker)
                    if corr_data is not None and not corr_data.empty:
                        fig_corr = px.bar(
                            x=corr_data.values,
                            y=corr_data.index,
                            orientation="h",
                            color=corr_data.values,
                            color_continuous_scale="RdBu",
                        )
                        fig_corr.update_layout(template="plotly_dark", height=400, xaxis_title="Correlation Coefficient")
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Correlation data unavailable.")

            # --- TAB 5: DASHBOARD ---
            with tab5:
                last = df.iloc[-1]
                dash_data = {
                    "Metric": ["God Mode Score", "Apex Trend", "Vector Trend", "Gann Trend", "EVWM Momentum", "RVOL"],
                    "Value": [
                        f"{last['GM_Score']:.0f}",
                        "BULL" if last["Apex_Trend"] == 1 else "BEAR",
                        "BULL" if last["DarkVector_Trend"] == 1 else "BEAR",
                        "BULL" if last["Gann_Trend"] == 1 else "BEAR",
                        f"{last['EVWM']:.2f}",
                        f"{last['RVOL']:.1f}x",
                    ],
                }
                st.dataframe(pd.DataFrame(dash_data), use_container_width=True)

            # --- TAB 6: SMC ---
            with tab6:
                st.subheader("🏦 Smart Money Concepts")
                smc = calculate_smc(df)
                fig_smc = go.Figure(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]))

                for ob in smc["order_blocks"]:
                    fig_smc.add_shape(type="rect", x0=ob["x0"], x1=ob["x1"], y0=ob["y0"], y1=ob["y1"], fillcolor=ob["color"], opacity=0.5, line_width=0)
                for fvg in smc["fvgs"]:
                    fig_smc.add_shape(type="rect", x0=fvg["x0"], x1=fvg["x1"], y0=fvg["y0"], y1=fvg["y1"], fillcolor=fvg["color"], opacity=0.5, line_width=0)
                for struct in smc["structures"]:
                    fig_smc.add_shape(type="line", x0=struct["x0"], x1=struct["x1"], y0=struct["y"], y1=struct["y"], line=dict(color=struct["color"], width=1, dash="dot"))
                    fig_smc.add_annotation(
                        x=struct["x1"],
                        y=struct["y"],
                        text=struct["label"],
                        showarrow=False,
                        yshift=10 if struct["color"] == "green" else -10,
                        font=dict(color=struct["color"], size=10),
                    )

                fig_smc.update_layout(height=600, template="plotly_dark", title="SMC Analysis")
                st.plotly_chart(fig_smc, use_container_width=True)

            # --- TAB 7: QUANT ---
            with tab7:
                st.subheader("🔮 Quantitative Forecasting")
                mc = run_monte_carlo(df)
                fig_mc = go.Figure()

                lines = min(50, mc.shape[1])
                for i in range(lines):
                    fig_mc.add_trace(go.Scatter(y=mc[:, i], mode="lines", line=dict(color="rgba(255,255,255,0.05)"), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=np.mean(mc, axis=1), mode="lines", name="Mean", line=dict(color="orange")))

                fig_mc.update_layout(height=500, template="plotly_dark", title="Monte Carlo Forecast (30 Days)")
                st.plotly_chart(fig_mc, use_container_width=True)

            # --- TAB 8: VOLUME PROFILE ---
            with tab8:
                st.subheader("📊 Volume Profile")
                vp, poc = calc_volume_profile(df)
                fig_vp = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.7, 0.3])

                fig_vp.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]), row=1, col=1)
                fig_vp.add_trace(go.Bar(x=vp["Volume"], y=vp["Price"], orientation="h", marker_color="rgba(0,200,255,0.3)"), row=1, col=2)

                if np.isfinite(poc):
                    fig_vp.add_hline(y=poc, line_color="yellow")

                fig_vp.update_layout(height=600, template="plotly_dark", title="Volume Profile (VPVR)")
                st.plotly_chart(fig_vp, use_container_width=True)

            # --- TAB 10: BROADCAST ---
            with tab10:
                st.subheader("📡 Social Command Center")

                tv_interval_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D", "1wk": "W"}
                tv_int = tv_interval_map.get(interval, "D")

                tv_ticker = ticker.replace("-", "") if "BTC" in ticker else ticker

                tv_widget_html = f"""
                <div class="tradingview-widget-container">
                    <div id="tradingview_widget"></div>
                    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                    <script type="text/javascript">
                    new TradingView.widget(
                    {{
                        "width": "100%",
                        "height": 500,
                        "symbol": "{tv_ticker}",
                        "interval": "{tv_int}",
                        "timezone": "Etc/UTC",
                        "theme": "dark",
                        "style": "1",
                        "locale": "en",
                        "toolbar_bg": "#f1f3f6",
                        "enable_publishing": false,
                        "hide_side_toolbar": false,
                        "allow_symbol_change": true,
                        "container_id": "tradingview_widget"
                    }}
                    );
                    </script>
                </div>
                """
                st.components.v1.html(tv_widget_html, height=500)
                st.caption("Drawing Tools are enabled on the left sidebar.")

                st.markdown("---")
                st.markdown("#### 🚀 Broadcast Signal")

                last_r = df.iloc[-1]

                # Macro context
                spy_chg = m_chg.get("S&P 500", 0)
                btc_chg = m_chg.get("Bitcoin", 0)
                dxy_chg = m_chg.get("DXY Index", 0)
                gold_chg = m_chg.get("Gold", 0)

                macro_text = f"🌍 SPY: {spy_chg:+.2f}% | BTC: {btc_chg:+.2f}%\n💵 DXY: {dxy_chg:+.2f}% | 🟡 Gold: {gold_chg:+.2f}%"
                gm_emoji = "🟢" if last_r["GM_Score"] > 0 else "🔴"

                signal_text = (
                    f"🔥 {ticker} ({interval}) TITAN\n\n"
                    f"Price: ${last_r['Close']:.2f}\n"
                    f"{gm_emoji} Titan Score: {last_r['GM_Score']:.0f}/5\n\n"
                    f"Apex: {'🐂 BULL' if last_r['Apex_Trend']==1 else '🐻 BEAR'}\n"
                    f"Vector: {'↗️ BULL' if last_r['DarkVector_Trend']==1 else '↘️ BEAR'}\n"
                    f"Squeeze: {'💥 ON' if bool(last_r['Squeeze_On']) else '💤 OFF'}\n\n"
                    f"📊 RSI: {last_r['RSI']:.1f}\n"
                    f"🔋 Vol: {last_r['RVOL']:.1f}x\n\n"
                    f"{macro_text}\n\n"
                    f"🤖 AI Outlook: {ai_verdict}\n\n"
                    f"#Trading #DarkPool #Titan"
                )

                msg = st.text_area("Message Preview", value=signal_text, height=170)

                uploaded_file = st.file_uploader("Upload Chart Screenshot (Optional but Recommended)", type=["png", "jpg", "jpeg"])

                col_b1, col_b2 = st.columns(2)

                if col_b1.button("Send to Telegram 🚀"):
                    if tg_token and tg_chat:
                        try:
                            timeout = 12

                            if uploaded_file:
                                url_photo = f"https://api.telegram.org/bot{tg_token}/sendPhoto"
                                data_photo = {"chat_id": tg_chat, "caption": f"🔥 Analysis: {ticker}"}
                                files = {"photo": uploaded_file.getvalue()}
                                r1 = requests.post(url_photo, data=data_photo, files=files, timeout=timeout)
                                r1.raise_for_status()

                            url_msg = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                            clean_msg = msg.replace("###", "").strip()

                            max_length = 2000
                            if len(clean_msg) <= max_length:
                                r2 = requests.post(url_msg, data={"chat_id": tg_chat, "text": clean_msg}, timeout=timeout)
                                r2.raise_for_status()
                            else:
                                part = 1
                                for i in range(0, len(clean_msg), max_length):
                                    chunk = clean_msg[i : i + max_length]
                                    r2 = requests.post(url_msg, data={"chat_id": tg_chat, "text": f"(Part {part}) {chunk}"}, timeout=timeout)
                                    r2.raise_for_status()
                                    part += 1

                            st.success("✅ Sent to Telegram (Split into multiple parts to prevent cutoff)!")
                        except Exception as e:
                            st.error(f"Failed: {e}")
                    else:
                        st.warning("⚠️ Enter Telegram Keys in Sidebar.")

                if col_b2.button("Post to X (Twitter)"):
                    encoded_msg = urllib.parse.quote(msg)
                    st.link_button("🐦 Launch Tweet", f"https://twitter.com/intent/tweet?text={encoded_msg}")
```
