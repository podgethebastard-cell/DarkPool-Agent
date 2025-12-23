import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import openai
import urllib.parse
import time
import os
from pathlib import Path
import json
import traceback

# --- 1. SETUP PAGE CONFIGURATION (Mobile Template) ---
st.set_page_config(
    page_title="Macro Mobile",
    page_icon="🦅",
    layout="centered",  # Mobile friendly
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS (Merged Mobile + Macro Styling) ---
custom_style = """
    <style>
    /* --- MOBILE TEMPLATE STYLES --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Increase button size for touch targets */
    div.stButton > button:first-child {
        width: 100%;
        height: 3em;
        font-weight: bold;
        border-radius: 20px;
    }

    /* --- MACRO INSIGHTER STYLES --- */
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 12px; /* Slightly rounder for mobile */
        transition: transform 0.2s;
    }

    /* Description Text */
    .metric-desc {
        font-size: 0.8rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 10px;
        font-style: italic;
    }

    /* Radio Button Styling */
    .stRadio > label {
        font-weight: bold;
    }

    /* Tiny badge pills */
    .pill {
        display: inline-block;
        padding: 0.12rem 0.5rem;
        border-radius: 999px;
        font-size: 0.75rem;
        border: 1px solid #333;
        background: rgba(255,255,255,0.06);
        color: #ddd;
        margin-right: 0.35rem;
        margin-top: 0.25rem;
    }
    .pill-good { border-color: rgba(0,255,0,0.35); }
    .pill-bad { border-color: rgba(255,75,75,0.55); }
    .pill-warn { border-color: rgba(255,193,7,0.55); }
    </style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# --- 3. DATA MAPPING (The Brain) ---

# Structure: "Label": ("Ticker", "Description")
TICKERS = {
    "✅ MASTER CORE": {
        "S&P 500": ("^GSPC", "US Large Cap Benchmark"),
        "Nasdaq 100": ("^NDX", "Tech & Growth Core"),
        "DXY": ("DX-Y.NYB", "Global Liquidity Engine"),
        "US 10Y": ("^TNX", "Global Asset Pricing Anchor"),
        "US 02Y": ("^IRX", "Fed Policy Sensitivity"),
        "VIX": ("^VIX", "Fear & Volatility Index"),
        "WTI Crude": ("CL=F", "Industrial Energy Demand"),
        "Gold": ("GC=F", "Real Money / Inflation Hedge"),
        "Copper": ("HG=F", "Global Growth Proxy (Dr. Copper)"),
        "HYG (Junk)": ("HYG", "Credit Risk Appetite"),
        "TLT (Long Bond)": ("TLT", "Duration / Recession Hedge"),
        "Bitcoin": ("BTC-USD", "Digital Liquidity Sponge"),
        "Ethereum": ("ETH-USD", "Web3 / Tech Platform Risk")
    },
    "✅ Global Equity Indices": {
        "S&P 500": ("^GSPC", "US Risk-On Core"),
        "Nasdaq 100": ("^NDX", "US Tech/Growth"),
        "Dow Jones": ("^DJI", "US Industrial/Value"),
        "Russell 2000": ("^RUT", "US Small Caps / Domestic Econ"),
        "DAX (DE)": ("^GDAXI", "Europe Industrial Core"),
        "FTSE (UK)": ("^FTSE", "UK/Global Banks & Energy"),
        "CAC (FR)": ("^FCHI", "French Luxury/Consumer"),
        "STOXX50": ("^STOXX50E", "Eurozone Blue Chips"),
        "Nikkei (JP)": ("^N225", "Japan Exporters / YCC Play"),
        "Hang Seng (HK)": ("^HSI", "China Tech / Real Estate"),
        "Shanghai": ("000001.SS", "China Mainland Economy"),
        "KOSPI": ("^KS11", "Korean Tech / Chips"),
        "ACWI": ("ACWI", "All Country World Index"),
        "VT (World)": ("VT", "Total World Stock Market"),
        "EEM (Emerging)": ("EEM", "Emerging Markets Risk")
    },
    "✅ Volatility & Fear": {
        "VIX": ("^VIX", "S&P 500 Implied Volatility"),
        "VXN (Nasdaq)": ("^VXN", "Tech Sector Volatility"),
        "VXD (Dow)": ("^VXD", "Industrial Volatility"),
        "MOVE Proxy (ICE BofA)": ("MOVE.MX", "Bond Market Volatility (Stress)")
    },
    "✅ Interest Rates": {
        "US 10Y": ("^TNX", "Benchmark Long Rate"),
        "US 02Y": ("^IRX", "Fed Policy Expectations"),
        "US 30Y": ("^TYX", "Long Duration / Inflation Exp"),
        "US 05Y": ("^FVX", "Medium Term Rates"),
        "TLT": ("TLT", "20Y+ Treasury Bond ETF"),
        "IEF": ("IEF", "7-10Y Treasury Bond ETF"),
        "SHY": ("SHY", "1-3Y Short Duration Cash"),
        "LQD": ("LQD", "Investment Grade Corporate"),
        "HYG": ("HYG", "High Yield Junk Bonds"),
        "TIP": ("TIP", "Inflation Protected Securities")
    },
    "✅ Currencies": {
        "DXY": ("DX-Y.NYB", "US Dollar vs Major Peers"),
        "EUR/USD": ("EURUSD=X", "Euro Strength"),
        "GBP/USD": ("GBPUSD=X", "British Pound / Risk"),
        "USD/JPY": ("USDJPY=X", "Yen Carry Trade Key"),
        "USD/CNY": ("USDCNY=X", "Yuan / China Export Strength"),
        "AUD/USD": ("AUDUSD=X", "Commodity Currency Proxy"),
        "USD/CHF": ("USDCHF=X", "Swiss Franc Safe Haven"),
        "USD/MXN": ("USDMXN=X", "Emerging Mkt Risk Gauge")
    },
    "✅ Commodities": {
        "WTI": ("CL=F", "US Crude Oil"),
        "Brent": ("BZ=F", "Global Sea-Borne Oil"),
        "NatGas": ("NG=F", "US Heating/Industrial Energy"),
        "Gold": ("GC=F", "Safe Haven / Monetary Metal"),
        "Silver": ("SI=F", "Industrial + Monetary Metal"),
        "Platinum": ("PL=F", "Auto Catalyst / Industrial"),
        "Palladium": ("PA=F", "Tech / Industrial Metal"),
        "Copper": ("HG=F", "Construction / Econ Growth"),
        "Wheat": ("KE=F", "Global Food Supply"),
        "Corn": ("ZC=F", "Feed / Energy / Food"),
        "Soybeans": ("ZS=F", "Global Ag Export Demand")
    },
    "✅ Real Estate": {
        "VNQ (US REITs)": ("VNQ", "US Commercial Real Estate"),
        "REET (Global)": ("REET", "Global Property Market"),
        "XLRE": ("XLRE", "S&P 500 Real Estate Sector")
    },
    "✅ Crypto Macro": {
        "BTC.D (Proxy)": ("BTC-USD", "Bitcoin Dominance Pct"),
        "Total Cap (Proxy)": ("BTC-USD", "Total Crypto Market"),
        "BTC": ("BTC-USD", "Digital Gold / Liquidity"),
        "ETH": ("ETH-USD", "Smart Contract Platform")
    }
}

# Structure: "Label": ("Num_Ticker", "Den_Ticker", "Description")
RATIO_GROUPS = {
    "✅ CRYPTO RELATIVE STRENGTH": {
        "BTC / ETH (Risk Appetite)": ("BTC-USD", "ETH-USD", "Higher = Risk Off / Bitcoin Safety"),
        "BTC / SPX (Adoption)": ("BTC-USD", "^GSPC", "Crypto vs TradFi Correlation"),
        "BTC / NDX (Tech Corr)": ("BTC-USD", "^NDX", "Bitcoin vs Tech Stocks"),
        "ETH / SPX": ("ETH-USD", "^GSPC", "Ethereum Beta to Stocks"),
        "ETH / NDX": ("ETH-USD", "^NDX", "Ethereum vs Nasdaq"),
        "BTC / DXY (Liquidity)": ("BTC-USD", "DX-Y.NYB", "Higher = Liquidity Expansion"),
        "BTC / US10Y (Yields)": ("BTC-USD", "^TNX", "Crypto Sensitivity to Rates"),
        "BTC / VIX (Vol)": ("BTC-USD", "^VIX", "Price vs Fear Index"),
        "BTC / Gold (Hard Money)": ("BTC-USD", "GC=F", "Digital vs Analog Gold")
    },
    "✅ CRYPTO DOMINANCE (Calculated)": {
        "TOTAL 3 / TOTAL": ("SPECIAL_TOTAL3", "SPECIAL_TOTAL", "Altseason Indicator (No BTC/ETH)"),
        "TOTAL 2 / TOTAL": ("SPECIAL_TOTAL2", "SPECIAL_TOTAL", "Alts + ETH Strength"),
        "BTC.D (BTC/Total)": ("BTC-USD", "SPECIAL_TOTAL", "Bitcoin Market Share"),
        "ETH.D (ETH/Total)": ("ETH-USD", "SPECIAL_TOTAL", "Ethereum Market Share"),
        "USDT.D (Tether/Total)": ("USDT-USD", "SPECIAL_TOTAL", "Stablecoin Flight to Safety")
    },
    "✅ EQUITY RISK ROTATION": {
        "SPY / TLT (Risk On/Off)": ("SPY", "TLT", "Rising = Stocks Outperform Bonds"),
        "QQQ / IEF (Growth/Rates)": ("QQQ", "IEF", "Tech vs 7-10Y Treasuries"),
        "XLF / XLU (Fin/Util)": ("XLF", "XLU", "Cyclical vs Defensive"),
        "XLY / XLP (Disc/Staples)": ("XLY", "XLP", "Consumer Confident vs Defensive"),
        "IWM / SPY (Small/Large)": ("IWM", "SPY", "Risk Appetite (Small Caps)"),
        "EEM / SPY (Emerging/US)": ("EEM", "SPY", "Global Growth vs US Exceptionalism"),
        "HYG / TLT (Credit/Safe)": ("HYG", "TLT", "Junk Bond Demand vs Safety"),
        "JNK / TLT": ("JNK", "TLT", "Credit Risk Appetite"),
        "KRE / XLF (Regional/Big)": ("KRE", "XLF", "Bank Stress Indicator"),
        "SMH / SPY (Semi Lead)": ("SMH", "SPY", "Semi-Conductors Leading Market")
    },
    "✅ BOND & YIELD POWER": {
        "10Y / 2Y (Curve)": ("^TNX", "^IRX", "Recession Signal (Inversion)"),
        "10Y / 3M (Recession)": ("^TNX", "^IRX", "Deep Recession Signal"),
        "TLT / SHY (Duration)": ("TLT", "SHY", "Long Duration Demand"),
        "TLT / SPY (Safety/Risk)": ("TLT", "SPY", "Flight to Safety Ratio"),
        "IEF / SHY": ("IEF", "SHY", "Medium vs Short Duration"),
        "MOVE / VIX (Stress)": ("MOVE.MX", "^VIX", "Bond Vol vs Equity Vol")
    },
    "✅ DOLLAR & LIQUIDITY": {
        "DXY / Gold": ("DX-Y.NYB", "GC=F", "Fiat Strength vs Hard Money"),
        "DXY / Oil": ("DX-Y.NYB", "CL=F", "Dollar Purchasing Power (Energy)"),
        "EURUSD / DXY": ("EURUSD=X", "DX-Y.NYB", "Euro Relative Strength"),
        "USDJPY / DXY": ("USDJPY=X", "DX-Y.NYB", "Yen Weakness Isolation"),
        "EEM / DXY": ("EEM", "DX-Y.NYB", "Emerging Market Currency Health"),
    },
    "✅ COMMODITIES & INFLATION": {
        "Gold / Silver": ("GC=F", "SI=F", "Mint Ratio (High = Deflation/Fear)"),
        "Copper / Gold": ("HG=F", "GC=F", "Growth vs Safety (Dr. Copper)"),
        "Oil / Gold": ("CL=F", "GC=F", "Energy Costs vs Monetary Base"),
        "Oil / Copper": ("CL=F", "HG=F", "Energy vs Industrial Demand"),
        "Brent / WTI": ("BZ=F", "CL=F", "Geopolitical Spread")
    },
    "✅ EQUITIES vs REAL ASSETS": {
        "SPX / Gold": ("^GSPC", "GC=F", "Stocks priced in Real Money"),
        "SPX / Copper": ("^GSPC", "HG=F", "Financial vs Real Economy"),
        "SPX / Oil": ("^GSPC", "CL=F", "Stocks vs Energy Costs"),
        "VNQ / SPY (RE/Stocks)": ("VNQ", "SPY", "Real Estate vs Broad Market"),
        "XLE / SPX (Energy/Mkt)": ("XLE", "^GSPC", "Old Economy vs New Economy")
    },
    "✅ TRADE & MACRO STRESS": {
        "XLI / SPX (Ind/Mkt)": ("XLI", "^GSPC", "Industrial Strength"),
        "ITA / SPX (Defense/Mkt)": ("ITA", "^GSPC", "War Premium / Geopolitics"),
        "HYG / JNK (Quality Junk)": ("HYG", "JNK", "High Yield Dispersion")
    }
}

# --- 3B. TRUTH FLAGS / VALIDATION LAYER (No assumptions, just flags) ---
# This DOES NOT change your tickers; it only surfaces potential interpretation hazards.
TRUTH_FLAGS = {
    # NOTE: ^IRX is commonly a 13-week T-Bill yield proxy, not a 2Y. Flag only.
    ("US 02Y", "^IRX"): "Label says 2Y but ticker ^IRX is often a 13-week T-Bill yield proxy.",
    # Crypto macro proxies
    ("BTC.D (Proxy)", "BTC-USD"): "Dominance cannot be derived from BTC price alone; this is a proxy label.",
    ("Total Cap (Proxy)", "BTC-USD"): "Total cap cannot be derived from BTC price alone; this is a proxy label.",
    # Curve ratios
    ("10Y / 3M (Recession)", "^IRX"): "Denominator uses ^IRX again (often 13-week). This may not equal 3M consistently.",
}

# --- SESSION STATE (Favorites / persistent UX) ---
if "fav_tickers" not in st.session_state:
    st.session_state["fav_tickers"] = set()  # keys like "TICKER::S&P 500" or similar
if "fav_ratios" not in st.session_state:
    st.session_state["fav_ratios"] = set()   # keys like "RATIO::SPY / TLT (Risk On/Off)"
if "last_broadcast_pack" not in st.session_state:
    st.session_state["last_broadcast_pack"] = None
if "last_snapshot_date" not in st.session_state:
    st.session_state["last_snapshot_date"] = None

# --- 4. HELPER FUNCTIONS ---

def _now_utc():
    return datetime.now(timezone.utc)

def _safe_key(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_:: " else "_" for ch in s)

@st.cache_data(ttl=120)
def get_market_data(tickers_list, period="1y", interval="1d", max_retries=3):
    """
    Fetches data for a list of tickers safely.
    Returns: (close_df, missing_tickers, fetched_at_utc_iso, raw_shape_note)
    """
    valid_tickers = [t for t in tickers_list if not str(t).startswith("SPECIAL_")]
    valid_tickers = sorted(list(set(valid_tickers)))

    if not valid_tickers:
        return pd.DataFrame(), [], _now_utc().isoformat(), "no_valid_tickers"

    last_exc = None
    for attempt in range(max_retries):
        try:
            # yfinance can return:
            # - DataFrame with OHLC columns for single ticker
            # - MultiIndex columns when multiple tickers
            data = yf.download(
                valid_tickers,
                period=period,
                interval=interval,
                progress=False,
                group_by="ticker",
                auto_adjust=False,
                threads=True
            )

            if data is None or len(data) == 0:
                return pd.DataFrame(), valid_tickers, _now_utc().isoformat(), "empty_download"

            close_df = pd.DataFrame(index=data.index)

            # Single ticker returns columns like ['Open','High','Low','Close'...]
            if isinstance(data.columns, pd.Index) and "Close" in data.columns:
                t = valid_tickers[0]
                close_df[t] = data["Close"]
                raw_shape_note = "single_ticker_frame"

            # Multi ticker returns columns level0=ticker, level1=field OR vice versa
            else:
                raw_shape_note = "multi_ticker_frame"

                # Case 1: level0 is tickers (group_by="ticker" typical)
                if isinstance(data.columns, pd.MultiIndex):
                    lvl0 = list(data.columns.levels[0])
                    lvl1 = list(data.columns.levels[1])

                    # If tickers are level0 and 'Close' exists in level1:
                    if "Close" in lvl1:
                        for t in valid_tickers:
                            if t in lvl0 and ("Close" in data[t].columns):
                                close_df[t] = data[t]["Close"]
                    # If reversed (rare): 'Close' in level0 and tickers in level1:
                    elif "Close" in lvl0:
                        # columns like ('Close', 'AAPL')
                        for t in valid_tickers:
                            if t in lvl1 and ("Close", t) in data.columns:
                                close_df[t] = data[("Close", t)]
                else:
                    # Unexpected shape; best effort
                    raw_shape_note = "unexpected_columns_shape"
                    if "Close" in data:
                        close_df = data[["Close"]].copy()

            # determine missing: tickers not present or all-NaN
            missing = []
            for t in valid_tickers:
                if t not in close_df.columns:
                    missing.append(t)
                else:
                    s = close_df[t]
                    if s.dropna().empty:
                        missing.append(t)

            return close_df, missing, _now_utc().isoformat(), raw_shape_note

        except Exception as e:
            last_exc = e
            # Exponential backoff
            time.sleep(min(2 ** attempt, 8))

    # Failed after retries
    return pd.DataFrame(), valid_tickers, _now_utc().isoformat(), f"failed: {repr(last_exc)}"

def get_crypto_total_proxy(data_df):
    """Constructs synthetic TOTAL, TOTAL2, TOTAL3 indices (price-sum proxy)."""
    coins = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD", "AVAX-USD", "LINK-USD"]
    available = [c for c in coins if c in data_df.columns]

    if not available:
        return None, None, None

    sub_df = data_df[available].ffill()
    total = sub_df.sum(axis=1)

    ex_btc = [c for c in available if c != "BTC-USD"]
    total2 = sub_df[ex_btc].sum(axis=1) if ex_btc else pd.Series(dtype="float64")

    ex_btc_eth = [c for c in available if c not in ["BTC-USD", "ETH-USD"]]
    total3 = sub_df[ex_btc_eth].sum(axis=1) if ex_btc_eth else pd.Series(dtype="float64")

    return total, total2, total3

def calculate_change(series):
    if series is None or len(series) < 2:
        return None, None
    series = series.dropna()
    if len(series) < 2:
        return None, None
    latest = series.iloc[-1]
    prev = series.iloc[-2]
    if prev == 0:
        return latest, 0
    pct = ((latest - prev) / prev) * 100
    return latest, pct

def color_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def plot_sparkline(series, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f"rgba({','.join([str(c) for c in color_to_rgb(color)])}, 0.1)"
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=50, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def zscore(series: pd.Series, window: int = 60):
    s = series.dropna()
    if len(s) < max(10, window // 2):
        return None
    roll = s.rolling(window=window)
    mu = roll.mean()
    sd = roll.std(ddof=0)
    z = (s - mu) / sd.replace(0, pd.NA)
    return z

def pill_html(text, kind="warn"):
    cls = {"good": "pill pill-good", "bad": "pill pill-bad", "warn": "pill pill-warn"}.get(kind, "pill")
    return f"<span class='{cls}'>{text}</span>"

def resolve_series(t, market_data, syn_total, syn_total2, syn_total3):
    """Resolves normal tickers or SPECIAL_* synthetic series."""
    if t == "SPECIAL_TOTAL":
        return syn_total
    if t == "SPECIAL_TOTAL2":
        return syn_total2
    if t == "SPECIAL_TOTAL3":
        return syn_total3
    if t in market_data.columns:
        return market_data[t]
    return None

def compute_ratio_series(num_t, den_t, market_data, syn_total, syn_total2, syn_total3):
    series_num = resolve_series(num_t, market_data, syn_total, syn_total2, syn_total3)
    series_den = resolve_series(den_t, market_data, syn_total, syn_total2, syn_total3)
    if series_num is None or series_den is None:
        return None
    common_idx = series_num.dropna().index.intersection(series_den.dropna().index)
    if common_idx.empty:
        return None
    rs = series_num.loc[common_idx] / series_den.loc[common_idx]
    return rs

def build_regime_dashboard(market_data, syn_total, syn_total2, syn_total3):
    """
    Compact composites based on your existing ratios.
    Returns dict of regime metrics.
    """
    # Core ratios (best effort; missing tolerated)
    r_spy_tlt = compute_ratio_series("SPY", "TLT", market_data, syn_total, syn_total2, syn_total3)
    r_hyg_tlt = compute_ratio_series("HYG", "TLT", market_data, syn_total, syn_total2, syn_total3)
    r_iwm_spy = compute_ratio_series("IWM", "SPY", market_data, syn_total, syn_total2, syn_total3)
    r_cu_au = compute_ratio_series("HG=F", "GC=F", market_data, syn_total, syn_total2, syn_total3)
    r_oil_au = compute_ratio_series("CL=F", "GC=F", market_data, syn_total, syn_total2, syn_total3)
    r_dxy_au = compute_ratio_series("DX-Y.NYB", "GC=F", market_data, syn_total, syn_total2, syn_total3)
    r_btc_dxy = compute_ratio_series("BTC-USD", "DX-Y.NYB", market_data, syn_total, syn_total2, syn_total3)
    r_btc_spx = compute_ratio_series("BTC-USD", "^GSPC", market_data, syn_total, syn_total2, syn_total3)

    def last_pct(rs):
        if rs is None:
            return None, None
        v, p = calculate_change(rs)
        return v, p

    # Scores: sign of daily change as quick regime tilt
    comps = {}
    comps["Risk-On Tilt"] = None
    comps["Inflation Tilt"] = None
    comps["USD Pressure"] = None
    comps["Crypto vs TradFi"] = None

    # Risk: SPY/TLT, HYG/TLT, IWM/SPY
    risk_parts = []
    for rs in [r_spy_tlt, r_hyg_tlt, r_iwm_spy]:
        _, p = last_pct(rs)
        if p is not None:
            risk_parts.append(1 if p > 0 else (-1 if p < 0 else 0))
    if risk_parts:
        comps["Risk-On Tilt"] = sum(risk_parts) / len(risk_parts)

    # Inflation: Copper/Gold, Oil/Gold
    infl_parts = []
    for rs in [r_cu_au, r_oil_au]:
        _, p = last_pct(rs)
        if p is not None:
            infl_parts.append(1 if p > 0 else (-1 if p < 0 else 0))
    if infl_parts:
        comps["Inflation Tilt"] = sum(infl_parts) / len(infl_parts)

    # USD pressure: DXY/Gold up = tighter; BTC/DXY down = tighter
    usd_parts = []
    v, p = last_pct(r_dxy_au)
    if p is not None:
        usd_parts.append(1 if p > 0 else (-1 if p < 0 else 0))
    v, p = last_pct(r_btc_dxy)
    if p is not None:
        usd_parts.append(-1 if p > 0 else (1 if p < 0 else 0))  # inverse interpretation
    if usd_parts:
        comps["USD Pressure"] = sum(usd_parts) / len(usd_parts)

    # Crypto vs TradFi: BTC/SPX up = crypto leadership
    v, p = last_pct(r_btc_spx)
    if p is not None:
        comps["Crypto vs TradFi"] = 1 if p > 0 else (-1 if p < 0 else 0)

    # Add raw values for display (best effort)
    raw = {
        "SPY/TLT": last_pct(r_spy_tlt),
        "HYG/TLT": last_pct(r_hyg_tlt),
        "IWM/SPY": last_pct(r_iwm_spy),
        "Copper/Gold": last_pct(r_cu_au),
        "Oil/Gold": last_pct(r_oil_au),
        "DXY/Gold": last_pct(r_dxy_au),
        "BTC/DXY": last_pct(r_btc_dxy),
        "BTC/SPX": last_pct(r_btc_spx),
    }
    return comps, raw

def generate_ai_analysis(summary_text, cfg):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            return "⚠️ Missing OpenAI API Key in st.secrets."

        client = openai.OpenAI(api_key=api_key)

        focus = []
        if cfg.get("focus_risk"): focus.append("Risk regime (risk-on/off)")
        if cfg.get("focus_inflation"): focus.append("Inflation/deflation impulse")
        if cfg.get("focus_crypto"): focus.append("Crypto structure (dominance/relative strength)")
        if cfg.get("focus_rates"): focus.append("Rates/curve & duration signals")
        focus_txt = ", ".join(focus) if focus else "Full macro read"

        style = cfg.get("style", "Bullet Points")
        style_instr = "Be concise, bullet points." if style == "Bullet Points" else "Write a short memo (6–10 sentences), direct and actionable."

        model = cfg.get("model", "gpt-4o")
        temp = float(cfg.get("temperature", 0.2))

        prompt = f"""
Act as a Global Macro Strategist.

Focus: {focus_txt}

Analyze these key market ratios and changes:
{summary_text}

Deliver:
1) Regime call: Risk-On vs Risk-Off (name the strongest confirms + biggest contradictions)
2) Inflation vs Deflation impulse (what’s leading, what’s lagging)
3) USD/liquidity pressure (tightening vs easing)
4) Crypto: structure + relative strength (if included)

{style_instr}
"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def ensure_snapshots_dir():
    p = Path("snapshots")
    p.mkdir(exist_ok=True)
    return p

def save_snapshot_csv(snapshot_rows, filename_prefix="macro_snapshot"):
    """
    snapshot_rows: list of dicts
    """
    p = ensure_snapshots_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fp = p / f"{filename_prefix}_{ts}.csv"
    df = pd.DataFrame(snapshot_rows)
    df.to_csv(fp, index=False)
    return str(fp)

def build_broadcast_pack(regime_comps, regime_raw, top_movers, timestamp_iso):
    """
    Returns a dict with text templates and a simple card HTML.
    """
    def tilt_word(v):
        if v is None:
            return "Neutral / Unknown"
        if v >= 0.34:
            return "Risk-On"
        if v <= -0.34:
            return "Risk-Off"
        return "Mixed"

    risk = tilt_word(regime_comps.get("Risk-On Tilt"))
    infl = tilt_word(regime_comps.get("Inflation Tilt"))
    usd = tilt_word(regime_comps.get("USD Pressure"))
    crypto = tilt_word(regime_comps.get("Crypto vs TradFi"))

    movers_txt = "\n".join([f"- {m['label']}: {m['pct']:+.2f}%" for m in top_movers[:5]]) if top_movers else "- (no movers available)"

    text = f"""🦅 Macro Mobile — {timestamp_iso}
Regime: {risk} | Inflation: {infl} | USD: {usd} | Crypto: {crypto}

Top movers:
{movers_txt}

#macro #markets #rates #crypto
"""

    # Simple HTML card (not an image, but looks like one)
    card = f"""
<div style="border:1px solid #333;border-radius:16px;padding:14px;background:#111;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="font-size:16px;font-weight:700;">🦅 Macro Mobile</div>
    <div style="font-size:12px;color:#888;">{timestamp_iso}</div>
  </div>
  <div style="margin-top:10px;">
    <span class="pill pill-warn">Regime: {risk}</span>
    <span class="pill pill-warn">Inflation: {infl}</span>
    <span class="pill pill-warn">USD: {usd}</span>
    <span class="pill pill-warn">Crypto: {crypto}</span>
  </div>
  <div style="margin-top:12px;color:#ddd;font-weight:600;">Top Movers</div>
  <div style="margin-top:6px;color:#bbb;white-space:pre-line;font-size:13px;">
{urllib.parse.quote(movers_txt).replace("%0A", "<br/>").replace("%2D", "-")}
  </div>
</div>
"""
    return {
        "timestamp": timestamp_iso,
        "regime": {"risk": risk, "inflation": infl, "usd": usd, "crypto": crypto},
        "top_movers": top_movers[:10],
        "text": text,
        "card_html": card
    }

# --- 5. GLOBAL CONTROLS (Period / Interval / UX) ---
st.markdown("### 🦅 Macro Mobile")

# Guardrails for Yahoo intervals (best effort)
INTERVAL_CHOICES = ["1d", "1h", "30m", "15m", "5m"]
PERIOD_CHOICES = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]

c1, c2 = st.columns(2)
with c1:
    period = st.selectbox("Timeframe (period)", PERIOD_CHOICES, index=3)
with c2:
    interval = st.selectbox("Granularity (interval)", INTERVAL_CHOICES, index=0)

# Guardrail note (no assumptions, just alert)
if interval != "1d" and period not in ["1mo", "3mo", "6mo"]:
    st.info("Intraday intervals often have limited lookback on Yahoo. If data looks thin, reduce timeframe to 1–6 months.")

show_diagnostics = st.toggle("Show Data Diagnostics", value=False)

st.markdown("---")

# --- 6. DATA PREPARATION (Run Once) ---

# Prepare List of All Tickers Needed
all_needed_tickers = set()
for cat in TICKERS.values():
    for t_data in cat.values():
        all_needed_tickers.add(t_data[0])

for cat in RATIO_GROUPS.values():
    for t_data in cat.values():
        all_needed_tickers.add(t_data[0])
        all_needed_tickers.add(t_data[1])

crypto_components = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "USDT-USD", "USDC-USD"]
all_needed_tickers.update(crypto_components)

with st.spinner("Fetching Global Market Data..."):
    market_data, missing_tickers, fetched_at_iso, shape_note = get_market_data(
        list(all_needed_tickers),
        period=period,
        interval=interval
    )
    syn_total, syn_total2, syn_total3 = get_crypto_total_proxy(market_data)

# Coverage & last updated (A2)
loaded = len(market_data.columns) if isinstance(market_data, pd.DataFrame) else 0
requested = len([t for t in all_needed_tickers if not str(t).startswith("SPECIAL_")])
st.caption(f"Last updated (UTC): {fetched_at_iso}  •  Coverage: {loaded}/{requested} tickers  •  Shape: {shape_note}")

if show_diagnostics:
    if missing_tickers:
        st.warning(f"Missing / empty tickers ({len(missing_tickers)}): {', '.join(missing_tickers[:30])}{' ...' if len(missing_tickers) > 30 else ''}")
    if market_data is None or market_data.empty:
        st.error("Market data is empty. Yahoo may be throttling or the selected interval/period is too strict.")

# --- 7. APP NAVIGATION (Mobile Tabs) ---
tab1, tab2, tab3 = st.tabs(["📈 Markets", "➗ Ratios", "🧠 Intel"])

# --- 8. AI SUMMARY BASE (Decoupled from UI filtering) ---
# This ensures AI can work even if user filters down the UI heavily.
AI_CORE_TICKERS = [
    ("S&P 500", "^GSPC"),
    ("Nasdaq 100", "^NDX"),
    ("DXY", "DX-Y.NYB"),
    ("US 10Y", "^TNX"),
    ("VIX", "^VIX"),
    ("Gold", "GC=F"),
    ("Copper", "HG=F"),
    ("WTI", "CL=F"),
    ("HYG", "HYG"),
    ("TLT", "TLT"),
    ("BTC", "BTC-USD"),
    ("ETH", "ETH-USD"),
]

AI_CORE_RATIOS = [
    ("SPY / TLT", "SPY", "TLT"),
    ("HYG / TLT", "HYG", "TLT"),
    ("Copper / Gold", "HG=F", "GC=F"),
    ("Oil / Gold", "CL=F", "GC=F"),
    ("DXY / Gold", "DX-Y.NYB", "GC=F"),
    ("BTC / SPX", "BTC-USD", "^GSPC"),
    ("BTC / DXY", "BTC-USD", "DX-Y.NYB"),
]

def build_ai_summary_base(market_data, syn_total, syn_total2, syn_total3):
    rows = []
    # tickers
    for label, t in AI_CORE_TICKERS:
        if t in market_data.columns:
            val, pct = calculate_change(market_data[t])
            if val is not None and pct is not None:
                rows.append(f"{label}: {pct:+.2f}%")
    # ratios
    for label, n, d in AI_CORE_RATIOS:
        rs = compute_ratio_series(n, d, market_data, syn_total, syn_total2, syn_total3)
        if rs is not None:
            val, pct = calculate_change(rs)
            if val is not None and pct is not None:
                rows.append(f"{label}: {val:.4f} ({pct:+.2f}%)")
    return rows

ai_summary_base = build_ai_summary_base(market_data, syn_total, syn_total2, syn_total3)
data_summary_for_ai = []  # still kept from your foundation loops (union later)

# --- TAB 1: STANDARD TICKERS ---
with tab1:
    # Favorites view injected without altering your mapping structure
    tick_cats = list(TICKERS.keys())
    if len(st.session_state["fav_tickers"]) > 0:
        tick_cats = ["⭐ Favorites"] + tick_cats

    selected_category = st.selectbox("Asset Class", tick_cats)

    search_query = st.text_input("Search (tickers)", value="", placeholder="Type to filter labels...")

    # Build items depending on category
    if selected_category == "⭐ Favorites":
        # Flatten all tickers and filter to favorites
        flat = []
        for cat_name, cat in TICKERS.items():
            for label, (ticker, desc) in cat.items():
                fav_key = f"TICKER::{label}::{ticker}"
                if fav_key in st.session_state["fav_tickers"]:
                    flat.append((label, ticker, desc, cat_name))
        # Create pseudo items dict preserving label->(ticker,desc)
        items = {f"{label}  ·  {cat_name}": (ticker, desc) for (label, ticker, desc, cat_name) in flat}
        if not items:
            st.info("No favorites yet. Tap ⭐ under any metric to add it.")
    else:
        items = TICKERS[selected_category]

    # Apply search filter
    if search_query.strip():
        q = search_query.strip().lower()
        items = {k: v for k, v in items.items() if q in k.lower()}

    cols = st.columns(2)

    for i, (label, (ticker, desc)) in enumerate(items.items()):
        col_idx = i % 2

        with cols[col_idx]:
            # A3 truth flags (no changes, just surfacing)
            flag_msg = TRUTH_FLAGS.get((label, ticker))

            fav_key = f"TICKER::{label}::{ticker}"
            is_fav = fav_key in st.session_state["fav_tickers"]

            if ticker in market_data.columns:
                series = market_data[ticker].dropna()
                val, pct = calculate_change(series)

                if val is not None:
                    data_summary_for_ai.append(f"{label}: {pct:+.2f}%")

                    color = "#00FF00" if pct >= 0 else "#FF4B4B"
                    st.metric(label, f"{val:,.2f}", f"{pct:+.2f}%")
                    st.caption(desc)

                    # Signal layer (C1) using z-score
                    z = zscore(series, window=60)
                    if z is not None and len(z.dropna()) > 0:
                        z_last = float(z.dropna().iloc[-1])
                        if z_last >= 1.5:
                            st.markdown(pill_html(f"Z60 {z_last:+.2f} EXTENDED HIGH", "warn"), unsafe_allow_html=True)
                        elif z_last <= -1.5:
                            st.markdown(pill_html(f"Z60 {z_last:+.2f} EXTENDED LOW", "warn"), unsafe_allow_html=True)
                        else:
                            st.markdown(pill_html(f"Z60 {z_last:+.2f}", "good"), unsafe_allow_html=True)

                    if flag_msg:
                        st.markdown(pill_html("Truth Flag", "warn"), unsafe_allow_html=True)
                        st.caption(f"⚠️ {flag_msg}")

                    st.plotly_chart(
                        plot_sparkline(series, color),
                        use_container_width=True,
                        config={'displayModeBar': False},
                        key=f"std_{_safe_key(label)}_{i}"
                    )
            else:
                st.warning(f"{label}: No Data")

            # Favorites toggle (E1)
            new_fav = st.toggle("⭐ Favorite", value=is_fav, key=f"fav_t_{_safe_key(label)}_{_safe_key(ticker)}")
            if new_fav and not is_fav:
                st.session_state["fav_tickers"].add(fav_key)
            if (not new_fav) and is_fav:
                st.session_state["fav_tickers"].remove(fav_key)

    # Export (F1) — download currently displayed category
    st.markdown("---")
    export_cols = st.columns(2)
    with export_cols[0]:
        if st.button("⬇️ Download This View (CSV)", key="dl_tab1"):
            # Build DF for this view
            rows = []
            for label, (ticker, desc) in items.items():
                if ticker in market_data.columns:
                    s = market_data[ticker].dropna()
                    if not s.empty:
                        rows.append({"label": label, "ticker": ticker, "desc": desc, "latest": s.iloc[-1]})
            df_view = pd.DataFrame(rows)
            st.download_button(
                "Download CSV",
                df_view.to_csv(index=False).encode("utf-8"),
                file_name=f"markets_view_{_safe_key(selected_category)}_{period}_{interval}.csv",
                mime="text/csv",
                key="dl_btn_tab1"
            )
    with export_cols[1]:
        st.caption("Tip: Favorites appear under ⭐ Favorites.")

# --- TAB 2: RATIOS ---
with tab2:
    ratio_cats = list(RATIO_GROUPS.keys())
    if len(st.session_state["fav_ratios"]) > 0:
        ratio_cats = ["⭐ Favorites"] + ratio_cats

    selected_ratio_cat = st.selectbox("Ratio Strategy", ratio_cats)

    search_ratio = st.text_input("Search (ratios)", value="", placeholder="Type to filter ratio labels...")

    if selected_ratio_cat == "⭐ Favorites":
        flat = []
        for cat_name, cat in RATIO_GROUPS.items():
            for label, (num_t, den_t, desc) in cat.items():
                fav_key = f"RATIO::{label}::{num_t}::{den_t}"
                if fav_key in st.session_state["fav_ratios"]:
                    flat.append((label, num_t, den_t, desc, cat_name))
        items = {f"{label}  ·  {cat_name}": (num_t, den_t, desc) for (label, num_t, den_t, desc, cat_name) in flat}
        if not items:
            st.info("No ratio favorites yet. Tap ⭐ under any ratio to add it.")
    else:
        items = RATIO_GROUPS[selected_ratio_cat]

    if search_ratio.strip():
        q = search_ratio.strip().lower()
        items = {k: v for k, v in items.items() if q in k.lower()}

    cols = st.columns(2)

    # We will also build a view DF for export
    ratio_view_rows = []

    for i, (label, (num_t, den_t, desc)) in enumerate(items.items()):
        col_idx = i % 2

        with cols[col_idx]:
            fav_key = f"RATIO::{label}::{num_t}::{den_t}"
            is_fav = fav_key in st.session_state["fav_ratios"]

            # Calculate Ratio (handles SPECIAL_* via resolver)
            ratio_series = compute_ratio_series(num_t, den_t, market_data, syn_total, syn_total2, syn_total3)

            # Truth flags: either endpoint flagged
            flag_msg = None
            # We can try to match your label mapping flags for denominator cases
            if (label, num_t) in TRUTH_FLAGS:
                flag_msg = TRUTH_FLAGS[(label, num_t)]
            if (label, den_t) in TRUTH_FLAGS:
                flag_msg = flag_msg or TRUTH_FLAGS[(label, den_t)]

            if ratio_series is not None:
                val, pct = calculate_change(ratio_series)
                if val is not None:
                    data_summary_for_ai.append(f"{label}: {val:.4f} ({pct:+.2f}%)")
                    ratio_view_rows.append({
                        "label": label,
                        "num": num_t,
                        "den": den_t,
                        "desc": desc,
                        "latest": float(val),
                        "pct": float(pct),
                    })

                    # Color logic preserved (your style)
                    color = "#3498db"
                    if "Risk On" in label or "BTC/SPX" in label:
                        color = "#00FF00" if pct > 0 else "#FF4B4B"

                    st.metric(label, f"{val:.4f}", f"{pct:+.2f}%")
                    st.caption(desc)

                    # Z-score badge (C1)
                    z = zscore(ratio_series, window=60)
                    if z is not None and len(z.dropna()) > 0:
                        z_last = float(z.dropna().iloc[-1])
                        if z_last >= 1.5:
                            st.markdown(pill_html(f"Z60 {z_last:+.2f} EXTENDED HIGH", "warn"), unsafe_allow_html=True)
                        elif z_last <= -1.5:
                            st.markdown(pill_html(f"Z60 {z_last:+.2f} EXTENDED LOW", "warn"), unsafe_allow_html=True)
                        else:
                            st.markdown(pill_html(f"Z60 {z_last:+.2f}", "good"), unsafe_allow_html=True)

                    if flag_msg:
                        st.markdown(pill_html("Truth Flag", "warn"), unsafe_allow_html=True)
                        st.caption(f"⚠️ {flag_msg}")

                    st.plotly_chart(
                        plot_sparkline(ratio_series, color),
                        use_container_width=True,
                        config={'displayModeBar': False},
                        key=f"ratio_{_safe_key(label)}_{i}"
                    )
            else:
                st.info(f"{label}: Insufficient Data")

            # Favorites toggle (E1)
            new_fav = st.toggle("⭐ Favorite", value=is_fav, key=f"fav_r_{_safe_key(label)}_{_safe_key(num_t)}_{_safe_key(den_t)}")
            if new_fav and not is_fav:
                st.session_state["fav_ratios"].add(fav_key)
            if (not new_fav) and is_fav:
                st.session_state["fav_ratios"].remove(fav_key)

    # Export (F1) — download currently displayed ratios
    st.markdown("---")
    df_ratio_view = pd.DataFrame(ratio_view_rows)
    st.download_button(
        "⬇️ Download This Ratio View (CSV)",
        df_ratio_view.to_csv(index=False).encode("utf-8"),
        file_name=f"ratios_view_{_safe_key(selected_ratio_cat)}_{period}_{interval}.csv",
        mime="text/csv",
        key="dl_btn_tab2"
    )

# --- TAB 3: AI & BROADCAST (Tools) ---
with tab3:
    st.header("Institutional Intel")

    # C2: Macro Regime Dashboard (composites)
    regime_comps, regime_raw = build_regime_dashboard(market_data, syn_total, syn_total2, syn_total3)

    # Present dashboard compactly
    d1, d2 = st.columns(2)
    with d1:
        ro = regime_comps.get("Risk-On Tilt")
        label = "Risk-On Tilt"
        if ro is None:
            st.metric(label, "n/a", "n/a")
        else:
            st.metric(label, f"{ro:+.2f}", "↑ Risk-On" if ro > 0 else ("↓ Risk-Off" if ro < 0 else "Flat"))
    with d2:
        it = regime_comps.get("Inflation Tilt")
        label = "Inflation Tilt"
        if it is None:
            st.metric(label, "n/a", "n/a")
        else:
            st.metric(label, f"{it:+.2f}", "↑ Inflation" if it > 0 else ("↓ Disinflation" if it < 0 else "Flat"))

    d3, d4 = st.columns(2)
    with d3:
        up = regime_comps.get("USD Pressure")
        label = "USD Pressure"
        if up is None:
            st.metric(label, "n/a", "n/a")
        else:
            st.metric(label, f"{up:+.2f}", "↑ Tightening" if up > 0 else ("↓ Easing" if up < 0 else "Flat"))
    with d4:
        cvt = regime_comps.get("Crypto vs TradFi")
        label = "Crypto vs TradFi"
        if cvt is None:
            st.metric(label, "n/a", "n/a")
        else:
            st.metric(label, f"{cvt:+.0f}", "↑ Crypto Leads" if cvt > 0 else ("↓ TradFi Leads" if cvt < 0 else "Flat"))

    if show_diagnostics:
        st.caption("Regime raw ratios (latest, daily %):")
        # Show a few key raws
        raw_rows = []
        for k, (v, p) in regime_raw.items():
            if v is not None and p is not None:
                raw_rows.append({"ratio": k, "latest": v, "pct": p})
        st.dataframe(pd.DataFrame(raw_rows))

    st.markdown("---")

    # D1: AI controls
    st.subheader("🧠 AI Report Controls")
    cA, cB = st.columns(2)
    with cA:
        focus_risk = st.checkbox("Risk Regime", value=True)
        focus_inflation = st.checkbox("Inflation/Deflation", value=True)
    with cB:
        focus_crypto = st.checkbox("Crypto", value=True)
        focus_rates = st.checkbox("Rates/Curve", value=True)

    cC, cD = st.columns(2)
    with cC:
        style = st.selectbox("Output Style", ["Bullet Points", "Short Memo"], index=0)
    with cD:
        model = st.text_input("Model", value="gpt-4o")

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    cfg = {
        "focus_risk": focus_risk,
        "focus_inflation": focus_inflation,
        "focus_crypto": focus_crypto,
        "focus_rates": focus_rates,
        "style": style,
        "model": model,
        "temperature": temperature
    }

    # Merge AI summaries (base + whatever user rendered via UI), dedup preserving order
    def dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    ai_summary_all = dedup(ai_summary_base + data_summary_for_ai)

    if st.button("Generate AI Report", type="primary"):
        with st.spinner("Analyzing Market Structure..."):
            if ai_summary_all:
                report = generate_ai_analysis("\n".join(ai_summary_all), cfg)
                st.success("Report Generated")
                st.markdown(report)
            else:
                st.error("No data available for analysis. Check data connection.")

    st.markdown("---")

    # D2: Broadcast Pack (X-Factor-ish) + F2 snapshot logging
    st.subheader("📢 Broadcast Signal")

    # Compute top movers from MASTER CORE shown tickers (best effort)
    movers = []
    try:
        # use MASTER CORE as stable broadcast base
        for label, (ticker, desc) in TICKERS.get("✅ MASTER CORE", {}).items():
            if ticker in market_data.columns:
                s = market_data[ticker].dropna()
                v, p = calculate_change(s)
                if p is not None:
                    movers.append({"label": label, "ticker": ticker, "pct": float(p), "latest": float(v)})
        movers = sorted(movers, key=lambda x: abs(x["pct"]), reverse=True)
    except Exception:
        movers = []

    timestamp_iso = fetched_at_iso

    if st.button("Build Broadcast Pack"):
        pack = build_broadcast_pack(regime_comps, regime_raw, movers, timestamp_iso)
        st.session_state["last_broadcast_pack"] = pack
        st.success("Broadcast Pack Ready")

    pack = st.session_state.get("last_broadcast_pack")
    if pack:
        st.markdown("**Preview Card**")
        st.markdown(pack["card_html"], unsafe_allow_html=True)

        st.markdown("**Post Text (copy/paste)**")
        broadcast_msg = st.text_area("Message Preview", value=pack["text"], height=220)

        encoded_msg = urllib.parse.quote(broadcast_msg)

        col_x, col_tg = st.columns(2)
        with col_x:
            st.link_button("X (Post)", f"https://twitter.com/intent/tweet?text={encoded_msg}")
        with col_tg:
            st.link_button("Telegram", f"https://t.me/share/url?url=&text={encoded_msg}")

        st.download_button(
            "⬇️ Download Broadcast Pack (JSON)",
            data=json.dumps(pack, indent=2).encode("utf-8"),
            file_name=f"broadcast_pack_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # F2 snapshot logging
        st.markdown("---")
        st.subheader("🧾 Snapshot Logging")

        enable_auto_daily = st.toggle("Auto-save once per UTC day (on pack build)", value=False)
        manual_save = st.button("Save Snapshot Now")

        snapshot_rows = []
        # Save regime + movers + AI base summary
        snapshot_rows.append({
            "timestamp_utc": timestamp_iso,
            "type": "regime",
            "risk_on_tilt": regime_comps.get("Risk-On Tilt"),
            "inflation_tilt": regime_comps.get("Inflation Tilt"),
            "usd_pressure": regime_comps.get("USD Pressure"),
            "crypto_vs_tradfi": regime_comps.get("Crypto vs TradFi"),
        })
        for m in movers[:10]:
            snapshot_rows.append({
                "timestamp_utc": timestamp_iso,
                "type": "mover",
                "label": m["label"],
                "ticker": m["ticker"],
                "pct": m["pct"],
                "latest": m["latest"]
            })
        for line in ai_summary_base[:50]:
            snapshot_rows.append({
                "timestamp_utc": timestamp_iso,
                "type": "ai_base",
                "text": line
            })

        def utc_day(iso_str):
            try:
                dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
                return dt.date().isoformat()
            except Exception:
                return datetime.utcnow().date().isoformat()

        today = utc_day(timestamp_iso)

        did_save = False
        if manual_save:
            fp = save_snapshot_csv(snapshot_rows)
            st.success(f"Saved: {fp}")
            did_save = True

        if enable_auto_daily and not did_save:
            last_day = st.session_state.get("last_snapshot_date")
            if last_day != today:
                fp = save_snapshot_csv(snapshot_rows, filename_prefix="macro_daily")
                st.session_state["last_snapshot_date"] = today
                st.success(f"Auto-saved daily snapshot: {fp}")

    st.markdown("---")
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
