# app.py
import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# -----------------------------
# DarkPool UI (DPC CSS)
# -----------------------------
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

.metric-wrap {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 14px;
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


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _annualize_factor(interval: str) -> float:
    # Conservative mapping for common Yahoo intervals
    # 1d: 252 trading days; 1wk: 52; 1mo: 12; intraday assumes 252*6.5h trading with typical bars
    if interval.endswith("d"):
        return 252.0
    if interval.endswith("wk"):
        return 52.0
    if interval.endswith("mo"):
        return 12.0
    if interval.endswith("h"):
        # approximate trading hours/day
        hours = _safe_float(interval.replace("h", ""))
        if not math.isfinite(hours) or hours <= 0:
            return 252.0 * 6.5
        return 252.0 * (6.5 / hours)
    if interval.endswith("m"):
        mins = _safe_float(interval.replace("m", ""))
        if not math.isfinite(mins) or mins <= 0:
            return 252.0 * 6.5 * 60.0
        return 252.0 * (6.5 * 60.0 / mins)
    return 252.0


def _to_utc_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance sometimes returns timezone-aware. Normalize to naive for consistency in Streamlit.
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_convert("UTC").tz_localize(None)
        except Exception:
            pass
    return df


def _flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance can return MultiIndex columns (Ticker, OHLCV) or single.
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # if multiple tickers, keep top-level ticker groups; handled elsewhere
        return df
    # Standardize column names
    df.columns = [str(c).strip().title() for c in df.columns]
    return df


def compute_indicators(px: pd.DataFrame, ema_len: int, sma_len: int, rsi_len: int, atr_len: int) -> pd.DataFrame:
    """
    px: DataFrame with columns: Open, High, Low, Close, Adj Close (optional), Volume
    Returns df with extra columns: EMA, SMA, RSI, ATR, LogRet
    """
    df = px.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float) if "High" in df.columns else close
    low = df["Low"].astype(float) if "Low" in df.columns else close

    # Log returns
    df["LogRet"] = np.log(close).diff()

    # EMA/SMA
    df[f"EMA_{ema_len}"] = close.ewm(span=ema_len, adjust=False).mean()
    df[f"SMA_{sma_len}"] = close.rolling(window=sma_len, min_periods=sma_len).mean()

    # RSI (Wilder-style smoothing approximated via ewm alpha=1/len)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_len, adjust=False, min_periods=rsi_len).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_len, adjust=False, min_periods=rsi_len).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df[f"RSI_{rsi_len}"] = 100.0 - (100.0 / (1.0 + rs))

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df[f"ATR_{atr_len}"] = tr.ewm(alpha=1.0 / atr_len, adjust=False, min_periods=atr_len).mean()

    return df


def risk_metrics(logret: pd.Series, ann_factor: float, var_alpha: float = 0.05) -> Dict[str, float]:
    r = logret.dropna().astype(float)
    out = {
        "AnnVol": float("nan"),
        "AnnRet": float("nan"),
        "Sharpe": float("nan"),
        "VaR": float("nan"),
        "CVaR": float("nan"),
    }
    if len(r) < 10:
        return out

    mu = r.mean()
    sig = r.std(ddof=1)
    out["AnnVol"] = float(sig * math.sqrt(ann_factor))
    out["AnnRet"] = float(mu * ann_factor)

    # Sharpe (risk-free assumed 0 here; user can adjust if desired)
    if sig > 0:
        out["Sharpe"] = float((mu / sig) * math.sqrt(ann_factor))

    # Historical VaR/CVaR on log returns
    q = np.quantile(r.values, var_alpha)
    out["VaR"] = float(q)
    tail = r[r <= q]
    out["CVaR"] = float(tail.mean()) if len(tail) > 0 else float("nan")

    return out


def max_drawdown(close: pd.Series) -> float:
    c = close.dropna().astype(float)
    if len(c) < 2:
        return float("nan")
    peak = c.cummax()
    dd = (c / peak) - 1.0
    return float(dd.min())


def make_price_figure(df: pd.DataFrame, symbol: str, show_ema: bool, show_sma: bool, show_atr: bool,
                      ema_len: int, sma_len: int, atr_len: int) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=f"{symbol} OHLC",
        increasing_line_color="#00FF9A",
        decreasing_line_color="#FF2E63",
        increasing_fillcolor="rgba(0,255,154,0.15)",
        decreasing_fillcolor="rgba(255,46,99,0.15)",
        line_width=1,
    ))

    if show_ema and f"EMA_{ema_len}" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"EMA_{ema_len}"],
            mode="lines",
            name=f"EMA {ema_len}",
            line=dict(color="#7ee7ff", width=1.6),
        ))

    if show_sma and f"SMA_{sma_len}" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"SMA_{sma_len}"],
            mode="lines",
            name=f"SMA {sma_len}",
            line=dict(color="#ffd166", width=1.4, dash="dot"),
        ))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=f"{symbol} — Price Tape", x=0.01),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(rangeslider_visible=False)

    # Optional ATR overlay as a "volatility band" (not on price axis; displayed as annotation)
    if show_atr and f"ATR_{atr_len}" in df.columns:
        last_atr = df[f"ATR_{atr_len}"].dropna()
        if len(last_atr) > 0:
            atr_v = float(last_atr.iloc[-1])
            fig.add_annotation(
                text=f"ATR({atr_len}) ≈ {atr_v:,.4f}",
                xref="paper", yref="paper",
                x=0.99, y=0.02,
                xanchor="right", yanchor="bottom",
                showarrow=False,
                font=dict(color="rgba(0,255,180,0.85)", size=12),
                bgcolor="rgba(0,255,180,0.06)",
                bordercolor="rgba(0,255,180,0.25)",
                borderwidth=1,
            )

    return fig


def make_rsi_figure(df: pd.DataFrame, rsi_len: int) -> go.Figure:
    col = f"RSI_{rsi_len}"
    fig = go.Figure()
    if col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            mode="lines",
            name=f"RSI {rsi_len}",
            line=dict(color="#00FF9A", width=1.5),
        ))
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.20)")
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.20)")

    fig.update_layout(
        template="plotly_dark",
        height=240,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Momentum — RSI", x=0.01),
        xaxis=dict(showgrid=False),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
    )
    return fig


def make_returns_figure(logret: pd.Series, symbol: str) -> go.Figure:
    r = logret.dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r,
        nbinsx=60,
        name=f"{symbol} log returns",
        marker_color="rgba(126,231,255,0.70)",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Microstructure — Return Distribution", x=0.01),
        xaxis=dict(showgrid=False, title="Log return"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Count"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_drawdown_figure(close: pd.Series) -> go.Figure:
    c = close.dropna().astype(float)
    if len(c) == 0:
        c = pd.Series(dtype=float)
    peak = c.cummax()
    dd = (c / peak) - 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode="lines",
        name="Drawdown",
        line=dict(color="#FF2E63", width=1.6),
        fill="tozeroy",
        fillcolor="rgba(255,46,99,0.15)",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Risk — Drawdown", x=0.01),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickformat=".0%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbols: Tuple[str, ...], start: date, end: date, interval: str) -> pd.DataFrame:
    # yfinance end is exclusive-ish; extend by 1 day for daily/weekly/monthly to include end date
    _end = end
    if interval in ("1d", "1wk", "1mo"):
        _end = end + timedelta(days=1)

    try:
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
    except Exception as e:
        raise RuntimeError(f"yfinance download failed: {e}") from e

    if df is None or len(df) == 0:
        raise RuntimeError("No data returned. Check symbol(s), interval, or date range.")

    df = _to_utc_naive_index(df)
    df = _flatten_ohlcv(df)
    return df


def extract_single_ticker(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    # If MultiIndex columns: (Ticker, OHLCV). If single: already OHLCV.
    if isinstance(df.columns, pd.MultiIndex):
        if symbol not in df.columns.get_level_values(0):
            # fallback: attempt case-insensitive match
            tops = list(map(str, df.columns.get_level_values(0).unique()))
            m = {t.upper(): t for t in tops}
            key = symbol.upper()
            if key in m:
                symbol = m[key]
            else:
                raise KeyError(f"Ticker '{symbol}' not found in returned dataset.")
        sub = df[symbol].copy()
        sub.columns = [str(c).strip().title() for c in sub.columns]
        return sub.dropna(how="all")
    else:
        # Single ticker request
        return df.copy().dropna(how="all")


def align_multi_close(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    # Return close series for each symbol aligned on index
    closes = {}
    if isinstance(df.columns, pd.MultiIndex):
        for sym in symbols:
            if sym in df.columns.get_level_values(0):
                c = df[sym]["Close"].rename(sym)
                closes[sym] = c
            else:
                # case-insensitive fallback
                tops = list(map(str, df.columns.get_level_values(0).unique()))
                m = {t.upper(): t for t in tops}
                if sym.upper() in m:
                    real = m[sym.upper()]
                    c = df[real]["Close"].rename(sym)
                    closes[sym] = c
    else:
        # single series
        closes[symbols[0]] = df["Close"].rename(symbols[0])

    out = pd.concat(closes.values(), axis=1).dropna(how="all")
    return out


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="DarkPool Terminal", page_icon="🕳️", layout="wide")
st.markdown(DPC_CSS, unsafe_allow_html=True)

st.title("DarkPool Terminal")
st.caption("Institutional-grade tape view: price, momentum, risk. No fluff. Just signal.")

with st.sidebar:
    st.markdown("### Controls")
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    symbols_raw = st.text_input(
        "Tickers (comma-separated)",
        value="SPY, QQQ",
        help="Example: SPY, QQQ, AAPL, MSFT. Uses Yahoo Finance tickers."
    )
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    if len(symbols) == 0:
        symbols = ["SPY"]

    interval = st.selectbox(
        "Interval",
        options=["1d", "1h", "30m", "15m", "5m", "1wk", "1mo"],
        index=0,
        help="Intraday intervals may have limited lookback depending on Yahoo Finance."
    )

    # Date defaults: 1 year for daily, 60 days for intraday
    today = date.today()
    if interval in ("1h", "30m", "15m", "5m"):
        default_start = today - timedelta(days=60)
    elif interval in ("1wk", "1mo"):
        default_start = today - timedelta(days=365 * 3)
    else:
        default_start = today - timedelta(days=365)

    start = st.date_input("Start", value=default_start)
    end = st.date_input("End", value=today)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Indicators")

    ema_len = st.slider("EMA length", min_value=5, max_value=200, value=21, step=1)
    sma_len = st.slider("SMA length", min_value=5, max_value=300, value=50, step=1)
    rsi_len = st.slider("RSI length", min_value=5, max_value=50, value=14, step=1)
    atr_len = st.slider("ATR length", min_value=5, max_value=50, value=14, step=1)

    show_ema = st.toggle("Show EMA", value=True)
    show_sma = st.toggle("Show SMA", value=True)
    show_rsi = st.toggle("Show RSI", value=True)
    show_atr = st.toggle("Show ATR tag", value=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Risk")
    var_alpha = st.slider("VaR/CVaR alpha", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
    rf = st.number_input("Risk-free (annual, decimal)", value=0.00, step=0.005, format="%.3f",
                         help="Used only for an adjusted Sharpe estimate in the table (optional).")

    reload_btn = st.button("Reload data", use_container_width=True)

# Reload button: clear cache
if reload_btn:
    st.cache_data.clear()

# Sanity checks
if start >= end:
    st.error("Start date must be before end date.")
    st.stop()

ann_factor = _annualize_factor(interval)

# Data load
try:
    raw = load_prices(tuple(symbols), start, end, interval)
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

# Single ticker detailed view uses the first symbol
primary = symbols[0]

try:
    px = extract_single_ticker(raw, primary)
except Exception as e:
    st.error(f"Failed to extract ticker '{primary}': {e}")
    st.stop()

# Ensure required columns exist for candlestick
need = {"Open", "High", "Low", "Close"}
missing = need.difference(set(px.columns))
if missing:
    st.error(f"Missing required columns for {primary}: {sorted(list(missing))}")
    st.stop()

df = compute_indicators(px, ema_len=ema_len, sma_len=sma_len, rsi_len=rsi_len, atr_len=atr_len)

# Metrics
last_close = float(df["Close"].dropna().iloc[-1]) if df["Close"].notna().any() else float("nan")
last_logret = df["LogRet"].dropna()
last_ret = float(last_logret.iloc[-1]) if len(last_logret) else float("nan")
mdd = max_drawdown(df["Close"])
rm = risk_metrics(df["LogRet"], ann_factor=ann_factor, var_alpha=var_alpha)

# Adjust Sharpe by rf if possible
sharpe_adj = float("nan")
r = df["LogRet"].dropna()
if len(r) > 10:
    mu = r.mean() * ann_factor
    sig = r.std(ddof=1) * math.sqrt(ann_factor)
    if sig > 0 and math.isfinite(rf):
        sharpe_adj = float((mu - rf) / sig)

# Header badges
badge_html = f"""
<div>
  <span class="badge">PRIMARY: {primary}</span>
  <span class="badge">INTERVAL: {interval}</span>
  <span class="badge">WINDOW: {start.isoformat()} → {end.isoformat()}</span>
  <span class="badge">BARS: {len(df):,}</span>
</div>
"""
st.markdown(badge_html, unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Tape KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last", f"{last_close:,.2f}" if math.isfinite(last_close) else "—")
c2.metric("Last log ret", f"{last_ret:+.4%}" if math.isfinite(last_ret) else "—")
c3.metric("Ann vol", f"{rm['AnnVol']:.2%}" if math.isfinite(rm["AnnVol"]) else "—")
c4.metric("Max DD", f"{mdd:.2%}" if math.isfinite(mdd) else "—")
c5.metric("Sharpe (adj)", f"{sharpe_adj:.2f}" if math.isfinite(sharpe_adj) else "—")

# Price chart
left, right = st.columns([1.45, 1.0], gap="large")
with left:
    st.plotly_chart(
        make_price_figure(
            df=df,
            symbol=primary,
            show_ema=show_ema,
            show_sma=show_sma,
            show_atr=show_atr,
            ema_len=ema_len,
            sma_len=sma_len,
            atr_len=atr_len,
        ),
        use_container_width=True,
    )
with right:
    if show_rsi:
        st.plotly_chart(make_rsi_figure(df, rsi_len=rsi_len), use_container_width=True)

    st.markdown("#### Risk Tape")
    risk_table = pd.DataFrame(
        {
            "Metric": ["Annualized return (log)", "Annualized vol", "Sharpe (0 rf)", "Sharpe (rf adj)", f"VaR α={var_alpha:.2f}", f"CVaR α={var_alpha:.2f}"],
            "Value": [
                rm["AnnRet"],
                rm["AnnVol"],
                rm["Sharpe"],
                sharpe_adj,
                rm["VaR"],
                rm["CVaR"],
            ],
        }
    )

    def _fmt_metric(row):
        m = row["Metric"]
        v = row["Value"]
        if not math.isfinite(_safe_float(v)):
            return "—"
        if "Sharpe" in m:
            return f"{float(v):.2f}"
        if "VaR" in m or "CVaR" in m or "return" in m:
            return f"{float(v):+.2%}"
        if "vol" in m:
            return f"{float(v):.2%}"
        return f"{float(v):.4f}"

    risk_table["Value"] = risk_table.apply(_fmt_metric, axis=1)
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

# Returns + drawdown
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
cA, cB = st.columns([1.1, 1.1], gap="large")
with cA:
    st.plotly_chart(make_returns_figure(df["LogRet"], symbol=primary), use_container_width=True)
with cB:
    st.plotly_chart(make_drawdown_figure(df["Close"]), use_container_width=True)

# Cross-asset compare (if multiple)
if len(symbols) > 1:
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Cross-Asset Compare")

    closes = align_multi_close(raw, symbols)
    if closes.shape[1] >= 2:
        norm = closes / closes.iloc[0]
        fig = go.Figure()
        colors = ["#00FF9A", "#7ee7ff", "#ffd166", "#FF2E63", "#b39ddb", "#80cbc4", "#ffab91"]
        for i, sym in enumerate(norm.columns):
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm[sym],
                mode="lines",
                name=sym,
                line=dict(width=1.7, color=colors[i % len(colors)]),
            ))
        fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            title=dict(text="Normalized Performance (base=1.0)", x=0.01),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap of log returns
        logrets = np.log(closes).diff()
        corr = logrets.corr()

        heat = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="corr"),
            )
        )
        heat.update_layout(
            template="plotly_dark",
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            title=dict(text="Return Correlation Matrix", x=0.01),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Not enough overlapping data across tickers to compare.")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="small">Notes: Yahoo intraday history can be limited; risk metrics are based on log returns and assume 0 slippage/fees. '
    'Sharpe (rf adj) uses your sidebar risk-free rate.</div>',
    unsafe_allow_html=True,
)
