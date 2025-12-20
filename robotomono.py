# app.py
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


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
# Utilities
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


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbols: Tuple[str, ...], start: date, end: date, interval: str) -> pd.DataFrame:
    _end = end + timedelta(days=1) if interval in ("1d", "1wk", "1mo") else end
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


def ensure_ohlcv(px: pd.DataFrame) -> pd.DataFrame:
    need = {"Open", "High", "Low", "Close", "Volume"}
    missing = need.difference(set(px.columns))
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}")
    px = px.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        px[c] = pd.to_numeric(px[c], errors="coerce")
    return px.dropna(subset=["Open", "High", "Low", "Close"])


def heikin_ashi(px: pd.DataFrame) -> pd.DataFrame:
    """Heikin Ashi transform (for Gann calc option)."""
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


def resample_ohlcv(px: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLCV resample for MTF-like mode (works best for intraday)."""
    o = px["Open"].resample(rule).first()
    h = px["High"].resample(rule).max()
    l = px["Low"].resample(rule).min()
    c = px["Close"].resample(rule).last()
    v = px["Volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    return out.dropna()


# ============================================================
# Core Indicator Building Blocks
# ============================================================
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
    return 100.0 - (100.0 / (1.0 + ratio))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    w = np.arange(1, length + 1, dtype=float)
    denom = w.sum()

    def _wma(x: np.ndarray) -> float:
        return float(np.dot(x, w) / denom)

    return series.rolling(length, min_periods=length).apply(_wma, raw=True)


def hma(series: pd.Series, length: int) -> pd.Series:
    n = int(length)
    n2 = max(int(n / 2), 1)
    ns = max(int(math.sqrt(n)), 1)
    return wma(2.0 * wma(series, n2) - wma(series, n), ns)


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


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


# ============================================================
# DPLR Mk II (Python translation)
# ============================================================
def compute_dplr(df: pd.DataFrame, len_reactor: int, len_mfi: int, vol_mult: float, show_wyckoff: bool) -> pd.DataFrame:
    out = df.copy()
    vol_avg = out["Volume"].rolling(50, min_periods=50).mean()
    vol_std = out["Volume"].rolling(50, min_periods=50).std(ddof=1)
    out["DPLR_vol_avg"] = vol_avg
    out["DPLR_vol_std"] = vol_std

    out["DPLR_obv"] = obv(out["Close"], out["Volume"])
    out["DPLR_mfi"] = mfi(out["High"], out["Low"], out["Close"], out["Volume"], len_mfi)

    # slope(src,len) = (src - src[len]) / len
    out["DPLR_slope_price"] = (out["Close"] - out["Close"].shift(len_reactor)) / float(len_reactor)
    out["DPLR_slope_obv"] = (out["DPLR_obv"] - out["DPLR_obv"].shift(len_reactor)) / float(len_reactor)

    # Block trade
    out["DPLR_is_block"] = out["Volume"] > (vol_avg + (vol_std * vol_mult))

    # Accum / Dist
    out["DPLR_is_accum"] = (out["DPLR_slope_price"] < 0.0) & (out["DPLR_slope_obv"] > 0.0)
    out["DPLR_is_dist"] = (out["DPLR_slope_price"] > 0.0) & (out["DPLR_slope_obv"] < 0.0)

    # Wyckoff extremes
    lowest_low = out["Low"].rolling(20, min_periods=20).min()
    highest_high = out["High"].rolling(20, min_periods=20).max()
    mid = (out["High"] + out["Low"]) / 2.0

    is_spring = show_wyckoff & (out["Low"] <= lowest_low.shift(1)) & (out["Close"] > mid) & (out["Volume"] > vol_avg * 1.1)
    is_upthrust = show_wyckoff & (out["High"] >= highest_high.shift(1)) & (out["Close"] < mid) & (out["Volume"] > vol_avg * 1.1)

    out["DPLR_is_spring"] = is_spring
    out["DPLR_is_upthrust"] = is_upthrust

    # Flow vector / delta_vol
    up_pressure = (out["High"] - out["Open"]) + (out["Close"] - out["Low"])
    down_pressure = (out["Open"] - out["Low"]) + (out["High"] - out["Close"])
    buy_vol_ratio = up_pressure / (up_pressure + down_pressure + 1e-5)
    out["DPLR_delta_vol"] = out["Volume"] * (buy_vol_ratio - 0.5) * 2.0

    # Reactor bar color classification (string label)
    # Priority matches Pine
    cls = np.where(out["DPLR_is_block"], "BLOCK",
          np.where(out["DPLR_is_spring"], "SPRING",
          np.where(out["DPLR_is_upthrust"], "UPTHRUST",
          np.where(out["DPLR_is_accum"], "ACCUM",
          np.where(out["DPLR_is_dist"], "DIST", "STASIS")))))
    out["DPLR_class"] = cls

    return out


# ============================================================
# Apex Trend & Liquidity Master (Python translation)
# ============================================================
@dataclass
class Zone:
    kind: str  # "Supply" or "Demand"
    left: pd.Timestamp
    right: pd.Timestamp
    top: float
    bottom: float


def pivots_centered(series: pd.Series, left_right: int, mode: str) -> pd.Series:
    """Vectorized pivot detection similar to ta.pivothigh/low(high, L, L).
    Returns True at the pivot *center bar*.
    """
    w = 2 * left_right + 1
    if mode == "high":
        extreme = series.rolling(w, center=True, min_periods=w).max()
        is_p = series.eq(extreme)
    else:
        extreme = series.rolling(w, center=True, min_periods=w).min()
        is_p = series.eq(extreme)
    return is_p.fillna(False)


def compute_apex(
    df: pd.DataFrame,
    ma_type: str,
    len_main: int,
    mult: float,
    src_col: str,
    show_liq: bool,
    liq_len: int,
    zone_ext_bars: int,
    filter_liq: bool,
    use_vol: bool,
    use_rsi: bool,
) -> Tuple[pd.DataFrame, List[Zone]]:
    out = df.copy()
    src = out[src_col].astype(float)

    baseline = get_ma(ma_type, src, len_main)
    atr = atr_wilder(out["High"], out["Low"], out["Close"], len_main)
    upper = baseline + atr * mult
    lower = baseline - atr * mult

    out["APX_baseline"] = baseline
    out["APX_atr"] = atr
    out["APX_upper"] = upper
    out["APX_lower"] = lower

    # Trend: breaks set to +/-1; else hold previous -> ffill
    trend_sig = pd.Series(np.nan, index=out.index, dtype=float)
    trend_sig[out["Close"] > upper] = 1.0
    trend_sig[out["Close"] < lower] = -1.0
    trend = trend_sig.ffill().fillna(0.0).astype(int)
    out["APX_trend"] = trend

    vol_ma = out["Volume"].rolling(20, min_periods=20).mean()
    high_vol = out["Volume"] > vol_ma
    out["APX_vol_ma"] = vol_ma
    out["APX_high_vol"] = high_vol

    rsi_val = 100.0 - (100.0 / (1.0 + (out["Close"].diff().clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
                                       / (-out["Close"].diff().clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean().replace(0, np.nan)))))
    out["APX_rsi"] = rsi_val

    cond_vol = (~use_vol) | high_vol
    rsi_ok_buy = (~use_rsi) | (rsi_val < 70)
    rsi_ok_sell = (~use_rsi) | (rsi_val > 30)

    sig_buy = (trend == 1) & (trend.shift(1) != 1) & cond_vol & rsi_ok_buy
    sig_sell = (trend == -1) & (trend.shift(1) != -1) & cond_vol & rsi_ok_sell
    out["APX_sig_buy"] = sig_buy.fillna(False)
    out["APX_sig_sell"] = sig_sell.fillna(False)

    # Liquidity zones (supply/demand)
    zones: List[Zone] = []
    if show_liq and len(out) >= (2 * liq_len + 5):
        ph_center = pivots_centered(out["High"], liq_len, mode="high")
        pl_center = pivots_centered(out["Low"], liq_len, mode="low")

        # Pine draws zone when pivot confirmed; we anchor at pivot bar.
        # Filter by trend at pivot (trend[liq_len] in Pine at bar where pivot is detected).
        # Here, at pivot center index t, "detected" occurs at t+liq_len. trend_at_pivot ~ trend[t].
        trend_at = out["APX_trend"]

        # Iterate pivots to build last few zones with mitigation check later
        pivot_high_idx = out.index[ph_center]
        pivot_low_idx = out.index[pl_center]

        # Keep only last N pivots for plotting
        pivot_high_idx = pivot_high_idx[-20:]
        pivot_low_idx = pivot_low_idx[-20:]

        # Build zones
        for t in pivot_high_idx:
            if filter_liq and trend_at.loc[t] == 1:
                continue
            box_top = float(out.loc[t, "High"])
            box_bot = float(max(out.loc[t, "Open"], out.loc[t, "Close"]))
            zones.append(Zone("Supply", left=t, right=t, top=box_top, bottom=box_bot))

        for t in pivot_low_idx:
            if filter_liq and trend_at.loc[t] == -1:
                continue
            box_bot = float(out.loc[t, "Low"])
            box_top = float(min(out.loc[t, "Open"], out.loc[t, "Close"]))
            zones.append(Zone("Demand", left=t, right=t, top=box_top, bottom=box_bot))

        # Extend zones and mitigate based on subsequent price
        zones = sorted(zones, key=lambda z: z.left)

        # Convert "bars extension" into timestamps using index positions
        idx = out.index
        pos_map = {ts: i for i, ts in enumerate(idx)}

        active_supply: List[Zone] = []
        active_demand: List[Zone] = []

        for z in zones:
            if z.kind == "Supply":
                active_supply.append(z)
                if len(active_supply) > 5:
                    active_supply.pop(0)
            else:
                active_demand.append(z)
                if len(active_demand) > 5:
                    active_demand.pop(0)

        # For plotting, we compute each zone right edge as min(pivot+zone_ext_bars, end),
        # then if mitigated (price crosses), we drop it.
        plotted: List[Zone] = []
        last_i = len(out) - 1

        def right_ts(pivot_ts: pd.Timestamp) -> pd.Timestamp:
            i = pos_map.get(pivot_ts, None)
            if i is None:
                return idx[-1]
            return idx[min(i + zone_ext_bars, last_i)]

        for z in (active_supply + active_demand):
            i0 = pos_map.get(z.left, None)
            if i0 is None:
                continue
            i1 = min(i0 + max(zone_ext_bars, 1), last_i)
            window = out.iloc[i0:i1 + 1]

            if z.kind == "Supply":
                # Pine: if close > box bottom => delete
                if (window["Close"] > z.bottom).any():
                    continue
            else:
                # Demand: if close < box top => delete
                if (window["Close"] < z.top).any():
                    continue

            plotted.append(Zone(z.kind, left=z.left, right=right_ts(z.left), top=z.top, bottom=z.bottom))

        zones = plotted

    return out, zones


# ============================================================
# Gann High Low Activator (Python translation)
# ============================================================
def dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder DMI/ADX approximation (close enough for dashboard/filtering)."""
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


def gann_activator(high: pd.Series, low: pd.Series, close: pd.Series, length: int, use_close: bool) -> Tuple[pd.Series, pd.Series]:
    """State machine: returns activator line and direction (+1 bull, -1 bear)."""
    sma_high = high.rolling(length, min_periods=length).mean()
    sma_low = low.rolling(length, min_periods=length).mean()

    activ = pd.Series(np.nan, index=close.index, dtype=float)
    direction = pd.Series(np.nan, index=close.index, dtype=float)

    d = 1
    for i, ts in enumerate(close.index):
        if i == 0:
            activ.iloc[i] = sma_low.iloc[i]
            direction.iloc[i] = d
            continue

        prev_activ = activ.iloc[i - 1]
        # If SMA not ready, carry forward NaNs
        if not np.isfinite(sma_high.iloc[i]) or not np.isfinite(sma_low.iloc[i]) or not np.isfinite(prev_activ):
            activ.iloc[i] = np.nan
            direction.iloc[i] = direction.iloc[i - 1]
            continue

        if d == 1:
            trigger = close.iloc[i] if use_close else low.iloc[i]
            if trigger < prev_activ:
                d = -1
                activ.iloc[i] = sma_high.iloc[i]
            else:
                activ.iloc[i] = sma_low.iloc[i]
        else:
            trigger = close.iloc[i] if use_close else high.iloc[i]
            if trigger > prev_activ:
                d = 1
                activ.iloc[i] = sma_low.iloc[i]
            else:
                activ.iloc[i] = sma_high.iloc[i]

        direction.iloc[i] = d

    direction = direction.ffill().fillna(1).astype(int)
    return activ, direction


def compute_gann(
    df: pd.DataFrame,
    gann_len: int,
    use_close: bool,
    calc_mode: str,
    use_mtf: bool,
    mtf_rule: str,
    use_ema: bool,
    ema_len: int,
    use_adx: bool,
    adx_thresh: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - out_main: main timeframe df with gann line forward-filled from mtf (if enabled)
      - out_calc: the dataframe used for gann calc (mtf and/or heikin)
    """
    px = df.copy()
    px_calc = px

    if calc_mode.lower().startswith("heikin"):
        px_calc = heikin_ashi(px_calc)

    if use_mtf:
        # mtf_rule e.g. "4H", "1D"
        px_calc = resample_ohlcv(px_calc, mtf_rule)

    # compute gann on calc data
    g_val, g_dir = gann_activator(px_calc["High"], px_calc["Low"], px_calc["Close"], gann_len, use_close)
    px_calc = px_calc.copy()
    px_calc["GANN_val"] = g_val
    px_calc["GANN_trend"] = g_dir

    # Filters on main timeframe
    ema_line = px["Close"].ewm(span=ema_len, adjust=False, min_periods=ema_len).mean()
    _, _, adx_val = dmi_adx(px["High"], px["Low"], px["Close"], length=14)
    is_chop = use_adx & (adx_val < adx_thresh)

    # Map MTF gann back to main timeframe by forward-fill reindex
    if use_mtf:
        g_val_main = px_calc["GANN_val"].reindex(px.index, method="ffill")
        g_dir_main = px_calc["GANN_trend"].reindex(px.index, method="ffill")
    else:
        g_val_main = px_calc["GANN_val"].reindex(px.index)
        g_dir_main = px_calc["GANN_trend"].reindex(px.index)

    out = px.copy()
    out["GANN_val"] = g_val_main
    out["GANN_trend"] = g_dir_main.astype("Int64")
    out["GANN_ema"] = ema_line
    out["GANN_adx"] = adx_val
    out["GANN_is_chop"] = is_chop

    trend_changed = out["GANN_trend"].ne(out["GANN_trend"].shift(1))
    is_bull = out["GANN_trend"] == 1
    is_bear = out["GANN_trend"] == -1

    ema_pass = (~use_ema) | ((is_bull & (out["Close"] > ema_line)) | (is_bear & (out["Close"] < ema_line)))
    adx_pass = (~use_adx) | (~is_chop)

    out["GANN_buy"] = (trend_changed & is_bull & ema_pass & adx_pass).fillna(False)
    out["GANN_sell"] = (trend_changed & is_bear & ema_pass & adx_pass).fillna(False)

    return out, px_calc


# ============================================================
# Plotly Builders
# ============================================================
COL_BULL_NEON = "#00f2ff"  # DPLR bull neon
COL_BEAR_NEON = "#ff0055"  # DPLR bear neon
COL_ACC_GOLD = "#ffe600"   # DPLR block

APX_BULL = "#00E676"
APX_BEAR = "#FF1744"
NEUT = "#78909C"


def base_layout(title: str, height: int) -> Dict:
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


def fig_price_master(df: pd.DataFrame, symbol: str, dplr: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=f"{symbol} OHLC",
        increasing_line_color="#00FF9A",
        decreasing_line_color="#FF2E63",
        increasing_fillcolor="rgba(0,255,154,0.15)",
        decreasing_fillcolor="rgba(255,46,99,0.15)",
        line_width=1,
    ))

    # DPLR events overlays
    spring = dplr["DPLR_is_spring"].fillna(False)
    upth = dplr["DPLR_is_upthrust"].fillna(False)
    block = dplr["DPLR_is_block"].fillna(False) & (~spring) & (~upth)

    # plot markers near highs/lows to mimic Pine shapes
    if spring.any():
        fig.add_trace(go.Scatter(
            x=df.index[spring],
            y=df["Low"][spring] * (1 - 0.002),
            mode="markers",
            name="Wyckoff Spring",
            marker=dict(symbol="triangle-up", size=10, color=COL_BULL_NEON, line=dict(width=0)),
        ))
    if upth.any():
        fig.add_trace(go.Scatter(
            x=df.index[upth],
            y=df["High"][upth] * (1 + 0.002),
            mode="markers",
            name="Wyckoff Upthrust",
            marker=dict(symbol="triangle-down", size=10, color=COL_BEAR_NEON, line=dict(width=0)),
        ))
    if block.any():
        fig.add_trace(go.Scatter(
            x=df.index[block],
            y=df["High"][block] * (1 + 0.002),
            mode="markers",
            name="Block Trade",
            marker=dict(symbol="diamond", size=9, color=COL_ACC_GOLD, line=dict(width=0)),
        ))

    fig.update_layout(**base_layout(f"{symbol} — Price Master + DPLR Events", 540))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def fig_dplr_reactor(dplr: pd.DataFrame, symbol: str, len_mfi: int) -> go.Figure:
    fig = go.Figure()

    # MFI
    fig.add_trace(go.Scatter(
        x=dplr.index, y=dplr["DPLR_mfi"],
        mode="lines", name=f"MFI {len_mfi}",
        line=dict(color="#7ee7ff", width=1.6),
    ))
    fig.add_hline(y=80, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.20)")
    fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.20)")

    # Delta volume as bars on secondary axis
    dv = dplr["DPLR_delta_vol"].fillna(0.0)
    fig.add_trace(go.Bar(
        x=dplr.index, y=dv,
        name="Net Delta (synthetic)",
        marker_color=np.where(dv >= 0, "rgba(0,242,255,0.35)", "rgba(255,0,85,0.35)"),
        yaxis="y2",
        opacity=0.9,
    ))

    fig.update_layout(**base_layout(f"{symbol} — DPLR Reactor (MFI + Net Delta)", 420))
    fig.update_layout(
        yaxis=dict(title="MFI", range=[0, 100]),
        yaxis2=dict(
            title="DeltaVol",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.15)",
        ),
    )
    return fig


def fig_ghost_profile(df: pd.DataFrame, symbol: str, lookback: int, rows: int) -> go.Figure:
    """Volume profile approximation: price bins weighted by volume over lookback bars."""
    w = df.tail(max(lookback, 10)).copy()
    if len(w) < 10:
        return go.Figure(layout=base_layout(f"{symbol} — Ghost Profile (insufficient data)", 360))

    hi = float(w["High"].max())
    lo = float(w["Low"].min())
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        return go.Figure(layout=base_layout(f"{symbol} — Ghost Profile (bad range)", 360))

    edges = np.linspace(lo, hi, rows + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0

    # bucket by close (matching your Pine implementation)
    close = w["Close"].astype(float).values
    vol = w["Volume"].astype(float).values
    idx = np.clip(np.digitize(close, edges) - 1, 0, rows - 1)
    buckets = np.zeros(rows, dtype=float)
    np.add.at(buckets, idx, vol)

    max_v = buckets.max() if buckets.max() > 0 else 1.0

    fig = go.Figure()
    # horizontal bars: volume by price bucket
    colors = np.where(buckets == buckets.max(), "rgba(255,230,0,0.75)", "rgba(160,160,160,0.35)")
    fig.add_trace(go.Bar(
        x=buckets / max_v,
        y=mids,
        orientation="h",
        name="Profile",
        marker_color=colors,
        hovertemplate="Price=%{y:.2f}<br>RelVol=%{x:.3f}<extra></extra>",
    ))

    fig.update_layout(**base_layout(f"{symbol} — Ghost Profile (Volume Profile)", 420))
    fig.update_layout(
        xaxis=dict(title="Relative Volume (normalized)", showgrid=False, range=[0, 1.05]),
        yaxis=dict(title="Price", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    )
    return fig


def fig_apex(apx: pd.DataFrame, symbol: str, zones: List[Zone], c_bull: str, c_bear: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=apx.index,
        open=apx["Open"], high=apx["High"], low=apx["Low"], close=apx["Close"],
        name="OHLC",
        increasing_line_color="#00FF9A",
        decreasing_line_color="#FF2E63",
        increasing_fillcolor="rgba(0,255,154,0.12)",
        decreasing_fillcolor="rgba(255,46,99,0.12)",
        line_width=1,
    ))

    # Trend cloud fill between upper/lower
    upper = apx["APX_upper"]
    lower = apx["APX_lower"]
    trend = apx["APX_trend"]
    col = np.where(trend.values == 1, c_bull, c_bear)

    # To make a filled cloud in Plotly: draw upper then lower reversed
    fig.add_trace(go.Scatter(
        x=apx.index, y=upper,
        mode="lines", name="Upper",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=apx.index, y=lower,
        mode="lines", name="Lower",
        fill="tonexty",
        fillcolor="rgba(0,230,118,0.14)" if (trend.iloc[-1] == 1) else "rgba(255,23,68,0.14)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Signals
    buy = apx["APX_sig_buy"].fillna(False)
    sell = apx["APX_sig_sell"].fillna(False)
    if buy.any():
        fig.add_trace(go.Scatter(
            x=apx.index[buy],
            y=apx["Low"][buy] * (1 - 0.003),
            mode="markers+text",
            name="BUY",
            marker=dict(symbol="triangle-up", size=11, color=c_bull),
            text=["BUY"] * int(buy.sum()),
            textposition="bottom center",
            textfont=dict(color="rgba(255,255,255,0.85)", size=10),
        ))
    if sell.any():
        fig.add_trace(go.Scatter(
            x=apx.index[sell],
            y=apx["High"][sell] * (1 + 0.003),
            mode="markers+text",
            name="SELL",
            marker=dict(symbol="triangle-down", size=11, color=c_bear),
            text=["SELL"] * int(sell.sum()),
            textposition="top center",
            textfont=dict(color="rgba(255,255,255,0.85)", size=10),
        ))

    # Liquidity zones as rectangles
    shapes = []
    for z in zones:
        if not (np.isfinite(z.top) and np.isfinite(z.bottom)):
            continue
        fill = "rgba(255,23,68,0.15)" if z.kind == "Supply" else "rgba(0,230,118,0.15)"
        line = "rgba(255,23,68,0.35)" if z.kind == "Supply" else "rgba(0,230,118,0.35)"
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="y",
            x0=z.left,
            x1=z.right,
            y0=z.bottom,
            y1=z.top,
            line=dict(color=line, width=1),
            fillcolor=fill,
            layer="below",
        ))

    fig.update_layout(**base_layout(f"{symbol} — Apex Trend & Liquidity (Cloud + Zones + Signals)", 560))
    fig.update_layout(shapes=shapes)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def fig_gann(gann: pd.DataFrame, symbol: str, col_bull: str, col_bear: str, col_neut: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=gann.index,
        open=gann["Open"], high=gann["High"], low=gann["Low"], close=gann["Close"],
        name="OHLC",
        increasing_line_color="#00FF9A",
        decreasing_line_color="#FF2E63",
        increasing_fillcolor="rgba(0,255,154,0.10)",
        decreasing_fillcolor="rgba(255,46,99,0.10)",
        line_width=1,
    ))

    trend = gann["GANN_trend"].astype("Int64")
    is_chop = gann["GANN_is_chop"].fillna(False)
    color_line = np.where(is_chop.values, col_neut, np.where(trend.values == 1, col_bull, col_bear))

    # Gann line
    fig.add_trace(go.Scatter(
        x=gann.index,
        y=gann["GANN_val"],
        mode="lines",
        name="Gann Activator",
        line=dict(color="rgba(255,255,255,0.85)", width=2),
    ))

    # Cloud fill between price and gann (approx)
    fig.add_trace(go.Scatter(
        x=gann.index,
        y=gann["Close"],
        mode="lines",
        name="Close (for cloud)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=gann.index,
        y=gann["GANN_val"],
        mode="lines",
        name="Cloud",
        fill="tonexty",
        fillcolor="rgba(0,230,118,0.12)" if (trend.iloc[-1] == 1) else "rgba(255,82,82,0.12)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # EMA filter line (if enabled) will be plotted regardless; visibility controlled in UI by setting NaNs
    if "GANN_ema" in gann.columns:
        fig.add_trace(go.Scatter(
            x=gann.index, y=gann["GANN_ema"],
            mode="lines", name="EMA Filter",
            line=dict(color="rgba(200,200,200,0.35)", width=1.2, dash="dot"),
        ))

    # Signals
    buy = gann["GANN_buy"].fillna(False)
    sell = gann["GANN_sell"].fillna(False)
    if buy.any():
        fig.add_trace(go.Scatter(
            x=gann.index[buy],
            y=gann["Low"][buy] * (1 - 0.003),
            mode="markers+text",
            name="BUY",
            marker=dict(symbol="label-up", size=16, color=col_bull),
            text=["BUY"] * int(buy.sum()),
            textfont=dict(color="white", size=9),
            textposition="middle center",
        ))
    if sell.any():
        fig.add_trace(go.Scatter(
            x=gann.index[sell],
            y=gann["High"][sell] * (1 + 0.003),
            mode="markers+text",
            name="SELL",
            marker=dict(symbol="label-down", size=16, color=col_bear),
            text=["SELL"] * int(sell.sum()),
            textfont=dict(color="white", size=9),
            textposition="middle center",
        ))

    fig.update_layout(**base_layout(f"{symbol} — Gann High/Low Activator (Line + Cloud + Labels)", 560))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# ============================================================
# Streamlit App
# ============================================================
st.set_page_config(page_title="DarkPool Terminal — Indicator Stack", page_icon="🕳️", layout="wide")
st.markdown(DPC_CSS, unsafe_allow_html=True)

st.title("DarkPool Terminal — Indicator Stack (DPLR + Apex + Gann)")
st.caption("Pine logic translated into pandas. Each module gets its own Plotly panel.")


with st.sidebar:
    st.markdown("### Data")
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    symbols_raw = st.text_input("Tickers (comma-separated)", value="SPY", help="Yahoo Finance tickers.")
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    if not symbols:
        symbols = ["SPY"]
    primary = symbols[0]

    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m", "1wk", "1mo"], index=0)

    today = date.today()
    if interval in ("1h", "30m", "15m", "5m"):
        default_start = today - timedelta(days=60)
    elif interval in ("1wk", "1mo"):
        default_start = today - timedelta(days=365 * 3)
    else:
        default_start = today - timedelta(days=365)

    start = st.date_input("Start", value=default_start)
    end = st.date_input("End", value=today)

    reload_btn = st.button("Reload data", use_container_width=True)
    if reload_btn:
        st.cache_data.clear()

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### DPLR Mk II")
    len_reactor = st.slider("Liquidity Lookback (slope)", 5, 100, 20, 1)
    len_mfi = st.slider("MFI Length", 5, 50, 14, 1)
    vol_mult = st.slider("Block Trade Size (xStd)", 0.5, 5.0, 2.0, 0.1)
    show_wyckoff = st.toggle("Show Wyckoff Events", value=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Ghost Profile")
    show_vp = st.toggle("Show Ghost Profile", value=True)
    vp_lookback = st.slider("Profile Depth (bars)", 50, 600, 100, 10)
    vp_rows = st.slider("Profile Resolution (rows)", 20, 150, 50, 5)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Apex Trend & Liquidity")
    ma_type = st.selectbox("Trend Algorithm", ["HMA", "EMA", "SMA", "RMA"], index=0)
    len_main = st.slider("Trend Length", 10, 200, 55, 1)
    mult = st.slider("Volatility Multiplier (ATR)", 0.5, 5.0, 1.5, 0.1)
    src_col = st.selectbox("Source", ["Close", "Open", "High", "Low"], index=0)

    show_liq = st.toggle("Show Smart Liquidity Zones", value=True)
    liq_len = st.slider("Pivot Lookback", 2, 30, 10, 1)
    zone_ext = st.slider("Zone Extension (bars)", 2, 50, 10, 1)
    filter_liq = st.toggle("Filter Zones by Trend?", value=False)

    use_vol = st.toggle("Volume Filter (signals)", value=True)
    use_rsi = st.toggle("RSI Filter (signals)", value=False)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Gann High/Low Activator")
    gann_len = st.slider("Gann Length", 1, 20, 3, 1)
    gann_use_close = st.toggle("Use Close Price", value=True)
    gann_mode = st.selectbox("Calculation Mode", ["Standard", "Heikin Ashi"], index=0)

    use_mtf = st.toggle("Enable MTF Mode", value=False)
    mtf_rule = st.selectbox("Higher Timeframe (resample rule)", ["4H", "1D", "1W"], index=0, disabled=not use_mtf)

    use_ema = st.toggle("Enable EMA Filter", value=False)
    ema_len = st.slider("EMA Length", 10, 400, 200, 5, disabled=not use_ema)

    use_adx = st.toggle("Enable ADX Filter (Anti-Chop)", value=False)
    adx_thresh = st.slider("ADX Threshold", 5, 40, 20, 1, disabled=not use_adx)


# ----------------------------
# Validate + Load
# ----------------------------
if start >= end:
    st.error("Start date must be before end date.")
    st.stop()

try:
    raw = load_prices(tuple(symbols), start, end, interval)
    px = extract_single_ticker(raw, primary)
    px = ensure_ohlcv(px)
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

ann_factor = _annualize_factor(interval)

badge_html = f"""
<div>
  <span class="badge">PRIMARY: {primary}</span>
  <span class="badge">INTERVAL: {interval}</span>
  <span class="badge">WINDOW: {start.isoformat()} → {end.isoformat()}</span>
  <span class="badge">BARS: {len(px):,}</span>
  <span class="badge">ANN_FACTOR≈ {ann_factor:.1f}</span>
</div>
"""
st.markdown(badge_html, unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ----------------------------
# Compute all modules
# ----------------------------
dplr = compute_dplr(px, len_reactor=len_reactor, len_mfi=len_mfi, vol_mult=vol_mult, show_wyckoff=show_wyckoff)

apx, zones = compute_apex(
    px,
    ma_type=ma_type,
    len_main=len_main,
    mult=mult,
    src_col=src_col,
    show_liq=show_liq,
    liq_len=liq_len,
    zone_ext_bars=zone_ext,
    filter_liq=filter_liq,
    use_vol=use_vol,
    use_rsi=use_rsi,
)

gann, gann_calc = compute_gann(
    px,
    gann_len=gann_len,
    use_close=gann_use_close,
    calc_mode=gann_mode,
    use_mtf=use_mtf,
    mtf_rule=mtf_rule,
    use_ema=use_ema,
    ema_len=ema_len if use_ema else 200,
    use_adx=use_adx,
    adx_thresh=float(adx_thresh) if use_adx else 20.0,
)

# ----------------------------
# KPIs (tight, tape-style)
# ----------------------------
last = float(px["Close"].dropna().iloc[-1]) if px["Close"].notna().any() else float("nan")
logret = np.log(px["Close"]).diff()
ann_vol = float(logret.dropna().std(ddof=1) * math.sqrt(ann_factor)) if logret.dropna().shape[0] > 10 else float("nan")
mfi_last = float(dplr["DPLR_mfi"].dropna().iloc[-1]) if dplr["DPLR_mfi"].notna().any() else float("nan")
delta_last = float(dplr["DPLR_delta_vol"].dropna().iloc[-1]) if dplr["DPLR_delta_vol"].notna().any() else float("nan")
apx_tr = int(apx["APX_trend"].iloc[-1]) if len(apx) else 0
g_tr = int(gann["GANN_trend"].dropna().iloc[-1]) if gann["GANN_trend"].notna().any() else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Last", f"{last:,.2f}" if np.isfinite(last) else "—")
c2.metric("Ann Vol", f"{ann_vol:.2%}" if np.isfinite(ann_vol) else "—")
c3.metric("DPLR MFI", f"{mfi_last:.1f}" if np.isfinite(mfi_last) else "—")
c4.metric("DPLR ΔVol", f"{delta_last:,.0f}" if np.isfinite(delta_last) else "—")
c5.metric("Apex Trend", "BULL" if apx_tr == 1 else "BEAR" if apx_tr == -1 else "HOLD")
c6.metric("Gann Trend", "BULL" if g_tr == 1 else "BEAR" if g_tr == -1 else "—")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ----------------------------
# Panels (each indicator gets Plotly)
# ----------------------------
st.plotly_chart(fig_price_master(px, primary, dplr), use_container_width=True)

row1a, row1b = st.columns([1.2, 1.0], gap="large")
with row1a:
    st.plotly_chart(fig_dplr_reactor(dplr, primary, len_mfi=len_mfi), use_container_width=True)
with row1b:
    if show_vp:
        st.plotly_chart(fig_ghost_profile(px, primary, lookback=vp_lookback, rows=vp_rows), use_container_width=True)
    else:
        st.info("Ghost Profile disabled in sidebar.")

st.plotly_chart(fig_apex(apx, primary, zones, c_bull=APX_BULL, c_bear=APX_BEAR), use_container_width=True)
st.plotly_chart(fig_gann(gann, primary, col_bull="#00E676", col_bear="#FF5252", col_neut=NEUT), use_container_width=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="small">'
    'Notes: '
    'DPLR delta-vol is a synthetic candle-geometry proxy (not true orderflow). '
    'Apex liquidity zones use pivot-window detection; mitigation approximates your Pine logic. '
    'Gann activator is a state machine (looped) to match Pine behavior. '
    'MTF mode uses pandas resampling and forward-filling back to the main timeframe.'
    '</div>',
    unsafe_allow_html=True,
)

