# app.py
# Port of: "Apex Trend & Liquidity Master (SMC)v7.2" Pine v6 -> Streamlit (Plotly)
# Hypothetical/educational. Not investment advice.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    import requests
except Exception:
    requests = None


# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Apex Trend & Liquidity Master — Pine Port", layout="wide")

DARK_BG = "#0b1220"
FG = "#e6edf3"
MUTED = "#9fb2c8"
ACCENT = "#4ea1ff"
GOOD = "#00E676"
BAD = "#FF1744"


# ----------------------------
# Pine-style helpers
# ----------------------------
def nz(x: pd.Series, y=0.0) -> pd.Series:
    return x.fillna(y) if isinstance(x, pd.Series) else (y if pd.isna(x) else x)

def rma(x: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / float(length)
    return x.ewm(alpha=alpha, adjust=False).mean()

def sma(x: pd.Series, length: int) -> pd.Series:
    return x.rolling(length, min_periods=length).mean()

def ema(x: pd.Series, length: int) -> pd.Series:
    return x.ewm(span=length, adjust=False, min_periods=length).mean()

def wma(x: pd.Series, length: int) -> pd.Series:
    w = np.arange(1, length + 1, dtype=float)
    return x.rolling(length, min_periods=length).apply(lambda v: np.dot(v, w) / w.sum(), raw=True)

def hma(x: pd.Series, length: int) -> pd.Series:
    # Pine ta.hma: WMA(2*WMA(x, len/2) - WMA(x, len), sqrt(len))
    half = max(1, int(length / 2))
    sqrt_len = max(1, int(np.sqrt(length)))
    return wma(2 * wma(x, half) - wma(x, length), sqrt_len)

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return rma(tr, length)

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)

def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)

def get_ma(ma_type: str, s: pd.Series, length: int) -> pd.Series:
    if ma_type == "SMA":
        return sma(s, length)
    if ma_type == "EMA":
        return ema(s, length)
    if ma_type == "HMA":
        return hma(s, length)
    # RMA
    return rma(s, length)


# ----------------------------
# DMI/ADX (Pine ta.dmi)
# ----------------------------
def dmi(df: pd.DataFrame, di_len: int = 14, adx_len: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    down = -l.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    tr_rma = rma(pd.Series(tr, index=df.index), di_len)
    plus_rma = rma(pd.Series(plus_dm, index=df.index), di_len)
    minus_rma = rma(pd.Series(minus_dm, index=df.index), di_len)

    di_plus = 100.0 * (plus_rma / tr_rma.replace(0.0, np.nan))
    di_minus = 100.0 * (minus_rma / tr_rma.replace(0.0, np.nan))

    dx = 100.0 * ((di_plus - di_minus).abs() / (di_plus + di_minus).replace(0.0, np.nan))
    adx = rma(dx, adx_len)
    return di_plus, di_minus, adx


# ----------------------------
# WaveTrend (per your Pine)
# ----------------------------
def wavetrend_tci(df: pd.DataFrame) -> pd.Series:
    ap = (df["high"] + df["low"] + df["close"]) / 3.0  # hlc3
    esa = ema(ap, 10)
    d = ema((ap - esa).abs(), 10)
    ci = (ap - esa) / (0.015 * d.replace(0.0, np.nan))
    tci = ema(ci, 21)
    return tci


# ----------------------------
# Data loading / dummy
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    path = os.getenv("OHLCV_CSV_PATH", "").strip()
    if path:
        try:
            raw = pd.read_csv(path)
            return validate_ohlcv(raw)
        except Exception:
            pass
    return make_dummy_ohlcv(["BTCUSDT", "ETHUSDT", "AAPL", "SPY"])

def make_dummy_ohlcv(symbols: List[str], start="2024-01-01", end="2025-12-21", freq="15min", seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    out = []
    for s in symbols:
        drift = rng.normal(0.00002, 0.00002)
        vol = rng.uniform(0.0008, 0.0020)
        rets = rng.normal(drift, vol, size=len(idx))
        p0 = rng.uniform(40, 500)
        close = p0 * np.exp(np.cumsum(rets))
        spread = np.abs(rng.normal(0, vol * 2, size=len(idx)))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        volume = rng.lognormal(mean=10, sigma=0.3, size=len(idx))
        out.append(pd.DataFrame({"timestamp": idx, "symbol": s, "open": open_, "high": high, "low": low, "close": close, "volume": volume}))
    return validate_ohlcv(pd.concat(out, ignore_index=True))

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    df = df[df["volume"] >= 0]
    mx = df[["open", "close"]].max(axis=1)
    mn = df[["open", "close"]].min(axis=1)
    df["high"] = np.maximum(df["high"].values, mx.values)
    df["low"] = np.minimum(df["low"].values, mn.values)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df

TF_MAP = {"15m": "15min", "30m": "30min", "1H": "1H", "4H": "4H", "1D": "1D"}

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


# ----------------------------
# Pivot logic (ta.pivothigh/low)
# Note: Pine pivot confirms after `right` bars. We return:
#  - pivot_val at confirmation bar index
#  - pivot_origin_index (center bar where pivot formed)
# ----------------------------
def pivothigh(high: pd.Series, left: int, right: int) -> Tuple[pd.Series, pd.Series]:
    n = left + right + 1
    roll_max = high.rolling(n, center=True, min_periods=n).max()
    is_pivot_center = (high == roll_max)
    center_val = high.where(is_pivot_center)

    # shift to confirmation bar (right bars after center)
    pivot_val_at_confirm = center_val.shift(right)
    origin_idx = pd.Series(np.nan, index=high.index, dtype="float64")
    # store origin position as integer offset in array sense (we'll map separately)
    # We'll keep origin timestamp by shifting index
    origin_ts = pd.Series(pd.NaT, index=high.index, dtype="datetime64[ns, UTC]")
    origin_ts.loc[pivot_val_at_confirm.notna()] = high.index.to_series().shift(right).loc[pivot_val_at_confirm.notna()]
    return pivot_val_at_confirm, origin_ts

def pivotlow(low: pd.Series, left: int, right: int) -> Tuple[pd.Series, pd.Series]:
    n = left + right + 1
    roll_min = low.rolling(n, center=True, min_periods=n).min()
    is_pivot_center = (low == roll_min)
    center_val = low.where(is_pivot_center)

    pivot_val_at_confirm = center_val.shift(right)
    origin_ts = pd.Series(pd.NaT, index=low.index, dtype="datetime64[ns, UTC]")
    origin_ts.loc[pivot_val_at_confirm.notna()] = low.index.to_series().shift(right).loc[pivot_val_at_confirm.notna()]
    return pivot_val_at_confirm, origin_ts


# ----------------------------
# Ported indicator logic
# ----------------------------
@dataclass
class ApexConfig:
    show_sig: bool = True
    show_sl: bool = True
    ma_type: str = "HMA"
    len_main: int = 55
    mult: float = 1.5
    liq_len: int = 10
    sd_ext: int = 20
    show_sd: bool = True
    show_bos: bool = True
    show_ob: bool = True
    show_fvg: bool = True
    fvg_mit: bool = True

def compute_apex(df: pd.DataFrame, cfg: ApexConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      - df_out with columns: baseline, atr_main, upper, lower, trend, tci, vol_ok, adx, sig_buy, sig_sell, trail_stop
      - zones dict with SD/OB/FVG lists (for plotting)
    """
    out = df.copy()

    src = out["close"]  # Pine input.source(close) default
    baseline = get_ma(cfg.ma_type, src, cfg.len_main)
    atr_main = atr(out, cfg.len_main)
    upper = baseline + (atr_main * cfg.mult)
    lower = baseline - (atr_main * cfg.mult)

    out["baseline"] = baseline
    out["atr_main"] = atr_main
    out["upper"] = upper
    out["lower"] = lower

    # Stateful trend: var int trend=0; if close>upper ->1 else if close<lower -> -1 else keep
    trend = pd.Series(0, index=out.index, dtype="int64")
    for i in range(1, len(out)):
        prev = int(trend.iat[i - 1])
        if out["close"].iat[i] > out["upper"].iat[i]:
            trend.iat[i] = 1
        elif out["close"].iat[i] < out["lower"].iat[i]:
            trend.iat[i] = -1
        else:
            trend.iat[i] = prev
    out["trend"] = trend

    # DMI/ADX filter: ta.dmi(14,14), adx>20
    di_plus, di_minus, adx_s = dmi(out, 14, 14)
    out["di_plus"] = di_plus
    out["di_minus"] = di_minus
    out["adx"] = adx_s
    out["adx_ok"] = out["adx"] > 20

    # WaveTrend: mom_buy = tci < 60 ; mom_sell = tci > -60 (as in Pine)
    out["tci"] = wavetrend_tci(out)
    out["mom_buy"] = out["tci"] < 60
    out["mom_sell"] = out["tci"] > -60

    # Volume filter
    out["vol_avg"] = sma(out["volume"], 20)
    out["vol_ok"] = out["volume"] > out["vol_avg"]

    # Signal logic: trend flips into 1/-1 + filters
    out["sig_buy"] = (out["trend"] == 1) & (out["trend"].shift(1) != 1) & out["vol_ok"] & out["mom_buy"] & out["adx_ok"]
    out["sig_sell"] = (out["trend"] == -1) & (out["trend"].shift(1) != -1) & out["vol_ok"] & out["mom_sell"] & out["adx_ok"]

    # Trailing stop: var float trail_stop=na ; trail_atr=atr(14)*2
    trail_atr = atr(out, 14) * 2.0
    trail_stop = pd.Series(np.nan, index=out.index, dtype="float64")
    for i in range(len(out)):
        t = int(out["trend"].iat[i])
        prev_t = int(out["trend"].iat[i - 1]) if i > 0 else 0
        prev_ts = trail_stop.iat[i - 1] if i > 0 else np.nan
        c = float(out["close"].iat[i])
        ta_ = float(trail_atr.iat[i]) if np.isfinite(trail_atr.iat[i]) else np.nan

        if not np.isfinite(ta_):
            trail_stop.iat[i] = prev_ts
            continue

        if t == 1:
            trail_stop.iat[i] = max(nz(prev_ts, c), c - ta_)
            if prev_t == -1:
                trail_stop.iat[i] = c - ta_
        elif t == -1:
            trail_stop.iat[i] = min(nz(prev_ts, c), c + ta_)
            if prev_t == 1:
                trail_stop.iat[i] = c + ta_
        else:
            trail_stop.iat[i] = prev_ts
    out["trail_stop"] = trail_stop

    # --- Pivots ---
    ph_val, ph_origin_ts = pivothigh(out["high"], cfg.liq_len, cfg.liq_len)
    pl_val, pl_origin_ts = pivotlow(out["low"], cfg.liq_len, cfg.liq_len)
    out["ph"] = ph_val
    out["pl"] = pl_val
    out["ph_origin_ts"] = ph_origin_ts
    out["pl_origin_ts"] = pl_origin_ts

    # --- SMC state ---
    last_ph = np.nan
    last_pl = np.nan
    lower_high = np.nan
    higher_low = np.nan

    bos_events = []   # (ts, price, kind)
    choch_events = [] # (ts, price, kind)

    # Zones for plotting
    sd_zones = []   # dict: {kind, x0, x1, top, bot}
    ob_zones = []   # dict: {kind, x0, x1, top, bot, active}
    fvg_zones = []  # dict: {kind, x0, x1, top, bot, active}

    # Helper for cap
    def cap_list(arr: list, lim: int):
        if len(arr) > lim:
            del arr[0 : (len(arr) - lim)]

    # Precompute ATR for FVG threshold
    atr_c = atr(out, 14)

    idx = out.index.to_list()
    for i, ts in enumerate(idx):
        # Update last pivots on confirmation bars
        if pd.notna(out["ph"].iat[i]):
            last_ph = float(out["high"].shift(cfg.liq_len).iat[i])  # high[liq_len] at confirm bar i
            if int(out["trend"].iat[i]) == -1:
                lower_high = last_ph
            if cfg.show_sd:
                # Supply zone: left=bar_index-liq_len; right=bar_index+sd_ext; top=high[liq_len]; bottom=max(open,close)[liq_len]
                o0 = float(out["open"].shift(cfg.liq_len).iat[i])
                c0 = float(out["close"].shift(cfg.liq_len).iat[i])
                top = float(out["high"].shift(cfg.liq_len).iat[i])
                bot = float(max(o0, c0))
                x0 = idx[i - cfg.liq_len] if (i - cfg.liq_len) >= 0 else ts
                x1 = idx[min(i + cfg.sd_ext, len(idx) - 1)]
                sd_zones.append({"kind": "supply", "x0": x0, "x1": x1, "top": top, "bot": bot})
                cap_list(sd_zones, 10)

        if pd.notna(out["pl"].iat[i]):
            last_pl = float(out["low"].shift(cfg.liq_len).iat[i])   # low[liq_len]
            if int(out["trend"].iat[i]) == 1:
                higher_low = last_pl
            if cfg.show_sd:
                o0 = float(out["open"].shift(cfg.liq_len).iat[i])
                c0 = float(out["close"].shift(cfg.liq_len).iat[i])
                bot = float(out["low"].shift(cfg.liq_len).iat[i])
                top = float(min(o0, c0))
                x0 = idx[i - cfg.liq_len] if (i - cfg.liq_len) >= 0 else ts
                x1 = idx[min(i + cfg.sd_ext, len(idx) - 1)]
                sd_zones.append({"kind": "demand", "x0": x0, "x1": x1, "top": top, "bot": bot})
                cap_list(sd_zones, 10)

        close_i = float(out["close"].iat[i])

        # Crosses vs stored levels (Pine x_ph = ta.crossover(close, last_ph))
        # Since last_ph/last_pl/lower_high/higher_low are scalars that can update,
        # emulate with previous close comparison.
        prev_close = float(out["close"].iat[i - 1]) if i > 0 else np.nan

        x_ph = np.isfinite(last_ph) and np.isfinite(prev_close) and (prev_close <= last_ph) and (close_i > last_ph)
        x_pl = np.isfinite(last_pl) and np.isfinite(prev_close) and (prev_close >= last_pl) and (close_i < last_pl)
        x_lh = np.isfinite(lower_high) and np.isfinite(prev_close) and (prev_close <= lower_high) and (close_i > lower_high)
        x_hl = np.isfinite(higher_low) and np.isfinite(prev_close) and (prev_close >= higher_low) and (close_i < higher_low)

        tr = int(out["trend"].iat[i])

        # BOS / CHoCH events (store for plotting)
        if cfg.show_bos:
            if tr == 1 and x_ph:
                bos_events.append((ts, float(last_ph), "BOS_BULL"))
            if tr == -1 and x_pl:
                bos_events.append((ts, float(last_pl), "BOS_BEAR"))
            if tr == -1 and x_lh:
                choch_events.append((ts, float(lower_high), "CHoCH_BULL"))
                higher_low = float(out["low"].iat[i])  # higher_low := low
            if tr == 1 and x_hl:
                choch_events.append((ts, float(higher_low), "CHoCH_BEAR"))
                lower_high = float(out["high"].iat[i])  # lower_high := high

        # Order blocks on x_ph/x_pl
        if cfg.show_ob:
            if tr == 1 and x_ph:
                # find last bearish candle in past 1..20 (close[i] < open[i])
                for j in range(1, 21):
                    if i - j < 0:
                        break
                    if float(out["close"].iat[i - j]) < float(out["open"].iat[i - j]):
                        top = float(out["high"].iat[i - j])
                        bot = float(out["low"].iat[i - j])
                        x0 = idx[i - j]
                        x1 = idx[min(i + cfg.sd_ext, len(idx) - 1)]
                        ob_zones.append({"kind": "ob_demand", "x0": x0, "x1": x1, "top": top, "bot": bot, "active": True})
                        cap_list(ob_zones, 5)
                        break
            if tr == -1 and x_pl:
                # find last bullish candle (close > open)
                for j in range(1, 21):
                    if i - j < 0:
                        break
                    if float(out["close"].iat[i - j]) > float(out["open"].iat[i - j]):
                        top = float(out["high"].iat[i - j])
                        bot = float(out["low"].iat[i - j])
                        x0 = idx[i - j]
                        x1 = idx[min(i + cfg.sd_ext, len(idx) - 1)]
                        ob_zones.append({"kind": "ob_supply", "x0": x0, "x1": x1, "top": top, "bot": bot, "active": True})
                        cap_list(ob_zones, 5)
                        break

        # FVG logic:
        # fvg_b = (low > high[2]) and (low - high[2] > atr_c * 0.5)
        # fvg_s = (high < low[2]) and (low[2] - high > atr_c * 0.5)
        if cfg.show_fvg and i >= 2:
            low_i = float(out["low"].iat[i])
            high_i = float(out["high"].iat[i])
            high_2 = float(out["high"].iat[i - 2])
            low_2 = float(out["low"].iat[i - 2])
            atr_i = float(atr_c.iat[i]) if np.isfinite(atr_c.iat[i]) else np.nan
            thr = (atr_i * 0.5) if np.isfinite(atr_i) else np.inf

            fvg_b = (low_i > high_2) and ((low_i - high_2) > thr)
            fvg_s = (high_i < low_2) and ((low_2 - high_i) > thr)

            if fvg_b:
                x0 = idx[i - 2]
                x1 = idx[min(i + cfg.sd_ext, len(idx) - 1)]
                # box.new(bar_index[2], high[2], ..., low)
                fvg_zones.append({"kind": "fvg_bull", "x0": x0, "x1": x1, "top": high_2, "bot": low_i, "active": True})
            if fvg_s:
                x0 = idx[i - 2]
                x1 = idx[min(i + cfg.sd_ext, len(idx) - 1)]
                # box.new(bar_index[2], low[2], ..., high)
                fvg_zones.append({"kind": "fvg_bear", "x0": x0, "x1": x1, "top": low_2, "bot": high_i, "active": True})
            cap_list(fvg_zones, 10)

        # Mitigation (delete mitigated, else extend right edge)
        if cfg.fvg_mit:
            # OB zones
            for z in ob_zones:
                if not z.get("active", True):
                    continue
                bt = max(z["top"], z["bot"])
                bb = min(z["top"], z["bot"])
                if (close_i > bt and bt > bb) or (close_i < bb and bb < bt):
                    z["active"] = False
                else:
                    z["x1"] = idx[min(i + 5, len(idx) - 1)]
            # FVG zones
            for z in fvg_zones:
                if not z.get("active", True):
                    continue
                bt = max(z["top"], z["bot"])
                bb = min(z["top"], z["bot"])
                if (close_i > bt and bt > bb) or (close_i < bb and bb < bt):
                    z["active"] = False
                else:
                    z["x1"] = idx[min(i + 5, len(idx) - 1)]

    zones = {
        "sd": [z for z in sd_zones],
        "ob": [z for z in ob_zones if z.get("active", True)],
        "fvg": [z for z in fvg_zones if z.get("active", True)],
        "bos": bos_events,
        "choch": choch_events,
    }
    return out, zones


# ----------------------------
# Simple signal backtest (optional): enter on sig_buy, exit on sig_sell or trailing stop break
# Executes next bar open to avoid look-ahead.
# ----------------------------
@dataclass
class BacktestConfig:
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    initial_equity: float = 10_000.0
    execute_on: str = "next_open"  # "next_open" or "close"

def _apply_costs(px: float, fee_bps: float, slippage_bps: float, side: str) -> float:
    adj = (fee_bps + slippage_bps) / 10_000.0
    return px * (1.0 + adj) if side == "buy" else px * (1.0 - adj)

def backtest_apex(df: pd.DataFrame, cfg: BacktestConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    in_pos = False
    entry_px = np.nan
    qty = 0.0
    equity = cfg.initial_equity
    peak = equity

    equity_curve = []
    dd_curve = []
    pos_curve = []
    trades = []

    idx = df.index.to_list()
    for i, ts in enumerate(idx):
        row = df.iloc[i]
        close_i = float(row["close"])
        total_equity = equity if not in_pos else qty * close_i
        peak = max(peak, total_equity)
        dd = (total_equity / peak) - 1.0

        equity_curve.append(total_equity)
        dd_curve.append(dd)
        pos_curve.append(1.0 if in_pos else 0.0)

        has_next = (i + 1) < len(idx)
        next_ts = idx[i + 1] if has_next else None

        # Exit: sig_sell OR close crosses under trail_stop in long trend
        if in_pos:
            trail = float(row.get("trail_stop", np.nan))
            prev_close = float(df["close"].iloc[i - 1]) if i > 0 else np.nan
            exit_signal = bool(row.get("sig_sell", False))
            exit_trail = np.isfinite(trail) and np.isfinite(prev_close) and (prev_close >= trail) and (close_i < trail)

            if exit_signal or exit_trail:
                if cfg.execute_on == "next_open" and has_next:
                    px = float(df["open"].iloc[i + 1])
                    exit_time = next_ts
                else:
                    px = close_i
                    exit_time = ts
                px = _apply_costs(px, cfg.fee_bps, cfg.slippage_bps, "sell")
                proceeds = qty * px
                pnl = proceeds - (qty * entry_px)
                equity = proceeds
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_px": entry_px,
                    "exit_px": px,
                    "qty": qty,
                    "pnl": pnl,
                    "pnl_pct": pnl / (qty * entry_px) if qty * entry_px else np.nan,
                    "reason": "sig_sell" if exit_signal else "trail_stop",
                })
                in_pos = False
                entry_px = np.nan
                qty = 0.0

        # Entry: sig_buy
        if (not in_pos) and bool(row.get("sig_buy", False)):
            if cfg.execute_on == "next_open":
                if not has_next:
                    continue
                px = float(df["open"].iloc[i + 1])
                entry_time = next_ts
            else:
                px = close_i
                entry_time = ts
            px = _apply_costs(px, cfg.fee_bps, cfg.slippage_bps, "buy")
            qty = equity / px
            entry_px = px
            in_pos = True

    eq = pd.DataFrame({"equity": equity_curve, "drawdown": dd_curve, "position": pos_curve}, index=df.index)
    trades_df = pd.DataFrame(trades)
    return eq, trades_df

def compute_kpis(eq: pd.DataFrame, trades: pd.DataFrame, bars_per_year: float) -> Dict[str, Any]:
    equity = eq["equity"].astype(float)
    ret = equity.pct_change().fillna(0.0)
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    sharpe = (ret.mean() * bars_per_year) / (ret.std(ddof=0) * np.sqrt(bars_per_year) + 1e-12)
    max_dd = float(eq["drawdown"].min())

    if trades is None or len(trades) == 0:
        win_rate = np.nan
        profit_factor = np.nan
        avg_trade = np.nan
        n_trades = 0
    else:
        n_trades = len(trades)
        wins = trades["pnl"] > 0
        win_rate = float(wins.mean())
        gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf
        avg_trade = float(trades["pnl_pct"].mean())

    exposure = float(eq["position"].mean())
    return {
        "Total Return": total_return,
        "Sharpe (approx)": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Avg Trade %": avg_trade,
        "Exposure %": exposure,
        "Trades": n_trades,
    }

def bars_per_year_from_rule(rule: str) -> float:
    if rule.endswith("min"):
        mins = float(rule.replace("min", ""))
        return (365.0 * 24.0 * 60.0) / mins
    if rule.endswith("H"):
        hrs = float(rule.replace("H", ""))
        return (365.0 * 24.0) / hrs
    if rule.endswith("D"):
        days = float(rule.replace("D", ""))
        return 365.0 / days
    return 365.0


# ----------------------------
# TradingView webhook sender (outbound)
# ----------------------------
def _get_secret(path: List[str], default: Optional[str] = None) -> Optional[str]:
    try:
        d = st.secrets
        for k in path:
            d = d[k]
        if isinstance(d, str) and d.strip():
            return d.strip()
    except Exception:
        pass
    env_key = "_".join([p.upper() for p in path])
    v = os.getenv(env_key, default)
    return v.strip() if isinstance(v, str) and v.strip() else default

def send_tradingview_webhook(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """
    POST JSON to your webhook receiver.
    Secrets:
      st.secrets['tradingview']['webhook_url']
      st.secrets['tradingview']['passphrase'] (optional)
    Env fallback:
      TRADINGVIEW_WEBHOOK_URL, TRADINGVIEW_PASSPHRASE
    """
    url = _get_secret(["tradingview", "webhook_url"])
    passphrase = _get_secret(["tradingview", "passphrase"])
    if not url:
        return False, "NO-OP (missing tradingview.webhook_url)"
    if requests is None:
        return False, "requests not available"
    safe_payload = dict(payload)
    if passphrase:
        safe_payload["passphrase"] = passphrase
    try:
        r = requests.post(url, json=safe_payload, timeout=10)
        return (200 <= r.status_code < 300), f"POST {r.status_code}"
    except Exception as e:
        return False, f"Error: {type(e).__name__}"


# ----------------------------
# UI
# ----------------------------
st.title("Apex Trend & Liquidity Master (SMC) v7.2 — Pine Port")
st.caption("Signals/zones are computed to mirror the Pine script. Backtest is hypothetical and executes on next-bar open.")

raw = load_data()

with st.sidebar:
    st.header("Filters")
    symbols = sorted(raw["symbol"].unique().tolist())
    symbol = st.selectbox("Symbol", symbols, index=0)
    tf = st.selectbox("Timeframe", list(TF_MAP.keys()), index=2)
    rule = TF_MAP[tf]

    sym = raw[raw["symbol"] == symbol].copy()
    sym["timestamp"] = pd.to_datetime(sym["timestamp"], utc=True)
    sym = sym.sort_values("timestamp")
    min_ts, max_ts = sym["timestamp"].min(), sym["timestamp"].max()
    start_date = st.date_input("Start (UTC)", value=min_ts.date())
    end_date = st.date_input("End (UTC)", value=max_ts.date())

    st.divider()
    st.header("Apex Inputs")
    show_sig = st.checkbox("Show Buy/Sell Signals", value=True)
    show_sl = st.checkbox("Show Trailing Stop", value=True)
    ma_type = st.selectbox("Trend Algorithm", ["EMA", "SMA", "HMA", "RMA"], index=2)
    len_main = st.number_input("Trend Length", min_value=10, max_value=500, value=55, step=1)
    mult = st.number_input("Volatility Multiplier", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

    st.divider()
    st.header("Classic Supply & Demand")
    show_sd = st.checkbox("Show Swing S/D Zones", value=True)
    liq_len = st.number_input("Pivot Lookback", min_value=2, max_value=50, value=10, step=1)
    sd_ext = st.number_input("Extension (bars)", min_value=5, max_value=200, value=20, step=1)

    st.divider()
    st.header("SMC")
    show_bos = st.checkbox("Show BOS/CHoCH", value=True)
    show_ob = st.checkbox("Show Order Blocks", value=True)
    show_fvg = st.checkbox("Show FVG", value=True)
    fvg_mit = st.checkbox("Auto-Delete Mitigated", value=True)

    st.divider()
    st.header("Backtest (optional)")
    do_bt = st.checkbox("Run simple backtest on BUY/SELL", value=True)
    fee_bps = st.number_input("Fee (bps/side)", 0.0, 200.0, 5.0, 0.5)
    slippage_bps = st.number_input("Slippage (bps/side)", 0.0, 200.0, 2.0, 0.5)
    exec_on = st.selectbox("Execution", ["next_open", "close"], index=0)

    st.divider()
    st.header("TradingView Webhook (Outbound)")
    st.caption("Uses secrets: tradingview.webhook_url (+ optional tradingview.passphrase). Never displayed.")
    enable_send = st.checkbox("Enable send button", value=True)

# Filter & resample
sym = sym[(sym["timestamp"].dt.date >= start_date) & (sym["timestamp"].dt.date <= end_date)]
sym = sym.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
if len(sym) < 100:
    st.warning("Not enough data in range.")
    st.stop()

df = resample_ohlcv(sym, rule)
if len(df) < 100:
    st.warning("Not enough bars after resample.")
    st.stop()

cfg = ApexConfig(
    show_sig=show_sig, show_sl=show_sl, ma_type=ma_type, len_main=int(len_main), mult=float(mult),
    show_sd=show_sd, liq_len=int(liq_len), sd_ext=int(sd_ext),
    show_bos=show_bos, show_ob=show_ob, show_fvg=show_fvg, fvg_mit=fvg_mit
)
df_out, zones = compute_apex(df, cfg)

# KPI cards
kcol = st.columns(5)
kcol[0].metric("Trend", "Bull" if int(df_out["trend"].iloc[-1]) == 1 else ("Bear" if int(df_out["trend"].iloc[-1]) == -1 else "Flat"))
kcol[1].metric("ADX", f"{float(df_out['adx'].iloc[-1]):.1f}")
kcol[2].metric("TCI", f"{float(df_out['tci'].iloc[-1]):.1f}")
kcol[3].metric("Vol OK", "Yes" if bool(df_out["vol_ok"].iloc[-1]) else "No")
kcol[4].metric("Last Signal", "BUY" if bool(df_out["sig_buy"].iloc[-1]) else ("SELL" if bool(df_out["sig_sell"].iloc[-1]) else "—"))

# Candles + overlays
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_out.index, open=df_out["open"], high=df_out["high"], low=df_out["low"], close=df_out["close"],
    name="OHLC", increasing_line_color=GOOD, decreasing_line_color=BAD
))

# Trend cloud (upper/lower fill)
fig.add_trace(go.Scatter(x=df_out.index, y=df_out["upper"], name="Upper", line=dict(color="rgba(0,0,0,0)"), showlegend=False))
fig.add_trace(go.Scatter(x=df_out.index, y=df_out["lower"], name="Lower", fill="tonexty",
                         fillcolor="rgba(0,105,92,0.25)" if int(df_out["trend"].iloc[-1]) == 1 else "rgba(183,28,28,0.25)",
                         line=dict(color="rgba(0,0,0,0)"), showlegend=False))

# Trailing stop
if cfg.show_sl:
    fig.add_trace(go.Scatter(
        x=df_out.index, y=df_out["trail_stop"], name="Trailing Stop",
        line=dict(color=GOOD if int(df_out["trend"].iloc[-1]) == 1 else BAD, width=2)
    ))

# BUY/SELL markers
if cfg.show_sig:
    buys = df_out.index[df_out["sig_buy"].fillna(False)]
    sells = df_out.index[df_out["sig_sell"].fillna(False)]
    if len(buys):
        fig.add_trace(go.Scatter(x=buys, y=df_out.loc[buys, "low"], mode="markers", name="BUY",
                                 marker=dict(symbol="triangle-up", size=11, color=GOOD)))
    if len(sells):
        fig.add_trace(go.Scatter(x=sells, y=df_out.loc[sells, "high"], mode="markers", name="SELL",
                                 marker=dict(symbol="triangle-down", size=11, color=BAD)))

# Zones (SD / OB / FVG) as rectangles
def add_zone_rect(z, color_rgba: str, name: str):
    fig.add_shape(
        type="rect",
        x0=z["x0"], x1=z["x1"],
        y0=min(z["top"], z["bot"]), y1=max(z["top"], z["bot"]),
        line=dict(width=0),
        fillcolor=color_rgba,
        layer="below",
    )

if cfg.show_sd:
    for z in zones["sd"]:
        if z["kind"] == "supply":
            add_zone_rect(z, "rgba(229,57,53,0.18)", "Supply")
        else:
            add_zone_rect(z, "rgba(67,160,71,0.18)", "Demand")

if cfg.show_ob:
    for z in zones["ob"]:
        if z["kind"] == "ob_demand":
            add_zone_rect(z, "rgba(185,246,202,0.18)", "OB Demand")
        else:
            add_zone_rect(z, "rgba(255,205,210,0.18)", "OB Supply")

if cfg.show_fvg:
    for z in zones["fvg"]:
        if z["kind"] == "fvg_bull":
            add_zone_rect(z, "rgba(185,246,202,0.18)", "FVG Bull")
        else:
            add_zone_rect(z, "rgba(255,205,210,0.18)", "FVG Bear")

# BOS / CHoCH as lines + annotations
if cfg.show_bos:
    for ts, price, kind in zones["bos"]:
        col = GOOD if "BULL" in kind else BAD
        fig.add_shape(type="line", x0=ts, x1=ts, y0=price, y1=price, line=dict(color=col, width=2))
        fig.add_annotation(x=ts, y=price, text="BOS", showarrow=False, font=dict(color=col, size=12), yshift=10)
    for ts, price, kind in zones["choch"]:
        col = GOOD if "BULL" in kind else BAD
        fig.add_shape(type="line", x0=ts, x1=ts, y0=price, y1=price, line=dict(color=col, width=2))
        fig.add_annotation(x=ts, y=price, text="CHoCH", showarrow=False, font=dict(color=col, size=12), yshift=10)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    height=700,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Backtest section
if do_bt:
    bt_cfg = BacktestConfig(fee_bps=float(fee_bps), slippage_bps=float(slippage_bps), execute_on=str(exec_on))
    eq, trades = backtest_apex(df_out, bt_cfg)
    kpis = compute_kpis(eq, trades, bars_per_year=bars_per_year_from_rule(rule))

    st.subheader("Backtest KPIs (BUY/SELL + trailing stop exit)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return", f"{kpis['Total Return']:.2%}")
    c2.metric("Sharpe (approx)", f"{kpis['Sharpe (approx)']:.2f}")
    c3.metric("Max Drawdown", f"{kpis['Max Drawdown']:.2%}")
    c4.metric("Win Rate", "—" if np.isnan(kpis["Win Rate"]) else f"{kpis['Win Rate']:.1%}")
    c5.metric("Profit Factor", "∞" if np.isinf(kpis["Profit Factor"]) else ("—" if np.isnan(kpis["Profit Factor"]) else f"{kpis['Profit Factor']:.2f}"))

    e1, e2 = st.columns([1.2, 0.8], gap="large")
    with e1:
        st.caption("Equity and drawdown are computed from a simple long-only simulation. Hypothetical.")
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=eq.index, y=eq["equity"], name="Equity", line=dict(color=ACCENT, width=2)))
        eq_fig.add_trace(go.Scatter(x=eq.index, y=eq["drawdown"], name="Drawdown", yaxis="y2",
                                    line=dict(color=BAD, width=1)))
        eq_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            height=320,
            margin=dict(l=10, r=10, t=25, b=10),
            yaxis=dict(title="Equity"),
            yaxis2=dict(title="Drawdown", overlaying="y", side="right", tickformat=".0%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(eq_fig, use_container_width=True)
    with e2:
        st.caption("Trades")
        if trades is None or len(trades) == 0:
            st.info("No trades produced.")
        else:
            st.dataframe(trades.sort_values("entry_time", ascending=False), use_container_width=True, hide_index=True)

# Webhook payload + send
st.subheader("TradingView Webhook (Outbound)")
st.caption("Sends JSON to your configured endpoint. Secrets never shown. If webhook URL missing → safe NO-OP.")

latest = df_out.iloc[-1]
latest_ts = df_out.index[-1]
signal = "NONE"
if bool(latest.get("sig_buy", False)):
    signal = "APEX_BUY"
elif bool(latest.get("sig_sell", False)):
    signal = "APEX_SELL"

payload = {
    "symbol": symbol,
    "timeframe": tf,
    "timestamp": str(latest_ts),
    "signal": signal,
    "trend": int(latest["trend"]),
    "close": float(latest["close"]),
    "upper": float(latest["upper"]) if np.isfinite(latest["upper"]) else None,
    "lower": float(latest["lower"]) if np.isfinite(latest["lower"]) else None,
    "trail_stop": float(latest["trail_stop"]) if np.isfinite(latest["trail_stop"]) else None,
    "adx": float(latest["adx"]) if np.isfinite(latest["adx"]) else None,
    "tci": float(latest["tci"]) if np.isfinite(latest["tci"]) else None,
    "vol_ok": bool(latest["vol_ok"]),
}

p1, p2 = st.columns([0.55, 0.45], gap="large")
with p1:
    st.json(payload)

with p2:
    if enable_send:
        if st.button("Send latest via webhook", type="primary", use_container_width=True):
            ok, msg = send_tradingview_webhook(payload)
            if ok:
                st.success(f"Webhook sent ({msg}).")
            else:
                st.warning(f"Webhook not sent: {msg}")

st.caption(
    "Configure `.streamlit/secrets.toml`:\n\n"
    "[tradingview]\n"
    "webhook_url = \"https://your-endpoint\"\n"
    "passphrase = \"optional\"\n\n"
    "Env fallback: TRADINGVIEW_WEBHOOK_URL / TRADINGVIEW_PASSPHRASE"
)
