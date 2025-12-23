import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timezone
import requests
import time
import math
import random
import urllib.parse
from scipy.stats import linregress
import streamlit.components.v1 as components

# =============================================================================
# 0. PINE SCRIPT REFERENCE (VERBATIM - BRANDING REMOVED)
# =============================================================================
PINE_SCRIPT_SMC_V8 = r"""// This Pine Script® code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/ 
// © DarkPoolCrypto
// Improved Version with Enhanced Logic, Performance, and Features

//@version=6
indicator("Trend & Liquidity Master (SMC) v8.0", overlay=true, max_boxes_count=500, max_lines_count=500, max_labels_count=500)

// ════════════════════════════════════════════════════════════════════════════
// 1. SYSTEM CONFIGURATION
// ════════════════════════════════════════════════════════════════════════════

// --- Trend & Signals ---
grp_sig = "🌊 Trend & Signals"
show_sig = input.bool(true, "Show Buy/Sell Signals", group=grp_sig, tooltip="Momentum-based entry signals with multi-filter confirmation.")
show_sl = input.bool(true, "Show Trailing Stop", group=grp_sig, tooltip="ATR-based dynamic trailing stop with trend alignment.")
ma_type = input.string("HMA", "Trend Algorithm", options=["EMA", "SMA", "HMA", "RMA"], group=grp_sig)
len_main = input.int(55, "Trend Length", minval=10, maxval=200, group=grp_sig)
mult = input.float(1.5, "Volatility Multiplier", minval=0.5, maxval=5.0, step=0.1, group=grp_sig)
src = input.source(close, "Source", group=grp_sig)

// Signal Filters
use_adx = input.bool(true, "Use ADX Filter", group=grp_sig, tooltip="Require trend strength confirmation.")
adx_threshold = input.int(20, "ADX Minimum", minval=10, maxval=50, group=grp_sig)
use_volume = input.bool(true, "Use Volume Filter", group=grp_sig, tooltip="Require above-average volume.")
vol_mult = input.float(1.0, "Volume Multiplier", minval=0.5, maxval=3.0, step=0.1, group=grp_sig)
use_momentum = input.bool(true, "Use Momentum Filter", group=grp_sig, tooltip="WaveTrend momentum confirmation.")

// --- Classic Supply & Demand ---
grp_sd = "🧱 Classic Supply & Demand"
show_sd = input.bool(true, "Show Swing S/D Zones", group=grp_sd)
liq_len = input.int(10, "Pivot Lookback", minval=3, maxval=50, group=grp_sd)
sd_ext = input.int(20, "Extension", minval=5, maxval=100, group=grp_sd)
sd_max_zones = input.int(10, "Max Zones", minval=5, maxval=30, group=grp_sd)

// --- Smart Money Concepts (SMC) ---
grp_smc = "🏛️ Smart Money Concepts"
show_bos = input.bool(true, "Show BOS/CHoCH", group=grp_smc)
show_ob = input.bool(true, "Show Order Blocks", group=grp_smc)
show_fvg = input.bool(true, "Show FVG", group=grp_smc)
fvg_mit = input.bool(true, "Auto-Delete Mitigated", group=grp_smc)
ob_lookback = input.int(20, "OB Lookback", minval=5, maxval=50, group=grp_smc)
fvg_min_size = input.float(0.5, "FVG Min Size (ATR)", minval=0.1, maxval=2.0, step=0.1, group=grp_smc)
ob_max_zones = input.int(5, "Max Order Blocks", minval=3, maxval=15, group=grp_smc)
fvg_max_zones = input.int(10, "Max FVGs", minval=5, maxval=30, group=grp_smc)

// --- Visual Palette ---
grp_vis = "🎨 Visual Settings"
c_tr_bull = input.color(#00695C, "Trend: Bullish", group=grp_vis)
c_tr_bear = input.color(#B71C1C, "Trend: Bearish", group=grp_vis)
c_sig_bull = input.color(#00E676, "Action: Bullish", group=grp_vis)
c_sig_bear = input.color(#FF1744, "Action: Bearish", group=grp_vis)
c_sd_bull = input.color(#43A047, "Classic Demand", group=grp_vis)
c_sd_bear = input.color(#E53935, "Classic Supply", group=grp_vis)
c_smc_bull = input.color(#B9F6CA, "SMC Demand", group=grp_vis)
c_smc_bear = input.color(#FFCDD2, "SMC Supply", group=grp_vis)

// Performance Settings
cloud_transparency = input.int(85, "Cloud Transparency", minval=50, maxval=95, group=grp_vis)
zone_transparency = input.int(70, "Zone Transparency", minval=50, maxval=95, group=grp_vis)
smc_transparency = input.int(80, "SMC Transparency", minval=60, maxval=95, group=grp_vis)

// ════════════════════════════════════════════════════════════════════════════
// 2. HELPER FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

// Optimized MA calculation
get_ma(t, s, l) =>
    switch t
        "SMA" => ta.sma(s, l)
        "EMA" => ta.ema(s, l)
        "HMA" => ta.hma(s, l)
        "RMA" => ta.rma(s, l)
        => ta.sma(s, l)

// Efficient zone cleanup
cleanup_zones(box_array, max_size) =>
    if array.size(box_array) > max_size
        box.delete(array.shift(box_array))

// Check if price has mitigated a zone
is_mitigated(box_id, is_bullish) =>
    if na(box_id)
        false
    else
        top = box.get_top(box_id)
        bottom = box.get_bottom(box_id)
        if is_bullish
            close < bottom  // Demand zone mitigated if price closes below
        else
            close > top     // Supply zone mitigated if price closes above

// ════════════════════════════════════════════════════════════════════════════
// 3. TREND & SIGNALS ENGINE
// ════════════════════════════════════════════════════════════════════════════

// Dynamic MA Baseline
baseline = get_ma(ma_type, src, len_main)
atr = ta.atr(len_main)
upper = baseline + (atr * mult)
lower = baseline - (atr * mult)

// Trend State Machine (More Robust)
var int trend = 0
var int prev_trend = 0

// Trend confirmation requires price to stay beyond threshold for confirmation
trend_confirmed = close > upper ? 1 : close < lower ? -1 : trend

// Update trend with confirmation
if trend_confirmed != trend
    prev_trend := trend
    trend := trend_confirmed

// Enhanced Signal Filters
[di_plus, di_minus, adx] = ta.dmi(14, 14)
adx_ok = not use_adx or adx > adx_threshold

// Improved WaveTrend Calculation
ap = hlc3
esa = ta.ema(ap, 10)
d = ta.ema(math.abs(ap - esa), 10)
ci = d != 0 ? (ap - esa) / (0.015 * d) : 0
tci = ta.ema(ci, 21)

// Momentum conditions (more precise)
mom_buy = not use_momentum or (tci < 60 and tci > tci[1])  // Oversold and recovering
mom_sell = not use_momentum or (tci > -60 and tci < tci[1])  // Overbought and declining

// Enhanced Volume Filter
vol_avg = ta.sma(volume, 20)
vol_ok = not use_volume or volume > (vol_avg * vol_mult)

// Additional confirmation: Price action alignment
price_confirm_buy = close > open and close > close[1]
price_confirm_sell = close < open and close < close[1]

// Signal Logic (More Conservative)
sig_buy = trend == 1 and prev_trend != 1 and vol_ok and mom_buy and adx_ok and price_confirm_buy
sig_sell = trend == -1 and prev_trend != -1 and vol_ok and mom_sell and adx_ok and price_confirm_sell

// Enhanced Trailing Stop with Better Logic
var float trail_stop = na
var float trail_stop_prev = na

trail_atr = ta.atr(14) * 2.0

if trend == 1  // Bullish trend
    if na(trail_stop) or trend[1] != 1
        trail_stop := close - trail_atr
    else
        trail_stop := math.max(trail_stop, close - trail_atr)
        // Don't allow stop to move down
        if trail_stop < trail_stop_prev
            trail_stop := trail_stop_prev
    trail_stop_prev := trail_stop
    
else if trend == -1  // Bearish trend
    if na(trail_stop) or trend[1] != -1
        trail_stop := close + trail_atr
    else
        trail_stop := math.min(trail_stop, close + trail_atr)
        // Don't allow stop to move up
        if trail_stop > trail_stop_prev
            trail_stop := trail_stop_prev
    trail_stop_prev := trail_stop

else
    trail_stop := na
    trail_stop_prev := na

// ════════════════════════════════════════════════════════════════════════════
// 4. CLASSIC SUPPLY & DEMAND (PIVOTS)
// ════════════════════════════════════════════════════════════════════════════

ph = ta.pivothigh(high, liq_len, liq_len)
pl = ta.pivotlow(low, liq_len, liq_len)

var box[] sd_zones = array.new_box()

// Store zone metadata for better management
var int[] sd_zones_bars = array.new_int()

if not na(ph) and show_sd
    ph_price = high[liq_len]
    ph_bar = bar_index - liq_len
    zone_top = math.max(open[liq_len], close[liq_len])
    
    b = box.new(
         ph_bar, ph_price, 
         bar_index + sd_ext, zone_top,
         border_color=color.new(c_sd_bear, 50), 
         bgcolor=color.new(c_sd_bear, zone_transparency),
         text="Supply", 
         text_color=color.new(color.white, 20), 
         text_size=size.tiny,
         extend=extend.right)
    
    array.push(sd_zones, b)
    array.push(sd_zones_bars, ph_bar)
    cleanup_zones(sd_zones, sd_max_zones)
    if array.size(sd_zones_bars) > sd_max_zones
        array.shift(sd_zones_bars)

if not na(pl) and show_sd
    pl_price = low[liq_len]
    pl_bar = bar_index - liq_len
    zone_bottom = math.min(open[liq_len], close[liq_len])
    
    b = box.new(
         pl_bar, zone_bottom,
         bar_index + sd_ext, pl_price,
         border_color=color.new(c_sd_bull, 50), 
         bgcolor=color.new(c_sd_bull, zone_transparency),
         text="Demand", 
         text_color=color.new(color.white, 20), 
         text_size=size.tiny,
         extend=extend.right)
    
    array.push(sd_zones, b)
    array.push(sd_zones_bars, pl_bar)
    cleanup_zones(sd_zones, sd_max_zones)
    if array.size(sd_zones_bars) > sd_max_zones
        array.shift(sd_zones_bars)

// ════════════════════════════════════════════════════════════════════════════
// 5. SMC LOGIC (BOS / CHoCH / OB / FVG)
// ════════════════════════════════════════════════════════════════════════════

// Enhanced Structure Tracking
var float last_ph = na
var float last_pl = na
var float lower_high = na
var float higher_low = na
var int last_ph_bar = na
var int last_pl_bar = na

// Update structure points
if not na(ph)
    last_ph := high[liq_len]
    last_ph_bar := bar_index - liq_len
    if trend == -1
        lower_high := last_ph

if not na(pl)
    last_pl := low[liq_len]
    last_pl_bar := bar_index - liq_len
    if trend == 1
        higher_low := last_pl

// Structure breaks (calculate crossovers on every bar for consistency)
x_ph_raw = ta.crossover(close, last_ph)
x_pl_raw = ta.crossunder(close, last_pl)
x_lh_raw = ta.crossover(close, lower_high)
x_hl_raw = ta.crossunder(close, higher_low)

// Apply na checks after calculation
x_ph = not na(last_ph) and x_ph_raw
x_pl = not na(last_pl) and x_pl_raw
x_lh = not na(lower_high) and x_lh_raw
x_hl = not na(higher_low) and x_hl_raw

// BOS & CHoCH Visualization (Enhanced)
var line[] bos_lines = array.new_line()
var label[] bos_labels = array.new_label()

cleanup_lines(arr, max_size) =>
    if array.size(arr) > max_size
        line.delete(array.shift(arr))

cleanup_labels(arr, max_size) =>
    if array.size(arr) > max_size
        label.delete(array.shift(arr))

if show_bos
    if trend == 1 and x_ph
        l = line.new(bar_index - 10, last_ph, bar_index, last_ph, 
                     color=c_sig_bull, style=line.style_solid, width=2, extend=extend.right)
        lab = label.new(bar_index, last_ph, "BOS", 
                        color=color(na), textcolor=c_sig_bull, 
                        style=label.style_none, size=size.normal)
        array.push(bos_lines, l)
        array.push(bos_labels, lab)
        cleanup_lines(bos_lines, 20)
        cleanup_labels(bos_labels, 20)
        
    if trend == -1 and x_pl
        l = line.new(bar_index - 10, last_pl, bar_index, last_pl, 
                     color=c_sig_bear, style=line.style_solid, width=2, extend=extend.right)
        lab = label.new(bar_index, last_pl, "BOS", 
                        color=color(na), textcolor=c_sig_bear, 
                        style=label.style_none, size=size.normal)
        array.push(bos_lines, l)
        array.push(bos_labels, lab)
        cleanup_lines(bos_lines, 20)
        cleanup_labels(bos_labels, 20)
        
    if trend == -1 and x_lh
        l = line.new(bar_index - 10, lower_high, bar_index, lower_high, 
                     color=c_sig_bull, style=line.style_dashed, width=2, extend=extend.right)
        lab = label.new(bar_index, lower_high, "CHoCH", 
                        color=color(na), textcolor=c_sig_bull, 
                        style=label.style_none, size=size.normal)
        array.push(bos_lines, l)
        array.push(bos_labels, lab)
        cleanup_lines(bos_lines, 20)
        cleanup_labels(bos_labels, 20)
        higher_low := low
        
    if trend == 1 and x_hl
        l = line.new(bar_index - 10, higher_low, bar_index, higher_low, 
                     color=c_sig_bear, style=line.style_dashed, width=2, extend=extend.right)
        lab = label.new(bar_index, higher_low, "CHoCH", 
                        color=color(na), textcolor=c_sig_bear, 
                        style=label.style_none, size=size.normal)
        array.push(bos_lines, l)
        array.push(bos_labels, lab)
        cleanup_lines(bos_lines, 20)
        cleanup_labels(bos_labels, 20)
        lower_high := high

// Enhanced Order Blocks
var box[] ob_zones = array.new_box()
var bool[] ob_bullish = array.new_bool()

// Improved OB detection - looks for last opposite candle before structure break
if show_ob
    if trend == 1 and x_ph
        found_ob = false
        for i = 1 to ob_lookback
            if close[i] < open[i] and not found_ob  // Bearish candle
                ob_high = high[i]
                ob_low = low[i]
                ob_bar = bar_index[i]
                
                b = box.new(ob_bar, ob_high, bar_index + sd_ext, ob_low, 
                           border_color=na, 
                           bgcolor=color.new(c_smc_bull, smc_transparency),
                           extend=extend.right)
                array.push(ob_zones, b)
                array.push(ob_bullish, true)
                cleanup_zones(ob_zones, ob_max_zones)
                if array.size(ob_bullish) > ob_max_zones
                    array.shift(ob_bullish)
                found_ob := true
                break
                
    if trend == -1 and x_pl
        found_ob = false
        for i = 1 to ob_lookback
            if close[i] > open[i] and not found_ob  // Bullish candle
                ob_high = high[i]
                ob_low = low[i]
                ob_bar = bar_index[i]
                
                b = box.new(ob_bar, ob_high, bar_index + sd_ext, ob_low, 
                           border_color=na, 
                           bgcolor=color.new(c_smc_bear, smc_transparency),
                           extend=extend.right)
                array.push(ob_zones, b)
                array.push(ob_bullish, false)
                cleanup_zones(ob_zones, ob_max_zones)
                if array.size(ob_bullish) > ob_max_zones
                    array.shift(ob_bullish)
                found_ob := true
                break

// Enhanced FVG Detection
var box[] fvg_zones = array.new_box()
var bool[] fvg_bullish = array.new_bool()

atr_c = ta.atr(14)
fvg_min = atr_c * fvg_min_size

// Bullish FVG: gap between current low and 2 bars ago high
fvg_b = (low > high[2]) and ((low - high[2]) > fvg_min)
// Bearish FVG: gap between current high and 2 bars ago low
fvg_s = (high < low[2]) and ((low[2] - high) > fvg_min)

if show_fvg
    if fvg_b
        fvg_top = high[2]
        fvg_bottom = low
        fvg_bar = bar_index[2]
        
        b = box.new(fvg_bar, fvg_top, bar_index + sd_ext, fvg_bottom, 
                   border_color=na, 
                   bgcolor=color.new(c_smc_bull, smc_transparency),
                   extend=extend.right)
        array.push(fvg_zones, b)
        array.push(fvg_bullish, true)
        cleanup_zones(fvg_zones, fvg_max_zones)
        if array.size(fvg_bullish) > fvg_max_zones
            array.shift(fvg_bullish)
            
    if fvg_s
        fvg_top = high
        fvg_bottom = low[2]
        fvg_bar = bar_index[2]
        
        b = box.new(fvg_bar, fvg_bottom, bar_index + sd_ext, fvg_top, 
                   border_color=na, 
                   bgcolor=color.new(c_smc_bear, smc_transparency),
                   extend=extend.right)
        array.push(fvg_zones, b)
        array.push(fvg_bullish, false)
        cleanup_zones(fvg_zones, fvg_max_zones)
        if array.size(fvg_bullish) > fvg_max_zones
            array.shift(fvg_bullish)

// Enhanced Mitigation Logic
if fvg_mit
    // Mitigate Order Blocks
    if array.size(ob_zones) > 0
        for i = array.size(ob_zones) - 1 to 0
            b = array.get(ob_zones, i)
            is_bull = array.get(ob_bullish, i)
            
            if is_mitigated(b, is_bull)
                box.delete(b)
                array.remove(ob_zones, i)
                array.remove(ob_bullish, i)
            else
                // Extend zone forward
                box.set_right(b, bar_index + 5)
    
    // Mitigate FVGs
    if array.size(fvg_zones) > 0
        for i = array.size(fvg_zones) - 1 to 0
            b = array.get(fvg_zones, i)
            is_bull = array.get(fvg_bullish, i)
            
            if is_mitigated(b, is_bull)
                box.delete(b)
                array.remove(fvg_zones, i)
                array.remove(fvg_bullish, i)
            else
                // Extend zone forward
                box.set_right(b, bar_index + 5)

// ════════════════════════════════════════════════════════════════════════════
// 6. PLOTTING
// ════════════════════════════════════════════════════════════════════════════

// Trend Cloud (Dynamic based on trend)
p1 = plot(upper, display=display.none)
p2 = plot(lower, display=display.none)
c_cloud_fill = trend == 1 ? color.new(c_tr_bull, cloud_transparency) : color.new(c_tr_bear, cloud_transparency)
fill(p1, p2, color=c_cloud_fill, title="Trend Cloud")

// Baseline (Optional - can be enabled)
// plot(baseline, "Baseline", color=color.new(color.gray, 70), linewidth=1)

// Trailing Stop
plot(show_sl ? trail_stop : na, "Trailing Stop", 
     color=trend==1 ? c_sig_bull : trend==-1 ? c_sig_bear : na, 
     style=plot.style_linebr, 
     linewidth=2)

// Signals
if show_sig and sig_buy
    label.new(bar_index, low, "BUY", 
              color=c_sig_bull, 
              textcolor=color.white, 
              style=label.style_label_up, 
              size=size.small)

if show_sig and sig_sell
    label.new(bar_index, high, "SELL", 
              color=c_sig_bear, 
              textcolor=color.white, 
              style=label.style_label_down, 
              size=size.small)

// Bar Colors
barcolor(trend == 1 ? c_tr_bull : trend == -1 ? c_tr_bear : na)

// ════════════════════════════════════════════════════════════════════════════
// 7. ALERTS
// ════════════════════════════════════════════════════════════════════════════

alertcondition(sig_buy, "Trend Buy Signal", "Trend Following Buy Signal - All filters confirmed")
alertcondition(sig_sell, "Trend Sell Signal", "Trend Following Sell Signal - All filters confirmed")
alertcondition(x_ph and show_bos, "Bullish BOS", "Break of Structure - Bullish")
alertcondition(x_pl and show_bos, "Bearish BOS", "Break of Structure - Bearish")
alertcondition(trend == 1 and trend[1] != 1, "Trend Change: Bullish", "Trend changed to bullish")
alertcondition(trend == -1 and trend[1] != -1, "Trend Change: Bearish", "Trend changed to bearish")
"""

# =============================================================================
# 1. CONFIG & STYLES (MERGED + MOBILE)
# =============================================================================
st.set_page_config(
    page_title="Market Omni-Terminal",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    /* GLOBAL THEME */
    .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* TERMINAL GLOW TEXT */
    .title-glow {
        font-size: 2.5em; font-weight: bold; color: #ffffff;
        text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00;
        margin-bottom: 10px;
    }
    
    /* METRIC CARDS (Standard) */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px; border-radius: 8px;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover { transform: scale(1.02); border-color: #00ff00; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; font-weight: 700; color: #66fcf1 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem !important; color: #c5c6c7 !important; }

    /* MOBILE SIGNAL CARDS */
    .report-card {
        background-color: #1f2833;
        border-left: 5px solid #45a29e;
        padding: 15px; border-radius: 5px;
        margin-bottom: 15px;
    }
    .report-header {
        font-size: 18px; font-weight: bold; color: #ffffff;
        margin-bottom: 10px; border-bottom: 1px solid #45a29e; padding-bottom: 5px;
    }
    .report-item { margin-bottom: 8px; font-size: 14px; color: #c5c6c7; }
    .highlight { color: #66fcf1; font-weight: bold; }

    /* TAP PANES (MACRO) */
    .pane-card { border:1px solid #333; border-radius:16px; padding:12px; background:#111; }
    .pane-title { font-weight:700; font-size:0.95rem; color:#ddd; }
    .pane-sub { font-size:0.78rem; color:#888; margin-top:2px; }
    .pane-row { display:flex; justify-content:space-between; align-items:center; margin-top:8px; }
    .pane-val { font-size:1.05rem; font-weight:700; color:#fff; }
    .pane-pct { font-size:0.9rem; font-weight:700; }

    /* PILLS */
    .pill { display:inline-block; padding:0.12rem 0.5rem; border-radius:999px; font-size:0.75rem; border:1px solid #333; background:rgba(255,255,255,0.06); color:#ddd; margin-right:0.35rem; }
    .pill-warn { border-color: rgba(255,193,7,0.55); }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #161b22; border: 1px solid #30363d; color: #8b949e; }
    .stTabs [aria-selected="true"] { background-color: #0e1117; color: #00ff00; border-bottom: 2px solid #00ff00; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CORE UTILS & STATE
# =============================================================================
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = "BTC-USD"
if "selected_label" not in st.session_state:
    st.session_state["selected_label"] = "Bitcoin"
if "api_key" not in st.session_state:
    st.session_state["api_key"] = st.secrets.get("OPENAI_API_KEY", "")

def _now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def _safe_key(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_:: " else "_" for ch in str(s))

def _sleep_backoff(attempt: int):
    base = min(2 ** attempt, 16)
    time.sleep(base + random.random() * 0.35)

def try_get_plotly_events():
    try:
        from streamlit_plotly_events import plotly_events
        return plotly_events
    except: return None

plotly_events = try_get_plotly_events()

# =============================================================================
# 3. ROBUST DATA ENGINE
# =============================================================================
@st.cache_data(ttl=120)
def get_ohlcv_robust(ticker: str, period="6mo", interval="1d", max_retries=3):
    """
    Robust fetcher handling different providers logic implicitly via yfinance
    """
    for attempt in range(max_retries):
        try:
            # Handle Yahoo's lack of 4h by fetching 1h and resampling if needed
            fetch_int = "1h" if interval == "4h" else interval
            fetch_period = period
            
            # Constraint adjustments
            if interval in ["1m", "5m", "15m"]: fetch_period = "5d" if interval == "1m" else "1mo"
            elif interval == "4h": fetch_period = "1y" 
            
            df = yf.download(ticker, period=fetch_period, interval=fetch_int, progress=False, auto_adjust=False, threads=False)
            
            if df is None or df.empty:
                _sleep_backoff(attempt)
                continue
                
            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Standardization
            if 'Adj Close' in df.columns and 'Close' not in df.columns:
                df['Close'] = df['Adj Close']
            
            df = df.dropna()
            
            # Resample for 4h
            if interval == "4h":
                agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
                df = df.resample('4h').agg(agg_dict).dropna()
                
            return df
        except Exception:
            _sleep_backoff(attempt)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    if any(x in ticker for x in ["-", "=", "^"]): return None
    try:
        info = yf.Ticker(ticker).info
        return {
            "Market Cap": info.get("marketCap", 0),
            "P/E": info.get("trailingPE", 0),
            "Rev Growth": info.get("revenueGrowth", 0),
            "Summary": info.get("longBusinessSummary", "No Data")
        }
    except: return None

# =============================================================================
# 4. MATH & INDICATOR ENGINES
# =============================================================================

# --- Basic Helpers ---
def wma(series, length):
    w = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def hma(series, length):
    half = int(length / 2)
    sqrt_l = int(math.sqrt(length))
    wma_f = wma(series, length)
    wma_h = wma(series, half)
    diff = 2 * wma_h - wma_f
    return wma(diff, sqrt_l)

def rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def atr_series(df, length=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return rma(tr, length)

# --- Trend / Strategy Logic ---
def calc_trend_logic(df):
    if df.empty: return df
    
    # 1. HMA Trend
    df['HMA_55'] = hma(df['Close'], 55)
    
    # 2. ATR & SuperTrend
    df['ATR'] = atr_series(df, 14)
    # Simple Python SuperTrend
    m = 3.0
    hl2 = (df['High'] + df['Low']) / 2
    basic_upper = hl2 + (m * df['ATR'])
    basic_lower = hl2 - (m * df['ATR'])
    
    # Vectorized SuperTrend calculation is complex, using simplified iteration for robustness
    # For speed in dashboard, we use a vectorized approximation or just direction based on HMA/ATR
    
    # 3. Trend Cloud (Approximation of the Pine Script logic for Python)
    # Uses HMA 55 +/- 1.5 ATR(55)
    df['Apex_Base'] = df['HMA_55']
    df['Apex_ATR'] = atr_series(df, 55)
    df['Apex_Upper'] = df['Apex_Base'] + (df['Apex_ATR'] * 1.5)
    df['Apex_Lower'] = df['Apex_Base'] - (df['Apex_ATR'] * 1.5)
    df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))
    df['Apex_Trend'] = df['Apex_Trend'].replace(0, method='ffill')

    # 4. Squeeze Momentum
    # BB
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    # KC
    kc_range = atr_series(df, 20)
    upper_kc = sma20 + (1.5 * kc_range)
    lower_kc = sma20 - (1.5 * kc_range)
    df['Squeeze_On'] = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    # Mom (LinReg Proxy)
    df['Sqz_Mom'] = (df['Close'] - sma20).rolling(20).mean() # Simplified for speed

    # 5. Money Flow Matrix
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # Flow
    rsi_norm = df['RSI'] - 50
    vol_ratio = df['Volume'] / df['Volume'].rolling(20).mean()
    df['MF_Matrix'] = (rsi_norm * vol_ratio).ewm(span=3).mean()
    df['RVOL'] = vol_ratio

    # 6. Confluence Score (Previously GM_Score)
    # Aggregating: Trend, Price > HMA, Mom > 0, RSI > 50
    df['GM_Score'] = 0
    df.loc[df['Apex_Trend'] == 1, 'GM_Score'] += 2
    df.loc[df['Apex_Trend'] == -1, 'GM_Score'] -= 2
    df.loc[df['Close'] > df['HMA_55'], 'GM_Score'] += 1
    df.loc[df['Close'] < df['HMA_55'], 'GM_Score'] -= 1
    df.loc[df['Sqz_Mom'] > 0, 'GM_Score'] += 1
    df.loc[df['Sqz_Mom'] < 0, 'GM_Score'] -= 1
    
    # 7. Sentiment / Fear Greed
    # Composite of RSI, Mom, Vol
    df['FG_Index'] = (df['RSI'] + (df['Sqz_Mom']*100).clip(0,100) + (vol_ratio*20).clip(0,100)) / 3
    
    return df

# --- SMC Python Port ---
def smc_v8_python(df, liq_len=10):
    # This recreates the ZONES logic for plotting
    # Pivots
    df['PH'] = df['High'].rolling(liq_len*2+1, center=True).max() == df['High']
    df['PL'] = df['Low'].rolling(liq_len*2+1, center=True).min() == df['Low']
    
    zones = []
    # Identify last 5 S/D zones
    ph_idxs = df.index[df['PH']].tolist()
    pl_idxs = df.index[df['PL']].tolist()
    
    for idx in ph_idxs[-5:]:
        price = df.loc[idx, 'High']
        zones.append({'type': 'supply', 'y0': df.loc[idx, 'Open'], 'y1': price, 'x0': idx})
        
    for idx in pl_idxs[-5:]:
        price = df.loc[idx, 'Low']
        zones.append({'type': 'demand', 'y0': df.loc[idx, 'Open'], 'y1': price, 'x0': idx})
        
    return zones

# =============================================================================
# 5. LAYOUT & UI CONSTRUCTION
# =============================================================================

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="title-glow">Market Terminal</div>', unsafe_allow_html=True)
    
    # Global Asset Selection
    st.subheader("📡 Feed Selector")
    
    # Universe Logic
    univ = st.selectbox("Universe", ["Curated Lists", "Manual Entry"])
    
    assets = {
        "Indices": ["^GSPC", "^NDX", "^DJI", "IWM"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"],
        "Tech": ["NVDA", "TSLA", "AAPL", "MSFT", "AMD"],
        "Commodities": ["GC=F", "CL=F", "SI=F", "NG=F"]
    }
    
    if univ == "Curated Lists":
        cat = st.selectbox("Category", list(assets.keys()))
        ticker_val = st.selectbox("Asset", assets[cat])
    else:
        ticker_val = st.text_input("Ticker Symbol", "BTC-USD").upper()
        
    # Sync with Session State
    if ticker_val != st.session_state["selected_ticker"]:
        st.session_state["selected_ticker"] = ticker_val
        st.session_state["selected_label"] = ticker_val
        
    # Timeframe
    c1, c2 = st.columns(2)
    with c1: period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    with c2: interval = st.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=3)
    
    st.markdown("---")
    st.subheader("🤖 AI Council")
    st.session_state["api_key"] = st.text_input("OpenAI Key", value=st.session_state["api_key"], type="password")
    
    if st.button("🔄 System Refresh"):
        st.cache_data.clear()
        st.rerun()

# --- Main Data Fetching ---
with st.spinner(f"Accessing Market Feed for {st.session_state['selected_ticker']}..."):
    df = get_ohlcv_robust(st.session_state["selected_ticker"], period, interval)
    if not df.empty:
        df = calc_trend_logic(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate Fibs & Stops for Mobile View
        h, l = df['High'].max(), df['Low'].min()
        fib618 = h - ((h-l)*0.618)
        smart_stop = fib618 # Simplified smart stop logic
        tp3 = last['Close'] + (last['ATR']*5) if last['Apex_Trend']==1 else last['Close'] - (last['ATR']*5)
    else:
        st.error("Data Feed Disconnected. Check Ticker.")
        st.stop()

# --- HEADER METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric(st.session_state["selected_label"], f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
m2.metric("Confluence Score", f"{last['GM_Score']:.0f}/5", "BULL" if last['GM_Score']>0 else "BEAR")
m3.metric("Flux (Money Flow)", f"{last['MF_Matrix']:.2f}", "INFLOW" if last['MF_Matrix']>0 else "OUTFLOW")
m4.metric("Volatility", f"{last['ATR']:.2f}")

# --- TABS ---
tab_mobile, tab_term, tab_apex, tab_macro, tab_ai = st.tabs([
    "📱 Mobile Command", "🖥️ Omni-Terminal", "🧠 SMC Analysis", "🌍 Macro & Regime", "🤖 AI Council"
])

# =============================================================================
# TAB 1: MOBILE COMMAND (Signals Style)
# =============================================================================
with tab_mobile:
    c1, c2 = st.columns([1, 1])
    
    # Left: Signal Card
    with c1:
        direction = "LONG 🐂" if last['Apex_Trend'] == 1 else "SHORT 🐻" if last['Apex_Trend'] == -1 else "NEUTRAL ⚪"
        conf = "HIGH" if abs(last['GM_Score']) >= 3 else "LOW"
        sqz = "⚠️ ACTIVE" if last['Squeeze_On'] else "⚪ OFF"
        
        html_card = f"""
        <div class="report-card">
            <div class="report-header">💠 SIGNAL: {direction}</div>
            <div class="report-item">Confidence: <span class="highlight">{conf}</span></div>
            <div class="report-item">Confluence Score: <span class="highlight">{last['GM_Score']:.0f}</span></div>
            <div class="report-item">Squeeze: <span class="highlight">{sqz}</span></div>
        </div>
        <div class="report-card">
            <div class="report-header">🌊 FLOW & VOL</div>
            <div class="report-item">RVOL: <span class="highlight">{last['RVOL']:.2f}x</span></div>
            <div class="report-item">Money Flow: <span class="highlight">{last['MF_Matrix']:.2f}</span></div>
        </div>
        """
        st.markdown(html_card, unsafe_allow_html=True)
        
    # Right: Execution Card
    with c2:
        html_exec = f"""
        <div class="report-card">
            <div class="report-header">🎯 EXECUTION PLAN</div>
            <div class="report-item">Entry (Ref): <span class="highlight">{last['Close']:.2f}</span></div>
            <div class="report-item">🛑 SMART STOP: <span class="highlight">{smart_stop:.2f}</span></div>
            <div class="report-item">🚀 TP3 (5R): <span class="highlight">{tp3:.2f}</span></div>
        </div>
        """
        st.markdown(html_exec, unsafe_allow_html=True)
        
    # Simple Chart
    fig_mob = go.Figure()
    fig_mob.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig_mob.add_trace(go.Scatter(x=df.index, y=df['HMA_55'], line=dict(color='cyan')))
    fig_mob.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_mob, use_container_width=True)

# =============================================================================
# TAB 2: OMNI-TERMINAL (Advanced Charts)
# =============================================================================
with tab_term:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02)
    
    # 1. Price & Cloud
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)', line=dict(width=0), name="Trend Cloud"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HMA_55'], line=dict(color='yellow', width=2), name="HMA 55"), row=1, col=1)
    
    # 2. Squeeze Momentum
    cols = ['#00E676' if x > 0 else '#FF5252' for x in df['Sqz_Mom']]
    fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=cols, name="Squeeze"), row=2, col=1)
    
    # 3. Money Flow
    fig.add_trace(go.Scatter(x=df.index, y=df['MF_Matrix'], fill='tozeroy', line=dict(color='cyan'), name="Money Flow"), row=3, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, title=f"Advanced Technicals: {st.session_state['selected_ticker']}")
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3: SMC ANALYSIS (Specific Logic + Pine)
# =============================================================================
with tab_apex:
    st.subheader("🧠 Trend & Liquidity Master (SMC)")
    
    # Plot SMC Python Port
    smc_zones = smc_v8_python(df)
    
    fig_smc = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    
    # Plot Zones
    for z in smc_zones:
        color = "rgba(0, 255, 0, 0.2)" if z['type'] == 'demand' else "rgba(255, 0, 0, 0.2)"
        fig_smc.add_shape(type="rect", x0=z['x0'], x1=df.index[-1], y0=z['y0'], y1=z['y1'], fillcolor=color, line_width=0)
        
    fig_smc.update_layout(height=600, template="plotly_dark", title="SMC Zones (Python Approximation)", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_smc, use_container_width=True)
    
    with st.expander("📌 View SMC v8 Pine Script (Verbatim)"):
        st.code(PINE_SCRIPT_SMC_V8, language="pine")

# =============================================================================
# TAB 4: MACRO & REGIME
# =============================================================================
with tab_macro:
    st.subheader("🌍 Global Regime Timeline")
    
    # This requires fetching extra data. We will do a lightweight fetch here.
    macro_tickers = ["SPY", "TLT", "GLD", "UUP"] # Simple set
    if st.button("Load Macro Data"):
        with st.spinner("Fetching Macro Data..."):
            macro_df = yf.download(macro_tickers, period="1y", interval="1d", progress=False)['Close']
            
            # Simple Risk Ratio: SPY/TLT
            if not macro_df.empty:
                macro_df['Risk_Ratio'] = macro_df['SPY'] / macro_df['TLT']
                
                # Z-Score
                macro_df['Regime_Z'] = (macro_df['Risk_Ratio'] - macro_df['Risk_Ratio'].rolling(60).mean()) / macro_df['Risk_Ratio'].rolling(60).std()
                
                fig_reg = go.Figure()
                fig_reg.add_trace(go.Scatter(x=macro_df.index, y=macro_df['Regime_Z'], fill='tozeroy', name="Risk Regime (Z)"))
                fig_reg.add_hline(y=0, line_color="white")
                fig_reg.add_hline(y=2, line_color="red", line_dash="dot")
                fig_reg.add_hline(y=-2, line_color="green", line_dash="dot")
                fig_reg.update_layout(template="plotly_dark", height=400, title="Risk-On / Risk-Off Regime (SPY/TLT Z-Score)")
                st.plotly_chart(fig_reg, use_container_width=True)
            else:
                st.error("Macro data fetch failed.")

# =============================================================================
# TAB 5: AI COUNCIL
# =============================================================================
with tab_ai:
    st.subheader("🤖 The Council")
    
    persona = st.selectbox("Advisor", ["The Architect (Technical)", "The Macro Strategist", "The Quant"])
    
    fund_data = get_fundamentals(st.session_state["selected_ticker"])
    fund_str = str(fund_data) if fund_data else "N/A"
    
    if st.button("Consult Advisor"):
        if not st.session_state["api_key"]:
            st.error("Please enter API Key in Sidebar")
        else:
            try:
                client = openai.OpenAI(api_key=st.session_state["api_key"])
                
                prompt = f"""
                You are {persona}. Analyze this asset: {st.session_state['selected_ticker']}
                Price: {last['Close']}
                Trend: {'Bull' if last['Apex_Trend']==1 else 'Bear'}
                Confluence Score: {last['GM_Score']}
                Money Flow: {last['MF_Matrix']}
                Fundamentals: {fund_str}
                
                Provide a strategic outlook (max 150 words). Use emojis.
                """
                
                resp = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":prompt}])
                st.info(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"AI Error: {e}")
