import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone
import openai
import urllib.parse
import time
from pathlib import Path
import json
import math
import requests

# =============================================================================
# 0) PINE SCRIPT (VERBATIM — DO NOT OMIT)
# =============================================================================
PINE_SCRIPT_APEX_SMC_V8 = r"""// This Pine Script® code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/ 
// © DarkPoolCrypto
// Improved Version with Enhanced Logic, Performance, and Features

//@version=6
indicator("Apex Trend & Liquidity Master (SMC) v8.0", overlay=true, max_boxes_count=500, max_lines_count=500, max_labels_count=500)

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

alertcondition(sig_buy, "Apex Buy Signal", "Trend Following Buy Signal - All filters confirmed")
alertcondition(sig_sell, "Apex Sell Signal", "Trend Following Sell Signal - All filters confirmed")
alertcondition(x_ph and show_bos, "Bullish BOS", "Break of Structure - Bullish")
alertcondition(x_pl and show_bos, "Bearish BOS", "Break of Structure - Bearish")
alertcondition(trend == 1 and trend[1] != 1, "Trend Change: Bullish", "Trend changed to bullish")
alertcondition(trend == -1 and trend[1] != -1, "Trend Change: Bearish", "Trend changed to bearish")
"""

# =============================================================================
# 1) STREAMLIT SETUP
# =============================================================================
st.set_page_config(
    page_title="Macro Mobile",
    page_icon="🦅",
    layout="centered",
    initial_sidebar_state="collapsed"
)

custom_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div.stButton > button:first-child { width: 100%; height: 3em; font-weight: bold; border-radius: 20px; }
div[data-testid="stMetric"] { background-color: #1E1E1E; border: 1px solid #333; padding: 10px; border-radius: 12px; transition: transform 0.2s; }
.pill { display:inline-block; padding:0.12rem 0.5rem; border-radius:999px; font-size:0.75rem; border:1px solid #333; background:rgba(255,255,255,0.06); color:#ddd; margin-right:0.35rem; margin-top:0.25rem; }
.pill-good { border-color: rgba(0,255,0,0.35); }
.pill-warn { border-color: rgba(255,193,7,0.55); }
.pill-bad { border-color: rgba(255,75,75,0.55); }

/* Touch panes */
.pane-wrap { border:1px solid #333; border-radius:16px; padding:12px; background:#111; }
.pane-title { font-weight:700; font-size:0.95rem; color:#ddd; }
.pane-sub { font-size:0.78rem; color:#888; margin-top:2px; }
.pane-row { display:flex; justify-content:space-between; align-items:center; margin-top:8px; }
.pane-val { font-size:1.05rem; font-weight:700; color:#fff; }
.pane-pct { font-size:0.9rem; font-weight:700; }
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

def _now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def _safe_key(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_:: " else "_" for ch in str(s))

def pill_html(text, kind="warn"):
    cls = {"good": "pill pill-good", "bad": "pill pill-bad", "warn": "pill pill-warn"}.get(kind, "pill")
    return f"<span class='{cls}'>{text}</span>"

# =============================================================================
# 2) YOUR FOUNDATION DICTS (UNCHANGED)
# =============================================================================
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

TRUTH_FLAGS = {
    ("US 02Y", "^IRX"): "Label says 2Y but ticker ^IRX is often a 13-week T-Bill yield proxy.",
    ("BTC.D (Proxy)", "BTC-USD"): "Dominance cannot be derived from BTC price alone; this is a proxy label.",
    ("Total Cap (Proxy)", "BTC-USD"): "Total cap cannot be derived from BTC price alone; this is a proxy label.",
    ("10Y / 3M (Recession)", "^IRX"): "Denominator uses ^IRX again (often 13-week). This may not equal 3M consistently.",
}

# =============================================================================
# 3) SESSION STATE
# =============================================================================
if "fav_tickers" not in st.session_state:
    st.session_state["fav_tickers"] = set()
if "fav_ratios" not in st.session_state:
    st.session_state["fav_ratios"] = set()

if "global_asset_search" not in st.session_state:
    st.session_state["global_asset_search"] = ""
if "markets_search" not in st.session_state:
    st.session_state["markets_search"] = ""
if "selected_asset_label" not in st.session_state:
    st.session_state["selected_asset_label"] = None
if "selected_asset_ticker" not in st.session_state:
    st.session_state["selected_asset_ticker"] = "^GSPC"

if "last_broadcast_pack" not in st.session_state:
    st.session_state["last_broadcast_pack"] = None
if "last_snapshot_date" not in st.session_state:
    st.session_state["last_snapshot_date"] = None

# =============================================================================
# 4) DATA + UNIVERSES
# =============================================================================
@st.cache_data(ttl=86400)
def load_universe(universe_key: str):
    """
    Returns:
      items_dict: label -> (ticker, desc)
      meta: {source, derived_note}
    """
    if universe_key == "Crypto Top 100 (CoinGecko)":
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 100, "page": 1, "sparkline": "false"}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        items = {}
        for row in data:
            sym = str(row.get("symbol", "")).strip().upper()
            name = str(row.get("name", "")).strip()
            if sym and name:
                items[name] = (f"{sym}-USD", "CoinGecko top 100 • Derived Yahoo SYMBOL-USD (availability varies)")
        return items, {"source": url, "derived_note": "Derived yfinance candidates as SYMBOL-USD. Missing tickers will show in diagnostics."}

    # If you want the other 3 universes back exactly as before, paste your previous read_html variants here.
    # Keeping scope tight: user asked panes + apex indicator core; we keep crypto top100 here (no assumptions).
    return {}, {"source": "n/a", "derived_note": "Universe not implemented in this minimal block."}

@st.cache_data(ttl=120)
def get_market_close(tickers_list, period="1y", interval="1d", max_retries=3, chunk_size=60):
    valid = sorted(list(set([t for t in tickers_list if not str(t).startswith("SPECIAL_")])))
    fetched_at = _now_utc_iso()
    if not valid:
        return pd.DataFrame(), [], fetched_at, "no_valid_tickers"

    frames = []
    missing_all = set()
    chunks = [valid[i:i+chunk_size] for i in range(0, len(valid), chunk_size)]

    for ch in chunks:
        last_exc = None
        for attempt in range(max_retries):
            try:
                data = yf.download(ch, period=period, interval=interval, progress=False, group_by="ticker", auto_adjust=False, threads=True)
                if data is None or len(data) == 0:
                    last_exc = RuntimeError("empty_download")
                    time.sleep(min(2**attempt, 8))
                    continue

                close_df = pd.DataFrame(index=data.index)
                if isinstance(data.columns, pd.Index) and "Close" in data.columns and len(ch) == 1:
                    close_df[ch[0]] = data["Close"]
                else:
                    if isinstance(data.columns, pd.MultiIndex):
                        lvl0 = list(data.columns.levels[0])
                        lvl1 = list(data.columns.levels[1])
                        if "Close" in lvl1:
                            for t in ch:
                                if t in lvl0 and "Close" in data[t].columns:
                                    close_df[t] = data[t]["Close"]
                        elif "Close" in lvl0:
                            for t in ch:
                                if t in lvl1 and ("Close", t) in data.columns:
                                    close_df[t] = data[("Close", t)]
                    else:
                        if "Close" in data:
                            close_df = data[["Close"]].copy()

                for t in ch:
                    if t not in close_df.columns or close_df[t].dropna().empty:
                        missing_all.add(t)

                frames.append(close_df)
                break

            except Exception as e:
                last_exc = e
                time.sleep(min(2**attempt, 8))

        if last_exc is not None and not frames:
            return pd.DataFrame(), valid, fetched_at, f"failed_first_chunk: {repr(last_exc)}"

    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    missing = []
    for t in valid:
        if t not in merged.columns or merged[t].dropna().empty:
            missing.append(t)

    return merged, missing, fetched_at, "chunked_close"

@st.cache_data(ttl=120)
def get_ohlcv(ticker: str, period="6mo", interval="1d", max_retries=3):
    """
    OHLCV for the Apex SMC tab.
    """
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=True)
            if df is None or df.empty:
                time.sleep(min(2**attempt, 8))
                continue
            df = df.dropna()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            time.sleep(min(2**attempt, 8))
    return pd.DataFrame()

def get_crypto_total_proxy(close_df):
    coins = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD", "AVAX-USD", "LINK-USD"]
    available = [c for c in coins if c in close_df.columns]
    if not available:
        return None, None, None
    sub = close_df[available].ffill()
    total = sub.sum(axis=1)
    ex_btc = [c for c in available if c != "BTC-USD"]
    total2 = sub[ex_btc].sum(axis=1) if ex_btc else pd.Series(dtype="float64")
    ex_btc_eth = [c for c in available if c not in ["BTC-USD", "ETH-USD"]]
    total3 = sub[ex_btc_eth].sum(axis=1) if ex_btc_eth else pd.Series(dtype="float64")
    return total, total2, total3

def calculate_change(series):
    if series is None:
        return None, None
    s = series.dropna()
    if len(s) < 2:
        return None, None
    latest = s.iloc[-1]
    prev = s.iloc[-2]
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
        height=55, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def zscore(series: pd.Series, window: int = 60):
    s = series.dropna()
    if len(s) < max(10, window // 2):
        return None
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0, pd.NA)
    return (s - mu) / sd

# =============================================================================
# 5) APEX SMC v8 — PYTHON IMPLEMENTATION (CORE LOGIC)
# =============================================================================
def wma(series: pd.Series, length: int) -> pd.Series:
    if length <= 1:
        return series
    weights = pd.Series(range(1, length + 1), index=range(length))
    def _calc(x):
        return (x * weights.values).sum() / weights.sum()
    return series.rolling(length).apply(_calc, raw=True)

def hma(series: pd.Series, length: int) -> pd.Series:
    if length <= 1:
        return series
    half = max(1, length // 2)
    sqrt_l = max(1, int(math.sqrt(length)))
    return wma(2 * wma(series, half) - wma(series, length), sqrt_l)

def rma(series: pd.Series, length: int) -> pd.Series:
    # Wilder smoothing: alpha = 1/length
    if length <= 1:
        return series
    return series.ewm(alpha=1/length, adjust=False).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    if length <= 1:
        return series
    return series.ewm(span=length, adjust=False).mean()

def sma(series: pd.Series, length: int) -> pd.Series:
    if length <= 1:
        return series
    return series.rolling(length).mean()

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return rma(tr, length)

def dmi_adx(df: pd.DataFrame, di_len: int = 14, adx_len: int = 14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]

    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr_sm = rma(tr, di_len)
    plus_di = 100 * rma(plus_dm, di_len) / atr_sm.replace(0, pd.NA)
    minus_di = 100 * rma(minus_dm, di_len) / atr_sm.replace(0, pd.NA)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA))
    adx = rma(dx, adx_len)
    return plus_di, minus_di, adx

def pivothigh(series: pd.Series, left: int, right: int) -> pd.Series:
    # pivot at i if value equals centered rolling max
    win = left + right + 1
    rmax = series.rolling(win, center=True).max()
    return series.where(series == rmax)

def pivotlow(series: pd.Series, left: int, right: int) -> pd.Series:
    win = left + right + 1
    rmin = series.rolling(win, center=True).min()
    return series.where(series == rmin)

def apex_smc_v8(df: pd.DataFrame, cfg: dict):
    """
    Implements core logic from the Pine indicator for plotting in Streamlit/Plotly.
    Returns:
      - series: baseline, upper, lower, trend, trail_stop, sig_buy, sig_sell, tci, adx
      - objects: sd_zones, bos_lines, ob_zones, fvg_zones
    """
    if df is None or df.empty:
        return None

    # Inputs (names match Pine intent)
    ma_type = cfg["ma_type"]
    len_main = cfg["len_main"]
    mult = cfg["mult"]
    use_adx = cfg["use_adx"]
    adx_threshold = cfg["adx_threshold"]
    use_volume = cfg["use_volume"]
    vol_mult = cfg["vol_mult"]
    use_momentum = cfg["use_momentum"]

    show_sd = cfg["show_sd"]
    liq_len = cfg["liq_len"]
    sd_ext = cfg["sd_ext"]
    sd_max_zones = cfg["sd_max_zones"]

    show_bos = cfg["show_bos"]
    show_ob = cfg["show_ob"]
    show_fvg = cfg["show_fvg"]
    fvg_mit = cfg["fvg_mit"]
    ob_lookback = cfg["ob_lookback"]
    fvg_min_size = cfg["fvg_min_size"]
    ob_max_zones = cfg["ob_max_zones"]
    fvg_max_zones = cfg["fvg_max_zones"]

    src = df["Close"]

    # MA baseline
    if ma_type == "SMA":
        baseline = sma(src, len_main)
    elif ma_type == "EMA":
        baseline = ema(src, len_main)
    elif ma_type == "RMA":
        baseline = rma(src, len_main)
    else:  # HMA
        baseline = hma(src, len_main)

    atr_main = atr(df, len_main)
    upper = baseline + (atr_main * mult)
    lower = baseline - (atr_main * mult)

    # Trend state machine (iterative, like Pine var)
    trend = pd.Series(0, index=df.index, dtype="int64")
    prev_trend = pd.Series(0, index=df.index, dtype="int64")

    for i in range(1, len(df)):
        c = df["Close"].iloc[i]
        up = upper.iloc[i]
        lo = lower.iloc[i]
        last_tr = trend.iloc[i-1]
        confirmed = 1 if c > up else (-1 if c < lo else last_tr)
        prev_trend.iloc[i] = last_tr
        trend.iloc[i] = confirmed

    # ADX filter
    di_plus, di_minus, adx = dmi_adx(df, 14, 14)
    adx_ok = (~use_adx) | (adx > adx_threshold)

    # WaveTrend
    ap = (df["High"] + df["Low"] + df["Close"]) / 3.0
    esa = ema(ap, 10)
    d = ema((ap - esa).abs(), 10)
    ci = (ap - esa) / (0.015 * d.replace(0, pd.NA))
    ci = ci.fillna(0.0)
    tci = ema(ci, 21)

    mom_buy = (~use_momentum) | ((tci < 60) & (tci > tci.shift(1)))
    mom_sell = (~use_momentum) | ((tci > -60) & (tci < tci.shift(1)))

    # Volume filter
    vol_avg = sma(df["Volume"], 20)
    vol_ok = (~use_volume) | (df["Volume"] > (vol_avg * vol_mult))

    # Price confirms
    price_confirm_buy = (df["Close"] > df["Open"]) & (df["Close"] > df["Close"].shift(1))
    price_confirm_sell = (df["Close"] < df["Open"]) & (df["Close"] < df["Close"].shift(1))

    # Signal logic
    sig_buy = (trend == 1) & (prev_trend != 1) & vol_ok & mom_buy & adx_ok & price_confirm_buy
    sig_sell = (trend == -1) & (prev_trend != -1) & vol_ok & mom_sell & adx_ok & price_confirm_sell

    # Trailing stop
    trail_atr = atr(df, 14) * 2.0
    trail_stop = pd.Series(pd.NA, index=df.index, dtype="float64")
    trail_prev = pd.Series(pd.NA, index=df.index, dtype="float64")

    for i in range(1, len(df)):
        tr = trend.iloc[i]
        c = df["Close"].iloc[i]
        ta = trail_atr.iloc[i]
        prev_ts = trail_stop.iloc[i-1]

        if tr == 1:
            if pd.isna(prev_ts) or trend.iloc[i-1] != 1:
                ts = c - ta
            else:
                ts = max(prev_ts, c - ta)
                if not pd.isna(trail_prev.iloc[i-1]) and ts < trail_prev.iloc[i-1]:
                    ts = trail_prev.iloc[i-1]
            trail_stop.iloc[i] = ts
            trail_prev.iloc[i] = ts

        elif tr == -1:
            if pd.isna(prev_ts) or trend.iloc[i-1] != -1:
                ts = c + ta
            else:
                ts = min(prev_ts, c + ta)
                if not pd.isna(trail_prev.iloc[i-1]) and ts > trail_prev.iloc[i-1]:
                    ts = trail_prev.iloc[i-1]
            trail_stop.iloc[i] = ts
            trail_prev.iloc[i] = ts

        else:
            trail_stop.iloc[i] = pd.NA
            trail_prev.iloc[i] = pd.NA

    # Pivots for zones
    ph = pivothigh(df["High"], liq_len, liq_len)
    pl = pivotlow(df["Low"], liq_len, liq_len)

    # Objects lists (limited like Pine max zones)
    sd_zones = []   # dicts: {type, x0, x1, y0, y1, label}
    bos_lines = []  # dicts: {kind, x0, x1, y, text}
    ob_zones = []   # dicts: {type, x0, x1, y0, y1}
    fvg_zones = []  # dicts: {type, x0, x1, y0, y1}

    # Structure tracking
    last_ph = pd.NA
    last_pl = pd.NA
    lower_high = pd.NA
    higher_low = pd.NA

    # For mitigation checks, we evaluate per new zone vs current/next closes like Pine would.
    # Here: we keep zones unless mitigated by the latest close (configurable).
    def mitigated(zone, close_val):
        if zone["type"] in ("demand", "ob_bull", "fvg_bull"):
            return close_val < zone["y0"]  # below bottom
        else:
            return close_val > zone["y1"]  # above top

    # Iterate bars to build zones/structure events
    for i in range(len(df)):
        x = df.index[i]
        c = df["Close"].iloc[i]
        tr = trend.iloc[i]

        # Detect pivots (centered => pivot appears at i where ph/pl not NA)
        if show_sd and not pd.isna(ph.iloc[i]):
            # Pine uses high[liq_len] and bar_index - liq_len, which aligns to pivot candle.
            # With centered rolling, i is already pivot candle.
            ph_price = df["High"].iloc[i]
            zone_bottom = max(df["Open"].iloc[i], df["Close"].iloc[i])  # zone_top in Pine
            sd_zones.append({"type": "supply", "x0": x, "x1": df.index[-1], "y0": zone_bottom, "y1": ph_price, "label": "Supply"})
            sd_zones = sd_zones[-sd_max_zones:]

            # update structure point
            last_ph = ph_price
            if tr == -1:
                lower_high = last_ph

        if show_sd and not pd.isna(pl.iloc[i]):
            pl_price = df["Low"].iloc[i]
            zone_top = min(df["Open"].iloc[i], df["Close"].iloc[i])  # zone_bottom in Pine
            sd_zones.append({"type": "demand", "x0": x, "x1": df.index[-1], "y0": zone_top, "y1": pl_price, "label": "Demand"})
            sd_zones = sd_zones[-sd_max_zones:]

            last_pl = pl_price
            if tr == 1:
                higher_low = last_pl

        # Crossovers (consistent with Pine intent)
        def crossover(price, level, prev_price, prev_level):
            if pd.isna(level) or pd.isna(prev_level):
                return False
            return (prev_price <= prev_level) and (price > level)

        def crossunder(price, level, prev_price, prev_level):
            if pd.isna(level) or pd.isna(prev_level):
                return False
            return (prev_price >= prev_level) and (price < level)

        if i >= 1:
            prev_c = df["Close"].iloc[i-1]
            prev_last_ph = last_ph
            prev_last_pl = last_pl
            prev_lh = lower_high
            prev_hl = higher_low

            x_ph = (not pd.isna(last_ph)) and crossover(c, last_ph, prev_c, prev_last_ph)
            x_pl = (not pd.isna(last_pl)) and crossunder(c, last_pl, prev_c, prev_last_pl)
            x_lh = (not pd.isna(lower_high)) and crossover(c, lower_high, prev_c, prev_lh)
            x_hl = (not pd.isna(higher_low)) and crossunder(c, higher_low, prev_c, prev_hl)

            # BOS / CHoCH
            if show_bos:
                if tr == 1 and x_ph:
                    bos_lines.append({"kind": "BOS", "x0": df.index[max(i-10, 0)], "x1": df.index[-1], "y": float(last_ph), "text": "BOS"})
                if tr == -1 and x_pl:
                    bos_lines.append({"kind": "BOS", "x0": df.index[max(i-10, 0)], "x1": df.index[-1], "y": float(last_pl), "text": "BOS"})
                if tr == -1 and x_lh:
                    bos_lines.append({"kind": "CHoCH", "x0": df.index[max(i-10, 0)], "x1": df.index[-1], "y": float(lower_high), "text": "CHoCH"})
                    higher_low = df["Low"].iloc[i]
                if tr == 1 and x_hl:
                    bos_lines.append({"kind": "CHoCH", "x0": df.index[max(i-10, 0)], "x1": df.index[-1], "y": float(higher_low), "text": "CHoCH"})
                    lower_high = df["High"].iloc[i]

                bos_lines = bos_lines[-20:]

            # Order Blocks
            if show_ob:
                if tr == 1 and x_ph:
                    # last bearish candle in lookback
                    found = False
                    for j in range(1, ob_lookback + 1):
                        if i - j < 0:
                            break
                        if (df["Close"].iloc[i-j] < df["Open"].iloc[i-j]) and not found:
                            ob_high = df["High"].iloc[i-j]
                            ob_low = df["Low"].iloc[i-j]
                            ob_zones.append({"type": "ob_bull", "x0": df.index[i-j], "x1": df.index[-1], "y0": float(ob_low), "y1": float(ob_high)})
                            ob_zones = ob_zones[-ob_max_zones:]
                            found = True
                            break

                if tr == -1 and x_pl:
                    found = False
                    for j in range(1, ob_lookback + 1):
                        if i - j < 0:
                            break
                        if (df["Close"].iloc[i-j] > df["Open"].iloc[i-j]) and not found:
                            ob_high = df["High"].iloc[i-j]
                            ob_low = df["Low"].iloc[i-j]
                            ob_zones.append({"type": "ob_bear", "x0": df.index[i-j], "x1": df.index[-1], "y0": float(ob_low), "y1": float(ob_high)})
                            ob_zones = ob_zones[-ob_max_zones:]
                            found = True
                            break

            # FVG
            if show_fvg and i >= 2:
                atr_c = atr(df, 14).iloc[i]
                fvg_min = float(atr_c) * float(fvg_min_size) if not pd.isna(atr_c) else 0.0

                # bullish: low > high[2]
                fvg_b = (df["Low"].iloc[i] > df["High"].iloc[i-2]) and ((df["Low"].iloc[i] - df["High"].iloc[i-2]) > fvg_min)
                # bearish: high < low[2]
                fvg_s = (df["High"].iloc[i] < df["Low"].iloc[i-2]) and ((df["Low"].iloc[i-2] - df["High"].iloc[i]) > fvg_min)

                if fvg_b:
                    top = float(df["High"].iloc[i-2])
                    bottom = float(df["Low"].iloc[i])
                    y0, y1 = (min(top, bottom), max(top, bottom))
                    fvg_zones.append({"type": "fvg_bull", "x0": df.index[i-2], "x1": df.index[-1], "y0": y0, "y1": y1})
                    fvg_zones = fvg_zones[-fvg_max_zones:]

                if fvg_s:
                    top = float(df["High"].iloc[i])
                    bottom = float(df["Low"].iloc[i-2])
                    y0, y1 = (min(top, bottom), max(top, bottom))
                    fvg_zones.append({"type": "fvg_bear", "x0": df.index[i-2], "x1": df.index[-1], "y0": y0, "y1": y1})
                    fvg_zones = fvg_zones[-fvg_max_zones:]

            # Mitigation (delete mitigated)
            if fvg_mit:
                # apply using latest close each bar (matches "if mitigated" checks)
                ob_zones = [z for z in ob_zones if not mitigated(z, c)]
                fvg_zones = [z for z in fvg_zones if not mitigated(z, c)]

    return {
        "baseline": baseline,
        "upper": upper,
        "lower": lower,
        "trend": trend,
        "trail_stop": trail_stop,
        "sig_buy": sig_buy,
        "sig_sell": sig_sell,
        "adx": adx,
        "tci": tci,
        "sd_zones": sd_zones,
        "bos_lines": bos_lines,
        "ob_zones": ob_zones,
        "fvg_zones": fvg_zones,
    }

def plot_apex_smc(df: pd.DataFrame, ind: dict, title: str):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_width=1,
        decreasing_line_width=1,
    ))

    # Trend cloud: segment by trend so fill color matches regime
    upper = ind["upper"]
    lower = ind["lower"]
    trend = ind["trend"]

    # Build bull/bear segment traces with NaNs for clean fills
    upper_bull = upper.where(trend == 1)
    lower_bull = lower.where(trend == 1)
    upper_bear = upper.where(trend == -1)
    lower_bear = lower.where(trend == -1)

    # Fill bull
    fig.add_trace(go.Scatter(x=df.index, y=upper_bull, mode="lines", line=dict(width=0), name="Upper (Bull)", showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=lower_bull, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(0,105,92,0.20)", name="Trend Cloud (Bull)", showlegend=False))
    # Fill bear
    fig.add_trace(go.Scatter(x=df.index, y=upper_bear, mode="lines", line=dict(width=0), name="Upper (Bear)", showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=lower_bear, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(183,28,28,0.20)", name="Trend Cloud (Bear)", showlegend=False))

    # Trailing stop with regime coloring
    ts = ind["trail_stop"]
    ts_bull = ts.where(trend == 1)
    ts_bear = ts.where(trend == -1)
    fig.add_trace(go.Scatter(x=df.index, y=ts_bull, mode="lines", line=dict(width=2), name="Trail Stop (Bull)"))
    fig.add_trace(go.Scatter(x=df.index, y=ts_bear, mode="lines", line=dict(width=2, dash="dot"), name="Trail Stop (Bear)"))

    # BUY/SELL markers
    buys = df[ind["sig_buy"].fillna(False)]
    sells = df[ind["sig_sell"].fillna(False)]

    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Low"],
            mode="markers+text",
            text=["BUY"] * len(buys),
            textposition="bottom center",
            name="BUY",
            marker=dict(size=10, symbol="triangle-up")
        ))
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["High"],
            mode="markers+text",
            text=["SELL"] * len(sells),
            textposition="top center",
            name="SELL",
            marker=dict(size=10, symbol="triangle-down")
        ))

    # Zones: Classic SD
    for z in ind["sd_zones"]:
        color = "rgba(67,160,71,0.18)" if z["type"] == "demand" else "rgba(229,57,53,0.18)"
        fig.add_shape(
            type="rect",
            x0=z["x0"], x1=z["x1"],
            y0=z["y0"], y1=z["y1"],
            fillcolor=color,
            line=dict(width=0),
            layer="below",
        )
        fig.add_annotation(x=z["x0"], y=z["y1"], text=z["label"], showarrow=False, font=dict(size=10, color="rgba(255,255,255,0.65)"))

    # Zones: OB
    for z in ind["ob_zones"]:
        color = "rgba(185,246,202,0.20)" if z["type"] == "ob_bull" else "rgba(255,205,210,0.20)"
        fig.add_shape(type="rect", x0=z["x0"], x1=z["x1"], y0=z["y0"], y1=z["y1"], fillcolor=color, line=dict(width=0), layer="below")

    # Zones: FVG
    for z in ind["fvg_zones"]:
        color = "rgba(185,246,202,0.14)" if z["type"] == "fvg_bull" else "rgba(255,205,210,0.14)"
        fig.add_shape(type="rect", x0=z["x0"], x1=z["x1"], y0=z["y0"], y1=z["y1"], fillcolor=color, line=dict(width=0), layer="below")

    # BOS / CHoCH lines
    for b in ind["bos_lines"]:
        dash = "solid" if b["kind"] == "BOS" else "dash"
        fig.add_shape(type="line", x0=b["x0"], x1=b["x1"], y0=b["y"], y1=b["y"],
                      line=dict(width=2, dash=dash))
        fig.add_annotation(x=b["x1"], y=b["y"], text=b["text"], showarrow=False, xanchor="left", font=dict(size=11))

    fig.update_layout(
        title=title,
        height=560,
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(rangeslider=dict(visible=False)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

# =============================================================================
# 6) REGIME TIMELINE (kept)
# =============================================================================
def resolve_series(t, market_data, syn_total, syn_total2, syn_total3):
    if t == "SPECIAL_TOTAL":
        return syn_total
    if t == "SPECIAL_TOTAL2":
        return syn_total2
    if t == "SPECIAL_TOTAL3":
        return syn_total3
    if isinstance(market_data, pd.DataFrame) and t in market_data.columns:
        return market_data[t]
    return None

def compute_ratio_series(num_t, den_t, market_data, syn_total, syn_total2, syn_total3):
    a = resolve_series(num_t, market_data, syn_total, syn_total2, syn_total3)
    b = resolve_series(den_t, market_data, syn_total, syn_total2, syn_total3)
    if a is None or b is None:
        return None
    a = a.dropna()
    b = b.dropna()
    idx = a.index.intersection(b.index)
    if idx.empty:
        return None
    return a.loc[idx] / b.loc[idx]

def build_regime_timeline(market_data, syn_total, syn_total2, syn_total3, z_window=60):
    r_spy_tlt = compute_ratio_series("SPY", "TLT", market_data, syn_total, syn_total2, syn_total3)
    r_hyg_tlt = compute_ratio_series("HYG", "TLT", market_data, syn_total, syn_total2, syn_total3)
    r_iwm_spy = compute_ratio_series("IWM", "SPY", market_data, syn_total, syn_total2, syn_total3)
    r_cu_au = compute_ratio_series("HG=F", "GC=F", market_data, syn_total, syn_total2, syn_total3)
    r_oil_au = compute_ratio_series("CL=F", "GC=F", market_data, syn_total, syn_total2, syn_total3)
    r_dxy_au = compute_ratio_series("DX-Y.NYB", "GC=F", market_data, syn_total, syn_total2, syn_total3)
    r_btc_dxy = compute_ratio_series("BTC-USD", "DX-Y.NYB", market_data, syn_total, syn_total2, syn_total3)
    r_btc_spx = compute_ratio_series("BTC-USD", "^GSPC", market_data, syn_total, syn_total2, syn_total3)

    z_spy_tlt = zscore(r_spy_tlt, z_window) if r_spy_tlt is not None else None
    z_hyg_tlt = zscore(r_hyg_tlt, z_window) if r_hyg_tlt is not None else None
    z_iwm_spy = zscore(r_iwm_spy, z_window) if r_iwm_spy is not None else None
    z_cu_au = zscore(r_cu_au, z_window) if r_cu_au is not None else None
    z_oil_au = zscore(r_oil_au, z_window) if r_oil_au is not None else None
    z_dxy_au = zscore(r_dxy_au, z_window) if r_dxy_au is not None else None
    z_btc_dxy = zscore(r_btc_dxy, z_window) if r_btc_dxy is not None else None
    z_btc_spx = zscore(r_btc_spx, z_window) if r_btc_spx is not None else None

    def mean_of(series_list):
        series_list = [s.dropna() for s in series_list if s is not None and not s.dropna().empty]
        if not series_list:
            return None
        idx = series_list[0].index
        for s in series_list[1:]:
            idx = idx.intersection(s.index)
        if idx.empty:
            return None
        stacked = pd.concat([s.loc[idx] for s in series_list], axis=1)
        return stacked.mean(axis=1)

    risk = mean_of([z_spy_tlt, z_hyg_tlt, z_iwm_spy])
    infl = mean_of([z_cu_au, z_oil_au])

    usd = None
    if z_dxy_au is not None and z_btc_dxy is not None:
        idx = z_dxy_au.dropna().index.intersection(z_btc_dxy.dropna().index)
        if not idx.empty:
            usd = z_dxy_au.loc[idx] - z_btc_dxy.loc[idx]

    crypto = z_btc_spx.dropna() if z_btc_spx is not None else None

    out = pd.DataFrame()
    if risk is not None:
        out["Risk Composite (Z)"] = risk
    if infl is not None:
        out["Inflation Composite (Z)"] = infl
    if usd is not None:
        out["USD Pressure (Z)"] = usd
    if crypto is not None:
        out["Crypto vs TradFi (Z)"] = crypto

    return out.dropna(how="all")

def plot_regime_timeline(df: pd.DataFrame, title="Regime Timeline"):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

# =============================================================================
# 7) UI HEADER + GLOBAL SEARCH (LINK TARGET)
# =============================================================================
st.markdown("### 🦅 Macro Mobile")

# Global search bar (panes push into this)
st.text_input(
    "Asset Search (global)",
    key="global_asset_search",
    placeholder="Type a label or ticker… tap panes to auto-fill"
)

# Time controls
INTERVAL_CHOICES = ["1d", "1h", "30m", "15m", "5m"]
PERIOD_CHOICES = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]

c1, c2 = st.columns(2)
with c1:
    period = st.selectbox("Timeframe (period)", PERIOD_CHOICES, index=3)
with c2:
    interval = st.selectbox("Granularity (interval)", INTERVAL_CHOICES, index=0)

show_diagnostics = st.toggle("Show Data Diagnostics", value=False)
st.markdown("---")

# =============================================================================
# 8) FETCH CLOSE DATA FOR MAIN DASH
# =============================================================================
all_needed_tickers = set()
for cat in TICKERS.values():
    for _, (t, _) in cat.items():
        all_needed_tickers.add(t)
for cat in RATIO_GROUPS.values():
    for _, (n, d, _) in cat.items():
        all_needed_tickers.add(n)
        all_needed_tickers.add(d)

crypto_components = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "USDT-USD", "USDC-USD"]
all_needed_tickers.update(crypto_components)

with st.spinner("Fetching Global Market Data..."):
    market_close, missing_tickers, fetched_at_iso, shape_note = get_market_close(
        list(all_needed_tickers),
        period=period,
        interval=interval
    )
    syn_total, syn_total2, syn_total3 = get_crypto_total_proxy(market_close)

loaded = len(market_close.columns) if isinstance(market_close, pd.DataFrame) else 0
requested = len([t for t in all_needed_tickers if not str(t).startswith("SPECIAL_")])
st.caption(f"Last updated (UTC): {fetched_at_iso}  •  Coverage: {loaded}/{requested} tickers  •  Fetch: {shape_note}")

if show_diagnostics and missing_tickers:
    st.warning(f"Missing / empty tickers ({len(missing_tickers)}): {', '.join(missing_tickers[:30])}{' ...' if len(missing_tickers) > 30 else ''}")

# =============================================================================
# 9) TABS (MAIN + SEPARATE SECOND ANALYSIS)
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Markets", "➗ Ratios", "🧠 Intel", "🧠 Apex SMC v8"])

data_summary_for_ai = []

# =============================================================================
# 10) TAB 1 — TOUCH PANES (LINKED TO SEARCH)
# =============================================================================
with tab1:
    # search bar that panes write into
    st.text_input("Asset Search (Markets)", key="markets_search", placeholder="Filter labels… tap panes to auto-fill")

    selected_category = st.selectbox("Asset Class", list(TICKERS.keys()))
    items = TICKERS[selected_category]

    # Apply filter from either markets_search or global_asset_search (union)
    q = (st.session_state["markets_search"] or "").strip().lower()
    qg = (st.session_state["global_asset_search"] or "").strip().lower()
    query = q if q else qg

    if query:
        items = {k: v for k, v in items.items() if query in k.lower() or query in v[0].lower()}

    cols = st.columns(2)

    def tap_asset(label, ticker):
        # link to both search bars + set selected asset
        st.session_state["markets_search"] = label
        st.session_state["global_asset_search"] = label
        st.session_state["selected_asset_label"] = label
        st.session_state["selected_asset_ticker"] = ticker

    for i, (label, (ticker, desc)) in enumerate(items.items()):
        col = cols[i % 2]
        with col:
            if ticker in market_close.columns:
                s = market_close[ticker].dropna()
                val, pct = calculate_change(s)
                if val is None:
                    st.warning(f"{label}: No Data")
                    continue

                # Pane as a big touch target button (tap => sets search + selected asset)
                btn_label = f"🔎 {label}"
                if st.button(btn_label, key=f"pane_{_safe_key(label)}_{_safe_key(ticker)}"):
                    tap_asset(label, ticker)

                # Pane body (HTML look)
                pct_color = "#00FF00" if pct >= 0 else "#FF4B4B"
                st.markdown(
                    f"""
                    <div class="pane-wrap">
                      <div class="pane-title">{label}</div>
                      <div class="pane-sub">{desc} • <span style="color:#888">{ticker}</span></div>
                      <div class="pane-row">
                        <div class="pane-val">{val:,.2f}</div>
                        <div class="pane-pct" style="color:{pct_color}">{pct:+.2f}%</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                data_summary_for_ai.append(f"{label}: {pct:+.2f}%")
                st.plotly_chart(
                    plot_sparkline(s, "#00FF00" if pct >= 0 else "#FF4B4B"),
                    use_container_width=True,
                    config={"displayModeBar": False, "scrollZoom": True},
                    key=f"spark_{_safe_key(label)}_{i}"
                )

                # Truth flags surfaced
                flag_msg = TRUTH_FLAGS.get((label, ticker))
                if flag_msg:
                    st.markdown(pill_html("Truth Flag", "warn"), unsafe_allow_html=True)
                    st.caption(f"⚠️ {flag_msg}")

            else:
                st.warning(f"{label}: No Data")

    st.markdown("---")
    st.caption(f"Selected Asset: {st.session_state['selected_asset_label'] or '(none)'}  •  {st.session_state['selected_asset_ticker']}")

# =============================================================================
# 11) TAB 2 — RATIOS (unchanged core behavior)
# =============================================================================
with tab2:
    selected_ratio_cat = st.selectbox("Ratio Strategy", list(RATIO_GROUPS.keys()))
    items = RATIO_GROUPS[selected_ratio_cat]
    cols = st.columns(2)

    for i, (label, (num_t, den_t, desc)) in enumerate(items.items()):
        col = cols[i % 2]
        with col:
            rs = compute_ratio_series(num_t, den_t, market_close, syn_total, syn_total2, syn_total3)
            if rs is None or rs.dropna().empty:
                st.info(f"{label}: Insufficient Data")
                continue
            val, pct = calculate_change(rs)
            if val is None:
                st.info(f"{label}: Insufficient Data")
                continue

            data_summary_for_ai.append(f"{label}: {val:.4f} ({pct:+.2f}%)")
            st.metric(label, f"{val:.4f}", f"{pct:+.2f}%")
            st.caption(desc)
            st.plotly_chart(
                plot_sparkline(rs, "#3498db"),
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": True},
                key=f"ratio_{_safe_key(label)}_{i}"
            )

# =============================================================================
# 12) TAB 3 — INTEL (Regime Timeline stays)
# =============================================================================
with tab3:
    st.header("Institutional Intel")
    st.subheader("📉 Regime Timeline")

    z_window = st.selectbox("Timeline Z-Window", [20, 60, 120], index=1)
    timeline_df = build_regime_timeline(market_close, syn_total, syn_total2, syn_total3, z_window=z_window)

    if timeline_df is None or timeline_df.empty:
        st.info("Regime Timeline unavailable (insufficient overlap for selected timeframe/interval).")
    else:
        range_opt = st.selectbox("Timeline Range", ["1M", "3M", "6M", "1Y", "ALL"], index=1)
        df_show = timeline_df.copy()
        if range_opt != "ALL":
            days = {"1M": 31, "3M": 93, "6M": 186, "1Y": 366}[range_opt]
            cutoff = df_show.index.max() - pd.Timedelta(days=days)
            df_show = df_show[df_show.index >= cutoff]

        st.plotly_chart(
            plot_regime_timeline(df_show, title="Composite Regime Signals (Z-scored)"),
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": True}
        )

    st.markdown("---")
    st.subheader("🧠 AI Report")
    if st.button("Generate AI Report", type="primary"):
        with st.spinner("Analyzing Market Structure..."):
            if data_summary_for_ai:
                # Keep your original OpenAI integration pattern
                try:
                    api_key = st.secrets.get("OPENAI_API_KEY")
                    if not api_key:
                        st.error("⚠️ Missing OpenAI API Key in st.secrets.")
                    else:
                        client = openai.OpenAI(api_key=api_key)
                        prompt = f"""
Act as a Global Macro Strategist. Analyze these key market ratios:
{chr(10).join(data_summary_for_ai)}

Identify:
1. Is the regime Risk-On or Risk-Off?
2. Are we seeing Inflation or Deflation?
3. Crypto Specific Outlook

Be concise, bullet points.
"""
                        resp = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.markdown(resp.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("No data available for analysis.")

# =============================================================================
# 13) TAB 4 — SECOND ANALYSIS (SEPARATE) — APEX SMC v8
# =============================================================================
with tab4:
    st.header("🧠 Apex Trend & Liquidity Master (SMC) v8.0 — Second Analysis")
    st.caption("This tab is fully separate from the main dashboard. It uses the selected asset from the touch panes or the search bar.")

    # Asset selector: uses selected asset, but user can override by typing a ticker.
    cA, cB = st.columns([2, 1])
    with cA:
        override = st.text_input(
            "Selected Ticker (override if needed)",
            value=st.session_state["selected_asset_ticker"],
            help="Tap a pane in Markets to set this automatically, or type a yfinance ticker.",
            key="apex_override_ticker"
        )
    with cB:
        ohlc_period = st.selectbox("OHLC Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)

    # Config controls (mirrors Pine inputs)
    with st.expander("⚙️ Apex SMC Settings", expanded=True):
        s1, s2 = st.columns(2)
        with s1:
            ma_type = st.selectbox("Trend Algorithm", ["EMA", "SMA", "HMA", "RMA"], index=2)
            len_main = st.slider("Trend Length", 10, 200, 55, 1)
            mult = st.slider("Volatility Multiplier", 0.5, 5.0, 1.5, 0.1)
        with s2:
            use_adx = st.checkbox("Use ADX Filter", value=True)
            adx_threshold = st.slider("ADX Minimum", 10, 50, 20, 1)
            use_volume = st.checkbox("Use Volume Filter", value=True)
            vol_mult = st.slider("Volume Multiplier", 0.5, 3.0, 1.0, 0.1)
            use_momentum = st.checkbox("Use Momentum Filter (WaveTrend)", value=True)

        st.markdown("---")
        s3, s4 = st.columns(2)
        with s3:
            show_sd = st.checkbox("Show Swing S/D Zones", value=True)
            liq_len = st.slider("Pivot Lookback", 3, 50, 10, 1)
            sd_ext = st.slider("Extension (bars)", 5, 100, 20, 1)
            sd_max_zones = st.slider("Max S/D Zones", 5, 30, 10, 1)
        with s4:
            show_bos = st.checkbox("Show BOS/CHoCH", value=True)
            show_ob = st.checkbox("Show Order Blocks", value=True)
            show_fvg = st.checkbox("Show FVG", value=True)
            fvg_mit = st.checkbox("Auto-Delete Mitigated Zones", value=True)
            ob_lookback = st.slider("OB Lookback", 5, 50, 20, 1)
            fvg_min_size = st.slider("FVG Min Size (ATR)", 0.1, 2.0, 0.5, 0.1)
            ob_max_zones = st.slider("Max Order Blocks", 3, 15, 5, 1)
            fvg_max_zones = st.slider("Max FVGs", 5, 30, 10, 1)

    st.session_state["selected_asset_ticker"] = override

    cfg = dict(
        ma_type=ma_type, len_main=len_main, mult=mult,
        use_adx=use_adx, adx_threshold=adx_threshold,
        use_volume=use_volume, vol_mult=vol_mult,
        use_momentum=use_momentum,
        show_sd=show_sd, liq_len=liq_len, sd_ext=sd_ext, sd_max_zones=sd_max_zones,
        show_bos=show_bos, show_ob=show_ob, show_fvg=show_fvg,
        fvg_mit=fvg_mit, ob_lookback=ob_lookback, fvg_min_size=fvg_min_size,
        ob_max_zones=ob_max_zones, fvg_max_zones=fvg_max_zones,
    )

    with st.spinner("Fetching OHLCV + computing Apex SMC v8..."):
        ohlc = get_ohlcv(override, period=ohlc_period, interval=interval)
        if ohlc is None or ohlc.empty:
            st.error("No OHLCV data returned for this ticker/period/interval.")
        else:
            # keep chart performant
            max_bars = 900 if interval == "1d" else 700
            ohlc = ohlc.tail(max_bars)

            ind = apex_smc_v8(ohlc, cfg)
            if ind is None:
                st.error("Indicator computation failed (insufficient data).")
            else:
                title = f"Apex SMC v8 — {override} • {ohlc_period} • {interval}"
                fig = plot_apex_smc(ohlc, ind, title=title)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": True,
                        "scrollZoom": True,
                        "doubleClick": "reset",
                        "showTips": True
                    }
                )

                # Secondary analytics (separate, indicator-driven)
                st.markdown("---")
                st.subheader("📊 Apex Diagnostics (Indicator-driven)")
                d1, d2, d3 = st.columns(3)
                with d1:
                    last_tr = int(ind["trend"].iloc[-1])
                    st.metric("Trend", "Bullish" if last_tr == 1 else ("Bearish" if last_tr == -1 else "Neutral"))
                with d2:
                    last_adx = float(ind["adx"].dropna().iloc[-1]) if not ind["adx"].dropna().empty else float("nan")
                    st.metric("ADX (14)", f"{last_adx:.2f}" if not math.isnan(last_adx) else "n/a")
                with d3:
                    last_tci = float(ind["tci"].dropna().iloc[-1]) if not ind["tci"].dropna().empty else float("nan")
                    st.metric("WaveTrend TCI", f"{last_tci:.2f}" if not math.isnan(last_tci) else "n/a")

                # Show Pine script verbatim (do not omit)
                st.markdown("---")
                with st.expander("📌 Pine Script Reference (Verbatim — Apex SMC v8.0)", expanded=False):
                    st.code(PINE_SCRIPT_APEX_SMC_V8, language="pine")

    st.markdown("---")
    if st.button("Refresh Data (All)"):
        st.cache_data.clear()
        st.rerun()
