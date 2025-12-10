"""
TITAN INTRADAY PRO - Production-Ready Trading Dashboard

Version 20.0: Enhanced Architecture & Performance
"""

import time
import math
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime, timezone

# NEW IMPORT FOR AI (Wrapped to prevent crash if missing)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

@dataclass
class APIConfig:
    """API configuration constants"""
    BINANCE_BASE: str = "https://api.binance.us/api/v3"
    BYBIT_BASE: str = "https://api.bybit.com/v5/market/kline"
    COINBASE_BASE: str = "https://api.exchange.coinbase.com/products"
    HEADERS: Dict[str, str] = None
    
    def __post_init__(self):
        if self.HEADERS is None:
            self.HEADERS = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json"
            }


@dataclass
class DefaultParams:
    """Default parameter values"""
    AMPLITUDE: int = 10
    CHANNEL_DEV: float = 3.0
    HMA_LEN: int = 50
    GANN_LEN: int = 3
    TP1_R: float = 1.5
    TP2_R: float = 3.0
    TP3_R: float = 5.0
    MF_LEN: int = 14
    VOL_LEN: int = 20
    LIMIT: int = 300
    TIMEFRAME: str = "1h"


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="TITAN TERMINAL",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM CSS & STYLING
# =============================================================================

CUSTOM_CSS = """
<style>
    .main { background-color: #0b0c10; }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background: rgba(31, 40, 51, 0.7);
        border: 1px solid rgba(102, 252, 241, 0.1);
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    h1, h2, h3 { font-family: 'Roboto Mono', monospace; color: #c5c6c7; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #45a29e;
        color: #66fcf1;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: #45a29e;
        color: #0b0c10;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background-color: rgba(31, 40, 51, 0.5);
        border: 1px solid #45a29e;
        border-radius: 10px;
    }

    /* Report Table Styling */
    div[data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
        background-color: #1f2833;
        color: #c5c6c7;
    }
    div[data-testid="stMarkdownContainer"] th {
        background-color: #45a29e;
        color: #0b0c10;
        text-align: center;
        padding: 8px;
    }
    div[data-testid="stMarkdownContainer"] td {
        border-bottom: 1px solid #0b0c10;
        padding: 8px;
        text-align: center;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_symbol(symbol: str) -> str:
    """Normalize symbol input to standard format"""
    symbol = symbol.strip().upper().replace("/", "").replace("-", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    return symbol


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def calculate_hma(series: pd.Series, length: int) -> pd.Series:
    """Calculate Hull Moving Average"""
    if len(series) < length:
        return pd.Series(index=series.index, dtype=float)
    
    half_len = max(1, int(length / 2))
    sqrt_len = max(1, int(math.sqrt(length)))
    
    wma_f = series.rolling(length, min_periods=1).mean()
    wma_h = series.rolling(half_len, min_periods=1).mean()
    diff = 2 * wma_h - wma_f
    return diff.rolling(sqrt_len, min_periods=1).mean()


def calculate_fibonacci(df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels"""
    if len(df) < lookback:
        lookback = len(df)
    
    recent = df.iloc[-lookback:]
    h, l = recent['high'].max(), recent['low'].min()
    d = h - l
    
    if d == 0:
        mid = (h + l) / 2
        return {
            'fib_382': mid,
            'fib_500': mid,
            'fib_618': mid,
            'high': h,
            'low': l
        }
    
    return {
        'fib_382': h - (d * 0.382),
        'fib_500': h - (d * 0.500),
        'fib_618': h - (d * 0.618),
        'high': h,
        'low': l
    }


def calculate_fear_greed_index(df: pd.DataFrame) -> int:
    """Calculate custom Fear & Greed Index"""
    try:
        if len(df) < 90:
            return 50
        
        df = df.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        vol_30 = df['log_ret'].rolling(30, min_periods=1).std()
        vol_90 = df['log_ret'].rolling(90, min_periods=1).std()
        
        vol_score = 50 - ((vol_30.iloc[-1] - vol_90.iloc[-1]) / max(vol_90.iloc[-1], 1e-10)) * 100
        vol_score = max(0, min(100, vol_score))
        
        rsi = df.get('rsi', pd.Series([50]))[-1] if 'rsi' in df.columns else 50
        rsi = max(0, min(100, rsi))
        
        sma_50 = df['close'].rolling(50, min_periods=1).mean().iloc[-1]
        dist = safe_divide(df['close'].iloc[-1] - sma_50, sma_50, 0.0)
        trend_score = 50 + (dist * 1000)
        trend_score = max(0, min(100, trend_score))
        
        fg = (vol_score * 0.3) + (rsi * 0.4) + (trend_score * 0.3)
        return int(max(0, min(100, fg)))
    except Exception:
        return 50


# =============================================================================
# DATA FETCHING
# =============================================================================

@st.cache_data(ttl=5)
def get_klines(symbol_bin: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch klines from Binance API with error handling"""
    api_config = APIConfig()
    
    try:
        params = {
            "symbol": symbol_bin,
            "interval": interval,
            "limit": limit
        }
        
        response = requests.get(
            f"{api_config.BINANCE_BASE}/klines",
            params=params,
            headers=api_config.HEADERS,
            timeout=4
        )
        
        if response.status_code == 200:
            data = response.json()
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(
                data,
                columns=['t', 'o', 'h', 'l', 'c', 'v', 'T', 'q', 'n', 'V', 'Q', 'B']
            )
            
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            numeric_cols = ['o', 'h', 'l', 'c', 'v']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df[['timestamp', 'o', 'h', 'l', 'c', 'v']].rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
            })
        else:
            st.error(f"API Error: {response.status_code}")
            return pd.DataFrame()
            
    except requests.exceptions.Timeout:
        st.error("Request timeout - please try again")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return pd.DataFrame()


# =============================================================================
# TRADING ENGINES
# =============================================================================

def run_engines(
    df: pd.DataFrame,
    amp: int,
    dev: float,
    hma_l: int,
    hma_on: bool,
    tp1: float,
    tp2: float,
    tp3: float,
    mf_l: int,
    vol_l: int,
    gann_l: int
) -> pd.DataFrame:
    """Run all trading indicator engines"""
    if df.empty or len(df) < max(amp, hma_l, gann_l, 20):
        return df
    
    df = df.copy().reset_index(drop=True)
    
    # ATR Calculation
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    
    # HMA
    df['hma'] = calculate_hma(df['close'], hma_l)
    
    # VWAP Engine
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol_tp'] = df['tp'] * df['volume']
    df['vwap'] = df['vol_tp'].cumsum() / df['volume'].cumsum().replace(0, np.nan)
    df['vwap'] = df['vwap'].fillna(method='ffill')
    
    # Squeeze Engine (TTM Style)
    bb_basis = df['close'].rolling(20, min_periods=1).mean()
    bb_dev = df['close'].rolling(20, min_periods=1).std() * 2.0
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    
    kc_basis = df['close'].rolling(20, min_periods=1).mean()
    kc_dev = df['atr'] * 1.5
    kc_upper = kc_basis + kc_dev
    kc_lower = kc_basis - kc_dev
    
    df['in_squeeze'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + safe_divide(gain, loss, 1)))
    
    # Relative Volume
    vol_ma = df['volume'].rolling(vol_l, min_periods=1).mean()
    df['rvol'] = safe_divide(df['volume'], vol_ma, 1.0)
    
    # Money Flow
    rsi_source = df['rsi'] - 50
    vol_sma = df['volume'].rolling(mf_l, min_periods=1).mean()
    mf_volume = safe_divide(df['volume'], vol_sma, 1.0)
    df['money_flow'] = (rsi_source * mf_volume).ewm(span=3, adjust=False).mean()
    
    # Hyper Wave
    pc = df['close'].diff()
    ds_pc = pc.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    ds_abs_pc = abs(pc).ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    df['hyper_wave'] = (100 * safe_divide(ds_pc, ds_abs_pc, 0)) / 2
    
    # Advanced Volume Metrics
    price_range = df['high'] - df['low']
    mfm = safe_divide(
        (df['close'] - df['low']) - (df['high'] - df['close']),
        price_range,
        0
    )
    mfm = mfm.fillna(0)
    mf_vol = mfm * df['volume']
    df['cmf'] = safe_divide(
        mf_vol.rolling(20, min_periods=1).sum(),
        df['volume'].rolling(20, min_periods=1).sum(),
        0
    )
    
    v_delta = df['volume'].diff()
    v_gain = (v_delta.where(v_delta > 0, 0)).rolling(14, min_periods=1).mean()
    v_loss = (-v_delta.where(v_delta < 0, 0)).rolling(14, min_periods=1).mean()
    df['vrsi'] = 100 - (100 / (1 + safe_divide(v_gain, v_loss, 1)))
    
    v_short = df['volume'].rolling(14, min_periods=1).mean()
    v_long = df['volume'].rolling(28, min_periods=1).mean()
    df['vol_osc'] = 100 * safe_divide(v_short - v_long, v_long, 0)
    
    # Hidden Liquidity
    body_size = (df['close'] - df['open']).abs()
    range_size = df['high'] - df['low']
    is_doji = body_size <= (range_size * 0.1)
    df['hidden_liq'] = is_doji & (df['rvol'] > 2.0)
    
    # TITAN Engine
    df['ll'] = df['low'].rolling(amp, min_periods=1).min()
    df['hh'] = df['high'].rolling(amp, min_periods=1).max()
    
    trend = np.zeros(len(df))
    stop = np.full(len(df), np.nan)
    curr_t = 0
    curr_s = np.nan
    
    for i in range(amp, len(df)):
        c = df.at[i, 'close']
        d = df.at[i, 'atr'] * dev
        
        if curr_t == 0:  # Bullish
            s = df.at[i, 'll'] + d
            curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
            if c < curr_s:
                curr_t = 1
                curr_s = df.at[i, 'hh'] - d
        else:  # Bearish
            s = df.at[i, 'hh'] - d
            curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
            if c > curr_s:
                curr_t = 0
                curr_s = df.at[i, 'll'] + d
        
        trend[i] = curr_t
        stop[i] = curr_s
    
    df['is_bull'] = trend == 0
    df['entry_stop'] = stop
    
    # Signal Generation
    cond_buy = (
        (df['is_bull']) &
        (~df['is_bull'].shift(1).fillna(False)) &
        (df['rvol'] > 1.15) &
        (df['rsi'] < 70)
    )
    cond_sell = (
        (~df['is_bull']) &
        (df['is_bull'].shift(1).fillna(True)) &
        (df['rvol'] > 1.15) &
        (df['rsi'] > 30)
    )
    
    if hma_on:
        cond_buy &= (df['close'] > df['hma'])
        cond_sell &= (df['close'] < df['hma'])
    
    df['buy'] = cond_buy
    df['sell'] = cond_sell
    
    # Ladder (TP Calculation)
    df['sig_id'] = (df['buy'] | df['sell']).cumsum()
    df['entry'] = np.where(df['buy'] | df['sell'], df['close'], np.nan)
    df['entry'] = df.groupby('sig_id')['entry'].ffill()
    df['stop_val'] = df.groupby('sig_id')['entry_stop'].ffill()
    
    risk = abs(df['entry'] - df['stop_val'])
    df['tp1'] = np.where(
        df['is_bull'],
        df['entry'] + (risk * tp1),
        df['entry'] - (risk * tp1)
    )
    df['tp2'] = np.where(
        df['is_bull'],
        df['entry'] + (risk * tp2),
        df['entry'] - (risk * tp2)
    )
    df['tp3'] = np.where(
        df['is_bull'],
        df['entry'] + (risk * tp3),
        df['entry'] - (risk * tp3)
    )
    
    # APEX Engine
    apex_base = calculate_hma(df['close'], 55)
    apex_atr = df['atr'] * 1.5
    df['apex_upper'] = apex_base + apex_atr
    df['apex_lower'] = apex_base - apex_atr
    
    apex_t = np.zeros(len(df))
    for i in range(1, len(df)):
        if df.at[i, 'close'] > df.at[i, 'apex_upper']:
            apex_t[i] = 1
        elif df.at[i, 'close'] < df.at[i, 'apex_lower']:
            apex_t[i] = -1
        else:
            apex_t[i] = apex_t[i-1]
    df['apex_trend'] = apex_t
    
    # Pivot Points
    df['pivot_high'] = df['high'].rolling(21, center=True, min_periods=1).max()
    df['pivot_low'] = df['low'].rolling(21, center=True, min_periods=1).min()
    df['is_res'] = (df['high'] == df['pivot_high'])
    df['is_sup'] = (df['low'] == df['pivot_low'])
    
    # GANN HILO Engine
    sma_high = df['high'].rolling(gann_l, min_periods=1).mean()
    sma_low = df['low'].rolling(gann_l, min_periods=1).mean()
    
    g_trend = np.full(len(df), np.nan)
    g_act = np.full(len(df), np.nan)
    
    if len(sma_low) > gann_l:
        curr_g_t = 1
        curr_g_a = sma_low.iloc[gann_l]
        
        for i in range(gann_l, len(df)):
            c = df.at[i, 'close']
            h_ma = sma_high.iloc[i]
            l_ma = sma_low.iloc[i]
            prev_a = g_act[i-1] if (i > 0 and not np.isnan(g_act[i-1])) else curr_g_a
            
            if np.isnan(prev_a):
                prev_a = l_ma
            
            if curr_g_t == 1:
                if c < prev_a:
                    curr_g_t = -1
                    curr_g_a = h_ma
                else:
                    curr_g_a = l_ma
            else:
                if c > prev_a:
                    curr_g_t = 1
                    curr_g_a = l_ma
                else:
                    curr_g_a = h_ma
            
            g_trend[i] = curr_g_t
            g_act[i] = curr_g_a
    
    df['gann_trend'] = g_trend
    df['gann_act'] = g_act
    
    return df


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_full_report(
    row: pd.Series,
    symbol: str,
    tf: str,
    fibs: Dict[str, float],
    fg_index: int,
    smart_stop: float
) -> Tuple[str, str]:
    """Generate comprehensive trading report"""
    is_bull = row['is_bull']
    direction = "LONG 🐂" if is_bull else "SHORT 🐻"
    
    # Alignment Logic
    titan_sig = 1 if is_bull else -1
    apex_sig = row['apex_trend']
    gann_sig = row['gann_trend']
    vwap_sig = 1 if row['close'] > row['vwap'] else -1
    
    score_val = 0
    if titan_sig == apex_sig:
        score_val += 1
    if titan_sig == gann_sig:
        score_val += 1
    if titan_sig == vwap_sig:
        score_val += 1
    
    confidence = "LOW"
    if score_val == 3:
        confidence = "INSTITUTIONAL 🔥"
    elif score_val == 2:
        confidence = "HIGH"
    elif score_val == 1:
        confidence = "MODERATE"
    
    # Market Regime
    regime = "TRENDING"
    if row['in_squeeze']:
        regime = "COMPRESSION (COILING)"
    elif abs(row['close'] - row['vwap']) / max(row['vwap'], 1e-10) > 0.05:
        regime = "OVER-EXTENDED"
    
    # Flow Analysis
    flow_state = "NEUTRAL"
    if row['money_flow'] > 0 and is_bull:
        flow_state = "Valid Accumulation"
    elif row['money_flow'] < 0 and not is_bull:
        flow_state = "Valid Distribution"
    elif row['money_flow'] < 0 and is_bull:
        flow_state = "⚠️ DIVERGENCE (Price Up, Vol Down)"
    elif row['money_flow'] > 0 and not is_bull:
        flow_state = "⚠️ ABSORPTION (Price Down, Vol Up)"
    
    # Volatility Context
    vol_desc = "Normal"
    if row['rvol'] > 2.0:
        vol_desc = "IGNITION BREAKOUT (High Vol)"
    elif row['rvol'] < 0.6:
        vol_desc = "Low Liquidity / Chop"
    
    squeeze_txt = "Energy Building 🔋" if row['in_squeeze'] else "Energy Released ⚡"
    
    # Commentary
    bias_color = "🟢" if is_bull else "🔴"
    vwap_dist = safe_divide((row['close'] - row['vwap']), row['vwap'], 0) * 100
    
    commentary = (
        f"Price is currently {abs(vwap_dist):.2f}% "
        f"{'above' if vwap_dist > 0 else 'below'} the institutional VWAP. "
    )
    
    if confidence == "INSTITUTIONAL 🔥":
        commentary += "All trend systems (Titan, Apex, Gann) are perfectly aligned. "
    else:
        commentary += "Mixed signals detected between short-term momentum and medium-term structure. "
    
    if row['in_squeeze']:
        commentary += "Volatility is compressing (TTM Squeeze); expect an explosive move soon. "
    
    # Format Report
    report_md = f"""
### 💠 TITAN AI DEEP DIVE: {symbol} [{tf}]

| **SIGNAL MATRIX** | **INSTITUTIONAL DATA** | **VOLATILITY & FLOW** |
| :--- | :--- | :--- |
| **DIR:** {direction} | **VWAP:** {row['vwap']:.4f} | **RVOL:** {row['rvol']:.2f} ({vol_desc}) |
| **CONF:** {confidence} | **REGIME:** {regime} | **FLOW:** {flow_state} |
| **TITAN:** {bias_color} | **SENTIMENT:** {fg_index}/100 | **SQUEEZE:** {squeeze_txt} |
| **APEX:** {'🟢' if apex_sig == 1 else '🔴' if apex_sig == -1 else '⚪'} | **GANN:** {'🟢' if gann_sig == 1 else '🔴'} | **MOMENTUM:** {row['hyper_wave']:.2f} |

---

#### 🧠 STRATEGIC SYNTHESIS

> **{commentary}**

* **Structure:** The Apex Cloud is currently **{'Bullish' if apex_sig == 1 else 'Bearish' if apex_sig == -1 else 'Flat'}**. The Gann Activator is **{'Supporting' if gann_sig == titan_sig else 'Conflicting with'}** the primary trend.

* **Money Flow:** The Money Flow Index is at **{row['money_flow']:.2f}**. {flow_state}.

* **Execution Zone:**
    * **Current Price:** `{row['close']:.4f}`
    * **Smart Stop:** `{smart_stop:.4f}` (Placed behind Market Structure & Fib 0.618)
    * **TP1 (1.5R):** `{row['tp1']:.4f}` | **TP2 (3.0R):** `{row['tp2']:.4f}` | **TP3 (5.0R):** `{row['tp3']:.4f}**
"""
    
    return report_md, commentary


# =============================================================================
# BACKTESTING
# =============================================================================

def run_backtest(df: pd.DataFrame, tp1_r: float) -> Tuple[int, float, float]:
    """Run simple backtest on signals"""
    trades = []
    signals = df[(df['buy']) | (df['sell'])]
    
    if signals.empty:
        return 0, 0, 0
    
    for idx, row in signals.iterrows():
        future = df.loc[idx+1:idx+20]
        if future.empty:
            continue
        
        entry = row['close']
        stop = row['entry_stop']
        tp1 = row['tp1']
        is_long = row['is_bull']
        
        outcome = "PENDING"
        pnl = 0
        
        if is_long:
            if future['high'].max() >= tp1:
                outcome = "WIN"
                pnl = abs(entry - stop) * tp1_r
            elif future['low'].min() <= stop:
                outcome = "LOSS"
                pnl = -abs(entry - stop)
        else:
            if future['low'].min() <= tp1:
                outcome = "WIN"
                pnl = abs(entry - stop) * tp1_r
            elif future['high'].max() >= stop:
                outcome = "LOSS"
                pnl = -abs(entry - stop)
        
        if outcome != "PENDING":
            trades.append({'outcome': outcome, 'pnl': pnl})
    
    if not trades:
        return 0, 0, 0
    
    df_res = pd.DataFrame(trades)
    total_trades = len(df_res)
    win_rate = (len(df_res[df_res['outcome'] == 'WIN']) / total_trades) * 100
    net_r = (
        (len(df_res[df_res['outcome'] == 'WIN']) * tp1_r) -
        len(df_res[df_res['outcome'] == 'LOSS'])
    )
    
    return total_trades, win_rate, net_r


# =============================================================================
# TELEGRAM INTEGRATION
# =============================================================================

def send_telegram_msg(
    token: str,
    chat: str,
    msg: str,
    cooldown: int = 0
) -> bool:
    """Send message to Telegram with cooldown"""
    if not token or not chat:
        return False
    
    token = token.strip()
    chat = chat.strip()
    
    last = st.session_state.get("last_tg", 0)
    if time.time() - last < cooldown:
        return False
    
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"},
            timeout=5
        )
        
        if response.status_code == 200:
            st.session_state["last_tg"] = time.time()
            return True
    except Exception:
        pass
    
    return False


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_ticker_widget():
    """Render TradingView ticker widget"""
    components.html(
        """
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
          {
          "symbols": [
            {"proName": "BINANCE:BTCUSDT", "title": "BTC/USDT"},
            {"proName": "BINANCE:ETHUSDT", "title": "ETH/USDT"},
            {"proName": "BINANCE:SOLUSDT", "title": "SOL/USDT"},
            {"proName": "BINANCE:XRPUSDT", "title": "XRP/USDT"},
            {"proName": "BINANCE:BNBUSDT", "title": "BNB/USDT"}
          ],
          "showSymbolLogo": true,
          "colorTheme": "dark",
          "isTransparent": true,
          "displayMode": "adaptive",
          "locale": "en"
        }
          </script>
        </div>
        """,
        height=50
    )


def render_live_clock():
    """Render live UTC clock"""
    components.html(
        """
        <div id="live_clock"></div>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@700&display=swap');
            body { margin: 0; background-color: transparent; }
            #live_clock {
                font-family: 'Roboto Mono', monospace;
                font-size: 26px;
                color: #39ff14;
                text-shadow: 0 0 10px rgba(57, 255, 20, 0.8);
                font-weight: 800;
                text-align: right;
                padding: 10px;
                letter-spacing: 1px;
            }
        </style>
        <script>
        function updateTime() {
            const now = new Date();
            const timeString = now.toLocaleTimeString('en-GB', { timeZone: 'UTC' });
            document.getElementById('live_clock').innerHTML = 'UTC: ' + timeString;
        }
        setInterval(updateTime, 1000);
        updateTime();
        </script>
        """,
        height=60
    )


# =============================================================================
# MAIN APP EXECUTION
# =============================================================================

def main():
    """Main application entry point"""
    defaults = DefaultParams()
    
    # Render ticker
    render_ticker_widget()
    
    # Header
    c_head1, c_head2 = st.columns([3, 1])
    with c_head1:
        st.title("💠 TITAN TERMINAL v20.0")
        st.caption("FULL-SPECTRUM AI ANALYSIS ENGINE")
    with c_head2:
        render_live_clock()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ SYSTEM CONTROL")
        
        if st.button("🔄 FORCE REFRESH DATA", use_container_width=True):
            st.cache_data.clear()
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()
        
        with st.expander("📚 Engines Guide", expanded=False):
            st.markdown("""
            **1. TITAN:** ATR Trailing Stops.
            **2. APEX:** HMA Cloud Structure.
            **3. GANN:** HiLo Step Activator.
            **4. MATRIX:** Money Flow + Hyper Wave.
            **5. VWAP:** Institutional Average Price.
            **6. SQUEEZE:** Volatility Compression.
            """)
        
        st.subheader("📡 FEED")
        
        symbol_input = st.text_input("Asset Search (e.g. BTC)", value="BTC")
        symbol = normalize_symbol(symbol_input)
        
        timeframe = st.selectbox(
            "Interval",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            index=3
        )
        limit = st.slider("Depth", 100, 1000, defaults.LIMIT, 50)
        
        st.markdown("---")
        st.subheader("🧠 STRATEGY")
        
        amplitude = st.number_input("Amplitude", 2, 200, defaults.AMPLITUDE)
        channel_dev = st.number_input("ATR Dev", 0.5, 10.0, defaults.CHANNEL_DEV, 0.1)
        hma_len = st.number_input("HMA Len", 2, 400, defaults.HMA_LEN)
        gann_len = st.number_input("Gann Len", 1, 50, defaults.GANN_LEN)
        
        with st.expander("🎯 Targets (R)"):
            tp1_r = st.number_input("TP1", value=defaults.TP1_R, step=0.1)
            tp2_r = st.number_input("TP2", value=defaults.TP2_R, step=0.1)
            tp3_r = st.number_input("TP3", value=defaults.TP3_R, step=0.1)
        
        st.markdown("---")
        st.subheader("🤖 AI STATUS")
        
        openai_key = ""
        if OpenAI is None:
            st.error("Missing 'openai' library")
        else:
            try:
                openai_key = st.secrets["OPENAI_API_KEY"]
                st.success("🟢 AI Engine: ONLINE")
            except Exception:
                openai_key = ""
                st.error("🔴 AI Engine: OFFLINE (Missing Secret)")
        
        st.markdown("---")
        st.subheader("📊 VOL METRICS")
        
        hero_metric = st.selectbox(
            "Hero Metric",
            ["CMF", "Volume RSI", "Volume Oscillator", "RVOL"]
        )
        mf_len = st.number_input("MF Len", 2, 200, defaults.MF_LEN)
        vol_len = st.number_input("Vol Len", 5, 200, defaults.VOL_LEN)
        
        st.markdown("---")
        st.subheader("🤖 TELEGRAM")
        
        try:
            sec_token = st.secrets["TELEGRAM_TOKEN"]
        except Exception:
            sec_token = ""
        
        try:
            sec_chat = st.secrets["TELEGRAM_CHAT_ID"]
        except Exception:
            sec_chat = ""
        
        tg_token = st.text_input("Token", value=sec_token, type="password")
        tg_chat = st.text_input("Chat ID", value=sec_chat)
        
        if st.button("Test Link"):
            t_clean = tg_token.strip()
            c_clean = tg_chat.strip()
            
            if not t_clean or not c_clean:
                st.error("Missing Creds")
            else:
                try:
                    response = requests.post(
                        f"https://api.telegram.org/bot{t_clean}/sendMessage",
                        json={"chat_id": c_clean, "text": "💠 TITAN: ONLINE"},
                        timeout=5
                    )
                    if response.status_code == 200:
                        st.success("Linked! ✅")
                    else:
                        st.error(f"Error {response.status_code}")
                except Exception as e:
                    st.error(f"Failed: {e}")
    
    # Main Content
    with st.spinner("Initializing Terminal..."):
        df = get_klines(symbol, timeframe, limit)
    
    if df.empty:
        st.error("❌ Failed to fetch market data. Please check your symbol and try again.")
        return
    
    df = df.dropna(subset=['close'])
    
    if len(df) < 50:
        st.warning("⚠️ Insufficient data points. Please increase the depth or try a different timeframe.")
        return
    
    with st.spinner("Processing Algorithms..."):
        df = run_engines(
            df,
            int(amplitude),
            channel_dev,
            int(hma_len),
            True,
            tp1_r,
            tp2_r,
            tp3_r,
            int(mf_len),
            int(vol_len),
            int(gann_len)
        )
    
    if df.empty:
        st.error("❌ Engine processing failed.")
        return
    
    last = df.iloc[-1]
    fibs = calculate_fibonacci(df)
    fg_index = calculate_fear_greed_index(df)
    
    # Smart Stop Logic
    if last['is_bull']:
        smart_stop = min(last['entry_stop'], fibs['fib_618'] * 0.9995)
    else:
        smart_stop = max(last['entry_stop'], fibs['fib_618'] * 1.0005)
    
    # Generate Report
    ai_report, ai_context_str = generate_full_report(
        last, symbol, timeframe, fibs, fg_index, smart_stop
    )
    
    # Risk Metrics
    entry_price = last['close']
    risk_pct = abs(entry_price - smart_stop) / max(entry_price, 1e-10) * 100
    
    vol_tier = "🟢 LOW RISK"
    if risk_pct > 2.0:
        vol_tier = "🟡 MED RISK"
    if risk_pct > 5.0:
        vol_tier = "🔴 HIGH VOLATILITY"
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        tv_symbol = f"BINANCE:{symbol}"
        components.html(
            f"""
            <div class="tradingview-widget-container">
              <div class="tradingview-widget-container__widget"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js" async>
              {{
              "symbol": "{tv_symbol}",
              "width": "100%",
              "colorTheme": "dark",
              "isTransparent": true,
              "locale": "en"
            }}
              </script>
            </div>
            """,
            height=120
        )
    
    m2.metric(
        "GANN TREND",
        "BULL" if last['gann_trend'] == 1 else "BEAR"
    )
    m3.metric("SMART STOP", f"{smart_stop:.2f}")
    m4.metric("TP3 (5R)", f"{last['tp3']:.2f}")
    
    # Action Center
    c_act1, c_act2, c_act3 = st.columns([1, 2, 1])
    
    signal_txt = (
        f"🔥 *TITAN SIGNAL: {symbol}*\n"
        f"➖➖➖➖➖➖➖➖➖➖\n"
        f"📊 *STRATEGY INFO:*\n"
        f"• Type: *{'LONG 🟢' if last['is_bull'] else 'SHORT 🔴'}* | TF: {timeframe}\n"
        f"• Volatility: {vol_tier} (Risk: {risk_pct:.2f}%)\n"
        f"• Money Flow: {'🌊 INFLOW' if last['money_flow'] > 0 else '🩸 OUTFLOW'}\n"
        f"• Sentiment: {fg_index}/100\n"
        f"• VWAP Relation: {'Above' if last['close'] > last['vwap'] else 'Below'}\n"
        f"• Squeeze Status: {'⚠️ ACTIVE' if last['in_squeeze'] else '🚀 FIRING'}\n"
        f"• Confluence: {'Strong' if last['gann_trend'] == (1 if last['is_bull'] else -1) else 'Weak'}\n\n"
        f"📍 *ENTRY:* `{entry_price:.4f}`\n\n"
        f"🛡️ *SMART STOP:* `{smart_stop:.4f}`\n"
        f"_(Below 0.618 Golden Pocket)_\n\n"
        f"🎯 *TARGETS:*\n"
        f"1️⃣ TP1 (1.5R): `{last['tp1']:.4f}`\n"
        f"2️⃣ TP2 (3.0R): `{last['tp2']:.4f}`\n"
        f"3️⃣ TP3 (5.0R): `{last['tp3']:.4f}`\n"
        f"➖➖➖➖➖➖➖➖➖➖\n"
        f"⚠️ _NFA: Manage Risk_"
    )
    
    with c_act1:
        if st.button("🔥 BROADCAST SIGNAL", use_container_width=True):
            if tg_token and tg_chat:
                if send_telegram_msg(tg_token, tg_chat, signal_txt, 0):
                    st.success("SIGNAL SENT!")
                else:
                    st.error("FAILED")
            else:
                st.error("NO CREDS (Check secrets.toml or Inputs)")
    
    with c_act2:
        with st.expander("🤖 TITAN AI REPORT (LIVE)", expanded=True):
            st.markdown(ai_report, unsafe_allow_html=True)
            
            if st.button("✈️ POST THIS REPORT TO TG", use_container_width=True):
                if tg_token and tg_chat:
                    if send_telegram_msg(tg_token, tg_chat, ai_report, 0):
                        st.success("REPORT SENT!")
                    else:
                        st.error("FAILED")
                else:
                    st.error("NO CREDS")
    
    with c_act3:
        b_total, b_win, b_net = run_backtest(df, tp1_r)
        st.metric(
            "Backtest (Approx)",
            f"WR: {b_win:.1f}%",
            f"Net: {b_net:.1f}R ({b_total} Txs)"
        )
    
    # Main Chart
    st.markdown("### 🏹 EXECUTION")
    fig = go.Figure()
    
    fig.add_candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['hma'],
            mode='lines',
            name='HMA',
            line=dict(color='#66fcf1', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['vwap'],
            mode='lines',
            name='VWAP',
            line=dict(color='#9933ff', width=2, dash='solid')
        )
    )
    
    buys = df[df['buy']]
    sells = df[df['sell']]
    
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys['timestamp'],
                y=buys['low'] * 0.999,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='#00ff00'),
                name='BUY'
            )
        )
    
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells['timestamp'],
                y=sells['high'] * 1.001,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='#ff0000'),
                name='SELL'
            )
        )
    
    fig.update_layout(
        height=600,
        template='plotly_dark',
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_rangeslider_visible=False,
        yaxis=dict(autorange=True)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabbed Analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 GANN HILO", "🌊 APEX", "💸 MATRIX",
        "📉 VOL", "🧠 SENTIMENT", "🤖 AI ANALYST"
    ])
    
    with tab1:
        fig6 = go.Figure()
        fig6.add_candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        )
        
        mask = ~np.isnan(df['gann_act'])
        df_g = df[mask]
        
        if not df_g.empty:
            fig6.add_trace(
                go.Scatter(
                    x=df_g['timestamp'],
                    y=df_g['gann_act'],
                    mode='markers',
                    marker=dict(
                        color=np.where(df_g['gann_trend'] == 1, '#00ff00', '#ff0000'),
                        size=4
                    ),
                    name='Gann Activator'
                )
            )
        
        fig6.update_layout(
            height=500,
            template='plotly_dark',
            margin=dict(l=0, r=0, t=0, b=0),
            title="Gann High Low Activator"
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['apex_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['apex_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(102, 252, 241, 0.1)',
                name='Cloud'
            )
        )
        fig2.add_candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        )
        
        for i, r in df[df['is_sup']].iloc[-5:].iterrows():
            fig2.add_shape(
                type="rect",
                x0=r['timestamp'],
                y0=r['low'],
                x1=df['timestamp'].iloc[-1],
                y1=r['low'] - (r['atr'] * 0.5),
                fillcolor="rgba(0,255,0,0.2)",
                line_width=0
            )
        
        for i, r in df[df['is_res']].iloc[-5:].iterrows():
            fig2.add_shape(
                type="rect",
                x0=r['timestamp'],
                y0=r['high'],
                x1=df['timestamp'].iloc[-1],
                y1=r['high'] + (r['atr'] * 0.5),
                fillcolor="rgba(255,0,0,0.2)",
                line_width=0
            )
        
        fig2.update_layout(
            height=500,
            template='plotly_dark',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        fig4 = go.Figure()
        colors = ['#00e676' if x > 0 else '#ff1744' for x in df['money_flow']]
        fig4.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['money_flow'],
                marker_color=colors,
                name='Flow'
            )
        )
        fig4.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['hyper_wave'],
                mode='lines',
                name='Hyper Wave',
                line=dict(color='yellow', width=2)
            )
        )
        fig4.update_layout(
            height=500,
            template='plotly_dark',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    with tab4:
        fig5 = go.Figure()
        
        vol_map = {
            "CMF": df['cmf'],
            "Volume RSI": df['vrsi'],
            "Volume Oscillator": df['vol_osc'],
            "RVOL": df['rvol']
        }
        vol_d = vol_map.get(hero_metric, df['rvol'])
        
        fig5.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=vol_d,
                mode='lines',
                name=hero_metric,
                fill='tozeroy',
                line=dict(color='#b388ff')
            )
        )
        
        traps = df[df['hidden_liq']]
        if not traps.empty:
            fig5.add_trace(
                go.Scatter(
                    x=traps['timestamp'],
                    y=vol_d[df['hidden_liq']],
                    mode='markers',
                    marker=dict(symbol='diamond', size=10, color='yellow'),
                    name='Trap'
                )
            )
        
        fig5.update_layout(
            height=500,
            template='plotly_dark',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab5:
        fig3 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=fg_index,
                title={'text': "MARKET SENTIMENT"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, 25], 'color': '#ff1744'},
                        {'range': [75, 100], 'color': '#00b0ff'}
                    ]
                }
            )
        )
        fig3.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab6:
        st.subheader("🤖 TITAN AI ANALYST (GPT-4o)")
        
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{
                "role": "assistant",
                "content": (
                    f"I am TITAN AI. I am analyzing {symbol} on the {timeframe} timeframe.\n\n"
                    f"My live sensors indicate:\n"
                    f"- **Trend:** {'BULLISH' if last['is_bull'] else 'BEARISH'}\n"
                    f"- **RSI:** {last['rsi']:.2f}\n"
                    f"- **VWAP:** {last['vwap']:.2f}\n\n"
                    f"Ask me about price action, macro outlooks, or potential setups."
                )
            }]
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask Titan about the market..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            if not openai_key:
                err_msg = "⚠️ ACCESS DENIED. I cannot find your 'OPENAI_API_KEY' in the secrets.toml file."
                st.chat_message("assistant").markdown(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            elif OpenAI is None:
                err_msg = "⚠️ ERROR: OpenAI library not installed. Please run `pip install openai`."
                st.chat_message("assistant").markdown(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            else:
                try:
                    client = OpenAI(api_key=openai_key)
                    
                    system_prompt = f"""
                    You are TITAN, an elite quantitative trading assistant.
                    You are currently monitoring {symbol} on a {timeframe} timeframe.
                    
                    LIVE TECHNICAL DATA:
                    - Current Price: {last['close']}
                    - Trend Direction: {'UP' if last['is_bull'] else 'DOWN'}
                    - RSI (14): {last['rsi']:.2f}
                    - Volatility (RVOL): {last['rvol']:.2f}
                    - Squeeze Status: {'Active Squeeze (Coiling)' if last['in_squeeze'] else 'Released'}
                    - VWAP: {last['vwap']}
                    - Money Flow: {last['money_flow']:.2f}
                    - Sentiment Score: {fg_index}/100
                    - Technical Summary: {ai_context_str}
                    
                    INSTRUCTIONS:
                    - Answer concisely and professionally.
                    - If asked about trading advice, provide a technical outlook based on the data above but disclaim financial advice.
                    - If asked about Macro, use your general knowledge but tie it back to the current chart technicals if possible.
                    - Be confident but risk-aware.
                    """
                    
                    with st.chat_message("assistant"):
                        stream = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": system_prompt},
                            ] + [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            stream=True,
                        )
                        response = st.write_stream(stream)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    st.error(f"AI Engine Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    tv_interval = 'D' if timeframe == '1d' else '60'
    components.html(
        f"""
        <div id="tv" style="height:500px;border-radius:10px;overflow:hidden;"></div>
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>
        new TradingView.widget({{
            "autosize": true,
            "symbol": "BINANCE:{symbol}",
            "interval": "{tv_interval}",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "container_id": "tv",
            "hide_side_toolbar": false
        }});
        </script>
        """,
        height=500
    )


if __name__ == "__main__":
    main()
