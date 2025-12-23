import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import math
import time
import requests
import io
import warnings
import random
from typing import Optional, Tuple
import streamlit.components.v1 as components

# Optional dependencies
try:
    from scipy.special import zeta as scipy_zeta
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# 1. PAGE CONFIGURATION & QUANTUM THEME (MOBILE OPTIMIZED)
# =============================================================================
st.set_page_config(
    page_title="APEX OMNI-TERMINAL PRO",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded" 
)

QUANTUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@500;700&display=swap');
    
    :root {
        --bg: #050505;
        --card: #0F0F0F;
        --border: #222;
        --accent: #00F5D4;
        --purple: #9D4EDD;
        --danger: #FF006E;
        --success: #00E676;
        --warning: #FFD600;
        --text: #E0E0E0;
    }

    .stApp { background-color: var(--bg); color: var(--text); font-family: 'Rajdhani', sans-serif; }
    
    /* Mobile-First Metrics */
    div[data-testid="stMetric"] {
        background: rgba(20, 20, 20, 0.8);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        margin-bottom: 5px;
    }
    div[data-testid="stMetricLabel"] { font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #888; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono'; font-size: 1.4rem; font-weight: 700; color: #FFF; }

    /* Custom Cards */
    .quantum-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* Mobile Report Card Styling */
    .report-card {
        background-color: #1f2833;
        border-left: 5px solid #45a29e;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .report-header {
        font-family: 'JetBrains Mono';
        font-size: 1.1rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
        border-bottom: 1px solid #45a29e;
        padding-bottom: 5px;
        text-transform: uppercase;
    }
    .report-item {
        margin-bottom: 8px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        color: #c5c6c7;
    }
    .highlight { color: #66fcf1; font-weight: bold; }

    /* Touch-Friendly Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid var(--accent);
        color: var(--accent);
        font-weight: bold;
        height: 3em; 
        border-radius: 8px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: var(--accent);
        color: #000;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: transparent; overflow-x: auto; }
    .stTabs [data-baseweb="tab"] { 
        height: 45px; background-color: #111; 
        border: 1px solid #333; border-radius: 4px; 
        color: #888; font-family: 'JetBrains Mono'; font-size: 0.9rem;
        min-width: 80px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: var(--accent) !important; 
        color: #000 !important; font-weight: bold;
    }
</style>
"""
st.markdown(QUANTUM_CSS, unsafe_allow_html=True)

# =============================================================================
# 2. UNIVERSAL MATH LIBRARY
# =============================================================================
class QuantMath:
    """Optimized vector math for all engines"""
    
    @staticmethod
    def wma(series: pd.Series, length: int) -> pd.Series:
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def hma(series: pd.Series, length: int) -> pd.Series:
        half = int(length / 2)
        sqrt = int(math.sqrt(length))
        wmaf = QuantMath.wma(series, length)
        wmah = QuantMath.wma(series, half)
        diff = 2 * wmah - wmaf
        return QuantMath.wma(diff, sqrt)

    @staticmethod
    def rma(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return QuantMath.rma(tr, length)

    @staticmethod
    def normalize(series: pd.Series, window: int = 100) -> pd.Series:
        mn = series.rolling(window).min()
        mx = series.rolling(window).max()
        return (series - mn) / (mx - mn + 1e-10)

    @staticmethod
    def zscore(series: pd.Series, window: int = 60) -> pd.Series:
        mu = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mu) / (std + 1e-10)

# =============================================================================
# 3. ENGINE 1: TITAN / PENTAGRAM / HZQEO (Integrated)
# =============================================================================
def run_titan_engine(df, amp=10, dev=3.0, hma_l=50, tp1=1.5, tp2=3.0, tp3=5.0, mf_l=14, vol_l=20, gann_l=3):
    if df.empty: return df
    df = df.copy().reset_index(drop=True)
    
    # 1. Indicators
    df['atr'] = QuantMath.atr(df, 14)
    df['hma'] = QuantMath.hma(df['close'], hma_l)
    
    # VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol_tp'] = df['tp'] * df['volume']
    df['vwap'] = df['vol_tp'].cumsum() / df['volume'].cumsum()
    
    # Squeeze
    bb_basis = df['close'].rolling(20).mean(); bb_dev = df['close'].rolling(20).std() * 2.0
    kc_basis = df['close'].rolling(20).mean(); kc_dev = df['atr'] * 1.5
    df['in_squeeze'] = ((bb_basis - bb_dev) > (kc_basis - kc_dev)) & ((bb_basis + bb_dev) < (kc_basis + kc_dev))
    
    # Momentum & Flux
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean(); loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain/loss)))
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_l).mean()
    
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, 1e-9)
    eff = body / rng
    direction = np.sign(df['close'] - df['open'])
    raw_flux = direction * eff * (df['volume'] / df['volume'].rolling(50).mean())
    df['flux'] = raw_flux.ewm(span=14).mean()

    # Money Flow
    rsi_source = df['rsi'] - 50; vol_sma = df['volume'].rolling(mf_l).mean()
    df['money_flow'] = (rsi_source * (df['volume'] / vol_sma)).ewm(span=3).mean() 
    
    # Titan Trend (ATR Trailing Stop)
    df['ll'] = df['low'].rolling(amp).min(); df['hh'] = df['high'].rolling(amp).max()
    trend = np.zeros(len(df)); stop = np.full(len(df), np.nan)
    curr_t = 0; curr_s = np.nan
    for i in range(amp, len(df)):
        c = df.at[i,'close']; d = df.at[i,'atr']*dev
        if curr_t == 0:
            s = df.at[i,'ll'] + d
            curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
            if c < curr_s: curr_t = 1; curr_s = df.at[i,'hh'] - d
        else:
            s = df.at[i,'hh'] - d
            curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
            if c > curr_s: curr_t = 0; curr_s = df.at[i,'ll'] + d
        trend[i] = curr_t; stop[i] = curr_s
    
    df['is_bull'] = trend == 0
    df['entry_stop'] = stop
    
    # Signals
    cond_buy = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False)) & (df['rvol']>1.0)
    cond_sell = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True)) & (df['rvol']>1.0)
    df['buy'] = cond_buy; df['sell'] = cond_sell
    
    # Targets
    df['sig_id'] = (df['buy']|df['sell']).cumsum()
    df['entry'] = df.groupby('sig_id')['close'].ffill()
    df['stop_val'] = df.groupby('sig_id')['entry_stop'].ffill()
    risk = abs(df['entry'] - df['stop_val'])
    df['tp1'] = np.where(df['is_bull'], df['entry']+(risk*tp1), df['entry']-(risk*tp1))
    df['tp2'] = np.where(df['is_bull'], df['entry']+(risk*tp2), df['entry']-(risk*tp2))
    df['tp3'] = np.where(df['is_bull'], df['entry']+(risk*tp3), df['entry']-(risk*tp3))

    # Apex & Gann
    apex_base = QuantMath.hma(df['close'], 55); apex_atr = df['atr'] * 1.5
    df['apex_upper'] = apex_base + apex_atr; df['apex_lower'] = apex_base - apex_atr
    df['apex_trend'] = np.where(df['close'] > df['apex_upper'], 1, np.where(df['close'] < df['apex_lower'], -1, 0))

    sma_h = df['high'].rolling(gann_l).mean(); sma_l = df['low'].rolling(gann_l).mean()
    g_trend = np.full(len(df), np.nan); g_act = np.full(len(df), np.nan)
    curr_g_t = 1; curr_g_a = sma_l.iloc[gann_l] if len(sma_l) > gann_l else np.nan
    for i in range(gann_l, len(df)):
        c = df.at[i,'close']; h_ma = sma_h.iloc[i]; l_ma = sma_l.iloc[i]
        prev_a = g_act[i-1] if (i>0 and not np.isnan(g_act[i-1])) else curr_g_a
        if curr_g_t == 1:
            if c < prev_a: curr_g_t = -1; curr_g_a = h_ma
            else: curr_g_a = l_ma
        else:
            if c > prev_a: curr_g_t = 1; curr_g_a = l_ma
            else: curr_g_a = h_ma
        g_trend[i] = curr_g_t; g_act[i] = curr_g_a
    df['gann_trend'] = g_trend; df['gann_act'] = g_act
    
    return df

def calculate_fibonacci(df, lookback=50):
    recent = df.iloc[-lookback:]
    h, l = recent['high'].max(), recent['low'].min()
    d = h - l
    return {
        'fib_382': h - (d * 0.382), 'fib_500': h - (d * 0.500), 'fib_618': h - (d * 0.618),
        'high': h, 'low': l
    }

def calculate_fear_greed_index(df):
    try:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        vol_score = 50 - ((df['log_ret'].rolling(30).std().iloc[-1] - df['log_ret'].rolling(90).std().iloc[-1]) / df['log_ret'].rolling(90).std().iloc[-1]) * 100
        vol_score = max(0, min(100, vol_score))
        rsi = df['rsi'].iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        dist = (df['close'].iloc[-1] - sma_50) / sma_50
        trend_score = 50 + (dist * 1000)
        fg = (vol_score * 0.3) + (rsi * 0.4) + (max(0, min(100, trend_score)) * 0.3)
        return int(fg)
    except: return 50

def run_backtest(df, tp1_r):
    trades = []
    signals = df[(df['buy']) | (df['sell'])]
    for idx, row in signals.iterrows():
        future = df.loc[idx+1 : idx+20]
        if future.empty: continue
        entry = row['close']; stop = row['entry_stop']; tp1 = row['tp1']; is_long = row['is_bull']
        outcome = "PENDING"; pnl = 0
        if is_long:
            if future['high'].max() >= tp1: outcome = "WIN"; pnl = abs(entry - stop) * tp1_r
            elif future['low'].min() <= stop: outcome = "LOSS"; pnl = -abs(entry - stop)
        else:
            if future['low'].min() <= tp1: outcome = "WIN"; pnl = abs(entry - stop) * tp1_r
            elif future['high'].max() >= stop: outcome = "LOSS"; pnl = -abs(entry - stop)
        if outcome != "PENDING": trades.append({'outcome': outcome, 'pnl': pnl})
            
    if not trades: return 0, 0, 0
    df_res = pd.DataFrame(trades)
    total = len(df_res)
    win_rate = (len(df_res[df_res['outcome']=='WIN']) / total) * 100
    net_r = (len(df_res[df_res['outcome']=='WIN']) * tp1_r) - len(df_res[df_res['outcome']=='LOSS'])
    return total, win_rate, net_r

# =============================================================================
# 4. HZQEO ENGINE (Quantum Math)
# =============================================================================
class HZQEOEngine:
    def __init__(self, lookback=100, zeta_terms=30):
        self.lookback = lookback
        self.zeta_terms = zeta_terms

    def normalize_price(self, price, min_p, max_p):
        if max_p - min_p == 0: return 0.5
        return (price - min_p) / (max_p - min_p)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < self.lookback: return pd.DataFrame()
        
        # Simplified Vectorized Proxy for Speed
        df['norm_p'] = QuantMath.normalize(df['close'], self.lookback)
        
        # Entropy
        def get_entropy(window):
            counts, _ = np.histogram(window, bins=10, density=True)
            counts = counts[counts > 0]
            return -np.sum(counts * np.log(counts))
        
        df['entropy_f'] = df['close'].rolling(20).apply(get_entropy, raw=True)
        
        # Tunneling Prob (Barrier)
        roll_max = df['high'].rolling(50).max()
        roll_min = df['low'].rolling(50).min()
        barrier_w = roll_max - roll_min
        energy = df['close']
        
        diff = (roll_max - energy).clip(lower=0)
        df['tunnel_prob'] = np.exp(-2 * np.sqrt(diff) * (barrier_w / (df['close']*0.01)))
        
        # Zeta Oscillator (Synthetic)
        df['zeta_osc'] = np.tanh((df['norm_p'] - 0.5) * 5)
        
        # Final HZQEO
        df['hzqeo'] = np.tanh(df['zeta_osc'] * df['tunnel_prob'] * 2)
        return df

# =============================================================================
# 5. DATA HANDLER (Unified)
# =============================================================================
class UniversalData:
    def __init__(self):
        self.ccxt_ex = ccxt.kraken()
        self.binance_api = "https://api.binance.us/api/v3"
        
    @st.cache_data(ttl=60)
    def fetch_ohlcv(_self, source, symbol, tf, limit=500):
        try:
            if source == "Crypto (Binance API)":
                r = requests.get(f"{_self.binance_api}/klines", params={"symbol": symbol.replace("/",""), "interval": tf, "limit": limit})
                if r.status_code == 200:
                    df = pd.DataFrame(r.json(), columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
                    df.set_index('timestamp', inplace=True, drop=False)
                    return df[['timestamp','open','high','low','close','volume']]
            elif source == "Crypto (CCXT)":
                ohlcv = _self.ccxt_ex.fetch_ohlcv(symbol, tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True, drop=False)
                return df
            else: # Yahoo
                yf_int = "1d"; yf_per = "2y"
                if tf == "1h": yf_int = "1h"; yf_per = "730d"
                if tf == "15m": yf_int = "15m"; yf_per = "60d"
                df = yf.download(symbol, period=yf_per, interval=yf_int, progress=False, auto_adjust=False)
                if not df.empty:
                    df = df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    df['timestamp'] = pd.to_datetime(df.index)
                    return df
        except Exception as e:
            st.error(f"Data Error: {e}")
            return pd.DataFrame()
        return pd.DataFrame()

# =============================================================================
# 6. MOBILE REPORT GENERATOR
# =============================================================================
def generate_mobile_report(row, symbol, tf, fibs, fg_index, smart_stop):
    is_bull = row['is_bull']
    direction = "LONG 🐂" if is_bull else "SHORT 🐻"
    
    titan_sig = 1 if row['is_bull'] else -1
    apex_sig = row['apex_trend']
    gann_sig = row['gann_trend']
    
    score_val = 0
    if titan_sig == apex_sig: score_val += 1
    if titan_sig == gann_sig: score_val += 1
    
    confidence = "LOW"
    if score_val == 2: confidence = "MAX 🔥"
    elif score_val == 1: confidence = "HIGH"

    vol_desc = "Normal"
    if row['rvol'] > 2.0: vol_desc = "IGNITION 🚀"
    
    squeeze_txt = "⚠️ SQUEEZE ACTIVE" if row['in_squeeze'] else "⚪ NO SQUEEZE"
    
    report_html = f"""
    <div class="report-card">
        <div class="report-header">💠 SIGNAL: {direction}</div>
        <div class="report-item">Confidence: <span class="highlight">{confidence}</span></div>
        <div class="report-item">Sentiment: <span class="highlight">{fg_index}/100</span></div>
        <div class="report-item">Squeeze: <span class="highlight">{squeeze_txt}</span></div>
    </div>

    <div class="report-card">
        <div class="report-header">🌊 FLOW & VOL</div>
        <div class="report-item">RVOL: <span class="highlight">{row['rvol']:.2f} ({vol_desc})</span></div>
        <div class="report-item">Money Flow: <span class="highlight">{row['money_flow']:.2f}</span></div>
        <div class="report-item">VWAP Relation: <span class="highlight">{'Above' if row['close'] > row['vwap'] else 'Below'}</span></div>
    </div>

    <div class="report-card">
        <div class="report-header">🎯 EXECUTION PLAN</div>
        <div class="report-item">Entry: <span class="highlight">{row['close']:.4f}</span></div>
        <div class="report-item">🛑 SMART STOP: <span class="highlight">{smart_stop:.4f}</span></div>
        <div class="report-item">1️⃣ TP1 (1.5R): <span class="highlight">{row['tp1']:.4f}</span></div>
        <div class="report-item">2️⃣ TP2 (3.0R): <span class="highlight">{row['tp2']:.4f}</span></div>
        <div class="report-item">3️⃣ TP3 (5.0R): <span class="highlight">{row['tp3']:.4f}</span></div>
    </div>
    """
    return report_html

def send_telegram_msg(token, chat, msg):
    if not token or not chat: return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"}, timeout=5)
        return r.status_code == 200
    except: return False

# =============================================================================
# 7. MAIN APP CONTROLLER
# =============================================================================
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("💠 APEX OMNI")
        
        st.markdown("### 📡 Feed Config")
        source = st.selectbox("Source", ["Crypto (Binance API)", "Crypto (CCXT)", "TradFi (Yahoo)"])
        
        if "Crypto" in source:
            symbol = st.text_input("Asset", value="BTCUSDT" if "Binance" in source else "BTC/USD")
            timeframe = st.selectbox("Interval", ["15m", "1h", "4h", "1d"])
        else:
            symbol = st.text_input("Ticker", value="NVDA")
            timeframe = st.selectbox("Interval", ["1h", "1d", "1wk"])
            
        st.markdown("### ⚙️ Engine Params")
        amp = st.number_input("Titan Amp", 2, 200, 10)
        dev = st.number_input("Dev", 0.5, 10.0, 3.0)
        hz_lookback = st.slider("Quantum Lookback", 50, 200, 100)
        
        st.markdown("### 🤖 Notifications")
        tg_token = st.text_input("Bot Token", value=st.secrets.get("TELEGRAM_TOKEN", ""), type="password")
        tg_chat = st.text_input("Chat ID", value=st.secrets.get("TELEGRAM_CHAT_ID", ""))
        
        if st.button("🔄 Force Refresh"):
            st.cache_data.clear()
            st.rerun()

    # --- Main Logic ---
    data_engine = UniversalData()
    
    with st.spinner(f"Initializing Omni-Core for {symbol}..."):
        df = data_engine.fetch_ohlcv(source, symbol, timeframe)
        
        if not df.empty:
            # 1. Run Titan Logic
            df = run_titan_engine(df, int(amp), dev, 50, 1.5, 3.0, 5.0, 14, 20, 3)
            
            # 2. Run HZQEO Logic
            hz_engine = HZQEOEngine(lookback=hz_lookback)
            df_hz = hz_engine.calculate(df)
            
            # --- HUD Data ---
            last = df.iloc[-1]
            fibs = calculate_fibonacci(df)
            fg_index = calculate_fear_greed_index(df)
            
            if last['is_bull']: smart_stop = min(last['entry_stop'], fibs['fib_618'] * 0.9995) 
            else: smart_stop = max(last['entry_stop'], fibs['fib_618'] * 1.0005)

            # --- MOBILE METRICS GRID ---
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                # TradingView Widget customized for mobile height
                tv_symbol = f"BINANCE:{symbol}" if "Crypto" in source else symbol
                components.html(f"""
                <div class="tradingview-widget-container">
                  <div class="tradingview-widget-container__widget"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js" async>
                  {{ "symbol": "{tv_symbol}", "width": "100%", "colorTheme": "dark", "isTransparent": true, "locale": "en" }}
                  </script>
                </div>
                """, height=120)
            with c_m2:
                st.metric("TREND", "BULL 🐂" if last['gann_trend']==1 else "BEAR 🐻")

            c_m3, c_m4 = st.columns(2)
            with c_m3: st.metric("STOP", f"{smart_stop:.2f}")
            with c_m4: st.metric("TP3", f"{last['tp3']:.2f}")

            # --- REPORT & ACTIONS ---
            report_html = generate_mobile_report(last, symbol, timeframe, fibs, fg_index, smart_stop)
            st.markdown(report_html, unsafe_allow_html=True)
            
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("🔥 ALERT TG", use_container_width=True):
                    msg = f"TITAN SIGNAL: {symbol} | {'LONG' if last['is_bull'] else 'SHORT'} | EP: {last['close']}"
                    if send_telegram_msg(tg_token, tg_chat, msg): st.success("SENT")
                    else: st.error("FAIL")
            with b_col2:
                if st.button("📝 REPORT TG", use_container_width=True):
                     txt_rep = report_html.replace("<br>", "\n").replace("<div>", "").replace("</div>", "\n")
                     if send_telegram_msg(tg_token, tg_chat, f"REPORT: {symbol}\n{txt_rep}"): st.success("SENT")
                     else: st.error("FAIL")

            # --- MAIN CHART ---
            fig = go.Figure()
            fig.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price')
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', name='HMA', line=dict(color='#66fcf1', width=1)))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vwap'], mode='lines', name='VWAP', line=dict(color='#9933ff', width=2)))
            
            buys = df[df['buy']]; sells = df[df['sell']]
            if not buys.empty: fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00ff00'), name='BUY'))
            if not sells.empty: fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ff0000'), name='SELL'))
            
            fig.update_layout(height=400, template='plotly_dark', margin=dict(l=0,r=0,t=20,b=20), xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1, x=0))
            st.plotly_chart(fig, use_container_width=True)

            # --- TABS ---
            t1, t2, t3 = st.tabs(["📊 GANN", "🌊 FLOW", "⚛️ QUANTUM"])
            
            with t1:
                f1 = go.Figure()
                f1.add_candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
                df_g = df.dropna(subset=['gann_act'])
                f1.add_trace(go.Scatter(x=df_g['timestamp'], y=df_g['gann_act'], mode='markers', marker=dict(color=np.where(df_g['gann_trend']==1, '#00ff00', '#ff0000'), size=3)))
                f1.update_layout(height=300, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(f1, use_container_width=True)

            with t2:
                f2 = go.Figure()
                cols = ['#00e676' if x > 0 else '#ff1744' for x in df['money_flow']]
                f2.add_trace(go.Bar(x=df['timestamp'], y=df['money_flow'], marker_color=cols))
                f2.update_layout(height=300, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(f2, use_container_width=True)

            with t3:
                # 
                f3 = make_subplots(rows=2, cols=1, shared_xaxes=True)
                f3.add_trace(go.Scatter(x=df_hz.index, y=df_hz['hzqeo'], line=dict(color='#FFD600')), row=1, col=1)
                f3.add_hline(y=0.8, line_dash="dot", line_color="red", row=1, col=1)
                f3.add_hline(y=-0.8, line_dash="dot", line_color="green", row=1, col=1)
                f3.add_trace(go.Scatter(x=df_hz.index, y=df_hz['tunnel_prob'], fill='tozeroy', line=dict(color='#00E5FF')), row=2, col=1)
                f3.update_layout(height=300, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(f3, use_container_width=True)

        else:
            st.error("Data fetch failed. Check symbol or source.")

if __name__ == "__main__":
    main()
