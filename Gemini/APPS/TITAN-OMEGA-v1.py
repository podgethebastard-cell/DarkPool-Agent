"""
TITAN OMEGA - PRODUCTION TERMINAL
Version 21.0: Final Union | Enhanced Visualization | Deep AI Context
"""
import time
import math
import random
from typing import Dict, Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from datetime import datetime, timezone
import yfinance as yf

# AI Engine Import
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None 

# =============================================================================
# 1. PAGE CONFIGURATION & VISUAL THEME
# =============================================================================
st.set_page_config(
    page_title="TITAN OMEGA",
    layout="wide",
    page_icon="💀",
    initial_sidebar_state="expanded"
)

# Professional Dark/Neon CSS
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0e1117; }
    
    /* Metrics Cards - Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(30, 33, 39, 0.6);
        border: 1px solid rgba(102, 252, 241, 0.2);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        backdrop-filter: blur(5px);
    }
    
    /* Typography */
    h1, h2, h3 { font-family: 'Roboto Mono', monospace; color: #e0e0e0; letter-spacing: -0.5px; }
    div[data-testid="stMetricLabel"] { color: #8b9bb4; font-size: 14px; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #45a29e;
        color: #66fcf1;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #45a29e;
        color: #0b0c10;
        box-shadow: 0 0 10px #45a29e;
    }
    
    /* Chat & Tables */
    .stChatMessage { background-color: rgba(31, 40, 51, 0.5); border: 1px solid #45a29e; }
    div[data-testid="stMarkdownContainer"] table { width: 100%; border-collapse: collapse; background-color: #1e2127; }
    div[data-testid="stMarkdownContainer"] th { background-color: #45a29e; color: #0b0c10; padding: 10px; }
    div[data-testid="stMarkdownContainer"] td { border-bottom: 1px solid #333; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. UTILITY & GEOFENCING
# =============================================================================
def get_binance_url():
    """Auto-detects API reachability for Global vs US users"""
    try:
        requests.get("https://api.binance.com/api/v3/ping", timeout=1)
        return "https://api.binance.com/api/v3"
    except:
        return "https://api.binance.us/api/v3"

BINANCE_API_BASE = get_binance_url()
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# =============================================================================
# 3. CLASSIC ENGINE (Dark Singularity Logic)
# =============================================================================
def calculate_atr_classic(df: pd.DataFrame, length: int = 10) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def calculate_supertrend_classic(df: pd.DataFrame, period: int = 10, multiplier: float = 4.0):
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    atr = calculate_atr_classic(df, period).values
    m = len(df)
    
    basic_upper = (high + low) / 2 + (multiplier * atr)
    basic_lower = (high + low) / 2 - (multiplier * atr)
    final_upper, final_lower = np.zeros(m), np.zeros(m)
    supertrend, direction = np.zeros(m), np.zeros(m)
    
    for i in range(1, m):
        # Upper Band
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        # Lower Band
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        # Direction
        prev_dir = direction[i-1] if direction[i-1] != 0 else 1
        if prev_dir == 1:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close[i] > final_upper[i] else -1
            
        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
            
    df['supertrend'] = supertrend
    df['trend_dir'] = direction
    return df

def calculate_apex_vector_classic(df: pd.DataFrame, params: dict) -> pd.Series:
    body_abs = (df['close'] - df['open']).abs()
    range_abs = (df['high'] - df['low']).replace(0, 1e-9)
    efficiency = body_abs / range_abs
    vol_avg = df['volume'].rolling(params['vol_len']).mean().replace(0, 1)
    vol_fact = df['volume'] / vol_avg
    direction = np.sign(df['close'] - df['open'])
    return (direction * efficiency * vol_fact).ewm(span=params['smooth']).mean()

# =============================================================================
# 4. MODERN ENGINE (Titan AI Logic)
# =============================================================================
def calculate_hma(series, length):
    half_len = int(length / 2)
    sqrt_len = int(math.sqrt(length))
    wma_f = series.rolling(length).mean()
    wma_h = series.rolling(half_len).mean()
    diff = 2 * wma_h - wma_f
    return diff.rolling(sqrt_len).mean()

def calculate_fear_greed_index(df):
    try:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        # Volatility Score (Lower vol = Higher Greed usually in crypto bull markets, but we invert for safety)
        vol_score = 50 - ((df['log_ret'].rolling(30).std().iloc[-1] - df['log_ret'].rolling(90).std().iloc[-1]) / df['log_ret'].rolling(90).std().iloc[-1]) * 100
        vol_score = max(0, min(100, vol_score))
        
        rsi = df['rsi'].iloc[-1]
        # Trend Distance
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        dist = (df['close'].iloc[-1] - sma_50) / sma_50
        trend_score = 50 + (dist * 1000)
        
        fg = (vol_score * 0.3) + (rsi * 0.4) + (max(0, min(100, trend_score)) * 0.3)
        return int(fg)
    except: return 50

# =============================================================================
# 5. DATA FEED & PROCESSING HUB
# =============================================================================
@st.cache_data(ttl=5)
def get_klines(symbol_bin: str, interval: str, limit: int) -> pd.DataFrame:
    try:
        r = requests.get(f"{BINANCE_API_BASE}/klines", params={"symbol": symbol_bin, "interval": interval, "limit": limit}, headers=HEADERS, timeout=4)
        if r.status_code == 200:
            df = pd.DataFrame(r.json(), columns=['t','o','h','l','c','v','T','q','n','V','Q','B'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df[['open','high','low','close','volume']] = df[['o','h','l','c','v']].astype(float)
            return df[['timestamp','open','high','low','close','volume']]
    except: pass
    return pd.DataFrame()

def run_engines_universal(df, params):
    if df.empty: return df
    df = df.copy().reset_index(drop=True)
    
    # --- COMMON INDICATORS ---
    df['tr'] = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-df['close'].shift(1)), abs(df['low']-df['close'].shift(1))))
    df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
    df['vwap'] = df['vwap'].cumsum() / df['volume'].cumsum()
    
    # --- MODE SWITCHING ---
    if params['mode'] == 'CLASSIC (Dark Singularity)':
        # Classic Logic
        df = calculate_supertrend_classic(df, params['st_len'], params['st_mult'])
        
        # Choppiness (Log based)
        atr_sum = df['tr'].rolling(14).sum()
        range_diff = (df['high'].rolling(14).max() - df['low'].rolling(14).min()).replace(0, 1e-9)
        df['chop_index'] = 100 * (np.log10(atr_sum / range_diff) / np.log10(14))
        df['is_choppy'] = df['chop_index'] > params['chop_thresh']
        
        # Apex Vector
        df['vector'] = calculate_apex_vector_classic(df, {'vol_len': 50, 'smooth': 5})
        
        # Signals
        df['buy'] = (df['trend_dir'] == 1) & (df['trend_dir'].shift(1) == -1) & (~df['is_choppy'])
        df['sell'] = (df['trend_dir'] == -1) & (df['trend_dir'].shift(1) == 1) & (~df['is_choppy'])
        
        # Fillers for Modern Dashboard compatibility
        df['rsi'] = 50 
        df['money_flow'] = 0
        df['in_squeeze'] = False
        df['hyper_wave'] = 0
        df['rvol'] = 1
        df['hma'] = df['close'] # Dummy
        
    else: # MODERN (Titan AI)
        # HMA
        df['hma'] = calculate_hma(df['close'], params['hma_len'])
        
        # Titan Chandelier Trend
        df['ll'] = df['low'].rolling(10).min(); df['hh'] = df['high'].rolling(10).max()
        trend = np.zeros(len(df)); stop = np.full(len(df), np.nan)
        curr_t = 0; curr_s = np.nan
        for i in range(10, len(df)):
            c = df.at[i,'close']; d = df.at[i,'atr']*3.0
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
        df['trend_dir'] = np.where(df['is_bull'], 1, -1)
        df['entry_stop'] = stop
        
        # Volatility & Squeeze
        delta = df['close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14).mean(); loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain/loss)))
        df['rvol'] = df['volume'] / df['volume'].rolling(20).mean()
        
        bb_dev = df['close'].rolling(20).std() * 2.0
        kc_dev = df['atr'] * 1.5
        df['in_squeeze'] = bb_dev < kc_dev
        
        # Money Flow & HyperWave
        rsi_source = df['rsi'] - 50
        vol_sma = df['volume'].rolling(14).mean()
        df['money_flow'] = (rsi_source * (df['volume'] / vol_sma)).ewm(span=3).mean()
        
        pc = df['close'].diff()
        ds_pc = pc.ewm(span=25).mean().ewm(span=13).mean()
        ds_abs_pc = abs(pc).ewm(span=25).mean().ewm(span=13).mean()
        df['hyper_wave'] = (100 * (ds_pc / ds_abs_pc)) / 2
        
        # Signals
        cond_buy = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False)) & (df['rvol']>1.1)
        cond_sell = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True)) & (df['rvol']>1.1)
        df['buy'] = cond_buy; df['sell'] = cond_sell
        
        # Compatibility mapping
        df['vector'] = df['hyper_wave'] / 50
        df['supertrend'] = df['entry_stop']
        df['chop_index'] = 50

    # --- SHARED: GANN HILO & TARGETS ---
    sma_h = df['high'].rolling(params['gann_len']).mean()
    sma_l = df['low'].rolling(params['gann_len']).mean()
    g_trend = np.full(len(df), np.nan); g_act = np.full(len(df), np.nan)
    curr_g_t = 1; curr_g_a = sma_l.iloc[params['gann_len']] if len(sma_l) > params['gann_len'] else np.nan
    for i in range(params['gann_len'], len(df)):
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

    # Targets
    df['sig_id'] = (df['buy']|df['sell']).cumsum()
    df['entry'] = df.groupby('sig_id')['close'].ffill()
    if 'entry_stop' not in df.columns: df['entry_stop'] = df['supertrend']
    
    risk = abs(df['entry'] - df['entry_stop'])
    df['tp1'] = np.where(df['trend_dir']==1, df['entry']+(risk*params['tp1']), df['entry']-(risk*params['tp1']))
    df['tp2'] = np.where(df['trend_dir']==1, df['entry']+(risk*params['tp2']), df['entry']-(risk*params['tp2']))
    df['tp3'] = np.where(df['trend_dir']==1, df['entry']+(risk*params['tp3']), df['entry']-(risk*params['tp3']))

    return df

def run_backtest(df, params):
    trades = []
    signals = df[(df['buy']) | (df['sell'])]
    for idx, row in signals.iterrows():
        future = df.loc[idx+1 : idx+20]
        if future.empty: continue
        entry = row['close']; stop = row['entry_stop']; tp1 = row['tp1']
        is_long = row['trend_dir'] == 1
        outcome = "PENDING"; pnl = 0
        
        if is_long:
            if future['high'].max() >= tp1: outcome = "WIN"; pnl = abs(entry-stop)*params['tp1']
            elif future['low'].min() <= stop: outcome = "LOSS"; pnl = -abs(entry-stop)
        else:
            if future['low'].min() <= tp1: outcome = "WIN"; pnl = abs(entry-stop)*params['tp1']
            elif future['high'].max() >= stop: outcome = "LOSS"; pnl = -abs(entry-stop)
            
        if outcome != "PENDING": trades.append({'outcome': outcome, 'pnl': pnl})
    
    if not trades: return 0, 0.0, 0.0
    df_res = pd.DataFrame(trades)
    win_rate = (len(df_res[df_res['outcome']=='WIN']) / len(df_res)) * 100
    net_r = df_res['pnl'].sum()
    return len(df_res), win_rate, net_r

def send_telegram_msg(token, chat, msg):
    if not token or not chat: return False
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"}, timeout=5)
        return True
    except: return False

# =============================================================================
# 6. MAIN APPLICATION UI
# =============================================================================
# --- Header ---
c1, c2 = st.columns([3, 1])
with c1: st.title("TITAN OMEGA v21.0"); st.caption(f"PRODUCTION TERMINAL | {BINANCE_API_BASE.split('//')[1].split('/')[0]}")
with c2: components.html("""<script>setInterval(function(){document.getElementById('clk').innerHTML=new Date().toLocaleTimeString();},1000);</script><div id='clk' style='color:#00ffbb;font-family:monospace;font-size:24px;text-align:right;font-weight:bold;text-shadow:0 0 10px #00ffbb;'></div>""", height=50)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ SYSTEM CORE")
    mode = st.radio("Logic Engine", ["MODERN (Titan AI)", "CLASSIC (Dark Singularity)"])
    if st.button("🔄 FORCE REFRESH"): st.cache_data.clear(); st.rerun()
    
    st.subheader("📡 FEED")
    symbol = st.text_input("Asset", "BTCUSDT").upper()
    timeframe = st.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=1)
    
    with st.expander("Parameters"):
        st_len = st.slider("Trend Len", 5, 50, 10)
        st_mult = st.slider("Trend Mult", 1.0, 5.0, 4.0 if 'CLASSIC' in mode else 3.0)
        chop_thresh = st.slider("Chop Thresh", 40, 70, 60)
        gann_len = st.number_input("Gann Len", 3)
        tp1 = st.number_input("TP1 R", 1.5)
        tp2 = st.number_input("TP2 R", 3.0)
        tp3 = st.number_input("TP3 R", 5.0)

    st.subheader("🤖 AI & NOTIFICATIONS")
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    tg_token = st.text_input("TG Token", st.secrets.get("TELEGRAM_TOKEN", ""), type="password")
    tg_chat = st.text_input("TG Chat ID", st.secrets.get("TELEGRAM_CHAT_ID", ""))

# --- Execution ---
df = get_klines(symbol, timeframe, 500)

if not df.empty:
    params = {'mode': mode, 'st_len': st_len, 'st_mult': st_mult, 'chop_thresh': chop_thresh, 'gann_len': gann_len, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'hma_len': 50, 'vol_len':50, 'smooth':5}
    df = run_engines_universal(df, params)
    last = df.iloc[-1]
    
    # --- Metrics ---
    trend_color = "normal" if last['trend_dir'] == 1 else "inverse"
    trend_txt = "BULLISH" if last['trend_dir'] == 1 else "BEARISH"
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ASSET PRICE", f"{last['close']:.2f}")
    m2.metric("TREND REGIME", trend_txt, delta=1 if last['trend_dir']==1 else -1, delta_color="normal")
    
    if 'CLASSIC' in mode:
        m3.metric("CHOP INDEX", f"{last['chop_index']:.1f}", "LOCKED" if last['is_choppy'] else "OPEN", delta_color="inverse")
        m4.metric("APEX VECTOR", f"{last['vector']:.2f}")
    else:
        m3.metric("MARKET SENTIMENT", f"{calculate_fear_greed_index(df)}/100")
        m4.metric("MONEY FLOW", f"{last['money_flow']:.2f}")

    # --- EXCELLENT CHARTS (Pro Visualization) ---
    st.markdown("### 💠 TERMINAL VIEW")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    
    # 1. Main Candlestick (Pro Style)
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price',
        increasing_line_color='#00ffbb', decreasing_line_color='#ff1155',
        increasing_fillcolor='rgba(0, 255, 187, 0.1)', decreasing_fillcolor='rgba(255, 17, 85, 0.1)'
    ), row=1, col=1)
    
    # 2. Logic-Specific Overlays
    if 'CLASSIC' in mode:
        # SuperTrend Line
        st_color = '#00ffbb' if last['trend_dir']==1 else '#ff1155'
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['supertrend'], mode='lines', line=dict(color=st_color, width=2), name='SuperTrend'), row=1, col=1)
        # Classic Markers
        sigs = df[df['buy']]
        if not sigs.empty:
            fig.add_trace(go.Scatter(x=sigs['timestamp'], y=sigs['low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ffbb', line_width=1, line_color='white'), name='BUY SIGNAL'), row=1, col=1)
    else:
        # Modern Overlays (HMA + VWAP)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vwap'], mode='lines', line=dict(color='#ab47bc', width=2, dash='solid'), name='VWAP'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', line=dict(color='#29b6f6', width=1), name='HMA'), row=1, col=1)
        # Squeeze Highlights
        squeeze_on = df[df['in_squeeze']]
        if not squeeze_on.empty:
             fig.add_trace(go.Scatter(x=squeeze_on['timestamp'], y=squeeze_on['high']*1.005, mode='markers', marker=dict(symbol='circle', size=4, color='orange'), name='Squeeze Active'), row=1, col=1)

    # 3. Subplot (Vector or Money Flow)
    if 'CLASSIC' in mode:
        colors = ['#00ffbb' if v > 0 else '#ff1155' for v in df['vector']]
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['vector'], marker_color=colors, name='Apex Vector'), row=2, col=1)
    else:
        colors = ['#00ffbb' if v > 0 else '#ff1155' for v in df['money_flow']]
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['money_flow'], marker_color=colors, name='Money Flow'), row=2, col=1)

    # 4. Layout Polish
    fig.update_layout(
        template='plotly_dark',
        height=700,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='rgba(14, 17, 23, 1)',
        paper_bgcolor='rgba(14, 17, 23, 1)',
        xaxis_rangeslider_visible=False,
        showlegend=False,
        grid={'rows': 2, 'columns': 1, 'pattern': 'independent'}
    )
    # Refine Grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- ANALYSIS HUB ---
    t1, t2 = st.tabs(["📊 PERFORMANCE", "🤖 AI ANALYST"])
    
    with t1:
        # Backtest
        b_count, b_wr, b_net = run_backtest(df, params)
        c_b1, c_b2, c_b3 = st.columns(3)
        c_b1.metric("TOTAL TRADES", b_count)
        c_b2.metric("WIN RATE", f"{b_wr:.1f}%")
        c_b3.metric("NET RETURN", f"{b_net:.1f}R", delta=b_net)
        
    with t2:
        # AI Context Injection
        if openai_key and OpenAI:
            client = OpenAI(api_key=openai_key)
            
            # Construct Technical Context String
            tech_context = (
                f"MODE: {mode}\n"
                f"PRICE: {last['close']}\n"
                f"TREND: {trend_txt}\n"
                f"RSI: {last['rsi']:.1f}\n"
                f"VOLATILITY_RVOL: {last['rvol']:.2f}\n"
                f"VWAP: {last['vwap']:.2f}\n"
                f"SQUEEZE_ACTIVE: {last['in_squeeze']}\n"
                f"MONEY_FLOW: {last['money_flow']:.2f}\n"
                f"GANN_TREND: {'BULL' if last['gann_trend']==1 else 'BEAR'}\n"
                f"VECTOR: {last['vector'] if 'CLASSIC' in mode else last['hyper_wave']:.2f}\n"
            )
            
            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "Titan Omega AI initialized. I have access to all calculated indicators."}]
            
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
                
            if prompt := st.chat_input("Ask about the chart..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                
                with st.chat_message("assistant"):
                    stream = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": f"You are an elite quantitative analyst. Analyze the user query based on this LIVE DATA:\n{tech_context}\nBe concise, technical, and risk-aware."},
                        ] + st.session_state.messages,
                        stream=True
                    )
                    response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.info("💡 Add OPENAI_API_KEY to secrets.toml to enable the AI Analyst.")

    # --- BROADCAST ---
    if st.button("🔥 BROADCAST SIGNAL TO TELEGRAM", use_container_width=True):
        msg = (f"🔥 *TITAN OMEGA SIGNAL*\n"
               f"Asset: {symbol} [{timeframe}]\n"
               f"Mode: {mode}\n"
               f"Dir: {trend_txt}\n"
               f"Price: {last['close']}\n"
               f"TP1: {last['tp1']:.2f} | TP3: {last['tp3']:.2f}")
        if send_telegram_msg(tg_token, tg_chat, msg): st.success("Broadcast Sent!")
        else: st.error("Broadcast Failed. Check Creds.")
