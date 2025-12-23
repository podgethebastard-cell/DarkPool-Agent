import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ==========================================
# 1. SYSTEM IDENTITY & ARCHITECTURE
# ==========================================
st.set_page_config(
    page_title="TITAN | SINGULARITY CORE",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DARK POOL CSS INJECTION
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    /* GLOBAL RESET */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* INPUT HARDENING */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background-color: #0f0f0f; color: #00ffbb; border: 1px solid #222; font-family: 'Roboto Mono';
    }
    
    /* HUD METRIC CONTAINERS */
    .titan-metric {
        background: #111; border-left: 3px solid #333; padding: 10px; margin-bottom: 5px;
        transition: all 0.3s;
    }
    .bull-glow { border-left-color: #00ffbb; box-shadow: -5px 0 15px -5px rgba(0,255,187,0.2); }
    .bear-glow { border-left-color: #ff0055; box-shadow: -5px 0 15px -5px rgba(255,0,85,0.2); }
    .chop-glow { border-left-color: #ffcc00; }
    
    .metric-label { font-size: 0.7rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .metric-val { font-size: 1.2rem; font-weight: bold; color: #fff; }
    
    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] { background-color: #000; border-bottom: 1px solid #222; }
    .stTabs [data-baseweb="tab"] { color: #555; font-size: 0.8rem; }
    .stTabs [aria-selected="true"] { color: #00ffbb; border-bottom: 2px solid #00ffbb; }
    
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. QUANTITATIVE MATH ENGINE
# ==========================================

class TitanMath:
    @staticmethod
    def sma(series, length):
        return series.rolling(length).mean()

    @staticmethod
    def ema(series, length):
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def atr(df, length):
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(length).mean() # SMA of TR for consistency with Pine

    @staticmethod
    def rsi(series, length):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def supertrend(df, length, multiplier):
        high, low, close = df['high'], df['low'], df['close']
        atr = TitanMath.atr(df, length)
        hl2 = (high + low) / 2
        
        # Pine Script: src +/- (mult * atr)
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        upper = basic_upper.copy()
        lower = basic_lower.copy()
        trend = np.zeros(len(df))
        
        # Numba-style loop optimization for Python
        for i in range(1, len(df)):
            if basic_upper.iloc[i] < upper.iloc[i-1] or close.iloc[i-1] > upper.iloc[i-1]:
                upper.iloc[i] = basic_upper.iloc[i]
            else:
                upper.iloc[i] = upper.iloc[i-1]

            if basic_lower.iloc[i] > lower.iloc[i-1] or close.iloc[i-1] < lower.iloc[i-1]:
                lower.iloc[i] = basic_lower.iloc[i]
            else:
                lower.iloc[i] = lower.iloc[i-1]
                
            # Trend Logic
            prev_trend = trend[i-1] if i > 0 else 1
            
            if prev_trend == 1:
                if close.iloc[i] < lower.iloc[i-1]:
                    trend[i] = -1
                else:
                    trend[i] = 1
            else:
                if close.iloc[i] > upper.iloc[i-1]:
                    trend[i] = 1
                else:
                    trend[i] = -1
                    
        return trend, upper, lower

    @staticmethod
    def chop_index(df, length):
        high, low, close = df['high'], df['low'], df['close']
        # SUM(ATR(1), length)
        # Note: ATR(1) is just TR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        sum_tr = tr.rolling(length).sum()
        
        highest_high = high.rolling(length).max()
        lowest_low = low.rolling(length).min()
        
        # CI = 100 * LOG10( SUM_TR / (HH - LL) ) / LOG10(length)
        range_hl = highest_high - lowest_low
        # Avoid division by zero
        range_hl = range_hl.replace(0, np.nan) 
        
        x = sum_tr / range_hl
        ci = 100 * np.log10(x) / np.log10(length)
        return ci

# ==========================================
# 4. INDICATOR LOGIC (THE FUSION)
# ==========================================

def calculate_apex_vector(df, vec_len, vol_norm, eff_thresh):
    """Logic from 'Apex Vector [Flux + Efficiency]'"""
    close, open_, high, low, volume = df['close'], df['open'], df['high'], df['low'], df['volume']
    
    # 1. Geometric Efficiency
    range_abs = high - low
    body_abs = (close - open_).abs()
    raw_eff = body_abs / range_abs.replace(0, 1) # Prevent div/0
    efficiency = raw_eff.ewm(span=vec_len, adjust=False).mean()
    
    # 2. Volume Flux
    vol_avg = volume.rolling(vol_norm).mean()
    vol_fact = volume / vol_avg.replace(0, 1)
    
    # 3. Vector
    direction = np.sign(close - open_)
    vector_raw = direction * efficiency * vol_fact
    flux = vector_raw.ewm(span=5, adjust=False).mean()
    
    # States
    df['flux'] = flux
    df['is_super_bull'] = flux > eff_thresh
    df['is_super_bear'] = flux < -eff_thresh
    return df

def calculate_money_flow_matrix(df, mf_len, tsi_long, tsi_short):
    """Logic from 'Money Flow Matrix'"""
    close = df['close']
    volume = df['volume']
    
    # 1. Money Flow
    rsi_src = TitanMath.rsi(close, mf_len) - 50
    mf_vol = volume / volume.rolling(mf_len).mean()
    money_flow = (rsi_src * mf_vol).ewm(span=3).mean()
    
    # 2. Hyper Wave (TSI-like)
    pc = close.diff()
    # Double Smooth
    ss = pc.ewm(span=tsi_long).mean().ewm(span=tsi_short).mean()
    ss_abs = pc.abs().ewm(span=tsi_long).mean().ewm(span=tsi_short).mean()
    
    hyper_wave = np.where(ss_abs != 0, (100 * (ss / ss_abs)) / 2, 0)
    
    df['money_flow'] = money_flow
    df['hyper_wave'] = hyper_wave
    return df

def calculate_smc_structure(df, length, multiplier, fvg_threshold):
    """Logic from 'Apex Trend (SMC)' + FVG detection"""
    # 1. SuperTrend Cloud
    trend, upper, lower = TitanMath.supertrend(df, length, multiplier)
    df['trend_dir'] = trend
    df['trend_upper'] = upper
    df['trend_lower'] = lower
    
    # 2. FVG Detection (Bearish and Bullish Imbalances)
    # Bullish FVG: Low > High[2]
    # Bearish FVG: High < Low[2]
    high, low = df['high'], df['low']
    
    # Shifted for comparison (High[2] means High 2 bars ago)
    # In pandas, shift(2) gets the value from 2 rows above
    
    # Bull FVG: Current Bar's Low vs High of 2 bars ago
    # We detect it at the close of the current bar (index 0) looking at index -2
    # To map to chart, the gap exists between bar i-2 and bar i
    
    fvg_bull = (low > high.shift(2)) & ((low - high.shift(2)) > (TitanMath.atr(df, 14) * 0.5))
    fvg_bear = (high < low.shift(2)) & ((low.shift(2) - high) > (TitanMath.atr(df, 14) * 0.5))
    
    df['fvg_bull'] = fvg_bull
    df['fvg_bear'] = fvg_bear
    
    # 3. Order Blocks (Simplified for speed: Highest candle before down-trend)
    # Not fully implemented to save processing time, focusing on FVG
    return df

# ==========================================
# 5. DATA FEED
# ==========================================
@st.cache_data(ttl=15)
def get_crypto_data(symbol, timeframe, limit):
    try:
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Dark Pool Connection Failed: {e}")
        return pd.DataFrame()

# ==========================================
# 6. APP EXECUTION
# ==========================================

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("## ⚙️ TITAN CONFIG")
    
    st.markdown("### 📡 FEED")
    symbol = st.text_input("Ticker", value="BTC/USD")
    tf = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
    
    st.markdown("### 🌊 APEX PHYSICS")
    vec_len = st.number_input("Vector Length", 14)
    flux_thresh = st.slider("Super Threshold", 0.1, 1.0, 0.6)
    
    st.markdown("### 🏛️ SMC & TREND")
    st_len = st.number_input("Trend ATR", 10)
    st_mult = st.number_input("Trend Factor", 3.0)
    chop_filter = st.checkbox("Chop Gate (60)", value=True)
    
    st.markdown("### ⚖️ RISK CALC")
    acct_size = st.number_input("Equity ($)", 10000)
    risk_pct = st.number_input("Risk %", 1.0)

# --- Main Engine ---
df = get_crypto_data(symbol, tf, 500)

if not df.empty:
    # --- PROCESS INDICATORS ---
    df = calculate_apex_vector(df, vec_len, 55, flux_thresh)
    df = calculate_money_flow_matrix(df, 14, 25, 13)
    df = calculate_smc_structure(df, st_len, st_mult, 0.5)
    df['chop'] = TitanMath.chop_index(df, 14)

    # --- STATUS CHECK ---
    last = df.iloc[-1]
    is_bull = last['trend_dir'] == 1
    is_chop = last['chop'] > 60 and chop_filter
    
    trend_color = "#00ffbb" if is_bull else "#ff0055"
    if is_chop: trend_color = "#ffcc00"
    
    # --- HUD SECTION ---
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="titan-metric {'chop-glow' if is_chop else ('bull-glow' if is_bull else 'bear-glow')}">
            <div class="metric-label">MARKET REGIME</div>
            <div class="metric-val" style="color:{trend_color}">
                {'LOCKED (CHOP)' if is_chop else ('BULL PHASE' if is_bull else 'BEAR PHASE')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        flux_val = last['flux']
        f_col = "#00ffbb" if flux_val > 0 else "#ff0055"
        st.markdown(f"""
        <div class="titan-metric">
            <div class="metric-label">APEX FLUX</div>
            <div class="metric-val" style="color:{f_col}">{flux_val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        stop_level = last['trend_lower'] if is_bull else last['trend_upper']
        dist = abs(last['close'] - stop_level)
        risk_amt = acct_size * (risk_pct / 100)
        pos_size = risk_amt / dist if dist > 0 else 0
        
        st.markdown(f"""
        <div class="titan-metric">
            <div class="metric-label">RISK CALC ({risk_pct}%)</div>
            <div class="metric-val">{pos_size:.4f} Units</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        mf_val = last['money_flow']
        st.markdown(f"""
        <div class="titan-metric">
            <div class="metric-label">MONEY FLOW</div>
            <div class="metric-val" style="color:{'#00ffbb' if mf_val>0 else '#ff0055'}">{mf_val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # 7. HIGH-FIDELITY VISUALIZATION
    # ==========================================
    
    # Create Subplots
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, 
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # --- ROW 1: PRICE + SMC CLOUD + FVG ---
    
    # 1. Candlesticks
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price',
        increasing_line_color='#00ffbb', decreasing_line_color='#ff0055'
    ), row=1, col=1)
    
    # 2. SuperTrend Cloud
    # We plot two lines, fill between them
    # Logic: If Bull, Lower Band is active support. If Bear, Upper Band is active resistance.
    # To replicate the "Cloud", we fill between trend_upper and trend_lower
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['trend_upper'],
        mode='lines', line=dict(color='rgba(255, 0, 85, 0.5)', width=1), showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['trend_lower'],
        mode='lines', line=dict(color='rgba(0, 255, 187, 0.5)', width=1), fill='tonexty',
        fillcolor='rgba(100, 100, 100, 0.1)', showlegend=False
    ), row=1, col=1)

    # 3. FVG Rectangles (Approximation via Scatter Shapes)
    # Drawing shapes for every FVG is heavy, we'll use markers for the start of FVG
    bull_fvg_indices = df[df['fvg_bull']].index
    bear_fvg_indices = df[df['fvg_bear']].index
    
    # We plot markers to indicate FVG zones (High Performance)
    fig.add_trace(go.Scatter(
        x=df.loc[bull_fvg_indices, 'timestamp'], y=df.loc[bull_fvg_indices, 'low'],
        mode='markers', marker=dict(symbol='triangle-up', color='#00ffbb', size=8),
        name='Bull FVG'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.loc[bear_fvg_indices, 'timestamp'], y=df.loc[bear_fvg_indices, 'high'],
        mode='markers', marker=dict(symbol='triangle-down', color='#ff0055', size=8),
        name='Bear FVG'
    ), row=1, col=1)

    # --- ROW 2: APEX VECTOR (FLUX) ---
    colors_flux = ['#00ffbb' if v > 0 else '#ff0055' for v in df['flux']]
    fig.add_trace(go.Bar(
        x=df['timestamp'], y=df['flux'],
        marker_color=colors_flux, name='Flux Vector'
    ), row=2, col=1)
    
    # Threshold Lines
    fig.add_hline(y=flux_thresh, line_dash="dot", line_color="#333", row=2, col=1)
    fig.add_hline(y=-flux_thresh, line_dash="dot", line_color="#333", row=2, col=1)

    # --- ROW 3: MONEY FLOW MATRIX ---
    # Gradient Bar approach simulated via colors
    colors_mf = ['#006400' if v > 0 else '#8b0000' for v in df['money_flow']] # Darker base
    # Add overflow logic color
    colors_mf = [('#00ff00' if v > 0 else '#ff0000') if abs(v) > 50 else c for v, c in zip(df['money_flow'], colors_mf)]
    
    fig.add_trace(go.Bar(
        x=df['timestamp'], y=df['money_flow'],
        marker_color=colors_mf, name='Money Flow'
    ), row=3, col=1)
    
    # HyperWave Line
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['hyper_wave'],
        mode='lines', line=dict(color='#ffffff', width=1), name='HyperWave'
    ), row=3, col=1)

    # --- LAYOUT HARDENING ---
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
        height=800,
        margin=dict(l=0, r=50, t=20, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        hovermode="x unified"
    )
    
    fig.update_xaxes(showgrid=False, color="#333")
    fig.update_yaxes(showgrid=False, color="#333", side="right")
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Awaiting Data Feed...")

# ==========================================
# 8. FOOTER PROTOCOL
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #444; font-size: 0.8em;'>TITAN SYSTEM v3 | ARCHITECT PROTOCOL ACTIVE</div>", unsafe_allow_html=True)
