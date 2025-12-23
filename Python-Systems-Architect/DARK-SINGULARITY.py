# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# ----------------------------
# 1. SYSTEM CONFIGURATION & CSS
# ----------------------------
st.set_page_config(page_title="DARK SINGULARITY // APEX", layout="wide", page_icon="💀")

def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
        
        :root { 
            --bg: #0e1117; 
            --card: #1e2127; 
            --bull: #00ffbb; 
            --bear: #ff1155; 
            --chop: #546e7a;
        }
        
        .stApp { background-color: var(--bg); font-family: 'Roboto Mono', monospace; color: #e0e0e0; }
        
        /* Custom Metric Cards */
        div[data-testid="stMetric"] {
            background-color: var(--card);
            border: 1px solid #333;
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        div[data-testid="stMetricLabel"] { opacity: 0.8; font-size: 0.8rem; }
        div[data-testid="stMetricValue"] { font-weight: 700; font-size: 1.5rem; }
        
        /* Headers */
        h1, h2, h3 { color: #e0e0e0 !important; text-transform: uppercase; letter-spacing: 1px; }
        
        /* Plotly Background */
        .js-plotly-plot .plotly .main-svg { background: transparent !important; }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] { background-color: #0b0d10; border-right: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ----------------------------
# 2. VECTORIZED MATH ENGINE
# ----------------------------

def calculate_atr(df: pd.DataFrame, length: int = 10) -> pd.Series:
    """Wilder's Smoothing (RMA) based ATR to match TradingView."""
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # RMA initialization: First value is SMA, then RMA
    # Pandas ewm(alpha=1/length) is equivalent to RMA
    return tr.ewm(alpha=1/length, adjust=False).mean()

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 4.0):
    """
    Explicit Iterative SuperTrend implementation to match Pine Script recursive logic.
    Returns: supertrend (series), direction (series: 1=Bull, -1=Bear)
    """
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    atr = calculate_atr(df, period).values
    
    m = len(df)
    
    # Basic Bands
    basic_upper = (high + low) / 2 + (multiplier * atr)
    basic_lower = (high + low) / 2 - (multiplier * atr)
    
    final_upper = np.zeros(m)
    final_lower = np.zeros(m)
    supertrend = np.zeros(m)
    direction = np.zeros(m) # 1 = Bull (Trend Up), -1 = Bear (Trend Down)
    
    # Initialize first valid index (usually index=period)
    # We'll just start loops from 1 safely
    
    for i in range(1, m):
        # -- Upper Band Logic --
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        # -- Lower Band Logic --
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        # -- Trend Logic --
        prev_dir = direction[i-1] if direction[i-1] != 0 else 1 # Default Bull
        
        if prev_dir == 1: # Was Bull
            if close[i] < final_lower[i]:
                direction[i] = -1 # Flip Bear
            else:
                direction[i] = 1 # Stay Bull
        else: # Was Bear
            if close[i] > final_upper[i]:
                direction[i] = 1 # Flip Bull
            else:
                direction[i] = -1 # Stay Bear
        
        # -- SuperTrend Value --
        if direction[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]
            
    df['supertrend'] = supertrend
    df['trend_dir'] = direction
    return df

def calculate_choppiness(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Choppiness Index = 100 * LOG10( SUM(ATR(1), n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
    """
    high, low, close = df['High'], df['Low'], df['Close']
    
    # True Range for Summation
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(window=length).sum()
    
    high_max = high.rolling(window=length).max()
    low_min = low.rolling(window=length).min()
    
    range_diff = high_max - low_min
    
    # Avoid div by zero
    range_diff = range_diff.replace(0, 1e-9)
    
    numerator = np.log10(atr_sum / range_diff)
    denominator = np.log10(length)
    
    chop = 100 * (numerator / denominator)
    return chop

def calculate_apex_vector(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Vector = Direction * Efficiency * VolumeFlux
    Efficiency = Body / Range
    """
    # Efficiency
    body_abs = (df['Close'] - df['Open']).abs()
    range_abs = (df['High'] - df['Low']).replace(0, 1e-9)
    efficiency = body_abs / range_abs
    
    # Volume Flux
    vol_avg = df['Volume'].rolling(params['vol_len']).mean().replace(0, 1)
    vol_fact = df['Volume'] / vol_avg
    
    direction = np.sign(df['Close'] - df['Open'])
    
    vector_raw = direction * efficiency * vol_fact
    
    # Smoothing
    return vector_raw.ewm(span=params['smooth']).mean()

# ----------------------------
# 3. LOGIC CONTROLLER
# ----------------------------
def run_dark_singularity(df: pd.DataFrame, params: dict):
    # 1. SuperTrend
    df = calculate_supertrend(df, period=params['st_len'], multiplier=params['st_mult'])
    
    # 2. Choppiness
    df['chop_index'] = calculate_choppiness(df, length=params['chop_len'])
    
    # 3. Apex Vector
    df['vector'] = calculate_apex_vector(df, params)
    
    # 4. State Machine (Integration)
    # Logic: Signal only if NOT Choppy and Trend Flips
    df['is_choppy'] = df['chop_index'] > params['chop_thresh']
    
    # Identify Signal Candles
    # Bull Signal: Trend 1, Prev Trend -1, Not Choppy
    df['sig_long'] = (df['trend_dir'] == 1) & (df['trend_dir'].shift(1) == -1) & (~df['is_choppy'])
    
    # Bear Signal: Trend -1, Prev Trend 1, Not Choppy
    df['sig_short'] = (df['trend_dir'] == -1) & (df['trend_dir'].shift(1) == 1) & (~df['is_choppy'])
    
    return df

# ----------------------------
# 4. VISUALIZATION (PLOTLY)
# ----------------------------
def render_terminal(df: pd.DataFrame, ticker: str, params: dict):
    lookback = 200
    sub_df = df.tail(lookback).copy()
    
    # Colors
    c_bull = '#00ffbb'
    c_bear = '#ff1155'
    c_chop = '#546e7a'
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.75, 0.25],
        subplot_titles=(f"DARK SINGULARITY: {ticker}", "APEX VECTOR FLUX")
    )
    
    # --- CHART 1: PRICE + SUPERTREND ---
    
    # 1. Bar Coloring Logic
    # If Choppy -> Grey. Else -> Trend Color.
    colors = []
    for idx, row in sub_df.iterrows():
        if row['is_choppy'] and params['use_chop']:
            colors.append(c_chop) # Greyed out
        elif row['trend_dir'] == 1:
            colors.append(c_bull)
        else:
            colors.append(c_bear)
            
    fig.add_trace(go.Candlestick(
        x=sub_df.index,
        open=sub_df['Open'], high=sub_df['High'],
        low=sub_df['Low'], close=sub_df['Close'],
        name='Price',
        increasing_line_color=c_bull, decreasing_line_color=c_bear,
        increasing_fillcolor=c_bull, decreasing_fillcolor=c_bear,
        showlegend=False
    ), row=1, col=1)
    
    # Override colors with our custom logic (Plotly trick: separate scatter for custom colors or just use shape coloring)
    # Actually, simpler to just color the SuperTrend line vividly and let candles be standard or trend-colored.
    # To enforce "Regime Candles", we can't easily override internal Candlestick colors per-bar in standard Plotly without complex tricks.
    # We will trust the SuperTrend Line for regime indication.
    
    # 2. SuperTrend Line
    # Create segments to color the line correctly
    st_vals = sub_df['supertrend']
    
    # We draw points. For a continuous line that changes color, we need two traces or a gradient.
    # Simple approach: Two traces, masked.
    st_bull = st_vals.where(sub_df['trend_dir'] == 1)
    st_bear = st_vals.where(sub_df['trend_dir'] == -1)
    
    fig.add_trace(go.Scatter(
        x=sub_df.index, y=st_bull,
        mode='lines', line=dict(color=c_bull, width=3),
        name='SuperTrend Bull'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=sub_df.index, y=st_bear,
        mode='lines', line=dict(color=c_bear, width=3),
        name='SuperTrend Bear'
    ), row=1, col=1)
    
    # 3. Signals
    longs = sub_df[sub_df['sig_long']]
    shorts = sub_df[sub_df['sig_short']]
    
    if not longs.empty:
        fig.add_trace(go.Scatter(
            x=longs.index, y=longs['Low'] * 0.99,
            mode='markers+text', marker_symbol='triangle-up', marker_size=14, marker_color=c_bull,
            text='LONG', textposition='bottom center', textfont=dict(color=c_bull), name='Long Signal'
        ), row=1, col=1)
        
    if not shorts.empty:
        fig.add_trace(go.Scatter(
            x=shorts.index, y=shorts['High'] * 1.01,
            mode='markers+text', marker_symbol='triangle-down', marker_size=14, marker_color=c_bear,
            text='SHORT', textposition='top center', textfont=dict(color=c_bear), name='Short Signal'
        ), row=1, col=1)

    # --- CHART 2: APEX VECTOR ---
    vec_vals = sub_df['vector']
    vec_cols = [c_bull if v > 0.5 else (c_bear if v < -0.5 else c_chop) for v in vec_vals]
    
    fig.add_trace(go.Bar(
        x=sub_df.index, y=vec_vals,
        marker_color=vec_cols,
        name='Vector Flux'
    ), row=2, col=1)
    
    # Thresholds
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)

    # Styling
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=800,
        margin=dict(l=10, r=50, t=50, b=20),
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    
    return fig

# ----------------------------
# 5. MAIN APP INTERFACE
# ----------------------------
def main():
    st.sidebar.markdown("## 💀 THE ARCHITECT")
    st.sidebar.caption("Dark Pool // Singularity Engine")
    
    # INPUTS
    ticker = st.sidebar.text_input("TICKER", "BTC-USD").upper()
    
    with st.sidebar.expander("Trend Architecture", expanded=True):
        st_len = st.slider("ATR Length", 5, 50, 10)
        st_mult = st.slider("Trend Factor", 1.0, 10.0, 4.0, 0.1)
    
    with st.sidebar.expander("Noise Gate (Chop)", expanded=True):
        use_chop = st.checkbox("Active Chop Filter", True)
        chop_len = st.slider("Chop Lookback", 10, 30, 14)
        chop_thresh = st.slider("Chop Threshold", 40.0, 70.0, 60.0, 0.5)

    with st.sidebar.expander("Apex Vector", expanded=False):
        vol_len = st.slider("Volume Memory", 20, 100, 50)
        smooth = st.slider("Smoothing", 2, 20, 5)

    if st.sidebar.button("INITIALIZE", type="primary"):
        # DATA FETCH
        try:
            with st.spinner("Accessing Market Feed..."):
                df = yf.download(ticker, period="1y", interval="1d", progress=False)
                
            if df.empty:
                st.error("Feed connection failed.")
                return
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # RUN ENGINE
            params = {
                'st_len': st_len, 'st_mult': st_mult,
                'use_chop': use_chop, 'chop_len': chop_len, 'chop_thresh': chop_thresh,
                'vol_len': vol_len, 'smooth': smooth
            }
            
            df_calc = run_dark_singularity(df, params)
            
            # HEADS UP DISPLAY
            last = df_calc.iloc[-1]
            prev = df_calc.iloc[-2]
            
            # Status Logic
            is_bull = last['trend_dir'] == 1
            trend_txt = "BULLISH FLOW" if is_bull else "BEARISH FLOW"
            trend_col = "off" if not is_bull else "normal" # Streamlit metric delta color logic
            
            is_choppy = last['chop_index'] > chop_thresh
            gate_txt = "LOCKED (CHOP)" if (is_choppy and use_chop) else "OPEN (TRADE)"
            
            # METRICS
            c1, c2, c3, c4 = st.columns(4)
            
            c1.metric("PRICE", f"{last['Close']:.2f}", f"{last['Close'] - prev['Close']:.2f}")
            c2.metric("REGIME", trend_txt, delta=1 if is_bull else -1, delta_color="normal")
            c3.metric("GATE", gate_txt, f"{last['chop_index']:.1f}", delta_color="off")
            c4.metric("TRAILING STOP", f"{last['supertrend']:.2f}")
            
            # PLOT
            st.markdown("---")
            fig = render_terminal(df_calc, ticker, params)
            st.plotly_chart(fig, use_container_width=True)
            
            # DATA DUMP
            with st.expander("RAW FEED"):
                st.dataframe(df_calc.tail(10)[['Close', 'supertrend', 'trend_dir', 'chop_index', 'vector', 'sig_long', 'sig_short']])
                
        except Exception as e:
            st.error(f"Runtime Error: {e}")

if __name__ == "__main__":
    main()
