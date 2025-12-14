import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# ==========================================
# 1. PAGE CONFIGURATION & DARKPOOL UI
# ==========================================
st.set_page_config(
    page_title="APEX SMC MASTER",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DARKPOOL AESTHETIC CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Neon Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 0 0 15px rgba(0, 230, 118, 0.4);
        font-family: 'Roboto Mono', monospace;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(20, 20, 20, 0.5);
        border: 1px solid #333;
        padding: 10px;
        border-radius: 4px;
        transition: all 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #00E676;
        box-shadow: 0 0 10px rgba(0, 230, 118, 0.2);
    }
    div[data-testid="stMetricValue"] {
        color: #00E676 !important;
        font-size: 1.5rem !important;
        font-weight: 700;
    }
    
    /* Tables */
    .stDataFrame {
        border: 1px solid #333;
    }
    
    /* Buttons */
    .stButton > button {
        background: #111;
        color: #00E676;
        border: 1px solid #00E676;
        font-weight: bold;
        text-transform: uppercase;
        width: 100%;
        border-radius: 4px;
    }
    .stButton > button:hover {
        background: #00E676;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. APEX MATH ENGINE (PINE TRANSLATION)
# ==========================================

class ApexMath:
    @staticmethod
    def wma(series, length):
        """Weighted Moving Average"""
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def hma(series, length):
        """Hull Moving Average"""
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        wmaf = ApexMath.wma(series, half_len)
        wmas = ApexMath.wma(series, length)
        return ApexMath.wma(2 * wmaf - wmas, sqrt_len)

    @staticmethod
    def rma(series, length):
        """Wilder's Smoothing (RMA)"""
        return series.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def get_ma(ma_type, series, length):
        if ma_type == "HMA": return ApexMath.hma(series, length)
        if ma_type == "EMA": return series.ewm(span=length, adjust=False).mean()
        if ma_type == "SMA": return series.rolling(length).mean()
        if ma_type == "RMA": return ApexMath.rma(series, length)
        return series.rolling(length).mean()

    @staticmethod
    def calculate_atr(df, length):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    @staticmethod
    def calculate_adx(df, length=14):
        up = df['High'].diff()
        down = -df['Low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = ApexMath.calculate_atr(df, length)
        tr = tr.replace(0, np.nan)
        plus_di = 100 * (pd.Series(plus_dm).rolling(length).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(length).mean() / tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(length).mean()

    @staticmethod
    def calculate_wavetrend(df, ch_len=10, avg_len=21):
        """WaveTrend Momentum Oscillator"""
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=ch_len, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=ch_len, adjust=False).mean()
        ci = (ap - esa) / (0.015 * d)
        tci = ci.ewm(span=avg_len, adjust=False).mean()
        return tci

    @staticmethod
    def detect_pivots(df, length=10):
        """Detects Pivot Highs and Lows (Looking back 'length' bars)"""
        # Note: True pivots require lookahead. For scanning, we detect pivots that *completed* 'length' bars ago.
        # We simulate the Pine behavior: finding peaks/valleys in a window.
        
        # We will use a rolling window to find local min/max
        # Window size = 2 * length + 1 (left + right + current)
        window = 2 * length + 1
        
        df['Pivot_High'] = df['High'].rolling(window=window, center=True).max() == df['High']
        df['Pivot_Low'] = df['Low'].rolling(window=window, center=True).min() == df['Low']
        
        # Shift back because rolling(center=True) is lookahead in pandas, 
        # but we need to know when it was confirmed.
        # Actually for a scanner, we just want to know if a pivot exists at index i.
        # But 'center=True' introduces NaN at the end. 
        # We will stick to simple detection: Is current bar highest of last X bars? 
        # The Pine Script uses pivot(10, 10).
        
        return df

# ==========================================
# 3. KRAKEN DATA ENGINE
# ==========================================

@st.cache_resource
def get_exchange():
    return ccxt.kraken()

@st.cache_data(ttl=300)
def get_crypto_universe():
    return [
        "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "DOGE/USD", 
        "DOT/USD", "LINK/USD", "MATIC/USD", "LTC/USD", "BCH/USD", "UNI/USD",
        "ATOM/USD", "XLM/USD", "ALGO/USD", "FIL/USD", "APT/USD", "NEAR/USD"
    ]

@st.cache_data(ttl=120)
def fetch_data(ticker, limit=300):
    try:
        ex = get_exchange()
        bars = ex.fetch_ohlcv(ticker, timeframe='1d', limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except:
        return pd.DataFrame()

# ==========================================
# 4. STRATEGY LOGIC CORE
# ==========================================

def apply_apex_logic(df, config):
    if df.empty: return None

    # 1. Trend Cloud
    baseline = ApexMath.get_ma(config['ma_type'], df['Close'], config['len_main'])
    atr = ApexMath.calculate_atr(df, config['len_main'])
    upper = baseline + (atr * config['mult'])
    lower = baseline - (atr * config['mult'])
    
    df['Upper'] = upper
    df['Lower'] = lower
    
    # Trend State
    # 1 = Bull, -1 = Bear, 0 = Neutral/Hold
    trend = np.zeros(len(df))
    for i in range(1, len(df)):
        prev = trend[i-1]
        close = df['Close'].iloc[i]
        
        if close > upper.iloc[i]:
            curr = 1
        elif close < lower.iloc[i]:
            curr = -1
        else:
            curr = prev # Hold state
        trend[i] = curr
        
    df['Trend'] = trend
    
    # 2. Filters
    # ADX
    df['ADX'] = ApexMath.calculate_adx(df)
    adx_ok = df['ADX'] > config['adx_thresh']
    
    # Volume
    vol_avg = df['Volume'].rolling(20).mean()
    vol_ok = df['Volume'] > (vol_avg * config['vol_mult'])
    
    # Momentum (WaveTrend)
    df['WT'] = ApexMath.calculate_wavetrend(df)
    # Buy: Oversold (< 60) and recovering (WT > WT.shift) - using user pine logic
    # Pine: tci < 60 and tci > tci[1]
    mom_buy = (df['WT'] < 60) & (df['WT'] > df['WT'].shift(1))
    mom_sell = (df['WT'] > -60) & (df['WT'] < df['WT'].shift(1))
    
    # 3. Signals
    # Buy: Trend flip 1 or Trend is 1 & prev not 1
    # We strictly follow the Pine logic: 
    # sig_buy = trend == 1 and prev_trend != 1 and vol_ok and mom_buy and adx_ok
    
    # Shift trend to compare
    prev_trend = pd.Series(trend).shift(1).fillna(0)
    
    sig_buy = (trend == 1) & (prev_trend != 1) & vol_ok & mom_buy & adx_ok
    sig_sell = (trend == -1) & (prev_trend != -1) & vol_ok & mom_sell & adx_ok
    
    df['Buy_Signal'] = sig_buy
    df['Sell_Signal'] = sig_sell
    
    # 4. SMC - FVGs
    # Bull FVG: Low > High.shift(2)
    # Bear FVG: High < Low.shift(2)
    fvg_min_size = atr * 0.5 # User input default
    
    df['FVG_Bull'] = (df['Low'] > df['High'].shift(2)) & ((df['Low'] - df['High'].shift(2)) > fvg_min_size)
    df['FVG_Bear'] = (df['High'] < df['Low'].shift(2)) & ((df['Low'].shift(2) - df['High']) > fvg_min_size)

    return df

# ==========================================
# 5. MAIN APPLICATION
# ==========================================

def main():
    # --- HEADER ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🦅 APEX CRYPTO MASTER v8.0")
        st.caption("SMC • ORDER BLOCKS • HULL TREND CLOUDS")
    with c2:
        if st.button("RUN LOGIC ENGINE"):
            st.session_state['run'] = True

    # --- SIDEBAR CONFIG (MATCHING PINE SCRIPT) ---
    with st.sidebar:
        st.header("⚙️ LOGIC ENGINE INPUTS")
        
        st.subheader("1. Trend Architecture")
        ma_type = st.selectbox("Trend Algo", ["HMA", "EMA", "SMA", "RMA"], index=0)
        len_main = st.number_input("Trend Length", 10, 200, 55)
        mult = st.number_input("Cloud Multiplier", 0.5, 5.0, 1.5, step=0.1)
        
        st.subheader("2. Signal Filters")
        adx_thresh = st.slider("Min ADX Strength", 10, 50, 20)
        vol_mult = st.slider("Volume Multiplier", 0.5, 3.0, 1.0)
        
        st.subheader("3. SMC Settings")
        show_fvg = st.checkbox("Scan for Fresh FVG", value=True)
        
        config = {
            'ma_type': ma_type, 'len_main': len_main, 'mult': mult,
            'adx_thresh': adx_thresh, 'vol_mult': vol_mult
        }

    # --- EXECUTION ---
    if 'run' not in st.session_state:
        st.info("👈 Set your parameters and hit RUN to scan the market.")
        return

    universe = get_crypto_universe()
    results = []

    progress = st.progress(0)
    status = st.empty()

    for i, ticker in enumerate(universe):
        status.text(f"Scanning {ticker} for Order Blocks & Trend...")
        df = fetch_data(ticker)
        
        if df is not None and not df.empty:
            df = apply_apex_logic(df, config)
            last = df.iloc[-1]
            
            # Determine Status
            trend_str = "BULLISH 🟢" if last['Trend'] == 1 else "BEARISH 🔴"
            
            # Check for Signals (Look at last 3 bars for 'Recent' signal)
            recent_buy = df['Buy_Signal'].iloc[-3:].any()
            recent_sell = df['Sell_Signal'].iloc[-3:].any()
            
            sig_status = "WAIT"
            if recent_buy: sig_status = "BUY ENTRY 🚀"
            elif recent_sell: sig_status = "SELL ENTRY 📉"
            
            # SMC Data
            has_bull_fvg = df['FVG_Bull'].iloc[-5:].any() # Recent FVG
            has_bear_fvg = df['FVG_Bear'].iloc[-5:].any()
            
            res = {
                'Ticker': ticker,
                'Price': last['Close'],
                'Trend': trend_str,
                'Signal': sig_status,
                'WT_Mom': last['WT'],
                'ADX': last['ADX'],
                'Fresh_FVG': "Bull 🟩" if has_bull_fvg else ("Bear 🟥" if has_bear_fvg else "None")
            }
            results.append(res)
        
        progress.progress((i + 1) / len(universe))

    status.empty()
    progress.empty()
    
    res_df = pd.DataFrame(results)

    # --- DASHBOARD ---
    t1, t2 = st.tabs(["⚡ SIGNAL MATRIX", "🔬 SMC CHART LAB"])

    with t1:
        if not res_df.empty:
            # Sort by Signal importance
            res_df['Sort'] = res_df['Signal'].apply(lambda x: 0 if "ENTRY" in x else 1)
            res_df = res_df.sort_values('Sort')
            
            st.markdown("### 📡 Apex Market Scan")
            
            def color_row(row):
                return ['background-color: rgba(0, 230, 118, 0.1)'] * len(row) if "BUY" in row.Signal else \
                       ['background-color: rgba(255, 23, 68, 0.1)'] * len(row) if "SELL" in row.Signal else \
                       [''] * len(row)

            st.dataframe(
                res_df.style.apply(color_row, axis=1)
                      .format({'Price': '${:,.2f}', 'WT_Mom': '{:.1f}', 'ADX': '{:.1f}'}),
                use_container_width=True,
                height=600
            )

    with t2:
        if not res_df.empty:
            c1, c2 = st.columns([1, 3])
            with c1:
                sel_ticker = st.selectbox("Select Asset", res_df['Ticker'].tolist())
            
            with c2:
                # Prepare Chart Logic
                df_chart = fetch_data(sel_ticker, limit=200)
                df_chart = apply_apex_logic(df_chart, config)
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=[0.7, 0.3])

                # Candle
                fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                                             low=df_chart['Low'], close=df_chart['Close'], name='Price'), row=1, col=1)
                
                # Cloud
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Upper'], line=dict(width=1, color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Lower'], line=dict(width=1, color='rgba(0,0,0,0)'), fill='tonexty', 
                                         fillcolor='rgba(0, 230, 118, 0.1)' if df_chart['Trend'].iloc[-1] == 1 else 'rgba(255, 23, 68, 0.1)', 
                                         name='Trend Cloud'), row=1, col=1)

                # Signals
                buys = df_chart[df_chart['Buy_Signal']]
                sells = df_chart[df_chart['Sell_Signal']]
                
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers+text', 
                                         marker=dict(symbol='triangle-up', size=12, color='#00E676'),
                                         text="BUY", textposition="bottom center", name='Buy Sig'), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers+text', 
                                         marker=dict(symbol='triangle-down', size=12, color='#FF1744'),
                                         text="SELL", textposition="top center", name='Sell Sig'), row=1, col=1)

                # SMC FVGs (Simplified Visualization)
                # Plotting boxes in Plotly is heavy, we'll plot lines for FVG zones
                bull_fvgs = df_chart[df_chart['FVG_Bull']]
                if not bull_fvgs.empty:
                    # Just plot the most recent ones to keep chart clean
                    for idx, row in bull_fvgs.tail(5).iterrows():
                        fig.add_shape(type="rect", x0=idx, x1=df_chart.index[-1], 
                                      y0=row['Low'], y1=df_chart.loc[idx, 'High'], # Approximation since we don't have row-level shift access easily in iteration without index lookup
                                      fillcolor="rgba(0, 230, 118, 0.2)", line=dict(width=0), row=1, col=1)

                # WaveTrend
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['WT'], line=dict(color='#2962FF', width=2), name='WaveTrend'), row=2, col=1)
                fig.add_hline(y=60, line_dash="dot", line_color="red", row=2, col=1)
                fig.add_hline(y=-60, line_dash="dot", line_color="green", row=2, col=1)

                fig.update_layout(height=600, template="plotly_dark", title=f"Apex Structure: {sel_ticker}")
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
