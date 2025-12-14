import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# ==========================================
# 1. SYSTEM CONFIGURATION & UI
# ==========================================
st.set_page_config(
    page_title="APEX SMC MASTER v9",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DARKPOOL CSS ---
st.markdown("""
<style>
    /* Global Aesthetic */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* Neon Headers */
    h1, h2, h3 { 
        color: #ffffff !important; 
        text-shadow: 0 0 15px rgba(0, 255, 170, 0.4); 
        font-family: 'Roboto Mono', monospace; 
        text-transform: uppercase; 
        letter-spacing: 2px; 
    }
    
    /* Metrics & Cards */
    div[data-testid="metric-container"] {
        background: rgba(20, 20, 20, 0.6);
        border: 1px solid #333;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] { color: #00E676 !important; font-weight: 700; }
    
    /* Custom Sidebar */
    section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #111 0%, #000 100%);
        color: #00E676;
        border: 1px solid #00E676;
        font-weight: bold;
        text-transform: uppercase;
        border-radius: 4px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ADVANCED MATH & SMC ENGINE
# ==========================================

class ApexMath:
    @staticmethod
    def rma(series, length):
        """Wilder's Smoothing (RMA)"""
        return series.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def atr(df, length):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return ApexMath.rma(tr, length)

    @staticmethod
    def hma(series, length):
        """Hull Moving Average"""
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        
        def wma(s, l):
            weights = np.arange(1, l + 1)
            return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

        wmaf = wma(series, half_len)
        wmas = wma(series, length)
        return wma(2 * wmaf - wmas, sqrt_len)

    @staticmethod
    def wavetrend(df, ch_len=10, avg_len=21):
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=ch_len, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=ch_len, adjust=False).mean()
        ci = (ap - esa) / (0.015 * d)
        tci = ci.ewm(span=avg_len, adjust=False).mean()
        return tci

    @staticmethod
    def smc_processor(df, lookback=10):
        """
        Iterative SMC Logic Engine (Simulates Pine Script 'var' state).
        Detects Order Blocks (OB) and Fair Value Gaps (FVG) and tracks mitigation.
        """
        # --- Pre-calculate Pivots for speed ---
        # A pivot is a high surrounded by lower highs
        df['Pivot_H'] = df['High'].rolling(window=lookback*2+1, center=True).max() == df['High']
        df['Pivot_L'] = df['Low'].rolling(window=lookback*2+1, center=True).min() == df['Low']
        
        # Lists to store identified zones
        # Zone format: {'type': 'bull/bear', 'top': float, 'bottom': float, 'start_idx': int, 'mitigated': bool}
        order_blocks = []
        fvgs = []
        
        # State variables
        last_ph = None
        last_pl = None
        
        # Convert to numpy for faster iteration
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        piv_h = df['Pivot_H'].values
        piv_l = df['Pivot_L'].values
        times = df.index
        
        for i in range(lookback, len(df)-2):
            # 1. Update Structure Points
            if piv_h[i-lookback]: last_ph = highs[i-lookback]
            if piv_l[i-lookback]: last_pl = lows[i-lookback]
            
            # 2. Detect Break of Structure (BOS) -> Identify Order Block
            # Bullish BOS (Close breaks last Pivot High)
            if last_ph and closes[i] > last_ph and closes[i-1] <= last_ph:
                # Find last bearish candle before the break
                # Look back a bit to find the origin of the move
                for j in range(i, i-20, -1):
                    if closes[j] < opens[j]: # Bearish candle
                        ob = {
                            'type': 'bull',
                            'top': highs[j],
                            'bottom': lows[j],
                            'start_idx': times[j],
                            'mitigated': False,
                            'color': 'rgba(0, 230, 118, 0.3)'
                        }
                        order_blocks.append(ob)
                        break
            
            # Bearish BOS
            if last_pl and closes[i] < last_pl and closes[i-1] >= last_pl:
                for j in range(i, i-20, -1):
                    if closes[j] > opens[j]: # Bullish candle
                        ob = {
                            'type': 'bear',
                            'top': highs[j],
                            'bottom': lows[j],
                            'start_idx': times[j],
                            'mitigated': False,
                            'color': 'rgba(255, 23, 68, 0.3)'
                        }
                        order_blocks.append(ob)
                        break

            # 3. Detect FVG (Fair Value Gaps)
            # Bull FVG: Low[i] > High[i-2]
            if i >= 2:
                if lows[i] > highs[i-2]:
                    fvg = {
                        'type': 'bull',
                        'top': lows[i],
                        'bottom': highs[i-2],
                        'start_idx': times[i],
                        'mitigated': False,
                        'color': 'rgba(0, 230, 118, 0.15)'
                    }
                    fvgs.append(fvg)
                
                # Bear FVG: High[i] < Low[i-2]
                if highs[i] < lows[i-2]:
                    fvg = {
                        'type': 'bear',
                        'top': lows[i-2],
                        'bottom': highs[i],
                        'start_idx': times[i],
                        'mitigated': False,
                        'color': 'rgba(255, 23, 68, 0.15)'
                    }
                    fvgs.append(fvg)

            # 4. Check Mitigation (Active Zones Only)
            curr_low = lows[i]
            curr_high = highs[i]
            
            for ob in order_blocks:
                if not ob['mitigated'] and times[i] > ob['start_idx']:
                    if ob['type'] == 'bull' and curr_low < ob['bottom']: ob['mitigated'] = True
                    if ob['type'] == 'bear' and curr_high > ob['top']: ob['mitigated'] = True
                    
            for fvg in fvgs:
                if not fvg['mitigated'] and times[i] > fvg['start_idx']:
                    if fvg['type'] == 'bull' and curr_low < fvg['bottom']: fvg['mitigated'] = True
                    if fvg['type'] == 'bear' and curr_high > fvg['top']: fvg['mitigated'] = True

        return order_blocks, fvgs

# ==========================================
# 3. KRAKEN DATA HANDLER
# ==========================================

@st.cache_resource
def get_exchange():
    return ccxt.kraken()

@st.cache_data(ttl=300)
def get_tickers():
    return [
        "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "DOGE/USD", 
        "DOT/USD", "LINK/USD", "MATIC/USD", "LTC/USD", "BCH/USD", "UNI/USD",
        "ATOM/USD", "XLM/USD", "ALGO/USD", "FIL/USD", "APT/USD", "NEAR/USD",
        "AAVE/USD", "QNT/USD", "MKR/USD"
    ]

@st.cache_data(ttl=60)
def fetch_market_data(ticker, limit=300):
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
# 4. MAIN LOGIC LOOP
# ==========================================

def run_apex_logic(df, config):
    if df.empty: return None, [], []

    # 1. Trend Engine (HMA Cloud)
    baseline = ApexMath.hma(df['Close'], config['len_main'])
    atr = ApexMath.atr(df, config['len_main'])
    df['Upper'] = baseline + (atr * config['mult'])
    df['Lower'] = baseline - (atr * config['mult'])
    
    # 2. Trend State
    trend = np.where(df['Close'] > df['Upper'], 1, np.where(df['Close'] < df['Lower'], -1, 0))
    # Fill zeros with previous value to filter chop (State Persistence)
    # Using pandas ffill logic requires conversion to Series
    trend_s = pd.Series(trend).replace(0, np.nan).ffill().fillna(0).values
    df['Trend'] = trend_s
    
    # 3. Signals (WaveTrend + ADX)
    df['ADX'] = ApexMath.calculate_adx(df)
    df['WT'] = ApexMath.wavetrend(df)
    
    wt_buy = (df['WT'] < -50) & (df['WT'] > df['WT'].shift(1))
    wt_sell = (df['WT'] > 50) & (df['WT'] < df['WT'].shift(1))
    
    # Pine Signal Logic: Trend Match + Trigger
    df['Buy'] = (df['Trend'] == 1) & wt_buy & (df['ADX'] > config['adx_min'])
    df['Sell'] = (df['Trend'] == -1) & wt_sell & (df['ADX'] > config['adx_min'])
    
    # 4. Run SMC Engine (Heavy Calculation)
    obs, fvgs = ApexMath.smc_processor(df, lookback=10)
    
    return df, obs, fvgs

# ==========================================
# 5. UI & VISUALIZATION
# ==========================================

def plot_apex_chart(ticker, df, obs, fvgs, config):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # -- Price --
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    
    # -- Trend Cloud --
    if config['show_cloud']:
        color_fill = 'rgba(0, 230, 118, 0.1)' if df['Trend'].iloc[-1] == 1 else 'rgba(255, 23, 68, 0.1)'
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], fill='tonexty', fillcolor=color_fill, line=dict(width=0), name='Apex Cloud'), row=1, col=1)

    # -- SMC Zones (Rectangles) --
    # Filter for active zones if requested
    shapes = []
    
    # Draw Order Blocks
    if config['show_ob']:
        for ob in obs[-15:]: # Show last 15 detected OBs to avoid clutter
            if config['hide_mitigated'] and ob['mitigated']: continue
            
            shapes.append(dict(
                type="rect", x0=ob['start_idx'], x1=df.index[-1], 
                y0=ob['bottom'], y1=ob['top'],
                fillcolor=ob['color'], line=dict(width=0), xref="x", yref="y"
            ))

    # Draw FVGs
    if config['show_fvg']:
        for fvg in fvgs[-15:]:
            if config['hide_mitigated'] and fvg['mitigated']: continue
            
            shapes.append(dict(
                type="rect", x0=fvg['start_idx'], x1=df.index[-1], 
                y0=fvg['bottom'], y1=fvg['top'],
                fillcolor=fvg['color'], line=dict(width=0), xref="x", yref="y"
            ))

    fig.update_layout(shapes=shapes)

    # -- Signals --
    buys = df[df['Buy']]
    sells = df[df['Sell']]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00E676'), name='Apex Buy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#FF1744'), name='Apex Sell'), row=1, col=1)

    # -- Momentum --
    fig.add_trace(go.Scatter(x=df.index, y=df['WT'], line=dict(color='#2962FF', width=2), name='WaveTrend'), row=2, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='red', row=2, col=1)
    fig.add_hline(y=-50, line_dash='dot', line_color='green', row=2, col=1)
    
    fig.update_layout(height=700, template="plotly_dark", title=f"Apex Structure: {ticker}", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. APP EXECUTION
# ==========================================

def main():
    # --- HEADER ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🦅 APEX CRYPTO MASTER v9.0")
        st.markdown("**SMC LOGIC ENGINE • HMA TREND • ACTIVE ZONE TRACKING**")
    with c2:
        if st.button("🔄 REFRESH MARKET DATA", type="primary"):
            st.cache_data.clear()
            st.rerun()

    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.header("🛠️ LOGIC CONTROLS")
        
        st.subheader("1. Trend Engine")
        len_main = st.number_input("Trend Length", 20, 200, 55)
        mult = st.number_input("Cloud Multiplier", 0.5, 4.0, 1.5, step=0.1)
        
        st.subheader("2. SMC Visuals")
        show_ob = st.checkbox("Show Order Blocks", True)
        show_fvg = st.checkbox("Show FVGs", True)
        hide_mit = st.checkbox("Hide Mitigated Zones", True, help="Only show zones that price hasn't revisited yet.")
        
        st.subheader("3. Signal Filters")
        adx_min = st.slider("Min ADX Strength", 10, 50, 20)
        
        config = {
            'len_main': len_main, 'mult': mult, 'adx_min': adx_min,
            'show_cloud': True, 'show_ob': show_ob, 'show_fvg': show_fvg,
            'hide_mitigated': hide_mit
        }

    # --- MAIN LOOP ---
    universe = get_tickers()
    
    # Store results for dashboard
    scan_results = []
    
    progress_bar = st.progress(0)
    
    # We will display the chart of the first selected asset later
    # But first, we scan everything to build the dashboard
    
    with st.spinner("Processing SMC Logic on Kraken Universe..."):
        for i, ticker in enumerate(universe):
            df = fetch_market_data(ticker)
            if not df.empty:
                df, obs, fvgs = run_apex_logic(df, config)
                last = df.iloc[-1]
                
                # Active Zone Counts
                active_ob = len([x for x in obs if not x['mitigated']])
                active_fvg = len([x for x in fvgs if not x['mitigated']])
                
                # Signal Status
                status = "WAIT"
                if last['Buy']: status = "BUY 🟢"
                elif last['Sell']: status = "SELL 🔴"
                
                scan_results.append({
                    'Ticker': ticker,
                    'Price': last['Close'],
                    'Trend': "BULL" if last['Trend'] == 1 else "BEAR",
                    'Signal': status,
                    'Active OBs': active_ob,
                    'Active FVGs': active_fvg,
                    'ADX': last['ADX']
                })
            progress_bar.progress((i + 1) / len(universe))
            
    progress_bar.empty()
    
    # --- DASHBOARD & CHART ---
    if scan_results:
        df_res = pd.DataFrame(scan_results)
        
        # Style the dashboard
        def highlight_sig(val):
            color = '#00E676' if 'BUY' in val else '#FF1744' if 'SELL' in val else 'transparent'
            return f'background-color: {color}; color: white; font-weight: bold'

        t1, t2 = st.tabs(["⚡ APEX SIGNAL BOARD", "🔬 SMC CHART INSPECTOR"])
        
        with t1:
            st.dataframe(
                df_res.style.map(highlight_sig, subset=['Signal'])
                      .format({'Price': '${:,.2f}', 'ADX': '{:.1f}'}),
                use_container_width=True,
                height=500
            )
            
        with t2:
            c_sel, c_info = st.columns([1, 3])
            with c_sel:
                selected_ticker = st.selectbox("Inspect Asset", df_res['Ticker'].tolist())
            
            # Fetch and Run again for the selected ticker (to get full plotting data)
            df_plot = fetch_market_data(selected_ticker, limit=365) # Get more history for chart
            df_plot, obs_plot, fvgs_plot = run_apex_logic(df_plot, config)
            
            plot_apex_chart(selected_ticker, df_plot, obs_plot, fvgs_plot, config)

if __name__ == "__main__":
    main()
