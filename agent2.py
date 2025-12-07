import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import streamlit.components.v1 as components
from scipy.stats import linregress

# ==========================================
# 1. PREMIUM UI CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Titan Terminal: God Mode", page_icon="⚡")

# --- CUSTOM CSS: GLASSMORPHISM & NEON ---
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
        font-family: 'SF Mono', 'Roboto Mono', monospace;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00E676;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #fff;
    }
    
    /* Signal Colors */
    .bull { color: #00E676 !important; text-shadow: 0 0 10px rgba(0, 230, 118, 0.4); }
    .bear { color: #FF1744 !important; text-shadow: 0 0 10px rgba(255, 23, 68, 0.4); }
    .neu  { color: #B0BEC5 !important; }
    
    /* Header Glow */
    .titan-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E676, #2979FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: #121212;
        border-radius: 8px;
        color: #888;
        border: 1px solid #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00E676 !important;
        color: #000 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROL ---
st.sidebar.header("⚡ System Control")

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI Key", type="password")

# --- TOP 100+ ASSETS ---
crypto_assets = {
    "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Solana (SOL)": "SOL-USD",
    "BNB (BNB)": "BNB-USD", "XRP (XRP)": "XRP-USD", "Cardano (ADA)": "ADA-USD",
    "Avalanche (AVAX)": "AVAX-USD", "Dogecoin (DOGE)": "DOGE-USD", "Polkadot (DOT)": "DOT-USD",
    "Chainlink (LINK)": "LINK-USD", "Polygon (MATIC)": "MATIC-USD", "Shiba Inu (SHIB)": "SHIB-USD",
    "Litecoin (LTC)": "LTC-USD", "Near (NEAR)": "NEAR-USD", "Pepe (PEPE)": "PEPE24478-USD",
    "Bonk (BONK)": "BONK-USD", "Fetch.ai (FET)": "FET-USD", "Render (RNDR)": "RNDR-USD",
    "Optimism (OP)": "OP-USD", "Arbitrum (ARB)": "ARB11841-USD", "Injective (INJ)": "INJ-USD",
    "Kaspa (KAS)": "KAS-USD", "Stellar (XLM)": "XLM-USD", "Cosmos (ATOM)": "ATOM-USD",
    "Uniswap (UNI)": "UNI7083-USD", "Aptos (APT)": "APT21794-USD", "Sui (SUI)": "SUI20947-USD",
    "Thorchain (RUNE)": "RUNE-USD", "Fantom (FTM)": "FTM-USD", "Stacks (STX)": "STX4847-USD",
    "Immutable (IMX)": "IMX-USD", "Mantle (MNT)": "MNT27075-USD", "Celestia (TIA)": "TIA-USD"
}

ticker_name = st.sidebar.selectbox("Target Asset", list(crypto_assets.keys()))
ticker = crypto_assets[ticker_name]
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)

# ==========================================
# 2. ADVANCED MATH ENGINE
# ==========================================
def calculate_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    diff = 2 * wma_half - wma_full
    return calculate_wma(diff, sqrt_length)

def calculate_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return calculate_rma(tr, length)

def calculate_engine(df):
    """Core Calculation Pipeline"""
    
    # 1. APEX TREND MASTER
    apex_len = 55
    apex_mult = 1.5
    df['Apex_Base'] = calculate_hma(df['Close'], apex_len)
    df['Apex_ATR'] = calculate_atr(df, apex_len)
    df['Apex_Upper'] = df['Apex_Base'] + (df['Apex_ATR'] * apex_mult)
    df['Apex_Lower'] = df['Apex_Base'] - (df['Apex_ATR'] * apex_mult)
    
    # Trend State (1=Bull, -1=Bear, 0=Neutral/Hold)
    df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, 
                       np.where(df['Close'] < df['Apex_Lower'], -1, np.nan))
    df['Apex_Trend'] = df['Apex_Trend'].ffill().fillna(0)

    # Liquidity Zones (Pivots)
    liq_len = 10
    df['Pivot_High'] = df['High'].rolling(liq_len*2+1, center=True).max()
    df['Pivot_Low'] = df['Low'].rolling(liq_len*2+1, center=True).min()
    df['Supply_Zone'] = np.where(df['High'] == df['Pivot_High'], df['High'], np.nan)
    df['Demand_Zone'] = np.where(df['Low'] == df['Pivot_Low'], df['Low'], np.nan)

    # 2. VOLUME DELTA & MFI
    df['Vol_Delta'] = np.where(df['Close'] >= df['Open'], df['Volume'], -df['Volume'])
    
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    pos_flow = np.where(tp > tp.shift(1), rmf, 0)
    neg_flow = np.where(tp < tp.shift(1), rmf, 0)
    mfi_ratio = pd.Series(pos_flow).rolling(14).sum() / pd.Series(neg_flow).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # 3. OPTIMIZED ADX
    up = df['High'].diff()
    down = -df['Low'].diff()
    p_dm = np.where((up > down) & (up > 0), up, 0)
    m_dm = np.where((down > up) & (down > 0), down, 0)
    tr = calculate_atr(df, 14)
    p_di = 100 * calculate_rma(pd.Series(p_dm), 14) / tr
    m_di = 100 * calculate_rma(pd.Series(m_dm), 14) / tr
    dx = (np.abs(p_di - m_di) / (p_di + m_di)) * 100
    df['ADX'] = calculate_rma(dx, 14)

    # 4. DARKPOOL MACD (Volume Scaled)
    e12 = df['Close'].ewm(span=12).mean()
    e26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = e12 - e26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Hist'] = (df['MACD'] - df['Signal']) * (df['Volume'] / df['Volume'].rolling(20).mean())

    # 5. STRATEGY SIGNALS
    # Momentum (12)
    mom = df['Close'] - df['Close'].shift(12)
    df['Sig_Mom'] = np.where((mom > 0) & (mom.shift(1) > 0), 1, np.where((mom < 0) & (mom.shift(1) < 0), -1, 0))
    
    # Bollinger Breakout
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['Sig_BB'] = np.where(df['Close'] < (sma20 - 2*std20), 1, np.where(df['Close'] > (sma20 + 2*std20), -1, 0))

    return df

@st.cache_data(ttl=300)
def get_data(ticker, interval):
    p_map = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
    period = p_map.get(interval, "1y")
    d_int = "1h" if interval == "4h" else interval
    df = yf.download(ticker, period=period, interval=d_int, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if interval == "4h":
        agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
        df = df.resample('4h').agg(agg).dropna()
    return df

# ==========================================
# 3. DASHBOARD RENDERER
# ==========================================
st.markdown(f'<div class="titan-header">⚡ TITAN TERMINAL: {ticker_name}</div>', unsafe_allow_html=True)

# 1. TRADINGVIEW WIDGET (Top & Center)
tv_int_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
tv_sym = f"BINANCE:{ticker.replace('-USD', 'USDT')}"
components.html(
    f"""<div class="tradingview-widget-container"><div id="tradingview_widget"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{"width": "100%","height": 450,"symbol": "{tv_sym}","interval": "{tv_int_map[interval]}","timezone": "Etc/UTC","theme": "dark","style": "1","locale": "en","toolbar_bg": "#f1f3f6","enable_publishing": false,"hide_side_toolbar": false,"allow_symbol_change": true,"container_id": "tradingview_widget"}});</script></div>""",
    height=460,
)

df = get_data(ticker, interval)

if df is not None and not df.empty:
    df = calculate_engine(df)
    last = df.iloc[-1]
    
    # --- 2. SIGNAL HUD (Heads-Up Display) ---
    st.markdown("### 🧬 Market DNA")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # Helper to generate HTML card
    def card(label, value, condition):
        color_class = "bull" if condition == 1 else "bear" if condition == -1 else "neu"
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{value}</div>
        </div>
        """
    
    # Apex Trend Status
    apex_cond = last['Apex_Trend']
    apex_txt = "BULLISH" if apex_cond == 1 else "BEARISH" if apex_cond == -1 else "NEUTRAL"
    c1.markdown(card("Apex Trend", apex_txt, apex_cond), unsafe_allow_html=True)
    
    # Momentum Status
    mom_cond = last['Sig_Mom']
    mom_txt = "POSITIVE" if mom_cond == 1 else "NEGATIVE"
    c2.markdown(card("Momentum (12)", mom_txt, mom_cond), unsafe_allow_html=True)
    
    # Volatility Status
    adx_val = last['ADX']
    adx_cond = 1 if adx_val > 25 else 0
    c3.markdown(card("Trend Strength", f"{adx_val:.1f}", adx_cond), unsafe_allow_html=True)
    
    # Money Flow
    mfi_val = last['MFI']
    mfi_cond = 1 if mfi_val < 20 else -1 if mfi_val > 80 else 0
    mfi_txt = "OVERSOLD" if mfi_val < 20 else "OVERBOUGHT" if mfi_val > 80 else "NEUTRAL"
    c4.markdown(card("Money Flow", mfi_txt, mfi_cond), unsafe_allow_html=True)
    
    # Bollinger State
    bb_cond = last['Sig_BB']
    bb_txt = "DIP BUY" if bb_cond == 1 else "RIP SELL" if bb_cond == -1 else "WAIT"
    c5.markdown(card("Mean Reversion", bb_txt, bb_cond), unsafe_allow_html=True)

    # --- 3. DEEP DIVE TABS ---
    st.markdown("<br>", unsafe_allow_html=True)
    tab_apex, tab_macd, tab_ai = st.tabs(["🌊 Apex Master Chart", "📊 DarkPool Flows", "🤖 AI Strategist"])

    with tab_apex:
        # The Master Chart
        fig = go.Figure()
        
        # Candles
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        
        # Apex Cloud
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', 
            fillcolor='rgba(0, 230, 118, 0.15)' if last['Apex_Trend'] == 1 else 'rgba(255, 23, 68, 0.15)',
            line=dict(width=0), name="Apex Cloud"))
        
        # Baseline
        base_col = '#00E676' if last['Apex_Trend'] == 1 else '#FF1744'
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Base'], line=dict(color=base_col, width=2), name="Apex Baseline"))
        
        # Smart Liquidity Zones (Supply/Demand)
        supply = df['Supply_Zone'].dropna()
        demand = df['Demand_Zone'].dropna()
        fig.add_trace(go.Scatter(x=supply.index, y=supply, mode='markers', marker=dict(symbol='triangle-down', size=10, color='#FF1744'), name="Supply Zone"))
        fig.add_trace(go.Scatter(x=demand.index, y=demand, mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00E676'), name="Demand Zone"))

        fig.update_layout(height=600, template="plotly_dark", title=f"Apex Trend & Liquidity Master ({interval})", xaxis_rangeslider_visible=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    with tab_macd:
        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Histogram with DarkPool Colors
        cols = []
        for i in range(len(df)):
            curr = df['Hist'].iloc[i]
            prev = df['Hist'].iloc[i-1] if i > 0 else 0
            if curr >= 0: cols.append('#00E5FF' if curr > prev else '#2979FF') # Cyan/Blue
            else: cols.append('#FF1744' if curr < prev else '#FF9100') # Red/Orange
            
        fig_macd.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color=cols, name="DP Histogram"), row=1, col=1)
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='white', width=1), name="MACD"), row=1, col=1)
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='yellow', width=1), name="Signal"), row=1, col=1)
        
        # Volume Delta
        vd_col = ['#00E676' if v > 0 else '#FF1744' for v in df['Vol_Delta']]
        fig_macd.add_trace(go.Bar(x=df.index, y=df['Vol_Delta'], marker_color=vd_col, name="Vol Delta"), row=2, col=1)
        
        fig_macd.update_layout(height=500, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig_macd, use_container_width=True)

    with tab_ai:
        if st.button("Generate Alpha Report"):
            if not api_key:
                st.error("Missing OpenAI API Key")
            else:
                with st.spinner("Analyzing Apex Structures..."):
                    client = OpenAI(api_key=api_key)
                    prompt = f"""
                    Analyze {ticker} ({interval}) acting as an Institutional Quant.
                    
                    DATA FEED:
                    1. APEX TREND: {apex_txt} (Base: {last['Apex_Base']:.2f}).
                    2. MOMENTUM: {mom_txt}.
                    3. ADX: {adx_val:.2f} ({'Trending' if adx_val > 25 else 'Chop'}).
                    4. MONEY FLOW: {mfi_val:.1f} ({mfi_txt}).
                    5. LIQUIDITY: Last Supply Zone @ {last['Pivot_High']:.2f}, Demand @ {last['Pivot_Low']:.2f}.
                    
                    MISSION:
                    Synthesize these metrics into a precision trade plan. 
                    - Is the Apex Cloud supporting price? 
                    - Are we at a Liquidity Extremum (Supply/Demand)?
                    - VERDICT: LONG / SHORT / WAIT.
                    """
                    res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":prompt}])
                    st.info(res.choices[0].message.content)

else:
    st.error("Initializing Data Stream...")
