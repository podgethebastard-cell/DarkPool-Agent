import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import yfinance as yf
import ccxt
import requests
import time
import datetime
import math
from scipy.stats import linregress
from datetime import timedelta
import logging
import io

# NEW IMPORT FOR AI (Wrapped to prevent crash if missing)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# =============================================================================
# 1. GLOBAL SYSTEM CONFIGURATION & DARKPOOL CSS
# =============================================================================
st.set_page_config(
    page_title="Titan OS | Financial Singularity",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# THE TITAN DARKPOOL AESTHETIC
st.markdown("""
<style>
    /* GLOBAL RESET & FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;800&display=swap');
    
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto Mono', monospace;
        letter-spacing: -0.5px;
        color: #ffffff;
    }

    /* TITAN HEADER GLOW */
    .titan-header {
        background: linear-gradient(180deg, rgba(16,16,20,1) 0%, rgba(5,5,5,0) 100%);
        border-top: 3px solid #00ffbb;
        padding: 2rem 0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .titan-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00ffbb, #7d00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 255, 187, 0.2);
        margin: 0;
    }
    .titan-subtitle {
        font-family: 'Roboto Mono', monospace;
        color: #666;
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* METRIC CARDS */
    div[data-testid="metric-container"] {
        background: #0f0f0f;
        border: 1px solid #222;
        border-left: 4px solid #333;
        padding: 15px;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-left-color: #00ffbb;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetricLabel"] { font-family: 'Roboto Mono', monospace; font-size: 0.8rem; color: #888; }
    div[data-testid="stMetricValue"] { font-family: 'Inter', sans-serif; font-weight: 800; font-size: 1.8rem; color: #fff; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #111;
        border-radius: 4px;
        color: #666;
        border: 1px solid #222;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ffbb !important;
        color: #000 !important;
        border: none;
        font-weight: bold;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #080808;
        border-right: 1px solid #222;
    }
    
    /* BUTTONS */
    div.stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #333;
        color: #00ffbb;
        font-family: 'Roboto Mono', monospace;
        font-weight: bold;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        border-color: #00ffbb;
        box-shadow: 0 0 10px rgba(0, 255, 187, 0.2);
    }

    /* CUSTOM ALERTS */
    .titan-alert {
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.9rem;
    }
    .alert-bull { background: rgba(0, 255, 187, 0.1); border: 1px solid #00ffbb; color: #00ffbb; }
    .alert-bear { background: rgba(255, 17, 85, 0.1); border: 1px solid #ff1155; color: #ff1155; }
    
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CORE MATH & LOGIC LIBRARY (The Brain)
# =============================================================================
class TitanMath:
    @staticmethod
    def rma(series, period):
        return series.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return TitanMath.rma(tr, length)

    @staticmethod
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def hma(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma_half = TitanMath.wma(series, half_length)
        wma_full = TitanMath.wma(series, length)
        diff = 2 * wma_half - wma_full
        return TitanMath.wma(diff, sqrt_length)

    @staticmethod
    def rsi(series, length=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(series, length=20, std_dev=2.0):
        sma = series.rolling(window=length).mean()
        std = series.rolling(window=length).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def smc_detect(df, lookback=5):
        # Detect Order Blocks and Fair Value Gaps
        df['Pivot_High'] = df['High'].rolling(window=lookback*2+1, center=True).max() == df['High']
        df['Pivot_Low'] = df['Low'].rolling(window=lookback*2+1, center=True).min() == df['Low']
        
        # FVG Detection
        fvg_bull = (df['Low'] > df['High'].shift(2))
        fvg_bear = (df['High'] < df['Low'].shift(2))
        return fvg_bull, fvg_bear

    @staticmethod
    def squeeze_momentum(df):
        # Bollinger Bands
        upper, basis, lower = TitanMath.bollinger_bands(df['Close'], 20, 2.0)
        # Keltner Channels
        tr = TitanMath.atr(df, 20)
        k_upper = basis + (tr * 1.5)
        k_lower = basis - (tr * 1.5)
        
        squeeze_on = (lower > k_lower) & (upper < k_upper)
        
        # Momentum (Linear Regression)
        x = np.arange(20)
        # Simplified LinReg slope for speed
        mom = df['Close'].rolling(20).apply(lambda y: linregress(x, y - y.mean())[0] if len(y)==20 else 0, raw=True)
        return squeeze_on, mom

# =============================================================================
# 3. DATA ENGINE (Unified Feed)
# =============================================================================
class DataFeed:
    @staticmethod
    @st.cache_data(ttl=60)
    def fetch_crypto_kraken(symbol, timeframe, limit):
        try:
            exchange = ccxt.kraken()
            # Convert timeframe to CCXT format if needed (simplified)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            st.error(f"Kraken API Error: {e}")
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_stock_yfinance(symbol, period, interval):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            st.error(f"YFinance Error: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_macro_snapshot():
        # Fetches key macro data for the sidebar/header
        tickers = ['^GSPC', 'BTC-USD', 'DX-Y.NYB', '^TNX']
        data = yf.download(tickers, period="5d", interval="1d", progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        snapshot = {}
        for t in tickers:
            if t in data.columns:
                curr = data[t].iloc[-1]
                prev = data[t].iloc[-2]
                chg = ((curr - prev) / prev) * 100
                snapshot[t] = (curr, chg)
        return snapshot

# =============================================================================
# 4. MODULES (The Merged Functionality)
# =============================================================================

# --- MODULE A: LIVE TRADING TERMINAL (from app.py / agent4.py) ---
def render_live_terminal(api_key):
    st.markdown("### ⚡ Live Execution Terminal")
    
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        asset_class = st.selectbox("Asset Class", ["Crypto (Kraken)", "Stocks (Yahoo)"])
    with c2:
        if "Crypto" in asset_class:
            symbol = st.text_input("Symbol", "BTC/USD")
            tf = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)
        else:
            symbol = st.text_input("Symbol", "NVDA")
            tf = st.selectbox("Timeframe", ["15m", "1h", "1d"], index=1)
            
    # Data Fetching
    if "Crypto" in asset_class:
        df = DataFeed.fetch_crypto_kraken(symbol, tf, 500)
    else:
        p_map = {"15m": "5d", "1h": "1mo", "1d": "1y"}
        df = DataFeed.fetch_stock_yfinance(symbol, p_map.get(tf, "1mo"), tf)

    if df.empty:
        st.warning("No Data Available. Check Symbol.")
        return

    # --- ENGINE CALCS ---
    # Apex Trend (HMA)
    df['HMA'] = TitanMath.hma(df['Close'], 55)
    df['ATR'] = TitanMath.atr(df)
    df['Apex_Trend'] = np.where(df['Close'] > df['HMA'], 1, -1)
    
    # Squeeze
    df['Squeeze'], df['Mom'] = TitanMath.squeeze_momentum(df)
    
    # RSI & Vol
    df['RSI'] = TitanMath.rsi(df['Close'])
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Smart Money
    df['FVG_Bull'], df['FVG_Bear'] = TitanMath.smc_detect(df)

    last = df.iloc[-1]
    
    # --- UI METRICS ---
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price", f"${last['Close']:.2f}")
    m2.metric("Trend (Apex)", "BULL" if last['Apex_Trend']==1 else "BEAR", delta_color="normal")
    m3.metric("Momentum", f"{last['Mom']:.2f}", delta="Squeeze" if last['Squeeze'] else "Active")
    m4.metric("RSI (14)", f"{last['RSI']:.1f}")
    m5.metric("RVOL", f"{last['RVOL']:.1f}x")

    # --- CHARTING ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Main Chart
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='#00ffbb' if last['Apex_Trend']==1 else '#ff1155', width=2), name="Apex HMA"), row=1, col=1)
    
    # FVG Rectangles (Visual only for last 50 candles to keep chart clean)
    recent = df.iloc[-50:]
    for i, row in recent.iterrows():
        if row['FVG_Bull']:
            fig.add_shape(type="rect", x0=i, x1=i, y0=row['Low'], y1=row['High'], fillcolor="green", opacity=0.3, line_width=0, xref="x", yref="y", row=1, col=1)
    
    # Subchart (Squeeze)
    colors = ['#00ffbb' if v > 0 else '#ff1155' for v in df['Mom']]
    fig.add_trace(go.Bar(x=df.index, y=df['Mom'], marker_color=colors, name="Momentum"), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig, use_container_width=True)

    # --- AI ANALYST ---
    if st.button("🤖 Summon Titan AI Analyst"):
        if OpenAI and api_key:
            client = OpenAI(api_key=api_key)
            prompt = f"""
            Analyze {symbol} ({tf}). 
            Price: {last['Close']}. Trend: {'Bull' if last['Apex_Trend']==1 else 'Bear'}.
            RSI: {last['RSI']:.1f}. RVOL: {last['RVOL']:.1f}.
            Squeeze: {'ON' if last['Squeeze'] else 'OFF'}.
            Momentum: {last['Mom']:.2f}.
            
            Give a 3-bullet executive summary on:
            1. Structure (Trend/SMC)
            2. Volatility State
            3. Actionable Bias (Long/Short/Wait)
            """
            try:
                resp = client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content":prompt}])
                st.success(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"AI Error: {e}")
        else:
            st.warning("OpenAI API Key required in settings.")

# --- MODULE B: MACRO INTELLIGENCE (from agent5.py / agentIE.py) ---
def render_macro_desk():
    st.markdown("### 🌍 Macro Intelligence Desk")
    
    # Ratio Definitions
    ratios = {
        "Risk On/Off (SPY/TLT)": ("SPY", "TLT"),
        "Inflation (GLD/TLT)": ("GLD", "TLT"),
        "Tech Strength (QQQ/SPY)": ("QQQ", "SPY"),
        "Crypto Dominance (BTC/ETH)": ("BTC-USD", "ETH-USD")
    }
    
    # Fetch Data Batch
    all_tickers = set([t for pair in ratios.values() for t in pair])
    data = yf.download(list(all_tickers), period="6mo", interval="1d", progress=False)['Close']
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    
    cols = st.columns(len(ratios))
    for i, (name, (num, den)) in enumerate(ratios.items()):
        if num in data and den in data:
            ratio_series = data[num] / data[den]
            curr = ratio_series.iloc[-1]
            prev = ratio_series.iloc[-20] # Monthly change
            chg = ((curr - prev) / prev) * 100
            
            fig = px.line(ratio_series, title=name, template="plotly_dark")
            fig.update_layout(height=200, margin=dict(l=0,r=0,t=30,b=0), xaxis_visible=False)
            
            with cols[i]:
                st.metric(name, f"{curr:.4f}", f"{chg:.2f}% (1M)")
                st.plotly_chart(fig, use_container_width=True)

# --- MODULE C: DEEP SCANNER (from quantfactorinvest.py / ECVSScreener.py) ---
def render_scanner():
    st.markdown("### 🔭 Deep Factor Scanner (ECVS)")
    
    st.info("Scanning S&P 500 subset for High Quality + Value + Momentum")
    
    if st.button("Run Scan"):
        # Simulated Universe (Top 20 for speed, usually this would be 500+)
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "UNH", "XOM", "JNJ", "PG", "LLY", "AVGO", "HD", "MA", "CVX", "MRK", "ABBV"]
        
        results = []
        progress = st.progress(0)
        
        for i, t in enumerate(tickers):
            try:
                tk = yf.Ticker(t)
                info = tk.info
                hist = tk.history(period="6mo")
                
                if hist.empty: continue
                
                # ECVS Logic (Simplified)
                pe = info.get('forwardPE', 20)
                peg = info.get('pegRatio', 1.5)
                mom = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
                
                score = 0
                if pe < 25: score += 1
                if peg < 2.0: score += 1
                if mom > 0.10: score += 1 # Positive Momentum
                if info.get('profitMargins', 0) > 0.20: score += 1
                
                results.append({
                    "Ticker": t,
                    "Score": score,
                    "P/E": pe,
                    "Momentum": mom,
                    "Sector": info.get('sector', 'N/A')
                })
            except: pass
            progress.progress((i+1)/len(tickers))
            
        df = pd.DataFrame(results).sort_values("Score", ascending=False)
        st.dataframe(df.style.background_gradient(subset=["Score"], cmap="Greens"), use_container_width=True)

# --- MODULE D: STRATEGY LAB (from gemini1.py / cursor2.py) ---
def render_strategy_lab():
    st.markdown("### 🧪 Strategy Backtest Lab")
    
    col1, col2 = st.columns(2)
    with col1:
        strat = st.selectbox("Strategy Logic", ["SMA Crossover", "RSI Mean Reversion", "Bollinger Breakout"])
    with col2:
        capital = st.number_input("Starting Capital", value=10000)
    
    if st.button("Run Backtest"):
        # Mock Data Generator (from cursor2.py)
        dates = pd.date_range(end=datetime.datetime.now(), periods=200, freq='D')
        prices = 100 + np.random.randn(200).cumsum()
        df = pd.DataFrame({'Close': prices}, index=dates)
        
        # Strategy Logic
        df['Signal'] = 0
        if strat == "SMA Crossover":
            df['SMA1'] = df['Close'].rolling(10).mean()
            df['SMA2'] = df['Close'].rolling(30).mean()
            df.loc[df['SMA1'] > df['SMA2'], 'Signal'] = 1
            df.loc[df['SMA1'] <= df['SMA2'], 'Signal'] = -1
        
        # Equity Curve
        df['Return'] = df['Close'].pct_change()
        df['Strat_Ret'] = df['Return'] * df['Signal'].shift(1)
        df['Equity'] = capital * (1 + df['Strat_Ret']).cumprod()
        
        # Metrics
        total_ret = ((df['Equity'].iloc[-1] - capital) / capital) * 100
        
        st.metric("Total Return", f"{total_ret:.2f}%", f"${df['Equity'].iloc[-1]:,.2f}")
        st.line_chart(df['Equity'])

# =============================================================================
# 5. MAIN ORCHESTRATOR (The Sidebar Menu)
# =============================================================================

# --- SIDEBAR NAV ---
with st.sidebar:
    st.markdown("## 💠 TITAN OS")
    
    # Macro Snapshot
    try:
        macro = DataFeed.get_macro_snapshot()
        c1, c2 = st.columns(2)
        c1.metric("SPX", f"{macro['^GSPC'][0]:.0f}", f"{macro['^GSPC'][1]:.1f}%")
        c2.metric("BTC", f"{macro['BTC-USD'][0]:.0f}", f"{macro['BTC-USD'][1]:.1f}%")
    except:
        st.caption("Macro Feed Connecting...")

    st.markdown("---")
    menu = st.radio("SYSTEM MODULES", ["Live Terminal", "Macro Intelligence", "Deep Scanner", "Strategy Lab"])
    
    st.markdown("---")
    st.markdown("### 🔐 Credentials")
    api_key = st.text_input("OpenAI Key", type="password", help="For AI Analyst")
    tg_key = st.text_input("Telegram Token", type="password")
    
    with st.expander("ℹ️ System Manual"):
        st.markdown("""
        **Titan OS Guide:**
        1. **Live Terminal:** Real-time crypto/stock charts with HMA/SMC logic.
        2. **Macro:** Cross-asset ratio analysis.
        3. **Scanner:** Institutional fundamental scoring.
        4. **Lab:** Rapid backtesting prototype.
        """)

# --- AUTH CHECK (Password) ---
if "PASSWORD" in st.secrets:
    pwd = st.sidebar.text_input("Access Password", type="password")
    if pwd != st.secrets["PASSWORD"]:
        st.error("🔒 SYSTEM LOCKED. ENTER PASSWORD.")
        st.stop()

# --- LOAD SECRETS INTO SESSION STATE ---
if "OPENAI_API_KEY" in st.secrets and not api_key:
    api_key = st.secrets["OPENAI_API_KEY"]

# --- MAIN ROUTING ---
st.markdown('<div class="titan-header"><h1 class="titan-title">TITAN OS</h1><div class="titan-subtitle">FINANCIAL SINGULARITY INTERFACE</div></div>', unsafe_allow_html=True)

if menu == "Live Terminal":
    render_live_terminal(api_key)
elif menu == "Macro Intelligence":
    render_macro_desk()
elif menu == "Deep Scanner":
    render_scanner()
elif menu == "Strategy Lab":
    render_strategy_lab()

# Footer
st.markdown("---")
st.markdown("<center style='color:#444'>TITAN ARCHITECT SYSTEM v1.0 | DEPLOYED BY QUANTUM CORE</center>", unsafe_allow_html=True)
