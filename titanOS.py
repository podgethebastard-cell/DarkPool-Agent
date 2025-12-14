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
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

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

# THE TITAN DARKPOOL AESTHETIC (Merged from agent4, agentIE, agent5)
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

    /* METRIC CARDS (Merged from agentIE) */
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

    /* TABS & SIDEBAR */
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
    section[data-testid="stSidebar"] { background-color: #080808; border-right: 1px solid #222; }
    
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

    /* ALERTS */
    .div-alert {
        background-color: rgba(255, 82, 82, 0.1);
        border: 1px solid #FF5252;
        color: #FF5252;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 0% { opacity: 0.8; } 50% { opacity: 1; } 100% { opacity: 0.8; } }
    
    /* AI BOX */
    .ai-box {
        background: #0a0a0a;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
        border-left: 3px solid #7d00ff;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. UNIFIED MATH & LOGIC ENGINE (TitanMath)
# Merges logic from agent4, agentIE, gemini1, and cursor2
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
        # From ApexCryptoSMCScout.py
        df['Pivot_High'] = df['High'].rolling(window=lookback*2+1, center=True).max() == df['High']
        df['Pivot_Low'] = df['Low'].rolling(window=lookback*2+1, center=True).min() == df['Low']
        # FVG
        fvg_bull = (df['Low'] > df['High'].shift(2))
        fvg_bear = (df['High'] < df['Low'].shift(2))
        return fvg_bull, fvg_bear

    @staticmethod
    def squeeze_momentum(df):
        # From gptagent1.py / agent4.py
        upper, basis, lower = TitanMath.bollinger_bands(df['Close'], 20, 2.0)
        tr = TitanMath.atr(df, 20)
        k_upper = basis + (tr * 1.5)
        k_lower = basis - (tr * 1.5)
        squeeze_on = (lower > k_lower) & (upper < k_upper)
        
        x = np.arange(20)
        # Vectorized LinReg approach
        mom = df['Close'].rolling(20).apply(lambda y: linregress(x, y - y.mean())[0] if len(y)==20 else 0, raw=True)
        return squeeze_on, mom

    @staticmethod
    def money_flow(df, length=14):
        # From agent4.py
        rsi_src = TitanMath.rsi(df['Close'], length) - 50
        mf_vol = df['Volume'] / df['Volume'].rolling(length).mean()
        return (rsi_src * mf_vol).fillna(0).ewm(span=3).mean()

# =============================================================================
# 3. UNIFIED DATA FEED (TitanFeed)
# Handles YFinance and CCXT seamlessly
# =============================================================================
class TitanFeed:
    @staticmethod
    @st.cache_data(ttl=60)
    def fetch_crypto(symbol, timeframe, limit):
        try:
            exchange = ccxt.kraken()
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
    def fetch_stock(symbol, period, interval):
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
        # From agentIE.py
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
# 4. MODULE A: LIVE TERMINAL (The "God Mode" View)
# Merges agent4.py, gptagent1.py, deepseekagent.py
# =============================================================================
def render_live_terminal(api_key):
    st.markdown("### ⚡ Live Execution Terminal")
    
    # 1. Controls
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        asset_class = st.selectbox("Market", ["Crypto (Kraken)", "Stocks (Yahoo)"], help="Select data source")
    with c2:
        default_sym = "BTC/USD" if "Crypto" in asset_class else "NVDA"
        symbol = st.text_input("Symbol", default_sym, help="Ticker symbol")
    with c3:
        tf = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    with c4:
        limit = st.slider("Lookback", 200, 1000, 500)

    # 2. Fetch
    if "Crypto" in asset_class:
        df = TitanFeed.fetch_crypto(symbol, tf, limit)
    else:
        p_map = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
        df = TitanFeed.fetch_stock(symbol, p_map.get(tf, "1mo"), tf)

    if df.empty:
        st.warning("No Data. Check symbol format.")
        return

    # 3. Engine Calculation (Vectorized)
    df['HMA'] = TitanMath.hma(df['Close'], 55)
    df['ATR'] = TitanMath.atr(df)
    df['Upper'] = df['HMA'] + (df['ATR'] * 1.5)
    df['Lower'] = df['HMA'] - (df['ATR'] * 1.5)
    df['Apex_Trend'] = np.where(df['Close'] > df['Upper'], 1, np.where(df['Close'] < df['Lower'], -1, 0))
    
    df['Squeeze'], df['Mom'] = TitanMath.squeeze_momentum(df)
    df['Money_Flow'] = TitanMath.money_flow(df)
    df['RSI'] = TitanMath.rsi(df['Close'])
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['FVG_Bull'], _ = TitanMath.smc_detect(df)

    last = df.iloc[-1]

    # 4. HUD
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price", f"${last['Close']:,.2f}")
    m2.metric("Trend", "BULL" if last['Apex_Trend']==1 else "BEAR" if last['Apex_Trend']==-1 else "CHOP", 
              delta="Strong" if abs(last['Mom']) > 1 else "Weak")
    m3.metric("Momentum", f"{last['Mom']:.2f}", delta="Squeeze ON" if last['Squeeze'] else "Released")
    m4.metric("Money Flow", f"{last['Money_Flow']:.2f}", delta="Inflow" if last['Money_Flow']>0 else "Outflow")
    m5.metric("RVOL", f"{last['RVOL']:.1f}x")

    # 5. Charts
    tab_main, tab_ind = st.tabs(["🕯️ Price & Structure", "📊 Oscillator Suite"])
    
    with tab_main:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        # Price
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        # Cloud
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], fill='tonexty', 
                                 fillcolor='rgba(0, 255, 187, 0.1)' if last['Apex_Trend']==1 else 'rgba(255, 17, 85, 0.1)', 
                                 line=dict(width=0), name="Apex Cloud"), row=1, col=1)
        # HMA
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='#00ffbb' if last['Apex_Trend']==1 else '#ff1155', width=2), name="HMA 55"), row=1, col=1)
        
        # Momentum
        cols = ['#00ffbb' if v > 0 else '#ff1155' for v in df['Mom']]
        fig.add_trace(go.Bar(x=df.index, y=df['Mom'], marker_color=cols, name="Sqz Mom"), row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    with tab_ind:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True)
        # Money Flow
        cols_mf = ['#00ffbb' if v > 0 else '#ff1155' for v in df['Money_Flow']]
        fig2.add_trace(go.Bar(x=df.index, y=df['Money_Flow'], marker_color=cols_mf, name="Money Flow Matrix"), row=1, col=1)
        # RSI
        fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
        fig2.add_hline(y=70, line_dash="dot", row=2, col=1); fig2.add_hline(y=30, line_dash="dot", row=2, col=1)
        fig2.update_layout(height=500, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig2, use_container_width=True)

    # 6. AI & Broadcast
    c_ai, c_cast = st.columns(2)
    with c_ai:
        if st.button("🤖 AI Tactical Report"):
            if OpenAI and api_key:
                client = OpenAI(api_key=api_key)
                prompt = f"""
                Analyze {symbol} on {tf}.
                Price: {last['Close']}. Trend: {last['Apex_Trend']} (1=Bull, -1=Bear).
                Momentum: {last['Mom']:.2f}. Money Flow: {last['Money_Flow']:.2f}.
                RVOL: {last['RVOL']:.2f}.
                Provide a trading plan: 1. Bias, 2. Key Levels, 3. Risks. Keep it concise.
                """
                with st.spinner("Analyzing..."):
                    try:
                        res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}])
                        st.markdown(f"<div class='ai-box'>{res.choices[0].message.content}</div>", unsafe_allow_html=True)
                    except Exception as e: st.error(str(e))
            else: st.warning("API Key missing")

    with c_cast:
        if st.button("📡 Broadcast Signal"):
            if st.secrets.get("TELEGRAM_TOKEN") and st.secrets.get("TELEGRAM_CHAT_ID"):
                msg = f"🔥 TITAN SIGNAL: {symbol}\nTrend: {last['Apex_Trend']}\nPrice: {last['Close']}\nMom: {last['Mom']}"
                requests.post(f"https://api.telegram.org/bot{st.secrets['TELEGRAM_TOKEN']}/sendMessage", 
                              data={"chat_id": st.secrets['TELEGRAM_CHAT_ID'], "text": msg})
                st.success("Signal Sent!")
            else: st.error("Telegram secrets missing")

# =============================================================================
# 5. MODULE B: DEEP SCANNER (Scanner/Screener)
# Merges ECVSScreener.py, quantfactorinvest.py, clivesminers.py, microcaps.py
# =============================================================================
def render_scanner():
    st.markdown("### 🔭 Deep Factor Scanner")
    
    scan_type = st.radio("Scanner Mode", ["Institutional (ECVS)", "Miners (Clive)", "Microcaps (Apex)", "Global 1000"], horizontal=True)
    
    tickers = []
    if scan_type == "Institutional (ECVS)":
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "UNH", "XOM", "JNJ", "PG", "LLY", "AVGO"]
    elif scan_type == "Miners (Clive)":
        tickers = ["NXE", "UEC", "UUUU", "DNN", "LAC", "SGML", "KGC", "EQX", "ERO", "HBM", "MP"]
    elif scan_type == "Microcaps (Apex)":
        tickers = ["VNDA", "ORMP", "SELB", "QUIK", "ATOM", "MARA", "RIOT", "LNN", "TA"]
    elif scan_type == "Global 1000":
        tickers = ["AAPL", "MSFT", "AZN.L", "SHEL.L", "7203.T", "6758.T", "0700.HK", "BHP.AX"]

    if st.button(f"Scan {len(tickers)} Targets"):
        results = []
        bar = st.progress(0)
        
        for i, t in enumerate(tickers):
            try:
                tk = yf.Ticker(t)
                info = tk.info
                hist = tk.history(period="6mo")
                
                if hist.empty: continue
                
                # Universal Metrics
                pe = info.get('forwardPE', 999)
                mkt_cap = info.get('marketCap', 0)
                mom = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
                vol = hist['Close'].pct_change().std() * np.sqrt(252)
                
                row = {
                    "Ticker": t,
                    "Price": hist['Close'].iloc[-1],
                    "Momentum (6M)": mom,
                    "Volatility": vol,
                    "Market Cap": mkt_cap
                }

                # Specialized Logic
                if scan_type == "Miners (Clive)":
                    cash = info.get('totalCash', 0)
                    debt = info.get('totalDebt', 0)
                    row['Net Cash'] = cash - debt
                    row['P/B'] = info.get('priceToBook', 0)
                    score = 1 if row['Net Cash'] > 0 else 0
                    if mom > 0: score += 1
                    
                elif scan_type == "Institutional (ECVS)":
                    peg = info.get('pegRatio', 999)
                    margin = info.get('profitMargins', 0)
                    row['PEG'] = peg
                    row['Margin'] = margin
                    score = 0
                    if pe < 25: score += 1
                    if peg < 1.5: score += 1
                    if margin > 0.2: score += 1
                
                else: # Generic Technical Score
                    score = 0
                    if mom > 0: score += 1
                    if vol < 0.5: score += 1
                
                row['Score'] = score
                results.append(row)
                
            except: pass
            bar.progress((i+1)/len(tickers))
            
        bar.empty()
        
        if results:
            df = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.dataframe(df.style.background_gradient(subset=["Score"], cmap="Greens"), use_container_width=True)
            
            # Export
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            st.download_button("📥 Download Excel", buffer, "scan_results.xlsx")

# =============================================================================
# 6. MODULE C: STRATEGY LAB (Backtester)
# Merges gemini1.py and cursor2.py
# =============================================================================
def render_lab():
    st.markdown("### 🧪 Strategy Lab")
    
    c1, c2 = st.columns(2)
    with c1:
        strat = st.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion", "Bollinger Breakout", "DCA"])
    with c2:
        capital = st.number_input("Starting Capital", value=10000)
    
    if st.button("Run Simulation"):
        # Mock Data for Speed (Real data can be swapped)
        dates = pd.date_range(end=datetime.datetime.now(), periods=365, freq='D')
        prices = 100 + np.random.randn(365).cumsum()
        df = pd.DataFrame({'Close': prices}, index=dates)
        
        # Logic
        df['Signal'] = 0
        if strat == "SMA Crossover":
            df['Fast'] = df['Close'].rolling(10).mean()
            df['Slow'] = df['Close'].rolling(50).mean()
            df.loc[df['Fast'] > df['Slow'], 'Signal'] = 1
            df.loc[df['Fast'] <= df['Slow'], 'Signal'] = -1
        elif strat == "RSI Mean Reversion":
            df['RSI'] = TitanMath.rsi(df['Close'])
            df.loc[df['RSI'] < 30, 'Signal'] = 1
            df.loc[df['RSI'] > 70, 'Signal'] = -1
        elif strat == "DCA":
            df['Signal'] = 1
            
        # PnL
        df['Ret'] = df['Close'].pct_change()
        df['Strat_Ret'] = df['Ret'] * df['Signal'].shift(1)
        df['Equity'] = capital * (1 + df['Strat_Ret'].fillna(0)).cumprod()
        
        # Metrics
        total_ret = ((df['Equity'].iloc[-1] - capital) / capital) * 100
        dd = (df['Equity'] / df['Equity'].cummax() - 1).min() * 100
        
        st.metric("Total Return", f"{total_ret:.2f}%", f"Drawdown: {dd:.2f}%")
        st.line_chart(df['Equity'])
        with st.expander("Trade Log"):
            st.dataframe(df)

# =============================================================================
# 7. MODULE D: MACRO DESK (agent5 / agentIE)
# =============================================================================
def render_macro():
    st.markdown("### 🌍 Macro Intelligence")
    
    ratios = {
        "Risk On/Off (SPY/TLT)": ("^GSPC", "TLT"),
        "Inflation (GLD/TLT)": ("GC=F", "TLT"),
        "Tech/Value (QQQ/DIA)": ("QQQ", "DIA"),
        "Crypto Dom (BTC/ETH)": ("BTC-USD", "ETH-USD")
    }
    
    # Batch Fetch
    tickers = list(set([item for sublist in ratios.values() for item in sublist]))
    data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    
    cols = st.columns(2)
    for i, (name, (num, den)) in enumerate(ratios.items()):
        if num in data and den in data:
            ratio = data[num] / data[den]
            with cols[i%2]:
                fig = px.line(ratio, title=name, template="plotly_dark")
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 8. MAIN ORCHESTRATOR
# =============================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 💠 TITAN OS")
        
        # Macro Snapshot
        try:
            macro = TitanFeed.get_macro_snapshot()
            c1, c2 = st.columns(2)
            c1.metric("SPX", f"{macro['^GSPC'][0]:.0f}", f"{macro['^GSPC'][1]:.1f}%")
            c2.metric("BTC", f"{macro['BTC-USD'][0]:.0f}", f"{macro['BTC-USD'][1]:.1f}%")
        except: st.caption("Macro Feed Loading...")
        
        st.markdown("---")
        mode = st.radio("MODULE", ["Live Terminal", "Quant Scanner", "Macro Desk", "Strategy Lab"])
        
        st.markdown("---")
        # SECRETS LOADING (Automatic)
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key: api_key = st.text_input("OpenAI Key", type="password")
        
        if "PASSWORD" in st.secrets:
            pwd = st.text_input("System Lock", type="password")
            if pwd != st.secrets["PASSWORD"]:
                st.error("LOCKED")
                st.stop()

    # Routing
    st.markdown('<div class="titan-header"><h1 class="titan-title">TITAN OS</h1><div class="titan-subtitle">FINANCIAL SINGULARITY INTERFACE</div></div>', unsafe_allow_html=True)
    
    if mode == "Live Terminal":
        render_live_terminal(api_key)
    elif mode == "Quant Scanner":
        render_scanner()
    elif mode == "Macro Desk":
        render_macro()
    elif mode == "Strategy Lab":
        render_lab()
        
    st.markdown("---")
    st.markdown("<center style='color:#666'>TITAN ARCHITECT v9.0 | QUANTUM CORE ACTIVE</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
