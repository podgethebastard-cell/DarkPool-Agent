import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import ccxt
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO
from openai import OpenAI
from scipy.stats import linregress
import time

# ==========================================
# 1. SYSTEM CONFIGURATION & DARKPOOL CSS
# ==========================================
st.set_page_config(
    page_title="TITAN ULTRA | AI HEDGE FUND OS",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL STYLING (The "Titan" Aesthetic) ---
st.markdown("""
<style>
    /* TITAN ULTRA PRIME THEME */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=Inter:wght@400;600;800&display=swap');

    /* BASE */
    .stApp { 
        background-color: #050505; 
        color: #e0e0e0; 
        font-family: 'Inter', sans-serif; 
    }
    
    /* TYPOGRAPHY */
    h1, h2, h3, h4, h5 {
        font-family: 'Roboto Mono', monospace;
        color: #fff;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    h1 {
        text-shadow: 0 0 20px rgba(0, 255, 187, 0.5);
    }
    
    /* CUSTOM METRIC CARDS */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(20,20,20,0.95), rgba(10,10,10,0.98));
        border: 1px solid #333;
        border-right: 4px solid #00ffbb;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-right-color: #00b887;
        box-shadow: 0 8px 25px rgba(0, 255, 187, 0.2);
    }
    label[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #888; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 800; color: #fff; }
    
    /* BUTTONS */
    div.stButton > button {
        background: linear-gradient(90deg, #00ffbb, #008f6b);
        color: #050505;
        font-family: 'Roboto Mono', monospace;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 800;
        letter-spacing: 1px;
        border-radius: 4px;
        text-transform: uppercase;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 20px rgba(0, 255, 187, 0.4);
        transform: scale(1.02);
    }
    
    /* INPUTS */
    .stTextInput > div > div > input { background-color: #111; color: #00ffbb; border: 1px solid #333; }
    .stSelectbox > div > div > div { background-color: #111; color: #fff; border: 1px solid #333; }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #020202;
        border-right: 1px solid #222;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #111;
        border-radius: 4px;
        color: #888;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ffbb !important;
        color: #000 !important;
        font-weight: bold;
    }
    
    /* UTILS */
    .titan-badge {
        background: rgba(0,255,187,0.1);
        border: 1px solid #00ffbb;
        color: #00ffbb;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-family: 'Roboto Mono';
        font-weight: bold;
    }
    .titan-risk-high { color: #ff3366; border-color: #ff3366; background: rgba(255, 51, 102, 0.1); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE SYSTEMS (The Engine)
# ==========================================

class TitanMath:
    """The unified mathematical engine for Titan Ultra."""
    
    @staticmethod
    def rma(series, length):
        """Roling Moving Average (Wilder's)"""
        return series.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return TitanMath.rma(tr, length)

    @staticmethod
    def rsi(series, length=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/length, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/length, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def hma(series, length):
        """Hull Moving Average"""
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma_half = series.rolling(half_length).apply(lambda x: np.dot(x, np.arange(1, half_length + 1)) / np.arange(1, half_length + 1).sum(), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length + 1)) / np.arange(1, length + 1).sum(), raw=True)
        diff = 2 * wma_half - wma_full
        return diff.rolling(sqrt_length).apply(lambda x: np.dot(x, np.arange(1, sqrt_length + 1)) / np.arange(1, sqrt_length + 1).sum(), raw=True)

    @staticmethod
    def squeeze_momentum(df):
        """LazyBear's Squeeze Momentum"""
        length = 20
        mult = 2.0
        length_kc = 20
        mult_kc = 1.5
        
        # BB
        m_avg = df['Close'].rolling(window=length).mean()
        m_std = df['Close'].rolling(window=length).std(ddof=0)
        upper_bb = m_avg + mult * m_std
        lower_bb = m_avg - mult * m_std
        
        # KC
        m_avg_kc = df['Close'].rolling(window=length_kc).mean()
        tr = TitanMath.atr(df, length_kc)
        upper_kc = m_avg_kc + mult_kc * tr
        lower_kc = m_avg_kc - mult_kc * tr
        
        # Squeeze
        squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        
        # Momentum
        highest = df['High'].rolling(20).max()
        lowest = df['Low'].rolling(20).min()
        m1 = (highest + lowest) / 2
        val = df['Close'] - (m1 + m_avg_kc)/2
        linreg = val.rolling(20).apply(lambda x: linregress(np.arange(len(x)), x)[0] * len(x) + linregress(np.arange(len(x)), x)[1], raw=True)
        
        return squeeze_on, linreg

    @staticmethod
    def supertrend(df, period=10, multiplier=3):
        hl2 = (df['High'] + df['Low']) / 2
        atr = TitanMath.atr(df, period)
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        # Vectorized SuperTrend calculation involves iteration, keeping simple for now
        return upperband, lowerband

class TitanData:
    """Unified Data Feed for Stock & Crypto"""
    
    @staticmethod
    @st.cache_data(ttl=60)
    def fetch_data(symbol, interval, limit=500, source="Crypto (CCXT)"):
        try:
            if source == "Crypto (CCXT)":
                exchange = ccxt.kraken()
                tf_map = {'15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
                if interval not in tf_map: return pd.DataFrame() # Fallback
                
                ohlcv = exchange.fetch_ohlcv(symbol, tf_map[interval], limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
                
            else: # Yahoo Finance
                # Map YF intervals
                df = yf.download(symbol, period="2y", interval=interval, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception as e:
            st.error(f"Data Feed Error: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_company_info(ticker):
        """For Fundamental Analysis"""
        try:
            stock = yf.Ticker(ticker)
            return stock.info
        except:
            return None

class TitanAI:
    """Interface for LLM Analysis"""
    @staticmethod
    def generate_report(prompt, api_key, model="gpt-4o"):
        if not api_key:
            return "⚠️ OpenAI API Key Missing. Please add it in the sidebar."
        
        client = OpenAI(api_key=api_key)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are CLIVE, a senior institutional quant analyst at a top hedge fund. You provide terse, high-value, data-driven insights. No fluff."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Error: {str(e)}"

# ==========================================
# 3. MODULES (DESKS)
# ==========================================

# --- TITAN PRO LOGIC ENGINE ---
def calculate_titan_pro_logic(df, params):
    """
    Titan Pro specialized logic engine.
    ported from gptagent1.py
    """
    if df.empty: return df
    df = df.copy()
    
    # 1. BASE INDICATORS
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['ATR'] = TitanMath.atr(df, 14)
    
    # 2. TITAN ENGINE (Trend + ATR Trailing Stop)
    amp = params.get('amplitude', 10)
    dev = params.get('channel_dev', 3.0)
    
    df['LL'] = df['Low'].rolling(amp).min()
    df['HH'] = df['High'].rolling(amp).max()
    
    trend = np.zeros(len(df)); stop = np.full(len(df), np.nan)
    curr_t = 0; curr_s = np.nan
    
    # Iterative Trend Logic (Numba would be faster, but keeping pure Python for compatibility)
    start_idx = max(amp, len(df)-2000) 
    
    # Pre-calc arrays for speed
    close_arr = df['Close'].values
    atr_arr = df['ATR'].values
    ll_arr = df['LL'].values
    hh_arr = df['HH'].values
    
    # We need to handle potential NaN values in arrays carefully if not filling
    curr_s = 0 
    
    for i in range(start_idx, len(df)):
        c = close_arr[i]
        d = atr_arr[i] * dev
        if np.isnan(d): continue

        if curr_t == 0: # Bullish
            s = ll_arr[i] + d
            curr_s = max(curr_s, s) if not np.isnan(curr_s) else s
            if c < curr_s: curr_t = 1; curr_s = hh_arr[i] - d
        else: # Bearish
            s = hh_arr[i] - d
            curr_s = min(curr_s, s) if not np.isnan(curr_s) else s
            if c > curr_s: curr_t = 0; curr_s = ll_arr[i] + d
            
        trend[i] = curr_t
        stop[i] = curr_s
        
    df['Titan_Trend'] = trend
    df['Titan_Stop'] = stop
    df['Is_Bull'] = df['Titan_Trend'] == 0
    
    # 3. APEX CLOUD (HMA Structure)
    hma_len = params.get('hma_len', 50)
    # Using simple close for HMA base as per original gptagent1
    base_hma = TitanMath.hma(df['Close'], hma_len)
    apex_atr = df['ATR'] * 1.5
    df['Apex_Upper'] = base_hma + apex_atr
    df['Apex_Lower'] = base_hma - apex_atr
    
    # Apex Trend State
    df['Apex_State'] = np.select(
        [df['Close'] > df['Apex_Upper'], df['Close'] < df['Apex_Lower']],
        [1, -1], default=0
    )
    # Fill zeros with previous valid state logic requires ffill
    df['Apex_State'] = df['Apex_State'].replace(to_replace=0, method='ffill')
    
    # 4. GANN HILO
    gann_len = params.get('gann_len', 3)
    df['Gann_H'] = df['High'].rolling(gann_len).mean()
    df['Gann_L'] = df['Low'].rolling(gann_len).mean()
    # Simplified vectorized Gann
    df['Gann_Trend'] = np.where(df['Close'] > df['Gann_H'].shift(1), 1, np.where(df['Close'] < df['Gann_L'].shift(1), -1, 0))
    
    # 5. SQUEEZE & FLOW
    df['Squeeze_On'], df['Mom_Val'] = TitanMath.squeeze_momentum(df)
    df['RSI'] = TitanMath.rsi(df['Close'])
    # Handle Division by zero if rolling mean is 0
    vol_mean = df['Volume'].rolling(20).mean()
    df['RVOL'] = np.where(vol_mean > 0, df['Volume'] / vol_mean, 0)
    
    # 6. SIGNALS
    # Buy: Bull Trend Start + RVOL + RSI Not Overbought
    cond_buy = (df['Is_Bull']) & (~df['Is_Bull'].shift(1).fillna(False)) & (df['RVOL'] > 1.2) & (df['RSI'] < 75)
    # Sell: Bear Trend Start + RVOL + RSI Not Oversold
    cond_sell = (~df['Is_Bull']) & (df['Is_Bull'].shift(1).fillna(True)) & (df['RVOL'] > 1.2) & (df['RSI'] > 25)
    
    df['Signal'] = np.where(cond_buy, 1, np.where(cond_sell, -1, 0))
    
    # TARGETS
    entry_price = df['Close']
    risk = abs(entry_price - df['Titan_Stop'])
    df['TP1'] = np.where(df['Is_Bull'], entry_price + (risk*1.5), entry_price - (risk*1.5))
    df['TP2'] = np.where(df['Is_Bull'], entry_price + (risk*3.0), entry_price - (risk*3.0))
    
    return df

def generate_gpt_prompt(row, symbol, interval):
    """Generates the text prompt for the AI Analyst"""
    direction = "BULLISH 🐂" if row['Is_Bull'] else "BEARISH 🐻"
    squeeze = "ACTIVE (Explosion Imminent)" if row['Squeeze_On'] else "RELEASED"
    apex = "BULLISH" if row['Apex_State'] == 1 else "BEARISH"
    
    prompt = f"""
    ANALYZE THIS MARKET STATE FOR {symbol} on {interval} timeframe.
    
    METRICS:
    - Current Price: {row['Close']:.2f}
    - Primary Trend (Titan): {direction}
    - Market Structure (Apex Cloud): {apex}
    - Momentum (Squeeze): {squeeze}
    - RSI: {row['RSI']:.1f}
    - Relative Volume: {row['RVOL']:.2f}x average
    - VWAP: {row['VWAP']:.2f} (Price is {"Above" if row['Close'] > row['VWAP'] else "Below"})
    
    TASK:
    Write a short, punchy institutional trading memo.
    1. Verdict: LONG, SHORT, or WAIT?
    2. Confidence Level (High/Med/Low) based on confluence.
    3. Key Levels (Stop Loss at {row['Titan_Stop']:.2f}, TP1 at {row['TP1']:.2f}).
    4. One risks factor.
    
    Format nicely with Markdown. Use emojis. Be professional but aggressive.
    """
    return prompt

def render_titan_pro_terminal(api_key):
    st.markdown("## ⚡ TITAN PRO: INTRADAY TERMINAL")
    
    # --- CONTROLS ---
    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1: 
        symbol = st.text_input("Asset Pair", "BTC/USD").upper()
    with c2: 
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=2)
    with c3:
        source = st.selectbox("Data Source", ["Crypto (CCXT)", "Stocks (YF)"])
    with c4:
        st.caption("Strategy Settings")
        amp = st.slider("Trend Amp", 5, 50, 10, key="tp_amp")
        
    params = {
        'amplitude': amp,
        'channel_dev': 3.0,
        'hma_len': 55,
        'gann_len': 3
    }
    
    if st.button("🚀 INITIATE SCAN", type="primary"):
        with st.spinner(f"Analyzing {symbol}..."):
            df = TitanData.fetch_data(symbol, timeframe, 1000, source)
            
            if not df.empty and len(df) > 100:
                # RUN ENGINE
                df = calculate_titan_pro_logic(df, params)
                last = df.iloc[-1]
                
                # --- HUD ---
                m1, m2, m3, m4 = st.columns(4)
                
                trend_color = "normal" if last['Is_Bull'] else "inverse"
                trend_txt = "BULLISH" if last['Is_Bull'] else "BEARISH"
                
                m1.metric("Price", f"${last['Close']:,.2f}", f"RVOL: {last['RVOL']:.1f}x")
                m2.metric("Titan Trend", trend_txt, f"Stop: {last['Titan_Stop']:,.2f}", delta_color=trend_color)
                m3.metric("Momentum", "SQUEEZE" if last['Squeeze_On'] else "Range", f"RSI: {last['RSI']:.1f}")
                m4.metric("Risk/Reward", f"1:3.0", f"TP2: {last['TP2']:,.2f}")
                
                # --- CHART ---
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
                
                # Candlesticks
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                
                # Titan Stop
                fig.add_trace(go.Scatter(x=df.index, y=df['Titan_Stop'], mode='markers', marker=dict(size=2, color='gray'), name="Trailing Stop"), row=1, col=1)
                
                # Apex Cloud
                fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 255, 187, 0.05)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
                
                # Signals
                buys = df[df['Signal'] == 1]
                sells = df[df['Signal'] == -1]
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', color='#00ffbb', size=12), name="BUY"), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', color='#ff3366', size=12), name="SELL"), row=1, col=1)
                
                # Subplot: Momentum
                colors = ['#00ffbb' if v >= 0 else '#ff3366' for v in df['Mom_Val']]
                fig.add_trace(go.Bar(x=df.index, y=df['Mom_Val'], marker_color=colors, name="Squeeze Mom"), row=2, col=1)
                
                fig.update_layout(height=700, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # --- ACTION CENTER ---
                c_ai, c_tg = st.columns([2, 1])
                
                with c_ai:
                    st.markdown("### 🤖 TITAN ANALYST")
                    if st.button("Generate AI Report"):
                        with st.spinner("Consulting Neural Matrix..."):
                            prompt = generate_gpt_prompt(last, symbol, timeframe)
                            report = TitanAI.generate_report(prompt, api_key)
                            st.markdown(report)
                            
                with c_tg:
                    st.markdown("### 📡 BROADCAST")
                    st.caption("Send Signal to Telegram")
                    tg_msg = f"TITAN SIGNAL: {symbol} is {trend_txt}. Price: {last['Close']:.2f}. Stop: {last['Titan_Stop']:.2f}."
                    if st.button("Broadcast Signal"):
                        st.info("Telegram integration requires Setup in Sidebar.")
                        st.code(tg_msg, language="text")

            else:
                st.error("Data insufficient or not found. Try Crypto (CCXT) for coins or Stocks for equities.")

# --- MACRO INTELLIGENCE RESOURCES ---

MACRO_TICKERS = {
    "✅ MASTER CORE": {
        "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "DXY": "DX-Y.NYB", 
        "US 10Y": "^TNX", "VIX": "^VIX", "Bitcoin": "BTC-USD", "Gold": "GC=F"
    },
    "✅ Global Equity": {
        "World (VT)": "VT", "Emerging (EEM)": "EEM", "Europe (STOXX)": "^STOXX50E", 
        "China (HSI)": "^HSI", "Japan (Nikkei)": "^N225"
    },
    "✅ Rates & Bonds": {
        "US 10Y": "^TNX", "US 2Y": "^IRX", "TLT (20Y+)": "TLT", 
        "HYG (Junk)": "HYG", "LQD (Corp)": "LQD"
    },
    "✅ Commodities": {
        "Oil (WTI)": "CL=F", "Copper": "HG=F", "Natural Gas": "NG=F", 
        "Silver": "SI=F", "Corn": "ZC=F"
    }
}

MACRO_RATIOS = {
    "Risk Appetite": {
        "SPY / TLT": ("SPY", "TLT"), "BTC / SPX": ("BTC-USD", "^GSPC"), 
        "AUD / JPY": ("AUDUSD=X", "JPY=X"), "Discretionary/Staples": ("XLY", "XLP")
    },
    "Inflation/Deflation": {
        "Copper / Gold": ("HG=F", "GC=F"), "TIPS / IEF": ("TIP", "IEF"), 
        "Oil / Gold": ("CL=F", "GC=F")
    },
    "Liquidity": {
        "BTC / DXY": ("BTC-USD", "DX-Y.NYB"), "Gold / DXY": ("GC=F", "DX-Y.NYB")
    }
}

def plot_sparkline(data, color="#00ffbb"):
    if data is None or len(data) < 2: return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=data.values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f"rgba(0, 255, 187, 0.1)" if color=="#00ffbb" else "rgba(255, 51, 102, 0.1)"
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=40, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def render_macro_intelligence(api_key):
    st.markdown("## 🌍 MACRO INTELLIGENCE DESK")
    
    # --- CONTROLS ---
    c1, c2 = st.columns([1,3])
    with c1:
        mode = st.radio("View Mode", ["Standard Tickers", "Deep Ratios"])
        category = st.selectbox("Category", list(MACRO_TICKERS.keys()) if mode == "Standard Tickers" else list(MACRO_RATIOS.keys()))
    
    with c2:
        if st.button("🔄 REFRESH MACRO DATA", type="primary"):
            st.rerun()

    st.markdown("---")
    
    # --- DATA FETCHING ---
    if mode == "Standard Tickers":
        items = MACRO_TICKERS[category]
        tickers = list(items.values())
        data = yf.download(tickers, period="3mo", interval="1d", progress=False)['Close']
        
        # Grid Display
        cols = st.columns(3)
        for i, (label, ticker) in enumerate(items.items()):
            if ticker not in data.columns: continue
            series = data[ticker].dropna()
            if series.empty: continue
            
            last = series.iloc[-1]
            prev = series.iloc[-2]
            chg = ((last - prev) / prev) * 100
            
            col = cols[i % 3]
            with col:
                with st.container():
                    st.markdown(f"**{label}**")
                    st.metric("Level", f"{last:,.2f}", f"{chg:+.2f}%")
                    spark = plot_sparkline(series, "#00ffbb" if chg >= 0 else "#ff3366")
                    st.plotly_chart(spark, use_container_width=True, config={'displayModeBar': False})
                    st.markdown("---")

    else: # Ratios
        items = MACRO_RATIOS[category]
        tickers = []
        for n, d in items.items():
            tickers.extend([d[0], d[1]])
        data = yf.download(list(set(tickers)), period="6mo", interval="1d", progress=False)['Close']
        
        cols = st.columns(3)
        ai_summary = []
        
        for i, (label, (num, den)) in enumerate(items.items()):
            if num not in data.columns or den not in data.columns: continue
            
            s1 = data[num]
            s2 = data[den]
            ratio = s1 / s2
            ratio = ratio.dropna()
            
            if ratio.empty: continue
            
            last = ratio.iloc[-1]
            prev = ratio.iloc[-2]
            chg = ((last - prev) / prev) * 100
            
            col = cols[i % 3]
            with col:
                st.markdown(f"**{label}**")
                st.metric("Ratio", f"{last:.4f}", f"{chg:+.2f}%")
                spark = plot_sparkline(ratio, "#00ffbb" if chg >= 0 else "#ff3366")
                st.plotly_chart(spark, use_container_width=True, config={'displayModeBar': False})
                st.markdown("---")
                
            ai_summary.append(f"{label}: {last:.4f} (Change: {chg:+.2f}%)")
            
        # --- AI SYNTHESIS ---
        if st.button("🧠 GENERATE MACRO SYNTHESIS"):
            prompt = f"Analyze these macro ratios for the '{category}' category and give a market regime assessment (Risk On/Off, Inflation/Deflation):\n" + "\n".join(ai_summary)
            report = TitanAI.generate_report(prompt, api_key)
            st.info("### 🤖 MACRO ANALYST VERDICT")
            st.markdown(report)

# --- CLIVE FUNDAMENTAL ENGINE ---

CLIVE_UNIVERSE = [
    # US Tech/Growth
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "CRWD", "PLTR",
    # Global Titans
    "ASML", "TSM", "NVO", "LVMUY", "SHEL",
    # Finance/Defensive
    "JPM", "V", "JNJ", "PG", "COST"
]

def calculate_clive_fundamentals(ticker_symbol):
    """Fetches deep fundamental data for screening."""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Market Cap Filter (> 2B)
        mkt_cap = info.get('marketCap', 0)
        if mkt_cap < 2e9: return None
        
        # Data Extraction
        data = {
            "Symbol": ticker_symbol,
            "Name": info.get('shortName', ticker_symbol),
            "Sector": info.get('sector', 'Unknown'),
            "P/E": info.get('forwardPE', 999),
            "PEG": info.get('pegRatio', 999),
            "P/S": info.get('priceToSalesTrailing12Months', 999),
            "Rev Growth": info.get('revenueGrowth', 0),
            "Margins": info.get('profitMargins', 0),
            "Debt/Eq": info.get('debtToEquity', 999),
            "FCF Yield": 0.0 # Calc below
        }
        
        # FCF Calculation
        try:
            fcf = info.get('freeCashflow')
            if fcf and mkt_cap > 0:
                data["FCF Yield"] = (fcf / mkt_cap) * 100
        except: pass
        
        # Scoring Logic (Simple Quality/Growth/Value)
        score = 0
        if data['P/E'] < 25: score += 1
        if data['Rev Growth'] > 0.15: score += 2
        if data['Margins'] > 0.20: score += 1
        if data['Debt/Eq'] < 50: score += 1
        if data['FCF Yield'] > 3.0: score += 1
        
        data['Clive Score'] = score
        return data
    except:
        return None

def analyze_with_clive_ai(row, api_key):
    if not api_key: return "⚠️ Please enter OpenAI API Key."
    
    prompt = f"""
    You are CLIVE, a Senior Institutional Analyst. Write a 3-sentence Buy-Side Memo for:
    {row['Name']} ({row['Symbol']})
    
    DATA:
    - P/E: {row['P/E']} | PEG: {row['PEG']}
    - Rev Growth: {row['Rev Growth']*100:.1f}%
    - Margins: {row['Margins']*100:.1f}%
    - Clive Score: {row['Clive Score']}/6
    
    Format:
    **THESIS:** [Bull/Bear case]
    **VALUATION:** [Comment on multiples]
    **VERDICT:** [BUY/HOLD/SELL]
    """
    return TitanAI.generate_report(prompt, api_key)

def render_clive_analyst(api_key):
    st.markdown("## 👔 CLIVE: INSTITUTIONAL ANALYST")
    
    st.markdown("""
    **Mission:** Identify high-quality compounders with reasonable valuations using the *Titan Fundamental Score*.
    """)
    
    if st.button("🔎 RUN DEEP SCREEN (UNIVERSE: 20 GLOBAL TITANS)"):
        results = []
        bar = st.progress(0)
        status = st.empty()
        
        for i, tic in enumerate(CLIVE_UNIVERSE):
            status.caption(f"Analyzing balance sheet of {tic}...")
            bar.progress((i+1)/len(CLIVE_UNIVERSE))
            
            fund_data = calculate_clive_fundamentals(tic)
            if fund_data:
                results.append(fund_data)
        
        bar.empty()
        status.empty()
        
        if results:
            df = pd.DataFrame(results).sort_values("Clive Score", ascending=False)
            st.session_state['clive_results'] = df
    
    # Display Results
    if 'clive_results' in st.session_state:
        df = st.session_state['clive_results']
        
        # Metrics
        top_pick = df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Top Pick", top_pick['Symbol'], f"Score: {top_pick['Clive Score']}")
        c2.metric("Avg P/E", f"{df['P/E'].mean():.1f}")
        c3.metric("Avg Growth", f"{df['Rev Growth'].mean()*100:.1f}%")
        
        st.dataframe(
            df.style.background_gradient(subset=['Clive Score'], cmap='Greens')
              .format({'Rev Growth': '{:.1%}', 'Margins': '{:.1%}', 'FCF Yield': '{:.2f}%', 'P/E': '{:.1f}'}),
            use_container_width=True
        )
        
        # AI Deep Dive
        st.markdown("### 🧠 AI INVESTMENT MEMO")
        selected_ticker = st.selectbox("Select Ticker for Memo", df['Symbol'].tolist())
        
        if st.button("GENERATE MEMO"):
            row = df[df['Symbol'] == selected_ticker].iloc[0]
            with st.spinner(f"Drafting memo for {selected_ticker}..."):
                memo = analyze_with_clive_ai(row, api_key)
                st.info(memo)

# --- STRATEGY LAB ENGINE ---

class BaseStrategy:
    def generate_signal(self, history): raise NotImplementedError

class MA_Cross_Strategy(BaseStrategy):
    def __init__(self, slow, fast): self.slow = slow; self.fast = fast
    def generate_signal(self, df):
        if len(df) < self.slow: return 'HOLD'
        fast_ma = df['Close'].rolling(self.fast).mean()
        slow_ma = df['Close'].rolling(self.slow).mean()
        curr_fast, curr_slow = fast_ma.iloc[-1], slow_ma.iloc[-1]
        prev_fast, prev_slow = fast_ma.iloc[-2], slow_ma.iloc[-2]
        if prev_fast <= prev_slow and curr_fast > curr_slow: return 'BUY'
        if prev_fast >= prev_slow and curr_fast < curr_slow: return 'SELL'
        return 'HOLD'

class RSI_Strategy(BaseStrategy):
    def __init__(self, period=14, buy=30, sell=70): self.p = period; self.buy = buy; self.sell = sell
    def generate_signal(self, df):
        if len(df) < self.p+1: return 'HOLD'
        rsi = TitanMath.rsi(df['Close'], self.p)
        curr, prev = rsi.iloc[-1], rsi.iloc[-2]
        if prev <= self.buy and curr > self.buy: return 'BUY'
        if prev >= self.sell and curr < self.sell: return 'SELL'
        return 'HOLD'

class MACD_Strategy(BaseStrategy):
    def __init__(self, fast=12, slow=26, sig=9):
        self.fast = fast; self.slow = slow; self.sig = sig
    def generate_signal(self, df):
        if len(df) < self.slow + self.sig: return 'HOLD'
        ema_f = df['Close'].ewm(span=self.fast, adjust=False).mean()
        ema_s = df['Close'].ewm(span=self.slow, adjust=False).mean()
        macd = ema_f - ema_s
        signal = macd.ewm(span=self.sig, adjust=False).mean()
        curr_m, curr_s = macd.iloc[-1], signal.iloc[-1]
        prev_m, prev_s = macd.iloc[-2], signal.iloc[-2]
        if prev_m <= prev_s and curr_m > curr_s: return 'BUY'
        if prev_m >= prev_s and curr_m < curr_s: return 'SELL'
        return 'HOLD'

class TitanBacktester:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.equity = []
        self.trades = []
        
    def run(self, df, strategy):
        self.equity = []
        self.trades = []
        self.position = 0
        self.cash = self.capital
        self.idx = df.index
        
        # We start loop where strategy has enough data
        start_i = 50 
        
        # Vectorized signal generation for speed would be better, but strategy pattern requires loop
        # We will iterate through the passed DataFrame
        # Optimization: Pre-calculate indicators if possible, but strategy logic is dynamic
        
        # Simple Loop
        for i in range(start_i, len(df)):
            subset = df.iloc[:i+1]
            signal = strategy.generate_signal(subset)
            price = df['Close'].iloc[i]
            date = self.idx[i]
            
            if signal == 'BUY' and self.position == 0:
                qty = self.cash / price
                self.position = qty
                self.cash = 0
                self.trades.append({'Date': date, 'Type': 'BUY', 'Price': price, 'Qty': qty})
                
            elif signal == 'SELL' and self.position > 0:
                self.cash = self.position * price
                self.trades.append({'Date': date, 'Type': 'SELL', 'Price': price, 'Qty': self.position})
                self.position = 0
                
            curr_val = self.cash + (self.position * price)
            self.equity.append(curr_val)
            
        return pd.DataFrame(self.trades), self.equity

def render_strategy_lab():
    st.markdown("## 🧪 STRATEGY LAB")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        ticker = st.text_input("Ticker", "BTC/USD", key="sl_tic").upper()
        source = st.selectbox("Source", ["Crypto (CCXT)", "Stocks (YF)"], key='sl_src')
    with c2:
        strat_name = st.selectbox("Strategy Model", ["MA Cross", "RSI Mean Reversion", "MACD Momentum"])
        capital = st.number_input("Capital ($)", 1000, 1000000, 10000)
    with c3:
        if strat_name == "MA Cross":
            p1 = st.number_input("Fast MA", 5, 50, 9)
            p2 = st.number_input("Slow MA", 10, 200, 21)
            strategy = MA_Cross_Strategy(p2, p1)
        elif strat_name == "RSI Mean Reversion":
            p1 = st.number_input("RSI Period", 2, 30, 14)
            p2 = st.slider("Oversold/Bought", 10, 90, (30,70))
            strategy = RSI_Strategy(p1, p2[0], p2[1])
        elif strat_name == "MACD Momentum":
            strategy = MACD_Strategy()
            st.caption("Standard (12, 26, 9)")

    if st.button("RUN SIMULATION", type="primary"):
        with st.spinner("Simulating..."):
            df = TitanData.fetch_data(ticker, "1d", 500, source)
            
            if not df.empty and len(df) > 50:
                bt = TitanBacktester(capital)
                trades, equity = bt.run(df, strategy)
                
                if len(equity) > 0:
                    ret = ((equity[-1] - capital) / capital) * 100
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Return", f"{ret:.2f}%")
                    m2.metric("Final Equity", f"${equity[-1]:,.2f}")
                    m3.metric("Trade Count", len(trades))
                    
                    st.line_chart(equity)
                    
                    if not trades.empty:
                        st.markdown("### 📜 Trade Log")
                        st.dataframe(trades.style.format({'Price': '{:.2f}', 'Qty': '{:.4f}'}), use_container_width=True)
                else:
                    st.warning("No trades executed.")
            else:
                st.error("Insufficient Data.")

# ==========================================
# 4. MAIN NAVIGATION
# ==========================================

def main():
    # --- SIDEBAR HEADER ---
    st.sidebar.markdown("# 💠 TITAN ULTRA")
    st.sidebar.caption("v1.0.0 | System Online")
    
    # --- API KEYS ---
    with st.sidebar.expander("🔐 SYSTEM KEYS"):
        openai_key = st.text_input("OpenAI API Key", type="password")
        
    # --- NAVIGATION ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ WORKSTATION")
    
    desk = st.sidebar.radio(
        "Select Desk:",
        ["Titan Pro Terminal", "Macro Intelligence", "Clive Analyst", "Strategy Lab"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("System Status: **NOMINAL**")
    
    # --- ROUTING ---
    if desk == "Titan Pro Terminal":
        render_titan_pro_terminal(openai_key)
    elif desk == "Macro Intelligence":
        render_macro_intelligence(openai_key)
    elif desk == "Clive Analyst":
        render_clive_analyst(openai_key)
    elif desk == "Strategy Lab":
        render_strategy_lab()

if __name__ == "__main__":
    main()
