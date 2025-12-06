import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="DarkPool Titan Terminal")
st.title("👁️ DarkPool Titan Terminal")
st.markdown("### Institutional-Grade Market Intelligence")

# --- API Key Management ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if "OPENAI_API_KEY" in st.secrets:
    st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
else:
    if not st.session_state.api_key:
        st.session_state.api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# ==========================================
# 2. DATA ENGINE (ROBUST & CACHED)
# ==========================================
@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    """Fetches key financial metrics safely."""
    if "-" in ticker or "=" in ticker or "^" in ticker: 
        return None 
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info: return None
        
        return {
            "Market Cap": info.get("marketCap", 0),
            "P/E Ratio": info.get("trailingPE", 0),
            "Rev Growth": info.get("revenueGrowth", 0),
            "Debt/Equity": info.get("debtToEquity", 0),
            "Summary": info.get("longBusinessSummary", "No Data Available")
        }
    except: return None

@st.cache_data(ttl=300)
def get_sector_data():
    """Fetches performance of key US Sectors."""
    sectors = {
        "Tech": "XLK", "Energy": "XLE", "Financials": "XLF", 
        "Healthcare": "XLV", "Consumer": "XLY"
    }
    try:
        # Download individually to avoid MultiIndex errors
        results = {}
        for name, ticker in sectors.items():
            df = yf.download(ticker, period="5d", interval="1d", progress=False)
            if not df.empty:
                # Handle MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    price = df.xs('Close', axis=1, level=0).iloc[-1].iloc[0]
                    prev = df.xs('Close', axis=1, level=0).iloc[-2].iloc[0]
                else:
                    price = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                
                change = ((price - prev) / prev) * 100
                results[name] = change
        
        return pd.Series(results).sort_values(ascending=False)
    except: return None

def get_news(ticker):
    """Fetches latest news headlines safely."""
    try:
        if "=" in ticker or "^" in ticker: return []
        stock = yf.Ticker(ticker)
        news_items = stock.news
        # Return empty list if news is None or empty
        if not news_items: return []
        return news_items[:5]
    except: return []

def safe_download(ticker, period, interval):
    """Robust price downloader."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # FIX: Flatten MultiIndex columns if they exist (Common yfinance issue)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: return None
        
        # Ensure 'Close' exists
        if 'Close' not in df.columns:
            # Fallback for weird column names
            if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
            else: return None
            
        return df
    except: return None

@st.cache_data(ttl=300)
def get_macro_data():
    """Fetches key macro indicators individually to prevent NaNs."""
    tickers = {
        "S&P 500": "SPY", "Bitcoin": "BTC-USD", 
        "10Y Yield": "^TNX", "VIX": "^VIX"
    }
    prices = {}
    changes = {}
    
    for name, sym in tickers.items():
        try:
            df = yf.download(sym, period="5d", interval="1d", progress=False)
            if not df.empty:
                # Flatten columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                curr = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                chg = ((curr - prev) / prev) * 100
                
                prices[name] = curr
                changes[name] = chg
        except:
            prices[name] = 0.0
            changes[name] = 0.0
            
    return prices, changes

# ==========================================
# 3. MATH LIBRARY
# ==========================================
def calc_indicators(df):
    # 1. Apex Trend (SMA Proxy)
    df['HMA'] = df['Close'].rolling(55).mean()
    
    # 2. ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # 3. Support/Resistance
    df['Pivot_Resist'] = df['High'].rolling(20).max()
    df['Pivot_Support'] = df['Low'].rolling(20).min()
    
    # 4. Money Flow
    df['MFI'] = (df['Close'].diff() * df['Volume']).rolling(3).mean()
    
    # 5. Squeeze
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['KC_ATR'] = df['ATR'].rolling(20).mean()
    df['Squeeze_On'] = (df['BB_Mid'] + 2*df['BB_Std']) < (df['BB_Mid'] + 1.5*df['KC_ATR'])
    df['Mom'] = df['Close'] - df['Close'].rolling(20).mean()
    
    return df

# ==========================================
# 4. AI ANALYST (RISK MANAGER)
# ==========================================
def ask_ai_analyst(df, ticker, fundamentals, news_list, balance, risk_pct):
    if not st.session_state.api_key: 
        return "⚠️ Waiting for OpenAI API Key in the sidebar..."
    
    last = df.iloc[-1]
    
    # --- CRITICAL FIX FOR KEYERROR ---
    # We use .get('title', 'No Title') to ensure it never crashes if title is missing
    if news_list and isinstance(news_list, list):
        news_text = "\n".join([f"- {n.get('title', 'No Title Available')}" for n in news_list if isinstance(n, dict)])
    else:
        news_text = "No recent news found."
    
    # Technical States
    trend = "BULLISH" if last['Close'] > last['HMA'] else "BEARISH"
    
    # Risk Calculations
    risk_dollars = balance * (risk_pct / 100)
    
    if trend == "BULLISH":
        stop_level = last['Pivot_Support']
        direction = "LONG"
    else:
        stop_level = last['Pivot_Resist']
        direction = "SHORT"
        
    # Safety Check
    if pd.isna(stop_level) or abs(last['Close'] - stop_level) < (last['ATR']*0.5):
        stop_level = last['Close'] - (last['ATR']*2) if direction == "LONG" else last['Close'] + (last['ATR']*2)
        
    dist = abs(last['Close'] - stop_level)
    if dist == 0: dist = last['ATR']
    shares = risk_dollars / dist 
    target = last['Close'] + (dist * 2.5) if direction == "LONG" else last['Close'] - (dist * 2.5)
    
    fund_text = "N/A"
    if fundamentals:
        fund_text = f"P/E: {fundamentals.get('P/E Ratio', 'N/A')}. Growth: {fundamentals.get('Rev Growth', 0)*100:.1f}%."
    
    prompt = f"""
    Act as a Global Macro Strategist. Analyze {ticker} at ${last['Close']:.2f}.
    
    --- FUNDAMENTALS ---
    {fund_text}
    
    --- TECHNICALS ---
    Trend: {trend}. Money Flow: {last['MFI']:.0f}.
    
    --- NEWS ---
    {news_text}
    
    --- RISK PROTOCOL (1% Rule) ---
    Capital: ${balance}. Risk Budget: ${risk_dollars:.2f} ({risk_pct}%).
    Stop Loss: ${stop_level:.2f}. Position Size: {shares:.4f} units.
    
    --- MISSION ---
    1. **Verdict:** BUY, SELL, or WAIT.
    2. **Reasoning:** Combine Charts + News.
    3. **Trade Plan:** Entry, Stop, Target (2.5R), Size.
    """
    
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# ==========================================
# 5. UI DASHBOARD LAYOUT
# ==========================================
st.sidebar.header("🎛️ Terminal Controls")

input_mode = st.sidebar.radio("Input Mode:", ["Curated Lists", "Manual Search (Global)"])

if input_mode == "Curated Lists":
    assets = {
        "Indices": ["SPY", "QQQ", "IWM", "^VIX"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
        "Tech": ["NVDA", "TSLA", "AAPL", "MSFT"],
        "Macro": ["^TNX", "DX-Y.NYB", "TLT"]
    }
    cat = st.sidebar.selectbox("Asset Class", list(assets.keys()))
    ticker = st.sidebar.selectbox("Ticker", assets[cat])
else:
    st.sidebar.info("Type ticker (e.g. SHEL.L, 7203.T)")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()

interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=3)

st.sidebar.markdown("---")
st.sidebar.header("💰 Risk Parameters")
balance = st.sidebar.number_input("Capital ($)", 1000, 1000000, 10000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0)

# --- GLOBAL MACRO HEADER ---
m_price, m_chg = get_macro_data()
if m_price:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S&P 500", f"{m_price['S&P 500']:.2f}", f"{m_chg['S&P 500']:.2f}%")
    c2.metric("Bitcoin", f"{m_price['Bitcoin']:.2f}", f"{m_chg['Bitcoin']:.2f}%")
    c3.metric("10Y Yield", f"{m_price['10Y Yield']:.2f}", f"{m_chg['10Y Yield']:.2f}%")
    c4.metric("VIX", f"{m_price['VIX']:.2f}", f"{m_chg['VIX']:.2f}%")
    st.markdown("---")

# --- MAIN ANALYSIS TABS ---
tab1, tab2 = st.tabs(["📊 Technical Deep Dive", "🌍 Sector & Fundamentals"])

# SHARED TRIGGER
if st.button(f"Analyze {ticker}"):
    st.session_state['run_analysis'] = True

if st.session_state.get('run_analysis'):
    with st.spinner(f"Connecting to Global Exchanges for {ticker}..."):
        df = safe_download(ticker, "2y", interval)
        
        if df is not None:
            df = calc_indicators(df)
            fund = get_fundamentals(ticker)
            news = get_news(ticker)
            
            # --- TAB 1: TECHNICALS ---
            with tab1:
                st.subheader(f"🎯 Sniper Scope: {ticker}")
                
                # Charting
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='orange', width=2), name="Apex Trend"), row
