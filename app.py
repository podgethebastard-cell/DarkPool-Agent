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
        st.session_state.api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key here to unlock the AI Analyst features."
        )

# ==========================================
# 2. DATA ENGINE (PURE MATH & DATA)
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
def get_global_performance():
    """Fetches performance of a Global Multi-Asset Basket."""
    # UPGRADE: Expanded list to include Crypto, Commodities, and Bonds
    assets = {
        "Tech (XLK)": "XLK", 
        "Energy (XLE)": "XLE", 
        "Financials (XLF)": "XLF", 
        "Bitcoin (BTC)": "BTC-USD", 
        "Gold (GLD)": "GLD",
        "Oil (USO)": "USO",
        "Treasuries (TLT)": "TLT"
    }
    try:
        results = {}
        for name, ticker in assets.items():
            df = yf.download(ticker, period="5d", interval="1d", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    price = df.xs('Close', axis=1, level=0).iloc[-1].iloc[0]
                    prev = df.xs('Close', axis=1, level=0).iloc[-2].iloc[0]
                else:
                    price = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                
                change = ((price - prev) / prev) * 100
                results[name] = change
        
        return pd.Series(results).sort_values(ascending=True) # Sorted for the chart
    except: return None

def safe_download(ticker, period, interval):
    """Robust price downloader."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: return None
        
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
            else: return None
            
        return df
    except: return None

@st.cache_data(ttl=300)
def get_macro_data():
    """Fetches key macro indicators."""
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
    df['HMA'] = df['Close'].rolling(55).mean()
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    df['Pivot_Resist'] = df['High'].rolling(20).max()
    df['Pivot_Support'] = df['Low'].rolling(20).min()
    
    df['MFI'] = (df['Close'].diff() * df['Volume']).rolling(3).mean()
    
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['KC_ATR'] = df['ATR'].rolling(20).mean()
    df['Squeeze_On'] = (df['BB_Mid'] + 2*df['BB_Std']) < (df['BB_Mid'] + 1.5*df['KC_ATR'])
    df['Mom'] = df['Close'] - df['Close'].rolling(20).mean()
    
    return df

# ==========================================
# 4. AI ANALYST
# ==========================================
def ask_ai_analyst(df, ticker, fundamentals, balance, risk_pct):
    if not st.session_state.api_key: 
        return "⚠️ Waiting for OpenAI API Key in the sidebar..."
    
    last = df.iloc[-1]
    trend = "BULLISH" if last['Close'] > last['HMA'] else "BEARISH"
    risk_dollars = balance * (risk_pct / 100)
    
    if trend == "BULLISH":
        stop_level = last['Pivot_Support']
        direction = "LONG"
    else:
        stop_level = last['Pivot_Resist']
        direction = "SHORT"
        
    if pd.isna(stop_level) or abs(last['Close'] - stop_level) < (last['ATR']*0.5):
        stop_level = last['Close'] - (last['ATR']*2) if direction == "LONG" else last['Close'] + (last['ATR']*2)
        
    dist = abs(last['Close'] - stop_level)
    if dist == 0: dist = last['ATR']
    shares = risk_dollars / dist 
    
    fund_text = "N/A"
    if fundamentals:
        fund_text = f"P/E: {fundamentals.get('P/E Ratio', 'N/A')}. Growth: {fundamentals.get('Rev Growth', 0)*100:.1f}%."
    
    prompt = f"""
    Act as a Global Macro Strategist. Analyze {ticker} at ${last['Close']:.2f}.
    --- FUNDAMENTALS ---
    {fund_text}
    --- TECHNICALS ---
    Trend: {trend}. Money Flow: {last['MFI']:.0f}. Volatility (ATR): {last['ATR']:.2f}.
    --- RISK PROTOCOL (1% Rule) ---
    Capital: ${balance}. Risk Budget: ${risk_dollars:.2f} ({risk_pct}%).
    Stop Loss: ${stop_level:.2f}. Position Size: {shares:.4f} units.
    --- MISSION ---
    1. Verdict: BUY, SELL, or WAIT.
    2. Reasoning: Market Structure, Trend, Fundamentals.
    3. Trade Plan: Entry, Stop, Target (2.5R), Size.
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

# --- INPUT MODE SELECTION ---
input_mode = st.sidebar.radio(
    "Input Mode:", 
    ["Curated Lists", "Manual Search (Global)"],
    index=1,
    help="Choose 'Curated Lists' to select from preset menus, or 'Manual Search' to type any ticker symbol yourself."
)

if input_mode == "Curated Lists":
    assets = {
        "Indices": ["SPY", "QQQ", "IWM", "^VIX"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
        "Tech": ["NVDA", "TSLA", "AAPL", "MSFT"],
        "Macro": ["^TNX", "DX-Y.NYB", "TLT"],
        "LSE/Commodities": ["SSLN.L", "SGLN.L", "SHEL.L", "BP.L"] 
    }
    cat = st.sidebar.selectbox(
        "Asset Class", 
        list(assets.keys()),
        help="Select a category of assets to filter the ticker list."
    )
    ticker = st.sidebar.selectbox(
        "Ticker", 
        assets[cat],
        help="Choose a specific asset from the selected category."
    )
else:
    # --- SEARCH BOX FEATURE ---
    st.sidebar.info("Type any ticker (e.g. SSLN.L, BTC-USD)")
    ticker = st.sidebar.text_input(
        "Search Ticker Symbol", 
        value="SSLN.L",
        help="Type any valid Yahoo Finance ticker here. Works for Stocks, Crypto, Indices, and Forex."
    ).upper()

interval = st.sidebar.selectbox(
    "Interval", 
    ["15m", "1h", "4h", "1d", "1wk"], 
    index=3,
    help="Select the timeframe for the chart bars (e.g., 1 Day, 1 Hour)."
)
st.sidebar.markdown("---")

balance = st.sidebar.number_input(
    "Capital ($)", 
    1000, 1000000, 10000,
    help="Enter your total trading capital for position sizing calculations."
)
risk_pct = st.sidebar.slider(
    "Risk %", 
    0.5, 3.0, 1.0,
    help="Adjust the percentage of capital you are willing to risk on this trade."
)

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

if st.button(f"Analyze {ticker}", help="Click to run the data pipeline and AI analysis for the selected ticker."):
    st.session_state['run_analysis'] = True

if st.session_state.get('run_analysis'):
    with st.spinner(f"Analyzing {ticker}..."):
        df = safe_download(ticker, "2y", interval)
        
        if df is not None:
            df = calc_indicators(df)
            fund = get_fundamentals(ticker)
            
            with tab1:
                st.subheader(f"🎯 Sniper Scope: {ticker}")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='orange', width=2), name="Apex Trend"), row=1, col=1)
                
                if not pd.isna(df['Pivot_Resist'].iloc[-1]):
                    fig.add_hline(y=df['Pivot_Resist'].iloc[-1], line_dash="dash", line_color="red", row=1, col=1)
                if not pd.isna(df['Pivot_Support'].iloc[-1]):
                    fig.add_hline(y=df['Pivot_Support'].iloc[-1], line_dash="dash", line_color="green", row=1, col=1)
                
                colors = ['#00ff00' if v > 0 else '#ff0000' for v in df['MFI']]
                fig.add_trace(go.Bar(x=df.index, y=df['MFI'], marker_color=colors, name="Smart Money"), row=2, col=1)
                
                fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 🤖 Strategy Briefing")
                verdict = ask_ai_analyst(df, ticker, fund, balance, risk_pct)
                st.info(verdict)

            with tab2:
                st.subheader(f"🏢 Fundamental Health")
                if fund:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("P/E Ratio", f"{fund.get('P/E Ratio', 'N/A')}")
                    c2.metric("Rev Growth", f"{fund.get('Rev Growth', 0)*100:.1f}%")
                    c3.metric("Debt/Equity", f"{fund.get('Debt/Equity', 'N/A')}")
                    st.write(f"**Summary:** {fund.get('Summary', 'No Data')[:300]}...")
                else:
                    st.warning("Fundamentals not available for this asset.")
                
                st.markdown("---")
                # UPGRADE: Replaced Dataframe with Plotly Heatmap
                st.subheader("🔥 Global Market Heatmap")
                s_data = get_global_performance()
                if s_data is not None:
                    # Create a horizontal bar chart
                    fig_sector = go.Figure()
                    
                    # Color logic: Green for positive, Red for negative
                    colors = ['#00ff00' if v >= 0 else '#ff0000' for v in s_data.values]
                    
                    fig_sector.add_trace(go.Bar(
                        x=s_data.values,
                        y=s_data.index,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{v:.2f}%" for v in s_data.values],
                        textposition='auto'
                    ))
                    
                    fig_sector.update_layout(
                        height=400, 
                        template="plotly_dark", 
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="5-Day Performance (%)"
                    )
                    st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.error("Data connection failed. Try another ticker.")
