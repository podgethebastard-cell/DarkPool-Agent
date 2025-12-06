import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="DarkPool Ultimate Architect")
st.title("👁️ DarkPool Ultimate Architect")

# Load API Key
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# ==========================================
# 2. MATH & DATA ENGINE
# ==========================================
@st.cache_data(ttl=300) # Cache data for 5 mins to speed up the app
def get_global_data():
    # The "Big Board" Ticker List
    tickers = {
        "Indices": ["SPY", "QQQ", "IWM", "DIA", "^VIX"],
        "Global": ["EEM", "VGK", "FXI", "EWJ"], # Emerging, Europe, China, Japan
        "Rates & Fx": ["^TNX", "DX-Y.NYB", "UUP", "TLT"],
        "Commodities": ["GC=F", "SI=F", "CL=F", "HG=F"], # Gold, Silver, Oil, Copper
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"]
    }
    
    flat_list = [item for sublist in tickers.values() for item in sublist]
    
    try:
        data = yf.download(flat_list, period="1mo", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close']
        else:
            df = data['Close']
        return df
    except: return None

def calc_indicators(df):
    # Calculate Apex, Vector, etc (The "Sniper" Logic)
    df['HMA'] = df['Close'].rolling(55).mean() # Simplified HMA for speed
    df['ATR'] = df['High'] - df['Low'] # Simplified TR
    
    # Vector Scalper Logic (Simplified for Python Speed)
    df['Vector_Stop'] = df['Low'].rolling(5).min() # Basic trailing support
    
    # Money Flow
    df['MFI'] = (df['Close'].diff() * df['Volume']).rolling(3).mean()
    
    return df

# ==========================================
# 3. AI PERSONAS
# ==========================================
def ask_cio(df_change):
    if not api_key: return "⚠️ API Key Missing."
    
    # Prepare a summary string of performance
    summary = df_change.to_string()
    
    prompt = f"""
    Act as a Chief Investment Officer (CIO) for a Macro Hedge Fund.
    Here is the Daily Performance (%) of key global assets:
    {summary}
    
    YOUR MISSION:
    1. **Regime Identification:** Are we Risk-On (Tech/Crypto up, Dollar down) or Risk-Off (Gold/Dollar up)?
    2. **Anomalies:** Is Copper dropping while Stocks rise? (Bad signal). Is Yield (^TNX) spiking?
    3. **Strategy:** Where should capital flow today? (e.g., "Rotate into Commodities" or "Cash is King").
    4. **Warning:** One sentence on the biggest danger in the market right now.
    """
    
    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
    return res.choices[0].message.content

def ask_sniper(ticker, price, stop, atr, balance):
    if not api_key: return "⚠️ API Key Missing."
    
    risk_amt = balance * 0.02
    shares = risk_amt / (price - stop) if price > stop else 0
    
    prompt = f"""
    Act as a Senior Execution Trader.
    Asset: {ticker}. Price: {price}. Stop Loss Level: {stop}. ATR: {atr}.
    Risk Budget: ${risk_amt} (2% of ${balance}).
    
    TASK:
    1. **Signal:** BUY, SELL, or WAIT.
    2. **The Math:** Confirm position size is {shares:.4f} units.
    3. **Targets:** Set TP1 at {price + (2*atr):.2f} and TP2 at {price + (4*atr):.2f}.
    """
    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
    return res.choices[0].message.content

# ==========================================
# 4. MAIN INTERFACE (TABS)
# ==========================================
tab1, tab2 = st.tabs(["🌍 Global Macro War Room", "🎯 Sniper Scope (Charts)"])

# --- TAB 1: THE MACRO VIEW ---
with tab1:
    st.subheader("Global Market Pulse")
    df_macro = get_global_data()
    
    if df_macro is not None:
        # Calculate % Changes
        daily_change = df_macro.pct_change().iloc[-1] * 100
        
        # 1. METRIC ROW
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("S&P 500", f"{df_macro['SPY'].iloc[-1]:.2f}", f"{daily_change['SPY']:.2f}%")
        c2.metric("Bitcoin", f"{df_macro['BTC-USD'].iloc[-1]:.2f}", f"{daily_change['BTC-USD']:.2f}%")
        c3.metric("10Y Yield", f"{df_macro['^TNX'].iloc[-1]:.2f}", f"{daily_change['^TNX']:.2f}%")
        c4.metric("Gold", f"{df_macro['GC=F'].iloc[-1]:.2f}", f"{daily_change['GC=F']:.2f}%")
        c5.metric("VIX (Fear)", f"{df_macro['^VIX'].iloc[-1]:.2f}", f"{daily_change['^VIX']:.2f}%")
        
        st.markdown("---")
        
        # 2. CORRELATION MATRIX & CIO BRIEF
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.write("#### 📊 Asset Correlation Matrix (30 Days)")
            # Compute correlation on last 30 days
            corr_matrix = df_macro.tail(30).corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_right:
            st.write("#### 🧠 CIO Morning Brief")
            if st.button("Generate Macro Report"):
                with st.spinner("Analyzing Global Flows..."):
                    report = ask_cio(daily_change)
                    st.info(report)
            else:
                st.info("Click to let the AI analyze cross-asset flows.")

# --- TAB 2: THE SNIPER SCOPE ---
with tab2:
    st.sidebar.header("Sniper Controls")
    
    # Enhanced Asset List
    assets = ["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "QQQ", "NVDA", "TSLA", "MSTR", "COIN", "GC=F", "CL=F", "EURUSD=X"]
    ticker = st.sidebar.selectbox("Target Asset", assets)
    timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=2)
    balance = st.sidebar.number_input("Account Balance ($)", 1000, 1000000, 10000)
    
    if st.button("Analyze Target"):
        with st.spinner(f"Deploying Algorithms on {ticker}..."):
            # Get specific data
            data = yf.download(ticker, period="6mo", interval=timeframe, progress=False)
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
            
            # --- MATH CALCS (Re-implemented for the chart) ---
            # 1. Apex
            data['HMA'] = data['Close'].rolling(55).mean() # Proxy for full calc
            data['ATR'] = (data['High'] - data['Low']).rolling(14).mean()
            
            # 2. ATR Stops
            stop_long = data['Low'] - (data['ATR'] * 2)
            stop_short = data['High'] + (data['ATR'] * 2)
            
            # 3. Money Flow
            data['Flow'] = (data['Close'].diff() * data['Volume']).rolling(3).mean()
            
            # --- CHARTING ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            
            # Candles
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"), row=1, col=1)
            
            # ATR Stops (The Channels)
            fig.add_trace(go.Scatter(x=data.index, y=stop_long, line=dict(color='cyan', width=1), name="Long Stop"), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=stop_short, line=dict(color='magenta', width=1), name="Short Stop"), row=1, col=1)
            
            # Money Flow
            colors = ['#00ff00' if v > 0 else '#ff0000' for v in data['Flow']]
            fig.add_trace(go.Bar(x=data.index, y=data['Flow'], marker_color=colors, name="Smart Money"), row=2, col=1)
            
            fig.update_layout(height=700, template="plotly_dark", title=f"{ticker} Institutional View")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- AI SNIPER ---
            st.markdown("---")
            st.subheader("🎯 Execution Plan")
            
            # Get latest values
            last_price = data['Close'].iloc[-1]
            last_stop = stop_long.iloc[-1]
            last_atr = data['ATR'].iloc[-1]
            
            plan = ask_sniper(ticker, last_price, last_stop, last_atr, balance)
            st.success(plan)
            
            with st.expander("⚠️ Risk Disclaimer"):
                st.caption("Not financial advice. For educational and research purposes only.")
