import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from openai import OpenAI

# 1. SETUP PAGE
st.set_page_config(layout="wide", page_title="DarkPool Omni-Agent")
st.title("👁️ DarkPool Omni-Agent")
st.markdown("### The Convergence of Trend, Volume, and Volatility")

# 2. LOAD KEY
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# 3. SIDEBAR SETTINGS
st.sidebar.header("Mission Control")
ticker = st.sidebar.text_input("Asset", value="BTC-USD", help="Ticker Symbol")
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=2)

# --- MATH LIBRARY (The Translation Layer) ---
def calc_hma(series, length):
    """Hull Moving Average"""
    wma1 = series.rolling(length // 2).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
    wma2 = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
    diff = 2 * wma1 - wma2
    sqrt_len = int(np.sqrt(length))
    return diff.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)

def calc_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 4. DATA ENGINE (The Heavy Lifting)
def get_data(ticker, interval):
    try:
        df = yf.download(ticker, period="6mo", interval=interval)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty: return None

        # --- A. APEX TREND (HMA Cloud) ---
        length, mult = 55, 1.5
        df['HMA'] = calc_hma(df['Close'], length)
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(length).mean()
        df['Apex_Upper'] = df['HMA'] + (df['ATR'] * mult)
        df['Apex_Lower'] = df['HMA'] - (df['ATR'] * mult)
        # Trend State (1=Bull, -1=Bear)
        df['Apex_State'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))
        # Fill zeros (Neutral) with previous state
        df['Apex_State'] = df['Apex_State'].replace(to_replace=0, method='ffill')

        # --- B. SQUEEZE PRO (LazyBear) ---
        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Up'] = df['BB_Mid'] + (2.0 * df['BB_Std'])
        df['BB_Low'] = df['BB_Mid'] - (2.0 * df['BB_Std'])
        # Keltner Channels
        df['KC_Mid'] = df['BB_Mid'] # Share baseline
        df['KC_ATR'] = df['TR'].rolling(20).mean()
        df['KC_Up'] = df['KC_Mid'] + (1.5 * df['KC_ATR'])
        df['KC_Low'] = df['KC_Mid'] - (1.5 * df['KC_ATR'])
        # Squeeze Condition (BB inside KC)
        df['Squeeze_On'] = (df['BB_Low'] > df['KC_Low']) & (df['BB_Up'] < df['KC_Up'])
        # Momentum (Simplified LinReg)
        df['Momentum'] = df['Close'] - df['Close'].rolling(20).mean()

        # --- C. MONEY FLOW MATRIX ---
        # Normalized RSI weighted by Volume
        rsi = calc_rsi(df['Close'], 14)
        df['MFI_Raw'] = (rsi - 50) * (df['Volume'] / df['Volume'].rolling(20).mean())
        df['Matrix_Flow'] = df['MFI_Raw'].ewm(span=3).mean() # Smoothed

        return df
    except Exception as e:
        return None

# 5. THE AI SYNTHESIS
def ask_omni_agent(df, ticker):
    if not api_key: return "⚠️ Please insert API Key."
    
    last = df.iloc[-1]
    
    # Extract Metrics
    price = float(last['Close'])
    apex = "BULLISH 🟢" if last['Apex_State'] == 1 else "BEARISH 🔴"
    sqz = "FIRING (Active)" if last['Squeeze_On'] else "RELEASED (Trending)"
    mom = float(last['Momentum'])
    flow = float(last['Matrix_Flow'])
    
    flow_status = "Instits Buying" if flow > 0 else "Instits Selling"
    
    prompt = f"""
    You are the 'DarkPool Omni-Agent'. Synthesize these 3 elite indicators for {ticker}:
    
    1. APEX TREND (Structure): {apex}
    2. SQUEEZE PRO (Volatility): Status={sqz}, Momentum={mom:.2f} (Positive=Bull, Negative=Bear).
    3. MONEY FLOW MATRIX (Volume): Score={flow:.2f} ({flow_status}).
    
    DATA CONTEXT:
    {df.tail(5)[['Close', 'Apex_State', 'Momentum', 'Matrix_Flow']].to_string()}
    
    YOUR MISSION:
    - Determine if there is CONFLUENCE (Do Trend + Volume agree?).
    - If Apex is Bullish but Money Flow is Negative, warn of a "Fakeout".
    - If Squeeze is Firing and Apex is Neutral, warn of an "Explosive Move Soon".
    - Give a final verdict: LONG, SHORT, or WAIT.
    """
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 6. RENDER UI
if st.sidebar.button("Initialize Omni-Agent"):
    with st.spinner("Crunching DarkPool Data..."):
        df = get_data(ticker, interval)
        
        if df is not None:
            # --- ROW 1: PRICE & APEX ---
            st.subheader(f"1. Apex Trend Architecture ({ticker})")
            fig1 = go.Figure()
            # Candles
            fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            # Apex Cloud
            fig1.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(color='green', width=1), name="Apex Upper"))
            fig1.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], line=dict(color='red', width=1), name="Apex Lower"))
            fig1.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)
            
            # --- ROW 2: MATRIX & SQUEEZE ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("2. Money Flow Matrix (Volume)")
                fig2 = go.Figure()
                # Color logic for bars
                colors = ['#00ff00' if v > 0 else '#ff0000' for v in df['Matrix_Flow']]
                fig2.add_trace(go.Bar(x=df.index, y=df['Matrix_Flow'], marker_color=colors, name="Net Flow"))
                fig2.update_layout(height=300, template="plotly_dark", title="Institutional Volume Flow")
                st.plotly_chart(fig2, use_container_width=True)
                
            with col2:
                st.subheader("3. Squeeze Momentum (Volatility)")
                fig3 = go.Figure()
                # Squeeze Dots (0 line)
                sqz_col = ['red' if s else 'gray' for s in df['Squeeze_On']]
                fig3.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='markers', marker=dict(color=sqz_col, size=5), name="Squeeze Status"))
                # Momentum Hist
                mom_col = ['cyan' if m > 0 else 'purple' for m in df['Momentum']]
                fig3.add_trace(go.Bar(x=df.index, y=df['Momentum'], marker_color=mom_col, name="Momentum"))
                fig3.update_layout(height=300, template="plotly_dark", title="Squeeze Pro")
                st.plotly_chart(fig3, use_container_width=True)

            # --- ROW 3: AI BRAIN ---
            st.markdown("---")
            st.subheader("🧠 Omni-Agent Final Verdict")
            analysis = ask_omni_agent(df, ticker)
            st.info(analysis)
            
        else:
            st.error("Error fetching data. Try a standard ticker like BTC-USD or AAPL.")
