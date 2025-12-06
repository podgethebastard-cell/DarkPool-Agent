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

# --- MATH LIBRARY ---
def calc_hma(series, length):
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

# 4. DATA ENGINE
def get_data(ticker, interval):
    try:
        df = yf.download(ticker, period="6mo", interval=interval)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty: return None

        # A. APEX TREND
        length, mult = 55, 1.5
        df['HMA'] = calc_hma(df['Close'], length)
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(length).mean()
        df['Apex_Upper'] = df['HMA'] + (df['ATR'] * mult)
        df['Apex_Lower'] = df['HMA'] - (df['ATR'] * mult)
        df['Apex_State'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))
        df['Apex_State'] = df['Apex_State'].replace(to_replace=0, method='ffill')

        # B. SQUEEZE PRO
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Up'] = df['BB_Mid'] + (2.0 * df['BB_Std'])
        df['BB_Low'] = df['BB_Mid'] - (2.0 * df['BB_Std'])
        df['KC_Mid'] = df['BB_Mid']
        df['KC_ATR'] = df['TR'].rolling(20).mean()
        df['KC_Up'] = df['KC_Mid'] + (1.5 * df['KC_ATR'])
        df['KC_Low'] = df['KC_Mid'] - (1.5 * df['KC_ATR'])
        df['Squeeze_On'] = (df['BB_Low'] > df['KC_Low']) & (df['BB_Up'] < df['KC_Up'])
        df['Momentum'] = df['Close'] - df['Close'].rolling(20).mean()

        # C. MONEY FLOW MATRIX
        rsi = calc_rsi(df['Close'], 14)
        df['MFI_Raw'] = (rsi - 50) * (df['Volume'] / df['Volume'].rolling(20).mean())
        df['Matrix_Flow'] = df['MFI_Raw'].ewm(span=3).mean()

        return df
    except Exception as e:
        return None

# 5. AI SYNTHESIS
def ask_omni_agent(df, ticker):
    if not api_key: return "⚠️ Please insert API Key."
    
    last = df.iloc[-1]
    price = float(last['Close'])
    apex = "BULLISH 🟢" if last['Apex_State'] == 1 else "BEARISH 🔴"
    sqz = "FIRING (Active)" if last['Squeeze_On'] else "RELEASED (Trending)"
    mom = float(last['Momentum'])
    flow = float(last['Matrix_Flow'])
    
    prompt = f"""
    Analyze {ticker}. Price: {price:.2f}.
    
    1. TREND: {apex}
    2. VOLATILITY: Squeeze is {sqz}. Momentum is {mom:.2f}.
    3. VOLUME: Money Flow is {flow:.2f}.
    
    Tell me: LONG, SHORT, or WAIT? Explain why using the data above.
    """
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 6. RENDER UI
if st.sidebar.button("Initialize Omni-Agent"):
    with st.spinner("Crunching Data..."):
        df = get_data(ticker, interval)
        
        if df is not None:
            st.subheader(f"1. Apex Trend ({ticker})")
            fig1 = go.Figure()
            fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            fig1.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(color='green', width=1), name="Upper"))
            fig1.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], line=dict(color='red', width=1), name="Lower"))
            fig1.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("🧠 Omni-Agent Verdict")
            st.info(ask_omni_agent(df, ticker))
        else:
            st.error("Error fetching data.")
