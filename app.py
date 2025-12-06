import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from openai import OpenAI

# 1. SETUP PAGE
st.set_page_config(layout="wide", page_title="DarkPool Omni-Agent")
st.title("👁️ DarkPool Omni-Agent")

# 2. LOAD KEY
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# --- MACRO DASHBOARD FUNCTION ---
def get_macro_data():
    # Tickers: S&P500, Nasdaq, 10Y Yield, Dollar, VIX, Bitcoin
    tickers = ["SPY", "QQQ", "^TNX", "DX-Y.NYB", "^VIX", "BTC-USD"]
    try:
        # Bulk download for speed
        data = yf.download(tickers, period="5d", interval="1d")
        
        # Clean multi-index columns if necessary
        if isinstance(data.columns, pd.MultiIndex):
            # Extract just the 'Close' prices for easier handling
            df_close = data['Close']
        else:
            df_close = data['Close']

        # Calculate daily % change
        changes = df_close.pct_change().iloc[-1] * 100
        prices = df_close.iloc[-1]
        
        return prices, changes
    except Exception as e:
        return None, None

# --- RENDER MACRO DASHBOARD ---
st.markdown("### 🌍 Global Macro Regime")
macro_price, macro_change = get_macro_data()

if macro_price is not None:
    # Determine Regime
    # Simple Logic: If SPY & QQQ are Green -> Risk On. If VIX & DXY are Green -> Risk Off.
    spy_chg = macro_change.get("SPY", 0)
    vix_price = macro_price.get("^VIX", 0)
    
    regime = "NEUTRAL 🟡"
    if spy_chg > 0.5 and vix_price < 20:
        regime = "RISK-ON (Bullish) 🟢"
    elif spy_chg < -0.5 or vix_price > 25:
        regime = "RISK-OFF (Bearish) 🔴"
        
    st.info(f"**Current Market Regime:** {regime}")

    # Display Metrics in a Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    def show_metric(col, label, ticker_key, fmt="{:.2f}"):
        val = macro_price.get(ticker_key, 0)
        chg = macro_change.get(ticker_key, 0)
        col.metric(label, fmt.format(val), f"{chg:.2f}%")

    show_metric(m1, "S&P 500", "SPY")
    show_metric(m2, "Nasdaq", "QQQ")
    show_metric(m3, "Bitcoin", "BTC-USD")
    show_metric(m4, "10Y Yield", "^TNX")
    show_metric(m5, "Dollar (DXY)", "DX-Y.NYB")
    show_metric(m6, "VIX (Fear)", "^VIX")
    
    st.markdown("---") # Divider line

# 3. SIDEBAR SETTINGS
st.sidebar.header("Mission Control")

asset_options = {
    "--- CRYPTO ---": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"],
    "--- INDICES ---": ["SPY", "QQQ", "IWM", "DIA"], 
    "--- BIG TECH ---": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD"],
    "--- ENERGY & METALS ---": ["CL=F", "BZ=F", "NG=F", "GC=F", "SI=F", "HG=F", "PL=F", "PA=F"],
    "--- AGRO COMMODITIES ---": ["ZC=F", "ZS=F", "ZW=F", "CC=F", "KC=F"],
    "--- MACRO & RATES ---": ["DX-Y.NYB", "^TNX", "^FVX", "^TYX", "^VIX", "HYG", "TLT"],
    "--- FOREX ---": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
}

flat_options = []
for category, tickers in asset_options.items():
    flat_options.append(category)
    flat_options.extend(tickers)

ticker = st.sidebar.selectbox("Select Asset", options=flat_options, index=1)
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)

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
    if "---" in ticker: return None
    try:
        df = yf.download(ticker, period="1y", interval=interval)
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
    atr = float(last['ATR'])
    
    stop_long = price - (2 * atr)
    
    prompt = f"""
    Analyze {ticker}. Price: {price:.2f}.
    
    1. TREND: {apex}
    2. VOLATILITY: Squeeze is {sqz}. Momentum is {mom:.2f}.
    3. VOLUME: Money Flow is {flow:.2f}.
    4. RISK: ATR is {atr:.2f}.
    
    YOUR MISSION:
    1. Verdict: LONG, SHORT, or WAIT?
    2. Why? (Cite indicators).
    3. Risk: Suggest Stop Loss around {stop_long:.2f} (if Long).
    """
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 6. RENDER UI
if st.sidebar.button("Initialize Omni-Agent"):
    with st.spinner(f"Analyzing {ticker}..."):
        df = get_data(ticker, interval)
        
        if df is not None:
            # Row 1: Main Chart
            st.subheader(f"1. Apex Trend ({ticker})")
            fig1 = go.Figure()
            fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            fig1.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(color='green', width=1), name="Upper"))
            fig1.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], line=dict(color='red', width=1), name="Lower"))
            fig1.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Row 2: Indicators
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("2. Money Flow")
                fig2 = go.Figure()
                colors = ['#00ff00' if v > 0 else '#ff0000' for v in df['Matrix_Flow']]
                fig2.add_trace(go.Bar(x=df.index, y=df['Matrix_Flow'], marker_color=colors, name="Net Flow"))
                fig2.update_layout(height=300, template="plotly_dark", title="Volume Flow")
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                st.subheader("3. Squeeze Pro")
                fig3 = go.Figure()
                sqz_col = ['red' if s else 'gray' for s in df['Squeeze_On']]
                fig3.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='markers', marker=dict(color=sqz_col, size=5)))
                mom_col = ['cyan' if m > 0 else 'purple' for m in df['Momentum']]
                fig3.add_trace(go.Bar(x=df.index, y=df['Momentum'], marker_color=mom_col))
                fig3.update_layout(height=300, template="plotly_dark", title="Momentum")
                st.plotly_chart(fig3, use_container_width=True)

            # Row 3: Verdict
            st.markdown("---")
            st.subheader("🧠 Expert Verdict")
            st.info(ask_omni_agent(df, ticker))
        else:
            st.error("Select a valid asset.")
