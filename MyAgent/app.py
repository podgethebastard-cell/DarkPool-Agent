import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from openai import OpenAI

# ==========================================
# 1. PAGE SETUP & AUTH
# ==========================================
st.set_page_config(layout="wide", page_title="DarkPool Omni-Agent")
st.title("👁️ DarkPool Omni-Agent")

# Load API Key safely
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# ==========================================
# 2. MACRO DASHBOARD (The "Heads Up" Display)
# ==========================================
def get_macro_data():
    # Tickers: S&P500, Nasdaq, 10Y Yield, Dollar, VIX, Bitcoin
    tickers = ["SPY", "QQQ", "^TNX", "DX-Y.NYB", "^VIX", "BTC-USD"]
    try:
        data = yf.download(tickers, period="5d", interval="1d")
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close']
        else:
            df_close = data['Close']
        
        # Calculate daily changes
        changes = df_close.pct_change().iloc[-1] * 100
        prices = df_close.iloc[-1]
        return prices, changes
    except:
        return None, None

st.markdown("### 🌍 Global Macro Regime")
macro_price, macro_change = get_macro_data()

if macro_price is not None:
    # Determine Risk Regime
    spy_chg = macro_change.get("SPY", 0)
    vix_price = macro_price.get("^VIX", 0)
    
    regime = "NEUTRAL 🟡"
    if spy_chg > 0.5 and vix_price < 20:
        regime = "RISK-ON (Bullish) 🟢"
    elif spy_chg < -0.5 or vix_price > 25:
        regime = "RISK-OFF (Bearish) 🔴"
        
    st.info(f"**Current Market Regime:** {regime}")

    # Metrics Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def show_metric(col, label, key):
        val = macro_price.get(key, 0)
        chg = macro_change.get(key, 0)
        col.metric(label, f"{val:.2f}", f"{chg:.2f}%")

    show_metric(c1, "S&P 500", "SPY")
    show_metric(c2, "Nasdaq", "QQQ")
    show_metric(c3, "Bitcoin", "BTC-USD")
    show_metric(c4, "10Y Yield", "^TNX")
    show_metric(c5, "Dollar Index", "DX-Y.NYB")
    show_metric(c6, "VIX (Fear)", "^VIX")
    st.markdown("---")

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Mission Control")

# Institutional Asset List
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

# ==========================================
# 4. MATH ENGINE (Indicators)
# ==========================================
def calc_hma(series, length):
    """Hull Moving Average Calculation"""
    wma1 = series.rolling(length // 2).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
    wma2 = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
    diff = 2 * wma1 - wma2
    sqrt_len = int(np.sqrt(length))
    return diff.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)

def calc_rsi(series, period):
    """RSI Calculation"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_data(ticker, interval):
    if "---" in ticker: return None
    try:
        df = yf.download(ticker, period="1y", interval=interval)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty: return None

        # 1. APEX TREND (HMA + ATR Cloud)
        length, mult = 55, 1.5
        df['HMA'] = calc_hma(df['Close'], length)
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(length).mean()
        df['Apex_Upper'] = df['HMA'] + (df['ATR'] * mult)
        df['Apex_Lower'] = df['HMA'] - (df['ATR'] * mult)
        
        # State: 1=Bull, -1=Bear
        df['Apex_State'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))
        df['Apex_State'] = df['Apex_State'].replace(to_replace=0, method='ffill')

        # 2. SQUEEZE PRO
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

        # 3. MONEY FLOW MATRIX
        rsi = calc_rsi(df['Close'], 14)
        df['MFI_Raw'] = (rsi - 50) * (df['Volume'] / df['Volume'].rolling(20).mean())
        df['Matrix_Flow'] = df['MFI_Raw'].ewm(span=3).mean()

        return df
    except Exception as e:
        return None

# ==========================================
# 5. AI BRAIN
# ==========================================
def ask_omni_agent(df, ticker):
    if not api_key: return "⚠️ Please insert API Key in Side Menu."
    
    last = df.iloc[-1]
    price = float(last['Close'])
    
    # Human-readable states
    apex = "BULLISH (Green Cloud)" if last['Apex_State'] == 1 else "BEARISH (Red Cloud)"
    sqz = "FIRING (Active)" if last['Squeeze_On'] else "RELEASED"
    mom = float(last['Momentum'])
    flow = float(last['Matrix_Flow'])
    atr = float(last['ATR'])
    
    # Calculate Risk Levels
    stop_loss = price - (2 * atr)
    take_profit = price + (3 * atr)
    
    prompt = f"""
    Act as a Senior Institutional Trader. Analyze {ticker} at price {price:.2f}.
    
    TECHNICAL DASHBOARD:
    1. TREND (Apex System): {apex}
    2. MOMENTUM (Squeeze Pro): Status is {sqz}. Momentum Value: {mom:.2f} (Positive=Bull, Negative=Bear).
    3. VOLUME (Money Flow): {flow:.2f} (Green=Buying, Red=Selling).
    4. VOLATILITY (ATR): {atr:.2f}.
    
    YOUR MISSION:
    1. Verdict: Give a clear LONG, SHORT, or WAIT signal.
    2. Confluence: Explain WHY. Do trend and volume agree?
    3. Trade Plan: If Long, suggested Stop Loss: {stop_loss:.2f}. If Short, invert this logic.
    """
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ==========================================
# 6. MAIN APP RENDER
# ==========================================
if st.sidebar.button("Initialize Omni-Agent"):
    with st.spinner(f"Analyzing {ticker} Market Structure..."):
        df = get_data(ticker, interval)
        
        if df is not None:
            # CHART 1: APEX TREND
            st.subheader(f"1. Apex Trend Architecture ({ticker})")
            fig1 = go.Figure()
            # Candles
            fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            # Clouds
            fig1.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(color='green', width=1), name="Apex Upper"))
            fig1.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], line=dict(color='red', width=1), name="Apex Lower"))
            fig1.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)
            
            # CHART 2: INDICATORS
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("2. Money Flow Matrix")
                fig2 = go.Figure()
                # Color logic: Green if buying, Red if selling
                colors = ['#00ff00' if v > 0 else '#ff0000' for v in df['Matrix_Flow']]
                fig2.add_trace(go.Bar(x=df.index, y=df['Matrix_Flow'], marker_color=colors, name="Net Flow"))
                fig2.update_layout(height=300, template="plotly_dark", title="Institutional Volume")
                st.plotly_chart(fig2, use_container_width=True)
                
            with col2:
                st.subheader("3. Squeeze Pro")
                fig3 = go.Figure()
                # Squeeze Dots: Red=Squeeze, Gray=Release
                sqz_col = ['red' if s else 'gray' for s in df['Squeeze_On']]
                fig3.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='markers', marker=dict(color=sqz_col, size=5), name="Squeeze Status"))
                # Momentum Histogram
                mom_col = ['cyan' if m > 0 else 'purple' for m in df['Momentum']]
                fig3.add_trace(go.Bar(x=df.index, y=df['Momentum'], marker_color=mom_col, name="Momentum"))
                fig3.update_layout(height=300, template="plotly_dark", title="Volatility & Momentum")
                st.plotly_chart(fig3, use_container_width=True)

            # AI SECTION
            st.markdown("---")
            st.subheader("🧠 Omni-Agent Verdict")
            st.info(ask_omni_agent(df, ticker))
            
            # LEGAL FOOTER
            st.markdown("---")
            with st.expander("⚠️ Legal Disclaimer & Risk Warning"):
                st.markdown("""
                **For Research Purposes Only.**
                This application is a prototype powered by Artificial Intelligence. It is designed for technical analysis experimentation and **does not constitute financial advice**.
                
                * **No Guarantees:** Past performance of indicators (Apex, Squeeze, Money Flow) does not guarantee future results.
                * **High Risk:** Trading assets involves substantial risk of loss. You could lose your entire investment.
                * **Data:** Market data is provided by third-party sources (Yahoo Finance) and may be delayed.
                
                **By using this tool, you agree that you are solely responsible for your own trading decisions.**
                """)
        else:
            st.error("Could not fetch data. Please select a valid asset ticker.")
