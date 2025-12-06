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
    api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        help="Enter your personal OpenAI API Key to enable the AI Analyst features."
    )

# --- HELPER: DYNAMIC PERIOD SELECTOR ---
def get_valid_period(interval):
    """
    Maps the selected interval to a valid Yahoo Finance history period
    to prevent 'No data found' errors on intraday timeframes.
    """
    if interval in ["1m", "2m", "5m"]:
        return "5d"
    elif interval in ["15m", "30m", "60m", "1h"]:
        return "1mo"  # Max buffer for intraday is usually ~60d
    elif interval == "4h":
        return "1mo"
    else:
        return "1y"  # Daily/Weekly can handle long history

# --- MACRO DASHBOARD FUNCTION ---
@st.cache_data(ttl=600) # Added caching to prevent rate limiting
def get_macro_data():
    # Tickers: S&P500, Nasdaq, 10Y Yield, Dollar, VIX, Bitcoin
    tickers = ["SPY", "QQQ", "^TNX", "DX-Y.NYB", "^VIX", "BTC-USD"]
    try:
        data = yf.download(tickers, period="5d", interval="1d", progress=False)
        
        # Robust MultiIndex Handling
        if isinstance(data.columns, pd.MultiIndex):
            # If the columns are (Price, Ticker), we want 'Close'
            if 'Close' in data.columns.levels[0]:
                df_close = data['Close']
            else:
                df_close = data
        else:
            df_close = data['Close'] if 'Close' in data.columns else data

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
    
    st.markdown("---") 

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

# ADDED TOOLTIPS HERE
ticker = st.sidebar.selectbox(
    "Select Asset", 
    options=flat_options, 
    index=1,
    help="Choose the financial instrument you want to analyze. Sections starting with '---' are headers and cannot be selected."
)

interval = st.sidebar.selectbox(
    "Timeframe", 
    ["15m", "1h", "4h", "1d", "1wk"], 
    index=3,
    help="Select the candle duration. Note: 15m and 1h charts are limited to ~60 days of history."
)

# --- MATH LIBRARY ---
def calc_hma(series, length):
    # Simplified HMA logic to ensure stability
    wma1 = series.rolling(length // 2).mean() # Using Simple MA proxies for speed/stability in this demo
    wma2 = series.rolling(length).mean()
    diff = 2 * wma1 - wma2
    # In a full production app, implement true WMA, but this prevents numpy errors on short data
    return diff.rolling(int(np.sqrt(length))).mean()

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
        # DYNAMIC PERIOD FIX
        valid_period = get_valid_period(interval)
        
        df = yf.download(ticker, period=valid_period, interval=interval, progress=False)
        
        if df.empty: return None

        # Robust Column Handling
        if isinstance(df.columns, pd.MultiIndex):
            # Check if we can simply drop level, or if we need to select specific ticker
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass # If structure is different, we might already have single level

        # Drop NaNs to prevent math errors
        df = df.dropna()

        # A. APEX TREND
        length, mult = 55, 1.5
        df['HMA'] = calc_hma(df['Close'], length)
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(length).mean()
        df['Apex_Upper'] = df['HMA'] + (df['ATR'] * mult)
        df['Apex_Lower'] = df['HMA'] - (df['ATR'] * mult)
        
        # Vectorized State Calculation
        df['Apex_State'] = 0
        df.loc[df['Close'] > df['Apex_Upper'], 'Apex_State'] = 1
        df.loc[df['Close'] < df['Apex_Lower'], 'Apex_State'] = -1
        # Fill zeros (neutral) with previous state to show trend persistence
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

        return df.dropna() # Return only valid rows
    except Exception as e:
        st.error(f"Data Error: {e}")
        return None

# 5. AI SYNTHESIS
def ask_omni_agent(df, ticker):
    if not api_key: return "⚠️ Please insert API Key."
    
    try:
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
    except Exception as e:
        return f"AI Error: {str(e)}"

# 6. RENDER UI
# ADDED TOOLTIP HERE
if st.sidebar.button("Initialize Omni-Agent", help="Click to pull live data and generate the technical analysis report."):
    with st.spinner(f"Analyzing {ticker}..."):
        df = get_data(ticker, interval)
        
        if df is not None and not df.empty:
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
                # Handle boolean coloring for squeeze
                sqz_col = ['red' if s else 'gray' for s in df['Squeeze_On']]
                # Use a zero line with markers for the Squeeze dots
                fig3.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='markers', marker=dict(color=sqz_col, size=5), name="Squeeze Dots"))
                mom_col = ['cyan' if m > 0 else 'purple' for m in df['Momentum']]
                fig3.add_trace(go.Bar(x=df.index, y=df['Momentum'], marker_color=mom_col, name="Momentum"))
                fig3.update_layout(height=300, template="plotly_dark", title="Momentum")
                st.plotly_chart(fig3, use_container_width=True)

            # Row 3: Verdict
            st.markdown("---")
            st.subheader("🧠 Expert Verdict")
            st.info(ask_omni_agent(df, ticker))
        else:
            st.error("Select a valid asset. If using intraday (15m/1h), ensure the market is open or data is available.")
