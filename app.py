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
st.set_page_config(layout="wide", page_title="DarkPool Quant Terminal")
st.title("👁️ DarkPool Quant Terminal")
st.markdown("### Multi-Timeframe Quantitative Analysis")

if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if "OPENAI_API_KEY" in st.secrets:
    st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
else:
    if not st.session_state.api_key:
        st.session_state.api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# ==========================================
# 2. DATA ENGINE (MULTI-TIMEFRAME)
# ==========================================
def safe_download(ticker, period, interval):
    """Robust downloader."""
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
def get_quant_data(ticker, timeframe):
    """Downloads BOTH the trading timeframe AND the Daily timeframe for context."""
    # 1. Trading Data (e.g., 4h)
    df_trade = safe_download(ticker, "1y", timeframe)
    
    # 2. Macro Data (Daily)
    df_daily = safe_download(ticker, "2y", "1d")
    
    if df_trade is None or df_daily is None: return None, None
    
    return df_trade, df_daily

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    if "-" in ticker or "=" in ticker or "^" in ticker: return None 
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info: return None
        return {
            "Market Cap": info.get("marketCap", 0),
            "P/E": info.get("trailingPE", 0),
            "Growth": info.get("revenueGrowth", 0),
            "Sector": info.get("sector", "Unknown")
        }
    except: return None

# ==========================================
# 3. MATH ENGINE (QUANT LOGIC)
# ==========================================
def calculate_quant_metrics(df):
    # --- 1. TREND ---
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    df['EMA_200'] = df['Close'].ewm(span=200).mean()
    df['Trend'] = np.where(df['Close'] > df['EMA_50'], 1, -1) # 1=Bull, -1=Bear
    
    # --- 2. MOMENTUM (RSI) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- 3. VOLATILITY (ATR & REGIME) ---
    high_low = df['High'] - df['Low']
    df['ATR'] = high_low.rolling(14).mean()
    # Volatility Regime: Is current range > average range?
    df['Vol_Regime'] = np.where(high_low > df['ATR'], "EXPANDING", "COMPRESSING")
    
    # --- 4. MONEY FLOW ---
    df['MFI'] = (df['Close'].diff() * df['Volume']).rolling(3).mean()
    
    # --- 5. SUPPORT/RESISTANCE ---
    df['Resist'] = df['High'].rolling(20).max()
    df['Support'] = df['Low'].rolling(20).min()
    
    return df

def generate_score(df_trade, df_daily):
    """Calculates a 0-100 Quant Score based on confluence."""
    last_t = df_trade.iloc[-1]
    last_d = df_daily.iloc[-1]
    
    score = 50 # Start Neutral
    
    # Trend Alignment (+30)
    if last_t['Close'] > last_t['EMA_50']: score += 15
    if last_d['Close'] > last_d['EMA_50']: score += 15 # Daily trend is powerful
    
    # Momentum (+20)
    if 50 < last_t['RSI'] < 70: score += 10
    if 50 < last_d['RSI'] < 70: score += 10
    
    # Volume (+10)
    if last_t['MFI'] > 0: score += 10
    
    # Penalties
    if last_t['RSI'] > 75: score -= 10 (Overbought)
    if last_t['RSI'] < 25: score -= 10 (Oversold - catching knife)
    
    return max(0, min(100, score))

# ==========================================
# 4. AI QUANT BRAIN
# ==========================================
def ask_quant_brain(df_trade, df_daily, ticker, score, balance, risk_pct):
    if not st.session_state.api_key: return "⚠️ API Key Missing"
    
    last_t = df_trade.iloc[-1]
    last_d = df_daily.iloc[-1]
    
    # Determine Trend Alignment
    trend_micro = "BULLISH" if last_t['Close'] > last_t['EMA_50'] else "BEARISH"
    trend_macro = "BULLISH" if last_d['Close'] > last_d['EMA_50'] else "BEARISH"
    
    alignment = "PERFECT ALIGNMENT ✅" if trend_micro == trend_macro else "CONFLICT ⚠️"
    
    # Risk Math
    risk_dollars = balance * (risk_pct / 100)
    stop = last_t['Support'] if trend_micro == "BULLISH" else last_t['Resist']
    if pd.isna(stop): stop = last_t['Close'] * 0.95
    
    dist = abs(last_t['Close'] - stop)
    if dist == 0: dist = last_t['ATR']
    shares = risk_dollars / dist
    
    prompt = f"""
    Act as a Quantitative Hedge Fund Manager. Analyze {ticker}.
    
    --- QUANT METRICS ---
    Quant Score: {score}/100.
    Micro Trend ({last_t.name}): {trend_micro}.
    Macro Trend (Daily): {trend_macro}.
    Status: {alignment}.
    Volatility: {last_t['Vol_Regime']}.
    RSI: {last_t['RSI']:.1f}.
    
    --- RISK PARAMETERS ---
    Capital: ${balance}. Risk: ${risk_dollars:.2f} ({risk_pct}%).
    Stop Level: ${stop:.2f}. Position Size: {shares:.4f} units.
    
    --- MISSION ---
    1. **Verdict:** STRONG BUY, BUY, NEUTRAL, SELL, STRONG SELL.
    2. **Analysis:** Analyze the Alignment. Is the Daily chart supporting the trade?
    3. **The Plan:** Entry, Stop, Target (2R), and Size.
    """
    
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
        return res.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# ==========================================
# 5. UI DASHBOARD
# ==========================================
st.sidebar.header("🎛️ Quant Controls")

# Input
input_mode = st.sidebar.radio("Input Mode:", ["Curated Lists", "Manual Search (Global)"])
if input_mode == "Curated Lists":
    assets = {"Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"], "Indices": ["SPY", "QQQ"], "Tech": ["NVDA", "TSLA", "AAPL"]}
    cat = st.sidebar.selectbox("Category", list(assets.keys()))
    ticker = st.sidebar.selectbox("Ticker", assets[cat])
else:
    st.sidebar.info("Type ticker (e.g. SHEL.L)")
    ticker = st.sidebar.text_input("Symbol", value="AAPL").upper()

timeframe = st.sidebar.selectbox("Trading Timeframe", ["15m", "1h", "4h"], index=2)
st.sidebar.markdown("---")
balance = st.sidebar.number_input("Capital ($)", 1000, 1000000, 10000)
risk_pct = st.sidebar.slider("Risk %", 0.5, 3.0, 1.0)

# MAIN TRIGGER
if st.button("Run Quant Analysis"):
    with st.spinner(f"Running Multi-Timeframe Analysis on {ticker}..."):
        # Fetch TWO datasets
        df_trade, df_daily = get_quant_data(ticker, timeframe)
        
        if df_trade is not None and df_daily is not None:
            # Process Indicators
            df_trade = calculate_quant_metrics(df_trade)
            df_daily = calculate_quant_metrics(df_daily)
            
            # Generate Quant Score
            score = generate_score(df_trade, df_daily)
            
            # --- SCORECARD HEADER ---
            c1, c2, c3 = st.columns(3)
            c1.metric("Quant Score", f"{score}/100", delta="Bullish" if score > 60 else "Bearish")
            
            # Trend Alignment
            t_micro = "🟢 Up" if df_trade['Trend'].iloc[-1] == 1 else "🔴 Down"
            t_macro = "🟢 Up" if df_daily['Trend'].iloc[-1] == 1 else "🔴 Down"
            c2.metric("Trend Alignment", f"{t_micro} vs {t_macro}")
            
            # Volatility
            c3.metric("Volatility Regime", df_trade['Vol_Regime'].iloc[-1])
            
            st.markdown("---")
            
            # --- CHARTING (The "Triple Screen" View) ---
            tab_chart, tab_data = st.tabs(["Charts", "Raw Data"])
            
            with tab_chart:
                # We plot the Trading Timeframe with Daily Context levels
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                
                # Price
                fig.add_trace(go.Candlestick(x=df_trade.index, open=df_trade['Open'], high=df_trade['High'], low=df_trade['Low'], close=df_trade['Close'], name="Price"), row=1, col=1)
                
                # EMAs
                fig.add_trace(go.Scatter(x=df_trade.index, y=df_trade['EMA_50'], line=dict(color='cyan', width=1), name="EMA 50"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_trade.index, y=df_trade['EMA_200'], line=dict(color='blue', width=2), name="EMA 200"), row=1, col=1)
                
                # Support/Resist
                last_r = df_trade['Resist'].iloc[-1]
                last_s = df_trade['Support'].iloc[-1]
                if not pd.isna(last_r): fig.add_hline(y=last_r, line_dash="dash", line_color="red", row=1, col=1)
                if not pd.isna(last_s): fig.add_hline(y=last_s, line_dash="dash", line_color="green", row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(x=df_trade.index, y=df_trade['RSI'], line=dict(color='purple', width=2), name="RSI"), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)
                
                fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, title=f"{ticker} ({timeframe}) Structure")
                st.plotly_chart(fig, use_container_width=True)
            
            # --- AI VERDICT ---
            st.subheader("🧠 Quant Manager Briefing")
            verdict = ask_quant_brain(df_trade, df_daily, ticker, score, balance, risk_pct)
            st.success(verdict)
            
        else:
            st.error("Data fetch error. Check ticker symbol.")
