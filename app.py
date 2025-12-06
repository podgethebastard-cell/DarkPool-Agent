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

# Load API Key
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# ==========================================
# 2. DATA ENGINE (ROBUST & CACHED)
# ==========================================
@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    """Fetches key financial metrics for Stocks (UTP Rule I, Fundamentals)"""
    if "-" in ticker or "=" in ticker or "^" in ticker: return None 
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Market Cap": info.get("marketCap", 0),
            "P/E Ratio": info.get("trailingPE", 0),
            "PEG Ratio": info.get("pegRatio", 0),
            "Rev Growth": info.get("revenueGrowth", 0),
            "Debt/Equity": info.get("debtToEquity", 0),
            "Currency": info.get('currency', 'USD'),
            "Summary": info.get("longBusinessSummary", "No Data")
        }
    except: return None

@st.cache_data(ttl=300)
def get_sector_data():
    """Fetches performance of key US Sectors for Rotation Analysis"""
    sectors = {
        "Tech (XLK)": "XLK", "Energy (XLE)": "XLE", "Financials (XLF)": "XLF", 
        "Healthcare (XLV)": "XLV", "Utilities (XLU)": "XLU", "Consumer (XLY)": "XLY",
        "Industrial (XLI)": "XLI", "Materials (XLB)": "XLB", "Real Estate (XLRE)": "XLRE"
    }
    try:
        data = yf.download(list(sectors.values()), period="5d", interval="1d", progress=False)['Close']
        changes = data.pct_change().iloc[-1] * 100
        results = {name: changes[ticker] for name, ticker in sectors.items()}
        return pd.Series(results).sort_values(ascending=False)
    except: return None

def get_news(ticker):
    """Fetches latest news headlines safely"""
    try:
        if "=" in ticker or "^" in ticker: return []
        stock = yf.Ticker(ticker)
        return stock.news[:5] 
    except: return []

def safe_download(ticker, period, interval):
    """Robust price downloader that handles multi-index issues"""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.droplevel(1) 
            except: pass 
        if df.empty or 'Close' not in df.columns: return None
        return df
    except: return None

# ==========================================
# 3. MATH LIBRARY (FULL INDICATOR STACK)
# ==========================================
def calc_indicators(df):
    # 1. Apex Trend (HMA 55)
    df['HMA'] = df['Close'].rolling(55).mean()
    
    # 2. ATR (Volatility)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # 3. Auto Support/Resistance (Pivots - UTP Rule III)
    df['Pivot_Resist'] = df['High'].rolling(20, center=False).max()
    df['Pivot_Support'] = df['Low'].rolling(20, center=False).min()
    
    # 4. Money Flow Matrix (Smart Volume)
    df['MFI'] = (df['Close'].diff() * df['Volume']).rolling(3).mean()
    
    # 5. Squeeze Pro
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['KC_ATR'] = df['ATR'].rolling(20).mean()
    df['Sq_On'] = (df['BB_Mid'] + 2*df['BB_Std']) < (df['BB_Mid'] + 1.5*df['KC_ATR'])
    df['Mom'] = df['Close'] - df['Close'].rolling(20).mean()
    
    return df

# ==========================================
# 4. AI TITAN BRAIN (RISK MANAGER)
# ==========================================
def ask_ai_analyst(df, ticker, fundamentals, news_list, balance, risk_pct):
    if not api_key: return "⚠️ Waiting for API Key..."
    
    last = df.iloc[-1]
    
    # --- FIX APPLIED HERE: Use .get() ---
    news_text = "\n".join([f"- {n.get('title', 'Headline Missing')}" for n in news_list]) if news_list else "No specific news headlines available."
    
    # Technical States
    trend = "BULLISH" if last['Close'] > last['HMA'] else "BEARISH"
    
    # Risk Calculations (UTP Rule I & II)
    risk_dollars = balance * (risk_pct / 100)
    
    if trend == "BULLISH":
        stop_level = last['Pivot_Support']
        direction = "LONG"
    else:
        stop_level = last['Pivot_Resist']
        direction = "SHORT"
        
    # Safety Check for Stop Loss
    if pd.isna(stop_level) or abs(last['Close'] - stop_level) < (last['ATR']*0.5):
        stop_level = last['Close'] - (last['ATR']*2) if direction == "LONG" else last['Close'] + (last['ATR']*2)
        
    dist = abs(last['Close'] - stop_level)
    shares = risk_dollars / dist if dist > 0 else 0
    target = last['Close'] + (dist * 2.5) if direction == "LONG" else last['Close'] - (dist * 2.5)
    
    # Format Fundamentals for AI
    fund_text = "N/A"
    if fundamentals:
        fund_text = f"P/E: {fundamentals.get('P/E Ratio', 'N/A')}. Growth: {fundamentals.get('Rev Growth', 0)*100:.1f}%. Debt/Eq: {fundamentals.get('Debt/Equity', 'N/A')}."
    
    prompt = f"""
    Act as a Global Macro Strategist and Risk Manager. Analyze {ticker} at ${last['Close']:.2f}.
    
    --- 1. FUNDAMENTALS ---
    {fund_text}
    
    --- 2. TECHNICALS ---
    Trend: {trend}. Momentum: {last['Mom']:.2f}. Money Flow: {last['MFI']:.0f}.
    
    --- 3. NEWS SENTIMENT ---
    {news_text}
    
    --- 4. RISK PROTOCOL ---
    Capital: ${balance}. Risk Budget: ${risk_dollars:.2f} ({risk_pct}%).
    Stop Loss: ${stop_level:.2f}. Position Size: {shares:.4f} units.
    
    --- MISSION ---
    1. **The Verdict:** STRONG BUY, ACCUMULATE, WAIT, DISTRIBUTE, or STRONG SELL.
    2. **The Thesis:** Combine Technicals (Timing) and Fundamentals (Quality).
    3. **Execution Card:**
       - **Action:** (Clear instruction)
       - **Entry:** Market
       - **Stop Loss:** ${stop_level:.2f} (UTP Compliant)
       - **Take Profit (2.5R):** ${target:.2f}
       - **Size:** {shares:.4f} units
    """
    
    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
    return res.choices[0].message.content

# ==========================================
# 5. UI DASHBOARD LAYOUT
# ==========================================
st.sidebar.header("🎛️ Terminal Controls")

# Input Mode Switcher
input_mode = st.sidebar.radio("Input Mode:", ["Curated Lists", "Manual Search (Global)"])

if input_mode == "Curated Lists":
    assets = {
        "Indices": ["SPY", "QQQ", "IWM", "^VIX"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
        "Commodities": ["GC=F", "CL=F", "HG=F", "NG=F"],
        "Tech Giants": ["NVDA", "TSLA", "AAPL", "MSFT", "AMD"],
        "Macro Rates": ["^TNX", "DX-Y.NYB", "TLT", "HYG"]
    }
    cat = st.sidebar.selectbox("Asset Class", list(assets.keys()))
    ticker = st.sidebar.selectbox("Ticker", assets[cat])
else:
    st.sidebar.info("Type ANY global ticker (e.g. SHEL.L, 7203.T)")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()

interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=3)

st.sidebar.markdown("---")
st.sidebar.header("💰 Risk Parameters")
balance = st.sidebar.number_input("Capital ($)", 1000, 1000000, 10000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0)

# --- GLOBAL MACRO HEADER ---
st.write("Fetching Global Data...") # Placeholder to prevent empty space during load

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
                
                # 1. Price & Trend (Top Chart)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='orange', width=2), name="Apex Trend"), row=1, col=1)
                
                # Auto Support/Resistance (Dashed Lines)
                last_res = df['Pivot_Resist'].iloc[-1]
                last_sup = df['Pivot_Support'].iloc[-1]
                if not pd.isna(last_res): fig.add_hline(y=last_res, line_dash="dash", line_color="red", annotation_text="Resist", row=1, col=1)
                if not pd.isna(last_sup): fig.add_hline(y=last_sup, line_dash="dash", line_color="green", annotation_text="Support", row=1, col=1)
                
                # 2. Money Flow (Bottom Chart)
                colors = ['#00ff00' if v > 0 else '#ff0000' for v in df['MFI']]
                fig.add_trace(go.Bar(x=df.index, y=df['MFI'], marker_color=colors, name="Smart Money"), row=2, col=1)
                
                fig.update_layout(height=700, template="plotly_dark", title=f"Institutional View: {ticker}", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # --- LIVE INDICATOR DASHBOARD ---
                m1, m2, m3, m4 = st.columns(4)
                last_bar = df.iloc[-1]
                
                m1.metric("Current Price", f"${last_bar['Close']:.2f}")
                m2.metric("Money Flow", f"{last_bar['MFI']:.0f}", delta_color="normal" if last_bar['MFI'] > 0 else "inverse")
                m3.metric("Squeeze", "FIRING 🔥" if last_bar['Squeeze_On'] else "OFF", delta_color="off")
                m4.metric("Volatility (ATR)", f"{last_bar['ATR']:.2f}")
                
                # --- AI VERDICT ---
                st.markdown("### 🤖 Strategy Briefing")
                verdict = ask_ai_analyst(df, ticker, fund, news, balance, risk_pct)
                st.info(verdict)

            # --- TAB 2: FUNDAMENTALS & MACRO ---
            with tab2:
                st.subheader(f"🏢 Fundamental Health: {ticker}")
                if fund:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("P/E Ratio", f"{fund.get('P/E Ratio', 'N/A')}")
                    c2.metric("Rev Growth", f"{fund.get('Rev Growth', 0)*100:.1f}%")
                    c3.metric("Debt/Equity", f"{fund.get('Debt/Equity', 'N/A')}")
                    st.write(f"**Business Summary:** {fund.get('Summary', 'No Data')[:400]}...")
                else:
                    st.warning("Fundamental data not available for this asset class (Crypto/Forex/Futures).")
                
                st.markdown("---")
                st.subheader("🏆 Sector Performance")
                # Sector Data is cached and ready to display
                sector_data = get_sector_data()
                if sector_data is not None:
                    # FIX: DataFrame styling error
                    styled_df = sector_data.to_frame(name="Change").style.format("{:.2f}%").background_gradient(cmap="RdYlGn", vmin=-2, vmax=2)
                    st.dataframe(styled_df, height=400)
                
                st.markdown("---")
                st.subheader("📰 Live News Wire")
                for n in news:
                    # --- FIX APPLIED HERE ALSO ---
                    st.write(f"• **{n.get('title', 'Headline Missing')}**")
                    st.caption(f"Source: {n.get('publisher', 'N/A')} | [Read Story]({n.get('link', '#')})")
                    
        else:
            st.error("Data connection failed. Try another ticker or wait 1 min.")
