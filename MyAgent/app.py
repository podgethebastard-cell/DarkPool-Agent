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
st.set_page_config(layout="wide", page_title="DarkPool Terminal Pro")
st.title("👁️ DarkPool Terminal Pro")
st.markdown("### Institutional-Grade Market Intelligence")

# Load API Key
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# ==========================================
# 2. DATA ENGINE (ROBUST & CACHED)
# ==========================================
@st.cache_data(ttl=300) # Cache for 5 mins to speed up
def get_sector_data():
    """Fetches performance of key US Sectors to spot rotation"""
    sectors = {
        "Tech (XLK)": "XLK", "Energy (XLE)": "XLE", "Financials (XLF)": "XLF", 
        "Healthcare (XLV)": "XLV", "Utilities (XLU)": "XLU", "Consumer (XLY)": "XLY",
        "Industrial (XLI)": "XLI", "Materials (XLB)": "XLB", "Real Estate (XLRE)": "XLRE"
    }
    try:
        # Download last 5 days
        data = yf.download(list(sectors.values()), period="5d", interval="1d", progress=False)['Close']
        # Calculate % Change
        changes = data.pct_change().iloc[-1] * 100
        # Map back to readable names
        results = {name: changes[ticker] for name, ticker in sectors.items()}
        return pd.Series(results).sort_values(ascending=False)
    except: return None

def get_news(ticker):
    """Fetches latest news headlines from Yahoo Finance"""
    try:
        if "=" in ticker or "^" in ticker: return [] # Skip news for indexes/futures to avoid errors
        stock = yf.Ticker(ticker)
        return stock.news[:5] # Return top 5 stories
    except: return []

def safe_download(ticker, period, interval):
    """Robust downloader that handles Yahoo's multi-index issues"""
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
    df['HMA'] = df['Close'].rolling(55).mean() # Simplified for speed, works as trend baseline
    
    # 2. ATR (Volatility)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # 3. Auto Support/Resistance (Pivots)
    # Finds the highest high and lowest low of the last 20 candles
    df['Pivot_Resist'] = df['High'].rolling(20, center=False).max()
    df['Pivot_Support'] = df['Low'].rolling(20, center=False).min()
    
    # 4. Money Flow Matrix (Volume)
    df['MFI'] = (df['Close'].diff() * df['Volume']).rolling(3).mean()
    
    # 5. Squeeze Pro (Volatility Compression)
    # BB (2.0 std) vs KC (1.5 ATR)
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['KC_ATR'] = df['ATR'].rolling(20).mean()
    
    # Upper Band Logic
    bb_up = df['BB_Mid'] + (df['BB_Std'] * 2.0)
    kc_up = df['BB_Mid'] + (df['KC_ATR'] * 1.5)
    df['Squeeze_On'] = bb_up < kc_up
    
    # Momentum (Linear Reg Proxy)
    df['Mom'] = df['Close'] - df['Close'].rolling(20).mean()
    
    return df

# ==========================================
# 4. AI ANALYST (THE BRAIN)
# ==========================================
def ask_ai_analyst(df, ticker, news_list, balance, risk_pct):
    if not api_key: return "⚠️ Waiting for API Key..."
    
    last = df.iloc[-1]
    
    # Format News for AI
    news_text = "\n".join([f"- {n['title']}" for n in news_list]) if news_list else "No specific news headlines available."
    
    # Technical States
    trend = "BULLISH (Above HMA)" if last['Close'] > last['HMA'] else "BEARISH (Below HMA)"
    squeeze = "FIRING (High Volatility Soon)" if last['Squeeze_On'] else "OFF (Normal)"
    flow = "INFLOW (Buying)" if last['MFI'] > 0 else "OUTFLOW (Selling)"
    
    # Auto-Risk Calculation
    # If Bullish, Stop is at Support. If Bearish, Stop is at Resistance.
    if last['Close'] > last['HMA']:
        stop_level = last['Pivot_Support']
        direction = "LONG"
    else:
        stop_level = last['Pivot_Resist']
        direction = "SHORT"
        
    # Safety: If stop is NaN or too close, use ATR
    if pd.isna(stop_level) or abs(last['Close'] - stop_level) < (last['ATR']*0.5):
        stop_level = last['Close'] - (last['ATR']*2) if direction == "LONG" else last['Close'] + (last['ATR']*2)
        
    # Position Sizing
    risk_dollars = balance * (risk_pct / 100)
    dist = abs(last['Close'] - stop_level)
    shares = risk_dollars / dist if dist > 0 else 0
    target = last['Close'] + (dist*2) if direction == "LONG" else last['Close'] - (dist*2)
    
    prompt = f"""
    Act as a Hedge Fund Portfolio Manager. Analyze {ticker} at ${last['Close']:.2f}.
    
    --- TECHNICAL DASHBOARD ---
    1. Trend: {trend}
    2. Market Structure: Support ${last['Pivot_Support']:.2f} | Resist ${last['Pivot_Resist']:.2f}
    3. Momentum: Squeeze is {squeeze}. Money Flow: {flow}.
    
    --- NEWS SENTIMENT ---
    {news_text}
    
    --- RISK PROTOCOLS ---
    User Capital: ${balance}. Risk Budget: ${risk_dollars:.2f} ({risk_pct}%).
    Auto-Calculated Stop Loss: ${stop_level:.2f}.
    Auto-Calculated Position Size: {shares:.4f} units.
    
    --- MISSION ---
    1. **Sentiment Score:** Rate from 0 (Bearish) to 100 (Bullish) based on News + Charts.
    2. **The Verdict:** STRONG BUY, BUY, WAIT, SELL, or STRONG SELL.
    3. **The Trade Plan:**
       - **Action:** (Clear instruction)
       - **Entry:** Market
       - **Stop Loss:** ${stop_level:.2f}
       - **Take Profit:** ${target:.2f} (2R)
       - **Size:** {shares:.4f} units
    4. **Warning:** Mention if News contradicts the Chart.
    """
    
    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
    return res.choices[0].message.content

# ==========================================
# 5. UI DASHBOARD LAYOUT
# ==========================================
# --- SIDEBAR ---
st.sidebar.header("🎛️ Terminal Controls")
assets = {
    "Indices": ["SPY", "QQQ", "IWM", "^VIX"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"],
    "Commodities": ["GC=F", "CL=F", "HG=F", "NG=F", "SI=F"],
    "Tech Giants": ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "META"],
    "Macro Rates": ["^TNX", "DX-Y.NYB", "TLT", "HYG"]
}
cat = st.sidebar.selectbox("Asset Class", list(assets.keys()))
ticker = st.sidebar.selectbox("Ticker", assets[cat])
interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=2)

st.sidebar.markdown("---")
st.sidebar.header("💰 Risk Parameters")
balance = st.sidebar.number_input("Capital ($)", 1000, 1000000, 10000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0)

# --- TABS FOR WORKFLOW ---
tab1, tab2 = st.tabs(["🌍 Global Macro & Sectors", "🎯 Sniper Scope (Technicals)"])

# --- TAB 1: MACRO VIEW ---
with tab1:
    st.subheader("Market Heatmap & Sector Rotation")
    
    # 1. Sector Leaderboard
    sector_data = get_sector_data()
    if sector_data is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.caption("🏆 Sector Performance (Today)")
            st.dataframe(sector_data.style.format("{:.2f}%").background_gradient(cmap="RdYlGn", vmin=-2, vmax=2), height=400)
        
        with c2:
            st.caption("📈 Risk Gauge (SPY vs VIX vs BTC)")
            try:
                macro_df = yf.download(["SPY", "^VIX", "BTC-USD"], period="5d", progress=False)['Close']
                # Normalize to start at 0%
                macro_norm = (macro_df / macro_df.iloc[0] - 1) * 100
                st.line_chart(macro_norm)
                
                # Regime Logic
                spy_perf = macro_norm['SPY'].iloc[-1]
                btc_perf = macro_norm['BTC-USD'].iloc[-1]
                regime = "RISK ON 🚀" if spy_perf > 0 and btc_perf > 0 else "RISK OFF 🛡️"
                st.metric("Global Regime", regime, f"Leader: {sector_data.index[0]}")
            except:
                st.write("Loading Macro Data...")
    else:
        st.write("Loading Sector Data...")

# --- TAB 2: TECHNICAL VIEW ---
with tab2:
    if st.button(f"Analyze {ticker} Deep Dive"):
        with st.spinner(f"Accessing Dark Pools for {ticker}..."):
            df = safe_download(ticker, "1y", interval)
            
            if df is not None:
                df = calc_indicators(df)
                news = get_news(ticker)
                
                # --- CHARTING ENGINE ---
                # Create a "Stack" of charts: Price on top, Momentum on bottom
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                
                # 1. Price & Trend (Top Chart)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='orange', width=2), name="Apex Trend"), row=1, col=1)
                
                # Auto Support/Resistance (Dashed Lines)
                last_res = df['Pivot_Resist'].iloc[-1]
                last_sup = df['Pivot_Support'].iloc[-1]
                if not pd.isna(last_res):
                    fig.add_hline(y=last_res, line_dash="dash", line_color="red", annotation_text="Resistance", row=1, col=1)
                if not pd.isna(last_sup):
                    fig.add_hline(y=last_sup, line_dash="dash", line_color="green", annotation_text="Support", row=1, col=1)
                
                # 2. Money Flow & Squeeze (Bottom Chart)
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
                
                # --- AI STRATEGY & NEWS ---
                st.markdown("### 🤖 Strategy Briefing")
                col_ai, col_news = st.columns([2, 1])
                
                with col_ai:
                    verdict = ask_ai_analyst(df, ticker, news, balance, risk_pct)
                    st.info(verdict)
                    
                with col_news:
                    st.caption("📰 Live Sentiment Wire")
                    if news:
                        for n in news:
                            st.write(f"• [{n['title']}]({n['link']})")
                    else:
                        st.write("No recent news found.")
                    
            else:
                st.error("Data Error. Yahoo Finance might be blocking requests. Try again in 1 min.")
