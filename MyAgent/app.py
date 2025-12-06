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
st.set_page_config(layout="wide", page_title="DarkPool Titan Global")
st.title("👁️ DarkPool Titan Global")
st.markdown("### The Ultimate Institutional Terminal")

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# ==========================================
# 2. DATA ENGINE (GLOBAL)
# ==========================================
@st.cache_data(ttl=300)
def get_macro_data():
    """Fetches Global Indices for the War Room"""
    tickers = {
        "S&P 500": "SPY", 
        "Nasdaq": "QQQ", 
        "Bitcoin": "BTC-USD", 
        "10Y Yield": "^TNX", 
        "VIX": "^VIX",
        "FTSE 100 (UK)": "^FTSE", 
        "Nikkei 225 (JP)": "^N225", 
        "DAX (DE)": "^GDAXI"
    }
    try:
        data = yf.download(list(tickers.values()), period="5d", interval="1d", progress=False)['Close']
        # Calculate daily change
        changes = data.pct_change().iloc[-1] * 100
        prices = data.iloc[-1]
        return prices, changes, tickers
    except: return None, None, None

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    """Robust Fundamental Fetcher"""
    if "-" in ticker or "=" in ticker or "^" in ticker: return None 
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        currency = info.get('currency', 'USD')
        return {
            "Market Cap": info.get("marketCap", 0),
            "P/E Ratio": info.get("trailingPE", 0),
            "PEG Ratio": info.get("pegRatio", 0),
            "Rev Growth": info.get("revenueGrowth", 0),
            "Debt/Equity": info.get("debtToEquity", 0),
            "Target Price": info.get("targetMeanPrice", 0),
            "Currency": currency,
            "Summary": info.get("longBusinessSummary", "No Data")
        }
    except: return None

def get_news(ticker):
    try:
        if "=" in ticker or "^" in ticker: return []
        stock = yf.Ticker(ticker)
        return stock.news[:5]
    except: return []

def safe_download(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.droplevel(1) 
            except: pass 
        if df.empty or 'Close' not in df.columns: return None
        return df
    except: return None

# ==========================================
# 3. MATH LIBRARY (INDICATORS)
# ==========================================
def calc_indicators(df):
    # 1. Apex Trend (HMA)
    df['HMA'] = df['Close'].rolling(55).mean()
    
    # 2. ATR (Volatility)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # 3. Auto Support/Resistance (Pivots)
    df['Pivot_Resist'] = df['High'].rolling(20).max()
    df['Pivot_Support'] = df['Low'].rolling(20).min()
    
    # 4. Money Flow Matrix
    df['MFI'] = (df['Close'].diff() * df['Volume']).rolling(3).mean()
    
    # 5. Squeeze Pro
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['KC_ATR'] = df['ATR'].rolling(20).mean()
    df['Sq_On'] = (df['BB_Mid'] + 2*df['BB_Std']) < (df['BB_Mid'] + 1.5*df['KC_ATR'])
    df['Mom'] = df['Close'] - df['Close'].rolling(20).mean()
    
    return df

# ==========================================
# 4. AI TITAN BRAIN (GLOBAL CONTEXT)
# ==========================================
def ask_ai_analyst(df, ticker, fundamentals, news, balance, risk_pct):
    if not api_key: return "⚠️ Waiting for API Key..."
    
    last = df.iloc[-1]
    
    # Technicals
    trend = "BULLISH" if last['Close'] > last['HMA'] else "BEARISH"
    squeeze = "FIRING" if last['Sq_On'] else "OFF"
    
    # Fundamentals
    fund_text = "N/A"
    if fundamentals:
        curr = fundamentals['Currency']
        fund_text = f"Valuation: {fundamentals['P/E Ratio']:.2f} P/E. Debt/Eq: {fundamentals['Debt/Equity']}. Currency: {curr}."
    
    # News
    news_text = "\n".join([f"- {n['title']}" for n in news]) if news else "No major news."
    
    # Risk
    risk_dollars = balance * (risk_pct / 100)
    stop_level = last['Pivot_Support'] if trend == "BULLISH" else last['Pivot_Resist']
    if pd.isna(stop_level): stop_level = last['Close'] * 0.95
    
    dist = abs(last['Close'] - stop_level)
    shares = risk_dollars / dist if dist > 0 else 0
    
    prompt = f"""
    Act as a Global Macro Strategist. Analyze {ticker} at {last['Close']:.2f}.
    
    --- 1. FUNDAMENTALS (Global Context) ---
    {fund_text}
    
    --- 2. TECHNICALS (Timing) ---
    Trend: {trend}. Momentum: {last['Mom']:.2f}. Squeeze: {squeeze}.
    
    --- 3. NEWS ---
    {news_text}
    
    --- 4. RISK MANAGEMENT ---
    Capital: ${balance}. Risk Budget: ${risk_dollars:.2f} ({risk_pct}%).
    Stop Loss: {stop_level:.2f}. Position Size: {shares:.4f} units.
    
    --- MISSION ---
    1. **Verdict:** STRONG BUY, ACCUMULATE, WAIT, SELL.
    2. **Analysis:** Combine chart trend with fundamental quality.
    3. **Plan:** Entry, Stop, Target (2R).
    """
    
    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
    return res.choices[0].message.content

# ==========================================
# 5. UI DASHBOARD
# ==========================================
# --- SIDEBAR: ASSET SELECTION ---
st.sidebar.header("🎛️ Terminal Controls")

# Input Mode Switcher
input_mode = st.sidebar.radio("Input Mode:", ["Curated Lists", "Manual Search (Global)"])

if input_mode == "Curated Lists":
    assets = {
        "Indices": ["SPY", "QQQ", "IWM", "^VIX"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
        "Commodities": ["GC=F", "CL=F", "HG=F", "NG=F"],
        "Tech": ["NVDA", "TSLA", "AAPL", "MSFT", "AMD"],
        "Macro": ["^TNX", "DX-Y.NYB", "TLT", "HYG"]
    }
    cat = st.sidebar.selectbox("Category", list(assets.keys()))
    ticker = st.sidebar.selectbox("Ticker", assets[cat])
else:
    st.sidebar.info("Type ANY ticker (e.g. SHEL.L, 7203.T)")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()

interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=3)

# Global Suffix Guide (Sidebar)
with st.sidebar.expander("🌍 International Suffix Guide"):
    st.markdown("""
    * **USA:** No suffix (e.g., AAPL, TSLA)
    * **London:** .L (e.g., SHEL.L, RR.L)
    * **Europe:** .PA (Paris), .DE (Frankfurt), .AS (Amsterdam)
    * **Tokyo:** .T (e.g., 7203.T for Toyota)
    * **Hong Kong:** .HK (e.g., 0700.HK)
    * **Toronto:** .TO (e.g., SHOP.TO)
    * **Sydney:** .AX (e.g., BHP.AX)
    * **Crypto:** -USD (e.g., PEPE-USD)
    """)

st.sidebar.markdown("---")
st.sidebar.header("💰 Risk Parameters")
balance = st.sidebar.number_input("Capital ($)", 1000, 1000000, 10000)
risk_pct = st.sidebar.slider("Risk %", 0.5, 3.0, 1.0)

# --- GLOBAL MACRO HEADER ---
st.subheader("🌍 Global Market Regime")
m_price, m_chg, m_names = get_macro_data()

if m_price is not None:
    # Logic: If SPY is Up AND VIX is Down = Risk On
    spy_up = m_chg.get("SPY", 0) > 0
    vix_down = m_chg.get("^VIX", 0) < 0
    regime = "RISK ON 🚀" if spy_up and vix_down else "RISK OFF 🛡️"
    
    st.info(f"**Global Status:** {regime} | **Nikkei (Japan):** {m_chg.get('^N225',0):.2f}% | **FTSE (UK):** {m_chg.get('^FTSE',0):.2f}%")
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def show_metric(col, label, key):
        val = m_price.get(key, 0)
        chg = m_chg.get(key, 0)
        col.metric(label, f"{val:,.2f}", f"{chg:.2f}%")

    show_metric(c1, "S&P 500", "SPY")
    show_metric(c2, "Nasdaq", "QQQ")
    show_metric(c3, "Bitcoin", "BTC-USD")
    show_metric(c4, "10Y Yield", "^TNX")
    show_metric(c5, "FTSE 100", "^FTSE")
    show_metric(c6, "VIX", "^VIX")
    st.markdown("---")

# --- MAIN ANALYSIS TABS ---
tab1, tab2 = st.tabs(["📊 Technical Deep Dive", "🏢 Fundamentals & News"])

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
                
                # Chart
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='orange', width=2), name="Apex Trend"), row=1, col=1)
                
                # Pivot Lines
                last_res = df['Pivot_Resist'].iloc[-1]
                last_sup = df['Pivot_Support'].iloc[-1]
                if not pd.isna(last_res): fig.add_hline(y=last_res, line_dash="dash", line_color="red", row=1, col=1)
                if not pd.isna(last_sup): fig.add_hline(y=last_sup, line_dash="dash", line_color="green", row=1, col=1)
                
                # Money Flow
                colors = ['#00ff00' if v > 0 else '#ff0000' for v in df['MFI']]
                fig.add_trace(go.Bar(x=df.index, y=df['MFI'], marker_color=colors, name="Money Flow"), row=2, col=1)
                
                fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # AI Verdict
                st.markdown("### 🤖 Titan Strategy Verdict")
                verdict = ask_ai_analyst(df, ticker, fund, news, balance, risk_pct)
                st.success(verdict)

            # --- TAB 2: FUNDAMENTALS ---
            with tab2:
                if fund:
                    st.subheader(f"🏢 Fundamental Health ({fund.get('Currency', 'USD')})")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Market Cap", f"{fund['Market Cap']:,}")
                    c2.metric("P/E Ratio", f"{fund['P/E Ratio']}")
                    c3.metric("Rev Growth", f"{fund['Rev Growth']*100:.1f}%")
                    st.write(f"**Business Summary:** {fund['Summary'][:500]}...")
                else:
                    st.warning("Fundamentals N/A for this asset class.")
                
                st.markdown("---")
                st.subheader("📰 Global News Wire")
                for n in news:
                    st.write(f"**{n['title']}**")
                    st.caption(f"Source: {n['publisher']} | [Read]({n['link']})")
                    
        else:
            st.error(f"Could not find ticker '{ticker}'. Check the suffix guide in the sidebar.")
