import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import calendar
import datetime
import requests
import urllib.parse
import math

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="DarkPool Titan Terminal v6", page_icon="👁️")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .title-glow {
        font-size: 3em; font-weight: bold; color: #ffffff;
        text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 40px #00ff00;
        margin-bottom: 20px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px; border-radius: 8px; transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover { transform: scale(1.02); border-color: #00ff00; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #161b22;
        border-radius: 4px 4px 0px 0px; border: 1px solid #30363d; color: #8b949e;
    }
    .stTabs [aria-selected="true"] { background-color: #0e1117; color: #00ff00; border-bottom: 2px solid #00ff00; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-glow">👁️ DarkPool Titan v6</div>', unsafe_allow_html=True)
st.markdown("##### *Institutional-Grade Market Intelligence // Full Spectrum Analysis*")
st.markdown("---")

# --- API KEY ---
if 'api_key' not in st.session_state: st.session_state.api_key = None
if "OPENAI_API_KEY" in st.secrets: st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
else:
    if not st.session_state.api_key:
        st.session_state.api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# ==========================================
# 2. DATA ENGINE (OPTIMIZED)
# ==========================================
@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    if any(x in ticker for x in ["-", "=", "^"]): return None
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info: return None
        return {
            "Market Cap": info.get("marketCap", 0), "P/E Ratio": info.get("trailingPE", 0),
            "Rev Growth": info.get("revenueGrowth", 0), "Debt/Equity": info.get("debtToEquity", 0),
            "Summary": info.get("longBusinessSummary", "No Data Available")
        }
    except: return None

@st.cache_data(ttl=300)
def get_global_performance():
    assets = { "Tech": "XLK", "Energy": "XLE", "Financials": "XLF", "Bitcoin": "BTC-USD", "Gold": "GLD", "Oil": "USO", "Bonds": "TLT" }
    try:
        data = yf.download(list(assets.values()), period="5d", interval="1d", progress=False)['Close']
        res = {}
        for n, t in assets.items():
            if t in data.columns:
                res[n] = ((data[t].iloc[-1] - data[t].iloc[-2])/data[t].iloc[-2])*100
        return pd.Series(res).sort_values()
    except: return None

def safe_download(ticker, period, interval):
    # Strict limits for speed optimization
    if interval == "1m": period = "5d"
    elif interval == "5m": period = "1mo"
    elif interval in ["15m", "30m"]: period = "1mo"
    elif interval == "1h": period = "1y"
    
    try:
        dl_int = "1h" if interval == "4h" else interval
        df = yf.download(ticker, period=period, interval=dl_int, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return None
        if 'Close' not in df.columns and 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
        return df
    except: return None

@st.cache_data(ttl=300)
def get_macro_data():
    groups = {
        "🇺🇸 US Equities": {"S&P 500": "SPY", "Nasdaq": "QQQ"},
        "⚠️ Risk Assets": {"Bitcoin": "BTC-USD", "VIX": "^VIX"},
        "🏦 Rates": {"10Y Yield": "^TNX", "DXY": "DX-Y.NYB"},
        "🥇 Metals": {"Gold": "GC=F", "Silver": "SI=F"}
    }
    tickers = [t for g in groups.values() for t in g.values()]
    try:
        data = yf.download(tickers, period="5d", interval="1d", progress=False)['Close']
        p, c = {}, {}
        for t in tickers:
            if t in data.columns:
                p[t] = data[t].iloc[-1]
                c[t] = ((data[t].iloc[-1]-data[t].iloc[-2])/data[t].iloc[-2])*100
        return groups, p, c
    except: return groups, {}, {}

# ==========================================
# 3. MATH & ALGORITHMS (ALL 5 NEW ENGINES)
# ==========================================
def calc_linreg_slope(series, window):
    y = series.values
    x = np.arange(window)
    sum_x, sum_x_sq = np.sum(x), np.sum(x**2)
    div = window * sum_x_sq - sum_x**2
    res = np.full(len(series), np.nan)
    for i in range(window, len(series)):
        y_w = y[i-window:i]
        res[i] = (window * np.sum(x*y_w) - sum_x * np.sum(y_w)) / div
    return pd.Series(res, index=series.index)

def calc_hma(series, length):
    half, sqrt_l = int(length/2), int(np.sqrt(length))
    wma_half = series.rolling(half).apply(lambda x: np.dot(x, np.arange(1, half+1))/np.arange(1, half+1).sum(), raw=True)
    wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1))/np.arange(1, length+1).sum(), raw=True)
    raw = 2 * wma_half - wma_full
    return raw.rolling(sqrt_l).apply(lambda x: np.dot(x, np.arange(1, sqrt_l+1))/np.arange(1, sqrt_l+1).sum(), raw=True)

def double_smooth(src, long, short):
    return src.ewm(span=long).mean().ewm(span=short).mean()

def calc_titan_indicators_v6(df):
    """
    COMPREHENSIVE CALCULATION ENGINE
    Includes: Squeeze Pro, EVWM, Money Flow, Ultimate S&R, Apex Trend, SMC
    """
    # 1. Classics (Apex Trend, Vector, RVOL, ATR, RSI)
    df['HMA'] = calc_hma(df['Close'], 55)
    df['Apex_Trend'] = np.where(df['Close'] > df['HMA'], 'BULLISH', 'BEARISH')
    
    ha_close = (df['Open']+df['High']+df['Low']+df['Close'])/4
    ha_open = (df['Open'].shift(1)+df['Close'].shift(1))/2
    df['Vector_Color'] = np.where(ha_close > ha_open, 'GREEN', 'RED')
    
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['RVOL'] = df['Volume'] / df['Vol_SMA']
    
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    delta = df['Close'].diff()
    df['RSI'] = 100 - (100 / (1 + (delta.where(delta>0,0).rolling(14).mean() / -delta.where(delta<0,0).rolling(14).mean())))

    # 2. Ultimate S&R (Cluster Logic)
    period = 10
    pivot_h = df['High'].rolling(period*2+1, center=True).max() == df['High']
    pivot_l = df['Low'].rolling(period*2+1, center=True).min() == df['Low']
    
    pivots = df[pivot_h]['High'].tolist() + df[pivot_l]['Low'].tolist()
    sr_levels = []
    if pivots:
        pivots.sort()
        curr = [pivots[0]]
        thresh = df['Close'].iloc[-1] * 0.005
        for p in pivots[1:]:
            if p - curr[-1] < thresh: curr.append(p)
            else:
                sr_levels.append(sum(curr)/len(curr))
                curr = [p]
        sr_levels.append(sum(curr)/len(curr))

    # 3. EVWM (Elastic Volume Weighted Momentum)
    hull = calc_hma(df['Close'], 21)
    elasticity = (df['Close'] - hull) / df['ATR']
    force = np.sqrt((df['Volume']/df['Volume'].rolling(21).mean()).rolling(5).mean())
    df['EVWM'] = elasticity * force
    std = df['EVWM'].rolling(42).std()
    df['EVWM_Upper'] = df['EVWM'].rolling(42).mean() + (2*std)
    df['EVWM_Lower'] = df['EVWM'].rolling(42).mean() - (2*std)

    # 4. Squeeze Pro
    basis = df['Close'].rolling(20).mean()
    bb_dev = df['Close'].rolling(20).std()
    df['SQZ_On'] = (basis - 2*bb_dev) > (basis - 1.5*df['ATR']) # BB inside KC
    df['SQZ_Mom'] = calc_linreg_slope(df['Close'] - basis, 20)

    # 5. Money Flow Matrix
    mf = ((df['RSI']-50) * (df['Volume']/df['Volume'].rolling(14).mean())).ewm(span=3).mean()
    df['MF_Matrix'] = mf
    pc = df['Close'].diff()
    df['Hyper_Wave'] = (100 * (double_smooth(pc,25,13) / double_smooth(abs(pc),25,13))) / 2
    
    return df, sr_levels[-10:] # Keep last 10 levels for chart clarity

def calc_fear_greed_v4(df):
    df['FG_RSI'] = df['RSI']
    trend = np.where(df['Close'] > df['Close'].rolling(50).mean(), 60, 40)
    df['FG_Index'] = (df['RSI']*0.6) + (trend*0.4)
    return df

# --- RESTORED FULL FUNCTIONS (NO PLACEHOLDERS) ---
def calculate_smc(df, swing_length=5):
    """Full Smart Money Concepts Logic"""
    smc = {'structures': [], 'order_blocks': [], 'fvgs': []}
    
    # FVG
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            smc['fvgs'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'color': 'rgba(0,255,100,0.3)'})
        if df['High'].iloc[i] < df['Low'].iloc[i-2]:
            smc['fvgs'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['Low'].iloc[i-2], 'y1': df['High'].iloc[i], 'color': 'rgba(255,0,0,0.3)'})

    # Structure & OB
    ph = df['High'].rolling(swing_length*2+1, center=True).max() == df['High']
    pl = df['Low'].rolling(swing_length*2+1, center=True).min() == df['Low']
    
    last_h, last_l = None, None
    trend = 0
    
    for i in range(swing_length, len(df)):
        if ph.iloc[i-swing_length]: last_h = {'p': df['High'].iloc[i-swing_length], 'i': i-swing_length, 'idx': df.index[i-swing_length]}
        if pl.iloc[i-swing_length]: last_l = {'p': df['Low'].iloc[i-swing_length], 'i': i-swing_length, 'idx': df.index[i-swing_length]}
        
        curr = df['Close'].iloc[i]
        if last_h and curr > last_h['p']:
            lbl = "BOS" if trend == 1 else "CHoCH"
            trend = 1
            smc['structures'].append({'x0': last_h['idx'], 'x1': df.index[i], 'y': last_h['p'], 'c': 'green', 'l': lbl})
            if last_l:
                subset = df.iloc[last_l['i']:i]
                ob_idx = subset['Low'].idxmin()
                smc['order_blocks'].append({'x0': ob_idx, 'x1': df.index[-1], 'y0': df['Low'].loc[ob_idx], 'y1': df['High'].loc[ob_idx], 'c': 'rgba(0,255,0,0.2)'})
            last_h = None
            
        if last_l and curr < last_l['p']:
            lbl = "BOS" if trend == -1 else "CHoCH"
            trend = -1
            smc['structures'].append({'x0': last_l['idx'], 'x1': df.index[i], 'y': last_l['p'], 'c': 'red', 'l': lbl})
            if last_h:
                subset = df.iloc[last_h['i']:i]
                ob_idx = subset['High'].idxmax()
                smc['order_blocks'].append({'x0': ob_idx, 'x1': df.index[-1], 'y0': df['Low'].loc[ob_idx], 'y1': df['High'].loc[ob_idx], 'c': 'rgba(255,0,0,0.2)'})
            last_l = None
            
    return smc

def calc_mtf_trend(ticker):
    """Full MTF Trend Radar"""
    res = {}
    tf_map = {"1H": "1h", "4H": "1h", "Daily": "1d", "Weekly": "1wk"}
    for name, interval in tf_map.items():
        try:
            d = yf.download(ticker, period="1y", interval=interval, progress=False)
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            
            if name == "4H": # Manual Resample
                d = d.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            
            d['EMA20'] = d['Close'].ewm(span=20).mean()
            d['EMA50'] = d['Close'].ewm(span=50).mean()
            last = d.iloc[-1]
            
            trend = "BULL" if last['Close'] > last['EMA20'] > last['EMA50'] else "BEAR" if last['Close'] < last['EMA20'] < last['EMA50'] else "NEUTRAL"
            res[name] = {"Trend": trend, "Price": f"{last['Close']:.2f}"}
        except: res[name] = {"Trend": "N/A", "Price": "0.00"}
    return pd.DataFrame(res).T

def calc_correlations(ticker):
    """Full Cross-Asset Correlation"""
    targets = {"SPY": "SPY", "BTC": "BTC-USD", "10Y": "^TNX", "DXY": "DX-Y.NYB", "GOLD": "GC=F"}
    d = yf.download(list(targets.values()) + [ticker], period="6mo", interval="1d", progress=False)['Close']
    corr = d.corr()[ticker].drop(ticker, errors='ignore').sort_values(ascending=False)
    inv = {v: k for k, v in targets.items()}
    corr.index = [inv.get(i, i) for i in corr.index]
    return corr

def calc_volume_profile(df, bins=50):
    price_bins = np.linspace(df['Low'].min(), df['High'].max(), bins)
    df['Bin'] = pd.cut(df['Close'], bins=price_bins, include_lowest=True)
    vp = df.groupby('Bin')['Volume'].sum().reset_index()
    vp['Price'] = [i.mid for i in vp['Bin']]
    return vp, vp.loc[vp['Volume'].idxmax(), 'Price']

def run_monte_carlo(df, days=30, sims=200):
    rets = df['Close'].pct_change().dropna()
    paths = np.zeros((days, sims))
    paths[0] = df['Close'].iloc[-1]
    shocks = np.random.normal(rets.mean(), rets.std(), (days, sims))
    for t in range(1, days):
        paths[t] = paths[t-1] * (1 + shocks[t])
    return paths

# ==========================================
# 4. AI ANALYST (COMPLETE PROMPT)
# ==========================================
def ask_ai_analyst(df, ticker, balance, risk_pct, interval):
    if not st.session_state.api_key: return "⚠️ Waiting for OpenAI API Key..."
    last = df.iloc[-1]
    
    # State Logic
    evwm_state = "IGNITION" if last['EVWM'] > last['EVWM_Upper'] else "NEUTRAL"
    sqz_state = "FIRING" if (not last['SQZ_On'] and df['SQZ_On'].iloc[-2]) else "SQUEEZING" if last['SQZ_On'] else "NORMAL"
    
    prompt = f"""
    Analyze {ticker} ({interval}) at ${last['Close']:.2f}.
    
    --- DATA PACKET ---
    1. Trend: {last['Apex_Trend']} (HMA)
    2. Momentum (EVWM): {last['EVWM']:.2f} [{evwm_state}]
    3. Volatility (Squeeze): {last['SQZ_Mom']:.4f} [{sqz_state}]
    4. Money Flow: {last['MF_Matrix']:.2f}
    5. Hyper Wave: {last['Hyper_Wave']:.1f}
    6. RSI: {last['RSI']:.1f} | ATR: {last['ATR']:.2f}
    
    Account: ${balance}. Risk: {risk_pct}%.
    
    TASK: Provide a professional trading verdict (BUY/SELL/WAIT).
    Cite specific indicators (EVWM, Squeeze, Money Flow) in your reasoning.
    Calculated Position Size for 1% Risk.
    """
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        return client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}], max_tokens=800).choices[0].message.content
    except Exception as e: return str(e)

# ==========================================
# 5. UI LAYOUT
# ==========================================
st.sidebar.header("🎛️ Terminal Controls")

# Broadcaster Inputs
st.sidebar.subheader("📢 Broadcaster")
if 'tg_token' not in st.session_state: st.session_state.tg_token = ""
if 'tg_chat' not in st.session_state: st.session_state.tg_chat = ""
if "TELEGRAM_TOKEN" in st.secrets: st.session_state.tg_token = st.secrets["TELEGRAM_TOKEN"]
if "TELEGRAM_CHAT_ID" in st.secrets: st.session_state.tg_chat = st.secrets["TELEGRAM_CHAT_ID"]

tg_token = st.sidebar.text_input("Bot Token", value=st.session_state.tg_token, type="password")
tg_chat = st.sidebar.text_input("Chat ID", value=st.session_state.tg_chat)

# Asset Selection
mode = st.sidebar.radio("Input", ["Lists", "Search"], index=1)
if mode == "Lists":
    assets = { "Indices": ["SPY", "QQQ"], "Crypto": ["BTC-USD", "ETH-USD"], "Macro": ["^TNX"] }
    cat = st.sidebar.selectbox("Category", list(assets.keys()))
    ticker = st.sidebar.selectbox("Ticker", assets[cat])
else:
    ticker = st.sidebar.text_input("Ticker", "BTC-USD").upper()

interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=4)
st.sidebar.markdown("---")
balance = st.sidebar.number_input("Balance", 1000, 1000000, 10000)
risk = st.sidebar.slider("Risk %", 0.5, 3.0, 1.0)

# Macro Header
mg, mp, mc = get_macro_data()
if mp:
    cols = st.columns(4)
    cols[0].metric("SPY", f"{mp.get('SPY',0):.2f}", f"{mc.get('SPY',0):.2f}%")
    cols[1].metric("BTC", f"{mp.get('BTC-USD',0):.2f}", f"{mc.get('BTC-USD',0):.2f}%")
    cols[2].metric("10Y", f"{mp.get('^TNX',0):.2f}", f"{mc.get('^TNX',0):.2f}%")
    cols[3].metric("Gold", f"{mp.get('Gold',0):.1f}", f"{mc.get('Gold',0):.2f}%")
st.markdown("---")

# Main Logic
if st.button("🚀 Analyze"): st.session_state.run = True

if st.session_state.get('run'):
    with st.spinner("Processing..."):
        df = safe_download(ticker, "1y", interval)
        if interval == "4h" and df is not None:
             df = df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        
        if df is not None:
            df, sr_levels = calc_titan_indicators_v6(df)
            df = calc_fear_greed_v4(df)
            last = df.iloc[-1]
            ai_text = ask_ai_analyst(df, ticker, balance, risk, interval)

            tabs = st.tabs(["Chart", "Squeeze", "Money Flow", "AI", "SMC", "MTF", "Monte Carlo", "Profile", "Broadcast"])

            # 1. Main Chart
            with tabs[0]:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='orange', width=2), name='Apex Trend'), row=1, col=1)
                for l in sr_levels: fig.add_hline(y=l, line_dash='dot', line_color='rgba(255,255,255,0.3)', row=1, col=1)
                
                cols = np.where(df['EVWM']>0, '#00ffaa', '#ff0062')
                fig.add_trace(go.Bar(x=df.index, y=df['EVWM'], marker_color=cols, name='EVWM Momentum'), row=2, col=1)
                fig.update_layout(height=700, template='plotly_dark', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            # 2. Squeeze
            with tabs[1]:
                fig = go.Figure()
                c = np.where(df['SQZ_Mom']>0, 'green', 'red')
                fig.add_trace(go.Bar(x=df.index, y=df['SQZ_Mom'], marker_color=c, name='Momentum'))
                fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='markers', marker=dict(color=np.where(df['SQZ_On'],'red','green'), size=5), name='Squeeze State'))
                fig.update_layout(template='plotly_dark', title="Squeeze Pro")
                st.plotly_chart(fig, use_container_width=True)

            # 3. Money Flow
            with tabs[2]:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
                fig.add_trace(go.Bar(x=df.index, y=df['MF_Matrix'], marker_color=np.where(df['MF_Matrix']>0,'lime','red'), name='MF Matrix'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Hyper_Wave'], line=dict(color='cyan'), name='Hyper Wave'), row=2, col=1)
                fig.update_layout(template='plotly_dark', height=600)
                st.plotly_chart(fig, use_container_width=True)

            # 4. AI
            with tabs[3]: st.info(ai_text)

            # 5. SMC
            with tabs[4]:
                smc = calculate_smc(df)
                fig = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                for s in smc['structures']: 
                    fig.add_shape(type="line", x0=s['x0'], x1=s['x1'], y0=s['y'], y1=s['y'], line=dict(color=s['c'], dash='dot'))
                for ob in smc['order_blocks']:
                    fig.add_shape(type="rect", x0=ob['x0'], x1=ob['x1'], y0=ob['y0'], y1=ob['y1'], fillcolor=ob['c'], line_width=0)
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

            # 6. MTF
            with tabs[5]:
                mtf = calc_mtf_trend(ticker)
                corr = calc_correlations(ticker)
                c1, c2 = st.columns(2)
                c1.dataframe(mtf)
                c2.bar_chart(corr)

            # 7. Monte Carlo
            with tabs[6]:
                mc = run_monte_carlo(df)
                fig = go.Figure()
                for i in range(50): fig.add_trace(go.Scatter(y=mc[:,i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

            # 8. Profile
            with tabs[7]:
                vp, poc = calc_volume_profile(df)
                fig = go.Figure(go.Bar(x=vp['Volume'], y=vp['Price'], orientation='h'))
                fig.add_hline(y=poc, line_color='yellow')
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

            # 9. Broadcast (COMPREHENSIVE)
            with tabs[8]:
                st.subheader("📡 Pro Broadcast")
                
                # Auto-Generated Report with ALL Indicators
                sqz_txt = "FIRING" if (not last['SQZ_On'] and df['SQZ_On'].iloc[-2]) else "SQUEEZING" if last['SQZ_On'] else "NEUTRAL"
                
                full_report = f"""🔥 {ticker} ({interval}) INTELLIGENCE REPORT
💰 Price: ${last['Close']:.2f}
🌊 Squeeze Status: {sqz_txt} (Mom: {last['SQZ_Mom']:.4f})
⚡ EVWM Momentum: {last['EVWM']:.2f}
💸 Money Flow: {last['MF_Matrix']:.2f}
🎯 Apex Trend: {last['Apex_Trend']}

🤖 AI STRATEGY:
{ai_text}

#DarkPoolTitan #Trading #{ticker}"""

                msg_input = st.text_area("Final Message Draft", full_report, height=300)
                img = st.file_uploader("Attach Chart Screenshot", type=['png', 'jpg'])
                
                c1, c2 = st.columns(2)
                
                if c1.button("🚀 Send to Telegram"):
                    if tg_token and tg_chat:
                        try:
                            # 1. Send Photo
                            if img:
                                requests.post(
                                    f"https://api.telegram.org/bot{tg_token}/sendPhoto",
                                    data={'chat_id': tg_chat, 'caption': f"📊 {ticker} Chart"},
                                    files={'photo': img.getvalue()}
                                )
                            
                            # 2. Send Text (Chunked Safe Send)
                            limit = 2000
                            for i in range(0, len(msg_input), limit):
                                chunk = msg_input[i:i+limit]
                                requests.post(
                                    f"https://api.telegram.org/bot{tg_token}/sendMessage",
                                    data={'chat_id': tg_chat, 'text': chunk}
                                )
                            st.success("Broadcast Sent Successfully!")
                        except Exception as e:
                            st.error(f"Telegram Error: {e}")
                    else:
                        st.warning("Please enter Telegram Bot Token & Chat ID in the Sidebar.")

                if c2.button("🐦 Tweet (X)"):
                    safe_txt = urllib.parse.quote(msg_input[:280]) # Twitter limit warning
                    st.link_button("Launch Tweet", f"https://twitter.com/intent/tweet?text={safe_txt}")

        else: st.error("No Data Found. Try a standard ticker like 'BTC-USD'.")
