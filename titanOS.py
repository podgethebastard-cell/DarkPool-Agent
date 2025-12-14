import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import yfinance as yf
import ccxt
import requests
import time
import datetime
import math
from scipy.stats import linregress
from datetime import timedelta
import logging
import io
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict

# Suppress Warnings
warnings.filterwarnings('ignore')

# GLOBAL AI LOADER
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# =============================================================================
# 1. GLOBAL SYSTEM CONFIGURATION & DARKPOOL CSS
# =============================================================================
st.set_page_config(
    page_title="Titan OS | Financial Singularity",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# THE TITAN DARKPOOL AESTHETIC
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;800&display=swap');
    
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    
    /* TITAN HEADER GLOW */
    .titan-header {
        background: linear-gradient(180deg, rgba(16,16,20,1) 0%, rgba(5,5,5,0) 100%);
        border-top: 3px solid #00ffbb;
        padding: 2rem 0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .titan-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00ffbb, #7d00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 255, 187, 0.2);
        margin: 0;
    }
    .titan-subtitle {
        font-family: 'Roboto Mono', monospace;
        color: #666;
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* METRIC CARDS */
    div[data-testid="metric-container"] {
        background: #0f0f0f;
        border: 1px solid #222;
        border-left: 4px solid #333;
        padding: 15px;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-left-color: #00ffbb;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetricLabel"] { font-family: 'Roboto Mono', monospace; font-size: 0.8rem; color: #888; }
    div[data-testid="stMetricValue"] { font-family: 'Inter', sans-serif; font-weight: 800; font-size: 1.6rem; color: #fff; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: #111;
        border-radius: 4px;
        color: #666;
        border: 1px solid #222;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ffbb !important;
        color: #000 !important;
        border: 1px solid #00ffbb;
        font-weight: bold;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #080808; border-right: 1px solid #222; }
    
    /* BUTTONS */
    div.stButton > button {
        background: linear-gradient(135deg, #1f2833, #0b0c10);
        border: 1px solid #333;
        color: #00ffbb;
        font-family: 'Roboto Mono', monospace;
        font-weight: bold;
        transition: all 0.2s;
        border-radius: 6px;
        height: 3em;
        width: 100%;
    }
    div.stButton > button:hover {
        border-color: #00ffbb;
        box-shadow: 0 0 10px rgba(0, 255, 187, 0.2);
        color: #fff;
    }

    /* ALERTS & AI BOX */
    .titan-alert { padding: 10px; border-radius: 6px; margin-bottom: 10px; font-family: 'Roboto Mono', monospace; font-size: 0.9rem; text-align: center; font-weight: bold; }
    .alert-bull { background: rgba(0, 255, 187, 0.1); border: 1px solid #00ffbb; color: #00ffbb; }
    .alert-bear { background: rgba(255, 17, 85, 0.1); border: 1px solid #ff1155; color: #ff1155; }
    .ai-box { background: #0a0a0a; border: 1px solid #333; padding: 20px; border-radius: 5px; margin-top: 20px; border-left: 3px solid #7d00ff; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CORE LOGIC ENGINE (TitanMath)
# =============================================================================
class TitanMath:
    @staticmethod
    def rma(series, period):
        return series.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return TitanMath.rma(tr, length)

    @staticmethod
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def hma(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma_half = TitanMath.wma(series, half_length)
        wma_full = TitanMath.wma(series, length)
        diff = 2 * wma_half - wma_full
        return TitanMath.wma(diff, sqrt_length)

    @staticmethod
    def rsi(series, length=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(series, length=20, std_dev=2.0):
        sma = series.rolling(window=length).mean()
        std = series.rolling(window=length).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def smc_detect(df, lookback=5):
        df['Pivot_High'] = df['High'].rolling(window=lookback*2+1, center=True).max() == df['High']
        df['Pivot_Low'] = df['Low'].rolling(window=lookback*2+1, center=True).min() == df['Low']
        
        # FVG Detection
        fvg_bull = (df['Low'] > df['High'].shift(2))
        fvg_bear = (df['High'] < df['Low'].shift(2))
        return fvg_bull, fvg_bear

    @staticmethod
    def squeeze_momentum(df):
        upper, basis, lower = TitanMath.bollinger_bands(df['Close'], 20, 2.0)
        tr = TitanMath.atr(df, 20)
        k_upper = basis + (tr * 1.5)
        k_lower = basis - (tr * 1.5)
        
        squeeze_on = (lower > k_lower) & (upper < k_upper)
        
        x = np.arange(20)
        # Vectorized LinReg approach for Momentum
        mom = df['Close'].rolling(20).apply(lambda y: linregress(x, y - y.mean())[0] if len(y)==20 else 0, raw=True)
        return squeeze_on, mom

    @staticmethod
    def money_flow(df, length=14):
        rsi_src = TitanMath.rsi(df['Close'], length) - 50
        mf_vol = df['Volume'] / df['Volume'].rolling(length).mean()
        return (rsi_src * mf_vol).fillna(0).ewm(span=3).mean()

    @staticmethod
    def evwm(df, length=21, vol_smooth=5):
        baseline = TitanMath.hma(df['Close'], length)
        atr = TitanMath.atr(df, length)
        elasticity = (df['Close'] - baseline) / atr
        rvol = df['Volume'] / df['Volume'].rolling(length).mean()
        force = np.sqrt(rvol.rolling(vol_smooth).mean())
        return elasticity * force

    @staticmethod
    def calculate_all_indicators(df):
        if df.empty or len(df) < 55:
            return df
            
        # Ensure Numeric Data
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)

        df['HMA'] = TitanMath.hma(df['Close'], 55)
        df['ATR'] = TitanMath.atr(df, 14)
        df['RSI'] = TitanMath.rsi(df['Close'], 14)
        df['Money_Flow'] = TitanMath.money_flow(df)
        df['Squeeze'], df['Mom'] = TitanMath.squeeze_momentum(df)
        df['EVWM'] = TitanMath.evwm(df)
        df['FVG_Bull'], df['FVG_Bear'] = TitanMath.smc_detect(df)
        
        # Apex Trend Logic (HMA + ATR Bands)
        df['Upper'] = df['HMA'] + (df['ATR'] * 1.5)
        df['Lower'] = df['HMA'] - (df['ATR'] * 1.5)
        df['Apex_Trend'] = np.where(df['Close'] > df['Upper'], 1, np.where(df['Close'] < df['Lower'], -1, 0))
        
        # RVOL
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        return df

# =============================================================================
# 3. UNIFIED DATA FEED (TitanFeed)
# =============================================================================
class TitanFeed:
    @staticmethod
    def _clean_yf(df):
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    @st.cache_data(ttl=60)
    def fetch_crypto_kraken(symbol, timeframe, limit):
        try:
            exchange = ccxt.kraken()
            # Mapping common timeframes
            tf_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "1440"}
            ccxt_tf = tf_map.get(timeframe, "60")
            
            ohlcv = exchange.fetch_ohlcv(symbol, ccxt_tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            st.error(f"Kraken API Error: {e}")
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_stock_yfinance(symbol, period, interval):
        try:
            # Handle UK/Japan suffixes if needed
            if "." not in symbol and len(symbol) <= 5: 
                pass # US Stock
            
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            df = TitanFeed._clean_yf(df)
            
            if df.empty: return pd.DataFrame()

            # 4H resampling for stocks (Yahoo doesn't support 4h natively)
            if interval == "4h":
                df = yf.download(symbol, period=period, interval="1h", progress=False, auto_adjust=False)
                df = TitanFeed._clean_yf(df)
                df = df.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            
            return df
        except Exception as e:
            st.error(f"YFinance Error: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_macro_snapshot():
        tickers = ['^GSPC', 'BTC-USD', 'DX-Y.NYB', '^TNX']
        try:
            data = yf.download(tickers, period="5d", interval="1d", progress=False)['Close']
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
            snapshot = {}
            for t in tickers:
                if t in data.columns:
                    # Check if series has enough data
                    series = data[t].dropna()
                    if len(series) > 1:
                        curr = series.iloc[-1]
                        prev = series.iloc[-2]
                        chg = ((curr - prev) / prev) * 100
                        snapshot[t] = (curr, chg)
                    else:
                        snapshot[t] = (0, 0)
            return snapshot
        except: return {}

# =============================================================================
# 4. MODULE: LIVE TERMINAL (Desk 1)
# =============================================================================
def render_live_terminal(api_key):
    st.markdown("### ⚡ Live Execution Terminal")
    
    # 1. Configuration
    with st.expander("📡 Market & Strategy Configuration", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            asset_class = st.selectbox("Asset Class", ["Crypto (Kraken)", "Stocks (Yahoo)"])
        with c2:
            default_sym = "BTC/USD" if "Crypto" in asset_class else "NVDA"
            symbol = st.text_input("Symbol", default_sym).upper()
        with c3:
            tf = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
        with c4:
            limit = st.slider("Lookback Candles", 200, 2000, 500)
            
        st.markdown("---")
        l1, l2, l3, l4 = st.columns(4)
        with l1: tp1_mult = st.number_input("TP1 (ATR)", 1.0, 5.0, 1.5)
        with l2: tp2_mult = st.number_input("TP2 (ATR)", 1.0, 10.0, 3.0)
        with l3: tp3_mult = st.number_input("TP3 (ATR)", 1.0, 20.0, 5.0)
        with l4: sl_mult = st.number_input("Stop Loss (ATR)", 1.0, 5.0, 2.0)

    # 2. Fetch Data
    if "Crypto" in asset_class:
        df = TitanFeed.fetch_crypto_kraken(symbol, tf, limit)
    else:
        p_map = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "2y"}
        df = TitanFeed.fetch_stock_yfinance(symbol, p_map.get(tf, "1mo"), tf)

    if df.empty:
        st.error("No data found. Please check ticker symbol or API status.")
        return

    # 3. Process Engine
    try:
        df = TitanMath.calculate_all_indicators(df)
        last = df.iloc[-1]
    except Exception as e:
        st.error(f"Calculation Error (Check ticker): {e}")
        return

    # 4. Ladder Logic
    atr = last['ATR']
    # Dynamic levels based on Apex Trend direction
    if last['Apex_Trend'] == 1: # Bull
        stop_price = last['Close'] - (sl_mult * atr)
        tp1 = last['Close'] + (tp1_mult * atr)
        tp2 = last['Close'] + (tp2_mult * atr)
        tp3 = last['Close'] + (tp3_mult * atr)
        trend_str = "BULLISH 🐂"
        color = "#00ffbb"
    else: # Bear
        stop_price = last['Close'] + (sl_mult * atr)
        tp1 = last['Close'] - (tp1_mult * atr)
        tp2 = last['Close'] - (tp2_mult * atr)
        tp3 = last['Close'] - (tp3_mult * atr)
        trend_str = "BEARISH 🐻"
        color = "#ff1155"

    # 5. HUD Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price", f"${last['Close']:,.2f}")
    m2.metric("Trend", trend_str, delta="Apex System")
    m3.metric("Momentum", f"{last['Mom']:.2f}", delta="Squeeze ON" if last['Squeeze'] else "Released")
    m4.metric("Money Flow", f"{last['Money_Flow']:.2f}", delta="Inflow" if last['Money_Flow']>0 else "Outflow")
    m5.metric("RVOL", f"{last['RVOL']:.1f}x")

    # 6. Titan Charting (Plotly)
    tab_chart, tab_ladder, tab_ai = st.tabs(["🕯️ God Mode Chart", "🪜 Execution Ladder", "🤖 AI Analyst"])
    
    with tab_chart:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
        
        # Main Price + HMA + Cloud + FVG
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color=color, width=2), name="Apex HMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], fill='tonexty', fillcolor='rgba(0, 255, 187, 0.1)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
        
        # Plot TP/SL Lines
        fig.add_hline(y=stop_price, line_dash="dash", line_color="red", annotation_text="STOP", row=1, col=1)
        fig.add_hline(y=tp1, line_dash="dot", line_color="green", annotation_text="TP1", row=1, col=1)
        
        # FVG Visuals (Last 50 bars)
        recent = df.iloc[-50:]
        for i, row in recent.iterrows():
            if row['FVG_Bull']:
                fig.add_shape(type="rect", x0=i, x1=i, y0=row['Low'], y1=row['High'], fillcolor="green", opacity=0.3, line_width=0, xref="x", yref="y", row=1, col=1)
        
        # Subchart 1: Momentum
        cols = ['#00ffbb' if v > 0 else '#ff1155' for v in df['Mom']]
        fig.add_trace(go.Bar(x=df.index, y=df['Mom'], marker_color=cols, name="Momentum"), row=2, col=1)
        
        # Subchart 2: Money Flow + EVWM
        fig.add_trace(go.Scatter(x=df.index, y=df['Money_Flow'], fill='tozeroy', line=dict(color='cyan'), name="Money Flow"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EVWM'], line=dict(color='white', dash='dot'), name="EVWM"), row=3, col=1)
        
        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    with tab_ladder:
        st.markdown(f"""
        ### 🪜 Execution Ladder: {symbol}
        **Direction:** {trend_str}
        
        | Level | Price | Distance | R-Multiple |
        | :--- | :--- | :--- | :--- |
        | **ENTRY** | `${last['Close']:,.2f}` | 0.00% | 0.0R |
        | **STOP** | `${stop_price:,.2f}` | -{(abs(last['Close']-stop_price)/last['Close'])*100:.2f}% | -1.0R |
        | **TP1** | `${tp1:,.2f}` | +{(abs(last['Close']-tp1)/last['Close'])*100:.2f}% | +{tp1_mult}R |
        | **TP2** | `${tp2:,.2f}` | +{(abs(last['Close']-tp2)/last['Close'])*100:.2f}% | +{tp2_mult}R |
        | **TP3** | `${tp3:,.2f}` | +{(abs(last['Close']-tp3)/last['Close'])*100:.2f}% | +{tp3_mult}R |
        """)
        
        # TELEGRAM BROADCAST
        if st.button("📡 Broadcast Full Signal to Telegram"):
            # Construct the comprehensive message
            ai_snippet = "AI Analysis Pending..."
            if api_key and OpenAI:
                try:
                    client = OpenAI(api_key=api_key)
                    r = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":f"Short sentiment analysis for {symbol} based on {trend_str} trend."}])
                    ai_snippet = r.choices[0].message.content
                except: pass

            msg = f"""
🔥 **TITAN SIGNAL: {symbol}** 🔥
--------------------------------
📊 **Timeframe:** {tf}
🚀 **Trend:** {trend_str}
💵 **Entry:** ${last['Close']:,.2f}

🛑 **STOP LOSS:** ${stop_price:,.2f}
🎯 **TP1 ({tp1_mult}R):** ${tp1:,.2f}
🎯 **TP2 ({tp2_mult}R):** ${tp2:,.2f}
🎯 **TP3 ({tp3_mult}R):** ${tp3:,.2f}

📉 **Technicals:**
• RSI: {last['RSI']:.1f}
• RVOL: {last['RVOL']:.1f}x
• Money Flow: {last['Money_Flow']:.2f}
• Squeeze: {'Active' if last['Squeeze'] else 'Off'}

🤖 **AI Insight:**
{ai_snippet}

#TitanOS #Trading
"""
            if st.secrets.get("TELEGRAM_TOKEN"):
                requests.post(f"https://api.telegram.org/bot{st.secrets['TELEGRAM_TOKEN']}/sendMessage", 
                              data={"chat_id": st.secrets['TELEGRAM_CHAT_ID'], "text": msg, "parse_mode": "Markdown"})
                st.success("Signal Sent!")
            else:
                st.error("Telegram Secrets Missing")

    with tab_ai:
        st.subheader("🤖 Titan AI Analyst")
        if st.button("Generate Tactical Report"):
            if OpenAI and api_key:
                try:
                    client = OpenAI(api_key=api_key)
                    # Comprehensive Prompt
                    prompt = f"""
                    Act as a Senior Hedge Fund Analyst. Analyze {symbol} on {tf} timeframe.
                    
                    DATA FEED:
                    - Price: {last['Close']}
                    - Trend: {last['Apex_Trend']} (1=Bull, -1=Bear)
                    - Momentum: {last['Mom']:.2f}
                    - Money Flow: {last['Money_Flow']:.2f}
                    - FVG Detected: {last['FVG_Bull'] or last['FVG_Bear']}
                    - RVOL: {last['RVOL']:.2f}
                    - Volatility (ATR): {last['ATR']:.2f}
                    
                    Provide a professional trading plan in Markdown:
                    1. **Market Structure:** Is this a trend or chop? Mention SMC context.
                    2. **Volume Analysis:** Is smart money participating? (Check Money Flow/RVOL).
                    3. **Execution Plan:** Confirm if the Ladder levels (Stop/TPs) allow for a high probability trade.
                    4. **Verdict:** BULLISH / BEARISH / NEUTRAL with Confidence Score (1-10).
                    """
                    with st.spinner("Neural Engine Processing..."):
                        res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}])
                        st.markdown(f"<div class='ai-box'>{res.choices[0].message.content}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI Error: {e}")
            else:
                st.warning("OpenAI API Key Required")

# =============================================================================
# 6. MODULE: MACRO INTELLIGENCE (Desk 2)
# =============================================================================
def render_macro_desk():
    st.markdown("### 🌍 Macro Intelligence Desk")
    
    # Ratios
    ratios = {
        "Risk On/Off (SPY/TLT)": ("^GSPC", "TLT"),
        "Inflation (GLD/TLT)": ("GC=F", "TLT"),
        "Tech Strength (QQQ/SPY)": ("QQQ", "^GSPC"),
        "Crypto Dominance (BTC/ETH)": ("BTC-USD", "ETH-USD")
    }
    
    # Batch Fetch
    tickers = list(set([item for sublist in ratios.values() for item in sublist]))
    try:
        data = yf.download(tickers, period="6mo", interval="1d", progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    except:
        st.error("Macro Feed Failed")
        return
    
    cols = st.columns(2)
    for i, (name, (num, den)) in enumerate(ratios.items()):
        if num in data and den in data:
            ratio = data[num] / data[den]
            ratio = ratio.dropna()
            if len(ratio) > 20:
                curr = ratio.iloc[-1]
                chg = ((curr - ratio.iloc[-20]) / ratio.iloc[-20]) * 100
                
                with cols[i%2]:
                    st.metric(name, f"{curr:.4f}", f"{chg:.2f}% (1M)")
                    fig = px.line(ratio, title=name, template="plotly_dark")
                    fig.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0), xaxis_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 7. MODULE: DEEP SCANNER (Desk 3)
# =============================================================================
def render_scanner(api_key):
    st.markdown("### 🔭 Deep Factor Scanner")
    
    mode = st.radio("Scanner Mode", ["Institutional (ECVS)", "Miners (Clive)", "Microcaps (Apex)", "Crypto SMC"], horizontal=True)
    
    # Pre-defined Universes
    tickers = []
    if mode == "Institutional (ECVS)":
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "UNH", "XOM", "JNJ", "PG", "LLY", "AVGO"]
    elif mode == "Miners (Clive)":
        tickers = ["NXE", "UEC", "UUUU", "DNN", "LAC", "SGML", "KGC", "EQX", "ERO", "HBM", "MP", "CCJ"]
    elif mode == "Microcaps (Apex)":
        tickers = ["VNDA", "ORMP", "SELB", "QUIK", "ATOM", "MARA", "RIOT", "LNN", "TA", "PLTR", "SOFI"]
    elif mode == "Crypto SMC":
        tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "XRP-USD", "DOGE-USD", "LINK-USD", "MATIC-USD"]

    if st.button("🚀 Run Scan"):
        results = []
        bar = st.progress(0)
        status = st.empty()
        
        for i, t in enumerate(tickers):
            status.text(f"Scanning {t}...")
            try:
                # Fetch Data based on mode
                if mode == "Crypto SMC":
                    df = TitanFeed.fetch_crypto_kraken(t, "1h", 200)
                    if not df.empty and len(df) > 50:
                        df = TitanMath.calculate_all_indicators(df)
                        last = df.iloc[-1]
                        
                        score = 0
                        if last['Apex_Trend'] == 1: score += 1
                        if last['FVG_Bull']: score += 2
                        if last['Squeeze']: score += 1
                        if last['Money_Flow'] > 0: score += 1
                        
                        results.append({"Ticker": t, "Price": last['Close'], "Score": score, "Trend": last['Apex_Trend'], "FVG": last['FVG_Bull']})
                    
                else: # Stocks (YFinance)
                    tk = yf.Ticker(t)
                    info = tk.info
                    hist = tk.history(period="6mo")
                    if not hist.empty:
                        mom = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
                        
                        if mode == "Miners (Clive)":
                            cash = info.get('totalCash', 0)
                            debt = info.get('totalDebt', 0)
                            net_cash = cash - debt
                            pb = info.get('priceToBook', 0)
                            results.append({"Ticker": t, "Net Cash": net_cash, "P/B": pb, "Momentum": mom})
                            
                        elif mode == "Institutional (ECVS)":
                            pe = info.get('forwardPE', 99)
                            peg = info.get('pegRatio', 99)
                            margin = info.get('profitMargins', 0)
                            
                            score = 0
                            if pe < 25: score += 1
                            if peg < 1.5: score += 1
                            if margin > 0.2: score += 1
                            if mom > 0: score += 1
                            
                            results.append({"Ticker": t, "Score": score, "P/E": pe, "PEG": peg, "Momentum": mom})
                        
                        else: # Microcaps Logic
                            vol = hist['Close'].pct_change().std()
                            results.append({"Ticker": t, "Momentum": mom, "Volatility": vol})
            
            except Exception as e:
                # Keep scanning other tickers even if one fails
                pass
            
            bar.progress((i+1)/len(tickers))
            time.sleep(0.1) # Prevent Rate Limit
            
        bar.empty()
        status.empty()
        
        if results:
            df_res = pd.DataFrame(results)
            if "Score" in df_res.columns: df_res = df_res.sort_values("Score", ascending=False)
            st.success(f"Scan Complete: {len(df_res)} Hits")
            st.dataframe(df_res.style.background_gradient(cmap="Greens"), use_container_width=True)
            
            # AI Analysis on Top Pick
            if not df_res.empty and api_key and OpenAI:
                top_pick = df_res.iloc[0]
                if st.button(f"Analyze Top Pick: {top_pick['Ticker']}"):
                    try:
                        client = OpenAI(api_key=api_key)
                        prompt = f"Analyze investment thesis for {top_pick['Ticker']} based on gathered metrics: {top_pick.to_dict()}. Be concise."
                        res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}])
                        st.info(res.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI Error: {e}")

# =============================================================================
# 8. MODULE: STRATEGY LAB (Desk 4)
# =============================================================================
def render_strategy_lab():
    st.markdown("### 🧪 Strategy Lab (Backtesting)")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        strat = st.selectbox("Strategy Logic", ["SMA Crossover", "RSI Mean Reversion", "Apex Trend Follower", "Bollinger Breakout"])
    with c2:
        symbol = st.text_input("Test Symbol", "BTC-USD")
    with c3:
        capital = st.number_input("Starting Capital", 10000)
        
    if st.button("Run Simulation"):
        df = TitanFeed.fetch_stock_yfinance(symbol, "2y", "1d")
        if df.empty:
            st.error("No Data")
            return
            
        # Apply Logic
        df = TitanMath.calculate_all_indicators(df)
        df['Signal'] = 0
        
        if strat == "SMA Crossover":
            df['Fast'] = df['Close'].rolling(20).mean()
            df['Slow'] = df['Close'].rolling(50).mean()
            df.loc[df['Fast'] > df['Slow'], 'Signal'] = 1
            df.loc[df['Fast'] <= df['Slow'], 'Signal'] = -1
            
        elif strat == "Apex Trend Follower":
            df.loc[df['Apex_Trend'] == 1, 'Signal'] = 1
            df.loc[df['Apex_Trend'] == -1, 'Signal'] = -1
            
        elif strat == "RSI Mean Reversion":
            df.loc[df['RSI'] < 30, 'Signal'] = 1
            df.loc[df['RSI'] > 70, 'Signal'] = -1
            
        elif strat == "Bollinger Breakout":
            upper, _, lower = TitanMath.bollinger_bands(df['Close'])
            df.loc[df['Close'] > upper, 'Signal'] = 1
            df.loc[df['Close'] < lower, 'Signal'] = -1
            
        # Vectorized PnL Calculation
        df['Ret'] = df['Close'].pct_change()
        df['Strat_Ret'] = df['Ret'] * df['Signal'].shift(1)
        df['Equity'] = capital * (1 + df['Strat_Ret'].fillna(0)).cumprod()
        
        if not df['Equity'].dropna().empty:
            total_ret = ((df['Equity'].iloc[-1] - capital) / capital) * 100
            dd = (df['Equity'] / df['Equity'].cummax() - 1).min() * 100
            
            st.metric("Total Return", f"{total_ret:.2f}%", f"Drawdown: {dd:.2f}%")
            st.line_chart(df['Equity'])
            
            with st.expander("Trade Log"):
                st.dataframe(df[['Close', 'Signal', 'Equity']])
        else:
            st.error("Backtest Error: Insufficient Data for Strategy")

# =============================================================================
# 9. MAIN ORCHESTRATOR
# =============================================================================
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 💠 TITAN OS")
        
        # Macro Snapshot
        try:
            macro = TitanFeed.get_macro_snapshot()
            if macro:
                c1, c2 = st.columns(2)
                spx_val, spx_chg = macro.get('^GSPC', (0,0))
                btc_val, btc_chg = macro.get('BTC-USD', (0,0))
                c1.metric("SPX", f"{spx_val:.0f}", f"{spx_chg:.1f}%")
                c2.metric("BTC", f"{btc_val:.0f}", f"{btc_chg:.1f}%")
        except: st.caption("Macro Feed Loading...")
        
        st.markdown("---")
        mode = st.radio("SYSTEM DESK", ["Live Terminal", "Macro Intelligence", "Deep Scanner", "Strategy Lab"])
        
        st.markdown("---")
        # SECRETS LOADING
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key: api_key = st.text_input("OpenAI Key", type="password")
        
        if "PASSWORD" in st.secrets:
            pwd = st.text_input("System Lock", type="password")
            if pwd != st.secrets["PASSWORD"]:
                st.error("LOCKED")
                st.stop()

    # Routing
    st.markdown('<div class="titan-header"><h1 class="titan-title">TITAN OS</h1><div class="titan-subtitle">FINANCIAL SINGULARITY INTERFACE</div></div>', unsafe_allow_html=True)
    
    if mode == "Live Terminal":
        render_live_terminal(api_key)
    elif mode == "Macro Intelligence":
        render_macro_desk()
    elif mode == "Deep Scanner":
        render_scanner(api_key)
    elif mode == "Strategy Lab":
        render_strategy_lab()
        
    st.markdown("---")
    st.markdown("<center style='color:#666'>TITAN ARCHITECT v10.0 | QUANTUM CORE ACTIVE</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
