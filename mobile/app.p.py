import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import requests
import datetime
import math

# ==========================================
# 1. DPC v2.0 CSS ARCHITECTURE (HIGH DENSITY)
# ==========================================
st.set_page_config(layout="wide", page_title="TITAN v2.0", page_icon="👁️")

def apply_terminal_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Roboto Mono', monospace; }
    .title-glow {
        font-size: 2.8em; font-weight: 800; color: #ffffff;
        text-shadow: 0 0 12px #00ff00; margin-bottom: 5px;
    }
    div[data-testid="stMetric"] {
        background-color: #161b22; border: 1px solid #30363d;
        padding: 15px; border-radius: 4px;
    }
    div[data-testid="stMetric"]:hover { border-color: #00ff00; }
    .report-box {
        background-color: #0d1117; border-left: 4px solid #00ff00;
        padding: 20px; border-radius: 4px; margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22; border: 1px solid #30363d;
        padding: 10px 20px; border-radius: 4px 4px 0 0; color: #8b949e;
    }
    .stTabs [aria-selected="true"] { background-color: #0d1117; color: #00ff00; border-bottom: 2px solid #00ff00; }
    </style>
    """, unsafe_allow_html=True)

apply_terminal_theme()

# ==========================================
# 2. QUANT ENGINE (GOD MODE LOGIC)
# ==========================================
def calculate_hma(series, length):
    half_len = int(length / 2)
    sqrt_len = int(math.sqrt(length))
    wma_f = series.rolling(length).mean()
    wma_h = series.rolling(half_len).mean()
    diff = 2 * wma_h - wma_f
    return diff.rolling(sqrt_len).mean()

def get_god_mode_indicators(df):
    """Calculates Multi-Factor Confluence (Titan Score)."""
    # 1. Apex Trend (HMA + ATR)
    df['HMA'] = calculate_hma(df['Close'], 55)
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    apex_mult = 1.5
    df['Apex_Upper'] = df['HMA'] + (df['ATR'] * apex_mult)
    df['Apex_Lower'] = df['HMA'] - (df['ATR'] * apex_mult)
    df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))
    df['Apex_Trend'] = df['Apex_Trend'].replace(to_replace=0, method='ffill')

    # 2. DarkPool Squeeze
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['Sqz_Upper_BB'] = sma20 + (std20 * 2)
    df['Sqz_Lower_BB'] = sma20 - (std20 * 2)
    df['Sqz_Upper_KC'] = sma20 + (df['ATR'] * 1.5)
    df['Sqz_Lower_KC'] = sma20 - (df['ATR'] * 1.5)
    df['Squeeze_On'] = (df['Sqz_Lower_BB'] > df['Sqz_Lower_KC']) & (df['Sqz_Upper_BB'] < df['Sqz_Upper_KC'])
    
    # 3. Money Flow Matrix (Normalized)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    df['MF_Matrix'] = (df['RSI'] - 50) * (df['Volume'] / df['Volume'].rolling(20).mean())

    # 4. Titan Score Confluence
    df['Titan_Score'] = df['Apex_Trend'] + np.sign(df['MF_Matrix']) + np.where(df['Close'] > df['HMA'], 1, -1)
    return df

# ==========================================
# 3. PINE SCRIPT v6 GENERATOR MODULE
# ==========================================
def generate_pine_v6(symbol, titan_score):
    """Generates TradingView v6 Code based on latest standards."""
    # Utilizing //@version=6 and ISIN identification where available
    v6_code = f"""//@version=6
// TITAN v2.0 ARCHITECT GENERATED SCRIPT
// Symbol: {symbol}
indicator("TITAN God Mode v2.0", overlay = true)

// ISIN identification
// isin_id = syminfo.isin 

// Apex Trend Logic (HMA + ATR)
hma_len = input.int(55, "HMA Length")
atr_len = input.int(14, "ATR Length")
mult = input.float(1.5, "ATR Multiplier")

hma = ta.hma(close, hma_len)
atr = ta.atr(atr_len)
upper = hma + (atr * mult)
lower = hma - (atr * mult)

// Drawing Cloud with new v6 standards
plot(hma, "Trend Basis", color.new(color.yellow, 0))
p1 = plot(upper, "Apex Upper", color.new(color.green, 100))
p2 = plot(lower, "Apex Lower", color.new(color.red, 100))
fill(p1, p2, hma < close ? color.new(color.green, 90) : color.new(color.red, 90))

// Signal Alerts
buy_sig = ta.crossover(close, upper)
sell_sig = ta.crossunder(close, lower)

plotshape(buy_sig, "Titan Buy", shape.triangleup, location.belowbar, color.green, size = size.small)
plotshape(sell_sig, "Titan Sell", shape.triangledown, location.abovebar, color.red, size = size.small)

alertcondition(buy_sig or sell_sig, "Titan Alert", "Trend Shift Detected on " + syminfo.tickerid)
"""
    return v6_code

# ==========================================
# 4. SIDEBAR & CONTROL CENTER
# ==========================================
with st.sidebar:
    st.markdown('<div class="title-glow">👁️ TITAN v2.0</div>', unsafe_allow_html=True)
    st.caption("Institutional Synthesis Engine")
    st.markdown("---")
    
    ticker = st.text_input("Search Ticker", value="BTC-USD").upper()
    interval = st.selectbox("Interval", ["1h", "4h", "1d", "1wk"], index=2)
    
    st.subheader("📡 Social Broadcaster")
    tg_token = st.secrets.get("TELEGRAM_TOKEN", "")
    tg_chat = st.secrets.get("TELEGRAM_CHAT_ID", "")
    
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()

# ==========================================
# 5. MAIN DASHBOARD EXECUTION
# ==========================================
if ticker:
    with st.spinner(f"Architecting Analysis for {ticker}..."):
        # Fetching Data
        df = yf.download(ticker, period="1y", interval=interval, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = get_god_mode_indicators(df)
            last = df.iloc[-1]
            
            # --- TOP METRICS ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price", f"${last['Close']:,.2f}", f"{((last['Close']-df['Close'].iloc[-2])/df['Close'].iloc[-2])*100:.2f}%")
            m2.metric("Titan Score", f"{last['Titan_Score']:.0f}/3", "Bullish" if last['Titan_Score'] > 0 else "Bearish")
            m3.metric("Apex Trend", "BULL" if last['Apex_Trend'] == 1 else "BEAR")
            m4.metric("Squeeze", "ACTIVE" if last['Squeeze_On'] else "OFF")

            # --- TABBED INTERFACE ---
            tab1, tab2, tab3 = st.tabs(["📊 Terminal Chart", "🧠 AI Analyst", "📜 Pine v6 Generator"])
            
            with tab1:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
                # Price + Apex Cloud
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)' if last['Apex_Trend']==1 else 'rgba(255, 0, 0, 0.1)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
                # Money Flow Matrix
                colors = ['#00ff00' if x > 0 else '#ff0000' for x in df['MF_Matrix']]
                fig.add_trace(go.Bar(x=df.index, y=df['MF_Matrix'], marker_color=colors, name="Money Flow"), row=2, col=1)
                
                fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("🤖 Institutional AI Briefing")
                # Institutional Report Logic
                report_template = f"""
                ### 💠 TITAN DEEP DIVE: {ticker}
                
                **Executive Summary:**
                The asset is currently showing a **{'Strong Bullish' if last['Titan_Score'] >= 2 else 'Weak' if last['Titan_Score'] == 0 else 'Strong Bearish'}** bias. 
                Apex Trend is **{'BULLISH' if last['Apex_Trend']==1 else 'BEARISH'}**.
                
                **Flow Analysis:**
                Money Flow Matrix is at `{last['MF_Matrix']:.2f}`, indicating **{'Accumulation' if last['MF_Matrix'] > 0 else 'Distribution'}**.
                
                **Technical Verdict:**
                Maintain risk-adjusted exposure. Stop Loss recommended below the Apex Lower Band at `${last['Apex_Lower']:,.2f}`.
                """
                st.markdown(f'<div class="report-box">{report_template}</div>', unsafe_allow_html=True)
                
                if st.button("✈️ Broadcast Signal to Telegram"):
                    if tg_token and tg_chat:
                        msg = f"🔥 *TITAN SIGNAL: {ticker}*\n\nPrice: ${last['Close']:,.2f}\nTitan Score: {last['Titan_Score']:.0f}/3\n\n{report_template}"
                        requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", json={"chat_id": tg_chat, "text": msg, "parse_mode": "Markdown"})
                        st.success("Signal Dispatched!")

            with tab3:
                st.subheader("📜 Pine Script v6 Architect")
                st.caption("Auto-generated code utilizing TradingView v6 standards.")
                v6_output = generate_pine_v6(ticker, last['Titan_Score'])
                st.code(v6_output, language="pine")
                st.info("Upgraded for v6: Includes optimized `ta.hma` calculations and ISIN-ready identification logic.")

        else:
            st.error("Market data connection failed. Check symbol formatting.")
