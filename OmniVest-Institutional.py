import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import openai
import requests

# ==========================================
# 1. SETUP & CORPORATE STYLING
# ==========================================
st.set_page_config(page_title="OmniVest Institutional", page_icon="🏛️", layout="wide")

st.markdown("""
<style>
    /* Clean Corporate Theme */
    .stApp { background-color: #f0f2f6; color: #2c3e50; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; font-weight: 300; }
    .stMetric { background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .stButton>button {
        background-color: #2980b9; color: white; border-radius: 5px; border: none;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. TREND INDICATORS
# ==========================================
def calculate_trend_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    # Volume MA
    df['Vol_SMA'] = df['Volume'].rolling(window=20).mean()
    return df

# ==========================================
# 3. SWING STRATEGY ENGINE
# ==========================================
def get_swing_signal(df):
    if df is None or len(df) < 200: return "INSUFFICIENT DATA", []
    
    curr = df.iloc[-1]
    reasons = []
    trend = "NEUTRAL"
    
    # Golden Cross / Death Cross Check
    if curr['SMA_50'] > curr['SMA_200']:
        trend = "BULLISH TREND"
        # Check for pullback to value
        dist = (curr['Close'] - curr['SMA_50']) / curr['SMA_50']
        if dist < 0.02 and dist > -0.02:
            reasons.append("Price testing SMA 50 Support")
            trend = "BUY OPPORTUNITY"
    else:
        trend = "BEARISH TREND"
    
    # Volume Confirmation
    if curr['Volume'] > curr['Vol_SMA'] * 1.5:
        reasons.append("High Volume Anomaly Detected")

    return trend, reasons

# ==========================================
# 4. AI & BROADCAST
# ==========================================
def generate_corporate_brief(ticker, trend, reasons, price):
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key: return "API Key Missing"
    
    client = openai.OpenAI(api_key=api_key)
    prompt = f"""
    Draft a professional investment memorandum for Telegram regarding {ticker}.
    Tone: Institutional, calm, risk-averse.
    Current Trend: {trend}
    Key Observations: {', '.join(reasons)}
    Price: ${price:.2f}
    
    Structure:
    1. Executive Summary
    2. Technical Confluence
    3. Risk Assessment (Standard Disclaimer)
    """
    
    try:
        res = client.chat.completions.create(
            model="gpt-4", # Using GPT-4 for better reasoning if available
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as e: return "Analysis unavailable."

def send_telegram(msg):
    token = st.secrets.get("telegram", {}).get("bot_token")
    cid = st.secrets.get("telegram", {}).get("channel_id")
    if token and cid:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": cid, "text": msg})
        return True
    return False

# ==========================================
# 5. MAIN APP
# ==========================================
st.title("🏛️ OmniVest | Institutional Analytics")
st.markdown("### Equity & Forex Trend Analysis")
st.divider()

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Instrument")
    ticker = st.text_input("Symbol", "MSFT")
    
    if ticker:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = calculate_trend_indicators(df)
            trend, reasons = get_swing_signal(df)
            
            st.info(f"Market Phase: {trend}")
            for r in reasons:
                st.caption(f"• {r}")
                
            if st.button("Draft Investor Brief"):
                brief = generate_corporate_brief(ticker, trend, reasons, df['Close'].iloc[-1])
                st.text_area("Review Brief", brief, height=300)
                if st.button("Publish to Channel"):
                    send_telegram(brief)
                    st.success("Brief Published.")

with col2:
    if ticker and not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=2), name='SMA 50 (Institutional Support)'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='black', width=2), name='SMA 200 (Long Term Trend)'))
        
        fig.update_layout(template="simple_white", height=600, title=f"{ticker} Long Term Trend Analysis")
        st.plotly_chart(fig, use_container_width=True)
