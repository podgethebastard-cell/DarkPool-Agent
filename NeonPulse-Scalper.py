import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import openai
import requests
import numpy as np

# ==========================================
# 1. SETUP & CYBERPUNK STYLING
# ==========================================
st.set_page_config(page_title="NeonPulse Scalper", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    /* Cyberpunk Theme */
    .stApp { background-color: #050505; color: #00ff41; font-family: 'Courier New', monospace; }
    h1, h2, h3 { color: #00ff41 !important; text-shadow: 0 0 10px #00ff41; }
    .stButton>button {
        background-color: #000000; color: #00ff41; border: 1px solid #00ff41;
        border-radius: 0px; text-transform: uppercase; letter-spacing: 2px;
    }
    .stButton>button:hover {
        background-color: #00ff41; color: #000000; box-shadow: 0 0 20px #00ff41;
    }
    div[data-testid="stMetricValue"] { text-shadow: 0 0 5px white; color: white; }
    .stat-box { border: 1px solid #333; padding: 10px; background: #111; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOMENTUM INDICATORS
# ==========================================
def calculate_scalping_indicators(df):
    # EMA Ribbon (Fast moving averages)
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # Stochastic RSI
    window = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Stoch calc on RSI
    stoch_min = rsi.rolling(window=14).min()
    stoch_max = rsi.rolling(window=14).max()
    df['Stoch_K'] = 100 * (rsi - stoch_min) / (stoch_max - stoch_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    return df

# ==========================================
# 3. SCALPING STRATEGY ENGINE
# ==========================================
def get_scalp_signal(df):
    if df is None or len(df) < 50: return "WAIT", []
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    reasons = []
    score = 0
    
    # Logic: EMA Crossover + Stoch RSI Confirmation
    bullish_cross = (curr['EMA_8'] > curr['EMA_21']) and (prev['EMA_8'] <= prev['EMA_21'])
    bearish_cross = (curr['EMA_8'] < curr['EMA_21']) and (prev['EMA_8'] >= prev['EMA_21'])
    
    if bullish_cross:
        reasons.append("EMA 8/21 Bull Cross")
        score += 2
    if bearish_cross:
        reasons.append("EMA 8/21 Bear Cross")
        score -= 2
        
    # Stoch Momentum
    if curr['Stoch_K'] < 20 and curr['Stoch_K'] > curr['Stoch_D']:
        reasons.append("Stoch RSI Oversold Bounce")
        score += 1
    if curr['Stoch_K'] > 80 and curr['Stoch_K'] < curr['Stoch_D']:
        reasons.append("Stoch RSI Overbought Dump")
        score -= 1
        
    if score >= 2: return "LONG 🚀", reasons
    elif score <= -2: return "SHORT 🩸", reasons
    else: return "HODL ✋", reasons

# ==========================================
# 4. AI & BROADCAST
# ==========================================
def generate_degen_alert(ticker, signal, reasons, price):
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key: return "⚠️ Missing API Key"
    
    client = openai.OpenAI(api_key=api_key)
    prompt = f"""
    Write a crypto scalping telegram alert for {ticker}.
    Style: High energy, concise, use fire/rocket emojis. "Degen" trading style.
    Signal: {signal}
    Price: {price}
    Triggers: {', '.join(reasons)}
    Include exact entry and a quick scalp target (+1.5%).
    Max 4 lines.
    """
    
    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as e: return str(e)

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
st.title("⚡ NEON PULSE // SCALPER")
col1, col2 = st.columns([1, 3])

with col1:
    ticker = st.text_input("ASSET", "BTC-USD").upper()
    interval = st.selectbox("TIMEFRAME", ["1m", "5m", "15m", "1h"], index=1)
    
if ticker:
    # Handle yfinance limitations on 1m data
    p_map = {"1m": "5d", "5m": "5d", "15m": "1mo", "1h": "3mo"}
    df = yf.download(ticker, period=p_map[interval], interval=interval, progress=False)
    
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = calculate_scalping_indicators(df)
        signal, reasons = get_scalp_signal(df)
        
        # Display Signal
        st.markdown(f"<h1 style='text-align: center; font-size: 80px; margin:0;'>{signal}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #888;'>{', '.join(reasons)}</p>", unsafe_allow_html=True)
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_8'], line=dict(color='#00ff41', width=1), name='EMA 8'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], line=dict(color='#ff00ff', width=1), name='EMA 21'))
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Action
        if st.button("GENERATE & BLAST SIGNAL 📡"):
            with st.spinner("Calculating Alpha..."):
                alert = generate_degen_alert(ticker, signal, reasons, df['Close'].iloc[-1])
                st.code(alert, language="markdown")
                if send_telegram(alert):
                    st.success("SIGNAL BROADCASTED")
    else:
        st.error("NO DATA FEED")
