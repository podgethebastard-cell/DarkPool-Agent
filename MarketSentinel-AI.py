import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import requests
import numpy as np
from datetime import datetime

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="MarketSentinel AI",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for a professional dark theme look
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #262730;
        border-radius: 5px;
        padding: 15px;
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    .signal-buy { color: #00CC96; font-weight: bold; }
    .signal-sell { color: #EF553B; font-weight: bold; }
    .signal-neutral { color: #636EFA; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# TECHNICAL ANALYSIS LIBRARY (Native Pandas)
# ==========================================
class TechnicalAnalysis:
    @staticmethod
    def calculate_rsi(data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data, slow=26, fast=12, signal=9):
        exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_bollinger_bands(data, window=20, no_of_std=2):
        rolling_mean = data['Close'].rolling(window).mean()
        rolling_std = data['Close'].rolling(window).std()
        upper_band = rolling_mean + (rolling_std * no_of_std)
        lower_band = rolling_mean - (rolling_std * no_of_std)
        return upper_band, rolling_mean, lower_band

    @staticmethod
    def calculate_atr(data, window=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=window).mean()

    @staticmethod
    def calculate_smas(data):
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        return data

# ==========================================
# DATA HANDLER
# ==========================================
def get_market_data(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        
        # Flatten MultiIndex columns if necessary (common in newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Calculate Indicators
        df['RSI'] = TechnicalAnalysis.calculate_rsi(df)
        df['MACD'], df['MACD_Signal'] = TechnicalAnalysis.calculate_macd(df)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = TechnicalAnalysis.calculate_bollinger_bands(df)
        df['ATR'] = TechnicalAnalysis.calculate_atr(df)
        df = TechnicalAnalysis.calculate_smas(df)
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ==========================================
# STRATEGY ENGINE
# ==========================================
def analyze_signal(df):
    """
    Analyzes the latest candle to determine a trading signal.
    Strategies:
    1. RSI Extremes: Buy < 30, Sell > 70
    2. MACD Crossover
    3. Price vs Bollinger Bands
    4. Golden/Death Cross
    """
    if df is None or len(df) < 200:
        return "NEUTRAL", "Insufficient Data", {}

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0
    reasons = []

    # 1. RSI Strategy
    if curr['RSI'] < 30:
        score += 2
        reasons.append(f"RSI Oversold ({curr['RSI']:.2f})")
    elif curr['RSI'] > 70:
        score -= 2
        reasons.append(f"RSI Overbought ({curr['RSI']:.2f})")
    
    # 2. MACD Strategy
    if curr['MACD'] > curr['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
        score += 1
        reasons.append("MACD Bullish Crossover")
    elif curr['MACD'] < curr['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
        score -= 1
        reasons.append("MACD Bearish Crossover")

    # 3. Bollinger Bands
    if curr['Close'] < curr['BB_Lower']:
        score += 1
        reasons.append("Price below Lower Bollinger Band (Dip Buy)")
    elif curr['Close'] > curr['BB_Upper']:
        score -= 1
        reasons.append("Price above Upper Bollinger Band (Possible Reversal)")

    # 4. Golden/Death Cross
    if curr['SMA_50'] > curr['SMA_200']:
        score += 0.5 # Bullish trend context
    else:
        score -= 0.5 # Bearish trend context

    # Final Verdict
    if score >= 2:
        signal = "STRONG BUY"
    elif score >= 1:
        signal = "BUY"
    elif score <= -2:
        signal = "STRONG SELL"
    elif score <= -1:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    # Context Data for AI
    context = {
        "Price": curr['Close'],
        "RSI": curr['RSI'],
        "MACD_Hist": curr['MACD'] - curr['MACD_Signal'],
        "ATR": curr['ATR'],
        "SMA_50": curr['SMA_50'],
        "SMA_200": curr['SMA_200']
    }

    return signal, reasons, context

# ==========================================
# AI INTEGRATION
# ==========================================
def generate_ai_analysis(ticker, signal, reasons, context):
    """Generates a smart summary using OpenAI."""
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key:
        return "OpenAI API Key not found in secrets."

    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    You are an expert financial analyst. Analyze the following technical data for {ticker}.
    
    Signal: {signal}
    Technical Factors: {', '.join(reasons)}
    Current Price: {context['Price']:.2f}
    RSI: {context['RSI']:.2f}
    ATR (Volatility): {context['ATR']:.2f}
    Trend Context: SMA50 is {'above' if context['SMA_50'] > context['SMA_200'] else 'below'} SMA200.

    Write a concise, professional Telegram broadcast message. 
    Includes:
    1. A catchy header with emojis.
    2. The Trade Setup (Why we are entering).
    3. Suggested Stop Loss (based on 1.5x ATR).
    4. Suggested Take Profit (Risk/Reward 1:2).
    5. A brief risk warning.
    Keep it under 150 words.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a professional crypto/stock trader."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Analysis Failed: {e}"

# ==========================================
# TELEGRAM BROADCASTER
# ==========================================
def send_telegram_alert(message):
    token = st.secrets.get("telegram", {}).get("bot_token")
    chat_id = st.secrets.get("telegram", {}).get("channel_id")
    
    if not token or not chat_id:
        return False, "Telegram secrets missing."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return True, "Message sent successfully!"
        else:
            return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

# ==========================================
# UI LAYOUT
# ==========================================
st.title("🦅 MarketSentinel AI")
st.markdown("### Institutional Grade Technical Analysis & Broadcasting")
st.divider()

# Sidebar
with st.sidebar:
    st.header("Asset Configuration")
    ticker = st.text_input("Ticker Symbol", value="BTC-USD").upper()
    interval = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=3)
    period_map = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y", "1wk": "2y"}
    
    st.header("Strategy Settings")
    rsi_window = st.slider("RSI Window", 10, 30, 14)
    show_entry_exits = st.checkbox("Show AI Signals on Chart", value=True)
    
    st.info("Ensure secrets.toml is configured for OpenAI & Telegram features.")

# Main Logic
if ticker:
    with st.spinner(f"Fetching data for {ticker}..."):
        df = get_market_data(ticker, period=period_map[interval], interval=interval)

    if df is not None:
        # Calculate Signal
        signal, reasons, context = analyze_signal(df)
        
        # --- Top Metrics Row ---
        col1, col2, col3, col4 = st.columns(4)
        current_price = df['Close'].iloc[-1]
        price_change = current_price - df['Close'].iloc[-2]
        
        col1.metric("Current Price", f"{current_price:.2f}", f"{price_change:.2f}")
        col2.metric("RSI (14)", f"{context['RSI']:.2f}")
        col3.metric("MACD Signal", f"{context['MACD_Hist']:.4f}")
        
        # Signal Badge
        color_map = {"STRONG BUY": "green", "BUY": "lightgreen", "NEUTRAL": "gray", "SELL": "orange", "STRONG SELL": "red"}
        col4.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: {color_map.get(signal, 'gray')}; border-radius: 5px;">
                <h3 style="margin:0; color: white;">{signal}</h3>
            </div>
        """, unsafe_allow_html=True)

        # --- Plotly Chart ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=(f'{ticker} Price Action', 'RSI & Volume'), 
                            row_width=[0.2, 0.7])

        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)

        # SMAs & Bollinger
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='BB Lower'), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
        fig.add_shape(type="line", x0=df.index[0], y0=70, x1=df.index[-1], y1=70, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
        fig.add_shape(type="line", x0=df.index[0], y0=30, x1=df.index[-1], y1=30, line=dict(color="green", width=1, dash="dash"), row=2, col=1)

        fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # --- AI Analysis & Broadcast Section ---
        st.divider()
        st.subheader("🤖 AI Smart Analysis & Broadcasting")
        
        col_ai, col_preview = st.columns([1, 1])
        
        with col_ai:
            if st.button("Generate AI Signal Report"):
                with st.spinner("Consulting the Oracle..."):
                    ai_report = generate_ai_analysis(ticker, signal, reasons, context)
                    st.session_state['ai_report'] = ai_report
            
            if 'ai_report' in st.session_state:
                st.text_area("AI Generated Report", st.session_state['ai_report'], height=250)
        
        with col_preview:
            st.markdown("#### Broadcast Preview")
            if 'ai_report' in st.session_state:
                st.info(st.session_state['ai_report'])
                if st.button("🚀 Broadcast to Telegram Channel"):
                    success, msg = send_telegram_alert(st.session_state['ai_report'])
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
            else:
                st.write("Generate a report to preview the broadcast.")

    else:
        st.warning("No data found for this ticker.")
