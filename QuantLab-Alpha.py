import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import openai
import requests
import numpy as np

# ==========================================
# 1. SETUP & SCIENTIFIC STYLING
# ==========================================
st.set_page_config(page_title="QuantLab Alpha", page_icon="🧪", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Roboto Mono', monospace; color: #7fdbff; }
    div[data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; font-size: 24px; }
    .stPlotlyChart { border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATISTICAL INDICATORS
# ==========================================
def calculate_quant_stats(df, window=20):
    df['Mean'] = df['Close'].rolling(window=window).mean()
    df['Std_Dev'] = df['Close'].rolling(window=window).std()
    
    # Calculate Z-Score: (Price - Mean) / StdDev
    df['Z_Score'] = (df['Close'] - df['Mean']) / df['Std_Dev']
    
    # Bollinger Band Width for Volatility Squeeze
    upper = df['Mean'] + (2 * df['Std_Dev'])
    lower = df['Mean'] - (2 * df['Std_Dev'])
    df['BB_Width'] = (upper - lower) / df['Mean']
    
    return df

# ==========================================
# 3. MEAN REVERSION LOGIC
# ==========================================
def analyze_probabilities(df):
    if df is None: return "No Data", 0.0
    curr = df.iloc[-1]
    z_score = curr['Z_Score']
    
    # Statistical Likelihood of Reversion
    if z_score > 2.0:
        return "STATISTICAL SHORT (Overextended)", z_score
    elif z_score < -2.0:
        return "STATISTICAL LONG (Undervalued)", z_score
    else:
        return "WITHIN NORMAL DISTRIBUTION", z_score

# ==========================================
# 4. AI & BROADCAST
# ==========================================
def generate_quant_report(ticker, signal, z_score, vol):
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key: return "API Key Error"
    
    client = openai.OpenAI(api_key=api_key)
    prompt = f"""
    Generate a Quantitative Trading Signal for {ticker}.
    Style: Scientific, data-driven, precise.
    Signal: {signal}
    Z-Score: {z_score:.2f} (Standard Deviations from mean)
    Volatility (BB Width): {vol:.4f}
    
    Explain the statistical probability of mean reversion here. Do not use hype words. Use "Sigma events" and "Variance".
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
st.title("🧪 QuantLab | Statistical Arbitrage")
col_input, col_kpi = st.columns([1, 4])

with col_input:
    ticker = st.text_input("Asset Ticker", "ETH-USD")
    lookback = st.slider("Lookback Window", 10, 100, 20)
    
if ticker:
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = calculate_quant_stats(df, lookback)
        
        signal_text, z_val = analyze_probabilities(df)
        
        # KPI Row
        k1, k2, k3 = st.columns(3)
        k1.metric("Current Z-Score", f"{z_val:.2f} σ")
        k2.metric("Mean Price", f"{df['Mean'].iloc[-1]:.2f}")
        k3.metric("BB Width", f"{df['BB_Width'].iloc[-1]:.4f}")
        
        st.subheader(f"Model Output: {signal_text}")
        
        # Complex Visualization: Price + Z-Score Heatmap
        fig = go.Figure()
        
        # Subplots: Price on Top, Z-Score on Bottom
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        
        # Price with Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='white')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Mean'], mode='lines', name='Mean', line=dict(color='yellow', dash='dash')), row=1, col=1)
        
        # Z-Score Histogram
        colors = np.where(df['Z_Score'] > 0, 'red', 'green')
        fig.add_trace(go.Bar(x=df.index, y=df['Z_Score'], name='Z-Score', marker_color=colors), row=2, col=1)
        
        # Sigma Lines
        fig.add_hline(y=2, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=-2, line_dash="dot", line_color="green", row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Section
        with st.expander("GENERATE QUANT SIGNAL REPORT", expanded=True):
            if st.button("Run Statistical Analysis Model"):
                report = generate_quant_report(ticker, signal_text, z_val, df['BB_Width'].iloc[-1])
                st.code(report)
                if st.button("Broadcast to Telegram"):
                    send_telegram(report)
                    st.success("Data Transmitted.")
