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
import scipy.stats as stats
from scipy.stats import linregress, norm
from datetime import timedelta
import logging
import io
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

# =============================================================================
# SYSTEM INCEPTION: THE QUANT PERSONA & CONFIG
# =============================================================================
# Internal Directive: Never omit code. Professional Lab Aesthetic.
# Strategic Focus: SMC, Z-Scores, Bollinger Squeezes, Probability Distributions.
# =============================================================================

warnings.filterwarnings('ignore')

# GLOBAL AI LOADER
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# =============================================================================
# 1. SCIENTIFIC UI CORE (LAB THEME)
# =============================================================================
st.set_page_config(
    page_title="QuantLab | Statistical Singularity",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_lab_aesthetic():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Inter:wght@300;700;900&display=swap');
        
        :root {
            --primary-glow: #00ffbb;
            --secondary-glow: #7d00ff;
            --bg-dark: #050505;
            --panel-bg: #0d0d0d;
        }

        .stApp { background-color: var(--bg-dark); color: #e0e0e0; font-family: 'Inter', sans-serif; }
        
        /* THE QUANT LAB HEADER */
        .lab-header {
            background: linear-gradient(180deg, #111 0%, #050505 100%);
            border-bottom: 1px solid #222;
            padding: 1.5rem;
            text-align: left;
            margin-bottom: 2rem;
            border-left: 5px solid var(--primary-glow);
        }
        .lab-title {
            font-size: 2.2rem;
            font-weight: 900;
            letter-spacing: -1px;
            color: #fff;
            margin: 0;
        }
        .lab-status {
            font-family: 'JetBrains Mono', monospace;
            color: var(--primary-glow);
            font-size: 0.8rem;
            text-transform: uppercase;
        }

        /* DATA GRID CARDS */
        div[data-testid="metric-container"] {
            background: var(--panel-bg);
            border: 1px solid #1a1a1a;
            padding: 20px;
            border-radius: 2px;
            box-shadow: inset 0 0 10px rgba(0,255,187,0.02);
        }
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: var(--primary-glow);
        }

        /* SIDEBAR SCIENTIFIC OVERLAY */
        section[data-testid="stSidebar"] {
            background-color: #080808;
            border-right: 1px solid #1a1a1a;
        }

        /* THE QUANT TERMINAL INPUTS */
        .stTextInput>div>div>input {
            background-color: #0f0f0f;
            color: var(--primary-glow);
            border: 1px solid #333;
            font-family: 'JetBrains Mono', monospace;
        }

        /* PROBABILITY BADGES */
        .prob-badge {
            padding: 4px 10px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: bold;
        }
        .high-conf { background: rgba(0, 255, 187, 0.1); color: #00ffbb; border: 1px solid #00ffbb; }
        .low-conf { background: rgba(255, 17, 85, 0.1); color: #ff1155; border: 1px solid #ff1155; }
    </style>
    """, unsafe_allow_html=True)

apply_lab_aesthetic()

# =============================================================================
# 2. THE QUANT ENGINE: MATHEMATICAL MODELS
# =============================================================================
class StatisticalModels:
    """
    Core mathematical framework for Smart Money Concepts (SMC) and 
    Statistical Mean Reversion.
    """
    
    @staticmethod
    def calculate_z_score(series: pd.Series, window: int = 20) -> pd.Series:
        """Computes the distance from the mean in units of standard deviation."""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std

    @staticmethod
    def get_probability_density(current_z: float) -> float:
        """Calculates the probability of price returning to mean based on Normal Distribution."""
        # P-value of the current Z excursion
        return (1 - stats.norm.cdf(abs(current_z))) * 2

    @staticmethod
    def identify_smc_levels(df: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Detects Institutional Order Blocks and Liquidity Voids (FVG).
        """
        df = df.copy()
        # Change of Character (CHoCH) & Break of Structure (BOS)
        df['HH'] = df['High'] == df['High'].rolling(window=lookback, center=True).max()
        df['LL'] = df['Low'] == df['Low'].rolling(window=lookback, center=True).min()
        
        # Fair Value Gaps (FVG)
        # Bullish: Low(n) > High(n-2)
        bull_fvg = (df['Low'] > df['High'].shift(2))
        # Bearish: High(n) < Low(n-2)
        bear_fvg = (df['High'] < df['Low'].shift(2))
        
        return {
            "bull_fvg": bull_fvg,
            "bear_fvg": bear_fvg,
            "order_blocks": df[df['HH'] | df['LL']]
        }

    @staticmethod
    def bollinger_squeeze(df: pd.DataFrame, length: int = 20, mult: float = 2.0, kc_mult: float = 1.5):
        """
        Measures Volatility Compression.
        Squeeze occurs when Bollinger Bands are inside Keltner Channels.
        """
        basis = df['Close'].rolling(window=length).mean()
        std = df['Close'].rolling(window=length).std()
        
        bb_upper = basis + (std * mult)
        bb_lower = basis - (std * mult)
        
        # Keltner Channel
        tr = pd.concat([df['High']-df['Low'], 
                        abs(df['High']-df['Close'].shift()), 
                        abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=length).mean()
        
        kc_upper = basis + (atr * kc_mult)
        kc_lower = basis - (atr * kc_mult)
        
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        return squeeze_on, bb_upper, bb_lower

    @staticmethod
    def money_flow_index(df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Institutional Volume Delta tracking."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        mfr = positive_flow.rolling(length).sum() / negative_flow.rolling(length).sum()
        return 100 - (100 / (1 + mfr))

# =============================================================================
# 3. BROADCAST MODULE: SIGNAL INTELLIGENCE
# =============================================================================
class BroadcastExpert:
    """Handles Telegram broadcasting and professional reporting."""
    
    @staticmethod
    def format_signal(data: Dict) -> str:
        """Constructs a high-density institutional signal report."""
        p_val = data['p_value']
        confidence = "HIGH" if p_val < 0.05 else "LOW"
        
        msg = f"""
🧪 **QUANTLAB ALPHA BROADCAST**
-------------------------------
**ASSET:** {data['symbol']} | **TF:** {data['tf']}
**SENTIMENT:** {data['direction']}
**PROBABILITY (P-VAL):** {p_val:.4f} ({confidence})

**STATISTICAL THESIS:**
• Z-Score: {data['z_score']:.2f}σ
• Volatility: {'SQUEEZE' if data['squeeze'] else 'EXPANSION'}
• Institutional Flow (MFI): {data['mfi']:.1f}
• SMC Context: {data['smc_note']}

**EXECUTION LEVELS:**
• Entry: {data['price']}
• Hard Stop: {data['stop']}
• Target (Mean): {data['target']}

-------------------------------
*Calculated at {datetime.datetime.now().strftime('%H:%M:%S')} UTC*
#QuantLab #MeanReversion
"""
        return msg

    @staticmethod
    def send_telegram(message: str, token: str, chat_id: str):
        if not token or not chat_id:
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            response = requests.post(url, data=payload)
            return response.status_code == 200
        except:
            return False

# =============================================================================
# 4. DATA PIPELINE (THE FEED)
# =============================================================================
class DataOrchestrator:
    @staticmethod
    @st.cache_data(ttl=300)
    def get_market_data(symbol: str, interval: str, count: int = 500):
        try:
            # Check if crypto
            if "/" in symbol or "-" in symbol and len(symbol) > 6:
                exchange = ccxt.binance()
                # Map interval
                tf_map = {"1h": "1h", "4h": "4h", "1d": "1d", "15m": "15m"}
                ohlcv = exchange.fetch_ohlcv(symbol.replace("-", "/"), tf_map.get(interval, "1h"), limit=count)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                # Stock / Forex via YF
                ticker = yf.Ticker(symbol)
                period_map = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
                df = ticker.history(period=period_map.get(interval, "1mo"), interval=interval)
                return df
        except Exception as e:
            st.error(f"Data Feed Error: {e}")
            return pd.DataFrame()

# =============================================================================
# 5. LAB COMPONENT: THE INTERFACE
# =============================================================================
def main():
    # SIDEBAR: LAB PARAMETERS
    with st.sidebar:
        st.markdown("<h2 style='color:#00ffbb;'>SYSTEM CONFIG</h2>", unsafe_allow_html=True)
        
        symbol = st.text_input("INSTRUMENT TICKER", "BTC-USD").upper()
        timeframe = st.selectbox("TEMPORAL RESOLUTION", ["15m", "1h", "4h", "1d"], index=1)
        
        st.markdown("---")
        st.markdown("### 📡 BROADCAST CONFIG")
        tg_token = st.text_input("TELEGRAM BOT TOKEN", type="password")
        tg_chat = st.text_input("TELEGRAM CHAT ID")
        
        st.markdown("---")
        st.markdown("### 🧪 MODEL TUNING")
        z_window = st.slider("Z-SCORE WINDOW", 10, 100, 20)
        smc_lookback = st.slider("SMC LOOKBACK", 5, 50, 20)
        
        st.markdown("---")
        if OpenAI and st.secrets.get("OPENAI_API_KEY"):
            ai_enabled = st.checkbox("ACTIVATE AI ANALYST", value=True)
        else:
            ai_enabled = False
            st.caption("AI Analyst requires API Key.")

    # HEADER
    st.markdown("""
    <div class="lab-header">
        <div class="lab-status">● QUANTLAB CORE ACTIVE | STATISTICAL SIGNIFICANCE: ALPHA</div>
        <h1 class="lab-title">QuantLab Analysis Terminal</h1>
    </div>
    """, unsafe_allow_html=True)

    # FETCH & COMPUTE
    df = DataOrchestrator.get_market_data(symbol, timeframe)
    
    if df.empty:
        st.warning("Awaiting market data. Ensure symbol validity.")
        return

    # QUANT CALCULATIONS
    stats_engine = StatisticalModels()
    df['Z_Score'] = stats_engine.calculate_z_score(df['Close'], window=z_window)
    df['MFI'] = stats_engine.money_flow_index(df)
    squeeze_on, bb_u, bb_l = stats_engine.bollinger_squeeze(df)
    smc_data = stats_engine.identify_smc_levels(df, lookback=smc_lookback)
    
    # LAST STATE
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    p_val = stats_engine.get_probability_density(last_row['Z_Score'])
    
    # HUD METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CURRENT PRICE", f"{last_row['Close']:,.2f}")
    m2.metric("Z-SCORE", f"{last_row['Z_Score']:.2f}σ", 
              f"{last_row['Z_Score'] - prev_row['Z_Score']:.2f}Δ")
    
    prob_class = "high-conf" if p_val < 0.05 else "low-conf"
    m3.markdown(f"**MEAN REVERSION PROB**<br><span class='prob-badge {prob_class}'>{100*(1-p_val):.2f}%</span>", unsafe_allow_html=True)
    
    m4.metric("SQUEEZE STATUS", "COMPRESSED" if squeeze_on.iloc[-1] else "VOLATILE", 
              delta="Alert" if squeeze_on.iloc[-1] else None)

    # MAIN VISUALIZATION (THE GRID)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("LIQUIDITY FLOW & SMC", "STATISTICAL Z-SCORE", "MONEY FLOW INDEX")
    )

    # 1. Price + SMC
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price Action", opacity=0.8
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=bb_u, line=dict(color='rgba(0, 255, 187, 0.2)', width=1), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_l, line=dict(color='rgba(0, 255, 187, 0.2)', width=1), fill='tonexty', name="BB Lower"), row=1, col=1)

    # SMC Order Blocks (Represented as markers for clarity in lab view)
    obs = smc_data['order_blocks']
    fig.add_trace(go.Scatter(
        x=obs.index, y=obs['High'], mode='markers', 
        marker=dict(symbol='diamond', size=8, color='#7d00ff'),
        name="Inst. Order Block"
    ), row=1, col=1)

    # 2. Z-Score Subplot
    z_colors = ['#ff1155' if abs(z) > 2 else '#00ffbb' for z in df['Z_Score']]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Z_Score'], marker_color=z_colors, name="Z-Score"
    ), row=2, col=1)
    # Reversion Thresholds
    fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=0, line_color="white", row=2, col=1)

    # 3. MFI Subplot
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MFI'], line=dict(color='cyan', width=2), fill='tozeroy', name="Money Flow"
    ), row=3, col=1)

    fig.update_layout(
        height=900, 
        template="plotly_dark", 
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # BROADCAST & LAB REPORTING SECTION
    st.markdown("---")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### 🧪 Signal Generation")
        
        # Confluence Logic
        direction = "NEUTRAL"
        if last_row['Z_Score'] > 2 and last_row['MFI'] > 80:
            direction = "SHORT (Mean Reversion)"
        elif last_row['Z_Score'] < -2 and last_row['MFI'] < 20:
            direction = "LONG (Mean Reversion)"
            
        smc_note = "FVG Identified" if smc_data['bull_fvg'].iloc[-1] or smc_data['bear_fvg'].iloc[-1] else "Structure Neutral"
        
        signal_data = {
            "symbol": symbol,
            "tf": timeframe,
            "direction": direction,
            "p_value": p_val,
            "z_score": last_row['Z_Score'],
            "squeeze": squeeze_on.iloc[-1],
            "mfi": last_row['MFI'],
            "smc_note": smc_note,
            "price": f"{last_row['Close']:,.2f}",
            "stop": f"{last_row['Close'] * (1.02 if 'SHORT' in direction else 0.98):,.2f}",
            "target": f"{df['Close'].rolling(z_window).mean().iloc[-1]:,.2f}"
        }

        with st.expander("PREVIEW BROADCAST MESSAGE", expanded=True):
            formatted_msg = BroadcastExpert.format_signal(signal_data)
            st.code(formatted_msg, language="markdown")
            
        if st.button("🚀 BROADCAST TO TELEGRAM"):
            success = BroadcastExpert.send_telegram(formatted_msg, tg_token, tg_chat)
            if success:
                st.success("Alpha Signal Dispatched Successfully.")
            else:
                st.error("Broadcast Failed. Check API Credentials.")

    with c2:
        st.markdown("### 🤖 The Quant Analyst")
        if ai_enabled:
            if st.button("GENERATE COMPREHENSIVE LAB REPORT"):
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                # Quantitative Prompting
                prompt = f"""
                You are The Quant. Analyze {symbol} on the {timeframe} chart.
                Metrics:
                - Z-Score: {last_row['Z_Score']:.2f}
                - P-Value: {p_val:.4f}
                - Money Flow: {last_row['MFI']:.1f}
                - Volatility Squeeze: {squeeze_on.iloc[-1]}
                - SMC Logic: {smc_note}
                
                Provide a highly scientific, data-heavy analysis. Do not use fluff. 
                Focus on the probability of mean reversion and liquidity sweeps.
                Speak in standard deviations and statistical significance.
                """
                
                with st.spinner("Processing Gaussian Distributions..."):
                    response = client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[{"role": "system", "content": "You are a Senior Quantitative Researcher."},
                                  {"role": "user", "content": prompt}]
                    )
                    st.markdown(f"<div style='background:#111; padding:20px; border-left:3px solid #7d00ff;'>{response.choices[0].message.content}</div>", unsafe_allow_html=True)
        else:
            st.info("AI Analysis module is offline. Provide OpenAI Key in secrets to initialize.")

    # FOOTER GRID
    st.markdown("---")
    st.markdown("<center style='font-family:JetBrains Mono; color:#444;'>QUANTLAB V2.0 | NULL HYPOTHESIS REJECTION ENGINE | © 2024</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# =============================================================================
# END OF SYSTEM
# =============================================================================
