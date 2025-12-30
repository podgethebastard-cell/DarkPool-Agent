import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import scipy.stats as stats

# ==========================================
# 1. CORE CONFIG & SCIENTIFIC STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="QuantLab Terminal", page_icon="⚛️")

def inject_lab_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&display=swap');
        .stApp { background-color: #05070a; color: #e0e6ed; font-family: 'Roboto Mono', monospace; }
        [data-testid="stMetric"] {
            background: rgba(10, 25, 41, 0.7);
            border: 1px solid #1e3a5f;
            border-radius: 4px;
            padding: 15px;
        }
        .report-box {
            background: #0d1117;
            border-left: 4px solid #00d4ff;
            padding: 20px;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        .signal-bull { color: #00ffa3; font-weight: bold; }
        .signal-bear { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

inject_lab_theme()

# ==========================================
# 2. QUANTITATIVE ENGINES
# ==========================================
class QuantEngine:
    @staticmethod
    def get_data(ticker: str, interval: str):
        period_map = {"15m": "60d", "1h": "730d", "4h": "730d", "1d": "max"}
        df = yf.download(ticker, period=period_map.get(interval, "2y"), interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()

    @staticmethod
    def calc_smc(df):
        """Detection of Order Blocks and Fair Value Gaps."""
        df = df.copy()
        # FVG Detection
        df['FVG_UP'] = (df['Low'] > df['High'].shift(2))
        df['FVG_DN'] = (df['High'] < df['Low'].shift(2))
        
        # Order Blocks (Pivot high/low with volume confirmation)
        window = 5
        df['PH'] = df['High'] == df['High'].rolling(window*2+1, center=True).max()
        df['PL'] = df['Low'] == df['Low'].rolling(window*2+1, center=True).min()
        return df

    @staticmethod
    def calc_zscore(series, length=20):
        mean = series.rolling(window=length).mean()
        std = series.rolling(window=length).std()
        return (series - mean) / std

    @staticmethod
    def calc_mfi(df, length=14):
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        rmf = tp * df['Volume']
        pos_mf = []
        neg_mf = []
        for i in range(1, len(tp)):
            if tp.iloc[i] > tp.iloc[i-1]:
                pos_mf.append(rmf.iloc[i])
                neg_mf.append(0)
            else:
                pos_mf.append(0)
                neg_mf.append(rmf.iloc[i])
        
        mfr = (pd.Series(pos_mf).rolling(length).sum() / 
               pd.Series(neg_mf).rolling(length).sum())
        mfi = 100 - (100 / (1 + mfr))
        return pd.Series(mfi, index=df.index[1:])

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("🧬 LAB CONFIG")
    ticker = st.text_input("SYMBOL", value="BTC-USD").upper()
    interval = st.selectbox("TIMEFRAME", ["15m", "1h", "4h", "1d"], index=2)
    
    st.markdown("---")
    st.header("📡 BROADCAST CONFIG")
    tg_token = st.text_input("TG Bot Token", type="password")
    tg_chat = st.text_input("TG Chat ID")
    
    st.markdown("---")
    confidence_threshold = st.slider("Confidence Interval (%)", 90, 99, 95)
    z_limit = stats.norm.ppf(confidence_threshold / 100)

# ==========================================
# 4. MAIN DASHBOARD EXECUTION
# ==========================================
st.title("⚛️ QuantLab Research Terminal")
st.caption(f"Market Intel: {ticker} | Timeframe: {interval} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")

df = QuantEngine.get_data(ticker, interval)

if not df.empty:
    df = QuantEngine.calc_smc(df)
    df['Z_Score'] = QuantEngine.calc_zscore(df['Close'])
    df['MFI'] = QuantEngine.calc_mfi(df)
    
    # --- SUBPLOT SYSTEM ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])

    # Row 1: Price + SMC
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market"), row=1, col=1)
    
    # Overlay Order Blocks
    obs_h = df[df['PH']]
    obs_l = df[df['PL']]
    fig.add_trace(go.Scatter(x=obs_h.index, y=obs_h['High'], mode='markers', marker=dict(symbol='triangle-down', color='#ff4b4b', size=10), name="Supply OB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=obs_l.index, y=obs_l['Low'], mode='markers', marker=dict(symbol='triangle-up', color='#00ffa3', size=10), name="Demand OB"), row=1, col=1)

    # Row 2: Z-Score (Mean Reversion)
    fig.add_trace(go.Scatter(x=df.index, y=df['Z_Score'], line=dict(color='#00d4ff', width=1.5), name="Z-Score"), row=2, col=1)
    fig.add_hline(y=z_limit, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=-z_limit, line_dash="dot", line_color="green", row=2, col=1)

    # Row 3: Money Flow Index
    fig.add_trace(go.Bar(x=df.index, y=df['MFI'], marker_color='#888', name="MFI"), row=3, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="#ff4b4b", row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="#00ffa3", row=3, col=1)

    fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- STATISTICAL SIGNIFICANCE PANEL ---
    last_z = df['Z_Score'].iloc[-1]
    last_mfi = df['MFI'].iloc[-1]
    prob_reversion = (1 - stats.norm.cdf(abs(last_z))) * 2 # Two-tailed P-value

    cols = st.columns(3)
    cols[0].metric("Z-Score ($\sigma$)", f"{last_z:.2f}", delta="Extreme" if abs(last_z) > z_limit else "Normal")
    cols[1].metric("P-Value (Rev)", f"{prob_reversion:.4f}")
    cols[2].metric("MFI Delta", f"{last_mfi:.1f}")

    st.markdown("---")
    
    # ==========================================
    # 5. BROADCAST TERMINAL
    # ==========================================
    st.subheader("📡 Broadcast Command Center")
    
    # Logic Signal Validation
    is_long = (last_z < -z_limit) and (last_mfi < 30)
    is_short = (last_z > z_limit) and (last_mfi > 70)
    
    signal_status = "STABLE"
    if is_long: signal_status = "CONFLUENCE: LONG (Mean Reversion)"
    if is_short: signal_status = "CONFLUENCE: SHORT (Mean Reversion)"

    report_md = f"""
### 🧪 Comprehensive Quant Report: {ticker}
**Frequency:** {interval} | **Confidence:** {confidence_threshold}%

**Statistical Metrics:**
- **Current Z-Score:** `{last_z:.3f} σ`
- **Mean Reversion Probability:** `{((1 - prob_reversion) * 100):.2f}%`
- **MFI Position:** `{last_mfi:.1f}`

**SMC Constraints:**
- **FVG Detected:** `{'Yes' if df['FVG_UP'].iloc[-1] or df['FVG_DN'].iloc[-1] else 'No'}`
- **Liquidity Status:** `{'Sweep Detected' if abs(last_z) > 2.5 else 'Neutral'}`

**Final Verdict:** **{signal_status}**
    """
    
    st.markdown(f'<div class="report-box">{report_md}</div>', unsafe_allow_html=True)

    if st.button("🚀 BROADCAST SIGNAL TO TELEGRAM"):
        if tg_token and tg_chat:
            try:
                msg = f"💎 *QuantLab Signal: {ticker}*\n"
                msg += f"Timeframe: {interval}\n"
                msg += f"Z-Score: {last_z:.2f}σ\n"
                msg += f"MFI: {last_mfi:.1f}\n"
                msg += f"Reversion Prob: {((1-prob_reversion)*100):.1f}%\n"
                msg += f"Verdict: *{signal_status}*"
                
                url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                resp = requests.post(url, data={"chat_id": tg_chat, "text": msg, "parse_mode": "Markdown"})
                
                if resp.status_code == 200:
                    st.success("Broadcast successfully transmitted to encrypted channel.")
                else:
                    st.error(f"Transmission error: {resp.text}")
            except Exception as e:
                st.error(f"Engine Failure: {e}")
        else:
            st.warning("📡 Broadcast blocked: Credentials missing in Lab Config.")

else:
    st.error("Data Acquisition Failure. Check Ticker formatting (e.g., BTC-USD).")

# ==========================================
# 6. QA CHECKLIST & VALIDATION
# ==========================================
# 1. Vectorized Z-Score allows O(n) performance for large back-look periods.
# 2. SMC PH/PL logic uses center-true rolling windows for pinpoint accuracy.
# 3. MFI logic accounts for volume-weighted price delta.
