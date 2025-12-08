
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime, time
import pandas_ta as ta
from openai import OpenAI

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="LSE Mining & Commodities AI Analyst",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Excellent UI"
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .stAlert {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #444;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS (MATH)
# ==========================================
def calculate_hma(series, length):
    return ta.hma(series, length=length)

def calculate_rma(series, length):
    return ta.rma(series, length=length)

def calculate_money_flow(df, length=14, smooth=3):
    # Normalized Money Flow Logic from Pine Script
    rsi_source = ta.rsi(df['Close'], length=length) - 50
    mf_vol = df['Volume'] / ta.sma(df['Volume'], length=length)
    money_flow = ta.ema(rsi_source * mf_vol, length=smooth)
    return money_flow

# ==========================================
# 3. INDICATOR LOGIC (CONVERTED FROM PINE)
# ==========================================

# --- A. MACRO RISK TRAFFIC LIGHT ---
@st.cache_data(ttl=3600) # Cache macro data for 1 hour
def get_macro_data():
    # Mapping Pine Tickers to Yahoo Finance
    tickers = {
        'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SPX': '^GSPC', 
        'NDX': '^IXIC', 'VIX': '^VIX', 'TLT': 'TLT', 
        'HYG': 'HYG', 'DXY': 'DX-Y.NYB', 'GOLD': 'GC=F', 
        'SILVER': 'SI=F', 'COPPER': 'HG=F', 'OIL': 'CL=F', 
        'US10Y': '^TNX', 'US02Y': '^IRX' # Approximation
    }
    
    data = yf.download(list(tickers.values()), period="1y", interval="1d")['Close']
    
    # Calculate Ratios
    # Clean data (ffill for different market holidays)
    data = data.ffill().dropna()
    
    # Ratios & Logic
    try:
        df = pd.DataFrame(index=data.index)
        
        # Helper to safely get columns
        def get_col(key):
            return data[tickers[key]]

        # Calculations
        df['SPY_TLT'] = get_col('SPX') / get_col('TLT') # Proxy using SPX
        df['HYG_TLT'] = get_col('HYG') / get_col('TLT')
        df['BTC_SPX'] = get_col('BTC') / get_col('SPX')
        df['NDX_SPX'] = get_col('NDX') / get_col('SPX')
        df['COPPER_GOLD'] = get_col('COPPER') / get_col('GOLD')
        df['BTC_DXY'] = get_col('BTC') / get_col('DXY')
        df['VIX'] = get_col('VIX')
        
        # Yield Curve (using TNX/IRX proxies - Note: TNX is yield * 10)
        df['YIELD_CURVE'] = get_col('US10Y') - get_col('US02Y') 

        # Regime Function (EMA Spread)
        def get_regime(series, invert=False):
            e20 = ta.ema(series, 20)
            e50 = ta.ema(series, 50)
            spread_pct = (e20 - e50) / e50 * 100
            signal = np.where(spread_pct > 0.05, 1, np.where(spread_pct < -0.05, -1, 0))
            return signal * -1 if invert else signal

        # Calculate Scores
        score = 0
        score += get_regime(df['SPY_TLT'])
        score += get_regime(df['HYG_TLT'])
        score += get_regime(df['BTC_SPX'])
        score += get_regime(df['NDX_SPX'])
        score += get_regime(df['COPPER_GOLD'])
        score += get_regime(df['BTC_DXY'])
        score += get_regime(df['VIX'], invert=True) # Rising VIX is bad
        
        # Yield Curve Logic (Simple Inversion check)
        yc_score = np.where(df['YIELD_CURVE'] > 0, 1, -1)
        score += yc_score
        
        last_score = score[-1]
        
        return last_score, df
    except Exception as e:
        return 0, None

# --- B. APEX TREND & LIQUIDITY ---
def calculate_apex_trend(df, len_main=55, mult=1.5):
    # Hull MA Baseline
    baseline = ta.hma(df['Close'], length=len_main)
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=len_main)
    
    upper = baseline + (atr * mult)
    lower = baseline - (atr * mult)
    
    # Trend State
    trend = pd.Series(0, index=df.index)
    # Vectorized approach for trend state (iterative needed for state persistence)
    curr_trend = 0
    trend_list = []
    
    for i in range(len(df)):
        close = df['Close'].iloc[i]
        u = upper.iloc[i]
        l = lower.iloc[i]
        
        if np.isnan(u):
            trend_list.append(0)
            continue
            
        if close > u:
            curr_trend = 1
        elif close < l:
            curr_trend = -1
        
        trend_list.append(curr_trend)
    
    df['Apex_Trend'] = trend_list
    df['Apex_Upper'] = upper
    df['Apex_Lower'] = lower
    return df

# --- C. EVWM (Elastic Volume Weighted Momentum) ---
def calculate_evwm(df, length=21, vol_smooth=5, mult=2.0):
    baseline = ta.hma(df['Close'], length=length)
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=length)
    
    elasticity = (df['Close'] - baseline) / atr
    
    # Volume Weighting
    rvol = df['Volume'] / ta.sma(df['Volume'], length=length)
    smooth_rvol = ta.sma(rvol, length=vol_smooth)
    final_force = np.sqrt(smooth_rvol)
    
    evwm = elasticity * final_force
    
    # Bands
    band_basis = ta.sma(evwm, length=length*2)
    band_dev = ta.stdev(evwm, length=length*2) * mult
    
    df['EVWM'] = evwm
    df['EVWM_Upper'] = band_basis + band_dev
    df['EVWM_Lower'] = band_basis - band_dev
    return df

# --- D. SUPPORT & RESISTANCE (Simplified Logic for Python Speed) ---
def calculate_sr(df, period=10):
    # Find Pivots
    df['Pivot_High'] = df['High'].rolling(period*2+1, center=True).max()
    df['Pivot_Low'] = df['Low'].rolling(period*2+1, center=True).min()
    
    # Identify where High == Pivot High
    is_pivot_high = (df['High'] == df['Pivot_High'])
    is_pivot_low = (df['Low'] == df['Pivot_Low'])
    
    # Get last 3 pivots
    res_levels = df[is_pivot_high]['High'].tail(3).values
    sup_levels = df[is_pivot_low]['Low'].tail(3).values
    
    return res_levels, sup_levels

# ==========================================
# 4. DATA FETCHING
# ==========================================
TICKERS = {
    "SGLN (iShares Physical Gold)": "SGLN.L",
    "SSLN (iShares Physical Silver)": "SSLN.L",
    "SPLT (iShares Physical Platinum)": "SPLT.L",
    "SPDM (iShares Physical Palladium)": "SPDM.L",
    "SILG (Global X Silver Miners)": "SILG.L",
    "GJGB (VanEck Junior Gold Miners)": "GJGB.L",
    "ESGP (HANetf Gold Miners)": "ESGP.L",
    "URJP (Sprott Junior Uranium)": "URJP.L",
    "COPP (Sprott Copper Miners)": "COPP.L",
    "SPGP (iShares Gold Producers)": "SPGP.L"
}

@st.cache_data(ttl=900) # 15 min cache for ticker data
def get_ticker_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d")
    return df

# ==========================================
# 5. MARKET CLOCK
# ==========================================
def render_clock():
    london_tz = pytz.timezone('Europe/London')
    now_london = datetime.now(london_tz)
    
    # LSE Hours: 08:00 - 16:30
    market_open = time(8, 0, 0)
    market_close = time(16, 30, 0)
    current_time = now_london.time()
    
    is_weekday = now_london.weekday() < 5
    is_open = is_weekday and (market_open <= current_time <= market_close)
    
    status_color = "green" if is_open else "red"
    status_text = "MARKET OPEN" if is_open else "MARKET CLOSED"
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("LSE Commodities & Miners Analyst")
    with col2:
        st.markdown(f"""
        <div style="text-align: right; padding: 10px; border: 2px solid {status_color}; border-radius: 10px;">
            <div style="font-size: 14px; color: #888;">London Time</div>
            <div style="font-size: 20px; font-weight: bold;">{now_london.strftime('%H:%M:%S')}</div>
            <div style="color: {status_color}; font-weight: bold;">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 6. AI ANALYST
# ==========================================
def generate_ai_report(ticker_name, df, macro_score, macro_state):
    
    # Extract latest metrics
    last_close = df['Close'].iloc[-1]
    last_apex = "BULLISH" if df['Apex_Trend'].iloc[-1] == 1 else "BEARISH"
    last_mf = df['Money_Flow'].iloc[-1]
    mf_state = "INFLOW" if last_mf > 0 else "OUTFLOW"
    last_evwm = df['EVWM'].iloc[-1]
    
    prompt = f"""
    You are an expert financial analyst for the London Stock Exchange.
    Analyze: {ticker_name}
    
    TECHNICAL DATA:
    1. Macro Risk Environment: Score {macro_score}/8 ({macro_state}). (High score = Risk On/Bullish, Low = Risk Off).
    2. Apex Trend Algorithm: Currently {last_apex}.
    3. Money Flow Matrix: Value is {last_mf:.2f} ({mf_state}).
    4. EVWM Momentum: Value is {last_evwm:.2f}.
    5. Price: {last_close:.2f} GBP/USD.
    
    Task:
    Provide a concise, professional trading report. 
    - Start with a "Sentiment Verdict" (Bullish/Bearish/Neutral).
    - Explain the confluence of the Macro score and the specific ticker technicals.
    - Highlight if Money Flow supports the Trend.
    - Keep it under 150 words.
    """
    
    try:
        client = OpenAI(api_key=st.session_state.get('openai_api_key'))
        response = client.chat.completions.create(
            model="gpt-4",  # Or gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Analysis Error: Please check API Key. {str(e)}"

# ==========================================
# 7. MAIN APP LAYOUT
# ==========================================

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # API Key Input
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        st.session_state['openai_api_key'] = api_key
    
    selected_name = st.selectbox("Select Ticker", list(TICKERS.keys()))
    selected_ticker = TICKERS[selected_name]
    
    st.markdown("---")
    st.info("Indicators Loaded:\n- Macro Risk Traffic Light\n- Apex Trend Master\n- Money Flow Matrix\n- EVWM Momentum\n- Ultimate S/R")

# Render Clock
render_clock()

# Macro Section
macro_score, macro_df = get_macro_data()
risk_state = "RISK ON (Bullish)" if macro_score >= 4 else ("RISK OFF (Bearish)" if macro_score <= -4 else "NEUTRAL")
risk_color = "green" if macro_score >= 4 else ("red" if macro_score <= -4 else "gray")

st.markdown(f"""
<div style="background-color: #262730; padding: 10px; border-radius: 5px; border-left: 5px solid {risk_color}; margin-bottom: 20px;">
    <h3>🌍 Global Macro Risk Score: {int(macro_score)} / 8</h3>
    <p>Regime: <b>{risk_state}</b> | Aggregates 8 signals (Yield Curve, VIX, BTC/SPX, etc.)</p>
</div>
""", unsafe_allow_html=True)

# Ticker Analysis
if selected_ticker:
    df = get_ticker_data(selected_ticker)
    
    if df.empty:
        st.error("No data found for this ticker.")
    else:
        # --- Run Calculations ---
        df = calculate_apex_trend(df)
        df['Money_Flow'] = calculate_money_flow(df)
        df = calculate_evwm(df)
        res_levels, sup_levels = calculate_sr(df)
        
        # --- Visualization (Plotly) ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=("Price & Apex Trend", "Money Flow Matrix", "EVWM Momentum"))

        # Row 1: Price + Apex Cloud + S/R
        # Candle
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        
        # Apex Cloud
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)', line=dict(width=0), name="Apex Cloud", hoverinfo='skip'), row=1, col=1)
        
        # S/R Lines (Last 3)
        for r in res_levels:
            fig.add_hline(y=r, line_dash="dash", line_color="red", row=1, col=1, opacity=0.5)
        for s in sup_levels:
            fig.add_hline(y=s, line_dash="dash", line_color="green", row=1, col=1, opacity=0.5)

        # Row 2: Money Flow
        colors = ['green' if x > 0 else 'red' for x in df['Money_Flow']]
        fig.add_trace(go.Bar(x=df.index, y=df['Money_Flow'], marker_color=colors, name="Money Flow"), row=2, col=1)
        
        # Row 3: EVWM
        fig.add_trace(go.Scatter(x=df.index, y=df['EVWM'], line_color='white', name="EVWM"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EVWM_Upper'], line_color='gray', line_dash='dot', showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EVWM_Lower'], line_color='gray', line_dash='dot', showlegend=False), row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # --- AI Report Section ---
        st.markdown("### 🤖 AI Analyst Report")
        
        col_ai, col_but = st.columns([4, 1])
        with col_but:
            if st.button("Generate Report"):
                if not st.session_state.get('openai_api_key'):
                    st.warning("Please enter OpenAI API Key in Sidebar")
                else:
                    with st.spinner("Analyzing technicals..."):
                        report = generate_ai_report(selected_name, df, int(macro_score), risk_state)
                        st.session_state['report'] = report
        
        if st.session_state.get('report'):
            st.markdown(f"""
            <div style="background-color: #111; padding: 20px; border-radius: 10px; border: 1px solid #444;">
                {st.session_state['report']}
            </div>
            """, unsafe_allow_html=True)
