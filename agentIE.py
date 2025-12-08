import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS (MATH - NATIVE PANDAS)
# ==========================================
def calculate_hma(series, length):
    # Ensure series is a Series (handle potential 1-col DataFrame)
    if isinstance(series, pd.DataFrame):
        series = series.squeeze()
        
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    def wma(s, l):
        weights = np.arange(1, l + 1)
        # Apply WMA
        return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    wmaf = wma(series, half_length)
    wmas = wma(series, length)
    diff = 2 * wmaf - wmas
    return wma(diff, sqrt_length)

def calculate_rma(series, length):
    if isinstance(series, pd.DataFrame):
        series = series.squeeze()
    return series.ewm(alpha=1/length, adjust=False).mean()

def calculate_money_flow(df, length=14, smooth=3):
    # Ensure inputs are Series
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = calculate_rma(gain, length)
    avg_loss = calculate_rma(loss, length)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi_source = rsi - 50
    mf_vol = volume / volume.rolling(length).mean()
    money_flow = (rsi_source * mf_vol).ewm(span=smooth, adjust=False).mean()
    return money_flow

# ==========================================
# 3. INDICATOR LOGIC
# ==========================================

# --- A. MACRO RISK TRAFFIC LIGHT ---
@st.cache_data(ttl=3600)
def get_macro_data():
    tickers = {
        'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SPX': '^GSPC', 
        'NDX': '^IXIC', 'VIX': '^VIX', 'TLT': 'TLT', 
        'HYG': 'HYG', 'DXY': 'DX-Y.NYB', 'GOLD': 'GC=F', 
        'SILVER': 'SI=F', 'COPPER': 'HG=F', 'OIL': 'CL=F', 
        'US10Y': '^TNX', 'US02Y': '^IRX'
    }
    
    # Download and clean
    data = yf.download(list(tickers.values()), period="1y", interval="1d")['Close']
    data = data.ffill().dropna()
    
    try:
        df = pd.DataFrame(index=data.index)
        
        def get_col(key):
            # Safe column extraction
            col = tickers[key]
            if col in data.columns:
                return data[col]
            else:
                return pd.Series(0, index=data.index)

        # Ratios
        df['SPY_TLT'] = get_col('SPX') / get_col('TLT')
        df['HYG_TLT'] = get_col('HYG') / get_col('TLT')
        df['BTC_SPX'] = get_col('BTC') / get_col('SPX')
        df['NDX_SPX'] = get_col('NDX') / get_col('SPX')
        df['COPPER_GOLD'] = get_col('COPPER') / get_col('GOLD')
        df['BTC_DXY'] = get_col('BTC') / get_col('DXY')
        df['VIX'] = get_col('VIX')
        df['YIELD_CURVE'] = get_col('US10Y') - get_col('US02Y') 

        def get_regime(series, invert=False):
            e20 = series.ewm(span=20, adjust=False).mean()
            e50 = series.ewm(span=50, adjust=False).mean()
            spread_pct = (e20 - e50) / e50 * 100
            signal = np.where(spread_pct > 0.05, 1, np.where(spread_pct < -0.05, -1, 0))
            return signal * -1 if invert else signal

        # Score Calculation
        score = 0
        score += get_regime(df['SPY_TLT'])
        score += get_regime(df['HYG_TLT'])
        score += get_regime(df['BTC_SPX'])
        score += get_regime(df['NDX_SPX'])
        score += get_regime(df['COPPER_GOLD'])
        score += get_regime(df['BTC_DXY'])
        score += get_regime(df['VIX'], invert=True)
        
        yc_score = np.where(df['YIELD_CURVE'] > 0, 1, -1)
        score += yc_score
        
        last_score = score[-1]
        # Ensure scalar
        if hasattr(last_score, "item"):
            last_score = last_score.item()
            
        return last_score, df
    except Exception as e:
        st.error(f"Macro Data Error: {e}")
        return 0, None

# --- B. APEX TREND & LIQUIDITY ---
def calculate_apex_trend(df, len_main=55, mult=1.5):
    # 1. Clean Inputs
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    
    # 2. Calculate HMA Baseline
    baseline = calculate_hma(close, length=len_main)
    
    # 3. Calculate ATR (Manual Native)
    h_l = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low - close.shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    atr = calculate_rma(tr, len_main)
    
    # 4. Bands
    upper = baseline + (atr * mult)
    lower = baseline - (atr * mult)
    
    # 5. Trend Logic (Loop)
    trend_list = []
    
    # We must iterate over values safely
    # Convert series to numpy arrays for safe scalar indexing
    close_vals = close.values
    upper_vals = upper.values
    lower_vals = lower.values
    
    current_trend = 0
    
    for i in range(len(df)):
        c = close_vals[i]
        u = upper_vals[i]
        l = lower_vals[i]
        
        if np.isnan(u):
            trend_list.append(0)
            continue
            
        if c > u:
            current_trend = 1
        elif c < l:
            current_trend = -1
        # Else keep previous state
        
        trend_list.append(current_trend)
    
    df['Apex_Trend'] = trend_list
    df['Apex_Upper'] = upper
    df['Apex_Lower'] = lower
    return df

# --- C. EVWM ---
def calculate_evwm(df, length=21, vol_smooth=5, mult=2.0):
    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    
    baseline = calculate_hma(close, length=length)
    
    # ATR Calc
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    h_l = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low - close.shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    atr = calculate_rma(tr, length)
    
    elasticity = (close - baseline) / atr
    
    rvol = volume / volume.rolling(length).mean()
    smooth_rvol = rvol.rolling(vol_smooth).mean()
    final_force = np.sqrt(smooth_rvol)
    
    evwm = elasticity * final_force
    
    band_basis = evwm.rolling(length*2).mean()
    band_dev = evwm.rolling(length*2).std() * mult
    
    df['EVWM'] = evwm
    df['EVWM_Upper'] = band_basis + band_dev
    df['EVWM_Lower'] = band_basis - band_dev
    return df

# --- D. S/R ---
def calculate_sr(df, period=10):
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    
    df['Pivot_High'] = high.rolling(period*2+1, center=True).max()
    df['Pivot_Low'] = low.rolling(period*2+1, center=True).min()
    
    # Explicitly flatten boolean series
    is_pivot_high = (high == df['Pivot_High'])
    is_pivot_low = (low == df['Pivot_Low'])
    
    res_levels = df[is_pivot_high]['High'].tail(3).values
    sup_levels = df[is_pivot_low]['Low'].tail(3).values
    
    return res_levels, sup_levels

# ==========================================
# 4. DATA FETCHING (FIXED)
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

@st.cache_data(ttl=900)
def get_ticker_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d")
    
    # --- CRITICAL FIX: FLATTEN MULTI-INDEX ---
    # yfinance often returns columns like ('Close', 'SGLN.L')
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Drop the Ticker level, keeping just Open, High, Close etc.
            df.columns = df.columns.droplevel(1) 
        except:
            pass
            
    # Ensure it's not empty
    if df.empty:
        return df
        
    return df

# ==========================================
# 5. AI ANALYST
# ==========================================
def generate_ai_report(ticker_name, df, macro_score, macro_state):
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
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Analysis Error: Please check API Key. {str(e)}"

# ==========================================
# 6. MAIN APP LAYOUT
# ==========================================
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        st.session_state['openai_api_key'] = api_key
    
    selected_name = st.selectbox("Select Ticker", list(TICKERS.keys()))
    selected_ticker = TICKERS[selected_name]
    st.markdown("---")
    st.info("Indicators Loaded (Native Python)")

# Macro Section
macro_score, macro_df = get_macro_data()
risk_state = "RISK ON (Bullish)" if macro_score >= 4 else ("RISK OFF (Bearish)" if macro_score <= -4 else "NEUTRAL")
risk_color = "green" if macro_score >= 4 else ("red" if macro_score <= -4 else "gray")

st.markdown(f"""
<div style="background-color: #262730; padding: 10px; border-radius: 5px; border-left: 5px solid {risk_color}; margin-bottom: 20px;">
    <h3>🌍 Global Macro Risk Score: {int(macro_score)} / 8</h3>
    <p>Regime: <b>{risk_state}</b> | Aggregates 8 signals</p>
</div>
""", unsafe_allow_html=True)

# Ticker Analysis
if selected_ticker:
    df = get_ticker_data(selected_ticker)
    
    if df.empty:
        st.error("No data found for this ticker.")
    else:
        # Run Calculations
        df = calculate_apex_trend(df)
        df['Money_Flow'] = calculate_money_flow(df)
        df = calculate_evwm(df)
        res_levels, sup_levels = calculate_sr(df)
        
        # Plot
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=("Price & Apex Trend", "Money Flow Matrix", "EVWM Momentum"))

        # 1. Price
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)', line=dict(width=0), name="Apex Cloud", hoverinfo='skip'), row=1, col=1)
        
        # S/R
        for r in res_levels:
            fig.add_hline(y=r, line_dash="dash", line_color="red", row=1, col=1, opacity=0.5)
        for s in sup_levels:
            fig.add_hline(y=s, line_dash="dash", line_color="green", row=1, col=1, opacity=0.5)

        # 2. Money Flow
        colors = ['green' if x > 0 else 'red' for x in df['Money_Flow']]
        fig.add_trace(go.Bar(x=df.index, y=df['Money_Flow'], marker_color=colors, name="Money Flow"), row=2, col=1)
        
        # 3. EVWM
        fig.add_trace(go.Scatter(x=df.index, y=df['EVWM'], line_color='white', name="EVWM"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EVWM_Upper'], line_color='gray', line_dash='dot', showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EVWM_Lower'], line_color='gray', line_dash='dot', showlegend=False), row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Report
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
