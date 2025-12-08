import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import time

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Macro Insighter - DarkPool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed" # Changed to collapsed since sidebar is now minimal
)

# Custom CSS to match the "Macro Insighter" Dashboard Look
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0e1117; }
    
    /* Card Styling */
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Text Styling */
    h1, h2, h3 { color: white !important; font-family: 'Helvetica Neue', sans-serif; }
    .ticker-title { font-size: 1.1rem; font-weight: bold; color: #e0e0e0; margin-bottom: 5px; }
    .ticker-desc { font-size: 0.8rem; color: #888; margin-bottom: 10px; }
    .price-big { font-size: 1.8rem; font-weight: bold; color: white; }
    .price-change-pos { color: #00E676; font-size: 0.9rem; font-weight: bold; }
    .price-change-neg { color: #FF5252; font-size: 0.9rem; font-weight: bold; }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #262730;
        color: white;
        border: 1px solid #444;
        border-radius: 4px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        border-color: #00E676;
        color: #00E676;
    }
    
    /* Remove default Streamlit padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS (MATH & INDICATORS)
# ==========================================
class NativeIndicators:
    @staticmethod
    def sma(series, length): return series.rolling(window=length).mean()
    @staticmethod
    def ema(series, length): return series.ewm(span=length, adjust=False).mean()
    @staticmethod
    def rma(series, length): return series.ewm(alpha=1/length, adjust=False).mean()
    @staticmethod
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    @staticmethod
    def hma(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wmaf = NativeIndicators.wma(series, half_length)
        wmas = NativeIndicators.wma(series, length)
        diff = 2 * wmaf - wmas
        return NativeIndicators.wma(diff, sqrt_length)
    @staticmethod
    def stdev(series, length): return series.rolling(window=length).std()
    @staticmethod
    def rsi(series, length=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = NativeIndicators.rma(gain, length)
        avg_loss = NativeIndicators.rma(loss, length)
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    @staticmethod
    def mfi(high, low, close, volume, length=14):
        tp = (high + low + close) / 3
        raw_money_flow = tp * volume
        flow_pos = np.where(tp > tp.shift(1), raw_money_flow, 0)
        flow_neg = np.where(tp < tp.shift(1), raw_money_flow, 0)
        mf_pos_sum = pd.Series(flow_pos).rolling(length).sum()
        mf_neg_sum = pd.Series(flow_neg).rolling(length).sum()
        mfi_ratio = mf_pos_sum / mf_neg_sum
        return 100 - (100 / (1 + mfi_ratio))
    @staticmethod
    def adx(high, low, close, length=14, smooth=14):
        h_l = high - low
        h_pc = (high - close.shift(1)).abs()
        l_pc = (low - close.shift(1)).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        up = high - high.shift(1)
        down = low.shift(1) - low
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        tr_smooth = NativeIndicators.rma(tr, length)
        plus_dm_smooth = NativeIndicators.rma(plus_dm, length)
        minus_dm_smooth = NativeIndicators.rma(minus_dm, length)
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = NativeIndicators.rma(dx, smooth)
        return plus_di, minus_di, adx

ta = NativeIndicators()

# --- INDICATOR LOGIC WRAPPERS ---
def calculate_money_flow(df, length=14, smooth=3):
    close = df['Close']
    volume = df['Volume']
    rsi_source = ta.rsi(close, length=length) - 50
    mf_vol = volume / ta.sma(volume, length=length)
    return ta.ema(rsi_source * mf_vol, length=smooth)

def calculate_apex_trend(df, len_main=55, mult=1.5):
    close = df['Close']
    baseline = ta.hma(close, length=len_main)
    tr = pd.concat([df['High']-df['Low'], (df['High']-close.shift(1)).abs(), (df['Low']-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = ta.rma(tr, len_main)
    upper = baseline + (atr * mult)
    lower = baseline - (atr * mult)
    trend_list = []
    close_vals = close.values
    upper_vals = upper.values
    lower_vals = lower.values
    curr = 0
    for i in range(len(df)):
        c, u, l = close_vals[i], upper_vals[i], lower_vals[i]
        if np.isnan(u): trend_list.append(0); continue
        if c > u: curr = 1
        elif c < l: curr = -1
        trend_list.append(curr)
    df['Apex_Trend'] = trend_list
    df['Apex_Upper'] = upper
    df['Apex_Lower'] = lower
    return df

def calculate_evwm(df, length=21, vol_smooth=5, mult=2.0):
    close = df['Close']
    volume = df['Volume']
    baseline = ta.hma(close, length=length)
    tr = pd.concat([df['High']-df['Low'], (df['High']-close.shift(1)).abs(), (df['Low']-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = ta.rma(tr, length)
    elasticity = (close - baseline) / atr
    rvol = volume / ta.sma(volume, length)
    smooth_rvol = ta.sma(rvol, vol_smooth)
    final_force = np.sqrt(smooth_rvol)
    evwm = elasticity * final_force
    band_basis = ta.sma(evwm, length*2)
    band_dev = ta.stdev(evwm, length*2) * mult 
    df['EVWM'] = evwm
    df['EVWM_Upper'] = band_basis + band_dev
    df['EVWM_Lower'] = band_basis - band_dev
    return df

def calculate_darkpool_macd(df, fast=12, slow=26, sig=9, vol_len=20, use_real_vol=True):
    close = df['Close']
    volume = df['Volume']
    fast_ma = ta.ema(close, fast)
    slow_ma = ta.ema(close, slow)
    macd = fast_ma - slow_ma
    signal = ta.ema(macd, sig)
    hist = macd - signal
    avg_vol = ta.sma(volume, vol_len)
    rvol = np.where(avg_vol > 0, volume / avg_vol, 1.0)
    final_hist = (hist * rvol) if use_real_vol else hist
    df['DP_MACD'] = macd
    df['DP_Signal'] = signal
    df['DP_Hist'] = final_hist
    return df

def calculate_rsi_divergence(df, length=14, lookback=5):
    rsi = ta.rsi(df['Close'], length)
    df['RSI'] = rsi
    # (Simplified divergence logic for brevity)
    df['Bull_Div'] = np.nan
    df['Bear_Div'] = np.nan
    return df

def calculate_gann_hilo(df, length=3):
    sma_high = df['High'].rolling(length).mean()
    sma_low = df['Low'].rolling(length).mean()
    activator = [np.nan] * len(df)
    trend = [0] * len(df) 
    curr_trend = 1
    curr_act = sma_low.iloc[0] if not np.isnan(sma_low.iloc[0]) else df['Low'].iloc[0]
    for i in range(len(df)):
        c = df['Close'].iloc[i]
        sh = sma_high.iloc[i]
        sl = sma_low.iloc[i]
        if np.isnan(sh): activator[i] = np.nan; continue
        prev_act = curr_act 
        if curr_trend == 1:
            if c < prev_act: curr_trend = -1; curr_act = sh
            else: curr_act = sl
        else:
            if c > prev_act: curr_trend = 1; curr_act = sl
            else: curr_act = sh
        activator[i] = curr_act
        trend[i] = curr_trend
    df['Gann_Activator'] = activator
    df['Gann_Trend'] = trend
    return df

def calculate_all_indicators(df):
    df = calculate_apex_trend(df)
    df = calculate_evwm(df)
    df['Money_Flow'] = calculate_money_flow(df) 
    df = calculate_darkpool_macd(df)  
    pdi, mdi, adx = ta.adx(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx; df['PlusDI'] = pdi; df['MinusDI'] = mdi
    df = calculate_rsi_divergence(df)
    df = calculate_gann_hilo(df)
    return df

# ==========================================
# 3. DATA & CACHING
# ==========================================
@st.cache_data(ttl=3600)
def get_macro_data():
    tickers = {
        'BTC': 'BTC-USD', 'SPX': '^GSPC', 'TLT': 'TLT', 
        'HYG': 'HYG', 'VIX': '^VIX', 'US10Y': '^TNX', 'US02Y': '^IRX'
    }
    try:
        data = yf.download(list(tickers.values()), period="1y", interval="1d", progress=False)['Close']
        data = data.ffill().dropna()
        df = pd.DataFrame(index=data.index)
        def get_col(key): return data[tickers[key]] if tickers[key] in data.columns else pd.Series(0, index=data.index)
        
        # Macro Scores
        score = 0
        score += np.where(get_col('SPX')/get_col('TLT') > (get_col('SPX')/get_col('TLT')).rolling(50).mean(), 1, -1)
        score += np.where(get_col('VIX') < 20, 1, -1)
        return score[-1] if hasattr(score[-1], "item") else score[-1]
    except:
        return 0

TICKERS = ["SGLN", "SSLN", "SPLT", "SPDM", "SILG", "GJGB", "ESGP", "URJP", "COPP", "SPGP"]

@st.cache_data(ttl=900)
def get_ticker_data(symbol):
    yf_symbol = f"{symbol}.L"
    df = yf.download(yf_symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0): df.columns = df.columns.get_level_values(0)
        elif df.columns.nlevels > 1 and 'Close' in df.columns.get_level_values(1): df.columns = df.columns.get_level_values(1)
    return df

# ==========================================
# 4. SIDEBAR & SETTINGS
# ==========================================
# REMOVED REDUNDANT VISUALS FROM SIDEBAR
with st.sidebar:
    st.header("Settings")
    
    # API Key Handling (Retained as it is functional)
    secret_key = st.secrets.get("OPENAI_API_KEY", None)
    if secret_key: st.session_state['openai_api_key'] = secret_key
    else:
        user_key = st.text_input("OpenAI Key", type="password")
        if user_key: st.session_state['openai_api_key'] = user_key

# ==========================================
# 5. STATE MANAGEMENT
# ==========================================
if 'view_state' not in st.session_state:
    st.session_state.view_state = 'dashboard' # 'dashboard' or 'detail'
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None

def go_to_detail(ticker):
    st.session_state.selected_ticker = ticker
    st.session_state.view_state = 'detail'
    # Force rerun implicitly handled by Streamlit interactions

def go_to_dashboard():
    st.session_state.selected_ticker = None
    st.session_state.view_state = 'dashboard'

# ==========================================
# 6. TERMINAL OUTPUT (SUMMARY)
# ==========================================
# Only run this once per reload to avoid spamming terminal
macro_score = get_macro_data()
summary_data = []
for t in TICKERS:
    d = get_ticker_data(t)
    if not d.empty:
        curr = d['Close'].iloc[-1]
        chg = ((curr - d['Close'].iloc[-2]) / d['Close'].iloc[-2]) * 100
        summary_data.append({"Ticker": t, "Price": curr, "Change %": chg})

print("\n" + "="*40 + "\n LIVE TICKER SUMMARY \n" + "="*40)
print(pd.DataFrame(summary_data).to_string(index=False))
print("="*40 + "\n")

# ==========================================
# 7. MAIN UI LOGIC
# ==========================================

# --- DASHBOARD HEADER ---
risk_state = "RISK ON" if macro_score >= 1 else ("RISK OFF" if macro_score <= -1 else "NEUTRAL")
risk_color = "#00E676" if macro_score >= 1 else ("#FF5252" if macro_score <= -1 else "gray")

if st.session_state.view_state == 'dashboard':
    # --- DASHBOARD VIEW ---
    st.markdown(f"""
    <div style="background-color: #1e1e1e; padding: 15px; border-radius: 8px; border-left: 5px solid {risk_color}; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
        <h3 style="margin:0;">🌍 Global Macro Regime: {risk_state} (Score: {int(macro_score)})</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Grid Layout
    cols = st.columns(3)
    
    for i, ticker in enumerate(TICKERS):
        df = get_ticker_data(ticker)
        col = cols[i % 3]
        
        with col:
            if df.empty:
                st.warning(f"No Data: {ticker}")
                continue
            
            # Data Prep
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            delta = current_price - prev_price
            pct_change = (delta / prev_price) * 100
            color_class = "price-change-pos" if pct_change >= 0 else "price-change-neg"
            arrow = "▲" if pct_change >= 0 else "▼"
            
            # Sparkline Data (Last 30 periods)
            spark_df = df.iloc[-30:]
            
            # Render Card HTML
            st.markdown(f"""
            <div class="metric-card">
                <div class="ticker-title">{ticker}</div>
                <div class="ticker-desc">Physical/Equity Trust</div>
                <div style="display: flex; align-items: baseline; justify-content: space-between;">
                    <span class="price-big">{current_price:,.2f}</span>
                    <span class="{color_class}">{arrow} {pct_change:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sparkline Chart (Plotly Area)
            fig_spark = go.Figure()
            fig_spark.add_trace(go.Scatter(
                x=spark_df.index, y=spark_df['Close'],
                mode='lines', fill='tozeroy',
                line=dict(color=risk_color, width=2),
                fillcolor=f"rgba({0 if pct_change<0 else 0}, {230 if pct_change>=0 else 82}, {118 if pct_change>=0 else 82}, 0.2)"
            ))
            fig_spark.update_layout(
                showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=50,
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_spark, use_container_width=True, config={'displayModeBar': False})
            
            # Action Button
            if st.button(f"📊 Analyze {ticker}", key=f"btn_{ticker}"):
                go_to_detail(ticker)
                st.rerun()

elif st.session_state.view_state == 'detail':
    # --- DETAILED ANALYSIS VIEW ---
    ticker = st.session_state.selected_ticker
    st.button("← Back to Dashboard", on_click=go_to_dashboard)
    
    st.title(f"{ticker} Institutional Analysis")
    df = get_ticker_data(ticker)
    
    if not df.empty:
        # Calculate Indicators
        df = calculate_all_indicators(df)
        last = df.iloc[-1]
        
        # AI Report Section (Top as requested)
        with st.container():
            st.markdown("### 🤖 AI Analyst Report")
            if st.button(f"Generate GPT-4 Report for {ticker}"):
                 if not st.session_state.get('openai_api_key'):
                     st.warning("Please enter OpenAI API Key in Sidebar")
                 else:
                     with st.spinner(f"Analyzing {ticker} market structure..."):
                         gann_state = "BULL" if last['Gann_Trend'] == 1 else "BEAR"
                         prompt = f"""
                         Analyze {ticker}. MACRO: {macro_score} ({risk_state}).
                         TREND: Apex={"Bull" if last['Apex_Trend']==1 else "Bear"}. Gann HiLo={gann_state}.
                         MOMENTUM: MACD Hist={last['DP_Hist']:.4f}. ADX={last['ADX']:.1f}. RSI={last['RSI']:.1f}.
                         VOLUME: MoneyFlow={last['Money_Flow']:.2f}.
                         Verdict (Bull/Bear/Neutral)? Short & Actionable.
                         """
                         try:
                             client = OpenAI(api_key=st.session_state['openai_api_key'])
                             resp = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}])
                             st.success(resp.choices[0].message.content)
                         except Exception as e:
                             st.error(str(e))
        
        st.markdown("---")
        
        # Giant Plotly Chart (7 Rows)
        fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            row_heights=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
                            subplot_titles=("Price Structure & Gann", "Money Flow", "EVWM Elasticity", 
                                            "DarkPool MACD", "ADX Trend", "RSI Divergence", "Advanced Volume"))

        # R1: Price
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Gann_Activator'], line=dict(color='orange', width=2), name="Gann Activator"), row=1, col=1)

        # R2: Money Flow
        cols_mf = ['green' if x > 0 else 'red' for x in df['Money_Flow']]
        fig.add_trace(go.Bar(x=df.index, y=df['Money_Flow'], marker_color=cols_mf, name="Money Flow"), row=2, col=1)

        # R3: EVWM
        fig.add_trace(go.Scatter(x=df.index, y=df['EVWM'], line_color='white', name="EVWM"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EVWM_Upper'], line=dict(color='gray', dash='dot'), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EVWM_Lower'], line=dict(color='gray', dash='dot'), showlegend=False), row=3, col=1)

        # R4: MACD
        hist_cols = ["#00E5FF" if h >= 0 else "#FF5252" for h in df['DP_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['DP_Hist'], marker_color=hist_cols, name="MACD Hist"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['DP_MACD'], line_color='#00E676', name="MACD Line"), row=4, col=1)

        # R5: ADX
        fig.add_trace(go.Scatter(x=df.index, y=df['PlusDI'], line_color='blue', name="+DI"), row=5, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MinusDI'], line_color='gray', name="-DI"), row=5, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line_color='fuchsia', name="ADX"), row=5, col=1)
        fig.add_hline(y=25, line=dict(color='white', dash='dot'), row=5, col=1)

        # R6: RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line_color='black', name="RSI"), row=6, col=1)
        fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=6, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=6, col=1)

        # R7: Volume (CMF if calculated, defaulting to Vol for simplicity in master)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='dimgray', name="Volume"), row=7, col=1)

        fig.update_layout(height=1200, template="plotly_dark", margin=dict(l=0,r=0,t=40,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
