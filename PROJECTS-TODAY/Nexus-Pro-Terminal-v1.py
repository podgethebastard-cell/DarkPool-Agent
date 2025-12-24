import streamlit as st
import pandas as pd
import yfinance as yf
import openai
from datetime import datetime, date, timezone
import io
import xlsxwriter
import requests
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import linregress
import urllib.parse
import streamlit.components.v1 as components
import math
import calendar
from contextlib import contextmanager

# ==========================================
# 1. SYSTEM CONFIGURATION & PRO STYLING
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="Nexus Pro Terminal", 
    page_icon="📊", 
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL CSS THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@400;500&display=swap');

    /* Global Reset & Typography */
    .stApp {
        background-color: #0e1117; /* Deep Slate */
        color: #eceff1;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    
    /* KPI Cards - Clean & Flat */
    .kpi-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .kpi-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8b949e;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-family: 'Roboto Mono', monospace;
        font-size: 20px;
        font-weight: 600;
        color: #f0f6fc;
    }
    .kpi-sub {
        font-size: 11px;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Signal Colors */
    .c-up { color: #238636 !important; } /* Institutional Green */
    .c-down { color: #da3633 !important; } /* Institutional Red */
    .c-neu { color: #8b949e !important; }
    
    /* Metric Delta Styling Override */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 12px;
        border-radius: 6px;
    }
    div[data-testid="stMetricLabel"] { font-size: 12px; color: #8b949e; }
    div[data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; font-size: 18px; }

    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 6px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 4px;
        font-weight: 600;
        border: 1px solid #30363d;
        background-color: #21262d;
        color: #c9d1d9;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        border-color: #8b949e;
        color: #ffffff;
    }
    
    /* Mobile/Report Card Styling */
    .report-container {
        background-color: #161b22;
        border-left: 4px solid #1f6feb; /* Blue Accent */
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 12px;
        font-family: 'Roboto Mono', monospace;
    }
    .report-head { font-size: 14px; font-weight: 700; color: #fff; margin-bottom: 8px; border-bottom: 1px solid #30363d; padding-bottom: 4px; }
    .report-row { font-size: 13px; color: #c9d1d9; margin-bottom: 4px; display: flex; justify-content: space-between; }
    .hl { color: #58a6ff; font-weight: 600; }

    /* Custom Plotly Fixes */
    .js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CREDENTIAL AUTO-LOADER
# ==========================================
# Automatically loads secrets without user intervention if available
if "OPENAI_API_KEY" in st.secrets:
    st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
else:
    if 'api_key' not in st.session_state: st.session_state.api_key = ""

if "TELEGRAM_TOKEN" in st.secrets:
    st.session_state.tg_token = st.secrets["TELEGRAM_TOKEN"]
else:
    if 'tg_token' not in st.session_state: st.session_state.tg_token = ""

if "TELEGRAM_CHAT_ID" in st.secrets:
    st.session_state.tg_chat = st.secrets["TELEGRAM_CHAT_ID"]
else:
    if 'tg_chat' not in st.session_state: st.session_state.tg_chat = ""

# ==========================================
# 3. CORE LOGIC: TREND & SMC (Scanner)
# ==========================================
class TrendEngine:
    @staticmethod
    def calculate_hma(series, length):
        if len(series) < length: return pd.Series(0, index=series.index)
        def wma(s, l):
            weights = np.arange(1, l + 1)
            return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        wma_half = wma(series, int(length / 2))
        wma_full = wma(series, length)
        diff = 2 * wma_half - wma_full
        return wma(diff, int(np.sqrt(length)))

    @staticmethod
    def calculate_atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        return ranges.max(axis=1).ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def calculate_adx(df, length=14):
        up = df['High'].diff()
        down = -df['Low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = TrendEngine.calculate_atr(df, length)
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean() / tr)
        sum_di = (plus_di + minus_di).replace(0, 1)
        dx = 100 * np.abs(plus_di - minus_di) / sum_di
        return dx.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def calculate_wavetrend(df):
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=10, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=10, adjust=False).mean().replace(0, 0.0001)
        ci = (ap - esa) / (0.015 * d)
        return ci.ewm(span=21, adjust=False).mean()

    @staticmethod
    def detect_structure(df):
        lookback = 10
        # High/Low Rolling
        recent_high = df['High'].shift(1).rolling(20).max()
        # Break of Structure (BOS)
        bos_bull = (df['Close'] > recent_high) & (df['Close'].shift(1) <= recent_high.shift(1))
        # Fair Value Gap (FVG)
        fvg_bull = (df['Low'] > df['High'].shift(2))
        fvg_size = (df['Low'] - df['High'].shift(2))
        return bos_bull, fvg_bull, fvg_size

    @staticmethod
    def run_analysis(df):
        if len(df) < 60: return None
        # HMA Trend
        len_main = 55; mult = 1.5
        baseline = TrendEngine.calculate_hma(df['Close'], len_main)
        atr = TrendEngine.calculate_atr(df, len_main)
        upper = baseline + (atr * mult); lower = baseline - (atr * mult)
        
        trends = []
        curr_trend = 0
        close_vals = df['Close'].values; upper_vals = upper.values; lower_vals = lower.values
        for i in range(len(df)):
            if close_vals[i] > upper_vals[i]: curr_trend = 1
            elif close_vals[i] < lower_vals[i]: curr_trend = -1
            trends.append(curr_trend)
        df['Trend_State'] = trends
        
        # Momentum
        df['ADX'] = TrendEngine.calculate_adx(df)
        df['WT'] = TrendEngine.calculate_wavetrend(df)
        vol_ma = df['Volume'].rolling(20).mean()
        
        buy_signal = ((df['Trend_State'] == 1) & (df['WT'] < 60) & (df['WT'] > df['WT'].shift(1)) & (df['ADX'] > 20) & (df['Volume'] > vol_ma))
        bos, fvg, fvg_sz = TrendEngine.detect_structure(df)
        
        last = df.iloc[-1]
        return {
            "Price": last['Close'], 
            "Trend_State": last['Trend_State'],
            "Trend_Str": "Bullish" if last['Trend_State'] == 1 else "Bearish" if last['Trend_State'] == -1 else "Neutral",
            "WT": last['WT'], "ADX": last['ADX'],
            "Signal_Buy": buy_signal.tail(3).any(), 
            "Signal_BOS": bos.tail(3).any(),
            "Signal_FVG": fvg.iloc[-1], "FVG_Size": fvg_sz.iloc[-1] if fvg.iloc[-1] else 0
        }

# ==========================================
# 4. CORE LOGIC: INSTITUTIONAL (Terminal)
# ==========================================
class InstEngine:
    @staticmethod
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def hma(series, length):
        half = int(length / 2); sqrt = int(np.sqrt(length))
        wma_half = InstEngine.wma(series, half)
        wma_full = InstEngine.wma(series, length)
        return InstEngine.wma(2 * wma_half - wma_full, sqrt)

    @staticmethod
    def supertrend(df, period=10, multiplier=3):
        atr = TrendEngine.calculate_atr(df, period)
        hl2 = (df['High'] + df['Low']) / 2
        up = hl2 + (multiplier * atr); dn = hl2 - (multiplier * atr)
        
        st = np.zeros(len(df)); trend = np.zeros(len(df))
        close = df['Close'].values; up_val = up.values; dn_val = dn.values
        
        # Iterative Logic
        st[0] = dn_val[0]; trend[0] = 1
        for i in range(1, len(df)):
            if close[i-1] > st[i-1]: # Prev Up
                st[i] = max(dn_val[i], st[i-1]) if close[i] > st[i-1] else up_val[i]
                trend[i] = 1 if close[i] > st[i-1] else -1
                if close[i] < dn_val[i] and trend[i-1] == 1: st[i] = up_val[i]; trend[i] = -1
            else: # Prev Down
                st[i] = min(up_val[i], st[i-1]) if close[i] < st[i-1] else dn_val[i]
                trend[i] = -1 if close[i] < st[i-1] else 1
                if close[i] > up_val[i] and trend[i-1] == -1: st[i] = dn_val[i]; trend[i] = 1
        return pd.Series(st, index=df.index), pd.Series(trend, index=df.index)

    @staticmethod
    def calc_composite(df):
        # Base Indicators
        df['HMA55'] = InstEngine.hma(df['Close'], 55)
        df['ATR'] = TrendEngine.calculate_atr(df, 14)
        
        # Squeeze Momentum
        basis = df['Close'].rolling(20).mean()
        dev = df['Close'].rolling(20).std() * 2
        kelt = basis + (df['ATR'] * 1.5)
        df['Squeeze'] = ((basis - dev) > (basis - (df['ATR'] * 1.5))) & ((basis + dev) < kelt)
        
        # LinReg Momentum
        delta = df['Close'] - basis
        x = np.arange(20)
        df['Mom_Val'] = delta.rolling(20).apply(lambda y: linregress(x, y)[0], raw=True)
        
        # Money Flow (Simple)
        mf_mult = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        df['MF_Vol'] = (mf_mult * df['Volume']).ewm(span=3).mean()
        
        # Vector Trend (SuperTrend)
        _, df['Vector_Dir'] = InstEngine.supertrend(df, 10, 4)
        
        # Composite Score
        trend_score = np.where(df['Close'] > df['HMA55'], 1, -1)
        mom_score = np.sign(df['Mom_Val'])
        vec_score = df['Vector_Dir']
        
        df['Comp_Score'] = trend_score + mom_score + vec_score # Range -3 to +3
        return df

    @staticmethod
    def get_sentiment(df):
        # Simple Fear/Greed Proxy
        # RSI
        delta = df['Close'].diff()
        rs = delta.clip(lower=0).ewm(alpha=1/14).mean() / (-delta.clip(upper=0).ewm(alpha=1/14).mean())
        rsi = 100 - (100 / (1 + rs))
        
        # Volatility
        vol = df['Close'].pct_change().rolling(20).std()
        
        # Logic
        score = 50
        if rsi.iloc[-1] > 70: score += 20
        elif rsi.iloc[-1] < 30: score -= 20
        if vol.iloc[-1] > vol.mean(): score -= 10 # High vol = Fear
        
        return max(0, min(100, score))

# ==========================================
# 5. CORE LOGIC: QUANT FLOW (Prev. Dark Singularity)
# ==========================================
class QuantFlowEngine:
    @staticmethod
    def chop_index(df, length=14):
        tr = TrendEngine.calculate_atr(df, 1) # True Range
        atr_sum = tr.rolling(length).sum()
        h_max = df['High'].rolling(length).max()
        l_min = df['Low'].rolling(length).min()
        
        num = np.log10(atr_sum / (h_max - l_min).replace(0, 1))
        den = np.log10(length)
        return 100 * (num / den)

    @staticmethod
    def flow_vector(df, params):
        # Efficiency Ratio * Volume Flux
        eff = (df['Close'] - df['Open']).abs() / (df['High'] - df['Low']).replace(0, 1)
        v_flux = df['Volume'] / df['Volume'].rolling(params['vol_len']).mean()
        return (np.sign(df['Close'] - df['Open']) * eff * v_flux).ewm(span=params['smooth']).mean()

    @staticmethod
    def run_flow(df, params):
        # ST & Chop
        df['SuperTrend'], df['Trend_Dir'] = InstEngine.supertrend(df, params['st_len'], params['st_mult'])
        df['Chop'] = QuantFlowEngine.chop_index(df, params['chop_len'])
        df['Vector'] = QuantFlowEngine.flow_vector(df, params)
        
        df['Filter_Active'] = df['Chop'] > params['chop_thresh']
        
        # Signals (Trend Flip + No Chop)
        df['Sig_Long'] = (df['Trend_Dir'] == 1) & (df['Trend_Dir'].shift(1) == -1) & (~df['Filter_Active'])
        df['Sig_Short'] = (df['Trend_Dir'] == -1) & (df['Trend_Dir'].shift(1) == 1) & (~df['Filter_Active'])
        return df

# ==========================================
# 6. DATA UTILS
# ==========================================
@st.cache_data(ttl=300)
def fetch_market_data(ticker, interval="1d", period="1y"):
    try:
        # Handling Yahoo Finance Intervals
        y_int = "1h" if interval == "4h" else interval
        y_per = "3mo" if interval in ["1h", "4h"] else period
        
        df = yf.download(ticker, period=y_per, interval=y_int, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Manual Resample for 4H
        if interval == "4h":
            agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
            df = df.resample('4h').agg(agg).dropna()
            
        return df
    except: return None

def get_snapshot(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "Ticker": ticker, 
            "Name": info.get('shortName', ticker), 
            "Cap": info.get('marketCap', 0)
        }
    except: return None

# ==========================================
# 7. VISUALIZATION ENGINE (Plotly)
# ==========================================
def plot_pro_chart(df, ticker, levels=None, signals=None, title="Price Action"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75, 0.25])
    
    # Colors: Pro Slate
    c_up = '#238636'; c_dn = '#da3633'; c_line = '#a371f7'
    
    # Price
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color=c_up, decreasing_line_color=c_dn,
        increasing_fillcolor=c_up, decreasing_fillcolor=c_dn,
        name='Price'
    ), row=1, col=1)
    
    # Add Moving Averages or SuperTrend if present
    if 'HMA55' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA55'], line=dict(color='#e0e0e0', width=1), name='Trend Baseline'), row=1, col=1)
    if 'SuperTrend' in df.columns:
        # Split line for color
        st = df['SuperTrend']
        fig.add_trace(go.Scatter(x=df.index, y=st, line=dict(color='#58a6ff', width=1.5), name='Trailing Stop'), row=1, col=1)

    # Momentum / Volume
    if 'Vector' in df.columns:
        cols = [c_up if v > 0 else c_dn for v in df['Vector']]
        fig.add_trace(go.Bar(x=df.index, y=df['Vector'], marker_color=cols, name='Flow'), row=2, col=1)
    elif 'Volume' in df.columns:
        cols = [c_up if c > o else c_dn for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=cols, name='Volume'), row=2, col=1)

    # Signals
    if signals:
        longs = df[df['Sig_Long']] if 'Sig_Long' in df.columns else pd.DataFrame()
        if not longs.empty:
            fig.add_trace(go.Scatter(x=longs.index, y=longs['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#39d353'), name='Buy'), row=1, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=40, b=10),
        height=600,
        xaxis_rangeslider_visible=False,
        title=dict(text=f"{ticker} | {title}", font=dict(size=14, color="#8b949e"))
    )
    fig.update_xaxes(showgrid=False, linecolor='#30363d')
    fig.update_yaxes(showgrid=True, gridcolor='#21262d', linecolor='#30363d')
    
    return fig

# ==========================================
# 8. AI & ALERTS
# ==========================================
def generate_briefing(ticker, price, trend, mom, vol, api_key):
    if not api_key: return "⚠️ API Key Required"
    
    prompt = f"""
    Asset: {ticker} | Price: {price}
    Trend: {trend} | Momentum: {mom} | Volatility: {vol}
    
    As a senior quant analyst, provide a 3-bullet point executive summary. 
    1. Market Structure (Bull/Bear/Range)
    2. Key Volume/Liquidity Note
    3. Immediate Watch Level
    Keep it strictly professional and concise. No financial advice.
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}])
        return res.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

def send_alert(msg, file_obj=None):
    if st.session_state.tg_token and st.session_state.tg_chat:
        try:
            url = f"https://api.telegram.org/bot{st.session_state.tg_token}/sendMessage"
            requests.post(url, data={"chat_id": st.session_state.tg_chat, "text": msg})
            if file_obj:
                f_url = f"https://api.telegram.org/bot{st.session_state.tg_token}/sendDocument"
                requests.post(f_url, data={"chat_id": st.session_state.tg_chat}, files={"document": ("Report.xlsx", file_obj, "application/vnd.ms-excel")})
            return True
        except: return False
    return False

# ==========================================
# 9. MAIN APP CONTROLLER
# ==========================================
st.sidebar.title("NEXUS PRO")
mode = st.sidebar.radio("Module", ["Market Scanner", "Institutional Terminal", "Mobile Desk", "Quant Flow"])

# Credentials
with st.sidebar.expander("Settings & API"):
    k1 = st.text_input("OpenAI Key", value=st.session_state.api_key, type="password")
    if k1: st.session_state.api_key = k1
    k2 = st.text_input("TG Token", value=st.session_state.tg_token, type="password")
    if k2: st.session_state.tg_token = k2
    k3 = st.text_input("TG Chat ID", value=st.session_state.tg_chat)
    if k3: st.session_state.tg_chat = k3

# --------------------------
# MODULE 1: SCANNER
# --------------------------
if mode == "Market Scanner":
    st.markdown("## 🔭 Market Scanner")
    st.caption("Trend Engine // Smart Money Concepts // Volatility")
    
    scan_list = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "XRP-USD", "AVAX-USD", "LINK-USD"]
    
    if st.button("Initialize Scan"):
        results = []
        prog = st.progress(0)
        
        for i, t in enumerate(scan_list):
            prog.progress((i+1)/len(scan_list))
            df = fetch_market_data(t)
            if df is not None:
                res = TrendEngine.run_analysis(df)
                if res:
                    score = 0
                    if res['Trend_State'] == 1: score += 1
                    if res['Signal_Buy']: score += 2
                    if res['Signal_BOS']: score += 1
                    
                    results.append({
                        "Asset": t, 
                        "Price": res['Price'], 
                        "Trend": res['Trend_Str'], 
                        "Score": score,
                        "Signal": "BUY" if res['Signal_Buy'] else "-",
                        "BOS": "YES" if res['Signal_BOS'] else "-",
                        "FVG": "YES" if res['Signal_FVG'] else "-"
                    })
        prog.empty()
        
        if results:
            scan_df = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.dataframe(scan_df, use_container_width=True, height=400)
            
            # AI & Export
            c1, c2 = st.columns([2, 1])
            with c1:
                if st.button("Generate Analyst Briefing"):
                    top_pick = scan_df.iloc[0]
                    brief = generate_briefing(top_pick['Asset'], top_pick['Price'], top_pick['Trend'], "High", "High", st.session_state.api_key)
                    st.info(brief)
            
            with c2:
                # Excel Logic
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                    scan_df.to_excel(writer, sheet_name="Scan", index=False)
                st.download_button("Export Data", buf.getvalue(), "scan_results.xlsx")
                
                if st.button("Broadcast Top Pick"):
                    top = scan_df.iloc[0]
                    msg = f"📊 NEXUS ALERT: {top['Asset']}\nTrend: {top['Trend']}\nSignal: {top['Signal']}"
                    if send_alert(msg, io.BytesIO(buf.getvalue())):
                        st.success("Broadcast Sent")
                    else:
                        st.error("Config Error")

# --------------------------
# MODULE 2: TERMINAL
# --------------------------
elif mode == "Institutional Terminal":
    st.markdown("## 📊 Institutional Terminal")
    
    c1, c2 = st.columns([3, 1])
    ticker = c1.text_input("Asset Ticker", "BTC-USD").upper()
    tf = c2.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=3)
    
    if st.button("Load Data"):
        df = fetch_market_data(ticker, tf)
        if df is not None:
            df = InstEngine.calc_composite(df)
            last = df.iloc[-1]
            
            # KPI Row
            k1, k2, k3, k4 = st.columns(4)
            
            def kpi(col, label, val, sub):
                col.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)
            
            trend_c = "Bullish" if last['HMA55'] < last['Close'] else "Bearish"
            mom_c = "Positive" if last['Mom_Val'] > 0 else "Negative"
            
            kpi(k1, "Composite Score", f"{last['Comp_Score']:.0f} / 3", "Trend + Mom + Vector")
            kpi(k2, "Trend Baseline", trend_c, f"HMA: {last['HMA55']:.2f}")
            kpi(k3, "Momentum", mom_c, f"Squeeze: {'Active' if last['Squeeze'] else 'Inactive'}")
            kpi(k4, "Money Flow", f"{last['MF_Vol']:.2f}", "Vol Weighted")
            
            # Chart
            fig = plot_pro_chart(df, ticker, title="Composite Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # AI
            with st.expander("Analyst Commentary"):
                st.write(generate_briefing(ticker, last['Close'], trend_c, mom_c, "Avg", st.session_state.api_key))

# --------------------------
# MODULE 3: MOBILE
# --------------------------
elif mode == "Mobile Desk":
    st.markdown("## 📱 Mobile Desk")
    
    # Live Clock
    components.html("""
    <div style="font-family:'Roboto Mono'; color:#8b949e; text-align:center; font-size:14px;">
    MARKET TIME (UTC) <span id="clock" style="color:#f0f6fc; font-weight:bold;"></span>
    </div>
    <script>setInterval(()=>document.getElementById('clock').innerText=new Date().toLocaleTimeString('en-GB',{timeZone:'UTC'}),1000)</script>
    """, height=30)
    
    m_sym = st.text_input("Ticker", "BTC-USD").upper()
    
    if st.button("Refresh"):
        df = fetch_market_data(m_sym, "1h") # Fixed mobile TF
        if df is not None:
            # Need Mobile Specific Math (Re-using functions from previous mobile logic adapted here)
            # Simplified for consistency with Pro Theme:
            # Using InstEngine supertrend + HMA
            st_ser, t_dir = InstEngine.supertrend(df, 10, 3)
            df['SuperTrend'] = st_ser
            df['Trend'] = t_dir
            
            last = df.iloc[-1]
            trend_txt = "BULL" if last['Trend'] == 1 else "BEAR"
            
            st.markdown(f"""
            <div class="report-container">
                <div class="report-head">{m_sym} | {last['Close']:.2f}</div>
                <div class="report-row"><span>Trend</span> <span class="hl">{trend_txt}</span></div>
                <div class="report-row"><span>Stop Loss</span> <span class="hl">{last['SuperTrend']:.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Mini Chart
            fig = go.Figure(go.Scatter(x=df.index[-50:], y=df['Close'].tail(50), line=dict(color='#58a6ff', width=2)))
            fig.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Send Mobile Alert"):
                msg = f"📱 {m_sym} Update | Price: {last['Close']:.2f} | Trend: {trend_txt}"
                send_alert(msg)
                st.success("Sent")

# --------------------------
# MODULE 4: QUANT FLOW
# --------------------------
elif mode == "Quant Flow":
    st.markdown("## 💠 Quant Flow")
    st.caption("Algorithm Control // Vector Flux // Choppiness")
    
    q_sym = st.text_input("Ticker", "BTC-USD").upper()
    
    with st.expander("Algorithm Parameters"):
        st_len = st.slider("ATR Len", 5, 50, 10)
        st_mult = st.slider("Trend Factor", 1.0, 10.0, 4.0)
        chop_thr = st.slider("Chop Threshold", 40, 70, 60)
    
    if st.button("Execute Flow Analysis"):
        df = fetch_market_data(q_sym)
        if df is not None:
            params = {'st_len': st_len, 'st_mult': st_mult, 'chop_len': 14, 'chop_thresh': chop_thr, 'vol_len': 50, 'smooth': 5}
            df = QuantFlowEngine.run_flow(df, params)
            last = df.iloc[-1]
            
            # Status Indicators
            status = "TRENDING" if not last['Filter_Active'] else "CHOPPY / RANGING"
            s_col = "c-up" if not last['Filter_Active'] else "c-neu"
            
            st.markdown(f"""
            <div style="display:flex; gap:20px; margin-bottom:20px;">
                <div class="kpi-card" style="flex:1">
                    <div class="kpi-label">Market State</div>
                    <div class="kpi-value {s_col}">{status}</div>
                </div>
                <div class="kpi-card" style="flex:1">
                    <div class="kpi-label">Vector Strength</div>
                    <div class="kpi-value">{last['Vector']:.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Advanced Chart
            fig = plot_pro_chart(df, q_sym, signals=True, title="Flow Vector")
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Raw Data Feed"):
                st.dataframe(df.tail(10)[['Close', 'SuperTrend', 'Chop', 'Vector', 'Sig_Long', 'Sig_Short']])
