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
    page_icon="👁️", 
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL CSS THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@400;500&display=swap');

    /* Global Reset */
    .stApp {
        background-color: #0e1117; 
        color: #eceff1;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; letter-spacing: -0.5px; color: #ffffff; }
    
    /* BRAIN/CORTEX CARD */
    .brain-card {
        background: linear-gradient(135deg, #1c1c1c 0%, #0e0e0e 100%);
        border: 1px solid #333;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        border-left: 4px solid #7c4dff; /* Deep Purple Brain Accent */
    }
    .brain-title { font-size: 14px; color: #a688fa; letter-spacing: 1px; font-weight: 700; margin-bottom: 10px; text-transform: uppercase; }
    .brain-val { font-family: 'Roboto Mono', monospace; font-size: 28px; color: #fff; font-weight: 700; }
    .brain-sub { font-size: 12px; color: #666; margin-top: 5px; }

    /* KPI Cards */
    .kpi-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 10px;
        border-left: 3px solid #8b949e;
    }
    .kpi-card.bull { border-left-color: #238636; }
    .kpi-card.bear { border-left-color: #da3633; }
    .kpi-label { font-size: 11px; text-transform: uppercase; color: #8b949e; font-weight: 600; margin-bottom: 4px; }
    .kpi-value { font-family: 'Roboto Mono', monospace; font-size: 22px; font-weight: 600; color: #f0f6fc; }
    .kpi-sub { font-size: 11px; color: #8b949e; margin-top: 4px; }

    /* Utilities */
    .c-up { color: #238636 !important; }
    .c-down { color: #da3633 !important; } 
    .c-neu { color: #8b949e !important; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; }
    .stButton > button { border-radius: 4px; border: 1px solid #30363d; background-color: #21262d; color: #c9d1d9; }
    .stButton > button:hover { border-color: #8b949e; color: #ffffff; }
    
    /* Mobile Card */
    .report-container { background-color: #161b22; border-left: 4px solid #1f6feb; padding: 15px; margin-bottom: 12px; font-family: 'Roboto Mono', monospace; }
    .report-head { font-size: 14px; font-weight: 700; color: #fff; margin-bottom: 8px; border-bottom: 1px solid #30363d; }
    .report-row { font-size: 13px; color: #c9d1d9; margin-bottom: 4px; display: flex; justify-content: space-between; }
    .hl { color: #58a6ff; font-weight: 600; }
    .js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CREDENTIAL AUTO-LOADER
# ==========================================
if "OPENAI_API_KEY" in st.secrets: st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
else: st.session_state.api_key = st.session_state.get('api_key', "")

if "TELEGRAM_TOKEN" in st.secrets: st.session_state.tg_token = st.secrets["TELEGRAM_TOKEN"]
else: st.session_state.tg_token = st.session_state.get('tg_token', "")

if "TELEGRAM_CHAT_ID" in st.secrets: st.session_state.tg_chat = st.secrets["TELEGRAM_CHAT_ID"]
else: st.session_state.tg_chat = st.session_state.get('tg_chat', "")

# ==========================================
# 3. CORE ENGINES (ALL INCLUDED)
# ==========================================

# --- TREND ENGINE ---
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
        up, down = df['High'].diff(), -df['Low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = TrendEngine.calculate_atr(df, length)
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean() / tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        return dx.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def calculate_wavetrend(df):
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=10, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=10, adjust=False).mean().replace(0, 0.0001)
        ci = (ap - esa) / (0.015 * d)
        return ci.ewm(span=21, adjust=False).mean()

    @staticmethod
    def run_analysis(df):
        if len(df) < 60: return None
        base = TrendEngine.calculate_hma(df['Close'], 55)
        atr = TrendEngine.calculate_atr(df, 55)
        upper, lower = base + (atr * 1.5), base - (atr * 1.5)
        
        trends = []
        c_vals = df['Close'].values; u_vals = upper.values; l_vals = lower.values
        for i in range(len(df)):
            if c_vals[i] > u_vals[i]: trends.append(1)
            elif c_vals[i] < l_vals[i]: trends.append(-1)
            else: trends.append(0)
        df['Trend_State'] = trends
        
        df['ADX'] = TrendEngine.calculate_adx(df)
        df['WT'] = TrendEngine.calculate_wavetrend(df)
        vol_ma = df['Volume'].rolling(20).mean()
        buy = ((df['Trend_State'] == 1) & (df['WT'] < 60) & (df['WT'] > df['WT'].shift(1)) & (df['ADX'] > 20) & (df['Volume'] > vol_ma))
        
        rec_high = df['High'].shift(1).rolling(20).max()
        bos = (df['Close'] > rec_high) & (df['Close'].shift(1) <= rec_high.shift(1))
        fvg = (df['Low'] > df['High'].shift(2))
        
        last = df.iloc[-1]
        return {
            "Price": last['Close'], "Trend_State": last['Trend_State'],
            "Trend_Str": "Bullish" if last['Trend_State'] == 1 else "Bearish" if last['Trend_State'] == -1 else "Neutral",
            "WT": last['WT'], "ADX": last['ADX'], "Signal_Buy": buy.tail(3).any(), 
            "Signal_BOS": bos.tail(3).any(), "Signal_FVG": fvg.iloc[-1]
        }

# --- INSTITUTIONAL ENGINE ---
class InstEngine:
    @staticmethod
    def supertrend(df, period=10, multiplier=3):
        atr = TrendEngine.calculate_atr(df, period)
        hl2 = (df['High'] + df['Low']) / 2
        up = hl2 + (multiplier * atr); dn = hl2 - (multiplier * atr)
        st = np.zeros(len(df)); trend = np.zeros(len(df))
        close = df['Close'].values; up_val = up.values; dn_val = dn.values
        st[0] = dn_val[0]; trend[0] = 1
        for i in range(1, len(df)):
            if close[i-1] > st[i-1]:
                st[i] = max(dn_val[i], st[i-1]) if close[i] > st[i-1] else up_val[i]
                trend[i] = 1 if close[i] > st[i-1] else -1
                if close[i] < dn_val[i] and trend[i-1] == 1: st[i] = up_val[i]; trend[i] = -1
            else:
                st[i] = min(up_val[i], st[i-1]) if close[i] < st[i-1] else dn_val[i]
                trend[i] = -1 if close[i] < st[i-1] else 1
                if close[i] > up_val[i] and trend[i-1] == -1: st[i] = dn_val[i]; trend[i] = 1
        return pd.Series(st, index=df.index), pd.Series(trend, index=df.index)

    @staticmethod
    def calc_composite(df):
        df['HMA55'] = TrendEngine.calculate_hma(df['Close'], 55)
        df['ATR'] = TrendEngine.calculate_atr(df, 14)
        
        basis = df['Close'].rolling(20).mean()
        dev = df['Close'].rolling(20).std() * 2
        kelt = basis + (df['ATR'] * 1.5)
        df['Squeeze'] = ((basis - dev) > (basis - (df['ATR'] * 1.5))) & ((basis + dev) < kelt)
        
        delta = df['Close'] - basis
        x = np.arange(20)
        df['Mom_Val'] = delta.rolling(20).apply(lambda y: linregress(x, y)[0], raw=True)
        
        mf_mult = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        df['MF_Vol'] = (mf_mult * df['Volume']).ewm(span=3).mean()
        
        _, df['Vector_Dir'] = InstEngine.supertrend(df, 10, 4)
        df['Comp_Score'] = np.where(df['Close'] > df['HMA55'], 1, -1) + np.sign(df['Mom_Val']) + df['Vector_Dir']
        return df

    @staticmethod
    def get_macro_data():
        tickers = {"S&P 500": "SPY", "Bitcoin": "BTC-USD", "Gold": "GC=F", "10Y Yield": "^TNX", "VIX": "^VIX"}
        try:
            data = yf.download(list(tickers.values()), period="5d", progress=False)['Close']
            res = {}
            for k, v in tickers.items():
                if v in data.columns:
                    curr = data[v].iloc[-1]; prev = data[v].iloc[-2]
                    res[k] = {"Price": curr, "Chg": ((curr-prev)/prev)*100}
            return res
        except: return {}

# --- QUANT FLOW ENGINE ---
class QuantFlowEngine:
    @staticmethod
    def chop_index(df, length=14):
        tr = TrendEngine.calculate_atr(df, 1)
        atr_sum = tr.rolling(length).sum()
        h_max = df['High'].rolling(length).max()
        l_min = df['Low'].rolling(length).min()
        return 100 * np.log10(atr_sum / (h_max - l_min).replace(0, 1)) / np.log10(length)

    @staticmethod
    def flow_vector(df, params):
        eff = (df['Close'] - df['Open']).abs() / (df['High'] - df['Low']).replace(0, 1)
        v_flux = df['Volume'] / df['Volume'].rolling(params['vol_len']).mean()
        return (np.sign(df['Close'] - df['Open']) * eff * v_flux).ewm(span=params['smooth']).mean()

    @staticmethod
    def run_flow(df, params):
        df['SuperTrend'], df['Trend_Dir'] = InstEngine.supertrend(df, params['st_len'], params['st_mult'])
        df['Chop'] = QuantFlowEngine.chop_index(df, params['chop_len'])
        df['Vector'] = QuantFlowEngine.flow_vector(df, params)
        df['Filter_Active'] = df['Chop'] > params['chop_thresh']
        df['Sig_Long'] = (df['Trend_Dir'] == 1) & (df['Trend_Dir'].shift(1) == -1) & (~df['Filter_Active'])
        df['Sig_Short'] = (df['Trend_Dir'] == -1) & (df['Trend_Dir'].shift(1) == 1) & (~df['Filter_Active'])
        return df

# --- MOBILE ENGINE ---
class MobileEngine:
    @staticmethod
    def calculate_fibs(df):
        recent = df.iloc[-50:]
        h, l = recent['High'].max(), recent['Low'].min()
        diff = h - l
        return {'0.382': h-(diff*0.382), '0.618': h-(diff*0.618)}

    @staticmethod
    def generate_report(row, symbol, fibs, stop):
        trend_icon = "🐂" if row['Trend']==1 else "🐻"
        sqz_txt = "⚠️ ACTIVE" if row['Chop'] > 60 else "⚪ OPEN"
        html = f"""
        <div class="mobile-card">
            <div class="mobile-header">{symbol} | {row['Close']:.2f}</div>
            <div class="mobile-row"><span>DIRECTION</span> <span class="mobile-hl">{trend_icon} {row['Trend']}</span></div>
            <div class="mobile-row"><span>FLUX</span> <span>{row['Vector']:.2f}</span></div>
            <div class="mobile-row"><span>GATE</span> <span>{sqz_txt}</span></div>
        </div>
        <div class="mobile-card">
            <div class="mobile-header">EXECUTION</div>
            <div class="mobile-row"><span>ENTRY</span> <span class="mobile-hl">{row['Close']:.2f}</span></div>
            <div class="mobile-row"><span>STOP</span> <span style="color:#FF1744">{stop:.2f}</span></div>
            <div class="mobile-row"><span>FIB 618</span> <span>{fibs['0.618']:.2f}</span></div>
        </div>
        """
        return html

# ==========================================
# 4. BRAIN ENGINE (NEW: INTEGRATION LAYER)
# ==========================================
class BrainEngine:
    """
    The Cortex: Synthesizes data from Apex, Institutional, and Quant engines to form a unified opinion.
    """
    @staticmethod
    def analyze(df):
        # 1. Run Apex Logic
        apex_res = TrendEngine.run_analysis(df)
        
        # 2. Run Institutional Logic
        inst_df = InstEngine.calc_composite(df.copy())
        inst_last = inst_df.iloc[-1]
        
        # 3. Run Quant Logic
        q_params = {'st_len': 10, 'st_mult': 3, 'chop_len': 14, 'chop_thresh': 60, 'vol_len': 50, 'smooth': 5}
        quant_df = QuantFlowEngine.run_flow(df.copy(), q_params)
        quant_last = quant_df.iloc[-1]
        
        # 4. SYNTHESIS
        score = 0
        reasons = []
        
        # Trend Consensus
        if apex_res['Trend_State'] == 1: score += 20
        elif apex_res['Trend_State'] == -1: score -= 20
        
        if inst_last['Comp_Score'] > 1: score += 30; reasons.append("Inst. Flow (+)")
        elif inst_last['Comp_Score'] < -1: score -= 30; reasons.append("Inst. Flow (-)")
        
        if quant_last['Vector'] > 0.1: score += 10
        elif quant_last['Vector'] < -0.1: score -= 10
        
        # Filters
        if quant_last['Filter_Active']: score *= 0.5; reasons.append("Choppy (Dampened)")
        if apex_res['Signal_BOS']: score += 15; reasons.append("Struct Break")
        
        # Verdict
        if score > 40: verdict = "PRIME BULL"; color = "#00E676"
        elif score < -40: verdict = "PRIME BEAR"; color = "#FF1744"
        elif score > 10: verdict = "LEAN BULL"; color = "#69F0AE"
        elif score < -10: verdict = "LEAN BEAR"; color = "#FF5252"
        else: verdict = "NEUTRAL / CHOP"; color = "#888"
        
        return {
            "Score": score,
            "Verdict": verdict,
            "Color": color,
            "Reasons": ", ".join(reasons) if reasons else "No dominant drivers",
            "Inst_Score": inst_last['Comp_Score'],
            "Quant_Vec": quant_last['Vector'],
            "Apex_Trend": apex_res['Trend_Str']
        }

# ==========================================
# 5. DATA UTILS & VISUALS
# ==========================================
@st.cache_data(ttl=300)
def fetch_market_data(ticker, interval="1d", period="1y"):
    try:
        y_int = "1h" if interval == "4h" else interval
        y_per = "3mo" if interval in ["1h", "4h"] else period
        df = yf.download(ticker, period=y_per, interval=y_int, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if interval == "4h":
            agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
            df = df.resample('4h').agg(agg).dropna()
        return df
    except: return None

class VisualEngine:
    COLORS = {'bg': '#0e1117', 'grid': '#1f2937', 'bull': '#238636', 'bear': '#da3633', 'text': '#e5e7eb', 'hma': '#fbbf24', 'st_bull': '#238636', 'st_bear': '#da3633'}

    @staticmethod
    def create_master_chart(df, ticker, title="Market Analysis", show_signals=True):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3], subplot_titles=(f"{ticker} | PRICE", "FLOW VECTOR"))
        
        # Price
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], increasing_line_color=VisualEngine.COLORS['bull'], decreasing_line_color=VisualEngine.COLORS['bear'], name='Price'), row=1, col=1)
        
        # Indicators
        if 'HMA55' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['HMA55'], line=dict(color=VisualEngine.COLORS['hma'], width=2), name='HMA 55'), row=1, col=1)
        if 'SuperTrend' in df.columns:
            st_val = df['SuperTrend']; trend_dir = df['Trend_Dir'] if 'Trend_Dir' in df.columns else df['Vector_Dir']
            st_bull = st_val.where(trend_dir == 1); st_bear = st_val.where(trend_dir == -1)
            fig.add_trace(go.Scatter(x=df.index, y=st_bull, line=dict(color=VisualEngine.COLORS['st_bull'], width=2), name='Stop (Bull)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=st_bear, line=dict(color=VisualEngine.COLORS['st_bear'], width=2), name='Stop (Bear)'), row=1, col=1)

        # Flow
        if 'Vector' in df.columns:
            cols = [VisualEngine.COLORS['bull'] if v > 0 else VisualEngine.COLORS['bear'] for v in df['Vector']]
            fig.add_trace(go.Bar(x=df.index, y=df['Vector'], marker_color=cols, name='Vector'), row=2, col=1)
        elif 'Comp_Score' in df.columns:
            cols = [VisualEngine.COLORS['bull'] if v > 0 else VisualEngine.COLORS['bear'] for v in df['Comp_Score']]
            fig.add_trace(go.Bar(x=df.index, y=df['Comp_Score'], marker_color=cols, name='Composite'), row=2, col=1)

        fig.update_layout(template="plotly_dark", paper_bgcolor=VisualEngine.COLORS['bg'], plot_bgcolor=VisualEngine.COLORS['bg'], margin=dict(l=10, r=10, t=30, b=10), height=600, xaxis_rangeslider_visible=False)
        return fig

# ==========================================
# 6. MAIN CONTROLLER
# ==========================================
st.sidebar.title("NEXUS PRO")
# Default: DarkPool Terminal
mode = st.sidebar.radio("Command Module", ["👁️ DarkPool Terminal", "🛡️ Market Scanner", "💀 Quant Flow", "📱 Mobile Desk"])

with st.sidebar.expander("System Keys"):
    k1 = st.text_input("OpenAI Key", value=st.session_state.api_key, type="password")
    if k1: st.session_state.api_key = k1
    k2 = st.text_input("TG Token", value=st.session_state.tg_token, type="password")
    if k2: st.session_state.tg_token = k2
    k3 = st.text_input("TG Chat ID", value=st.session_state.tg_chat)
    if k3: st.session_state.tg_chat = k3

# ------------------------------------------------
# MODULE 1: DARKPOOL TERMINAL (DEFAULT + BRAIN)
# ------------------------------------------------
if mode == "👁️ DarkPool Terminal":
    st.markdown("### 👁️ DARKPOOL TERMINAL")
    st.caption("Nexus Cortex: Multi-Engine Synthesis")
    
    # Macro Bar
    macro = InstEngine.get_macro_data()
    if macro:
        cols = st.columns(len(macro))
        for i, (k,v) in enumerate(macro.items()):
            cols[i].metric(k, f"{v['Price']:.2f}", f"{v['Chg']:.2f}%")
            
    c1, c2 = st.columns([3, 1])
    ticker = c1.text_input("Asset Ticker", "BTC-USD").upper()
    tf = c2.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=3)
    
    if st.button("INITIALIZE CORTEX"):
        with st.spinner("Synthesizing Logic Engines..."):
            df = fetch_market_data(ticker, tf)
            if df is not None:
                # 1. BRAIN ANALYSIS
                brain = BrainEngine.analyze(df)
                
                # 2. RENDER BRAIN CARD
                st.markdown(f"""
                <div class="brain-card" style="border-left-color: {brain['Color']}">
                    <div class="brain-title">🧠 NEXUS CORTEX CONSENSUS</div>
                    <div class="brain-val" style="color: {brain['Color']}">{brain['Verdict']}</div>
                    <div class="brain-sub">SCORE: {brain['Score']:.0f} | DRIVERS: {brain['Reasons']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 3. DETAILED METRICS
                k1, k2, k3, k4 = st.columns(4)
                
                def kpi(col, title, value, sub, style=""):
                    col.markdown(f"""
                    <div class="kpi-card {style}">
                        <div class="kpi-label">{title}</div>
                        <div class="kpi-value">{value}</div>
                        <div class="kpi-sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)
                
                trend_c = "bull" if brain['Apex_Trend'] == "Bullish" else "bear"
                
                kpi(k1, "INSTITUTIONAL FLOW", f"{brain['Inst_Score']:.0f} / 3", "Composite Model")
                kpi(k2, "APEX TREND", brain['Apex_Trend'], "HMA 55 Baseline", trend_c)
                kpi(k3, "QUANT VECTOR", f"{brain['Quant_Vec']:.3f}", "Volume Flux Efficiency")
                kpi(k4, "SIGNAL CONFIDENCE", "HIGH" if abs(brain['Score']) > 40 else "LOW", "Multi-Factor")
                
                # 4. CHARTING (Enhanced)
                # We need to enrich DF for plotting based on selected view
                df = InstEngine.calc_composite(df)
                st.markdown("---")
                chart = VisualEngine.create_master_chart(df, ticker, title="INSTITUTIONAL COMPOSITE")
                st.plotly_chart(chart, use_container_width=True)
                
                # 5. AI BRIEF
                with st.expander("ACCESS ANALYST BRIEFING"):
                    prompt = f"Analyze {ticker}. Cortex Verdict: {brain['Verdict']} (Score {brain['Score']}). Apex Trend: {brain['Apex_Trend']}. Inst Score: {brain['Inst_Score']}. Provide strategic outlook."
                    if st.session_state.api_key:
                        try:
                            client = openai.OpenAI(api_key=st.session_state.api_key)
                            res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}])
                            st.write(res.choices[0].message.content)
                        except: st.error("AI Connection Failed")
                    else: st.warning("Enter API Key")

# ------------------------------------------------
# MODULE 2: MARKET SCANNER
# ------------------------------------------------
elif mode == "🛡️ Market Scanner":
    st.markdown("### 🛡️ MARKET SCANNER")
    st.caption("Apex Trend Engine // SMC Structure")
    
    scan_list = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "XRP-USD", "AVAX-USD", "LINK-USD"]
    
    if st.button("RUN SCAN"):
        results = []
        prog = st.progress(0)
        for i, t in enumerate(scan_list):
            prog.progress((i+1)/len(scan_list))
            df = fetch_market_data(t)
            if df is not None:
                res = TrendEngine.run_analysis(df)
                if res:
                    score = 0
                    if res['Trend_Str'] == "Bullish": score += 1
                    if res['Signal_Buy']: score += 2
                    if res['Signal_BOS']: score += 1
                    results.append({
                        "Asset": t, "Price": res['Price'], "Trend": res['Trend_Str'], 
                        "Score": score, "Signal": "BUY" if res['Signal_Buy'] else "-", 
                        "BOS": "YES" if res['Signal_BOS'] else "-"
                    })
        prog.empty()
        
        if results:
            scan_df = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.dataframe(scan_df, use_container_width=True)
            
            c1, c2 = st.columns(2)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf) as writer: scan_df.to_excel(writer, index=False)
            c1.download_button("EXPORT", buf.getvalue(), "scan.xlsx")

# ------------------------------------------------
# MODULE 3: QUANT FLOW
# ------------------------------------------------
elif mode == "💀 Quant Flow":
    st.markdown("### 💀 QUANT FLOW")
    st.caption("Singularity Engine: Vector Flux & Choppiness")
    
    q_sym = st.text_input("Ticker", "BTC-USD").upper()
    with st.expander("Parameters"):
        st_len = st.slider("ATR", 5, 50, 10)
        chop_thr = st.slider("Chop Threshold", 40, 70, 60)
        
    if st.button("EXECUTE FLOW"):
        df = fetch_market_data(q_sym)
        if df is not None:
            params = {'st_len': st_len, 'st_mult': 3, 'chop_len': 14, 'chop_thresh': chop_thr, 'vol_len': 50, 'smooth': 5}
            df = QuantFlowEngine.run_flow(df, params)
            last = df.iloc[-1]
            
            status = "TRENDING" if not last['Filter_Active'] else "CHOPPY"
            s_col = "bull" if not last['Filter_Active'] else "neu"
            
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="kpi-card {s_col}"><div class="kpi-label">STATE</div><div class="kpi-value">{status}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="kpi-card"><div class="kpi-label">VECTOR</div><div class="kpi-value">{last["Vector"]:.3f}</div></div>', unsafe_allow_html=True)
            
            chart = VisualEngine.create_master_chart(df, q_sym, title="FLOW ANALYSIS", show_signals=True)
            st.plotly_chart(chart, use_container_width=True)

# ------------------------------------------------
# MODULE 4: MOBILE DESK
# ------------------------------------------------
elif mode == "📱 Mobile Desk":
    st.markdown("### 📱 MOBILE DESK")
    components.html("""<div style="font-family:'Roboto Mono';color:#888;text-align:center;">UTC <span id="c" style="color:#fff;"></span></div><script>setInterval(()=>document.getElementById('c').innerText=new Date().toLocaleTimeString('en-GB',{timeZone:'UTC'}),1000)</script>""", height=30)
    
    m_sym = st.text_input("Ticker", "BTC-USD").upper()
    if st.button("REFRESH"):
        df = fetch_market_data(m_sym, "1h")
        if df is not None:
            params = {'st_len': 10, 'st_mult': 3, 'chop_len': 14, 'chop_thresh': 60, 'vol_len': 20, 'smooth': 3}
            df = QuantFlowEngine.run_flow(df, params)
            last = df.iloc[-1]
            fibs = MobileEngine.calculate_fibs(df)
            stop = min(last['SuperTrend'], fibs['0.618']) if last['Trend_Dir']==1 else max(last['SuperTrend'], fibs['0.618'])
            
            st.markdown(MobileEngine.generate_report(last, m_sym, fibs, stop), unsafe_allow_html=True)
            
            fig = go.Figure(go.Scatter(x=df.index[-24:], y=df['Close'].tail(24), line=dict(color='#00E676', width=2)))
            fig.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)
