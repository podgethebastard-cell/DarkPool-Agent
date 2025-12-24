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
# 1. SYSTEM CORE & UI CONFIGURATION
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="Titan Omni-System", 
    page_icon="💠", 
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL UI THEME (REVISED) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

    /* BASE THEME */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
        font-family: 'Rajdhani', sans-serif;
    }

    /* TYPOGRAPHY */
    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; text-transform: uppercase; letter-spacing: 1px; }
    .mono { font-family: 'Roboto Mono', monospace; }

    /* CARDS & CONTAINERS */
    .metric-card {
        background: #111;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 3px solid #444;
    }
    .metric-value { font-family: 'Roboto Mono', monospace; font-size: 1.4rem; font-weight: 700; color: #fff; }
    .metric-label { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }

    /* SIGNAL COLORS */
    .bull { color: #00E676 !important; }
    .bear { color: #FF1744 !important; }
    .neu { color: #888 !important; }

    /* MOBILE REPORT CARD (Specific Request) */
    .mobile-card {
        background-color: #1a1a1a;
        border-left: 4px solid #00E676;
        padding: 12px;
        margin-bottom: 8px;
        font-family: 'Roboto Mono', monospace;
        font-size: 13px;
    }
    .mobile-header { font-weight: bold; color: #fff; border-bottom: 1px solid #333; padding-bottom: 4px; margin-bottom: 6px; }
    .mobile-row { display: flex; justify-content: space-between; margin-bottom: 4px; color: #ccc; }
    .mobile-hl { color: #00E676; font-weight: bold; }

    /* CUSTOM COMPONENTS */
    div[data-testid="stMetric"] { background-color: #0a0a0a; border: 1px solid #222; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #111; border: 1px solid #333; color: #888; }
    .stTabs [aria-selected="true"] { background-color: #222; color: #fff; border-bottom: 2px solid #00E676; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CREDENTIAL MANAGER
# ==========================================
if "OPENAI_API_KEY" in st.secrets: st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
if "TELEGRAM_TOKEN" in st.secrets: st.session_state.tg_token = st.secrets["TELEGRAM_TOKEN"]
if "TELEGRAM_CHAT_ID" in st.secrets: st.session_state.tg_chat = st.secrets["TELEGRAM_CHAT_ID"]

# Init session state defaults
for key in ['api_key', 'tg_token', 'tg_chat', 'apex_df', 'apex_excel']:
    if key not in st.session_state: st.session_state[key] = None

# ==========================================
# 3. ENGINE: APEX (SMC & Trend)
# ==========================================
class ApexEngine:
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
        tr = ApexEngine.calculate_atr(df, length)
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
    def detect_smc(df):
        df['Pivot_High'] = df['High'].rolling(21, center=True).max()
        df['Pivot_Low'] = df['Low'].rolling(21, center=True).min()
        recent_high = df['High'].shift(1).rolling(20).max()
        bos = (df['Close'] > recent_high) & (df['Close'].shift(1) <= recent_high.shift(1))
        fvg = (df['Low'] > df['High'].shift(2))
        fvg_size = (df['Low'] - df['High'].shift(2))
        return bos, fvg, fvg_size

    @staticmethod
    def run_analysis(df):
        if len(df) < 60: return None
        # Trend
        base = ApexEngine.calculate_hma(df['Close'], 55)
        atr = ApexEngine.calculate_atr(df, 55)
        upper, lower = base + (atr*1.5), base - (atr*1.5)
        
        df['Apex_Trend'] = np.where(df['Close'] > upper, 1, np.where(df['Close'] < lower, -1, 0))
        df['Apex_Trend'] = df['Apex_Trend'].replace(0, method='ffill')
        
        # Signals
        df['ADX'] = ApexEngine.calculate_adx(df)
        df['WT'] = ApexEngine.calculate_wavetrend(df)
        vol_ma = df['Volume'].rolling(20).mean()
        
        buy = (df['Apex_Trend']==1) & (df['WT']<60) & (df['WT']>df['WT'].shift(1)) & (df['ADX']>20) & (df['Volume']>vol_ma)
        bos, fvg, fvg_sz = ApexEngine.detect_smc(df)
        
        last = df.iloc[-1]
        return {
            "Price": last['Close'], "Trend": "BULL" if last['Apex_Trend']==1 else "BEAR",
            "WT": last['WT'], "ADX": last['ADX'], "Buy_Sig": buy.tail(3).any(),
            "BOS": bos.tail(3).any(), "FVG": fvg.iloc[-1], "FVG_Size": fvg_sz.iloc[-1]
        }

# ==========================================
# 4. ENGINE: DARKPOOL (Institutional)
# ==========================================
class DarkPoolEngine:
    @staticmethod
    def calc_composite(df):
        # Base
        df['HMA55'] = ApexEngine.calculate_hma(df['Close'], 55)
        df['ATR'] = ApexEngine.calculate_atr(df, 14)
        
        # Squeeze
        basis = df['Close'].rolling(20).mean()
        dev = df['Close'].rolling(20).std() * 2
        kelt = basis + (df['ATR'] * 1.5)
        df['Squeeze'] = ((basis - dev) > (basis - (df['ATR'] * 1.5))) & ((basis + dev) < kelt)
        
        # LinReg Mom
        delta = df['Close'] - basis
        x = np.arange(20)
        df['Mom'] = delta.rolling(20).apply(lambda y: linregress(x, y)[0], raw=True)
        
        # Money Flow
        mf = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        df['MF_Vol'] = (mf * df['Volume']).ewm(span=3).mean()
        
        # Vector
        st_val, st_dir = DarkSingularityEngine.supertrend(df, 10, 4)
        df['Vector'] = st_dir
        
        # Score
        df['Score'] = np.where(df['Close'] > df['HMA55'], 1, -1) + np.sign(df['Mom']) + df['Vector']
        return df

    @staticmethod
    def get_macro_data():
        tickers = {
            "S&P 500": "SPY", "Bitcoin": "BTC-USD", "Gold": "GC=F", 
            "10Y Yield": "^TNX", "VIX": "^VIX", "DXY": "DX-Y.NYB"
        }
        try:
            data = yf.download(list(tickers.values()), period="5d", progress=False)['Close']
            res = {}
            for k, v in tickers.items():
                if v in data.columns:
                    curr = data[v].iloc[-1]
                    prev = data[v].iloc[-2]
                    res[k] = {"Price": curr, "Chg": ((curr-prev)/prev)*100}
            return res
        except: return {}

# ==========================================
# 5. ENGINE: DARK SINGULARITY (Quant)
# ==========================================
class DarkSingularityEngine:
    @staticmethod
    def supertrend(df, period=10, multiplier=3):
        atr = ApexEngine.calculate_atr(df, period)
        hl2 = (df['High'] + df['Low']) / 2
        up = hl2 + (multiplier * atr); dn = hl2 - (multiplier * atr)
        st = np.zeros(len(df)); trend = np.zeros(len(df))
        close = df['Close'].values; up_val = up.values; dn_val = dn.values
        
        # Optimized Iteration
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
    def chop_index(df, length=14):
        tr = ApexEngine.calculate_atr(df, 1)
        atr_sum = tr.rolling(length).sum()
        h_max = df['High'].rolling(length).max()
        l_min = df['Low'].rolling(length).min()
        return 100 * np.log10(atr_sum / (h_max - l_min).replace(0, 1)) / np.log10(length)

    @staticmethod
    def vector_flux(df, params):
        eff = (df['Close'] - df['Open']).abs() / (df['High'] - df['Low']).replace(0, 1)
        v_flux = df['Volume'] / df['Volume'].rolling(params['vol_len']).mean()
        return (np.sign(df['Close'] - df['Open']) * eff * v_flux).ewm(span=params['smooth']).mean()

    @staticmethod
    def run_engine(df, params):
        df['ST'], df['Trend'] = DarkSingularityEngine.supertrend(df, params['st_len'], params['st_mult'])
        df['Chop'] = DarkSingularityEngine.chop_index(df, params['chop_len'])
        df['Vector'] = DarkSingularityEngine.vector_flux(df, params)
        
        df['Signal_Long'] = (df['Trend']==1) & (df['Trend'].shift(1)==-1) & (df['Chop']<params['chop_thresh'])
        df['Signal_Short'] = (df['Trend']==-1) & (df['Trend'].shift(1)==1) & (df['Chop']<params['chop_thresh'])
        return df

# ==========================================
# 6. ENGINE: MOBILE (Intraday)
# ==========================================
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
# 7. UTILITIES: DATA & COMMS
# ==========================================
@st.cache_data(ttl=300)
def fetch_data(ticker, interval="1d", period="1y"):
    try:
        # Map for YF
        p_map = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
        per = p_map.get(interval, period)
        intv = "1h" if interval == "4h" else interval
        
        df = yf.download(ticker, period=per, interval=intv, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        if interval == "4h":
            agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
            df = df.resample('4h').agg(agg).dropna()
        return df
    except: return None

def ai_brief(ticker, data, key):
    if not key: return "⚠️ Connect OpenAI Key"
    client = openai.OpenAI(api_key=key)
    prompt = f"Analyze {ticker}: Price {data['Price']}, Trend {data['Trend']}, Momentum {data['WT']}. Give 3 bullet points: Structure, Volume, Execution."
    try:
        res = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}])
        return res.choices[0].message.content
    except Exception as e: return str(e)

def broadcast(msg, file=None):
    if st.session_state.tg_token and st.session_state.tg_chat:
        try:
            url = f"https://api.telegram.org/bot{st.session_state.tg_token}/sendMessage"
            requests.post(url, data={"chat_id": st.session_state.tg_chat, "text": msg})
            if file:
                f_url = f"https://api.telegram.org/bot{st.session_state.tg_token}/sendDocument"
                requests.post(f_url, data={"chat_id": st.session_state.tg_chat}, files={"document": ("Report.xlsx", file, "application/vnd.ms-excel")})
            return True
        except: return False
    return False

# ==========================================
# 8. MAIN UI CONTROLLER
# ==========================================
st.sidebar.markdown("## 💠 TITAN SYSTEM")
mode = st.sidebar.radio("MODE SELECT", ["🛡️ Apex Scanner", "👁️ DarkPool Terminal", "💀 Dark Singularity", "📱 Mobile Desk"])

with st.sidebar.expander("🔑 CONNECTIONS"):
    k1 = st.text_input("OpenAI Key", value=st.session_state.api_key, type="password")
    if k1: st.session_state.api_key = k1
    k2 = st.text_input("TG Token", value=st.session_state.tg_token, type="password")
    if k2: st.session_state.tg_token = k2
    k3 = st.text_input("TG Chat", value=st.session_state.tg_chat)
    if k3: st.session_state.tg_chat = k3

# ------------------------------------------------
# MODE 1: APEX SCANNER (Crypto Focus)
# ------------------------------------------------
if mode == "🛡️ Apex Scanner":
    st.markdown("### 🔭 APEX CRYPTO SCOUT")
    st.caption("Engine: HMA Trend // SMC Structure // WaveTrend Momentum")
    
    universe = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "XRP-USD", "AVAX-USD", "LINK-USD", "DOGE-USD", "MATIC-USD"]
    
    if st.button("RUN SCAN SEQUENCE"):
        prog = st.progress(0)
        results = []
        for i, t in enumerate(universe):
            prog.progress((i+1)/len(universe))
            df = fetch_data(t)
            if df is not None:
                res = ApexEngine.run_analysis(df)
                if res:
                    score = 0
                    if res['Trend'] == "BULL": score += 1
                    if res['Buy_Sig']: score += 2
                    if res['BOS']: score += 1
                    if res['FVG']: score += 1
                    
                    results.append({
                        "Ticker": t, "Price": res['Price'], "Trend": res['Trend'], 
                        "Score": score, "Signal": "BUY" if res['Buy_Sig'] else "-",
                        "BOS": "YES" if res['BOS'] else "-", "FVG": "YES" if res['FVG'] else "-"
                    })
        prog.empty()
        
        if results:
            df_res = pd.DataFrame(results).sort_values("Score", ascending=False)
            st.session_state.apex_df = df_res
            
            # Display
            st.dataframe(df_res, use_container_width=True)
            
            # AI Analysis on Top Pick
            top = df_res.iloc[0]
            st.markdown(f"#### 🧠 AI ANALYSIS: {top['Ticker']}")
            st.info(ai_brief(top['Ticker'], top, st.session_state.api_key))
            
            # Export
            buf = io.BytesIO()
            with pd.ExcelWriter(buf) as writer: df_res.to_excel(writer, index=False)
            
            c1, c2 = st.columns(2)
            c1.download_button("📥 EXPORT DATA", buf.getvalue(), "Apex_Scan.xlsx")
            if c2.button("📡 BROADCAST TOP PICK"):
                msg = f"🛡️ APEX ALERT: {top['Ticker']} | Trend: {top['Trend']} | Signal: {top['Signal']}"
                if broadcast(msg, io.BytesIO(buf.getvalue())): st.success("SENT")
                else: st.error("CHECK KEYS")

# ------------------------------------------------
# MODE 2: DARKPOOL TERMINAL (Institutional)
# ------------------------------------------------
elif mode == "👁️ DarkPool Terminal":
    st.markdown("### 👁️ DARKPOOL TERMINAL")
    
    # Macro Bar
    macro = DarkPoolEngine.get_macro_data()
    if macro:
        cols = st.columns(6)
        for i, (k,v) in enumerate(macro.items()):
            cols[i].metric(k, f"{v['Price']:.2f}", f"{v['Chg']:.2f}%")
            
    c1, c2 = st.columns([3, 1])
    ticker = c1.text_input("ASSET", "BTC-USD").upper()
    tf = c2.selectbox("TIMEFRAME", ["15m", "1h", "4h", "1d"], index=3)
    
    if st.button("LOAD INSTITUTIONAL DATA"):
        df = fetch_data(ticker, tf)
        if df is not None:
            df = DarkPoolEngine.calc_composite(df)
            last = df.iloc[-1]
            
            # KPIS
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f'<div class="metric-card"><div class="metric-label">COMPOSITE SCORE</div><div class="metric-value">{last["Score"]:.0f} / 3</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="metric-card"><div class="metric-label">TREND BASELINE</div><div class="metric-value">{last["HMA55"]:.2f}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="metric-card"><div class="metric-label">MOMENTUM FLUX</div><div class="metric-value">{last["Mom"]:.2f}</div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="metric-card"><div class="metric-label">MONEY FLOW</div><div class="metric-value">{last["MF_Vol"]:.2f}</div></div>', unsafe_allow_html=True)
            
            # Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            
            # Price & HMA
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['HMA55'], line=dict(color='#FFD700', width=2), name='HMA 55'), row=1, col=1)
            
            # Composite Bar
            cols = ['#00E676' if v > 0 else '#FF1744' for v in df['Score']]
            fig.add_trace(go.Bar(x=df.index, y=df['Score'], marker_color=cols, name='Composite'), row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor="#000", plot_bgcolor="#000")
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# MODE 3: DARK SINGULARITY (Quant)
# ------------------------------------------------
elif mode == "💀 Dark Singularity":
    st.markdown("### 💀 DARK SINGULARITY")
    st.caption("Engine: SuperTrend // Choppiness // Vector Flux")
    
    q_sym = st.text_input("TICKER", "BTC-USD").upper()
    
    with st.expander("QUANT PARAMETERS"):
        st_len = st.slider("ATR Length", 5, 50, 10)
        st_mult = st.slider("ST Multiplier", 1.0, 10.0, 4.0)
        chop_thr = st.slider("Chop Threshold", 40, 70, 60)
        
    if st.button("EXECUTE SINGULARITY ENGINE"):
        df = fetch_data(q_sym)
        if df is not None:
            params = {'st_len': st_len, 'st_mult': st_mult, 'chop_len': 14, 'chop_thresh': chop_thr, 'vol_len': 50, 'smooth': 5}
            df = DarkSingularityEngine.run_engine(df, params)
            last = df.iloc[-1]
            
            # Status
            state = "TRENDING" if last['Chop'] < chop_thr else "CHOPPY / RANGING"
            st_col = "bull" if last['Chop'] < chop_thr else "neu"
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><div class="metric-label">MARKET STATE</div><div class="metric-value {st_col}">{state}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-label">VECTOR STRENGTH</div><div class="metric-value">{last["Vector"]:.3f}</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-label">SUPERTREND</div><div class="metric-value">{last["ST"]:.2f}</div></div>', unsafe_allow_html=True)
            
            # Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25])
            
            # Candles
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
            
            # SuperTrend Line
            st_line = df['ST']
            fig.add_trace(go.Scatter(x=df.index, y=st_line, line=dict(color='#00E676', width=2), name='ST'), row=1, col=1)
            
            # Signals
            longs = df[df['Signal_Long']]
            if not longs.empty:
                fig.add_trace(go.Scatter(x=longs.index, y=longs['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00E676'), name='LONG'), row=1, col=1)
                
            # Vector
            v_cols = ['#00E676' if v > 0 else '#FF1744' for v in df['Vector']]
            fig.add_trace(go.Bar(x=df.index, y=df['Vector'], marker_color=v_cols), row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# MODE 4: MOBILE DESK (Titan)
# ------------------------------------------------
elif mode == "📱 Mobile Desk":
    st.markdown("### 📱 MOBILE COMMAND")
    
    # Clock
    components.html("""
    <div style="font-family:'Roboto Mono'; color:#888; text-align:center; font-size:14px; background:#000; padding:5px;">
    UTC <span id="clock" style="color:#fff; font-weight:bold;"></span>
    </div>
    <script>setInterval(()=>document.getElementById('clock').innerText=new Date().toLocaleTimeString('en-GB',{timeZone:'UTC'}),1000)</script>
    """, height=40)
    
    m_sym = st.text_input("ASSET", "BTC-USD").upper()
    
    if st.button("REFRESH MOBILE DATA"):
        df = fetch_data(m_sym, "1h")
        if df is not None:
            # Re-use Singularity Logic for Mobile
            params = {'st_len': 10, 'st_mult': 3, 'chop_len': 14, 'chop_thresh': 60, 'vol_len': 20, 'smooth': 3}
            df = DarkSingularityEngine.run_engine(df, params)
            last = df.iloc[-1]
            fibs = MobileEngine.calculate_fibs(df)
            stop = min(last['ST'], fibs['0.618']) if last['Trend']==1 else max(last['ST'], fibs['0.618'])
            
            # Render HTML Card
            st.markdown(MobileEngine.generate_report(last, m_sym, fibs, stop), unsafe_allow_html=True)
            
            # Mini Chart
            fig = go.Figure(go.Scatter(x=df.index[-24:], y=df['Close'].tail(24), line=dict(color='#00E676', width=2)))
            fig.update_layout(template="plotly_dark", height=200, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor="#000", plot_bgcolor="#000")
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("🔥 SEND ALERT"):
                msg = f"📱 {m_sym} | {last['Close']:.2f} | T: {last['Trend']}"
                if broadcast(msg): st.success("SENT")
                else: st.error("FAIL")
