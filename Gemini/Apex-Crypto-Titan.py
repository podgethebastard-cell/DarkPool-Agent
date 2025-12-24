import streamlit as st
import pandas as pd
import yfinance as yf
import openai
from datetime import datetime, date
import io
import xlsxwriter
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
import urllib.parse
import streamlit.components.v1 as components

# ==========================================
# 1. CONFIGURATION & STYLING (TITAN UI)
# ==========================================
st.set_page_config(layout="wide", page_title="Apex Crypto Titan", page_icon="⚡")

st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
        font-family: 'SF Mono', 'Roboto Mono', monospace;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00E676;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-sub {
        font-size: 0.7rem;
        color: #666;
        margin-top: 4px;
    }
    
    /* Signal Colors */
    .bull { color: #00E676 !important; text-shadow: 0 0 10px rgba(0, 230, 118, 0.4); }
    .bear { color: #FF1744 !important; text-shadow: 0 0 10px rgba(255, 23, 68, 0.4); }
    .neu  { color: #B0BEC5 !important; }
    
    /* Header Glow */
    .titan-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E676, #2979FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: #121212;
        border-radius: 8px;
        color: #888;
        border: 1px solid #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00E676 !important;
        color: #000 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIC ENGINES
# ==========================================

# --- A. APEX ENGINE (SMC, HMA, BOS, FVG) ---
class ApexEngine:
    @staticmethod
    def calculate_hma(series, length):
        """Calculates Hull Moving Average (HMA)"""
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
        true_range = ranges.max(axis=1)
        return true_range.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def calculate_adx(df, length=14):
        up = df['High'].diff()
        down = -df['Low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        tr = ApexEngine.calculate_atr(df, length)
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean() / tr)
        
        sum_di = plus_di + minus_di
        sum_di = sum_di.replace(0, 1) 
        
        dx = 100 * np.abs(plus_di - minus_di) / sum_di
        return dx.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def calculate_wavetrend(df):
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=10, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=10, adjust=False).mean()
        d = d.replace(0, 0.0001)
        ci = (ap - esa) / (0.015 * d)
        tci = ci.ewm(span=21, adjust=False).mean() 
        return tci

    @staticmethod
    def detect_smc(df):
        """Detects BOS (Break of Structure) and FVGs"""
        lookback = 10
        # Simple rolling max/min to identify local structures
        df['Pivot_High'] = df['High'].rolling(window=lookback*2+1, center=True).max()
        df['Pivot_Low'] = df['Low'].rolling(window=lookback*2+1, center=True).min()
        
        # BOS Detection
        recent_high = df['High'].shift(1).rolling(20).max()
        bos_bull = (df['Close'] > recent_high) & (df['Close'].shift(1) <= recent_high.shift(1))
        
        # FVG (Fair Value Gap) - Bullish
        fvg_bull = (df['Low'] > df['High'].shift(2))
        fvg_size = (df['Low'] - df['High'].shift(2))
        
        return bos_bull, fvg_bull, fvg_size

    @staticmethod
    def run_full_analysis(df):
        if len(df) < 60: return None
        
        # Trend
        len_main = 55
        mult = 1.5
        baseline = ApexEngine.calculate_hma(df['Close'], len_main)
        atr = ApexEngine.calculate_atr(df, len_main)
        upper = baseline + (atr * mult)
        lower = baseline - (atr * mult)
        
        # Trend State
        trends = []
        curr_trend = 0
        upper_vals = upper.values
        lower_vals = lower.values
        close_vals = df['Close'].values
        
        for i in range(len(df)):
            c = close_vals[i]
            u = upper_vals[i]
            l = lower_vals[i]
            
            if c > u: curr_trend = 1
            elif c < l: curr_trend = -1
            trends.append(curr_trend)
        
        df['Apex_Trend'] = trends
        
        # Signals
        df['ADX'] = ApexEngine.calculate_adx(df)
        df['WaveTrend'] = ApexEngine.calculate_wavetrend(df)
        vol_ma = df['Volume'].rolling(20).mean()
        
        buy_signal = (
            (df['Apex_Trend'] == 1) & 
            (df['WaveTrend'] < 60) & 
            (df['WaveTrend'] > df['WaveTrend'].shift(1)) &
            (df['ADX'] > 20) &
            (df['Volume'] > vol_ma)
        )
        
        bos_bull, fvg_bull, fvg_size = ApexEngine.detect_smc(df)
        df['BOS_Bull'] = bos_bull
        df['FVG_Bull'] = fvg_bull
        df['FVG_Size'] = fvg_size

        last = df.iloc[-1]
        recent_buy = buy_signal.tail(3).any()
        recent_bos = df['BOS_Bull'].tail(3).any()
        has_fvg = df['FVG_Bull'].iloc[-1]
        
        return {
            "Price": last['Close'],
            "Trend_Val": last['Apex_Trend'],
            "Trend": "Bullish 🟢" if last['Apex_Trend'] == 1 else "Bearish 🔴",
            "WaveTrend": last['WaveTrend'],
            "ADX": last['ADX'],
            "Apex_Buy_Signal": recent_buy,
            "BOS_Alert": recent_bos,
            "FVG_Detected": has_fvg,
            "FVG_Size": last['FVG_Size'] if has_fvg else 0
        }

# --- B. TITAN MATH ENGINE (Advanced Indicators) ---
def calculate_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def calculate_linreg_mom(series, length=20):
    x = np.arange(length)
    slope = series.rolling(length).apply(lambda y: linregress(x, y)[0], raw=True)
    return slope

def calculate_titan_engine(df, ticker):
    """
    Combines Titan Indicators (DarkPool, Ichimoku, etc.)
    Note: Re-uses some Apex functions for efficiency where they overlap (ATR/HMA)
    """
    # 1. APEX BASE (Required for visual cloud)
    apex_len = 55
    apex_mult = 1.5
    df['Apex_Base'] = ApexEngine.calculate_hma(df['Close'], apex_len)
    df['Apex_ATR'] = ApexEngine.calculate_atr(df, apex_len)
    df['Apex_Upper'] = df['Apex_Base'] + (df['Apex_ATR'] * apex_mult)
    df['Apex_Lower'] = df['Apex_Base'] - (df['Apex_ATR'] * apex_mult)
    # Trend logic is handled in ApexEngine, but we need the bands for plotting
    
    # 2. SQUEEZE MOMENTUM
    df['Basis'] = df['Close'].rolling(20).mean()
    df['Dev'] = df['Close'].rolling(20).std() * 2.0
    df['UpperBB'] = df['Basis'] + df['Dev']
    df['LowerBB'] = df['Basis'] - df['Dev']
    df['ATR'] = ApexEngine.calculate_atr(df, 20)
    df['UpperKC'] = df['Basis'] + (df['ATR'] * 1.5)
    df['LowerKC'] = df['Basis'] - (df['ATR'] * 1.5)
    df['Squeeze_On'] = (df['LowerBB'] > df['LowerKC']) & (df['UpperBB'] < df['UpperKC'])
    
    mean_hl = (df['High'].rolling(20).max() + df['Low'].rolling(20).min()) / 2
    avg_val = (mean_hl + df['Basis']) / 2
    delta = df['Close'] - avg_val
    df['Sqz_Mom'] = calculate_linreg_mom(delta, 20)

    # 3. GANN HIGH LOW
    gann_len = 3
    df['Gann_High'] = df['High'].rolling(gann_len).mean()
    df['Gann_Low'] = df['Low'].rolling(gann_len).mean()
    
    # 4. VOL DELTA & MFI
    df['Vol_Delta'] = np.where(df['Close'] >= df['Open'], df['Volume'], -df['Volume'])
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    pos_flow = np.where(tp > tp.shift(1), rmf, 0)
    neg_flow = np.where(tp < tp.shift(1), rmf, 0)
    mfi_ratio = pd.Series(pos_flow).rolling(14).sum() / pd.Series(neg_flow).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # 5. ICHIMOKU
    nine_high = df['High'].rolling(window=9).max()
    nine_low = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (nine_high + nine_low) / 2
    twenty_six_high = df['High'].rolling(window=26).max()
    twenty_six_low = df['Low'].rolling(window=26).min()
    df['Kijun'] = (twenty_six_high + twenty_six_low) / 2
    df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    fifty_two_high = df['High'].rolling(window=52).max()
    fifty_two_low = df['Low'].rolling(window=52).min()
    df['SpanB'] = ((fifty_two_high + fifty_two_low) / 2).shift(26)

    # 6. DARKPOOL MACD & STOCH
    e12 = df['Close'].ewm(span=12).mean()
    e26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = e12 - e26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Hist'] = (df['MACD'] - df['Signal']) * (df['Volume'] / df['Volume'].rolling(20).mean())
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    min_rsi = df['RSI'].rolling(14).min()
    max_rsi = df['RSI'].rolling(14).max()
    stoch = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    df['Stoch_K'] = stoch.rolling(3).mean() * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # 7. DARKPOOL MAs (Scoring)
    df['DP_MA10'] = df['Close'].ewm(span=10).mean()
    df['DP_MA20'] = df['Close'].ewm(span=20).mean()
    df['DP_MA50'] = df['Close'].ewm(span=50).mean()
    df['DP_MA100'] = df['Close'].ewm(span=100).mean()
    df['DP_MA200'] = df['Close'].ewm(span=200).mean()
    
    df['DP_Score'] = (
        np.where(df['Close'] > df['DP_MA10'], 1, 0) + 
        np.where(df['Close'] > df['DP_MA20'], 1, 0) + 
        np.where(df['Close'] > df['DP_MA50'], 1, 0) + 
        np.where(df['Close'] > df['DP_MA100'], 1, 0) + 
        np.where(df['Close'] > df['DP_MA200'], 1, 0)
    )

    return df

def calculate_strategies(df):
    """4 Core Strategy Signals"""
    # 1. Momentum (12)
    mom = df['Close'] - df['Close'].shift(12)
    df['Sig_Mom'] = np.where((mom > 0) & (mom.shift(1) > 0), 1, np.where((mom < 0) & (mom.shift(1) < 0), -1, 0))
    
    # 2. ADX Breakout (Close > 20 High + ADX < 25)
    box_high = df['High'].rolling(20).max().shift(1)
    box_low = df['Low'].rolling(20).min().shift(1)
    # Using Apex calculated ADX if available, else calc
    if 'ADX' not in df.columns: df['ADX'] = ApexEngine.calculate_adx(df)
        
    df['Sig_ADX'] = np.where((df['Close'] > box_high) & (df['ADX'] < 25), 1,
                       np.where((df['Close'] < box_low) & (df['ADX'] < 25), -1, 0))

    # 3. Bollinger Directed
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['Sig_BB'] = np.where(df['Close'] < (sma20 - 2*std20), 1, np.where(df['Close'] > (sma20 + 2*std20), -1, 0))

    # 4. RSI Strategy (30/70)
    if 'RSI' not in df.columns:
         delta = df['Close'].diff()
         gain = delta.where(delta > 0, 0)
         loss = -delta.where(delta < 0, 0)
         avg_gain = gain.rolling(14).mean()
         avg_loss = loss.rolling(14).mean()
         rs = avg_gain / avg_loss
         df['RSI'] = 100 - (100 / (1 + rs))
    df['Sig_RSI'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    
    return df

# ==========================================
# 3. DATA UTILS
# ==========================================

@st.cache_data(ttl=600)
def get_institutional_trend(ticker):
    """Fetches Daily and Weekly data for Institutional trend context"""
    try:
        df_d = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
        df_w = yf.download(ticker, period="2y", interval="1wk", progress=False)
        if isinstance(df_w.columns, pd.MultiIndex): df_w.columns = df_w.columns.get_level_values(0)
        ema_d = df_d['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        ema_w = df_w['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        return ema_d, ema_w
    except:
        return 0, 0

def get_financials_scan(ticker):
    """Fetches basic info for Scanner"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker": ticker,
            "name": info.get('shortName', ticker),
            "market_cap": info.get('marketCap', 0)
        }
    except:
        return None

def get_history_scan(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="1y") 
    except:
        return None

@st.cache_data(ttl=300)
def get_data_dashboard(ticker, interval):
    p_map = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
    period = p_map.get(interval, "1y")
    d_int = "1h" if interval == "4h" else interval
    df = yf.download(ticker, period=period, interval=d_int, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if interval == "4h":
        agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
        df = df.resample('4h').agg(agg).dropna()
    return df

# ==========================================
# 4. SIDEBAR & ASSETS
# ==========================================
st.sidebar.header("⚡ System Control")

# --- CREDENTIALS ---
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("OpenAI: Loaded")
else:
    api_key = st.sidebar.text_input("OpenAI Key", type="password")

if "TELEGRAM_TOKEN" in st.secrets: st.session_state.tg_token = st.secrets["TELEGRAM_TOKEN"]
if "TELEGRAM_CHAT_ID" in st.secrets: st.session_state.tg_chat = st.secrets["TELEGRAM_CHAT_ID"]

with st.sidebar.expander("📡 Broadcast Settings"):
    tg_token = st.text_input("Bot Token", value=st.session_state.get('tg_token', ""), type="password")
    tg_chat = st.text_input("Chat ID", value=st.session_state.get('tg_chat', ""))

# --- ASSET UNIVERSE ---
CRYPTO_UNIVERSE = [
    # Majors
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    # L1 / L2 / Infra
    "ADA-USD", "AVAX-USD", "LINK-USD", "MATIC-USD", "DOT-USD", "NEAR-USD",
    "ATOM-USD", "ARB-USD", "OP-USD", "SUI-USD", "APT-USD", "INJ-USD",
    # DeFi / Exchange
    "UNI-USD", "AAVE-USD", "MKR-USD", "RUNE-USD", "LDO-USD",
    # Meme / AI
    "DOGE-USD", "SHIB-USD", "PEPE-USD", "WIF-USD", "FET-USD", "RNDR-USD"
]

crypto_assets_dict = {
    "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Solana (SOL)": "SOL-USD",
    "BNB (BNB)": "BNB-USD", "XRP (XRP)": "XRP-USD", "Dogecoin (DOGE)": "DOGE-USD",
    "Avalanche (AVAX)": "AVAX-USD", "Chainlink (LINK)": "LINK-USD", "Pepe (PEPE)": "PEPE-USD",
    "Fetch.ai (FET)": "FET-USD", "Render (RNDR)": "RNDR-USD", "Dogwifhat (WIF)": "WIF-USD"
}

# --- MODE SELECTION ---
app_mode = st.sidebar.radio("Operation Mode", ["🛡️ Titan Terminal (Single)", " telescopes Apex Scanner (Batch)"])

# ==========================================
# 5. MODE: APEX SCANNER
# ==========================================
if app_mode == " telescopes Apex Scanner (Batch)":
    st.markdown(f'<div class="titan-header">🔭 Apex Crypto Scanner</div>', unsafe_allow_html=True)
    st.markdown("""
    **SMC & Trend Engine:** Scans the market for **Apex Trends** (HMA 55), **BOS** (Break of Structure), and **FVG** (Fair Value Gaps).
    """)

    if "apex_df" not in st.session_state: st.session_state.apex_df = None
    if "apex_excel" not in st.session_state: st.session_state.apex_excel = None

    if st.button("▶️ START MARKET SCAN"):
        progress_bar = st.progress(0)
        status = st.empty()
        results = []
        
        total = len(CRYPTO_UNIVERSE)
        for i, ticker in enumerate(CRYPTO_UNIVERSE):
            status.text(f"Analyzing Order Flow: {ticker}...")
            progress_bar.progress((i+1)/total)
            
            # 1. Fetch
            data = get_financials_scan(ticker)
            if not data: continue
            
            # 2. History
            hist = get_history_scan(ticker)
            if hist is None or len(hist) < 60: continue
            
            # 3. Engine
            apex_data = ApexEngine.run_full_analysis(hist)
            if not apex_data: continue
            
            # 4. Scoring
            score = 0
            tags = []
            
            if apex_data['Trend_Val'] == 1:
                score += 1
            if apex_data['Apex_Buy_Signal']:
                score += 3
                tags.append("BUY SIG")
            if apex_data['BOS_Alert']:
                score += 2
                tags.append("BOS")
            if apex_data['FVG_Detected']:
                score += 1
                tags.append("FVG")
                
            if score >= 1:
                row = data.copy()
                row.update(apex_data)
                row['Score'] = score
                row['Tags'] = ", ".join(tags)
                results.append(row)
        
        progress_bar.empty()
        status.empty()
        
        if results:
            df_res = pd.DataFrame(results).sort_values(by='Score', ascending=False).reset_index(drop=True)
            st.session_state.apex_df = df_res
            
            # Excel Generation
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                df_res.to_excel(writer, index=False, sheet_name="Apex SMC")
            buf.seek(0)
            st.session_state.apex_excel = buf.getvalue()
            st.success(f"Scan Complete. Found {len(df_res)} setups.")
        else:
            st.warning("No setups found matching criteria.")

    # DISPLAY RESULTS
    if st.session_state.apex_df is not None:
        df = st.session_state.apex_df
        st.dataframe(df[['ticker', 'Price', 'Trend', 'Tags', 'Score', 'ADX', 'WaveTrend']], use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Download Excel Report", st.session_state.apex_excel, "Apex_Crypto_Scan.xlsx")
        with c2:
            if tg_token and tg_chat and st.button("📡 Broadcast Top Pick"):
                top = df.iloc[0]
                msg = f"🏛️ **APEX SCAN WINNER**\n\nPair: {top['ticker']}\nPrice: ${top['Price']:.4f}\nTags: {top['Tags']}\nTrend: {top['Trend']}"
                try:
                    requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage", data={"chat_id": tg_chat, "text": msg})
                    # Send File
                    file_buf = io.BytesIO(st.session_state.apex_excel)
                    requests.post(f"https://api.telegram.org/bot{tg_token}/sendDocument",
                                  data={"chat_id": tg_chat},
                                  files={"document": ("Apex_Scan.xlsx", file_buf, "application/vnd.ms-excel")})
                    st.success("Sent to Telegram!")
                except Exception as e:
                    st.error(f"Telegram Error: {e}")

# ==========================================
# 6. MODE: TITAN TERMINAL (SINGLE ASSET)
# ==========================================
else:
    # Sidebar Selection for Terminal
    search_mode = st.sidebar.radio("Asset Selection", ["Top List", "Custom Search"], horizontal=True)
    if search_mode == "Top List":
        ticker_name = st.sidebar.selectbox("Target Asset", list(crypto_assets_dict.keys()))
        ticker = crypto_assets_dict[ticker_name]
    else:
        custom_symbol = st.sidebar.text_input("Enter Symbol (e.g. PEPE)", value="BTC").upper().strip()
        ticker = f"{custom_symbol}-USD" if "-" not in custom_symbol else custom_symbol
        ticker_name = ticker

    interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=2)

    st.markdown(f'<div class="titan-header">🛡️ TITAN TERMINAL: {ticker}</div>', unsafe_allow_html=True)

    # 1. TradingView Widget
    tv_int_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    tv_sym = ticker.replace("-USD", "USDT")
    tv_sym = f"BINANCE:{tv_sym}" # Assumption for major crypto
    
    components.html(
        f"""<div class="tradingview-widget-container"><div id="tradingview_widget"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{"width": "100%","height": 450,"symbol": "{tv_sym}","interval": "{tv_int_map[interval]}","timezone": "Etc/UTC","theme": "dark","style": "1","locale": "en","toolbar_bg": "#f1f3f6","enable_publishing": false,"hide_side_toolbar": false,"allow_symbol_change": true,"container_id": "tradingview_widget"}});</script></div>""",
        height=460,
    )

    # 2. Fetch & Calculate
    df = get_data_dashboard(ticker, interval)
    
    if df is not None and not df.empty:
        # RUN TITAN MATH
        df = calculate_titan_engine(df, ticker)
        df = calculate_strategies(df)
        
        # RUN APEX SMC MATH (Injection)
        apex_res = ApexEngine.run_full_analysis(df.copy()) # Use copy to avoid index conflicts
        
        last = df.iloc[-1]
        
        # Integration Variables
        curr_price = last['Close']
        atr_val = last['ATR']
        trend_val = apex_res['Trend_Val']
        trend_txt = apex_res['Trend']
        
        # Determine Direction for AI/Broadcast
        direction = "LONG" if trend_val == 1 else "SHORT" if trend_val == -1 else "NEUTRAL"
        
        if direction == "LONG":
            stop_loss = curr_price - (2 * atr_val)
            take_profit = curr_price + (3 * atr_val)
        else:
            stop_loss = curr_price + (2 * atr_val)
            take_profit = curr_price - (3 * atr_val)

        # --- METRICS GRID ---
        st.markdown("### 🧬 Market DNA")
        c1, c2, c3, c4, c5 = st.columns(5)
        
        def card(label, value, sub, condition):
            color_class = "bull" if condition == 1 else "bear" if condition == -1 else "neu"
            return f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {color_class}">{value}</div><div class="metric-sub">{sub}</div></div>"""
        
        # Apex Trend
        c1.markdown(card("Apex Trend", trend_txt, f"Cloud: {last['Apex_Base']:.2f}", trend_val), unsafe_allow_html=True)
        
        # SMC Status
        smc_txt = "BOS DETECTED" if apex_res['BOS_Alert'] else "FVG ZONE" if apex_res['FVG_Detected'] else "NO STRUCTURE"
        smc_cond = 1 if apex_res['BOS_Alert'] else (-1 if apex_res['FVG_Detected'] else 0) # Arbitrary color coding
        c2.markdown(card("Smart Money", smc_txt, f"FVG Size: {apex_res['FVG_Size']:.2f}", smc_cond), unsafe_allow_html=True)
        
        # Momentum
        mom_cond = last['Sig_Mom']
        mom_txt = "POSITIVE" if mom_cond == 1 else "NEGATIVE"
        c3.markdown(card("Momentum", mom_txt, f"Squeeze: {last['Sqz_Mom']:.2f}", mom_cond), unsafe_allow_html=True)
        
        # Institutional Trend
        ema_d, ema_w = get_institutional_trend(ticker)
        inst_cond = 1 if ema_d > ema_w else -1
        inst_txt = "MACRO BULL" if inst_cond == 1 else "MACRO BEAR"
        c4.markdown(card("Inst. Trend (1D/1W)", inst_txt, "EMA Cloud", inst_cond), unsafe_allow_html=True)
        
        # Money Flow
        mfi_val = last['MFI']
        mfi_cond = 1 if mfi_val < 20 else -1 if mfi_val > 80 else 0
        mfi_txt = "OVERSOLD" if mfi_val < 20 else "OVERBOUGHT" if mfi_val > 80 else "NEUTRAL"
        c5.markdown(card("Money Flow", mfi_txt, f"MFI: {mfi_val:.1f}", mfi_cond), unsafe_allow_html=True)

        # --- TABS ---
        st.markdown("<br>", unsafe_allow_html=True)
        tab_main, tab_smc, tab_dp, tab_ai, tab_cast = st.tabs(["🌊 Apex & Cloud", "🏛️ SMC & Volume", "💀 DarkPool", "🤖 AI Analyst", "📡 Broadcast"])
        
        with tab_main:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)' if trend_val == 1 else 'rgba(255, 23, 68, 0.1)', line=dict(width=0), name="Apex Cloud"))
            fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Base'], line=dict(color='#00E676' if trend_val == 1 else '#FF1744', width=2), name="Apex HMA 55"))
            fig.update_layout(height=600, template="plotly_dark", title=f"Apex Trend ({interval})", xaxis_rangeslider_visible=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)

        with tab_smc:
            fig_smc = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig_smc.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            # Add Pivot Points logic visual
            if 'Pivot_High' in df.columns:
                 fig_smc.add_trace(go.Scatter(x=df.index, y=df['Pivot_High'], mode='markers', marker=dict(color='red', size=5), name="Swing High"), row=1, col=1)
                 fig_smc.add_trace(go.Scatter(x=df.index, y=df['Pivot_Low'], mode='markers', marker=dict(color='green', size=5), name="Swing Low"), row=1, col=1)
            
            # Volume Delta
            colors_vol = ['#00E676' if v > 0 else '#FF1744' for v in df['Vol_Delta']]
            fig_smc.add_trace(go.Bar(x=df.index, y=df['Vol_Delta'], marker_color=colors_vol, name="Vol Delta"), row=2, col=1)
            fig_smc.update_layout(height=600, template="plotly_dark", title="SMC Structure & Volume", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_smc, use_container_width=True)

        with tab_dp:
            fig_dp = go.Figure()
            fig_dp.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA50'], line=dict(color='#00E676', width=2), name="EMA 50"))
            fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA200'], line=dict(color='#FF1744', width=2), name="EMA 200"))
            fig_dp.update_layout(height=600, template="plotly_dark", title="Institutional Moving Averages", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_dp, use_container_width=True)

        with tab_ai:
            if st.button("Generate Alpha Report"):
                if not api_key:
                    st.error("Missing OpenAI API Key")
                else:
                    with st.spinner("Analyzing Market Structure..."):
                        client = openai.OpenAI(api_key=api_key)
                        
                        prompt = f"""
                        Act as a Senior Crypto SMC & Quantitative Trader. 
                        Analyze {ticker} ({interval}) at Price ${curr_price:.2f}.
                        
                        --- TECHNICAL DATA ---
                        1. Apex Trend: {trend_txt} (HMA 55 Base: ${last['Apex_Base']:.2f})
                        2. Smart Money Concepts: 
                           - BOS (Break of Structure): {apex_res['BOS_Alert']}
                           - FVG (Fair Value Gap): {apex_res['FVG_Detected']} (Size: {apex_res['FVG_Size']:.4f})
                        3. Institutional Trend (Daily/Weekly EMA): {inst_txt}
                        4. Momentum: {mom_txt} (Squeeze: {last['Sqz_Mom']:.2f})
                        5. MFI (Money Flow): {mfi_val:.1f}
                        
                        --- CALCULATED LEVELS ---
                        - LONG SETUP: Stop < ${stop_loss:.2f}, Target > ${take_profit:.2f}
                        - SHORT SETUP: Stop > ${stop_loss:.2f}, Target < ${take_profit:.2f}
                        
                        --- MISSION ---
                        Synthesize a TRADE PLAN in strictly formatted MARKDOWN.
                        
                        OUTPUT FORMAT:
                        ### 📋 Trade Plan: {ticker}
                        
                        **1. VERDICT:** LONG / SHORT / WAIT (Confidence 1-10)
                        * **SMC Context:** (Explain the BOS/FVG situation)
                        * **Confluence:** (Combine Trend + Momentum + Institutional Flow)
                        
                        **2. EXECUTION**
                        * **Entry Zone:** (Price)
                        * **Stop Loss:** (Price - Explain technical reason)
                        * **Take Profit:** (Price - R:R Ratio)
                        
                        **3. ALPHA NOTE**
                        * (One sentence key insight about volume or liquidity)
                        """
                        try:
                            res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":prompt}])
                            st.info(res.choices[0].message.content)
                        except Exception as e:
                            st.error(f"AI Error: {e}")

        with tab_cast:
            st.subheader("📡 Broadcast Center")
            
            sig_emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "⚪"
            
            broadcast_msg = f"""
🔥 TITAN SIGNAL: {ticker} ({interval})
{sig_emoji} DIRECTION: {direction}
🚪 ENTRY: ${curr_price:,.4f}
🛑 STOP: ${stop_loss:,.4f}
🎯 TARGET: ${take_profit:,.4f}

🌊 Trend: {trend_txt}
🏛️ SMC: {smc_txt}
📊 Momentum: {mom_txt}
💀 Macro: {inst_txt}

⚠️ *Not financial advice.*
#DarkPool #Titan #Crypto #{ticker.split('-')[0]}
            """
            
            msg_preview = st.text_area("Message Preview", value=broadcast_msg, height=250)
            
            c_tg, c_x = st.columns(2)
            
            if c_tg.button("Send to Telegram 🚀"):
                if tg_token and tg_chat:
                    try:
                        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                        data = {"chat_id": tg_chat, "text": msg_preview}
                        requests.post(url, data=data)
                        st.success("Signal Broadcasted!")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                else:
                    st.warning("Enter Telegram Keys in Sidebar")
                    
            if c_x.button("Post to X (Twitter)"):
                encoded = urllib.parse.quote(msg_preview)
                st.link_button("🐦 Launch Tweet", f"https://twitter.com/intent/tweet?text={encoded}")
    else:
        st.info("Loading Data or Invalid Ticker...")
