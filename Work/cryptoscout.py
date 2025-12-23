import streamlit as st
import pandas as pd
import yfinance as yf
import openai
from datetime import datetime, date
import io
import xlsxwriter
import requests
import numpy as np

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(page_title="Apex Crypto SMC Scout", layout="wide")
st.title("🏛️ Apex Crypto Trend & Liquidity Master (SMC)")
st.markdown("""
**SMC & Trend Edition (Crypto):** This agent runs the **Apex v8.0 Pine Script logic** in Python.
It scans the Crypto market for:
* 🌊 **Apex Trends:** HMA-based trend following with Volatility Bands.
* 🏛️ **Smart Money:** Detects **BOS** (Break of Structure) and **FVG** (Fair Value Gaps).
* 🚀 **Momentum Signals:** Replicates the WaveTrend + ADX + Volume Buy Signals.
""")

# ------------------------------------------------------------------
# SIDEBAR & SECRETS
# ------------------------------------------------------------------
st.sidebar.header("Configuration")
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("OpenAI Key: Loaded")
else:
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

use_telegram = False
if "TELEGRAM_TOKEN" in st.secrets and "TELEGRAM_CHAT_ID" in st.secrets:
    tele_token = st.secrets["TELEGRAM_TOKEN"]
    tele_chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    use_telegram = True
    st.sidebar.success("Telegram: Connected")

# ------------------------------------------------------------------
# 1. UNIVERSE (Crypto Mix)
# ------------------------------------------------------------------
# Note: yfinance requires 'BTC-USD' format for crypto
CRYPTO_UNIVERSE = [
    # Majors
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
    # L1 / L2 / Infrastructure
    "ADA-USD", "AVAX-USD", "LINK-USD", "MATIC-USD", "DOT-USD", "NEAR-USD",
    "ATOM-USD", "ARB-USD", "OP-USD", "SUI-USD", "APT-USD",
    # DeFi / Exchange
    "UNI-USD", "AAVE-USD", "MKR-USD", "INJ-USD", "RUNE-USD",
    # High Beta / Meme / AI
    "DOGE-USD", "SHIB-USD", "PEPE-USD", "WIF-USD", "FET-USD", "RNDR-USD"
]

# ------------------------------------------------------------------
# 2. APEX ENGINE (Pine Script Logic in Python)
# ------------------------------------------------------------------
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
        
        # Avoid division by zero
        sum_di = plus_di + minus_di
        sum_di = sum_di.replace(0, 1) 
        
        dx = 100 * np.abs(plus_di - minus_di) / sum_di
        return dx.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def calculate_wavetrend(df):
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=10, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=10, adjust=False).mean()
        # Avoid division by zero
        d = d.replace(0, 0.0001)
        ci = (ap - esa) / (0.015 * d)
        tci = ci.ewm(span=21, adjust=False).mean() 
        return tci

    @staticmethod
    def detect_smc(df):
        """Detects BOS (Break of Structure) and FVGs"""
        # 1. PIVOTS (Lookback 10)
        lookback = 10
        # Simple rolling max/min to identify local structures
        df['Pivot_High'] = df['High'].rolling(window=lookback*2+1, center=True).max()
        df['Pivot_Low'] = df['Low'].rolling(window=lookback*2+1, center=True).min()
        
        # 2. BOS Detection (Price crossing recent confirmed pivot)
        recent_high = df['High'].shift(1).rolling(20).max()
        bos_bull = (df['Close'] > recent_high) & (df['Close'].shift(1) <= recent_high.shift(1))
        
        # 3. FVG (Fair Value Gap) - Bullish
        # Low of candle 0 > High of candle 2
        fvg_bull = (df['Low'] > df['High'].shift(2))
        fvg_size = (df['Low'] - df['High'].shift(2))
        
        return bos_bull, fvg_bull, fvg_size

    @staticmethod
    def run_full_analysis(df):
        if len(df) < 60: return None
        
        # --- 1. APEX TREND (HMA 55) ---
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
        
        # --- 2. SIGNALS (ADX + Vol + Momentum) ---
        df['ADX'] = ApexEngine.calculate_adx(df)
        df['WaveTrend'] = ApexEngine.calculate_wavetrend(df)
        vol_ma = df['Volume'].rolling(20).mean()
        
        # Buy Logic
        buy_signal = (
            (df['Apex_Trend'] == 1) & 
            (df['WaveTrend'] < 60) & 
            (df['WaveTrend'] > df['WaveTrend'].shift(1)) &
            (df['ADX'] > 20) &
            (df['Volume'] > vol_ma)
        )
        
        # --- 3. SMC ---
        bos_bull, fvg_bull, fvg_size = ApexEngine.detect_smc(df)
        df['BOS_Bull'] = bos_bull
        df['FVG_Bull'] = fvg_bull
        df['FVG_Size'] = fvg_size

        # Return latest data point
        last = df.iloc[-1]
        
        # Check for recent signals (last 3 days)
        recent_buy = buy_signal.tail(3).any()
        recent_bos = df['BOS_Bull'].tail(3).any()
        has_fvg = df['FVG_Bull'].iloc[-1]
        
        return {
            "Price": last['Close'],
            "Trend": "Bullish 🟢" if last['Apex_Trend'] == 1 else "Bearish 🔴",
            "WaveTrend": last['WaveTrend'],
            "ADX": last['ADX'],
            "Apex_Buy_Signal": recent_buy,
            "BOS_Alert": recent_bos,
            "FVG_Detected": has_fvg,
            "FVG_Size": last['FVG_Size'] if has_fvg else 0
        }

# ------------------------------------------------------------------
# 3. DATA & SCREENING
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_financials(ticker):
    """
    Fetches basic info. 
    FIX: Do NOT return 'stock_obj' (yf.Ticker) to avoid pickling errors.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker": ticker,
            "name": info.get('shortName', ticker), # shortName is often better for Crypto
            "sector": "Crypto",
            "market_cap": info.get('marketCap', 0)
        }
    except:
        return None

def get_history(ticker_symbol):
    """
    Instantiates Ticker locally to fetch history.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        # Need enough data for HMA 55 + ATR
        # Crypto trades 24/7, so 1y provides plenty of daily candles
        return stock.history(period="1y") 
    except:
        return None

def run_apex_screen(universe):
    progress_bar = st.progress(0)
    status = st.empty()
    results = []
    
    total = len(universe)
    for i, ticker in enumerate(universe):
        status.text(f"Running Apex Algorithm: {ticker}...")
        progress_bar.progress((i+1)/total)
        
        # 1. Get Cached Fundamentals
        data = get_financials(ticker)
        if not data: continue
        
        # 2. Get Fresh History (Non-cached object)
        hist = get_history(ticker)
        if hist is None or len(hist) < 60: continue
        
        # 3. RUN APEX ENGINE
        apex_data = ApexEngine.run_full_analysis(hist)
        if not apex_data: continue
        
        # 4. SCORING
        score = 0
        tags = []
        
        if apex_data['Trend'] == "Bullish 🟢":
            score += 1
            
        if apex_data['Apex_Buy_Signal']:
            score += 3
            tags.append("APEX BUY SIGNAL")
            
        if apex_data['BOS_Alert']:
            score += 2
            tags.append("BOS (Structure Break)")
            
        if apex_data['FVG_Detected']:
            score += 1
            tags.append("FVG Zone")
            
        if score >= 1:
            row = data.copy()
            row.update(apex_data)
            row['Score'] = score
            row['Tags'] = ", ".join(tags)
            results.append(row)
            
    progress_bar.empty()
    status.empty()
    return pd.DataFrame(results)

# ------------------------------------------------------------------
# 4. AI ANALYST (SMC AWARE)
# ------------------------------------------------------------------
def analyze_smc_with_ai(row, api_key):
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    Act as a Smart Money Concepts (SMC) Crypto Trader. Analyze this setup based on the Apex v8 indicator logic.
    
    [ASSET] {row['ticker']} (${row['Price']:.5f})
    [TREND] {row['Trend']} (HMA 55 Baseline)
    
    [SIGNALS]
    Apex Buy Signal: {row['Apex_Buy_Signal']} (WaveTrend + ADX + Vol confirmed)
    BOS (Break of Structure): {row['BOS_Alert']}
    FVG (Fair Value Gap): {row['FVG_Detected']} (Size: {row['FVG_Size']:.5f})
    
    [MOMENTUM]
    WaveTrend TCI: {row['WaveTrend']:.1f}
    ADX Strength: {row['ADX']:.1f}
    
    OUTPUT REQUIREMENTS:
    1. VERDICT: "Strong Long", "Scalp Long", "Wait", or "Short".
    2. SMC STRUCTURE: Explain the BOS/FVG context. (e.g. "Price created a FVG after breaking structure").
    3. EXECUTION: Where is the entry? (e.g. "Enter on FVG retest").
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an SMC master trader for Cryptocurrency."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content
        return content 
    except Exception as e:
        return f"Error: {e}"

# ------------------------------------------------------------------
# 5. TELEGRAM
# ------------------------------------------------------------------
def send_telegram(token, chat_id, text, file_buf, fname):
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat_id, "text": text})
        file_buf.seek(0)
        requests.post(f"https://api.telegram.org/bot{token}/sendDocument",
                      data={"chat_id": chat_id},
                      files={"document": (fname, file_buf, "application/vnd.ms-excel")})
        return True
    except: return False

# ------------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------------
if "apex_df" not in st.session_state: st.session_state.apex_df = None
if "apex_excel" not in st.session_state: st.session_state.apex_excel = None

if st.button("🏛️ Run Apex Crypto Scanner"):
    if not api_key:
        st.error("Need OpenAI API Key")
    else:
        st.subheader("1. Calculating HMA, BOS & Order Flow...")
        df = run_apex_screen(CRYPTO_UNIVERSE)
        
        if df.empty:
            st.warning("No setups found.")
        else:
            df = df.sort_values(by='Score', ascending=False).head(10).reset_index(drop=True)
            
            st.subheader("2. SMC Analyst Review...")
            ai_results = []
            prog = st.progress(0)
            for i, idx in enumerate(df.index):
                res = analyze_smc_with_ai(df.loc[idx], api_key)
                df.loc[idx, 'SMC_Analysis'] = res
                prog.progress((i+1)/len(df))
            prog.empty()
            
            st.session_state.apex_df = df
            
            # Excel
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name="Apex SMC")
            buf.seek(0)
            st.session_state.apex_excel = buf.getvalue()

if st.session_state.apex_df is not None:
    df = st.session_state.apex_df
    
    st.write("### 🏛️ Apex Signal Matrix")
    for i, row in df.iterrows():
        with st.expander(f"{row['ticker']} | {row['Tags']} | {row['Trend']}"):
            c1, c2, c3, c4 = st.columns(4)
            # Crypto often has high precision (e.g. 0.000024)
            c1.metric("Price", f"${row['Price']:.4f}")
            c2.metric("WaveTrend", f"{row['WaveTrend']:.1f}")
            c3.metric("ADX", f"{row['ADX']:.1f}")
            c4.metric("FVG Detected", str(row['FVG_Detected']))
            
            st.info(f"**SMC Analysis:** {row['SMC_Analysis']}")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Download Apex Report", st.session_state.apex_excel, "Apex_Crypto_SMC.xlsx")
    with c2:
        if use_telegram and st.button("📡 Broadcast Apex Signal"):
            top = df.iloc[0]
            msg = f"🏛️ **APEX CRYPTO ALERT**\n\nPair: {top['ticker']}\nTags: {top['Tags']}\nTrend: {top['Trend']}"
            send_telegram(tele_token, tele_chat_id, msg, io.BytesIO(st.session_state.apex_excel), "Apex_Crypto.xlsx")
            st.success("Sent!")
