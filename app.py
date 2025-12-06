import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI
import calendar
import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="DarkPool Titan Terminal")
st.title("👁️ DarkPool Titan Terminal")
st.markdown("### Institutional-Grade Market Intelligence")

# --- API Key Management ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if "OPENAI_API_KEY" in st.secrets:
    st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
else:
    if not st.session_state.api_key:
        st.session_state.api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key here to unlock the AI Analyst features."
        )

# ==========================================
# 2. DATA ENGINE (PURE MATH & DATA)
# ==========================================
@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    """Fetches key financial metrics safely."""
    if "-" in ticker or "=" in ticker or "^" in ticker: 
        return None 
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info: return None
        
        return {
            "Market Cap": info.get("marketCap", 0),
            "P/E Ratio": info.get("trailingPE", 0),
            "Rev Growth": info.get("revenueGrowth", 0),
            "Debt/Equity": info.get("debtToEquity", 0),
            "Summary": info.get("longBusinessSummary", "No Data Available")
        }
    except: return None

@st.cache_data(ttl=300)
def get_global_performance():
    """Fetches performance of a Global Multi-Asset Basket."""
    assets = {
        "Tech (XLK)": "XLK", 
        "Energy (XLE)": "XLE", 
        "Financials (XLF)": "XLF", 
        "Bitcoin (BTC)": "BTC-USD", 
        "Gold (GLD)": "GLD",
        "Oil (USO)": "USO",
        "Treasuries (TLT)": "TLT"
    }
    try:
        results = {}
        for name, ticker in assets.items():
            df = yf.download(ticker, period="5d", interval="1d", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    price = df.xs('Close', axis=1, level=0).iloc[-1].iloc[0]
                    prev = df.xs('Close', axis=1, level=0).iloc[-2].iloc[0]
                else:
                    price = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                
                change = ((price - prev) / prev) * 100
                results[name] = change
        
        return pd.Series(results).sort_values(ascending=True)
    except: return None

def safe_download(ticker, period, interval):
    """Robust price downloader."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: return None
        
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
            else: return None
            
        return df
    except: return None

@st.cache_data(ttl=300)
def get_macro_data():
    """Fetches expanded list of 28 global macro indicators."""
    # Defined in display order for logical grouping
    tickers = {
        # --- US INDICES ---
        "S&P 500": "SPY", 
        "Nasdaq 100": "QQQ",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        
        # --- GLOBAL INDICES ---
        "FTSE 100 (UK)": "^FTSE",
        "DAX (Germany)": "^GDAXI",
        "Euro Stoxx 50": "^STOXX50E",
        "Nikkei 225": "^N225",
        
        # --- RATES & VOLATILITY ---
        "10Y Yield": "^TNX", 
        "2Y Yield": "^IRX",
        "Dollar Index": "DX-Y.NYB",
        "VIX (Fear)": "^VIX",
        
        # --- CRYPTO & RISK ---
        "Bitcoin": "BTC-USD", 
        "Ethereum": "ETH-USD",
        "Mining (PICK)": "PICK",
        "Rare Earths": "REMX",

        # --- ENERGY ---
        "WTI Crude": "CL=F",
        "Brent Crude": "BZ=F",
        "Natural Gas": "NG=F",
        "Uranium": "URA",
        
        # --- PRECIOUS METALS ---
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Platinum": "PL=F",
        "Palladium": "PA=F",
        
        # --- INDUSTRIAL & AG ---
        "Copper": "HG=F",
        "Corn": "ZC=F",
        "Wheat": "ZW=F",
        "Soybeans": "ZS=F"
    }
    
    prices = {k: 0.0 for k in tickers.keys()}
    changes = {k: 0.0 for k in tickers.keys()}
    
    for name, sym in tickers.items():
        try:
            df = yf.download(sym, period="5d", interval="1d", progress=False)
            if not df.empty and len(df) >= 2:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                curr = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                chg = ((curr - prev) / prev) * 100
                
                prices[name] = curr
                changes[name] = chg
        except Exception:
            continue
            
    return prices, changes

# ==========================================
# 3. MATH LIBRARY & ALGORITHMS
# ==========================================
def calc_indicators(df):
    """Calculates Base Indicators + Dashboard V2 Logic"""
    df['HMA'] = df['Close'].rolling(55).mean()
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    df['Pivot_Resist'] = df['High'].rolling(20).max()
    df['Pivot_Support'] = df['Low'].rolling(20).min()
    
    df['MFI'] = (df['Close'].diff() * df['Volume']).rolling(14).mean() 
    
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['KC_ATR'] = df['ATR'].rolling(20).mean()
    df['Squeeze_On'] = (df['BB_Mid'] + 2*df['BB_Std']) < (df['BB_Mid'] + 1.5*df['KC_ATR'])
    df['Mom'] = df['Close'] - df['Close'].rolling(20).mean()

    # --- DASHBOARD V2 SPECIFIC CALCULATIONS ---
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']

    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    df['ROC'] = df['Close'].pct_change(14) * 100
    df['EMA_Fast'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / tr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / tr14)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    rsi_norm = (df['RSI'] - 50) * 2
    macd_norm = np.where(df['Hist'] > 0, np.minimum(df['Hist'] * 10, 100), np.maximum(df['Hist'] * 10, -100))
    stoch_norm = (df['Stoch_K'] - 50) * 2
    roc_norm = np.where(df['ROC'] > 0, np.minimum(df['ROC'] * 10, 100), np.maximum(df['ROC'] * 10, -100))
    
    df['Mom_Score'] = np.round((rsi_norm + macd_norm + stoch_norm + roc_norm) / 4)

    return df

def get_sr_channels(df, pivot_period=10, loopback=290, max_width_pct=5, min_strength=1):
    """Python implementation of 'Support Resistance Channels' logic."""
    if len(df) < loopback: loopback = len(df)
    window = df.iloc[-loopback:].copy()
    
    window['Is_Pivot_H'] = window['High'] == window['High'].rolling(pivot_period*2+1, center=True).max()
    window['Is_Pivot_L'] = window['Low'] == window['Low'].rolling(pivot_period*2+1, center=True).min()
    
    pivot_vals = []
    pivot_vals.extend(window[window['Is_Pivot_H']]['High'].tolist())
    pivot_vals.extend(window[window['Is_Pivot_L']]['Low'].tolist())
    
    if not pivot_vals: return []
    pivot_vals.sort()
    
    price_range = window['High'].max() - window['Low'].min()
    max_width = price_range * (max_width_pct / 100)
    
    potential_zones = []
    for i in range(len(pivot_vals)):
        seed = pivot_vals[i]
        cluster_min = seed
        cluster_max = seed
        pivot_count = 1
        
        for j in range(i + 1, len(pivot_vals)):
            curr = pivot_vals[j]
            if (curr - seed) <= max_width:
                cluster_max = curr
                pivot_count += 1
            else:
                break
        
        touches = ((window['High'] >= cluster_min) & (window['Low'] <= cluster_max)).sum()
        score = (pivot_count * 20) + touches
        
        potential_zones.append({'min': cluster_min, 'max': cluster_max, 'score': score})
        
    potential_zones.sort(key=lambda x: x['score'], reverse=True)
    
    final_zones = []
    for zone in potential_zones:
        is_overlapping = False
        for existing in final_zones:
            if (zone['min'] < existing['max']) and (zone['max'] > existing['min']):
                is_overlapping = True
                break
        if not is_overlapping:
            final_zones.append(zone)
            if len(final_zones) >= 6: break
                
    return final_zones

def calculate_smc(df, swing_length=5):
    """
    🏦 LuxAlgo Smart Money Concepts (Python Port)
    Calculates Structure (BOS/CHoCH), Order Blocks (OB), and Fair Value Gaps (FVG).
    """
    smc_data = {
        'structures': [], 
        'order_blocks': [], 
        'fvgs': [] 
    }
    
    # 1. FAIR VALUE GAPS (FVG)
    for i in range(2, len(df)):
        # Bullish FVG
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            smc_data['fvgs'].append({
                'x0': df.index[i-2], 'x1': df.index[i],
                'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i],
                'color': 'rgba(0, 255, 104, 0.3)' # Green
            })
        # Bearish FVG
        if df['High'].iloc[i] < df['Low'].iloc[i-2]:
            smc_data['fvgs'].append({
                'x0': df.index[i-2], 'x1': df.index[i],
                'y0': df['Low'].iloc[i-2], 'y1': df['High'].iloc[i],
                'color': 'rgba(255, 0, 8, 0.3)' # Red
            })
            
    # 2. MARKET STRUCTURE & ORDER BLOCKS
    df['Pivot_High'] = df['High'].rolling(window=swing_length*2+1, center=True).max() == df['High']
    df['Pivot_Low'] = df['Low'].rolling(window=swing_length*2+1, center=True).min() == df['Low']
    
    last_high = None
    last_low = None
    trend = 0 # 1=Bull, -1=Bear
    
    for i in range(swing_length, len(df)):
        curr_idx = df.index[i]
        curr_close = df['Close'].iloc[i]
        
        if df['Pivot_High'].iloc[i-swing_length]:
            last_high = {'price': df['High'].iloc[i-swing_length], 'idx': df.index[i-swing_length], 'i': i-swing_length}
        if df['Pivot_Low'].iloc[i-swing_length]:
            last_low = {'price': df['Low'].iloc[i-swing_length], 'idx': df.index[i-swing_length], 'i': i-swing_length}
            
        if last_high and curr_close > last_high['price']:
            if trend != 1:
                label = "CHoCH"
                trend = 1
            else:
                label = "BOS"
            
            smc_data['structures'].append({
                'x0': last_high['idx'], 'x1': curr_idx,
                'y': last_high['price'], 'color': 'green', 'label': label
            })
            
            if last_low:
                subset = df.iloc[last_low['i']:i]
                if not subset.empty:
                    ob_idx = subset['Low'].idxmin()
                    ob_row = df.loc[ob_idx]
                    smc_data['order_blocks'].append({
                        'x0': ob_idx, 'x1': df.index[-1],
                        'y0': ob_row['Low'], 'y1': ob_row['High'],
                        'color': 'rgba(33, 87, 243, 0.4)' # Blue
                    })
            last_high = None

        elif last_low and curr_close < last_low['price']:
            if trend != -1:
                label = "CHoCH"
                trend = -1
            else:
                label = "BOS"
                
            smc_data['structures'].append({
                'x0': last_low['idx'], 'x1': curr_idx,
                'y': last_low['price'], 'color': 'red', 'label': label
            })
            
            if last_high:
                subset = df.iloc[last_high['i']:i]
                if not subset.empty:
                    ob_idx = subset['High'].idxmax()
                    ob_row = df.loc[ob_idx]
                    smc_data['order_blocks'].append({
                        'x0': ob_idx, 'x1': df.index[-1],
                        'y0': ob_row['Low'], 'y1': ob_row['High'],
                        'color': 'rgba(255, 0, 0, 0.4)' # Red
                    })
            last_low = None

    return smc_data

def calc_fear_greed_v4(df):
    """DarkPool's Fear & Greed v4 Port"""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['FG_RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    df['FG_MACD'] = (50 + (hist * 10)).clip(0, 100)
    
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)
    df['FG_BB'] = ((df['Close'] - lower) / (upper - lower) * 100).clip(0, 100)
    
    sma50 = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200).mean()
    
    conditions = [
        (df['Close'] > sma50) & (sma50 > sma200),
        (df['Close'] > sma50),
        (df['Close'] < sma50) & (sma50 < sma200)
    ]
    choices = [75, 60, 25]
    df['FG_MA'] = np.select(conditions, choices, default=40)
    
    df['FG_Raw'] = (df['FG_RSI'] * 0.30) + (df['FG_MACD'] * 0.25) + (df['FG_BB'] * 0.25) + (df['FG_MA'] * 0.20)
    df['FG_Index'] = df['FG_Raw'].rolling(5).mean()
    
    vol_ma = df['Volume'].rolling(20).mean()
    high_vol = df['Volume'] > (vol_ma * 2.5)
    high_rsi = df['FG_RSI'] > 70
    momentum = df['Close'] > df['Close'].shift(3) * 1.02
    above_bb = df['Close'] > (upper * 1.0)
    
    df['IS_FOMO'] = high_vol & high_rsi & momentum & above_bb
    
    daily_drop = df['Close'].pct_change() * 100
    sharp_drop = daily_drop < -3.0
    panic_vol = df['Volume'] > (vol_ma * 3.0)
    low_rsi = df['FG_RSI'] < 30
    
    df['IS_PANIC'] = sharp_drop & panic_vol & (low_rsi | (daily_drop < -5.0))
    
    return df

@st.cache_data(ttl=3600)
def get_seasonality_stats(ticker):
    """Calculates Monthly Seasonality and Probability Stats."""
    try:
        df = yf.download(ticker, period="20y", interval="1mo", progress=False)
        if df.empty or len(df) < 12: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if 'Close' not in df.columns:
             if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
             else: return None

        df = df.dropna()
        df['Return'] = df['Close'].pct_change() * 100
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        
        heatmap_data = df.pivot_table(index='Year', columns='Month', values='Return')
        
        periods = [1, 3, 6, 12]
        hold_stats = {}
        for p in periods:
            rolling_ret = df['Close'].pct_change(periods=p) * 100
            rolling_ret = rolling_ret.dropna()
            
            win_count = (rolling_ret > 0).sum()
            total_count = len(rolling_ret)
            win_rate = (win_count / total_count * 100) if total_count > 0 else 0
            avg_ret = rolling_ret.mean()
            
            hold_stats[p] = {"Win Rate": win_rate, "Avg Return": avg_ret}
            
        month_stats = df.groupby('Month')['Return'].agg(['mean', lambda x: (x > 0).mean() * 100, 'count'])
        month_stats.columns = ['Avg Return', 'Win Rate', 'Count']
        
        return heatmap_data, hold_stats, month_stats
        
    except Exception as e:
        return None

def calc_day_of_week_dna(ticker, lookback, calc_mode):
    """DarkPool's Day of Week Seasonality DNA Port"""
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if df.empty: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.iloc[-lookback:].copy()
        
        if calc_mode == "Close to Close (Total)":
            df['Day_Return'] = df['Close'].pct_change() * 100
        else: # Open to Close (Intraday)
            df['Day_Return'] = ((df['Close'] - df['Open']) / df['Open']) * 100
            
        df = df.dropna()
        df['Day_Name'] = df.index.day_name()
        
        pivot_ret = df.pivot(columns='Day_Name', values='Day_Return').fillna(0)
        cum_ret = pivot_ret.cumsum()
        
        stats = df.groupby('Day_Name')['Day_Return'].agg(['count', 'sum', 'mean', lambda x: (x > 0).mean() * 100])
        stats.columns = ['Count', 'Total Return', 'Avg Return', 'Win Rate']
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        stats = stats.reindex([d for d in days_order if d in stats.index])
        
        return cum_ret, stats
        
    except Exception as e:
        return None

# ==========================================
# 4. AI ANALYST
# ==========================================
def ask_ai_analyst(df, ticker, fundamentals, balance, risk_pct):
    if not st.session_state.api_key: 
        return "⚠️ Waiting for OpenAI API Key in the sidebar..."
    
    last = df.iloc[-1]
    trend = "BULLISH" if last['Close'] > last['HMA'] else "BEARISH"
    risk_dollars = balance * (risk_pct / 100)
    
    if trend == "BULLISH":
        stop_level = last['Pivot_Support']
        direction = "LONG"
    else:
        stop_level = last['Pivot_Resist']
        direction = "SHORT"
        
    if pd.isna(stop_level) or abs(last['Close'] - stop_level) < (last['ATR']*0.5):
        stop_level = last['Close'] - (last['ATR']*2) if direction == "LONG" else last['Close'] + (last['ATR']*2)
        
    dist = abs(last['Close'] - stop_level)
    if dist == 0: dist = last['ATR']
    shares = risk_dollars / dist 
    
    fund_text = "N/A"
    if fundamentals:
        fund_text = f"P/E: {fundamentals.get('P/E Ratio', 'N/A')}. Growth: {fundamentals.get('Rev Growth', 0)*100:.1f}%."
    
    fg_val = last['FG_Index']
    fg_state = "EXTREME GREED" if fg_val >= 80 else "GREED" if fg_val >= 60 else "NEUTRAL" if fg_val >= 40 else "FEAR" if fg_val >= 20 else "EXTREME FEAR"
    psych_alert = ""
    if last['IS_FOMO']: psych_alert = "WARNING: ALGORITHMIC FOMO DETECTED."
    if last['IS_PANIC']: psych_alert = "WARNING: PANIC SELLING DETECTED."

    prompt = f"""
    Act as a Global Macro Strategist. Analyze {ticker} at ${last['Close']:.2f}.
    --- FUNDAMENTALS ---
    {fund_text}
    --- TECHNICALS ---
    Trend: {trend}. Volatility (ATR): {last['ATR']:.2f}.
    --- PSYCHOLOGY (DarkPool Index) ---
    Sentiment Score: {fg_val:.1f}/100 ({fg_state}).
    {psych_alert}
    --- RISK PROTOCOL (1% Rule) ---
    Capital: ${balance}. Risk Budget: ${risk_dollars:.2f} ({risk_pct}%).
    Stop Loss: ${stop_level:.2f}. Position Size: {shares:.4f} units.
    --- MISSION ---
    1. Verdict: BUY, SELL, or WAIT.
    2. Reasoning: Integrate Technicals, Fundamentals, and Market Psychology.
    3. Trade Plan: Entry, Stop, Target (2.5R), Size.
    """
    
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# ==========================================
# 5. UI DASHBOARD LAYOUT
# ==========================================
st.sidebar.header("🎛️ Terminal Controls")

# --- INPUT MODE SELECTION ---
input_mode = st.sidebar.radio(
    "Input Mode:", 
    ["Curated Lists", "Manual Search (Global)"],
    index=1,
    help="Choose 'Curated Lists' to select from preset menus, or 'Manual Search' to type any ticker symbol yourself."
)

if input_mode == "Curated Lists":
    assets = {
        "Indices": ["SPY", "QQQ", "IWM", "^VIX"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
        "Tech": ["NVDA", "TSLA", "AAPL", "MSFT"],
        "Macro": ["^TNX", "DX-Y.NYB", "TLT"],
        "LSE/Commodities": ["SSLN.L", "SGLN.L", "SHEL.L", "BP.L"] 
    }
    cat = st.sidebar.selectbox(
        "Asset Class", 
        list(assets.keys()),
        help="Select a category of assets to filter the ticker list."
    )
    ticker = st.sidebar.selectbox(
        "Ticker", 
        assets[cat],
        help="Choose a specific asset from the selected category."
    )
else:
    # --- SEARCH BOX FEATURE ---
    st.sidebar.info("Type any ticker (e.g. SSLN.L, BTC-USD)")
    ticker = st.sidebar.text_input(
        "Search Ticker Symbol", 
        value="SSLN.L",
        help="Type any valid Yahoo Finance ticker here. Works for Stocks, Crypto, Indices, and Forex."
    ).upper()

interval = st.sidebar.selectbox(
    "Interval", 
    ["15m", "1h", "4h", "1d", "1wk"], 
    index=3,
    help="Select the timeframe for the chart bars (e.g., 1 Day, 1 Hour)."
)
st.sidebar.markdown("---")

balance = st.sidebar.number_input(
    "Capital ($)", 
    1000, 1000000, 10000,
    help="Enter your total trading capital for position sizing calculations."
)
risk_pct = st.sidebar.slider(
    "Risk %", 
    0.5, 3.0, 1.0,
    help="Adjust the percentage of capital you are willing to risk on this trade."
)

# --- GLOBAL MACRO HEADER (DYNAMIC GRID) ---
m_price, m_chg = get_macro_data()
if m_price:
    # Convert dictionary to lists for iteration
    items = list(m_price.keys())
    
    # Create rows of 4 columns
    for i in range(0, len(items), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(items):
                name = items[i+j]
                val = m_price[name]
                pct = m_chg[name]
                
                # Format logic
                if "Yield" in name or "Index" in name or "VIX" in name:
                    fmt_val = f"{val:.2f}"
                else:
                    fmt_val = f"{val:,.2f}"
                    
                cols[j].metric(name, fmt_val, f"{pct:.2f}%")
                
    st.markdown("---")

# --- MAIN ANALYSIS TABS ---
# UPGRADE: Added "Smart Money Concepts" Tab
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Technical Deep Dive", "🌍 Sector & Fundamentals", "📅 Monthly Seasonality", "📆 Day of Week DNA", "📟 DarkPool Dashboard", "🏦 Smart Money Concepts"])

if st.button(f"Analyze {ticker}", help="Click to run the data pipeline and AI analysis for the selected ticker."):
    st.session_state['run_analysis'] = True

if st.session_state.get('run_analysis'):
    with st.spinner(f"Analyzing {ticker}..."):
        df = safe_download(ticker, "2y", interval)
        
        if df is not None:
            df = calc_indicators(df)
            df = calc_fear_greed_v4(df)
            fund = get_fundamentals(ticker)
            sr_zones = get_sr_channels(df) 
            
            # --- TAB 1: TECHNICALS ---
            with tab1:
                st.subheader(f"🎯 Sniper Scope: {ticker}")
                col_chart, col_gauge = st.columns([0.75, 0.25])
                
                with col_chart:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], line=dict(color='orange', width=2), name="Apex Trend"), row=1, col=1)
                    
                    last_close = df['Close'].iloc[-1]
                    for zone in sr_zones:
                        if last_close > zone['max']: col = "rgba(0, 255, 0, 0.15)"
                        elif last_close < zone['min']: col = "rgba(255, 0, 0, 0.15)"
                        else: col = "rgba(128, 128, 128, 0.15)"
                        fig.add_shape(type="rect", x0=df.index[0], x1=df.index[-1], xref="x", yref="y", y0=zone['min'], y1=zone['max'], fillcolor=col, line=dict(width=0), row=1, col=1)
                    
                    fomo_dates = df[df['IS_FOMO']].index
                    panic_dates = df[df['IS_PANIC']].index
                    fig.add_trace(go.Scatter(x=fomo_dates, y=df.loc[fomo_dates, 'High']*1.02, mode='markers', marker=dict(symbol='triangle-up', size=10, color='purple'), name="FOMO Algo"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=panic_dates, y=df.loc[panic_dates, 'Low']*0.98, mode='markers', marker=dict(symbol='triangle-down', size=10, color='yellow'), name="Panic Algo"), row=1, col=1)

                    colors = ['#00ff00' if v > 0 else '#ff0000' for v in df['MFI']]
                    fig.add_trace(go.Bar(x=df.index, y=df['MFI'], marker_color=colors, name="Smart Money"), row=2, col=1)
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_gauge:
                    curr_fg = df['FG_Index'].iloc[-1]
                    if np.isnan(curr_fg): curr_fg = 50
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = curr_fg,
                        title = {'text': "Fear & Greed Index"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "white", 'thickness': 0.2},
                            'steps': [{'range': [0, 20], 'color': "#FF0000"}, {'range': [20, 40], 'color': "#FFA500"}, {'range': [40, 60], 'color': "#808080"}, {'range': [60, 80], 'color': "#90EE90"}, {'range': [80, 100], 'color': "#00FF00"}]
                        }
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    st.markdown("#### Psychology Stats")
                    st.metric("RSI Contribution", f"{df['FG_RSI'].iloc[-1]:.1f}/100")
                    st.metric("MACD Momentum", f"{df['FG_MACD'].iloc[-1]:.1f}/100")
                    if df['IS_FOMO'].iloc[-1]: st.error("🚀 FOMO DETECTED")
                    if df['IS_PANIC'].iloc[-1]: st.warning("💥 PANIC DETECTED")

                st.markdown("### 🤖 Strategy Briefing")
                verdict = ask_ai_analyst(df, ticker, fund, balance, risk_pct)
                st.info(verdict)

            # --- TAB 2: SECTOR & FUNDAMENTALS ---
            with tab2:
                st.subheader(f"🏢 Fundamental Health")
                if fund:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("P/E Ratio", f"{fund.get('P/E Ratio', 'N/A')}")
                    c2.metric("Rev Growth", f"{fund.get('Rev Growth', 0)*100:.1f}%")
                    c3.metric("Debt/Equity", f"{fund.get('Debt/Equity', 'N/A')}")
                    st.write(f"**Summary:** {fund.get('Summary', 'No Data')[:300]}...")
                else:
                    st.warning("Fundamentals not available for this asset.")
                st.markdown("---")
                st.subheader("🔥 Global Market Heatmap")
                s_data = get_global_performance()
                if s_data is not None:
                    fig_sector = go.Figure()
                    colors = ['#00ff00' if v >= 0 else '#ff0000' for v in s_data.values]
                    fig_sector.add_trace(go.Bar(x=s_data.values, y=s_data.index, orientation='h', marker_color=colors, text=[f"{v:.2f}%" for v in s_data.values], textposition='auto'))
                    fig_sector.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), xaxis_title="5-Day Performance (%)")
                    st.plotly_chart(fig_sector, use_container_width=True)

            # --- TAB 3: MONTHLY SEASONALITY ---
            with tab3:
                st.subheader(f"📅 Monthly Seasonality: {ticker}")
                seas_res = get_seasonality_stats(ticker)
                if seas_res:
                    hm_data, hold_stats, month_stats = seas_res
                    hm_data = hm_data.reindex(columns=range(1, 13))
                    fig_hm = px.imshow(hm_data, labels=dict(x="Month", y="Year", color="Return %"), x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], color_continuous_scale='RdYlGn', text_auto='.1f')
                    fig_hm.update_layout(template="plotly_dark", height=500, xaxis_side="top")
                    st.plotly_chart(fig_hm, use_container_width=True)
                    st.markdown("---")
                    c_prob1, c_prob2 = st.columns(2)
                    with c_prob1:
                        st.markdown("#### 🎲 Holding Period Probabilities")
                        hold_df = pd.DataFrame(hold_stats).T
                        hold_df.columns = ["Win Rate %", "Avg Return %"]
                        st.dataframe(hold_df.style.format("{:.1f}%").background_gradient(subset=["Win Rate %"], cmap="RdYlGn", vmin=30, vmax=70))
                    with c_prob2:
                        st.markdown("#### 🔮 Forecast (Historical Odds)")
                        import datetime
                        curr_m = datetime.datetime.now().month
                        next_m = (curr_m % 12) + 1
                        m_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
                        try:
                            curr_data = month_stats.loc[curr_m]
                            next_data = month_stats.loc[next_m]
                            col1, col2 = st.columns(2)
                            col1.metric(f"Current ({m_names[curr_m]})", f"{curr_data['Win Rate']:.1f}% Win", f"{curr_data['Avg Return']:.2f}% Avg")
                            col2.metric(f"Next ({m_names[next_m]})", f"{next_data['Win Rate']:.1f}% Win", f"{next_data['Avg Return']:.2f}% Avg")
                        except: st.warning("Insufficient history for monthly forecast.")
                else: st.warning("Insufficient data to calculate seasonality stats.")

            # --- TAB 4: DAY OF WEEK DNA ---
            with tab4:
                st.subheader("📆 Day of Week DNA Analysis")
                c_opts1, c_opts2 = st.columns(2)
                with c_opts1:
                    dna_lookback = st.slider("DNA Lookback (Days)", 50, 2000, 250)
                with c_opts2:
                    dna_mode = st.selectbox("Calculation Mode", ["Close to Close (Total)", "Open to Close (Intraday)"])
                
                dna_res = calc_day_of_week_dna(ticker, dna_lookback, dna_mode)
                
                if dna_res:
                    cum_dna, stats_dna = dna_res
                    st.markdown(f"**Cumulative Performance by Weekday (Last {dna_lookback} Days)**")
                    fig_dna = go.Figure()
                    day_colors = {"Monday": "red", "Tuesday": "orange", "Wednesday": "yellow", "Thursday": "cyan", "Friday": "lime", "Saturday": "magenta", "Sunday": "gray"}
                    for col in cum_dna.columns:
                        if col in day_colors:
                            fig_dna.add_trace(go.Scatter(x=cum_dna.index, y=cum_dna[col], mode='lines', name=col, line=dict(color=day_colors[col], width=2)))
                    fig_dna.add_hline(y=0, line_dash="dot", line_color="gray")
                    fig_dna.update_layout(template="plotly_dark", height=500, xaxis_title="Date", yaxis_title="Cumulative Return (%)")
                    st.plotly_chart(fig_dna, use_container_width=True)
                    st.markdown("**Weekday Statistics**")
                    st.dataframe(stats_dna.style.format({"Total Return": "{:.1f}%", "Avg Return": "{:.2f}%", "Win Rate": "{:.1f}%", "Count": "{:.0f}"}).background_gradient(subset=["Win Rate"], cmap="RdYlGn", vmin=40, vmax=60), use_container_width=True)
                else:
                    st.warning("Insufficient daily data for DNA analysis.")

            # --- TAB 5: DARKPOOL DASHBOARD V2 ---
            with tab5:
                st.subheader("📟 DarkPool Dashboard v2")
                last = df.iloc[-1]
                mom_score = last['Mom_Score']
                signal = "STRONG BUY" if mom_score > 50 else "BUY" if mom_score > 20 else "SELL" if mom_score < -50 else "STRONG SELL" if mom_score < -20 else "HOLD"
                
                dash_data = {
                    "Metric": ["Momentum Score", "Signal", "RSI (14)", "Money Flow (MFI)", "Trend (EMA)", "Volume vs MA", "Volatility (Range)"],
                    "Value": [f"{mom_score:.0f}", signal, f"{last['RSI']:.2f}", f"{last['MFI']:.2f}", "BULLISH" if last['EMA_Fast'] > last['EMA_Slow'] else "BEARISH", f"{(last['Volume'] / last['Volume'].mean() * 100) - 100:.1f}%", f"{((last['High']-last['Low'])/last['Low']*100):.2f}%"]
                }
                st.dataframe(pd.DataFrame(dash_data), height=300, use_container_width=True)
                st.markdown("#### 🔬 Detailed Indicators")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ADX Strength", f"{last['ADX']:.2f}")
                c2.metric("Stoch K/D", f"{last['Stoch_K']:.0f} / {last['Stoch_D']:.0f}")
                c3.metric("MACD", f"{last['MACD']:.3f}")
                c4.metric("OBV Trend", "UP" if last['OBV'] > df['OBV'].iloc[-2] else "DOWN")
                delta_color = 'green' if last['Close'] > last['Open'] else 'red'
                st.markdown(f"**Volume Delta:** <span style='color:{delta_color}'>{'BUY PRESSURE' if delta_color=='green' else 'SELL PRESSURE'}</span>", unsafe_allow_html=True)

            # --- TAB 6: SMART MONEY CONCEPTS ---
            with tab6:
                st.subheader("🏦 LuxAlgo Smart Money Concepts (SMC)")
                
                # Input controls for SMC
                c_smc1, c_smc2 = st.columns(2)
                with c_smc1:
                    smc_swing_len = st.slider("SMC Swing Length", 3, 20, 5)
                
                # Calculate SMC
                smc_res = calculate_smc(df, smc_swing_len)
                
                fig_smc = go.Figure()
                
                # 1. Candlestick Chart
                fig_smc.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
                
                # 2. Draw Order Blocks (Rectangles)
                for ob in smc_res['order_blocks']:
                    fig_smc.add_shape(type="rect",
                        x0=ob['x0'], x1=ob['x1'], y0=ob['y0'], y1=ob['y1'],
                        fillcolor=ob['color'], line=dict(width=0), opacity=0.5
                    )
                    
                # 3. Draw Fair Value Gaps (Rectangles)
                for fvg in smc_res['fvgs']:
                     fig_smc.add_shape(type="rect",
                        x0=fvg['x0'], x1=fvg['x1'], y0=fvg['y0'], y1=fvg['y1'],
                        fillcolor=fvg['color'], line=dict(width=0), opacity=0.5
                    )
                
                # 4. Draw Structure Breaks (Lines & Annotations)
                for struct in smc_res['structures']:
                    # Dotted Line
                    fig_smc.add_shape(type="line",
                        x0=struct['x0'], x1=struct['x1'], y0=struct['y'], y1=struct['y'],
                        line=dict(color=struct['color'], width=1, dash="dot")
                    )
                    # Label
                    fig_smc.add_annotation(
                        x=struct['x1'], y=struct['y'], text=struct['label'],
                        showarrow=False, yshift=10 if struct['color']=='green' else -10,
                        font=dict(color=struct['color'], size=10)
                    )

                fig_smc.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False, title=f"SMC Analysis: {ticker}")
                st.plotly_chart(fig_smc, use_container_width=True)
                
                st.markdown("""
                **Legend:**
                * **BOS (Break of Structure):** Trend continuation signal.
                * **CHoCH (Change of Character):** Potential trend reversal signal.
                * **Blue/Red Boxes:** Order Blocks (Institutional Supply/Demand).
                * **Green/Red Rectangles:** Fair Value Gaps (Imbalances).
                """)

        else:
            st.error("Data connection failed. Try another ticker.")
