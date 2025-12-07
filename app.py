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
import requests
import urllib.parse

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM UI
# ==========================================
st.set_page_config(layout="wide", page_title="DarkPool Titan Terminal", page_icon="👁️")

# --- CUSTOM CSS FOR "DARKPOOL" AESTHETIC ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }
    .title-glow {
        font-size: 3em;
        font-weight: bold;
        color: #ffffff;
        text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 40px #00ff00;
        margin-bottom: 20px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: scale(1.02);
        border-color: #00ff00;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: 1px solid #30363d;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1117;
        color: #00ff00;
        border-bottom: 2px solid #00ff00;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #30363d !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="title-glow">👁️ DarkPool Titan Terminal</div>', unsafe_allow_html=True)
st.markdown("##### *Institutional-Grade Market Intelligence // v5.0 Titan Edition*")
st.markdown("---")

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
# 2. DATA ENGINE (FUNCTIONS DEFINED FIRST)
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
    """Fetches 40 global macro indicators grouped by sector."""
    groups = {
        "🇺🇸 US Equities": {
            "S&P 500": "SPY", "Nasdaq 100": "QQQ", "Dow Jones": "^DJI", "Russell 2000": "^RUT"
        },
        "🌍 Global Indices": {
            "FTSE 100": "^FTSE", "DAX": "^GDAXI", "Nikkei 225": "^N225", "Hang Seng": "^HSI"
        },
        "🏦 Rates & Bonds": {
            "10Y Yield": "^TNX", "2Y Yield": "^IRX", "30Y Yield": "^TYX", "T-Bond (TLT)": "TLT"
        },
        "💱 Forex & Volatility": {
            "DXY Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "USD/JPY": "JPY=X", "VIX (Fear)": "^VIX"
        },
        "⚠️ Risk Assets": {
            "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Semis (SMH)": "SMH", "Junk Bonds": "HYG"
        },
        "⚡ Energy": {
            "WTI Crude": "CL=F", "Brent Crude": "BZ=F", "Natural Gas": "NG=F", "Uranium": "URA"
        },
        "🥇 Precious Metals": {
            "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F"
        },
        "🏗️ Industrial & Ag": {
            "Copper": "HG=F", "Rare Earths": "REMX", "Corn": "ZC=F", "Wheat": "ZW=F"
        },
        "🇬🇧 UK Desk": {
            "GBP/USD": "GBPUSD=X", "GBP/JPY": "GBPJPY=X", "EUR/GBP": "EURGBP=X", "UK Gilts": "IGLT.L"
        },
        "📈 Growth & Real Assets": {
            "Emerging Mkts": "EEM", "China (FXI)": "FXI", "Real Estate": "VNQ", "Soybeans": "ZS=F"
        }
    }

    all_tickers = {}
    for g in groups.values():
        all_tickers.update(g)
    
    prices = {k: 0.0 for k in all_tickers.keys()}
    changes = {k: 0.0 for k in all_tickers.keys()}
    
    for name, sym in all_tickers.items():
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
            
    return groups, prices, changes

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

def calc_fear_greed_v4(df):
    """
    🔥 DarkPool's Fear & Greed v4 Port
    Calculates composite sentiment index, FOMO, and Panic states.
    """
    # 1. RSI Component (30% Weight)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['FG_RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD Component (25% Weight)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    df['FG_MACD'] = (50 + (hist * 10)).clip(0, 100)
    
    # 3. Bollinger Band Component (25% Weight)
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)
    df['FG_BB'] = ((df['Close'] - lower) / (upper - lower) * 100).clip(0, 100)
    
    # 4. Moving Average Trend (20% Weight)
    sma50 = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200).mean()
    
    conditions = [
        (df['Close'] > sma50) & (sma50 > sma200),
        (df['Close'] > sma50),
        (df['Close'] < sma50) & (sma50 < sma200)
    ]
    choices = [75, 60, 25]
    df['FG_MA'] = np.select(conditions, choices, default=40)
    
    # Composite Index
    df['FG_Raw'] = (df['FG_RSI'] * 0.30) + (df['FG_MACD'] * 0.25) + (df['FG_BB'] * 0.25) + (df['FG_MA'] * 0.20)
    df['FG_Index'] = df['FG_Raw'].rolling(5).mean()
    
    # --- FOMO LOGIC ---
    vol_ma = df['Volume'].rolling(20).mean()
    high_vol = df['Volume'] > (vol_ma * 2.5)
    high_rsi = df['FG_RSI'] > 70
    momentum = df['Close'] > df['Close'].shift(3) * 1.02
    above_bb = df['Close'] > (upper * 1.0)
    
    df['IS_FOMO'] = high_vol & high_rsi & momentum & above_bb
    
    # --- PANIC LOGIC ---
    daily_drop = df['Close'].pct_change() * 100
    sharp_drop = daily_drop < -3.0
    panic_vol = df['Volume'] > (vol_ma * 3.0)
    low_rsi = df['FG_RSI'] < 30
    
    df['IS_PANIC'] = sharp_drop & panic_vol & (low_rsi | (daily_drop < -5.0))
    
    return df

def run_monte_carlo(df, days=30, simulations=1000):
    """🔮 Monte Carlo Simulation."""
    last_price = df['Close'].iloc[-1]
    returns = df['Close'].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    
    daily_returns_sim = np.random.normal(mu, sigma, (days, simulations))
    price_paths = np.zeros((days, simulations))
    price_paths[0] = last_price
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * (1 + daily_returns_sim[t])
        
    return price_paths

def calc_volume_profile(df, bins=50):
    """📊 Institutional Volume Profile (VPVR)."""
    price_min = df['Low'].min()
    price_max = df['High'].max()
    price_bins = np.linspace(price_min, price_max, bins)
    
    df['Mid'] = (df['Close'] + df['Open']) / 2
    df['Bin'] = pd.cut(df['Mid'], bins=price_bins, labels=price_bins[:-1], include_lowest=True)
    
    vp = df.groupby('Bin')['Volume'].sum().reset_index()
    vp['Price'] = vp['Bin'].astype(float)
    poc_idx = vp['Volume'].idxmax()
    poc_price = vp.loc[poc_idx, 'Price']
    
    return vp, poc_price

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
    """🏦 LuxAlgo Smart Money Concepts."""
    smc_data = {'structures': [], 'order_blocks': [], 'fvgs': []}
    
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            smc_data['fvgs'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['High'].iloc[i-2], 'y1': df['Low'].iloc[i], 'color': 'rgba(0, 255, 104, 0.3)'})
        if df['High'].iloc[i] < df['Low'].iloc[i-2]:
            smc_data['fvgs'].append({'x0': df.index[i-2], 'x1': df.index[i], 'y0': df['Low'].iloc[i-2], 'y1': df['High'].iloc[i], 'color': 'rgba(255, 0, 8, 0.3)'})
            
    df['Pivot_High'] = df['High'].rolling(window=swing_length*2+1, center=True).max() == df['High']
    df['Pivot_Low'] = df['Low'].rolling(window=swing_length*2+1, center=True).min() == df['Low']
    
    last_high = None; last_low = None; trend = 0
    
    for i in range(swing_length, len(df)):
        curr_idx = df.index[i]; curr_close = df['Close'].iloc[i]
        
        if df['Pivot_High'].iloc[i-swing_length]:
            last_high = {'price': df['High'].iloc[i-swing_length], 'idx': df.index[i-swing_length], 'i': i-swing_length}
        if df['Pivot_Low'].iloc[i-swing_length]:
            last_low = {'price': df['Low'].iloc[i-swing_length], 'idx': df.index[i-swing_length], 'i': i-swing_length}
            
        if last_high and curr_close > last_high['price']:
            label = "CHoCH" if trend != 1 else "BOS"; trend = 1
            smc_data['structures'].append({'x0': last_high['idx'], 'x1': curr_idx, 'y': last_high['price'], 'color': 'green', 'label': label})
            if last_low:
                subset = df.iloc[last_low['i']:i]
                if not subset.empty:
                    ob_idx = subset['Low'].idxmin(); ob_row = df.loc[ob_idx]
                    smc_data['order_blocks'].append({'x0': ob_idx, 'x1': df.index[-1], 'y0': ob_row['Low'], 'y1': ob_row['High'], 'color': 'rgba(33, 87, 243, 0.4)'})
            last_high = None

        elif last_low and curr_close < last_low['price']:
            label = "CHoCH" if trend != -1 else "BOS"; trend = -1
            smc_data['structures'].append({'x0': last_low['idx'], 'x1': curr_idx, 'y': last_low['price'], 'color': 'red', 'label': label})
            if last_high:
                subset = df.iloc[last_high['i']:i]
                if not subset.empty:
                    ob_idx = subset['High'].idxmax(); ob_row = df.loc[ob_idx]
                    smc_data['order_blocks'].append({'x0': ob_idx, 'x1': df.index[-1], 'y0': ob_row['Low'], 'y1': ob_row['High'], 'color': 'rgba(255, 0, 0, 0.4)'})
            last_low = None

    return smc_data

def calc_correlations(ticker, lookback_days=180):
    """🧩 Cross-Asset Correlation Matrix."""
    macro_tickers = {
        "S&P 500": "SPY", "Bitcoin": "BTC-USD", "10Y Yield": "^TNX", 
        "Dollar (DXY)": "DX-Y.NYB", "Gold": "GC=F", "Oil": "CL=F"
    }
    
    df_main = yf.download(ticker, period="1y", interval="1d", progress=False)['Close']
    df_macro = yf.download(list(macro_tickers.values()), period="1y", interval="1d", progress=False)['Close']
    
    combined = df_macro.copy()
    combined[ticker] = df_main
    corr_matrix = combined.iloc[-lookback_days:].corr()
    target_corr = corr_matrix[ticker].drop(ticker).sort_values(ascending=False)
    
    inv_map = {v: k for k, v in macro_tickers.items()}
    target_corr.index = [inv_map.get(x, x) for x in target_corr.index]
    
    return target_corr

def calc_mtf_trend(ticker):
    """📡 Multi-Timeframe Trend Radar (Fixed for Multi-Index)."""
    timeframes = {"1H": "1h", "4H": "1h", "Daily": "1d", "Weekly": "1wk"}
    trends = {}
    
    for tf_name, tf_code in timeframes.items():
        try:
            period = "1mo" if tf_name == "1H" else "6mo" if tf_name == "4H" else "2y"
            df = yf.download(ticker, period=period, interval=tf_code, progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            if df.empty or len(df) < 50: 
                trends[tf_name] = {"Trend": "N/A", "RSI": "N/A", "EMA Spread": "N/A"}
                continue
            
            if tf_name == "4H":
                df = df.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            
            df['EMA20'] = df['Close'].ewm(span=20).mean()
            df['EMA50'] = df['Close'].ewm(span=50).mean()
            
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            last = df.iloc[-1]
            trend = "BULLISH" if last['Close'] > last['EMA20'] and last['EMA20'] > last['EMA50'] else "BEARISH" if last['Close'] < last['EMA20'] and last['EMA20'] < last['EMA50'] else "NEUTRAL"
            
            trends[tf_name] = {
                "Trend": trend,
                "RSI": f"{last['RSI']:.1f}",
                "EMA Spread": f"{(last['EMA20'] - last['EMA50']):.2f}"
            }
        except:
            trends[tf_name] = {"Trend": "N/A", "RSI": "N/A", "EMA Spread": "N/A"}
            
    return pd.DataFrame(trends).T

def calc_intraday_dna(ticker):
    """⏱️ Intraday Seasonality (Hour of Day)."""
    try:
        df = yf.download(ticker, period="60d", interval="1h", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: return None
        
        df['Return'] = df['Close'].pct_change() * 100
        df['Hour'] = df.index.hour
        
        hourly_stats = df.groupby('Hour')['Return'].agg(['mean', 'sum', 'count', lambda x: (x > 0).mean() * 100])
        hourly_stats.columns = ['Avg Return', 'Total Return', 'Count', 'Win Rate']
        
        return hourly_stats
    except: return None

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
        # --- FIX: Added max_tokens=2500 to prevent AI generation cutoff ---
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}], max_tokens=2500)
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {e}"

# ==========================================
# 5. UI DASHBOARD LAYOUT
# ==========================================
st.sidebar.header("🎛️ Terminal Controls")

# --- BROADCAST CENTER (NEW SIDEBAR) ---
st.sidebar.subheader("📢 Social Broadcaster")

# 1. Initialize Session State
if 'tg_token' not in st.session_state: st.session_state.tg_token = ""
if 'tg_chat' not in st.session_state: st.session_state.tg_chat = ""

# 2. Check Secrets
if "TELEGRAM_TOKEN" in st.secrets: st.session_state.tg_token = st.secrets["TELEGRAM_TOKEN"]
if "TELEGRAM_CHAT_ID" in st.secrets: st.session_state.tg_chat = st.secrets["TELEGRAM_CHAT_ID"]

# 3. Create Inputs (Auto-filled if secrets exist)
tg_token = st.sidebar.text_input("Telegram Bot Token", value=st.session_state.tg_token, type="password")
tg_chat = st.sidebar.text_input("Telegram Chat ID", value=st.session_state.tg_chat)

input_mode = st.sidebar.radio("Input Mode:", ["Curated Lists", "Manual Search (Global)"], index=1)

# --- MODIFIED ASSETS DICTIONARY (EXPANDED) ---
if input_mode == "Curated Lists":
    assets = {
        "Indices": ["SPY", "QQQ", "DIA", "IWM", "VTI"],
        "Crypto (Top 20)": [
            "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", 
            "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "TRX-USD", 
            "LINK-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "BCH-USD", 
            "XLM-USD", "ALGO-USD", "ATOM-USD", "UNI-USD", "FIL-USD"
        ],
        "Tech Giants (Top 10)": [
            "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", 
            "AMZN", "META", "AMD", "NFLX", "INTC"
        ],
        "Macro & Commodities": [
            "^TNX", "DX-Y.NYB", "GC=F", "SI=F", 
            "CL=F", "NG=F", "^VIX", "TLT"
        ]
    }
    cat = st.sidebar.selectbox("Asset Class", list(assets.keys()))
    ticker = st.sidebar.selectbox("Ticker", assets[cat])
else:
    st.sidebar.info("Type any ticker (e.g. SSLN.L, BTC-USD)")
    # --- FIX: Changed default from "SSLN.L" to "BTC-USD" ---
    ticker = st.sidebar.text_input("Search Ticker Symbol", value="BTC-USD").upper()

interval = st.sidebar.selectbox("Interval", ["15m", "1h", "4h", "1d", "1wk"], index=3)
st.sidebar.markdown("---")

balance = st.sidebar.number_input("Capital ($)", 1000, 1000000, 10000)
risk_pct = st.sidebar.slider("Risk %", 0.5, 3.0, 1.0)

# --- GLOBAL MACRO HEADER ---
macro_groups, m_price, m_chg = get_macro_data()

if m_price:
    group_names = list(macro_groups.keys())
    for i in range(0, len(group_names), 2): 
        cols = st.columns(2)
        g1 = group_names[i]
        with cols[0].container(border=True):
            st.markdown(f"#### {g1}")
            sc = st.columns(4) 
            for x, (n, s) in enumerate(macro_groups[g1].items()):
                fmt = "{:.3f}" if any(c in n for c in ["Yield","GBP","EUR","JPY"]) else "{:,.2f}"
                sc[x].metric(n.split('(')[0], fmt.format(m_price.get(n,0)), f"{m_chg.get(n,0):.2f}%")
        
        if i + 1 < len(group_names):
            g2 = group_names[i+1]
            with cols[1].container(border=True):
                st.markdown(f"#### {g2}")
                sc = st.columns(4)
                for x, (n, s) in enumerate(macro_groups[g2].items()):
                    fmt = "{:.3f}" if any(c in n for c in ["Yield","GBP","EUR","JPY"]) else "{:,.2f}"
                    sc[x].metric(n.split('(')[0], fmt.format(m_price.get(n,0)), f"{m_chg.get(n,0):.2f}%")
    st.markdown("---")

# --- MAIN ANALYSIS TABS ---
tab1, tab2, tab3, tab4, tab9, tab5, tab6, tab7, tab8, tab10 = st.tabs([
    "📊 Technical Deep Dive", 
    "🌍 Sector & Fundamentals", 
    "📅 Monthly Seasonality", 
    "📆 Day of Week DNA", 
    "🧩 Correlation & MTF", 
    "📟 DarkPool Dashboard", 
    "🏦 Smart Money Concepts",
    "🔮 Quantitative Forecasting",
    "📊 Volume Profile",
    "📡 Broadcast & TradingView"
])

if st.button(f"Analyze {ticker}", help="Run Analysis"):
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
                    for z in sr_zones:
                        col = "rgba(0, 255, 0, 0.15)" if df['Close'].iloc[-1] > z['max'] else "rgba(255, 0, 0, 0.15)"
                        fig.add_shape(type="rect", x0=df.index[0], x1=df.index[-1], xref="x", yref="y", y0=z['min'], y1=z['max'], fillcolor=col, line=dict(width=0), row=1, col=1)
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                with col_gauge:
                    fg_val = df['FG_Index'].iloc[-1]
                    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=fg_val, title={'text': "Fear & Greed"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "white"}, 'steps': [{'range': [0, 20], 'color': "#FF0000"}, {'range': [80, 100], 'color': "#00FF00"}]}))
                    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    st.metric("RSI", f"{df['FG_RSI'].iloc[-1]:.1f}")
                    st.metric("MACD", f"{df['FG_MACD'].iloc[-1]:.1f}")

                st.markdown("### 🤖 Strategy Briefing")
                ai_verdict = ask_ai_analyst(df, ticker, fund, balance, risk_pct)
                st.info(ai_verdict)

            # --- TAB 2: FUNDAMENTALS ---
            with tab2:
                if fund:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("P/E Ratio", f"{fund.get('P/E Ratio', 'N/A')}")
                    c2.metric("Rev Growth", f"{fund.get('Rev Growth', 0)*100:.1f}%")
                    c3.metric("Debt/Equity", f"{fund.get('Debt/Equity', 'N/A')}")
                    st.write(f"**Summary:** {fund.get('Summary', 'No Data')}")
                st.subheader("🔥 Global Market Heatmap")
                s_data = get_global_performance()
                if s_data is not None:
                    fig_sector = go.Figure(go.Bar(x=s_data.values, y=s_data.index, orientation='h', marker_color=['#00ff00' if v >= 0 else '#ff0000' for v in s_data.values]))
                    fig_sector.update_layout(height=400, template="plotly_dark")
                    st.plotly_chart(fig_sector, use_container_width=True)

            # --- TAB 3: SEASONALITY ---
            with tab3:
                seas = get_seasonality_stats(ticker)
                if seas:
                    hm, hold, month = seas
                    fig_hm = px.imshow(hm, color_continuous_scale='RdYlGn', text_auto='.1f')
                    fig_hm.update_layout(template="plotly_dark", height=500)
                    st.plotly_chart(fig_hm, use_container_width=True)
                    c1, c2 = st.columns(2)
                    c1.dataframe(pd.DataFrame(hold).T.style.format("{:.1f}%").background_gradient(cmap="RdYlGn"))
                    curr_m = datetime.datetime.now().month
                    c2.metric(f"Current Month Win Rate", f"{month.loc[curr_m, 'Win Rate']:.1f}%")

            # --- TAB 4: DNA & HOURLY ---
            with tab4:
                st.subheader("📆 Day & Hour DNA")
                c1, c2 = st.columns(2)
                
                # Day of Week
                dna_res = calc_day_of_week_dna(ticker, 250, "Close to Close (Total)")
                if dna_res:
                    cum, stats = dna_res
                    with c1:
                        st.markdown("**Day of Week Performance**")
                        fig_dna = go.Figure()
                        for c in cum.columns: fig_dna.add_trace(go.Scatter(x=cum.index, y=cum[c], name=c))
                        fig_dna.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig_dna, use_container_width=True)
                        st.dataframe(stats.style.background_gradient(subset=['Win Rate'], cmap="RdYlGn"))
                
                # Hourly DNA
                hourly_res = calc_intraday_dna(ticker)
                if hourly_res is not None:
                    with c2:
                        st.markdown("**Intraday (Hourly) Performance**")
                        fig_hr = px.bar(hourly_res, x=hourly_res.index, y='Avg Return', color='Win Rate', color_continuous_scale='RdYlGn')
                        fig_hr.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig_hr, use_container_width=True)
                        st.dataframe(hourly_res.style.format("{:.2f}"))

            # --- TAB 9: CORRELATION & MTF ---
            with tab9:
                st.subheader("🧩 Cross-Asset Intelligence")
                c1, c2 = st.columns([0.4, 0.6])
                
                with c1:
                    st.markdown("**📡 Multi-Timeframe Radar**")
                    mtf_df = calc_mtf_trend(ticker)
                    
                    def color_trend(val):
                        color = '#00ff00' if val == 'BULLISH' else '#ff0000' if val == 'BEARISH' else 'white'
                        return f'color: {color}; font-weight: bold'
                    
                    st.dataframe(mtf_df.style.map(color_trend, subset=['Trend']), use_container_width=True)
                    
                with c2:
                    st.markdown("**🔗 Macro Correlation Matrix (180 Days)**")
                    corr_data = calc_correlations(ticker)
                    fig_corr = px.bar(x=corr_data.values, y=corr_data.index, orientation='h', color=corr_data.values, color_continuous_scale='RdBu')
                    fig_corr.update_layout(template="plotly_dark", height=400, xaxis_title="Correlation Coefficient")
                    st.plotly_chart(fig_corr, use_container_width=True)

            # --- TAB 5: DASHBOARD ---
            with tab5:
                mom_score = df['Mom_Score'].iloc[-1]
                sig = "BUY" if mom_score > 20 else "SELL" if mom_score < -20 else "HOLD"
                dash_data = {"Metric": ["Momentum", "Signal", "RSI", "Trend"], "Value": [f"{mom_score:.0f}", sig, f"{df['RSI'].iloc[-1]:.1f}", "BULL" if df['Close'].iloc[-1] > df['EMA_50'].iloc[-1] else "BEAR"]}
                st.dataframe(pd.DataFrame(dash_data), use_container_width=True)

            # --- TAB 6: SMC ---
            with tab6:
                smc = calculate_smc(df)
                fig_smc = go.Figure(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                for ob in smc['order_blocks']: fig_smc.add_shape(type="rect", x0=ob['x0'], x1=ob['x1'], y0=ob['y0'], y1=ob['y1'], fillcolor=ob['color'], opacity=0.5, line_width=0)
                for fvg in smc['fvgs']: fig_smc.add_shape(type="rect", x0=fvg['x0'], x1=fvg['x1'], y0=fvg['y0'], y1=fvg['y1'], fillcolor=fvg['color'], opacity=0.5, line_width=0)
                for struct in smc['structures']: 
                    fig_smc.add_shape(type="line", x0=struct['x0'], x1=struct['x1'], y0=struct['y'], y1=struct['y'], line=dict(color=struct['color'], width=1, dash="dot"))
                    fig_smc.add_annotation(x=struct['x1'], y=struct['y'], text=struct['label'], showarrow=False, yshift=10 if struct['color']=='green' else -10, font=dict(color=struct['color'], size=10))

                fig_smc.update_layout(height=600, template="plotly_dark", title="SMC Analysis")
                st.plotly_chart(fig_smc, use_container_width=True)

            # --- TAB 7: QUANT ---
            with tab7:
                mc = run_monte_carlo(df)
                fig_mc = go.Figure()
                for i in range(50): fig_mc.add_trace(go.Scatter(y=mc[:,i], mode='lines', line=dict(color='rgba(255,255,255,0.05)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=np.mean(mc, axis=1), mode='lines', name='Mean', line=dict(color='orange')))
                fig_mc.update_layout(height=500, template="plotly_dark", title="Monte Carlo Forecast (30 Days)")
                st.plotly_chart(fig_mc, use_container_width=True)

            # --- TAB 8: VOLUME PROFILE ---
            with tab8:
                vp, poc = calc_volume_profile(df)
                fig_vp = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.7, 0.3])
                fig_vp.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
                fig_vp.add_trace(go.Bar(x=vp['Volume'], y=vp['Price'], orientation='h', marker_color='rgba(0,200,255,0.3)'), row=1, col=2)
                fig_vp.add_hline(y=poc, line_color="yellow")
                fig_vp.update_layout(height=600, template="plotly_dark", title="Volume Profile (VPVR)")
                st.plotly_chart(fig_vp, use_container_width=True)

            # --- TAB 10: BROADCAST ---
            with tab10:
                st.subheader("📡 Social Command Center")
                
                # TradingView Widget (Embed) - FIX: Added hide_side_toolbar: false
                tv_ticker = ticker.replace("-", "") if "BTC" in ticker else ticker 
                
                tv_widget_html = f"""
                <div class="tradingview-widget-container">
                  <div id="tradingview_widget"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                  <script type="text/javascript">
                  new TradingView.widget(
                  {{
                  "width": "100%",
                  "height": 500,
                  "symbol": "{tv_ticker}",
                  "interval": "D",
                  "timezone": "Etc/UTC",
                  "theme": "dark",
                  "style": "1",
                  "locale": "en",
                  "toolbar_bg": "#f1f3f6",
                  "enable_publishing": false,
                  "hide_side_toolbar": false,
                  "allow_symbol_change": true,
                  "container_id": "tradingview_widget"
                  }}
                  );
                  </script>
                </div>
                """
                st.components.v1.html(tv_widget_html, height=500)
                st.caption("Drawing Tools are enabled on the left sidebar.")
                
                st.markdown("---")
                st.markdown("#### 🚀 Broadcast Signal")
                
                # Signal Message Draft
                signal_text = f"🔥 {ticker} Analysis\n\nPrice: ${df['Close'].iloc[-1]:.2f}\nTrend: {'BULL' if df['Close'].iloc[-1] > df['EMA_50'].iloc[-1] else 'BEAR'}\nRSI: {df['RSI'].iloc[-1]:.1f}\n\n🤖 AI Verdict: {ai_verdict[:50]}...\n\n#Trading #DarkPool #Titan"
                
                msg = st.text_area("Message Preview", value=signal_text, height=150)
                
                # FIX: Added File Uploader for Screenshots
                uploaded_file = st.file_uploader("Upload Chart Screenshot (Optional but Recommended)", type=['png', 'jpg', 'jpeg'])
                
                col_b1, col_b2 = st.columns(2)
                
                # FIX: Telegram Infinite Split Message Logic
                if col_b1.button("Send to Telegram 🚀"):
                    if tg_token and tg_chat:
                        try:
                            # 1. Send Photo if uploaded
                            if uploaded_file:
                                files = {'photo': uploaded_file.getvalue()}
                                url_photo = f"https://api.telegram.org/bot{tg_token}/sendPhoto"
                                data_photo = {'chat_id': tg_chat, 'caption': f"🔥 Analysis: {ticker}", 'parse_mode': 'Markdown'}
                                requests.post(url_photo, data=data_photo, files=files)
                            
                            # 2. Send Full Text (Infinite Loop Splitting to avoid 4096 char limit cutoffs)
                            url_msg = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                            
                            # Clean up AI text to look good
                            clean_msg = msg.replace("###", "")
                            
                            # Chunking Loop - Fixed for Markdown Safety
                            # Reducing limit to 2000 chars to be ultra-safe
                            max_length = 2000 
                            
                            if len(clean_msg) <= max_length:
                                data_msg = {"chat_id": tg_chat, "text": clean_msg} # REMOVED parse_mode to avoid cutoff errors
                                requests.post(url_msg, data=data_msg)
                            else:
                                for i in range(0, len(clean_msg), max_length):
                                    chunk = clean_msg[i:i+max_length]
                                    data_msg = {"chat_id": tg_chat, "text": f"(Part {i//max_length + 1}) {chunk}"} # REMOVED parse_mode
                                    requests.post(url_msg, data=data_msg)

                            st.success("✅ Sent to Telegram (Split into multiple parts to prevent cutoff)!")
                            
                        except Exception as e:
                            st.error(f"Failed: {e}")
                    else:
                        st.warning("⚠️ Enter Telegram Keys in Sidebar.")
                
                if col_b2.button("Post to X (Twitter)"):
                    encoded_msg = urllib.parse.quote(msg)
                    st.link_button("🐦 Launch Tweet", f"https://twitter.com/intent/tweet?text={encoded_msg}")

        else:
            st.error("Data connection failed. Try another ticker.")
