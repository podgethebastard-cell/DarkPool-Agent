import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import streamlit.components.v1 as components
from scipy.stats import linregress
import requests
import urllib.parse

# ==========================================
# 1. PREMIUM UI CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Titan Terminal: God Mode", page_icon="⚡")

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

# --- SIDEBAR CONTROL ---
st.sidebar.header("⚡ System Control")

# API Keys
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI Key", type="password")

st.sidebar.subheader("📡 Broadcast Keys")
if 'tg_token' not in st.session_state: st.session_state.tg_token = ""
if 'tg_chat' not in st.session_state: st.session_state.tg_chat = ""

if "TELEGRAM_TOKEN" in st.secrets: st.session_state.tg_token = st.secrets["TELEGRAM_TOKEN"]
if "TELEGRAM_CHAT_ID" in st.secrets: st.session_state.tg_chat = st.secrets["TELEGRAM_CHAT_ID"]

tg_token = st.sidebar.text_input("Telegram Bot Token", value=st.session_state.tg_token, type="password")
tg_chat = st.sidebar.text_input("Telegram Chat ID", value=st.session_state.tg_chat)

# --- TOP 100 ASSETS ---
crypto_assets = {
    # 👑 THE KINGS
    "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Solana (SOL)": "SOL-USD",
    "BNB (BNB)": "BNB-USD", "XRP (XRP)": "XRP-USD", "Cardano (ADA)": "ADA-USD",
    "Avalanche (AVAX)": "AVAX-USD", "Dogecoin (DOGE)": "DOGE-USD", "Polkadot (DOT)": "DOT-USD",
    "Chainlink (LINK)": "LINK-USD", "Polygon (MATIC)": "MATIC-USD", "Shiba Inu (SHIB)": "SHIB-USD",
    "Litecoin (LTC)": "LTC-USD", "Bitcoin Cash (BCH)": "BCH-USD", "Near (NEAR)": "NEAR-USD",
    "Internet Computer (ICP)": "ICP-USD", "Toncoin (TON)": "TON11419-USD", "Tron (TRX)": "TRX-USD",
    
    # 🏗️ LAYER 1 & INFRASTRUCTURE
    "Aptos (APT)": "APT21794-USD", "Sui (SUI)": "SUI20947-USD", "Sei (SEI)": "SEI-USD",
    "Celestia (TIA)": "TIA-USD", "Kaspa (KAS)": "KAS-USD", "Injective (INJ)": "INJ-USD",
    "Stellar (XLM)": "XLM-USD", "Cosmos (ATOM)": "ATOM-USD", "Hedera (HBAR)": "HBAR-USD",
    "Monero (XMR)": "XMR-USD", "Ethereum Classic (ETC)": "ETC-USD", "Algorand (ALGO)": "ALGO-USD",
    "Fantom (FTM)": "FTM-USD", "Flow (FLOW)": "FLOW-USD", "Tezos (XTZ)": "XTZ-USD",
    "Neo (NEO)": "NEO-USD", "EOS (EOS)": "EOS-USD", "IOTA (IOTA)": "IOTA-USD",
    "Klaytn (KLAY)": "KLAY-USD", "MultiversX (EGLD)": "EGLD-USD", "Zcash (ZEC)": "ZEC-USD",
    "Mina (MINA)": "MINA-USD", "eCash (XEC)": "XEC-USD", "Conflux (CFX)": "CFX-USD",
    
    # 💻 LAYER 2 SCALING
    "Arbitrum (ARB)": "ARB11841-USD", "Optimism (OP)": "OP-USD", "Immutable (IMX)": "IMX-USD",
    "Mantle (MNT)": "MNT27075-USD", "Stacks (STX)": "STX4847-USD", "Starknet (STRK)": "STRK-USD",
    "Loopring (LRC)": "LRC-USD", "Metis (METIS)": "METIS-USD", "Skale (SKL)": "SKL-USD",
    
    # 🏦 DEFI & EXCHANGE
    "Uniswap (UNI)": "UNI7083-USD", "Maker (MKR)": "MKR-USD", "Aave (AAVE)": "AAVE-USD",
    "Thorchain (RUNE)": "RUNE-USD", "Synthetix (SNX)": "SNX-USD", "Lido DAO (LDO)": "LDO-USD",
    "Curve DAO (CRV)": "CRV-USD", "PancakeSwap (CAKE)": "CAKE-USD", "Compound (COMP)": "COMP-USD",
    "1inch (1INCH)": "1INCH-USD", "dYdX (DYDX)": "DYDX-USD", "Frax Share (FXS)": "FXS-USD",
    "GMX (GMX)": "GMX-USD", "Pendle (PENDLE)": "PENDLE-USD", "Woo (WOO)": "WOO-USD",
    "Osmosis (OSMO)": "OSMO-USD", "Jupiter (JUP)": "JUP-USD", "Pyth (PYTH)": "PYTH-USD",
    
    # 🎮 METAVERSE & GAMING
    "Render (RNDR)": "RNDR-USD", "The Graph (GRT)": "GRT6719-USD", "Axie Infinity (AXS)": "AXS-USD",
    "The Sandbox (SAND)": "SAND-USD", "Decentraland (MANA)": "MANA-USD", "Gala (GALA)": "GALA-USD",
    "Chiliz (CHZ)": "CHZ-USD", "Theta Network (THETA)": "THETA-USD", "Enjin Coin (ENJ)": "ENJ-USD",
    "ApeCoin (APE)": "APE18876-USD", "Beam (BEAM)": "BEAM-USD", "Ronin (RON)": "RON-USD",
    "Blur (BLUR)": "BLUR-USD",
    
    # 🐕 MEME COINS
    "Pepe (PEPE)": "PEPE24478-USD", "Bonk (BONK)": "BONK-USD", "Dogwifhat (WIF)": "WIF-USD",
    "Floki (FLOKI)": "FLOKI-USD", "Memecoin (MEME)": "MEME-USD", "Brett (BRETT)": "BRETT-USD",
    
    # 🧠 AI & STORAGE
    "Fetch.ai (FET)": "FET-USD", "SingularityNET (AGIX)": "AGIX-USD", "Filecoin (FIL)": "FIL-USD",
    "Arweave (AR)": "AR-USD", "Akash Network (AKT)": "AKT-USD", "Ocean Protocol (OCEAN)": "OCEAN-USD",
    "Bittensor (TAO)": "TAO22974-USD", "Worldcoin (WLD)": "WLD-USD", "Helium (HNT)": "HNT-USD"
}

ticker_name = st.sidebar.selectbox("Target Asset", list(crypto_assets.keys()))
ticker = crypto_assets[ticker_name]
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)

# ==========================================
# 2. ADVANCED MATH ENGINE
# ==========================================
def calculate_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    diff = 2 * wma_half - wma_full
    return calculate_wma(diff, sqrt_length)

def calculate_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return calculate_rma(tr, length)

def calculate_linreg_mom(series, length=20):
    x = np.arange(length)
    slope = series.rolling(length).apply(lambda y: linregress(x, y)[0], raw=True)
    return slope

# --- INSTITUTIONAL DATA FETCHING (1D & 1W) ---
@st.cache_data(ttl=600)
def get_institutional_trend(ticker):
    """
    Fetches Daily and Weekly data specifically for the 14th Indicator.
    Returns the latest 50 EMA for Daily and Weekly.
    """
    try:
        # Download 1 Year of Daily Data
        df_d = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
        
        # Download 2 Years of Weekly Data
        df_w = yf.download(ticker, period="2y", interval="1wk", progress=False)
        if isinstance(df_w.columns, pd.MultiIndex): df_w.columns = df_w.columns.get_level_values(0)
        
        # Calculate 50 EMAs
        ema_d = df_d['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        ema_w = df_w['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        
        return ema_d, ema_w
    except:
        return 0, 0

def calculate_engine(df, ticker):
    """Core Calculation Pipeline: 14 Indicators"""
    
    # 1. APEX TREND MASTER
    apex_len = 55
    apex_mult = 1.5
    df['Apex_Base'] = calculate_hma(df['Close'], apex_len)
    df['Apex_ATR'] = calculate_atr(df, apex_len)
    df['Apex_Upper'] = df['Apex_Base'] + (df['Apex_ATR'] * apex_mult)
    df['Apex_Lower'] = df['Apex_Base'] - (df['Apex_ATR'] * apex_mult)
    df['Apex_Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, 
                       np.where(df['Close'] < df['Apex_Lower'], -1, np.nan))
    df['Apex_Trend'] = df['Apex_Trend'].ffill().fillna(0)

    # 2. LIQUIDITY ZONES (Pivots)
    liq_len = 10
    df['Pivot_High'] = df['High'].rolling(liq_len*2+1, center=True).max()
    df['Pivot_Low'] = df['Low'].rolling(liq_len*2+1, center=True).min()
    df['Supply_Zone'] = np.where(df['High'] == df['Pivot_High'], df['High'], np.nan)
    df['Demand_Zone'] = np.where(df['Low'] == df['Pivot_Low'], df['Low'], np.nan)

    # 3. SQUEEZE MOMENTUM
    df['Basis'] = df['Close'].rolling(20).mean()
    df['Dev'] = df['Close'].rolling(20).std() * 2.0
    df['UpperBB'] = df['Basis'] + df['Dev']
    df['LowerBB'] = df['Basis'] - df['Dev']
    df['ATR'] = calculate_atr(df, 20)
    df['UpperKC'] = df['Basis'] + (df['ATR'] * 1.5)
    df['LowerKC'] = df['Basis'] - (df['ATR'] * 1.5)
    df['Squeeze_On'] = (df['LowerBB'] > df['LowerKC']) & (df['UpperBB'] < df['UpperKC'])
    
    mean_hl = (df['High'].rolling(20).max() + df['Low'].rolling(20).min()) / 2
    avg_val = (mean_hl + df['Basis']) / 2
    delta = df['Close'] - avg_val
    df['Sqz_Mom'] = calculate_linreg_mom(delta, 20)

    # 4. GANN HIGH LOW
    gann_len = 3
    df['Gann_High'] = df['High'].rolling(gann_len).mean()
    df['Gann_Low'] = df['Low'].rolling(gann_len).mean()
    df['Gann_Trend'] = np.where(df['Close'] > df['Gann_High'].shift(1), 1, 
                       np.where(df['Close'] < df['Gann_Low'].shift(1), -1, np.nan))
    df['Gann_Trend'] = df['Gann_Trend'].ffill().fillna(0)

    # 5. VOLUME DELTA
    df['Vol_Delta'] = np.where(df['Close'] >= df['Open'], df['Volume'], -df['Volume'])

    # 6. MFI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    pos_flow = np.where(tp > tp.shift(1), rmf, 0)
    neg_flow = np.where(tp < tp.shift(1), rmf, 0)
    mfi_ratio = pd.Series(pos_flow).rolling(14).sum() / pd.Series(neg_flow).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # 7. OPTIMIZED ADX
    up = df['High'].diff()
    down = -df['Low'].diff()
    p_dm = np.where((up > down) & (up > 0), up, 0)
    m_dm = np.where((down > up) & (down > 0), down, 0)
    tr = calculate_atr(df, 14)
    p_di = 100 * calculate_rma(pd.Series(p_dm), 14) / tr
    m_di = 100 * calculate_rma(pd.Series(m_dm), 14) / tr
    dx = (np.abs(p_di - m_di) / (p_di + m_di)) * 100
    df['ADX'] = calculate_rma(dx, 14)

    # 8. ICHIMOKU
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

    # 9. EMA
    df['EMA_200'] = df['Close'].ewm(span=200).mean()

    # 10. DARKPOOL MACD
    e12 = df['Close'].ewm(span=12).mean()
    e26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = e12 - e26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Hist'] = (df['MACD'] - df['Signal']) * (df['Volume'] / df['Volume'].rolling(20).mean())

    # 11. STOCHASTIC RSI
    rsi = 100 - (100 / (1 + (pd.Series(np.where(df['Close'].diff() > 0, df['Close'].diff(), 0)).rolling(14).mean() / pd.Series(np.where(df['Close'].diff() < 0, -df['Close'].diff(), 0)).rolling(14).mean())))
    min_rsi = rsi.rolling(14).min()
    max_rsi = rsi.rolling(14).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    df['Stoch_K'] = stoch_rsi.rolling(3).mean() * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # 12. ATR (Already calc for ADX/Apex)
    
    # 13. DARKPOOL MOVING AVERAGES (5-Layer System)
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

    # 14. INSTITUTIONAL TREND (1D & 1W EMA Cloud)
    # We fetch this once and apply it to the whole column for the AI to read
    ema_d, ema_w = get_institutional_trend(ticker)
    df['Inst_EMA_D'] = ema_d
    df['Inst_EMA_W'] = ema_w
    df['Inst_Trend'] = np.where(ema_d > ema_w, 1, -1) # 1 = Bullish, -1 = Bearish

    return df

def calculate_strategies(df):
    """4 Core Strategy Signals"""
    # 1. Momentum (12)
    mom = df['Close'] - df['Close'].shift(12)
    df['Sig_Mom'] = np.where((mom > 0) & (mom.shift(1) > 0), 1, np.where((mom < 0) & (mom.shift(1) < 0), -1, 0))
    
    # 2. ADX Breakout (Close > 20 High + ADX < 25)
    box_high = df['High'].rolling(20).max().shift(1)
    box_low = df['Low'].rolling(20).min().shift(1)
    df['Sig_ADX'] = np.where((df['Close'] > box_high) & (df['ADX'] < 25), 1,
                    np.where((df['Close'] < box_low) & (df['ADX'] < 25), -1, 0))

    # 3. Bollinger Directed
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['Sig_BB'] = np.where(df['Close'] < (sma20 - 2*std20), 1, np.where(df['Close'] > (sma20 + 2*std20), -1, 0))

    # 4. RSI Strategy (30/70)
    rsi = 100 - (100 / (1 + (pd.Series(np.where(df['Close'].diff() > 0, df['Close'].diff(), 0)).rolling(14).mean() / pd.Series(np.where(df['Close'].diff() < 0, -df['Close'].diff(), 0)).rolling(14).mean())))
    df['Sig_RSI'] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
    
    return df

@st.cache_data(ttl=300)
def get_data(ticker, interval):
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
# 3. DASHBOARD RENDERER
# ==========================================
st.markdown(f'<div class="titan-header">⚡ TITAN TERMINAL: {ticker_name}</div>', unsafe_allow_html=True)

# 1. TRADINGVIEW WIDGET
tv_int_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
tv_sym = f"BINANCE:{ticker.replace('-USD', 'USDT')}"
components.html(
    f"""<div class="tradingview-widget-container"><div id="tradingview_widget"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{"width": "100%","height": 450,"symbol": "{tv_sym}","interval": "{tv_int_map[interval]}","timezone": "Etc/UTC","theme": "dark","style": "1","locale": "en","toolbar_bg": "#f1f3f6","enable_publishing": false,"hide_side_toolbar": false,"allow_symbol_change": true,"container_id": "tradingview_widget"}});</script></div>""",
    height=460,
)

df = get_data(ticker, interval)

if df is not None and not df.empty:
    df = calculate_engine(df, ticker)
    df = calculate_strategies(df)
    last = df.iloc[-1]
    
    # --- PRE-CALCULATE LEVELS FOR BROADCAST & AI ---
    curr_price = last['Close']
    atr_val = last['ATR']
    trend_dir = "LONG" if last['Apex_Trend'] == 1 else "SHORT" if last['Apex_Trend'] == -1 else "NEUTRAL"
    
    if trend_dir == "LONG":
        stop_loss = curr_price - (2 * atr_val)
        take_profit = curr_price + (3 * atr_val)
        entry_price = curr_price
    else:
        stop_loss = curr_price + (2 * atr_val)
        take_profit = curr_price - (3 * atr_val)
        entry_price = curr_price

    # --- 2. SIGNAL HUD ---
    st.markdown("### 🧬 Market DNA")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    def card(label, value, condition):
        color_class = "bull" if condition == 1 else "bear" if condition == -1 else "neu"
        return f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {color_class}">{value}</div></div>"""
    
    # Apex Trend
    apex_cond = last['Apex_Trend']
    apex_txt = "BULLISH" if apex_cond == 1 else "BEARISH" if apex_cond == -1 else "NEUTRAL"
    c1.markdown(card("Apex Trend", apex_txt, apex_cond), unsafe_allow_html=True)
    
    # Momentum (Strat 1)
    mom_cond = last['Sig_Mom']
    mom_txt = "POSITIVE" if mom_cond == 1 else "NEGATIVE"
    c2.markdown(card("Momentum", mom_txt, mom_cond), unsafe_allow_html=True)
    
    # Institutional Trend (Indicator 14)
    inst_cond = last['Inst_Trend']
    inst_txt = "MACRO BULL" if inst_cond == 1 else "MACRO BEAR"
    c3.markdown(card("Inst. Trend (1D/1W)", inst_txt, inst_cond), unsafe_allow_html=True)
    
    # Money Flow
    mfi_val = last['MFI']
    mfi_cond = 1 if mfi_val < 20 else -1 if mfi_val > 80 else 0
    mfi_txt = "OVERSOLD" if mfi_val < 20 else "OVERBOUGHT" if mfi_val > 80 else "NEUTRAL"
    c4.markdown(card("Money Flow", mfi_txt, mfi_cond), unsafe_allow_html=True)
    
    # Bollinger Strat
    bb_cond = last['Sig_BB']
    bb_txt = "DIP BUY" if bb_cond == 1 else "RIP SELL" if bb_cond == -1 else "WAIT"
    c5.markdown(card("Bollinger", bb_txt, bb_cond), unsafe_allow_html=True)

    # --- 3. TABS ---
    st.markdown("<br>", unsafe_allow_html=True)
    tab_apex, tab_dpma, tab_macd, tab_cloud, tab_osc, tab_ai, tab_cast = st.tabs(["🌊 Apex Master", "💀 DarkPool Trends", "📊 Flows & Squeeze", "☁️ Ichimoku", "📈 Oscillators", "🤖 AI Analyst", "📡 Broadcast"])

    with tab_apex:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.15)' if last['Apex_Trend'] == 1 else 'rgba(255, 23, 68, 0.15)', line=dict(width=0), name="Apex Cloud"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Base'], line=dict(color='#00E676' if last['Apex_Trend'] == 1 else '#FF1744', width=2), name="Apex Base"))
        # Gann Trace
        fig.add_trace(go.Scatter(x=df.index, y=df['Gann_High'], line=dict(color='yellow', width=1, dash='dot'), name="Gann Level"))
        fig.update_layout(height=600, template="plotly_dark", title=f"Apex Trend & Liquidity ({interval})", xaxis_rangeslider_visible=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    with tab_dpma:
        # DarkPool MAs (#13)
        fig_dp = go.Figure()
        fig_dp.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA10'], line=dict(color='#00E5FF', width=1), name="EMA 10 (Fast)"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA20'], line=dict(color='#2979FF', width=1), name="EMA 20"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA50'], line=dict(color='#00E676', width=2), name="EMA 50 (Trend)"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA100'], line=dict(color='#FF9100', width=2), name="EMA 100"))
        fig_dp.add_trace(go.Scatter(x=df.index, y=df['DP_MA200'], line=dict(color='#FF1744', width=2), name="EMA 200 (Inst)"))
        fig_dp.update_layout(height=600, template="plotly_dark", title=f"DarkPool Institutional MAs ({interval})", xaxis_rangeslider_visible=False, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig_dp, use_container_width=True)

    with tab_macd:
        fig_macd = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05)
        colors_sqz = ['#00E676' if v > 0 else '#FF1744' for v in df['Sqz_Mom']]
        fig_macd.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=colors_sqz, name="Squeeze Mom"), row=1, col=1)
        fig_macd.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color='cyan', name="MACD Hist"), row=2, col=1)
        vd_col = ['#00E676' if v > 0 else '#FF1744' for v in df['Vol_Delta']]
        fig_macd.add_trace(go.Bar(x=df.index, y=df['Vol_Delta'], marker_color=vd_col, name="Vol Delta"), row=3, col=1)
        fig_macd.update_layout(height=600, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig_macd, use_container_width=True)

    with tab_cloud:
        fig_ichi = go.Figure()
        fig_ichi.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['Tenkan'], line=dict(color='#2962FF', width=1), name="Tenkan"))
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['Kijun'], line=dict(color='#B71C1C', width=1), name="Kijun"))
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['SpanA'], line=dict(width=0), showlegend=False))
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['SpanB'], fill='tonexty', fillcolor='rgba(67, 160, 71, 0.2)', line=dict(width=0), name="Kumo Cloud"))
        fig_ichi.update_layout(height=500, template="plotly_dark", title="Ichimoku Cloud", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_ichi, use_container_width=True)

    with tab_osc:
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line=dict(color='#2962FF', width=2), name="K%"))
        fig_stoch.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], line=dict(color='#FF6D00', width=2), name="D%"))
        fig_stoch.add_hline(y=80, line_color="gray", line_dash="dot")
        fig_stoch.add_hline(y=20, line_color="gray", line_dash="dot")
        fig_stoch.update_layout(height=400, template="plotly_dark", title="Stochastic RSI")
        st.plotly_chart(fig_stoch, use_container_width=True)

    with tab_ai:
        if st.button("Generate Alpha Report"):
            if not api_key:
                st.error("Missing OpenAI API Key")
            else:
                with st.spinner("Calculating Precise Levels..."):
                    client = OpenAI(api_key=api_key)
                    
                    # Pre-Calculate Levels
                    curr_price = last['Close']
                    atr_val = last['ATR']
                    swing_low = last['Pivot_Low'] if not pd.isna(last['Pivot_Low']) else curr_price - (2*atr_val)
                    swing_high = last['Pivot_High'] if not pd.isna(last['Pivot_High']) else curr_price + (2*atr_val)
                    
                    stop_long = curr_price - (2 * atr_val)
                    stop_short = curr_price + (2 * atr_val)
                    target_long = curr_price + (4 * atr_val)
                    target_short = curr_price - (4 * atr_val)
                    
                    cloud_top = last['Apex_Upper']
                    cloud_bot = last['Apex_Lower']

                    prompt = f"""
                    Act as a Senior Crypto Quantitative Trader. 
                    Analyze {ticker} ({interval}) at Price ${curr_price:.2f}.
                    
                    --- TECHNICAL DATA ---
                    1. ATR (Volatility): ${atr_val:.2f}
                    2. Apex Trend: {apex_txt} (Cloud Top: ${cloud_top:.2f}, Bot: ${cloud_bot:.2f})
                    3. DarkPool MAs: {last['DP_Score']:.0f}/5
                    4. Institutional Trend (1D/1W): {inst_txt}
                    5. Liquidity: Supply @ ${swing_high:.2f}, Demand @ ${swing_low:.2f}
                    6. Momentum: {mom_txt}
                    
                    --- STRATEGY SIGNALS ---
                    - ADX Breakout: {'YES' if last['Sig_ADX'] != 0 else 'NO'}
                    - Bollinger: {bb_txt}
                    - RSI: {last['Sig_RSI']}
                    
                    --- CALCULATED LEVELS (Use these if valid) ---
                    - LONG SETUP: Stop < ${stop_long:.2f}, Target > ${target_long:.2f}
                    - SHORT SETUP: Stop > ${stop_short:.2f}, Target < ${target_short:.2f}
                    
                    --- MISSION ---
                    Synthesize a TRADE PLAN in strictly formatted MARKDOWN.
                    Do not print messy raw numbers. Round to 2 decimal places.
                    
                    OUTPUT FORMAT:
                    ### 📋 Trade Plan: Alpha Report
                    
                    **1. VERDICT:** LONG / SHORT / WAIT (Confidence Level)
                    * **Rationale:** (1 sentence confluence summary)
                    
                    **2. ENTRY ZONE**
                    * **Price:** ${entry_price:.2f} (Current Market Price)
                    * **Rationale:** (Technical basis)
                    
                    **3. STOP LOSS**
                    * **Hard Stop:** ${stop_loss:.2f}
                    * **Rationale:** (e.g. Below Apex Cloud)
                    
                    **4. TAKE PROFIT**
                    * **Target:** ${take_profit:.2f}
                    * **Ratio:** 1:1.5 Risk/Reward
                    
                    **5. TRAILING STOP**
                    * **Dynamic Rule:** (e.g. Close below ${cloud_bot:.2f})
                    """
                    res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":prompt}])
                    st.info(res.choices[0].message.content)

    with tab_cast:
        st.subheader("📡 Broadcast Center")
        
        # UPGRADED TELEGRAM MESSAGE FORMAT
        sig_emoji = "🟢" if apex_cond == 1 else "🔴" if apex_cond == -1 else "⚪"
        direction = "LONG" if apex_cond == 1 else "SHORT" if apex_cond == -1 else "NEUTRAL"
        
        broadcast_msg = f"""
🔥 TITAN SIGNAL: {ticker} ({interval})
{sig_emoji} DIRECTION: {direction}

🚪 ENTRY: ${entry_price:,.2f}
🛑 STOP LOSS: ${stop_loss:,.2f}
🎯 TARGET: ${take_profit:,.2f}

Technical Confluence:
🌊 Trend: {apex_txt}
📊 Momentum: {mom_txt}
💰 Money Flow: {mfi_txt}
💀 Institutional Trend: {inst_txt}

#DarkPool #Titan #Crypto
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
                st.warning("⚠️ Enter Telegram Keys in Sidebar")
                
        if c_x.button("Post to X (Twitter)"):
            encoded = urllib.parse.quote(msg_preview)
            st.link_button("🐦 Launch Tweet", f"https://twitter.com/intent/tweet?text={encoded}")

else:
    st.error("Initializing Data Stream...")
