import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DarkPool Titan Terminal", layout="wide")

# ==============================================================================
# 1. TITAN ANALYTICS ENGINE (Mathematical Translation of Pine Script)
# ==============================================================================
class TitanAnalytics:
    @staticmethod
    def calculate_all(df):
        if df.empty: return df
        
        # Working on a copy to prevent SettingWithCopy warnings
        df = df.copy()
        
        # --- HELPER FUNCTIONS ---
        def get_wma(series, length):
            weights = np.arange(1, length + 1)
            return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

        def get_hma(series, length):
            half_length = int(length / 2)
            sqrt_length = int(np.sqrt(length))
            wma_half = get_wma(series, half_length)
            wma_full = get_wma(series, length)
            diff = 2 * wma_half - wma_full
            return get_wma(diff, sqrt_length)

        def get_rma(series, length):
            return series.ewm(alpha=1/length, adjust=False).mean()

        def get_atr(df, length=14):
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            return get_rma(true_range, length)

        # --- A. APEX TREND (HMA Logic) ---
        length_main = 55
        mult = 1.5
        
        df['Apex_Baseline'] = get_hma(df['Close'], length_main)
        df['Apex_ATR'] = get_atr(df, length_main)
        df['Apex_Upper'] = df['Apex_Baseline'] + (df['Apex_ATR'] * mult)
        df['Apex_Lower'] = df['Apex_Baseline'] - (df['Apex_ATR'] * mult)
        
        # Trend Logic
        trend_col = []
        for i in range(len(df)):
            close = df['Close'].iloc[i]
            upper = df['Apex_Upper'].iloc[i]
            lower = df['Apex_Lower'].iloc[i]
            
            if close > upper: 
                trend_col.append(1)
            elif close < lower: 
                trend_col.append(-1)
            else:
                # Carry forward previous trend if inside the cloud
                trend_col.append(trend_col[-1] if trend_col else 0)
        df['Apex_Trend'] = trend_col

        # --- B. SQUEEZE MOMENTUM (LazyBear) ---
        # Bollinger Bands
        bb_len = 20
        bb_mult = 2.0
        df['BB_Basis'] = df['Close'].rolling(bb_len).mean()
        df['BB_Dev'] = df['Close'].rolling(bb_len).std()
        df['BB_Upper'] = df['BB_Basis'] + (bb_mult * df['BB_Dev'])
        df['BB_Lower'] = df['BB_Basis'] - (bb_mult * df['BB_Dev'])
        
        # Keltner Channels
        kc_len = 20
        kc_mult = 1.5
        df['KC_Basis'] = df['Close'].rolling(kc_len).mean()
        df['KC_ATR'] = get_atr(df, kc_len)
        df['KC_Upper'] = df['KC_Basis'] + (kc_mult * df['KC_ATR'])
        df['KC_Lower'] = df['KC_Basis'] - (kc_mult * df['KC_ATR'])
        
        # Squeeze Status (ON if BB inside KC)
        df['Squeeze_On'] = (df['BB_Lower'] > df['KC_Lower']) & (df['BB_Upper'] < df['KC_Upper'])

        # --- C. GANN HIGH LOW ACTIVATOR ---
        gann_len = 3
        df['Gann_High'] = df['High'].rolling(gann_len).mean()
        df['Gann_Low'] = df['Low'].rolling(gann_len).mean()
        
        # --- D. FEAR & GREED COMPONENT ---
        # RSI + Volatility Composite
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Volatility_Score'] = (df['Close'].rolling(20).std() / df['Close']) * 100
        
        return df

# ==============================================================================
# 2. DEFINE ASSET LISTS
# ==============================================================================

# A. MACRO DASHBOARD ROWS
row1_core = [
    ("S&P 500", "^GSPC", "US Large Cap Benchmark."),
    ("Bitcoin", "BTC-USD", "Crypto/Liquidity Proxy."),
    ("10Y Yield", "^TNX", "US Risk-Free Rate."),
    ("VIX", "^VIX", "Implied Volatility (Fear Gauge).")
]

row2_uk = [
    ("FTSE 100", "^FTSE", "UK Blue Chips (Energy/Mining)."),
    ("FTSE 250", "^FTMC", "UK Domestic Economy."),
    ("GBP/USD", "GBPUSD=X", "Cable - Sterling vs USD."),
    ("EUR/GBP", "EURGBP=X", "Sterling vs Euro.")
]

row3_global = [
    ("Gold", "GC=F", "Inflation hedge & Safe-haven."),
    ("Brent Crude", "BZ=F", "North Sea Oil (UK/EU Benchmark)."),
    ("Dollar Index", "DX-Y.NYB", "DXY - Global Currency Strength."),
    ("Emerging Mkts", "EEM", "Global Growth Proxy.")
]

row4_eu_japan = [
    ("Nikkei 225", "^N225", "Japan Benchmark."),
    ("DAX", "^GDAXI", "Germany 40 (Industrial)."),
    ("CAC 40", "^FCHI", "France 40 (Luxury/Finance)."),
    ("Euro Stoxx 50", "^STOXX50E", "Eurozone Blue Chips.")
]

# B. SCREENSHOT ASSETS (Deep Dive Sections)
sect1_crypto_metals = [
    ("BTC/USD", "BTC-USD", "Bitcoin."), ("ETH/USD", "ETH-USD", "Ethereum."),
    ("Gold Futures", "GC=F", "Gold."), ("Silver Futures", "SI=F", "Silver."),
    ("Copper", "HG=F", "Copper (Dr. Copper).")
]

sect2_indices = [
    ("Nasdaq 100", "^NDX", "Tech Benchmark."), ("Invesco QQQ", "QQQ", "Nasdaq ETF."),
    ("UK 100", "^FTSE", "FTSE 100."), ("Euro Stoxx 50", "^STOXX50E", "EU Stoxx."),
    ("Nikkei 225", "^N225", "Japan.")
]

sect3_miners_themes = [
    ("Global Miners", "PICK", "Mining Producers."), ("Gold Miners", "RING", "Gold Miners."),
    ("Rare Earths", "REMX", "Strategic Metals."), ("Copper Miners", "COPX", "Copper Miners."),
    ("WTI Crude", "CL=F", "US Oil.")
]

sect4_forex_bonds = [
    ("EUR/USD", "EURUSD=X", "Euro/Dollar."), ("USD/JPY", "JPY=X", "Dollar/Yen."),
    ("US 10Y Yield", "^TNX", "10Y Treasury."), ("US 05Y Yield", "^FVX", "5Y Treasury.") 
]

# C. CRYPTO SNIPER LIST (Top 20)
top_crypto_assets = [
    ("BTC-USD", "Bitcoin (BTC)", "The Market King."),
    ("ETH-USD", "Ethereum (ETH)", "L1 King."),
    ("SOL-USD", "Solana (SOL)", "High Speed L1."),
    ("XRP-USD", "XRP (Ripple)", "Payments."),
    ("BNB-USD", "Binance Coin", "Exchange Token."),
    ("DOGE-USD", "Dogecoin", "Meme King."),
    ("ADA-USD", "Cardano", "Academic L1."),
    ("TRX-USD", "TRON", "USDT Network."),
    ("AVAX-USD", "Avalanche", "Subnets."),
    ("SHIB-USD", "Shiba Inu", "Meme Beta."),
    ("LINK-USD", "Chainlink", "Oracles."),
    ("BCH-USD", "Bitcoin Cash", "OG Fork."),
    ("DOT-USD", "Polkadot", "Interoperability."),
    ("LTC-USD", "Litecoin", "Digital Silver."),
    ("NEAR-USD", "NEAR", "Sharding."),
    ("UNI7083-USD", "Uniswap", "DEX King."),
    ("ICP-USD", "Internet Comp", "Web3 Cloud."),
    ("XLM-USD", "Stellar", "Payments."),
    ("HBAR-USD", "Hedera", "Enterprise."),
    ("FET-USD", "Fetch.ai", "AI Agent Proxy.")
]

# ==============================================================================
# 3. DATA FUNCTIONS (ROBUST)
# ==============================================================================

def get_all_data(all_lists_combined):
    """Fetches data for ALL assets in one go."""
    tickers = list(set([item[1] for item in all_lists_combined]))
    # Download data. Note: If tickers fail, yfinance handles them gracefully usually.
    # Group by ticker to handle multi-index outputs correctly
    data = yf.download(tickers, period="60d", interval="1d", progress=False, group_by='ticker')
    return data

def calculate_ratios(data):
    """Calculates Section 3 Ratios with Crash Protection."""
    ratios = {}
    
    # Robust helper to get latest price safely
    def get_price(ticker):
        try:
            # Check if ticker level exists in multi-index dataframe
            if ticker in data.columns.levels[0]: 
                # Access the Close column for that ticker
                series = data[ticker]['Close'].dropna()
                if not series.empty:
                    return series.iloc[-1]
            # Fallback for flat structure if only 1 ticker was fetched (rare but possible)
            elif 'Close' in data.columns and ticker in data.index: 
                 pass # Edge case handling skipped for brevity, standard is multi-index
        except Exception:
            pass
        return None

    # Fetch safely
    p_gold = get_price('GC=F')
    p_spx = get_price('^GSPC')
    p_btc = get_price('BTC-USD')
    p_copper = get_price('HG=F')
    y_10 = get_price('^TNX')
    y_5 = get_price('^FVX')

    # Compute Ratios (Only if both sides exist)
    if p_spx and p_gold: ratios["SPX/Gold"] = p_spx / p_gold
    if p_gold and p_btc: ratios["Gold/BTC"] = p_gold / p_btc
    if p_copper and p_gold: ratios["Copper/Gold"] = p_copper / p_gold
    if y_10 and y_5: ratios["10Y/5Y Spread"] = y_10 / y_5 

    return ratios

def render_row(title, asset_list, data_frame):
    st.markdown(f"#### {title}")
    cols = st.columns(len(asset_list))
    
    for col, (label, ticker, tip) in zip(cols, asset_list):
        try:
            # Safe Data Extraction
            if ticker in data_frame.columns.levels[0]:
                series = data_frame[ticker]['Close'].dropna()
            else:
                series = pd.Series()

            if not series.empty:
                curr = series.iloc[-1]
                prev = series.iloc[-2]
                delta = (curr - prev) / prev
                
                # Format
                if "Yield" in label or "VIX" in label: fmt = f"{curr:.2f}"
                elif "GBP" in label or "USD" in label or "EUR" in label: fmt = f"{curr:.4f}"
                else: fmt = f"{curr:,.2f}"

                with col:
                    st.metric(label=label, value=fmt, delta=f"{delta:.2%}", help=tip)
            else:
                with col: st.warning("No Data")
        except Exception:
             with col: st.metric(label=label, value="--")

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================

st.title("👁️ DarkPool Titan Terminal")
st.markdown("**Institutional-Grade Market Intelligence**")

# 1. Aggregate Tickers
master_list = (row1_core + row2_uk + row3_global + row4_eu_japan + 
               sect1_crypto_metals + sect2_indices + sect3_miners_themes + sect4_forex_bonds + top_crypto_assets)

# 2. Fetch & Compute
with st.spinner("Initializing Titan Data Feed..."):
    market_data = get_all_data(master_list)
    ratios = calculate_ratios(market_data)

# --- SECTION A: MACRO OUTLOOK ---
st.markdown("### 🌍 World Macro View")
st.markdown("---")
render_row("1. Core Drivers", row1_core, market_data)
st.markdown("")
render_row("2. UK Strategic View", row2_uk, market_data)
st.markdown("")
render_row("3. Global Commodities & Macro", row3_global, market_data)
st.markdown("")
render_row("4. Japan & Eurozone Indices", row4_eu_japan, market_data)
st.markdown("---")

# --- SECTION B: DEEP DIVE ---
st.markdown("### 🦅 Deep Dive: Sectors & Themes")
render_row("💎 Crypto & Precious Metals", sect1_crypto_metals, market_data)
st.divider()
render_row("📊 Global Indices (Expanded)", sect2_indices, market_data)
st.divider()
render_row("⛏️ Miners, Energy & Rare Earths", sect3_miners_themes, market_data)
st.divider()
render_row("💱 Currencies & Bonds", sect4_forex_bonds, market_data)
st.divider()

# --- SECTION C: RATIOS ---
st.markdown("#### ⚡ Inter-market Ratios")
r_cols = st.columns(4)
r_metrics = [
    ("SPX/Gold", ratios.get("SPX/Gold"), "Equities priced in Gold."),
    ("Gold/BTC", ratios.get("Gold/BTC"), "Old vs Digital Gold."),
    ("Copper/Gold", ratios.get("Copper/Gold"), "Growth vs Fear."),
    ("10Y/5Y Yield", ratios.get("10Y/5Y Spread"), "Curve Shape.")
]
for col, (label, val, tip) in zip(r_cols, r_metrics):
    with col:
        if val: st.metric(label=label, value=f"{val:.3f}", help=tip)
        else: st.metric(label=label, value="--")

st.markdown("---")

# --- SECTION D: CRYPTO SNIPER + TITAN INTELLIGENCE ---
st.markdown("### 🪙 Crypto Sniper Scope")

c1, c2 = st.columns([1, 3])

with c1:
    crypto_options = {name: (ticker, tip) for ticker, name, tip in top_crypto_assets}
    selected_name = st.selectbox("Select Asset:", options=list(crypto_options.keys()))
    sel_ticker, sel_tip = crypto_options[selected_name]

with c2:
    # 1. Fetch Specific History for Indicators (Independent Fetch)
    df_coin = yf.download(sel_ticker, period="3mo", interval="1d", progress=False)
    
    if not df_coin.empty:
        # Check structure: if multi-index (Ticker -> Close), flatten it
        if isinstance(df_coin.columns, pd.MultiIndex):
            df_coin = df_coin.xs(sel_ticker, axis=1, level=0)
            
        # 2. RUN TITAN ANALYTICS
        df_coin = TitanAnalytics.calculate_all(df_coin)
        latest = df_coin.iloc[-1]
        
        # --- TAB INTERFACE ---
        tab_chart, tab_tech = st.tabs(["📈 Price Action", "🧠 Titan Intelligence"])
        
        with tab_chart:
            curr = latest['Close']
            # Calculate delta safely
            try:
                prev = df_coin['Close'].iloc[-2]
                delta = (curr - prev) / prev
            except:
                delta = 0
                
            st.metric(label=f"{selected_name} Price", value=f"${curr:,.4f}", delta=f"{delta:.2%}", help=sel_tip)
            st.line_chart(df_coin['Close'], height=300)

        with tab_tech:
            st.markdown("#### 🛡️ DarkPool Indicator Matrix")
            
            # --- APEX TREND ---
            col_t1, col_t2, col_t3 = st.columns(3)
            # Safe access to trend data
            apex_val = latest.get('Apex_Trend', 0)
            apex_status = "BULLISH 🟢" if apex_val == 1 else "BEARISH 🔴"
            
            with col_t1:
                st.metric("Apex Trend (SMC)", apex_status, help="Hull MA + ATR Trailing Stop Logic")
            
            # --- SQUEEZE MOMENTUM ---
            sqz_val = latest.get('Squeeze_On', False)
            sqz_status = "ACTIVE 💥" if sqz_val else "RELEASED 💨"
            with col_t2:
                st.metric("Squeeze Pro", sqz_status, help="Bollinger Bands inside Keltner Channels")
            
            # --- GANN ACTIVATOR ---
            g_high = latest.get('Gann_High', 0)
            curr_price = latest['Close']
            gann_status = "LONG 🔼" if curr_price > g_high else "SHORT 🔽"
            with col_t3:
                st.metric("Gann Activator", gann_status, help="High/Low Moving Average Logic")
                
            st.divider()
            
            # --- METRICS GRID ---
            m1, m2, m3 = st.columns(3)
            with m1:
                rsi_val = latest.get('RSI', 50)
                st.metric("RSI (14)", f"{rsi_val:.1f}", help=">70 Overbought, <30 Oversold")
            with m2:
                vol_val = latest.get('Volatility_Score', 0)
                st.metric("Volatility Score", f"{vol_val:.2f}", help="Standard Deviation normalized")
            with m3:
                base_val = latest.get('Apex_Baseline', curr_price)
                dist_ma = curr_price - base_val
                st.metric("Dist to Baseline", f"{dist_ma:.2f}", help="Distance to Hull Moving Average")

    else:
        st.error(f"Data Unavailable for {selected_name}")
