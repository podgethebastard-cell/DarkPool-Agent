import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
import streamlit.components.v1 as components

# ==========================================
# 1. CONFIG & UI SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="Crypto Titan Terminal", page_icon="💎")

# Custom CSS for that "Dark Mode" feel
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
.metric-box {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
.signal-long { color: #00ff00; font-weight: bold; }
.signal-short { color: #ff0000; font-weight: bold; }
.signal-neutral { color: #888888; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("💎 Crypto Control")

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Crypto Asset List
crypto_assets = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Ripple (XRP)": "XRP-USD",
    "Cardano (ADA)": "ADA-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Polkadot (DOT)": "DOT-USD",
    "Polygon (MATIC)": "MATIC-USD",
    "Shiba Inu (SHIB)": "SHIB-USD",
    "Litecoin (LTC)": "LTC-USD"
}

ticker_name = st.sidebar.selectbox("Select Asset", list(crypto_assets.keys()))
ticker = crypto_assets[ticker_name]
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)

# ==========================================
# 2. STRATEGY ENGINE (PYTHON CONVERSION)
# ==========================================
def calculate_strategies(df):
    """
    Implements the 4 user-provided Pine Script strategies in Python.
    """
    # --- 1. Momentum Strategy ---
    # Pine: mom0 = price - price[length] (Length=12)
    mom_len = 12
    df['Mom_Val'] = df['Close'] - df['Close'].shift(mom_len)
    # Signal: Mom > 0 AND Mom[1] > 0 -> LONG
    df['Sig_Mom'] = np.where((df['Mom_Val'] > 0) & (df['Mom_Val'].shift(1) > 0), 1, 
                    np.where((df['Mom_Val'] < 0) & (df['Mom_Val'].shift(1) < 0), -1, 0))

    # --- 2. Rob Booker ADX Breakout ---
    # ADX Calc (14)
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    
    df['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr = pd.concat([df['High'] - df['Low'], 
                    (df['High'] - df['Close'].shift(1)).abs(), 
                    (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    
    atr = tr.rolling(14).mean()
    plus_di = 100 * (df['+DM'].ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (df['-DM'].ewm(alpha=1/14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()
    
    # Breakout Box (20 period lookback)
    box_lookback = 20
    # Pine: highest(high, 20)[1] -> Max of PREVIOUS 20 bars
    df['Box_High'] = df['High'].rolling(box_lookback).max().shift(1)
    df['Box_Low'] = df['Low'].rolling(box_lookback).min().shift(1)
    
    # Logic: Buy if Close > Box_High AND ADX < 30 (Consolidation Breakout)
    # Note: User input was 'ADX Lower Level' (default 18, but commonly 20-30 for crypto)
    adx_thresh = 25 
    df['Sig_ADX'] = np.where((df['Close'] > df['Box_High']) & (df['ADX'] < adx_thresh), 1,
                    np.where((df['Close'] < df['Box_Low']) & (df['ADX'] < adx_thresh), -1, 0))

    # --- 3. Bollinger Bands Directed ---
    # Pine: crossover(source, lower) -> Long
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    bb_upper = sma20 + (2.0 * std20)
    bb_lower = sma20 - (2.0 * std20)
    
    # Crossover Logic: Current < Lower, Previous > Lower (Dip Buy) - Wait, Pine 'crossover(source, lower)' means Source crosses OVER lower line (going up).
    # Correct logic: Close > Lower and Prev_Close < Prev_Lower
    bb_long = (df['Close'] > bb_lower) & (df['Close'].shift(1) < bb_lower.shift(1))
    bb_short = (df['Close'] < bb_upper) & (df['Close'].shift(1) > bb_upper.shift(1))
    
    df['Sig_BB'] = np.where(bb_long, 1, np.where(bb_short, -1, 0))

    # --- 4. RSI Strategy ---
    # Pine: RSI Crosses 30 (Long) / 70 (Short)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    rsi_long = (df['RSI'] > 30) & (df['RSI'].shift(1) < 30)
    rsi_short = (df['RSI'] < 70) & (df['RSI'].shift(1) > 70)
    
    df['Sig_RSI'] = np.where(rsi_long, 1, np.where(rsi_short, -1, 0))
    
    return df

@st.cache_data(ttl=300)
def get_crypto_data(ticker, interval):
    # Adjust yfinance interval mapping
    period_map = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
    period = period_map.get(interval, "1y")
    
    # yfinance doesn't support 4h natively, we download 1h and resample
    dl_interval = "1h" if interval == "4h" else interval
    
    df = yf.download(ticker, period=period, interval=dl_interval, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    if interval == "4h":
        agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        df = df.resample('4h').agg(agg_dict).dropna()
    
    return df

# ==========================================
# 3. MAIN DASHBOARD
# ==========================================
st.title(f"💎 Crypto Titan: {ticker_name}")

# --- TRADINGVIEW WIDGET ---
# Map interval to TradingView code
tv_int_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
tv_symbol = f"BINANCE:{ticker.replace('-USD', 'USDT')}" # Best guess for symbol

components.html(
    f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%",
        "height": 500,
        "symbol": "{tv_symbol}",
        "interval": "{tv_int_map[interval]}",
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
    """,
    height=500,
)

# --- DATA & SIGNALS ---
df = get_crypto_data(ticker, interval)

if df is not None and not df.empty:
    df = calculate_strategies(df)
    last = df.iloc[-1]
    
    st.markdown("### ⚡ Live Strategy Signals")
    
    col1, col2, col3, col4 = st.columns(4)
    
    def get_sig_display(val, label):
        color = "signal-long" if val == 1 else "signal-short" if val == -1 else "signal-neutral"
        text = "LONG 🟢" if val == 1 else "SHORT 🔴" if val == -1 else "NEUTRAL ⚪"
        return f"<div class='metric-box'><div style='font-size:0.9em'>{label}</div><div class='{color}' style='font-size:1.2em'>{text}</div></div>"

    col1.markdown(get_sig_display(last['Sig_Mom'], "Momentum (12)"), unsafe_allow_html=True)
    col2.markdown(get_sig_display(last['Sig_ADX'], "ADX Breakout"), unsafe_allow_html=True)
    col3.markdown(get_sig_display(last['Sig_BB'], "Bollinger Reversion"), unsafe_allow_html=True)
    col4.markdown(get_sig_display(last['Sig_RSI'], "RSI (30/70)"), unsafe_allow_html=True)

    # --- AI ANALYST ---
    st.markdown("---")
    st.subheader("🤖 Crypto Analyst")
    
    if st.button("Generate AI Report"):
        if not api_key:
            st.error("Please enter OpenAI Key in Sidebar")
        else:
            with st.spinner("Analyzing market structure..."):
                client = OpenAI(api_key=api_key)
                
                # Prepare Signal Summary
                signals = {
                    "Momentum": "BULLISH" if last['Sig_Mom'] == 1 else "BEARISH" if last['Sig_Mom'] == -1 else "NEUTRAL",
                    "ADX Breakout": "BREAKOUT UP" if last['Sig_ADX'] == 1 else "BREAKOUT DOWN" if last['Sig_ADX'] == -1 else "NO SIGNAL",
                    "Bollinger": "BUY DIP" if last['Sig_BB'] == 1 else "SELL RIP" if last['Sig_BB'] == -1 else "WAIT",
                    "RSI": f"{last['RSI']:.2f}"
                }
                
                prompt = f"""
                Analyze {ticker} on the {interval} timeframe based on these algorithmic signals:
                
                1. Momentum Strategy: {signals['Momentum']}
                2. ADX Breakout Strategy: {signals['ADX Breakout']} (ADX Value: {last['ADX']:.2f})
                3. Bollinger Strategy: {signals['Bollinger']}
                4. RSI: {signals['RSI']}
                
                Current Price: ${last['Close']:.2f}
                
                Synthesize these 4 signals into a coherent trade plan. 
                Are they confirming each other or conflicting? 
                Give a strict verdict: BUY, SELL, or WAIT.
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                st.info(response.choices[0].message.content)

else:
    st.error("Error fetching data. Please try a different asset or timeframe.")
