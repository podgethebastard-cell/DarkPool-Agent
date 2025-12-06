import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from openai import OpenAI

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="DarkPool Ultimate Architect")
st.title("👁️ DarkPool Ultimate Architect")
st.markdown("### Institutional Trade Planning Engine")

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# ==========================================
# 2. MATH LIBRARY
# ==========================================
def calc_hma(series, length):
    wma1 = series.rolling(length // 2).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
    wma2 = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
    diff = 2 * wma1 - wma2
    sqrt_len = int(np.sqrt(length))
    return diff.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)

def calc_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_rma(series, length):
    """Wilder's Smoothing (RMA) for ATR"""
    return series.ewm(alpha=1/length, adjust=False).mean()

def safe_download(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.droplevel(1) 
            except: pass 
        if df.empty or 'Close' not in df.columns: return None
        return df
    except: return None

# ==========================================
# 3. MACRO DASHBOARD
# ==========================================
def get_macro_data():
    tickers = ["SPY", "QQQ", "^TNX", "DX-Y.NYB", "^VIX", "BTC-USD"]
    try:
        data = yf.download(tickers, period="5d", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            df_close = data['Close']
        else:
            df_close = data['Close']
        changes = df_close.pct_change().iloc[-1] * 100
        prices = df_close.iloc[-1]
        return prices, changes
    except: return None, None

st.markdown("### 🌍 Global Macro Regime")
m_price, m_chg = get_macro_data()
if m_price is not None:
    spy_chg = m_chg.get("SPY", 0)
    vix = m_price.get("^VIX", 0)
    regime = "RISK-ON 🟢" if spy_chg > 0 and vix < 20 else "RISK-OFF 🔴" if vix > 25 else "NEUTRAL 🟡"
    st.info(f"**Market Regime:** {regime}")
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def show(col, lbl, k): 
        col.metric(lbl, f"{m_price.get(k,0):.2f}", f"{m_chg.get(k,0):.2f}%")
    show(c1,"S&P 500","SPY"); show(c2,"Nasdaq","QQQ"); show(c3,"Bitcoin","BTC-USD")
    show(c4,"10Y Yield","^TNX"); show(c5,"Dollar","DX-Y.NYB"); show(c6,"VIX","^VIX")
    st.markdown("---")

# ==========================================
# 4. DATA PROCESSING
# ==========================================
def get_full_analysis(ticker, interval):
    if "---" in ticker: return None
    df = safe_download(ticker, period="1y", interval=interval)
    if df is None: return None

    # --- 1. APEX TREND ---
    df['HMA'] = calc_hma(df['Close'], 55)
    # Basic ATR for Apex logic
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['TR'] = np.max(ranges, axis=1)
    
    df['ATR_55'] = df['TR'].rolling(55).mean()
    df['Apex_Up'] = df['HMA'] + (df['ATR_55'] * 1.5)
    df['Apex_Dn'] = df['HMA'] - (df['ATR_55'] * 1.5)
    df['Apex_State'] = np.where(df['Close'] > df['Apex_Up'], 1, np.where(df['Close'] < df['Apex_Dn'], -1, 0))
    df['Apex_State'] = df['Apex_State'].replace(to_replace=0, method='ffill')

    # --- 2. ATR STOP LOSS FINDER (The New Script) ---
    # Logic: ATR(14, RMA) * 1.5
    # Long Stop = Low - (ATR * 1.5)
    # Short Stop = High + (ATR * 1.5)
    atr_len = 14
    atr_mult = 1.5
    
    df['ATR_RMA'] = calc_rma(df['TR'], atr_len)
    df['ATR_Stop_Long'] = df['Low'] - (df['ATR_RMA'] * atr_mult)
    df['ATR_Stop_Short'] = df['High'] + (df['ATR_RMA'] * atr_mult)

    # --- 3. GANN HI/LO ACTIVATOR ---
    gl = 3
    df['Gann_Hi'] = df['High'].rolling(gl).mean()
    df['Gann_Lo'] = df['Low'].rolling(gl).mean()
    df['Gann_State'] = np.where(df['Close'] > df['Gann_Hi'].shift(1), 1, np.where(df['Close'] < df['Gann_Lo'].shift(1), -1, 0))
    df['Gann_State'] = df['Gann_State'].replace(to_replace=0, method='ffill')

    # --- 4. SQUEEZE PRO ---
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['KC_ATR'] = df['TR'].rolling(20).mean() # SMA ATR for Squeeze
    df['Sq_On'] = (df['BB_Mid'] - (2*df['BB_Std'])) > (df['BB_Mid'] - (1.5*df['KC_ATR']))
    df['Mom'] = df['Close'] - df['Close'].rolling(20).mean()

    # --- 5. MONEY FLOW MATRIX ---
    rsi = calc_rsi(df['Close'], 14)
    df['MFI'] = (rsi - 50) * (df['Volume'] / df['Volume'].rolling(20).mean())
    df['MFI_Smooth'] = df['MFI'].ewm(span=3).mean()

    return df

# ==========================================
# 5. AI CHAIRMAN (WITH ATR STOP LOGIC)
# ==========================================
def ask_chairman(df, ticker, balance):
    if not api_key: return "⚠️ API Key Missing."
    last = df.iloc[-1]
    
    price = float(last['Close'])
    # Get the new ATR Stops
    stop_long_level = float(last['ATR_Stop_Long'])
    stop_short_level = float(last['ATR_Stop_Short'])
    
    # --- POSITION SIZING (2% RISK) ---
    risk_amount = balance * 0.02
    
    # LONG MATH
    dist_long = price - stop_long_level
    if dist_long <= 0: dist_long = price * 0.01 # Safety
    qty_long = risk_amount / dist_long
    
    # SHORT MATH
    dist_short = stop_short_level - price
    if dist_short <= 0: dist_short = price * 0.01
    qty_short = risk_amount / dist_short
    
    # Targets (2R and 3R)
    tp1_long = price + (dist_long * 2)
    tp2_long = price + (dist_long * 3)
    
    tp1_short = price - (dist_short * 2)
    tp2_short = price - (dist_short * 3)
    
    # States
    apex = "BULL 🟢" if last['Apex_State'] == 1 else "BEAR 🔴"
    gann = "UP 🔼" if last['Gann_State'] == 1 else "DOWN 🔽"
    sqz = "ACTIVE ⚡" if last['Sq_On'] else "OFF"
    flow = "INFLOW 🟩" if last['MFI_Smooth'] > 0 else "OUTFLOW 🟥"
    
    prompt = f"""
    Act as a Hedge Fund Risk Manager. Analyze {ticker} at ${price:.2f}.
    User Balance: ${balance}. Max Risk per trade: ${risk_amount:.2f} (2%).
    
    --- TECHNICAL DASHBOARD ---
    1. MACRO TREND (Apex): {apex}
    2. SWING (Gann): {gann}
    3. MOMENTUM: Squeeze: {sqz}. Money Flow: {flow}.
    
    --- ATR STOP LOSS FINDER DATA ---
    * **IF LONG:** Stop Price: ${stop_long_level:.2f}. (Buy {qty_long:.4f} units).
    * **IF SHORT:** Stop Price: ${stop_short_level:.2f}. (Sell {qty_short:.4f} units).
    
    --- YOUR TASK ---
    1. **Verdict:** LONG, SHORT, or WAIT? (Is the trend clear?)
    2. **The Plan (Only for the winning side):**
       - **Action:** (e.g. Buy Market)
       - **Hard Stop:** (Use the ATR Stop level above)
       - **Position Size:** (Use the calculated units above)
       - **Target 1:** ${tp1_long:.2f} (Long) / ${tp1_short:.2f} (Short)
       - **Target 2:** ${tp2_long:.2f} (Long) / ${tp2_short:.2f} (Short)
    """
    
    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}])
    return res.choices[0].message.content

# ==========================================
# 6. UI RENDERER
# ==========================================
st.sidebar.header("Mission Control")
asset_options = {
    "--- CRYPTO ---": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"],
    "--- INDICES ---": ["SPY", "QQQ", "IWM", "DIA"], 
    "--- TECH ---": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD"],
    "--- COMMODITIES ---": ["GC=F", "SI=F", "CL=F", "HG=F", "NG=F"],
    "--- MACRO ---": ["DX-Y.NYB", "^TNX", "^VIX", "HYG", "TLT"]
}
flat = [i for s in asset_options.values() for i in s]
ticker = st.sidebar.selectbox("Asset", flat, index=0)
interval = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1wk"], index=2)
balance = st.sidebar.number_input("Account Balance ($)", min_value=100, value=10000, step=100)

if st.sidebar.button("Run Ultimate Analysis"):
    with st.spinner(f"Calculating ATR Stops & Strategy for {ticker}..."):
        df = get_full_analysis(ticker, interval)
        if df is not None:
            # --- MAIN CHART ---
            st.subheader(f"1. Strategic Command ({ticker})")
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=("Price & ATR Stops", "Money Flow Matrix", "Momentum & Squeeze"))

            # ROW 1: Price
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            # ATR Stops (Teal for Long Stop, Red for Short Stop)
            fig.add_trace(go.Scatter(x=df.index, y=df['ATR_Stop_Long'], line=dict(color='teal', width=1), name="ATR Support (Long Stop)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ATR_Stop_Short'], line=dict(color='red', width=1), name="ATR Resist (Short Stop)"), row=1, col=1)
            
            # Apex Cloud (Background Context)
            fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Up'], line=dict(color='rgba(0, 255, 0, 0.3)', width=1, dash='dot'), name="Apex Top"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Dn'], line=dict(color='rgba(255, 0, 0, 0.3)', width=1, dash='dot'), name="Apex Bot"), row=1, col=1)

            # ROW 2: Money Flow
            cols_mf = ['#00ff00' if v>0 else '#ff0000' for v in df['MFI_Smooth']]
            fig.add_trace(go.Bar(x=df.index, y=df['MFI_Smooth'], marker_color=cols_mf, name="Money Flow"), row=2, col=1)

            # ROW 3: Squeeze
            cols_mom = ['cyan' if m>0 else 'purple' for m in df['Mom']]
            fig.add_trace(go.Bar(x=df.index, y=df['Mom'], marker_color=cols_mom, name="Momentum"), row=3, col=1)
            sqz_y = [0] * len(df)
            cols_sqz = ['red' if s else 'gray' for s in df['Sq_On']]
            fig.add_trace(go.Scatter(x=df.index, y=sqz_y, mode='markers', marker=dict(color=cols_sqz, size=4), name="Squeeze"), row=3, col=1)

            fig.update_layout(height=900, xaxis_rangeslider_visible=False, template="plotly_dark", title_text=f"DarkPool Architecture: {ticker}")
            st.plotly_chart(fig, use_container_width=True)

            # --- AI VERDICT ---
            st.markdown("---")
            st.subheader("🧠 Chairman's Execution Plan")
            st.info(ask_chairman(df, ticker, balance))
            
            # --- DISCLAIMER ---
            st.markdown("---")
            with st.expander("⚠️ Disclaimer"):
                st.caption("Not Financial Advice. For Research Only.")
        else:
            st.error("Data Error. Try Rebooting.")
