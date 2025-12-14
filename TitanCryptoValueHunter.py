import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time

# ==========================================
# 1. PAGE CONFIGURATION & DARKPOOL UI
# ==========================================
st.set_page_config(
    page_title="KRAKEN TITAN FLOW",
    page_icon="🐙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DARKPOOL AESTHETIC CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Neon Glow Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(88, 101, 242, 0.5); /* Kraken Blue/Purple hue */
        font-family: 'Roboto Mono', monospace;
        font-weight: 800;
    }
    
    /* Custom Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(0, 0, 0, 0.2));
        border: 1px solid rgba(88, 101, 242, 0.3);
        padding: 15px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #5865F2;
        box-shadow: 0 0 15px rgba(88, 101, 242, 0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #5865F2 !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #888 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #161b22;
        border: 1px solid #30363d;
        color: #8b949e;
        border-radius: 6px;
        padding: 0 20px;
        transition: 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #5865F2 !important;
        color: #ffffff !important;
        border-color: #5865F2 !important;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1f2937;
        color: #5865F2;
        border: 1px solid #5865F2;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background: #5865F2;
        color: #fff;
        box-shadow: 0 0 15px rgba(88, 101, 242, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DARKPOOL FLOW ENGINE (50D / 50W)
# ==========================================

class DarkPoolMath:
    @staticmethod
    def calculate_sma(series, length):
        """Simple Moving Average"""
        return series.rolling(window=length).mean()

    @staticmethod
    def calculate_atr(df, length=14):
        """Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=length).mean()

    @staticmethod
    def calculate_rsi(series, length=14):
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_adx(df, length=14):
        """Average Directional Index (ADX)"""
        up = df['High'].diff()
        down = -df['Low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        tr = DarkPoolMath.calculate_atr(df, length)
        
        # Avoid division by zero
        tr = tr.replace(0, np.nan)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(length).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(length).mean() / tr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(length).mean()
        return adx

    @staticmethod
    def get_dual_anchors(daily_df):
        """
        Calculates:
        1. 50-Day SMA (Daily Data)
        2. 50-Week SMA (Weekly Data, resampled to Daily)
        """
        # 1. Daily Anchor (50-Day SMA)
        daily_df['SMA_50D'] = DarkPoolMath.calculate_sma(daily_df['Close'], 50)
        
        # 2. Weekly Anchor (50-Week SMA)
        # Resample to Weekly W-SUN
        weekly_df = daily_df.resample('W-SUN').agg({'Close': 'last'})
        weekly_df['SMA_50W'] = DarkPoolMath.calculate_sma(weekly_df['Close'], 50)
        
        # Reindex Weekly back to Daily (Forward Fill)
        # This projects the weekly level forward across the daily candles
        daily_anchors = weekly_df['SMA_50W'].reindex(daily_df.index, method='ffill')
        daily_df['SMA_50W'] = daily_anchors
        
        return daily_df

    @staticmethod
    def calculate_flow_score(row):
        """
        SCORING LOGIC (50D vs 50W - Crypto Adapted)
        """
        score = 0
        
        # 1. Major Trend Alignment (50D > 50W)
        is_uptrend = row['SMA_50D'] > row['SMA_50W']
        
        if is_uptrend:
            score += 40
        else:
            # In Crypto, we might punish downtrends harder
            return 0 
            
        # 2. Value Zone Logic
        price = row['Close']
        fast = row['SMA_50D']
        slow = row['SMA_50W']
        
        # Value Zone: Between 50W (Support) and 50D (Resistance/Momentum)
        if slow < price < fast:
            score += 40  # GOLDEN POCKET
        elif price > fast:
            # ABOVE TREND (Momentum)
            dist_pct = (price - fast) / fast * 100
            # Crypto allows for more extension than stocks
            if dist_pct < 10.0: 
                score += 25 
            else:
                score += 5 # Too extended
        elif price < slow:
            score -= 20 # Trend Broken

        # 3. ADX Filter (Trend Strength)
        if row['ADX'] > 20:
            score += 10
            
        # 4. RSI Check
        if 35 < row['RSI'] < 60: 
            score += 10
            
        return max(0, min(100, score))

# ==========================================
# 3. DATA ENGINE (KRAKEN / CCXT)
# ==========================================

@st.cache_data
def get_kraken_universe():
    """Curated list of Liquid Kraken Pairs (USD)."""
    return [
        "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", 
        "DOGE/USD", "DOT/USD", "LINK/USD", "MATIC/USD", "LTC/USD",
        "BCH/USD", "UNI/USD", "XLM/USD", "ATOM/USD", "FIL/USD",
        "AAVE/USD", "ALGO/USD", "NEAR/USD", "QNT/USD", "EOS/USD",
        "XTZ/USD", "GRT/USD", "MKR/USD", "SNX/USD", "SAND/USD"
    ]

@st.cache_resource
def init_exchange():
    return ccxt.kraken()

@st.cache_data(ttl=300)
def fetch_crypto_history(symbol, limit=720):
    """Fetches daily data from Kraken via CCXT."""
    try:
        exchange = init_exchange()
        # Fetch Daily (1d) candles
        bars = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=limit)
        if not bars:
            return pd.DataFrame()
            
        df = pd.DataFrame(bars, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()

def scan_market(tickers):
    """
    Scans Kraken assets using the 50D/50W logic.
    """
    data_list = []
    
    # Create a progress bar since we fetch serially with CCXT
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Scanning {ticker}...")
        try:
            # Need ~2 years of data for reliable 50W SMA calculation (50 weeks * 7 = 350 days, plus warmup)
            # Kraken max limit is often 720 candles for 1d, which is ~2 years. Perfect.
            df = fetch_crypto_history(ticker, limit=720)
            
            if df.empty or len(df) < 365: 
                progress_bar.progress((i + 1) / total)
                continue
            
            # --- DARKPOOL ENGINE ---
            # 1. Indicators
            df['RSI'] = DarkPoolMath.calculate_rsi(df['Close'], 14)
            df['ADX'] = DarkPoolMath.calculate_adx(df, 14)
            
            # 2. Calculate 50D and 50W Anchors
            df = DarkPoolMath.get_dual_anchors(df)
            
            # 3. Get Latest State
            last = df.iloc[-1]
            
            # Skip if 50W hasn't calculated yet (NaN)
            if pd.isna(last['SMA_50W']): 
                progress_bar.progress((i + 1) / total)
                continue

            info = {
                'Ticker': ticker,
                'Close': last['Close'],
                'SMA_50D': last['SMA_50D'],
                'SMA_50W': last['SMA_50W'],
                'RSI': last['RSI'],
                'ADX': last['ADX'],
                'Volume': last['Volume']
            }
            
            # 4. Score
            info['Flow_Score'] = DarkPoolMath.calculate_flow_score(info)
            
            # 5. Status String
            if info['SMA_50D'] > info['SMA_50W']:
                if info['SMA_50W'] < info['Close'] < info['SMA_50D']:
                    info['Status'] = "VALUE ZONE 💎"
                elif info['Close'] > info['SMA_50D']:
                    info['Status'] = "MOMENTUM 🚀"
                else:
                    info['Status'] = "CAUTION ⚠️"
            else:
                info['Status'] = "BEARISH 🐻"
                
            data_list.append(info)
            
        except Exception:
            pass
        
        progress_bar.progress((i + 1) / total)
    
    status_text.empty()
    progress_bar.empty()
            
    return pd.DataFrame(data_list)

# ==========================================
# 4. UI VISUALIZATION
# ==========================================

def render_chart(ticker):
    """Plots the 50D/50W Cloud Chart for Crypto."""
    with st.spinner(f"Rendering DarkPool Charts for {ticker}..."):
        df = fetch_crypto_history(ticker, limit=720)
        
        if df.empty:
            st.error("No data available.")
            return

        # Re-calc Logic for Chart
        df = DarkPoolMath.get_dual_anchors(df)
        df['RSI'] = DarkPoolMath.calculate_rsi(df['Close'])
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # CANDLESTICK
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        
        # 50-DAY SMA (Fast Anchor)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50D'], line=dict(color='#00E676', width=2), name='50-Day SMA'), row=1, col=1)
        
        # 50-WEEK SMA (Slow Anchor)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50W'], line=dict(color='#5865F2', width=2), name='50-Week SMA'), row=1, col=1)
        
        # Cloud Fill
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(df.index), pd.Series(df.index)[::-1]]),
            y=pd.concat([df['SMA_50D'], df['SMA_50W'][::-1]]),
            fill='toself',
            fillcolor='rgba(88, 101, 242, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Flow Cloud',
            showlegend=True
        ), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#e0e0e0', width=1), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#FF5252", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00E676", row=2, col=1)
        
        fig.update_layout(
            height=650,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False,
            title=f"Titan Flow Structure: {ticker}"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. MAIN APP
# ==========================================

def main():
    # Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🐙 TITAN KRAKEN HUNTER")
        st.caption("Strategy: 50-Day vs 50-Week Convergence (Crypto Edition)")
    with c2:
        if st.button("🔄 REFRESH"):
            st.cache_data.clear()
            st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ FLOW SETTINGS")
        st.info("Core Logic: Price interaction with the 50D/50W Cloud on Kraken pairs.")
        
        min_score = st.slider("Min Titan Score", 0, 100, 60, help="Higher = Better Technical Value Setup")
        
        st.subheader("Filters")
        only_cloud_retest = st.checkbox("Value Zone Only", value=False, help="Show assets strictly inside the 50D/50W Cloud")
        high_adx_only = st.checkbox("High Strength Only (ADX > 20)", value=True)

    # Main Logic
    universe = get_kraken_universe()
    
    # Check if scan has run
    if 'scan_run' not in st.session_state:
        st.info("👈 Hit 'RUN SCAN' below to analyze the Kraken Market.")
        if st.button("RUN KRAKEN SCAN", type="primary", use_container_width=True):
            st.session_state['scan_run'] = True
            st.rerun()
        return

    # Run Scan
    df = scan_market(universe)
    
    # Apply Filters
    if not df.empty:
        if high_adx_only:
            df = df[df['ADX'] > 20]
        
        if only_cloud_retest:
            df = df[ (df['Close'] > df['SMA_50W']) & (df['Close'] < df['SMA_50D']) ]
            
        df = df[df['Flow_Score'] >= min_score]
        df = df.sort_values(by="Flow_Score", ascending=False)

    # Display
    tab1, tab2 = st.tabs(["💎 CRYPTO VALUE", "🔬 CHART INSPECTOR"])

    with tab1:
        if not df.empty:
            # Top Picks Cards
            st.markdown("### 🏛️ Top Kraken Flow Targets")
            top_row = df.head(4)
            cols = st.columns(4)
            
            for i, (idx, row) in enumerate(top_row.iterrows()):
                with cols[i]:
                    st.metric(
                        label=row['Ticker'],
                        value=f"${row['Close']:,.2f}",
                        delta=f"{row['Status']}"
                    )
                    st.progress(int(row['Flow_Score']))
                    st.caption(f"50D: {row['SMA_50D']:,.2f} | 50W: {row['SMA_50W']:,.2f}")
            
            # Main Table
            st.markdown("### 📋 Full Scan Results")
            
            def highlight_status(val):
                if 'VALUE' in val: color = '#FFD700' # Gold
                elif 'MOMENTUM' in val: color = '#00E676' # Green
                else: color = '#FF5252' # Red
                return f'color: {color}; font-weight: bold'

            display_df = df[['Ticker', 'Close', 'Status', 'Flow_Score', 'SMA_50D', 'SMA_50W', 'RSI', 'ADX']].copy()
            
            st.dataframe(
                display_df.style.map(highlight_status, subset=['Status'])
                                .background_gradient(subset=['Flow_Score'], cmap='Greens')
                                .format({'SMA_50D': '{:,.2f}', 'SMA_50W': '{:,.2f}', 'RSI': '{:.1f}', 'ADX': '{:.1f}', 'Close': '${:,.2f}'}),
                use_container_width=True,
                height=600
            )
        else:
            st.warning("No assets match your current Value Filters. Try lowering the Score or disabling the ADX filter.")

    with tab2:
        st.markdown("### 🔭 Institutional Chart")
        if not df.empty:
            sel = st.selectbox("Select Asset", df['Ticker'].tolist())
            render_chart(sel)
            
            with st.expander("ℹ️ Understanding the 50/50 Crypto Strategy"):
                st.markdown("""
                **The Titan 50/50 System (Crypto Edition):**
                1.  **50-Week SMA (Blue):** The ultimate long-term baseline. In Crypto bull markets, this line is rarely broken.
                2.  **50-Day SMA (Green):** The medium-term institutional trend.
                3.  **Bullish Alignment:** We only look for buys when the **50-Day > 50-Week**.
                4.  **Value Zone (Cloud):** When price dips *below* the 50-Day but stays *above* the 50-Week, it is in the 'Value Zone'. This is often the best risk/reward entry during a bull cycle.
                """)
        else:
            st.info("Run the scan to populate the inspector.")

if __name__ == "__main__":
    main()
