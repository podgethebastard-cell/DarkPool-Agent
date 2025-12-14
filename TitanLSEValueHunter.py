import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# ==========================================
# 1. PAGE CONFIGURATION & DARKPOOL UI
# ==========================================
st.set_page_config(
    page_title="LSE TITAN 50/50",
    page_icon="🇬🇧",
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
        text-shadow: 0 0 10px rgba(0, 230, 118, 0.5);
        font-family: 'Roboto Mono', monospace;
        font-weight: 800;
    }
    
    /* Custom Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(0, 0, 0, 0.2));
        border: 1px solid rgba(0, 230, 118, 0.2);
        padding: 15px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #00E676;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.2);
    }
    div[data-testid="stMetricValue"] {
        color: #00E676 !important;
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
        background-color: #00E676 !important;
        color: #000000 !important;
        border-color: #00E676 !important;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1f2937;
        color: #00E676;
        border: 1px solid #00E676;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background: #00E676;
        color: #000;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.5);
    }
    
    /* Table Styling */
    .stDataFrame {
        border: 1px solid #333;
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
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(length).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(length).mean() / tr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(length).mean()
        return adx

    @staticmethod
    def get_dual_anchors(daily_df):
        """
        Calculates the user's preferred anchors:
        1. 50-Day SMA (Daily Data)
        2. 50-Week SMA (Weekly Data, resampled to Daily)
        """
        # 1. Daily Anchor (50-Day SMA)
        daily_df['SMA_50D'] = DarkPoolMath.calculate_sma(daily_df['Close'], 50)
        
        # 2. Weekly Anchor (50-Week SMA)
        weekly_df = daily_df.resample('W').agg({'Close': 'last'})
        weekly_df['SMA_50W'] = DarkPoolMath.calculate_sma(weekly_df['Close'], 50)
        
        # Reindex Weekly back to Daily (Forward Fill)
        # This aligns the 50W SMA to the daily chart
        daily_anchors = weekly_df['SMA_50W'].reindex(daily_df.index, method='ffill')
        daily_df['SMA_50W'] = daily_anchors
        
        return daily_df

    @staticmethod
    def calculate_flow_score(row):
        """
        SCORING LOGIC (50D vs 50W)
        1. Bull Trend: 50D > 50W
        2. Deep Value: Price inside the "Cloud" (Between 50D and 50W)
        """
        score = 0
        
        # 1. Major Trend Alignment (50D > 50W)
        # This confirms medium-term momentum is above long-term baseline
        is_uptrend = row['SMA_50D'] > row['SMA_50W']
        
        if is_uptrend:
            score += 40
        else:
            return 0 # Strict rule: We only want uptrends for value buying
            
        # 2. Value Zone Logic
        price = row['Close']
        fast = row['SMA_50D']
        slow = row['SMA_50W']
        
        # If Price is between Slow (50W) and Fast (50D), it's a deep pullback in an uptrend.
        if slow < price < fast:
            # GOLDEN VALUE ZONE
            score += 40 
        elif price > fast:
            # ABOVE TREND (Momentum)
            # Check how far extended it is
            dist_pct = (price - fast) / fast * 100
            if dist_pct < 5.0:
                score += 25 # Buying the breakout/support of 50D
            else:
                score += 10 # Extended
        elif price < slow:
            # BROKEN TREND (Below 50W) - High Risk
            score -= 20

        # 3. ADX Filter (Trend Strength)
        if row['ADX'] > 20:
            score += 10
            
        # 4. RSI Check
        if 35 < row['RSI'] < 60: # Ideal entry zone
            score += 10
            
        return max(0, min(100, score))

# ==========================================
# 3. DATA ENGINE (FTSE UNIVERSE)
# ==========================================

@st.cache_data
def get_lse_universe():
    """Curated list of Liquid LSE Stocks."""
    return [
        "RR.L", "SHEL.L", "BP.L", "AZN.L", "HSBA.L", "LLOY.L", "BARC.L", "GLEN.L", 
        "RIO.L", "ULVR.L", "GSK.L", "DGE.L", "BATS.L", "NG.L", "VOD.L", "TSCO.L", 
        "LSEG.L", "REL.L", "PRU.L", "AAL.L", "CRDA.L", "MKS.L", "NXT.L", "JD.L", 
        "WTB.L", "LAND.L", "BLND.L", "UU.L", "SVT.L", "CPG.L", "BA.L", "IMB.L",
        "TW.L", "PSN.L", "BDEV.L", "SBRY.L", "KGF.L", "EXPN.L", "SGE.L", "SMIN.L",
        "AUTO.L", "RMV.L", "ENT.L", "IAG.L", "EZJ.L", "WIZZ.L", "LGEN.L", "AV.L"
    ]

@st.cache_data(ttl=3600)
def scan_market(tickers):
    """
    Scans LSE using the 50D/50W logic.
    """
    data_list = []
    
    # Need 2+ years for 50W SMA calculation (50 weeks ~= 1 year, plus buffer)
    raw_data = yf.download(tickers, period="2y", interval="1d", group_by='ticker', progress=False)
    
    for ticker in tickers:
        try:
            df = raw_data[ticker].copy()
            if df.empty or len(df) < 300: continue
            
            df = df.dropna(subset=['Close'])
            
            # --- DARKPOOL ENGINE ---
            # 1. Indicators
            df['RSI'] = DarkPoolMath.calculate_rsi(df['Close'], 14)
            df['ADX'] = DarkPoolMath.calculate_adx(df, 14)
            
            # 2. Calculate 50D and 50W Anchors
            df = DarkPoolMath.get_dual_anchors(df)
            
            # 3. Get Latest State
            last = df.iloc[-1]
            
            # Skip if 50W hasn't calculated yet (NaN)
            if pd.isna(last['SMA_50W']): continue

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
            # Bullish Alignment: 50D > 50W
            if info['SMA_50D'] > info['SMA_50W']:
                if info['SMA_50W'] < info['Close'] < info['SMA_50D']:
                    info['Status'] = "VALUE ZONE 💎" # Inside the cloud
                elif info['Close'] > info['SMA_50D']:
                    info['Status'] = "MOMENTUM 🚀" # Above both
                else:
                    info['Status'] = "CAUTION ⚠️" # Below 50W
            else:
                info['Status'] = "BEARISH 🐻" # 50D < 50W
                
            data_list.append(info)
            
        except Exception as e:
            continue
            
    return pd.DataFrame(data_list)

# ==========================================
# 4. UI VISUALIZATION
# ==========================================

def render_chart(ticker):
    """Plots the 50D/50W Cloud Chart."""
    with st.spinner(f"Rendering 50/50 Analysis for {ticker}..."):
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y", interval="1d")
        
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
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50W'], line=dict(color='#2962FF', width=2), name='50-Week SMA'), row=1, col=1)
        
        # Cloud Fill (Between 50D and 50W)
        # Visualizes the "Value Zone"
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(df.index), pd.Series(df.index)[::-1]]),
            y=pd.concat([df['SMA_50D'], df['SMA_50W'][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 230, 118, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Value Cloud',
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
            title=f"Titan Value Structure: {ticker} (50D vs 50W)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. MAIN APP
# ==========================================

def main():
    # Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🇬🇧 TITAN LSE VALUE HUNTER")
        st.caption("Strategy: 50-Day SMA vs 50-Week SMA Convergence")
    with c2:
        if st.button("🔄 REFRESH SCAN"):
            st.cache_data.clear()
            st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ STRATEGY SETTINGS")
        st.info("Core Logic: Finding price interacting with the 50D/50W Cloud.")
        
        min_score = st.slider("Min Titan Score", 0, 100, 70, help="Higher = Better Technical Value Setup")
        
        st.subheader("Filters")
        only_cloud_retest = st.checkbox("Value Zone Only", value=False, help="Show stocks strictly inside the 50D/50W Cloud")
        high_adx_only = st.checkbox("High Strength Only (ADX > 25)", value=True)

    # Main Logic
    universe = get_lse_universe()
    
    with st.spinner(f"Calculating 50D/50W Models for {len(universe)} tickers..."):
        df = scan_market(universe)
        
        # Apply Sidebar Filters
        if high_adx_only:
            df = df[df['ADX'] > 25]
        
        if only_cloud_retest:
            # Value Zone = Price between 50W and 50D
            df = df[ (df['Close'] > df['SMA_50W']) & (df['Close'] < df['SMA_50D']) ]
            
        df = df[df['Flow_Score'] >= min_score]
        df = df.sort_values(by="Flow_Score", ascending=False)

    # Display
    tab1, tab2 = st.tabs(["💎 VALUE HUNTER", "🔬 CHART INSPECTOR"])

    with tab1:
        if not df.empty:
            # Top Picks Cards
            st.markdown("### 🏛️ Top 50D/50W Opportunities")
            top_row = df.head(4)
            cols = st.columns(4)
            
            for i, (idx, row) in enumerate(top_row.iterrows()):
                with cols[i]:
                    st.metric(
                        label=row['Ticker'],
                        value=f"£{row['Close']:.2f}",
                        delta=f"{row['Status']}"
                    )
                    st.progress(int(row['Flow_Score']))
                    st.caption(f"50D: {row['SMA_50D']:.2f} | 50W: {row['SMA_50W']:.2f}")
            
            # Main Table
            st.markdown("### 📋 Full Scan Results")
            
            # Color Styling Function
            def highlight_status(val):
                if 'VALUE' in val: color = '#FFD700' # Gold
                elif 'MOMENTUM' in val: color = '#00E676' # Green
                else: color = '#FF5252' # Red
                return f'color: {color}; font-weight: bold'

            display_df = df[['Ticker', 'Close', 'Status', 'Flow_Score', 'SMA_50D', 'SMA_50W', 'RSI']].copy()
            
            st.dataframe(
                display_df.style.map(highlight_status, subset=['Status'])
                                .background_gradient(subset=['Flow_Score'], cmap='Greens')
                                .format({'SMA_50D': '{:.2f}', 'SMA_50W': '{:.2f}', 'RSI': '{:.1f}'}),
                use_container_width=True,
                height=600
            )
        else:
            st.warning("No assets match your current Value Filters. Try lowering the Score.")

    with tab2:
        st.markdown("### 🔭 Institutional Chart")
        if not df.empty:
            sel = st.selectbox("Select Asset", df['Ticker'].tolist())
            render_chart(sel)
            
            # Logic Explanation
            with st.expander("ℹ️ Understanding the 50/50 Strategy"):
                st.markdown("""
                **The Titan 50/50 System:**
                1.  **50-Week SMA (Blue Line):** The ultimate long-term baseline. Institutions defend this level.
                2.  **50-Day SMA (Green Line):** The medium-term institutional trend.
                3.  **Bullish Alignment:** We only look for buys when the **50-Day > 50-Week**.
                4.  **Value Zone (Cloud):** When price dips *below* the 50-Day but stays *above* the 50-Week, it is in the 'Value Zone'. This is often the best risk/reward entry.
                """)
        else:
            st.info("Run the scan to populate the inspector.")

if __name__ == "__main__":
    main()
