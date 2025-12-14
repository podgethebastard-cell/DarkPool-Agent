import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import io
from openai import OpenAI

# ==========================================
# 1. PAGE CONFIGURATION & DARKPOOL UI
# ==========================================
st.set_page_config(
    page_title="TITAN ARCHITECT",
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
        text-shadow: 0 0 10px rgba(0, 255, 187, 0.6);
        font-family: 'Roboto Mono', monospace;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Custom Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(0, 0, 0, 0.2));
        border: 1px solid rgba(0, 255, 187, 0.3);
        padding: 15px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #00ffbb;
        box-shadow: 0 0 20px rgba(0, 255, 187, 0.2);
        transform: translateY(-2px);
    }
    div[data-testid="stMetricValue"] {
        color: #00ffbb !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #888 !important;
        font-size: 0.9rem !important;
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
        background-color: #00ffbb !important;
        color: #000000 !important;
        border-color: #00ffbb !important;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1f2937;
        color: #00ffbb;
        border: 1px solid #00ffbb;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        border-radius: 6px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #00ffbb;
        color: #000;
        box-shadow: 0 0 15px rgba(0, 255, 187, 0.5);
    }
    
    /* Sidebar Inputs */
    .stTextInput>div>div>input {
        background-color: #161b22;
        color: #00ffbb;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DYNAMIC MATH ENGINE
# ==========================================

class TitanMath:
    @staticmethod
    def calculate_ma(series, length, ma_type="SMA"):
        """Dynamic Moving Average Calculator"""
        if ma_type == "SMA":
            return series.rolling(window=length).mean()
        elif ma_type == "EMA":
            return series.ewm(span=length, adjust=False).mean()
        elif ma_type == "WMA":
            weights = np.arange(1, length + 1)
            return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        return series.rolling(window=length).mean() # Default to SMA

    @staticmethod
    def calculate_atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=length).mean()

    @staticmethod
    def calculate_rsi(series, length=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_adx(df, length=14):
        up = df['High'].diff()
        down = -df['Low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = TitanMath.calculate_atr(df, length)
        # Avoid division by zero
        tr = tr.replace(0, np.nan)
        plus_di = 100 * (pd.Series(plus_dm).rolling(length).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(length).mean() / tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(length).mean()

    @staticmethod
    def get_anchors(df, config):
        """
        Calculates Fast and Slow Anchors based on user configuration.
        Supports Daily vs Daily OR Daily vs Weekly.
        """
        # Fast Anchor (Always on base timeframe, e.g., Daily)
        df['Fast_Anchor'] = TitanMath.calculate_ma(df['Close'], config['fast_len'], config['fast_type'])
        
        # Slow Anchor
        if config['use_weekly_anchor']:
            # Resample to Weekly, Calculate, Reindex
            weekly = df.resample('W-FRI').agg({'Close': 'last'})
            weekly['Slow_Anchor'] = TitanMath.calculate_ma(weekly['Close'], config['slow_len'], config['slow_type'])
            # Forward fill daily to match weekly steps
            daily_slow = weekly['Slow_Anchor'].reindex(df.index, method='ffill')
            df['Slow_Anchor'] = daily_slow
        else:
            # Calculate on Daily data
            df['Slow_Anchor'] = TitanMath.calculate_ma(df['Close'], config['slow_len'], config['slow_type'])
            
        return df

    @staticmethod
    def score_asset(row, config):
        """
        Dynamic Scoring Engine based on Strategy Mode.
        """
        score = 0
        price = row['Close']
        fast = row['Fast_Anchor']
        slow = row['Slow_Anchor']
        
        # 1. Trend Filter
        bull_trend = fast > slow
        
        if config['strategy_mode'] == "Value Hunter (Cloud)":
            # Strategy: Buy pullbacks into the cloud (Between Fast and Slow) in an Uptrend
            if not bull_trend: return 0
            
            if slow < price < fast:
                score += 50 # Prime Value Zone
            elif price > fast:
                # Check extension
                dist = (price - fast) / fast * 100
                if dist < 3.0: score += 30 # Near breakout
                else: score += 10 # Extended
            elif price < slow:
                score -= 20 # Trend Broken
                
        elif config['strategy_mode'] == "Trend Momentum":
            # Strategy: Buy Breakouts (Price > Fast > Slow)
            if not bull_trend: return 0
            if price > fast: score += 50
            if slow < price < fast: score += 20 # Warning, momentum fading
            
        elif config['strategy_mode'] == "Deep Value (Contrarian)":
            # Strategy: Buy when Price < Slow (Oversold logic)
            # Dangerous, so we rely heavily on RSI
            if price < slow: score += 40
            if price < fast: score += 10

        # 2. ADX Filter
        if row['ADX'] > config['min_adx']:
            score += 10
            
        # 3. RSI Filter (Dynamic based on mode)
        rsi = row['RSI']
        if config['strategy_mode'] == "Trend Momentum":
            if 50 < rsi < 75: score += 20 # Strong momentum
        else:
            if rsi < config['max_rsi_entry']: score += 20 # Not overbought
            
        return max(0, min(100, score))

# ==========================================
# 3. DATA ENGINE
# ==========================================

@st.cache_data
def get_default_universe():
    """Default Liquid LSE Stocks."""
    return [
        "RR.L", "SHEL.L", "BP.L", "AZN.L", "HSBA.L", "LLOY.L", "BARC.L", "GLEN.L", 
        "RIO.L", "ULVR.L", "GSK.L", "DGE.L", "BATS.L", "NG.L", "VOD.L", "TSCO.L", 
        "LSEG.L", "REL.L", "PRU.L", "AAL.L", "CRDA.L", "MKS.L", "NXT.L", "JD.L", 
        "WTB.L", "LAND.L", "BLND.L", "UU.L", "SVT.L", "CPG.L", "BA.L", "IMB.L",
        "TW.L", "PSN.L", "BDEV.L", "SBRY.L", "KGF.L", "EXPN.L", "SGE.L", "SMIN.L",
        "AUTO.L", "RMV.L", "ENT.L", "IAG.L", "EZJ.L", "WIZZ.L", "LGEN.L", "AV.L"
    ]

@st.cache_data(ttl=3600)
def scan_market(tickers, config):
    data_list = []
    # Download 2.5y to allow for weekly calc warmup
    raw_data = yf.download(tickers, period="30mo", interval="1d", group_by='ticker', progress=False)
    
    for ticker in tickers:
        try:
            # Handle Single Ticker vs Multi Ticker return structure
            if len(tickers) == 1:
                df = raw_data.copy()
            else:
                df = raw_data[ticker].copy()
                
            if df.empty or len(df) < 300: continue
            
            df = df.dropna(subset=['Close'])
            
            # --- ENGINE ---
            df['RSI'] = TitanMath.calculate_rsi(df['Close'], 14)
            df['ADX'] = TitanMath.calculate_adx(df, 14)
            df = TitanMath.get_anchors(df, config)
            
            last = df.iloc[-1]
            if pd.isna(last['Slow_Anchor']): continue

            info = {
                'Ticker': ticker,
                'Close': last['Close'],
                'Fast_Anchor': last['Fast_Anchor'],
                'Slow_Anchor': last['Slow_Anchor'],
                'RSI': last['RSI'],
                'ADX': last['ADX'],
                'Volume': last['Volume']
            }
            
            info['Score'] = TitanMath.score_asset(info, config)
            
            # Status Text
            fast, slow, price = info['Fast_Anchor'], info['Slow_Anchor'], info['Close']
            if fast > slow:
                if slow < price < fast: info['Status'] = "CLOUD VALUE 💎"
                elif price > fast: info['Status'] = "MOMENTUM 🚀"
                else: info['Status'] = "WEAK BULL ⚠️"
            else:
                info['Status'] = "BEAR TREND 🐻"
                
            data_list.append(info)
            
        except Exception:
            continue
            
    return pd.DataFrame(data_list)

# ==========================================
# 4. UI COMPONENTS
# ==========================================

def render_chart(ticker, config):
    with st.spinner(f"Architecting Charts for {ticker}..."):
        stock = yf.Ticker(ticker)
        df = stock.history(period="30mo", interval="1d")
        
        # Apply Config Logic to single chart
        df = TitanMath.get_anchors(df, config)
        df['RSI'] = TitanMath.calculate_rsi(df['Close'])
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # Candle
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        
        # Anchors
        f_name = f"Fast ({config['fast_len']} {config['fast_type']})"
        s_name = f"Slow ({config['slow_len']} {config['slow_type']}{' W' if config['use_weekly_anchor'] else ' D'})"
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Fast_Anchor'], line=dict(color='#00ffbb', width=2), name=f_name), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Slow_Anchor'], line=dict(color='#2962FF', width=2), name=s_name), row=1, col=1)
        
        # Cloud
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(df.index), pd.Series(df.index)[::-1]]),
            y=pd.concat([df['Fast_Anchor'], df['Slow_Anchor'][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 255, 187, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Value Cloud',
            showlegend=True
        ), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#e0e0e0', width=1), name='RSI'), row=2, col=1)
        fig.add_hline(y=config['max_rsi_entry'], line_dash="dot", line_color="#00ffbb", row=2, col=1, annotation_text="Max Entry")
        fig.add_hline(y=30, line_dash="dot", line_color="#FF5252", row=2, col=1)
        
        fig.update_layout(
            height=600,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False,
            title=f"Titan Architect View: {ticker}"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. MAIN APP
# ==========================================

def main():
    # --- HEADER ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🇬🇧 TITAN ARCHITECT v3.0")
        st.caption("FULLY CUSTOMIZABLE ALGORITHMIC SCANNER")
    with c2:
        if st.button("RUN SCANNER", type="primary", use_container_width=True):
            st.session_state['run_scan'] = True

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.header("🛠️ STRATEGY ENGINE")
        
        # 1. Strategy Mode
        strat_mode = st.selectbox("Strategy Mode", 
                                  ["Value Hunter (Cloud)", "Trend Momentum", "Deep Value (Contrarian)"],
                                  index=0)
        
        st.markdown("---")
        st.subheader("1. Anchor Settings")
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            fast_len = st.number_input("Fast Len", 10, 200, 50)
            fast_type = st.selectbox("Fast Type", ["SMA", "EMA", "WMA"], index=0)
        with c_s2:
            slow_len = st.number_input("Slow Len", 20, 200, 50)
            slow_type = st.selectbox("Slow Type", ["SMA", "EMA", "WMA"], index=0)
            
        use_weekly = st.checkbox("Use Weekly Data for Slow Anchor?", value=True, help="If checked, Slow Anchor calculates on Weekly candles (e.g. 50W SMA).")
        
        st.markdown("---")
        st.subheader("2. Filter Logic")
        min_score = st.slider("Min Titan Score", 0, 100, 60)
        min_adx = st.slider("Min ADX (Strength)", 0, 50, 20)
        max_rsi = st.slider("Max RSI for Entry", 30, 90, 60)
        
        st.markdown("---")
        st.subheader("3. Universe")
        univ_mode = st.radio("Target Universe", ["FTSE Default", "Custom List"])
        
        custom_tickers = []
        if univ_mode == "Custom List":
            raw_txt = st.text_area("Enter Tickers (comma separated)", "AAPL, MSFT, TSLA, BTC-USD")
            custom_tickers = [x.strip() for x in raw_txt.split(',')]

        # Save config dict
        config = {
            'strategy_mode': strat_mode,
            'fast_len': fast_len, 'fast_type': fast_type,
            'slow_len': slow_len, 'slow_type': slow_type,
            'use_weekly_anchor': use_weekly,
            'min_adx': min_adx, 'max_rsi_entry': max_rsi
        }

    # --- MAIN LOGIC ---
    if univ_mode == "FTSE Default":
        universe = get_default_universe()
    else:
        universe = custom_tickers

    if 'run_scan' not in st.session_state:
        st.info("👈 Configure your Strategy Logic in the Sidebar and hit 'RUN SCANNER'.")
        return

    # Run Scan
    with st.spinner(f"Architecting Market Data for {len(universe)} assets..."):
        df = scan_market(universe, config)
        
        # Filters
        df = df[df['Score'] >= min_score]
        df = df.sort_values(by="Score", ascending=False)

    # --- TABS ---
    t1, t2, t3, t4 = st.tabs(["📊 RESULTS MATRIX", "🔬 CHART LAB", "🧮 RISK CALC", "🤖 AI ANALYST"])

    with t1:
        if not df.empty:
            st.markdown(f"### 🎯 Identified {len(df)} Candidates")
            
            # Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results (CSV)", csv, "titan_architect_scan.csv", "text/csv")
            
            # Styling
            def color_status(val):
                if "VALUE" in val: color = "#FFD700"
                elif "MOMENTUM" in val: color = "#00ffbb"
                elif "BEAR" in val: color = "#FF5252"
                else: color = "white"
                return f'color: {color}; font-weight: bold'

            show_df = df[['Ticker', 'Close', 'Status', 'Score', 'RSI', 'ADX', 'Fast_Anchor', 'Slow_Anchor']].copy()
            st.dataframe(
                show_df.style.map(color_status, subset=['Status'])
                             .background_gradient(subset=['Score'], cmap='Greens')
                             .format({'Close': '{:.2f}', 'RSI': '{:.1f}', 'ADX': '{:.1f}', 'Fast_Anchor': '{:.2f}', 'Slow_Anchor': '{:.2f}'}),
                use_container_width=True,
                height=600
            )
        else:
            st.warning("No assets matched your custom criteria.")

    with t2:
        if not df.empty:
            c1, c2 = st.columns([1, 3])
            with c1:
                sel = st.selectbox("Select Asset to Inspect", df['Ticker'].tolist())
            with c2:
                row = df[df['Ticker'] == sel].iloc[0]
                st.info(f"**Current Status:** {row['Status']} | **Score:** {row['Score']:.0f}")
            
            render_chart(sel, config)
        else:
            st.write("Run scan to populate.")

    with t3:
        st.markdown("### 🛡️ Position Size Architect")
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            acct_size = st.number_input("Account Size (£)", 1000, 1000000, 10000)
            risk_pct = st.number_input("Risk per Trade (%)", 0.1, 5.0, 1.0)
        with c_r2:
            entry_p = st.number_input("Entry Price", 0.0, 10000.0, 100.0)
            stop_p = st.number_input("Stop Loss Price", 0.0, 10000.0, 95.0)
        
        if entry_p > stop_p:
            risk_amt = acct_size * (risk_pct / 100)
            risk_per_share = entry_p - stop_p
            shares = risk_amt / risk_per_share
            total_cost = shares * entry_p
            
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Shares to Buy", f"{int(shares)}")
            m2.metric("Total Position Value", f"£{total_cost:,.2f}")
            m3.metric("Risk Amount", f"£{risk_amt:.2f}")
        else:
            st.warning("Stop Loss must be below Entry Price for Long positions.")

    with t4:
        st.markdown("### 🤖 Titan AI Analyst")
        api_key = st.text_input("OpenAI API Key (Optional)", type="password")
        
        if st.button("Generate Strategy Report"):
            if not df.empty and api_key:
                sel_ai = df.iloc[0] # Analyze top pick
                client = OpenAI(api_key=api_key)
                
                prompt = f"""
                Analyze {sel_ai['Ticker']} based on Titan Architect Data.
                Price: {sel_ai['Close']}. 
                Strategy Mode: {config['strategy_mode']}.
                Fast Anchor: {sel_ai['Fast_Anchor']}. Slow Anchor: {sel_ai['Slow_Anchor']}.
                RSI: {sel_ai['RSI']}. ADX: {sel_ai['ADX']}.
                Titan Score: {sel_ai['Score']}. Status: {sel_ai['Status']}.
                
                Provide a professional trade thesis (Bullish/Bearish), key risks, and whether the technicals support the strategy mode selected.
                """
                
                with st.spinner("AI Architect is thinking..."):
                    res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":prompt}])
                    st.markdown(res.choices[0].message.content)
            else:
                st.error("Need Scan Results and API Key.")

if __name__ == "__main__":
    main()
