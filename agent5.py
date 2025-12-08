import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import openai

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Macro Insighter",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Excellent UI"
st.markdown("""
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #4CAF50;
        transform: scale(1.02);
    }
    /* Chart Container */
    div[data-testid="stPlotlyChart"] {
        background-color: #0E1117;
        border-radius: 8px;
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA MAPPING (The Brain)
# -----------------------------------------------------------------------------

# Standard Tickers
TICKERS = {
    "15. MASTER CORE (The Non-Negotiables)": {
        "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Dollar Index (DXY)": "DX-Y.NYB",
        "US 10Y Yield": "^TNX", "US 02Y Yield": "^IRX", "VIX (Fear)": "^VIX",
        "Crude Oil (WTI)": "CL=F", "Gold": "GC=F", "Copper": "HG=F",
        "High Yield Corp (HYG)": "HYG", "Long Bonds (TLT)": "TLT",
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Global Liquidity (W M2)": "Wwow.L"
    },
    "1. Global Equity Indices": {
        "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Dow Jones": "^DJI", "Russell 2000": "^RUT",
        "DAX (Germany)": "^GDAXI", "FTSE 100 (UK)": "^FTSE", "CAC 40 (France)": "^FCHI",
        "Nikkei 225 (Japan)": "^N225", "Hang Seng (HK)": "^HSI", "KOSPI (Korea)": "^KS11",
        "All-World (ACWI)": "ACWI", "Emerging Markets (EEM)": "EEM"
    },
    "2. Volatility & Fear": {
        "VIX (S&P 500)": "^VIX", "VXN (Nasdaq)": "^VXN", "VXD (Dow)": "^VXD",
        "MOVE Index (Proxy)": "MOVE.MX" 
    },
    "3. Interest Rates & Bonds": {
        "US 10Y Yield": "^TNX", "US 2Y Yield": "^IRX", "US 30Y Yield": "^TYX",
        "TLT (20Y+ Bond ETF)": "TLT", "IEF (7-10Y Bond ETF)": "IEF",
        "SHY (1-3Y Bond ETF)": "SHY", "LQD (Inv Grade)": "LQD",
        "HYG (High Yield)": "HYG", "TIP (Inflation Prot)": "TIP"
    },
    "4. Currencies (FX)": {
        "Dollar Index": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X", "USD/CNY": "USDCNY=X", "USD/MXN": "USDMXN=X"
    },
    "7. Commodities": {
        "WTI Crude": "CL=F", "Brent Crude": "BZ=F", "Natural Gas": "NG=F",
        "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F",
        "Corn": "ZC=F", "Wheat": "KE=F"
    },
    "11. Crypto Macro": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD",
        "BNB": "BNB-USD", "XRP": "XRP-USD"
    },
    "12. Risk Rotation Signals": {
        "Stocks vs Bonds (SPY/TLT)": "SPY", # Calculated in logic
        "High Yield vs Safe (HYG/TLT)": "HYG",
        "Copper vs Gold": "HG=F"
    },
    "14. Geopolitics & Tail Risk": {
        "Defense (ITA)": "ITA", "Cybersecurity (HACK)": "HACK",
        "Energy (XLE)": "XLE", "Gold": "GC=F"
    }
}

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_market_data(ticker_dict, period="1y"): # Increased period for better charts
    """Fetches data for a dictionary of tickers."""
    symbols = list(ticker_dict.values())
    # Add ingredients for calculated ratios if needed
    symbols.extend(["SI=F", "GC=F", "BTC-USD", "^GSPC", "SPY", "TLT"]) 
    
    try:
        data = yf.download(symbols, period=period, progress=False)['Close']
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_metrics(df, ticker_symbol):
    """Calculates latest price and percentage change."""
    if df.empty or ticker_symbol not in df.columns:
        return None, None, None, None # FIX: Return 4 values
    
    series = df[ticker_symbol].dropna()
    if len(series) < 2:
        return None, None, None, None # FIX: Return 4 values
    
    latest_price = series.iloc[-1]
    prev_price = series.iloc[-2]
    change = latest_price - prev_price
    pct_change = (change / prev_price) * 100
    
    return latest_price, change, pct_change, series

def plot_mini_chart(series, color):
    """Creates a sparkline chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f"rgba({','.join([str(c) for c in color_to_rgb(color)])}, 0.1)"
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=60,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def color_to_rgb(color_hex):
    if color_hex == '#00FF00': return (0, 255, 0)
    return (255, 0, 0)

def generate_ai_analysis(market_text_summary):
    """Calls OpenAI API to analyze the market data."""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            return "⚠️ OpenAI API Key not found in st.secrets."
        
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        You are an institutional macro strategist (ex-Bridgewater/Blackrock). 
        Analyze the following market snapshot and provide a concise 'Risk-On' vs 'Risk-Off' assessment.
        Highlight anomalies in yields, crypto, or volatility.
        
        DATA SNAPSHOT:
        {market_text_summary}
        
        Output Format:
        **Market Regime:** [Risk-On / Risk-Off / Neutral]
        **Key Driver:** [Text]
        **Actionable Insight:** [Text]
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating AI report: {str(e)}"

# -----------------------------------------------------------------------------
# 4. MAIN APP LAYOUT
# -----------------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.header("🦅 Macro Monitor")
    st.caption("Institutional Risk Dashboard")
    
    selected_category = st.radio("Select Sector:", list(TICKERS.keys()) + ["🔥 SPECIAL: Custom Ratios", "💰 SPECIAL: Crypto Total"])
    
    st.markdown("---")
    
    # AI Report Button
    st.subheader("🤖 AI Analyst")
    if st.button("Generate Daily Report"):
        st.session_state['run_ai'] = True
    
    st.markdown("---")
    st.caption(f"Last Refresh: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("Refresh Data"):
        st.cache_data.clear()

# Main Header
st.title(f"{selected_category}")

# Fetch Data
with st.spinner('Accessing Global Markets...'):
    # We fetch ALL data to handle cross-sector ratios
    all_tickers = {}
    for cat in TICKERS:
        all_tickers.update(TICKERS[cat])
    
    market_data = get_market_data(all_tickers)

# -----------------------------------------------------------------------------
# 5. RENDER LOGIC
# -----------------------------------------------------------------------------

# A. STANDARD GRID SECTORS
if selected_category in TICKERS:
    # Logic to handle grid
    cols = st.columns(3)
    row_data_summary = [] # Collecting text for AI
    
    for i, (name, ticker) in enumerate(TICKERS[selected_category].items()):
        col_idx = i % 3
        
        latest, change, pct, series = calculate_metrics(market_data, ticker)
        
        if latest is not None:
            # Store for AI
            row_data_summary.append(f"{name}: {latest:.2f} ({pct:.2f}%)")
            
            with cols[col_idx]:
                color = "#00FF00" if change >= 0 else "#FF4B4B"
                with st.container():
                    # FIX: Added unique key using name and category
                    st.metric(
                        label=name,
                        value=f"{latest:,.2f}",
                        delta=f"{pct:,.2f}%"
                    )
                    st.plotly_chart(
                        plot_mini_chart(series, color), 
                        use_container_width=True, 
                        config={'displayModeBar': False},
                        key=f"chart_{name}_{i}" # FIX: Unique ID
                    )
        else:
            with cols[col_idx]:
                st.warning(f"{name}: Data Unavailable")

# B. SPECIAL: RATIOS (Gold/Silver, BTC/SPY)
elif selected_category == "🔥 SPECIAL: Custom Ratios":
    st.markdown("### ⚖️ Inter-Market Relationships")
    
    ratios = {
        "Gold / Silver Ratio": ("GC=F", "SI=F"),
        "BTC / Gold Ratio": ("BTC-USD", "GC=F"),
        "BTC / SPY Ratio": ("BTC-USD", "SPY"),
        "Copper / Gold Ratio": ("HG=F", "GC=F")
    }
    
    cols = st.columns(2)
    for i, (name, (num, den)) in enumerate(ratios.items()):
        if num in market_data and den in market_data:
            s_num = market_data[num].dropna()
            s_den = market_data[den].dropna()
            
            # Align dates
            common_idx = s_num.index.intersection(s_den.index)
            ratio_series = s_num.loc[common_idx] / s_den.loc[common_idx]
            
            latest = ratio_series.iloc[-1]
            prev = ratio_series.iloc[-2]
            pct = ((latest - prev) / prev) * 100
            
            with cols[i % 2]:
                st.metric(label=name, value=f"{latest:.4f}", delta=f"{pct:.2f}%")
                st.plotly_chart(plot_mini_chart(ratio_series, "#3498db"), use_container_width=True, key=f"ratio_{i}")

# C. SPECIAL: TOTAL CRYPTO PROXY
elif selected_category == "💰 SPECIAL: Crypto Total":
    st.info("ℹ️ Proxies calculated using Sum of Top 5 Assets (BTC, ETH, SOL, BNB, XRP) as Yahoo lacks a 'TOTAL' index.")
    
    # Calculate Total Cap Proxy
    components = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
    valid_comps = [c for c in components if c in market_data]
    
    if valid_comps:
        df_crypto = market_data[valid_comps].dropna()
        total_series = df_crypto.sum(axis=1)
        total_ex_btc_eth = df_crypto[["SOL-USD", "BNB-USD", "XRP-USD"]].sum(axis=1) if "SOL-USD" in df_crypto else pd.Series()

        c1, c2 = st.columns(2)
        
        # Chart 1: TOTAL PROXY
        with c1:
            latest = total_series.iloc[-1]
            pct = ((latest - total_series.iloc[-2])/total_series.iloc[-2])*100
            st.metric("Total Crypto Index (Proxy)", f"${latest:,.0f}", f"{pct:.2f}%")
            st.plotly_chart(plot_mini_chart(total_series, "#9b59b6"), use_container_width=True, key="total_crypto")

        # Chart 2: TOTAL 3 (Altcoins)
        with c2:
            if not total_ex_btc_eth.empty:
                latest_alt = total_ex_btc_eth.iloc[-1]
                pct_alt = ((latest_alt - total_ex_btc_eth.iloc[-2])/total_ex_btc_eth.iloc[-2])*100
                st.metric("Total 3 (Altcoin Proxy)", f"${latest_alt:,.0f}", f"{pct_alt:.2f}%")
                st.plotly_chart(plot_mini_chart(total_ex_btc_eth, "#e74c3c"), use_container_width=True, key="total_3")

# -----------------------------------------------------------------------------
# 6. AI ANALYSIS OUTPUT
# -----------------------------------------------------------------------------
if st.session_state.get('run_ai'):
    st.markdown("### 🧠 AI Analyst Insight")
    if 'row_data_summary' in locals() and row_data_summary:
        data_text = ", ".join(row_data_summary)
        with st.spinner("Analyzing market structure..."):
            insight = generate_ai_analysis(data_text)
            st.success("Analysis Complete")
            st.markdown(insight)
    else:
        st.warning("Please select a standard sector (like Global Equity) to analyze first.")
    st.session_state['run_ai'] = False
