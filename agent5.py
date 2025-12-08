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
    page_title="Macro Insighter Pro",
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
        border-color: #4CAF50; /* Green highlight on hover */
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    /* Chart Container */
    div[data-testid="stPlotlyChart"] {
        background-color: #0E1117;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    /* Section Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        color: #E0E0E0;
    }
    /* Radio Button Styling (to replace dropdown) */
    .stRadio > label {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA MAPPING (The Brain)
# -----------------------------------------------------------------------------

# A. SINGLE ASSETS (Original Request)
TICKERS = {
    "15. MASTER CORE": {
        "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "DXY": "DX-Y.NYB",
        "US 10Y": "^TNX", "US 02Y": "^IRX", "VIX": "^VIX",
        "WTI Crude": "CL=F", "Gold": "GC=F", "Copper": "HG=F",
        "HYG (Junk)": "HYG", "TLT (Long Bond)": "TLT",
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"
    },
    "1. Global Equity Indices": {
        "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Dow Jones": "^DJI", "Russell 2000": "^RUT",
        "DAX (DE)": "^GDAXI", "FTSE (UK)": "^FTSE", "CAC (FR)": "^FCHI", "STOXX50": "^STOXX50E",
        "Nikkei (JP)": "^N225", "Hang Seng (HK)": "^HSI", "Shanghai": "000001.SS", "KOSPI": "^KS11",
        "ACWI": "ACWI", "VT (World)": "VT", "EEM (Emerging)": "EEM"
    },
    "2. Volatility & Fear": {
        "VIX": "^VIX", "VXN (Nasdaq)": "^VXN", "VXD (Dow)": "^VXD",
        "MOVE Proxy (ICE BofA)": "MOVE.MX" 
    },
    "3. Interest Rates": {
        "US 10Y": "^TNX", "US 02Y": "^IRX", "US 30Y": "^TYX", "US 05Y": "^FVX",
        "TLT": "TLT", "IEF": "IEF", "SHY": "SHY", "LQD": "LQD", "HYG": "HYG", "TIP": "TIP"
    },
    "4. Currencies": {
        "DXY": "DX-Y.NYB", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X", "USD/CNY": "USDCNY=X", "AUD/USD": "AUDUSD=X",
        "USD/CHF": "USDCHF=X", "USD/MXN": "USDMXN=X"
    },
    "7. Commodities": {
        "WTI": "CL=F", "Brent": "BZ=F", "NatGas": "NG=F",
        "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F",
        "Copper": "HG=F", "Wheat": "KE=F", "Corn": "ZC=F", "Soybeans": "ZS=F"
    },
    "8. Real Estate": {
        "VNQ (US REITs)": "VNQ", "REET (Global)": "REET", "XLRE": "XLRE"
    },
    "11. Crypto Macro": {
        "BTC.D (Proxy)": "BTC-USD", "Total Cap (Proxy)": "BTC-USD", # Placeholders, handled by special logic
        "BTC": "BTC-USD", "ETH": "ETH-USD"
    }
}

# B. RATIO DEFINITIONS (The New Institutional Request)
# Format: "Label": ("Numerator_Ticker", "Denominator_Ticker")
RATIO_GROUPS = {
    "✅ CRYPTO RELATIVE STRENGTH": {
        "BTC / ETH (Risk Appetite)": ("BTC-USD", "ETH-USD"),
        "BTC / SPX (Adoption)": ("BTC-USD", "^GSPC"),
        "BTC / NDX (Tech Corr)": ("BTC-USD", "^NDX"),
        "ETH / SPX": ("ETH-USD", "^GSPC"),
        "ETH / NDX": ("ETH-USD", "^NDX"),
        "BTC / DXY (Liquidity)": ("BTC-USD", "DX-Y.NYB"),
        "BTC / US10Y (Yields)": ("BTC-USD", "^TNX"),
        "BTC / VIX (Vol)": ("BTC-USD", "^VIX"),
        "BTC / Gold (Hard Money)": ("BTC-USD", "GC=F")
    },
    "✅ CRYPTO DOMINANCE (Calculated)": {
        "TOTAL 3 / TOTAL": ("SPECIAL_TOTAL3", "SPECIAL_TOTAL"),
        "TOTAL 2 / TOTAL": ("SPECIAL_TOTAL2", "SPECIAL_TOTAL"),
        "BTC.D (BTC/Total)": ("BTC-USD", "SPECIAL_TOTAL"),
        "ETH.D (ETH/Total)": ("ETH-USD", "SPECIAL_TOTAL"),
        "USDT.D (Tether/Total)": ("USDT-USD", "SPECIAL_TOTAL")
    },
    "✅ EQUITY RISK ROTATION": {
        "SPY / TLT (Risk On/Off)": ("SPY", "TLT"),
        "QQQ / IEF (Growth/Rates)": ("QQQ", "IEF"),
        "XLF / XLU (Fin/Util)": ("XLF", "XLU"),
        "XLY / XLP (Disc/Staples)": ("XLY", "XLP"),
        "IWM / SPY (Small/Large)": ("IWM", "SPY"),
        "EEM / SPY (Emerging/US)": ("EEM", "SPY"),
        "HYG / TLT (Credit/Safe)": ("HYG", "TLT"),
        "JNK / TLT": ("JNK", "TLT"),
        "KRE / XLF (Regional/Big)": ("KRE", "XLF"),
        "SMH / SPY (Semi Lead)": ("SMH", "SPY")
    },
    "✅ BOND & YIELD POWER": {
        "10Y / 2Y (Curve)": ("^TNX", "^IRX"),
        "10Y / 3M (Recession)": ("^TNX", "^IRX"), # Using IRX as short rate proxy
        "TLT / SHY (Duration)": ("TLT", "SHY"),
        "TLT / SPY (Safety/Risk)": ("TLT", "SPY"),
        "IEF / SHY": ("IEF", "SHY"),
        "MOVE / VIX (Stress)": ("MOVE.MX", "^VIX")
    },
    "✅ DOLLAR & LIQUIDITY": {
        "DXY / Gold": ("DX-Y.NYB", "GC=F"),
        "DXY / Oil": ("DX-Y.NYB", "CL=F"),
        "EURUSD / DXY": ("EURUSD=X", "DX-Y.NYB"),
        "USDJPY / DXY": ("USDJPY=X", "DX-Y.NYB"),
        "EEM / DXY": ("EEM", "DX-Y.NYB"),
    },
    "✅ COMMODITIES & INFLATION": {
        "Gold / Silver": ("GC=F", "SI=F"),
        "Copper / Gold": ("HG=F", "GC=F"),
        "Oil / Gold": ("CL=F", "GC=F"),
        "Oil / Copper": ("CL=F", "HG=F"),
        "Brent / WTI": ("BZ=F", "CL=F")
    },
    "✅ EQUITIES vs REAL ASSETS": {
        "SPX / Gold": ("^GSPC", "GC=F"),
        "SPX / Copper": ("^GSPC", "HG=F"),
        "SPX / Oil": ("^GSPC", "CL=F"),
        "VNQ / SPY (RE/Stocks)": ("VNQ", "SPY"),
        "XLE / SPX (Energy/Mkt)": ("XLE", "^GSPC")
    },
    "✅ TRADE & MACRO STRESS": {
        "XLI / SPX (Ind/Mkt)": ("XLI", "^GSPC"),
        "ITA / SPX (Defense/Mkt)": ("ITA", "^GSPC"),
        "HYG / JNK (Quality Junk)": ("HYG", "JNK")
    }
}

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data(ttl=120) # Cache for 2 mins
def get_market_data(tickers_list, period="1y"):
    """Fetches data for a list of tickers safely."""
    # Filter out special keywords
    valid_tickers = [t for t in tickers_list if not t.startswith("SPECIAL_")]
    if not valid_tickers:
        return pd.DataFrame()
    
    try:
        data = yf.download(valid_tickers, period=period, progress=False)['Close']
        return data
    except Exception as e:
        return pd.DataFrame()

def get_crypto_total_proxy(data_df):
    """
    Constructs synthetic TOTAL, TOTAL2, TOTAL3 indices from available components.
    Yahoo does not have 'TOTAL' ticker, so we sum the majors.
    """
    # Components to build totals
    coins = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD", "AVAX-USD", "LINK-USD"]
    
    # Ensure they exist in df
    available = [c for c in coins if c in data_df.columns]
    if not available:
        return None, None, None
        
    sub_df = data_df[available].fillna(method='ffill')
    
    # 1. TOTAL (Sum of all tracked majors)
    total = sub_df.sum(axis=1)
    
    # 2. TOTAL 2 (Ex-BTC)
    ex_btc = [c for c in available if c != "BTC-USD"]
    total2 = sub_df[ex_btc].sum(axis=1) if ex_btc else pd.Series()
    
    # 3. TOTAL 3 (Ex-BTC & ETH)
    ex_btc_eth = [c for c in available if c not in ["BTC-USD", "ETH-USD"]]
    total3 = sub_df[ex_btc_eth].sum(axis=1) if ex_btc_eth else pd.Series()
    
    return total, total2, total3

def calculate_change(series):
    if series is None or len(series) < 2:
        return None, None
    latest = series.iloc[-1]
    prev = series.iloc[-2]
    if prev == 0: return latest, 0
    pct = ((latest - prev) / prev) * 100
    return latest, pct

def plot_sparkline(series, color):
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
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=50, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def color_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_ai_analysis(summary_text):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key: return "⚠️ Missing OpenAI API Key."
        client = openai.OpenAI(api_key=api_key)
        prompt = f"""
        Act as a Global Macro Strategist. Analyze these key market ratios:
        {summary_text}
        
        Identify:
        1. Is the regime Risk-On or Risk-Off? (Check SPY/TLT, AUD/JPY)
        2. Are we seeing Inflation or Deflation? (Check Copper/Gold, Tips)
        3. Crypto Specific Outlook (BTC/SPX, Dominance)
        
        Be concise, bullet points.
        """
        resp = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": prompt}])
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# -----------------------------------------------------------------------------
# 4. MAIN APP LOGIC
# -----------------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.title("🦅 Macro Insighter")
    st.caption("Institutional Dashboard")
    
    mode = st.radio("Select View Mode:", ["Standard Tickers", "Institutional Ratios"])
    
    st.markdown("---")
    
    # MODIFICATION: Using st.radio instead of st.selectbox to prevent "text editing" / typing
    if mode == "Standard Tickers":
        selected_category = st.radio("Asset Class", list(TICKERS.keys()))
    else:
        selected_category = st.radio("Ratio Strategy", list(RATIO_GROUPS.keys()))
        
    st.markdown("---")
    if st.button("Generate AI Report"):
        st.session_state['run_ai'] = True
    if st.button("Refresh Data"):
        st.cache_data.clear()

# Prepare List of All Tickers Needed
all_needed_tickers = set()
for cat in TICKERS.values():
    all_needed_tickers.update(cat.values())
for cat in RATIO_GROUPS.values():
    for num, den in cat.values():
        all_needed_tickers.add(num)
        all_needed_tickers.add(den)

# Add Crypto Components for synthetic indices
crypto_components = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "USDT-USD", "USDC-USD"]
all_needed_tickers.update(crypto_components)

# FETCH DATA ONCE
with st.spinner("Fetching Global Market Data..."):
    market_data = get_market_data(list(all_needed_tickers))
    
    # Generate Synthetic Indices
    syn_total, syn_total2, syn_total3 = get_crypto_total_proxy(market_data)

# HEADER
st.title(f"{selected_category}")
st.markdown("---")

# RENDER CONTENT
data_summary_for_ai = []

if mode == "Standard Tickers":
    # Standard Grid
    items = TICKERS[selected_category]
    cols = st.columns(3)
    for i, (label, ticker) in enumerate(items.items()):
        col_idx = i % 3
        
        if ticker in market_data.columns:
            series = market_data[ticker].dropna()
            val, pct = calculate_change(series)
            
            if val is not None:
                data_summary_for_ai.append(f"{label}: {pct:.2f}%")
                with cols[col_idx]:
                    color = "#00FF00" if pct >= 0 else "#FF4B4B"
                    st.metric(label, f"{val:,.2f}", f"{pct:.2f}%")
                    st.plotly_chart(plot_sparkline(series, color), use_container_width=True, config={'displayModeBar': False}, key=f"std_{i}")
        else:
            with cols[col_idx]:
                st.warning(f"{label}: No Data")

else: # RATIO MODE
    # Ratio Grid
    items = RATIO_GROUPS[selected_category]
    cols = st.columns(3)
    
    for i, (label, (num_t, den_t)) in enumerate(items.items()):
        col_idx = i % 3
        
        # Resolve Numerator
        series_num = None
        if num_t == "SPECIAL_TOTAL": series_num = syn_total
        elif num_t == "SPECIAL_TOTAL2": series_num = syn_total2
        elif num_t == "SPECIAL_TOTAL3": series_num = syn_total3
        elif num_t in market_data: series_num = market_data[num_t]
        
        # Resolve Denominator
        series_den = None
        if den_t == "SPECIAL_TOTAL": series_den = syn_total
        elif den_t in market_data: series_den = market_data[den_t]
        
        # Calculate Ratio
        if series_num is not None and series_den is not None:
            # Align Indices
            common_idx = series_num.index.intersection(series_den.index)
            if not common_idx.empty:
                ratio_series = series_num.loc[common_idx] / series_den.loc[common_idx]
                val, pct = calculate_change(ratio_series)
                
                if val is not None:
                    data_summary_for_ai.append(f"{label}: {val:.4f} ({pct:.2f}%)")
                    with cols[col_idx]:
                        color = "#3498db" # Blue for ratios usually, or conditonal
                        if "Risk On" in label or "BTC/SPX" in label:
                            color = "#00FF00" if pct > 0 else "#FF4B4B"
                        
                        st.metric(label, f"{val:.4f}", f"{pct:.2f}%")
                        st.plotly_chart(plot_sparkline(ratio_series, color), use_container_width=True, config={'displayModeBar': False}, key=f"ratio_{i}")
        else:
            with cols[col_idx]:
                st.info(f"{label}: Insufficient Data")

# AI Report Section
if st.session_state.get('run_ai'):
    st.markdown("### 🧠 AI Institutional Analysis")
    if data_summary_for_ai:
        report = generate_ai_analysis("\n".join(data_summary_for_ai))
        st.success("Report Generated")
        st.markdown(report)
    else:
        st.error("No data available for analysis.")
    st.session_state['run_ai'] = False
