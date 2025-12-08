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
    page_title="Macro Titan Outlook",
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
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    /* Description Text */
    .metric-desc {
        font-size: 0.8rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 10px;
        font-style: italic;
    }
    /* Chart Container */
    div[data-testid="stPlotlyChart"] {
        background-color: #0E1117;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    /* Radio Button Styling */
    .stRadio > label {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA MAPPING (The Brain)
# -----------------------------------------------------------------------------

# Structure: "Label": ("Ticker", "Description")
TICKERS = {
    "    MASTER CORE": {
        "S&P 500": ("^GSPC", "US Large Cap Benchmark"),
        "Nasdaq 100": ("^NDX", "Tech & Growth Core"),
        "DXY": ("DX-Y.NYB", "Global Liquidity Engine"),
        "US 10Y": ("^TNX", "Global Asset Pricing Anchor"),
        "US 02Y": ("^IRX", "Fed Policy Sensitivity"),
        "VIX": ("^VIX", "Fear & Volatility Index"),
        "WTI Crude": ("CL=F", "Industrial Energy Demand"),
        "Gold": ("GC=F", "Real Money / Inflation Hedge"),
        "Copper": ("HG=F", "Global Growth Proxy (Dr. Copper)"),
        "HYG (Junk)": ("HYG", "Credit Risk Appetite"),
        "TLT (Long Bond)": ("TLT", "Duration / Recession Hedge"),
        "Bitcoin": ("BTC-USD", "Digital Liquidity Sponge"),
        "Ethereum": ("ETH-USD", "Web3 / Tech Platform Risk")
    },
    "   Global Equity Indices": {
        "S&P 500": ("^GSPC", "US Risk-On Core"),
        "Nasdaq 100": ("^NDX", "US Tech/Growth"),
        "Dow Jones": ("^DJI", "US Industrial/Value"),
        "Russell 2000": ("^RUT", "US Small Caps / Domestic Econ"),
        "DAX (DE)": ("^GDAXI", "Europe Industrial Core"),
        "FTSE (UK)": ("^FTSE", "UK/Global Banks & Energy"),
        "CAC (FR)": ("^FCHI", "French Luxury/Consumer"),
        "STOXX50": ("^STOXX50E", "Eurozone Blue Chips"),
        "Nikkei (JP)": ("^N225", "Japan Exporters / YCC Play"),
        "Hang Seng (HK)": ("^HSI", "China Tech / Real Estate"),
        "Shanghai": ("000001.SS", "China Mainland Economy"),
        "KOSPI": ("^KS11", "Korean Tech / Chips"),
        "ACWI": ("ACWI", "All Country World Index"),
        "VT (World)": ("VT", "Total World Stock Market"),
        "EEM (Emerging)": ("EEM", "Emerging Markets Risk")
    },
    "   Volatility & Fear": {
        "VIX": ("^VIX", "S&P 500 Implied Volatility"),
        "VXN (Nasdaq)": ("^VXN", "Tech Sector Volatility"),
        "VXD (Dow)": ("^VXD", "Industrial Volatility"),
        "MOVE Proxy (ICE BofA)": ("MOVE.MX", "Bond Market Volatility (Stress)")
    },
    "   Interest Rates": {
        "US 10Y": ("^TNX", "Benchmark Long Rate"),
        "US 02Y": ("^IRX", "Fed Policy Expectations"),
        "US 30Y": ("^TYX", "Long Duration / Inflation Exp"),
        "US 05Y": ("^FVX", "Medium Term Rates"),
        "TLT": ("TLT", "20Y+ Treasury Bond ETF"),
        "IEF": ("IEF", "7-10Y Treasury Bond ETF"),
        "SHY": ("SHY", "1-3Y Short Duration Cash"),
        "LQD": ("LQD", "Investment Grade Corporate"),
        "HYG": ("HYG", "High Yield Junk Bonds"),
        "TIP": ("TIP", "Inflation Protected Securities")
    },
    "   Currencies": {
        "DXY": ("DX-Y.NYB", "US Dollar vs Major Peers"),
        "EUR/USD": ("EURUSD=X", "Euro Strength"),
        "GBP/USD": ("GBPUSD=X", "British Pound / Risk"),
        "USD/JPY": ("USDJPY=X", "Yen Carry Trade Key"),
        "USD/CNY": ("USDCNY=X", "Yuan / China Export Strength"),
        "AUD/USD": ("AUDUSD=X", "Commodity Currency Proxy"),
        "USD/CHF": ("USDCHF=X", "Swiss Franc Safe Haven"),
        "USD/MXN": ("USDMXN=X", "Emerging Mkt Risk Gauge")
    },
    "   Commodities": {
        "WTI": ("CL=F", "US Crude Oil"),
        "Brent": ("BZ=F", "Global Sea-Borne Oil"),
        "NatGas": ("NG=F", "US Heating/Industrial Energy"),
        "Gold": ("GC=F", "Safe Haven / Monetary Metal"),
        "Silver": ("SI=F", "Industrial + Monetary Metal"),
        "Platinum": ("PL=F", "Auto Catalyst / Industrial"),
        "Palladium": ("PA=F", "Tech / Industrial Metal"),
        "Copper": ("HG=F", "Construction / Econ Growth"),
        "Wheat": ("KE=F", "Global Food Supply"),
        "Corn": ("ZC=F", "Feed / Energy / Food"),
        "Soybeans": ("ZS=F", "Global Ag Export Demand")
    },
    "   Real Estate": {
        "VNQ (US REITs)": ("VNQ", "US Commercial Real Estate"),
        "REET (Global)": ("REET", "Global Property Market"),
        "XLRE": ("XLRE", "S&P 500 Real Estate Sector")
    },
    "    Crypto Macro": {
        "BTC.D (Proxy)": ("BTC-USD", "Bitcoin Dominance Pct"),
        "Total Cap (Proxy)": ("BTC-USD", "Total Crypto Market"), 
        "BTC": ("BTC-USD", "Digital Gold / Liquidity"),
        "ETH": ("ETH-USD", "Smart Contract Platform")
    }
}

# Structure: "Label": ("Num_Ticker", "Den_Ticker", "Description")
RATIO_GROUPS = {
    "   CRYPTO RELATIVE STRENGTH": {
        "BTC / ETH (Risk Appetite)": ("BTC-USD", "ETH-USD", "Higher = Risk Off / Bitcoin Safety"),
        "BTC / SPX (Adoption)": ("BTC-USD", "^GSPC", "Crypto vs TradFi Correlation"),
        "BTC / NDX (Tech Corr)": ("BTC-USD", "^NDX", "Bitcoin vs Tech Stocks"),
        "ETH / SPX": ("ETH-USD", "^GSPC", "Ethereum Beta to Stocks"),
        "ETH / NDX": ("ETH-USD", "^NDX", "Ethereum vs Nasdaq"),
        "BTC / DXY (Liquidity)": ("BTC-USD", "DX-Y.NYB", "Higher = Liquidity Expansion"),
        "BTC / US10Y (Yields)": ("BTC-USD", "^TNX", "Crypto Sensitivity to Rates"),
        "BTC / VIX (Vol)": ("BTC-USD", "^VIX", "Price vs Fear Index"),
        "BTC / Gold (Hard Money)": ("BTC-USD", "GC=F", "Digital vs Analog Gold")
    },
    "   CRYPTO DOMINANCE (Calculated)": {
        "TOTAL 3 / TOTAL": ("SPECIAL_TOTAL3", "SPECIAL_TOTAL", "Altseason Indicator (No BTC/ETH)"),
        "TOTAL 2 / TOTAL": ("SPECIAL_TOTAL2", "SPECIAL_TOTAL", "Alts + ETH Strength"),
        "BTC.D (BTC/Total)": ("BTC-USD", "SPECIAL_TOTAL", "Bitcoin Market Share"),
        "ETH.D (ETH/Total)": ("ETH-USD", "SPECIAL_TOTAL", "Ethereum Market Share"),
        "USDT.D (Tether/Total)": ("USDT-USD", "SPECIAL_TOTAL", "Stablecoin Flight to Safety")
    },
    "   EQUITY RISK ROTATION": {
        "SPY / TLT (Risk On/Off)": ("SPY", "TLT", "Rising = Stocks Outperform Bonds"),
        "QQQ / IEF (Growth/Rates)": ("QQQ", "IEF", "Tech vs 7-10Y Treasuries"),
        "XLF / XLU (Fin/Util)": ("XLF", "XLU", "Cyclical vs Defensive"),
        "XLY / XLP (Disc/Staples)": ("XLY", "XLP", "Consumer Confident vs Defensive"),
        "IWM / SPY (Small/Large)": ("IWM", "SPY", "Risk Appetite (Small Caps)"),
        "EEM / SPY (Emerging/US)": ("EEM", "SPY", "Global Growth vs US Exceptionalism"),
        "HYG / TLT (Credit/Safe)": ("HYG", "TLT", "Junk Bond Demand vs Safety"),
        "JNK / TLT": ("JNK", "TLT", "Credit Risk Appetite"),
        "KRE / XLF (Regional/Big)": ("KRE", "XLF", "Bank Stress Indicator"),
        "SMH / SPY (Semi Lead)": ("SMH", "SPY", "Semi-Conductors Leading Market")
    },
    "   BOND & YIELD POWER": {
        "10Y / 2Y (Curve)": ("^TNX", "^IRX", "Recession Signal (Inversion)"),
        "10Y / 3M (Recession)": ("^TNX", "^IRX", "Deep Recession Signal"),
        "TLT / SHY (Duration)": ("TLT", "SHY", "Long Duration Demand"),
        "TLT / SPY (Safety/Risk)": ("TLT", "SPY", "Flight to Safety Ratio"),
        "IEF / SHY": ("IEF", "SHY", "Medium vs Short Duration"),
        "MOVE / VIX (Stress)": ("MOVE.MX", "^VIX", "Bond Vol vs Equity Vol")
    },
    "   DOLLAR & LIQUIDITY": {
        "DXY / Gold": ("DX-Y.NYB", "GC=F", "Fiat Strength vs Hard Money"),
        "DXY / Oil": ("DX-Y.NYB", "CL=F", "Dollar Purchasing Power (Energy)"),
        "EURUSD / DXY": ("EURUSD=X", "DX-Y.NYB", "Euro Relative Strength"),
        "USDJPY / DXY": ("USDJPY=X", "DX-Y.NYB", "Yen Weakness Isolation"),
        "EEM / DXY": ("EEM", "DX-Y.NYB", "Emerging Market Currency Health"),
    },
    "   COMMODITIES & INFLATION": {
        "Gold / Silver": ("GC=F", "SI=F", "Mint Ratio (High = Deflation/Fear)"),
        "Copper / Gold": ("HG=F", "GC=F", "Growth vs Safety (Dr. Copper)"),
        "Oil / Gold": ("CL=F", "GC=F", "Energy Costs vs Monetary Base"),
        "Oil / Copper": ("CL=F", "HG=F", "Energy vs Industrial Demand"),
        "Brent / WTI": ("BZ=F", "CL=F", "Geopolitical Spread")
    },
    "   EQUITIES vs REAL ASSETS": {
        "SPX / Gold": ("^GSPC", "GC=F", "Stocks priced in Real Money"),
        "SPX / Copper": ("^GSPC", "HG=F", "Financial vs Real Economy"),
        "SPX / Oil": ("^GSPC", "CL=F", "Stocks vs Energy Costs"),
        "VNQ / SPY (RE/Stocks)": ("VNQ", "SPY", "Real Estate vs Broad Market"),
        "XLE / SPX (Energy/Mkt)": ("XLE", "^GSPC", "Old Economy vs New Economy")
    },
    "   TRADE & MACRO STRESS": {
        "XLI / SPX (Ind/Mkt)": ("XLI", "^GSPC", "Industrial Strength"),
        "ITA / SPX (Defense/Mkt)": ("ITA", "^GSPC", "War Premium / Geopolitics"),
        "HYG / JNK (Quality Junk)": ("HYG", "JNK", "High Yield Dispersion")
    }
}

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data(ttl=120)
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
    """Constructs synthetic TOTAL, TOTAL2, TOTAL3 indices."""
    coins = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD", "AVAX-USD", "LINK-USD"]
    available = [c for c in coins if c in data_df.columns]
    
    if not available:
        return None, None, None
        
    sub_df = data_df[available].fillna(method='ffill')
    total = sub_df.sum(axis=1)
    
    ex_btc = [c for c in available if c != "BTC-USD"]
    total2 = sub_df[ex_btc].sum(axis=1) if ex_btc else pd.Series()
    
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
    for t_data in cat.values():
        all_needed_tickers.add(t_data[0]) # Extract ticker symbol (index 0)

for cat in RATIO_GROUPS.values():
    for t_data in cat.values():
        all_needed_tickers.add(t_data[0]) # Numerator
        all_needed_tickers.add(t_data[1]) # Denominator

# Add Crypto Components for synthetic indices
crypto_components = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "USDT-USD", "USDC-USD"]
all_needed_tickers.update(crypto_components)

# FETCH DATA ONCE
with st.spinner("Fetching Global Market Data..."):
    market_data = get_market_data(list(all_needed_tickers))
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
    for i, (label, (ticker, desc)) in enumerate(items.items()):
        col_idx = i % 3
        
        if ticker in market_data.columns:
            series = market_data[ticker].dropna()
            val, pct = calculate_change(series)
            
            if val is not None:
                data_summary_for_ai.append(f"{label}: {pct:.2f}%")
                with cols[col_idx]:
                    color = "#00FF00" if pct >= 0 else "#FF4B4B"
                    st.metric(label, f"{val:,.2f}", f"{pct:.2f}%")
                    st.caption(desc) # Display the description
                    st.plotly_chart(plot_sparkline(series, color), use_container_width=True, config={'displayModeBar': False}, key=f"std_{i}")
        else:
            with cols[col_idx]:
                st.warning(f"{label}: No Data")

else: # RATIO MODE
    # Ratio Grid
    items = RATIO_GROUPS[selected_category]
    cols = st.columns(3)
    
    for i, (label, (num_t, den_t, desc)) in enumerate(items.items()):
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
            common_idx = series_num.index.intersection(series_den.index)
            if not common_idx.empty:
                ratio_series = series_num.loc[common_idx] / series_den.loc[common_idx]
                val, pct = calculate_change(ratio_series)
                
                if val is not None:
                    data_summary_for_ai.append(f"{label}: {val:.4f} ({pct:.2f}%)")
                    with cols[col_idx]:
                        color = "#3498db" # Default Blue
                        if "Risk On" in label or "BTC/SPX" in label:
                            color = "#00FF00" if pct > 0 else "#FF4B4B"
                        
                        st.metric(label, f"{val:.4f}", f"{pct:.2f}%")
                        st.caption(desc) # Display the description
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
