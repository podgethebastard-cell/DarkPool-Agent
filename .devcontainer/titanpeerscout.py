import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime

# ==========================================
# 1. PAGE CONFIGURATION & TITAN CSS
# ==========================================
st.set_page_config(
    page_title="Titan Peer Scout",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DARKPOOL AESTHETIC ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Header Glow */
    .titan-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E676, #2979FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 230, 118, 0.3);
        margin-bottom: 10px;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: scale(1.02);
        border-color: #2979FF;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700;
        color: #fff;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #888;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #161b22;
        border-radius: 4px;
        color: #8b949e;
        border: 1px solid #30363d;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1117;
        color: #00E676;
        border-bottom: 2px solid #00E676;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #161b22;
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE
# ==========================================

DEFAULT_PEERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA"]

# Time Horizon Mapping for yfinance
HORIZON_MAP = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "YTD": "ytd",
    "1 Year": "1y",
    "3 Years": "3y",
    "5 Years": "5y"
}

@st.cache_data(ttl=3600)
def fetch_market_data(tickers, period):
    """
    Fetches adjusted close prices for multiple tickers.
    Vectorized data handling.
    """
    if not tickers:
        return pd.DataFrame()
    
    try:
        # Batch download is faster
        df = yf.download(tickers, period=period, group_by='ticker', progress=False)
        
        # Extract Close prices into a single clean DataFrame
        close_df = pd.DataFrame()
        
        for t in tickers:
            # Handle cases where yfinance returns MultiIndex or single index
            try:
                if len(tickers) > 1:
                    if 'Adj Close' in df[t].columns:
                        close_df[t] = df[t]['Adj Close']
                    elif 'Close' in df[t].columns:
                        close_df[t] = df[t]['Close']
                else:
                    # Single ticker case
                    if 'Adj Close' in df.columns:
                        close_df[t] = df['Adj Close']
                    elif 'Close' in df.columns:
                        close_df[t] = df['Close']
            except KeyError:
                continue # Skip tickers that failed
                
        return close_df.dropna()
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

def process_peer_analytics(df):
    """
    Calculates Normalized Performance, Peer Averages, and Deltas.
    """
    if df.empty:
        return None, None, None
    
    # 1. Normalize to percentage return (Start = 0%)
    # formula: (Price / StartPrice) - 1
    norm_df = (df / df.iloc[0]) - 1
    
    # 2. Calculate Peer Average (Equal Weight Index)
    df['Peer_Avg'] = df.mean(axis=1)
    norm_df['Peer_Avg'] = (df['Peer_Avg'] / df['Peer_Avg'].iloc[0]) - 1
    
    # 3. Calculate Alpha (Delta vs Peer Avg)
    # How much did Stock X beat the group average?
    delta_df = pd.DataFrame()
    for col in df.columns:
        if col != 'Peer_Avg':
            # Alpha is the difference in normalized returns
            delta_df[col] = norm_df[col] - norm_df['Peer_Avg']
            
    return norm_df, delta_df, df

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### 🔭 Control Deck")
    
    # Ticker Selection
    selected_tickers = st.multiselect(
        "Select Assets",
        options=sorted(list(set(DEFAULT_PEERS + ["AMD", "INTC", "QCOM", "AVGO", "TXN", "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE"]))),
        default=DEFAULT_PEERS[:5]
    )
    
    # Add custom ticker
    custom_ticker = st.text_input("Add Custom Ticker (e.g. BRK-B)", "")
    if custom_ticker:
        custom_ticker = custom_ticker.upper().strip()
        if custom_ticker not in selected_tickers:
            selected_tickers.append(custom_ticker)
    
    st.markdown("---")
    
    # Timeframe
    selected_horizon_label = st.selectbox("Lookback Period", list(HORIZON_MAP.keys()), index=2)
    selected_horizon = HORIZON_MAP[selected_horizon_label]
    
    st.markdown("---")
    st.info("💡 **Titan Tip:** Compare stocks within the same sector for valid alpha generation.")

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.markdown('<div class="titan-header">🔭 Titan Peer Scout</div>', unsafe_allow_html=True)

if not selected_tickers or len(selected_tickers) < 2:
    st.warning("⚠️ Please select at least 2 tickers to perform peer analysis.")
    st.stop()

# --- LOAD & PROCESS DATA ---
with st.spinner("Initializing Data Matrix..."):
    raw_data = fetch_market_data(selected_tickers, selected_horizon)
    norm_data, delta_data, price_data = process_peer_analytics(raw_data)

if norm_data is None or norm_data.empty:
    st.error("❌ Insufficient data returned. Check tickers or try a different timeframe.")
    st.stop()

# --- KPI ROW ---
# Calculate leaders based on total return over period
total_return = norm_data.iloc[-1].drop('Peer_Avg')
best_stock = total_return.idxmax()
best_val = total_return.max()
worst_stock = total_return.idxmin()
worst_val = total_return.min()

# Calculate Alpha Leader (Beat avg by most)
alpha_scores = delta_data.iloc[-1]
alpha_king = alpha_scores.idxmax()
alpha_val = alpha_scores.max()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("🏆 Relative Strength King", best_stock, f"{best_val:.1%}")
with k2:
    st.metric("🛡️ Alpha Leader", alpha_king, f"+{alpha_val:.1%} vs Peers")
with k3:
    st.metric("🥀 Laggard", worst_stock, f"{worst_val:.1%}")
with k4:
    # Dispersion (Standard Deviation of returns at the end)
    dispersion = total_return.std()
    st.metric("📊 Group Dispersion", f"{dispersion*100:.1f}%", help="Standard deviation of returns within the group.")

# --- TABS LAYOUT ---
tab_perf, tab_alpha, tab_corr, tab_vol = st.tabs([
    "📈 Normalized Performance", 
    "🎯 Peer Alpha (Delta)", 
    "🧩 Correlation Matrix", 
    "⚡ Volatility Engine"
])

# --- TAB 1: NORMALIZED PERFORMANCE ---
with tab_perf:
    st.markdown("##### 🟢 Performance from Start Date (Base = 0%)")
    
    # Plotly Line Chart
    fig_norm = go.Figure()
    
    # Plot Peers
    for col in norm_data.columns:
        if col != 'Peer_Avg':
            # Highlight Best/Worst visually? Maybe later.
            fig_norm.add_trace(go.Scatter(
                x=norm_data.index, y=norm_data[col],
                mode='lines', name=col,
                hovertemplate='%{y:.1%}'
            ))
            
    # Plot Average (Thick Dashed Line)
    fig_norm.add_trace(go.Scatter(
        x=norm_data.index, y=norm_data['Peer_Avg'],
        mode='lines', name='PEER AVERAGE',
        line=dict(color='white', width=3, dash='dash'),
        hovertemplate='%{y:.1%}'
    ))
    
    fig_norm.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(tickformat=".0%"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_norm, use_container_width=True)

# --- TAB 2: PEER ALPHA (DELTA) ---
with tab_alpha:
    st.markdown("##### ⚔️ Alpha Generation (Performance vs Peer Average)")
    st.caption("Positive values indicate outperformance against the group index. Negative values indicate underperformance.")
    
    # Dynamic Grid for Delta Charts
    # We create small charts for each ticker showing Area plot of alpha
    
    cols = st.columns(3) # 3 charts per row
    charts = []
    
    for i, ticker in enumerate(delta_data.columns):
        col_idx = i % 3
        with cols[col_idx]:
            # Color logic: Green if ending positive, Red if negative
            final_val = delta_data[ticker].iloc[-1]
            fill_color = 'rgba(0, 230, 118, 0.2)' if final_val >= 0 else 'rgba(255, 23, 68, 0.2)'
            line_color = '#00E676' if final_val >= 0 else '#FF1744'
            
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(
                x=delta_data.index, y=delta_data[ticker],
                fill='tozeroy',
                fillcolor=fill_color,
                mode='lines',
                line=dict(color=line_color, width=2),
                name=ticker
            ))
            
            fig_delta.update_layout(
                template="plotly_dark",
                height=200,
                title=dict(text=f"<b>{ticker}</b>: {final_val:+.1%}", x=0.05, y=0.9),
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis=dict(showgrid=False, tickformat=".0%", zeroline=True, zerolinewidth=1, zerolinecolor='white'),
                xaxis=dict(showgrid=False),
                showlegend=False
            )
            st.plotly_chart(fig_delta, use_container_width=True)

# --- TAB 3: CORRELATION MATRIX ---
with tab_corr:
    st.markdown("##### 🔗 Return Correlation Matrix")
    
    # Calculate Log Returns for Correlation
    log_rets = np.log(price_data.drop(columns=['Peer_Avg']) / price_data.drop(columns=['Peer_Avg']).shift(1)).dropna()
    corr_matrix = log_rets.corr()
    
    # Plotly Heatmap
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r", # Red = Low Corr, Blue = High Corr
        zmin=-1, zmax=1
    )
    
    fig_corr.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 4: VOLATILITY ENGINE ---
with tab_vol:
    st.markdown("##### ⚡ Volatility Analysis (Annualized)")
    
    # Calculate Annualized Volatility (Std Dev of Daily Returns * sqrt(252))
    volatility = log_rets.std() * np.sqrt(252) * 100
    volatility = volatility.sort_values(ascending=False)
    
    # Bar Chart
    fig_vol = go.Figure(go.Bar(
        x=volatility.values,
        y=volatility.index,
        orientation='h',
        marker=dict(
            color=volatility.values,
            colorscale='Viridis',
            showscale=False
        ),
        text=[f"{v:.1f}%" for v in volatility.values],
        textposition='auto'
    ))
    
    fig_vol.update_layout(
        template="plotly_dark",
        height=500,
        title="Annualized Volatility (Risk)",
        xaxis_title="Volatility %",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Sharpe Ratio Estimate (Assuming risk-free rate ~ 4%)
    st.markdown("##### ⚖️ Sharpe Ratio Estimate (Rf=4%)")
    rf = 0.04
    # Annualized Return
    total_days = (raw_data.index[-1] - raw_data.index[0]).days
    years = total_days / 365.25
    cagr = (price_data.drop(columns=['Peer_Avg']).iloc[-1] / price_data.drop(columns=['Peer_Avg']).iloc[0]) ** (1/years) - 1
    
    sharpe = (cagr - rf) / (volatility / 100)
    sharpe = sharpe.sort_values(ascending=False)
    
    fig_sharpe = go.Figure(go.Bar(
        x=sharpe.values,
        y=sharpe.index,
        orientation='h',
        marker=dict(color='#2979FF')
    ))
    
    fig_sharpe.update_layout(
        template="plotly_dark",
        height=500,
        title="Sharpe Ratio (Risk-Adjusted Return)",
        xaxis_title="Ratio",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption(f"Titan Terminal | Data Source: Yahoo Finance | Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
