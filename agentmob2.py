import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats
import scipy.special
from openai import OpenAI
import yfinance as yf
import time

# ==========================================
# 1. PAGE CONFIGURATION & DPC CSS ARCHITECTURE
# ==========================================
st.set_page_config(
    page_title="TITAN | QUANTUM MACRO",
    layout="wide",
    page_icon="Ω",
    initial_sidebar_state="expanded"
)

# DPC Dark Mode CSS
st.markdown("""
    <style>
    /* MAIN CONTAINER */
    .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* TEXT UTILS */
    h1, h2, h3, h4 { color: #ffffff !important; font-family: 'Roboto Mono', monospace; letter-spacing: -1px; }
    .neon-text-blue { color: #00E5FF; text-shadow: 0 0 10px rgba(0, 229, 255, 0.6); }
    .neon-text-pink { color: #FF0055; text-shadow: 0 0 10px rgba(255, 0, 85, 0.6); }
    
    /* CUSTOM METRIC CARDS */
    .metric-card {
        background: #151920;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 3px solid #333;
    }
    .metric-card.bull { border-left-color: #00E5FF; }
    .metric-card.bear { border-left-color: #FF0055; }
    
    /* INPUTS */
    .stTextInput > div > div > input { background-color: #151920; color: #e0e0e0; border: 1px solid #333; }
    .stSelectbox > div > div > div { background-color: #151920; color: #e0e0e0; }
    
    /* BUTTONS */
    div.stButton > button {
        background-color: #151920;
        border: 1px solid #00E5FF;
        color: #00E5FF;
        width: 100%;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #00E5FF;
        color: #000;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.5);
    }
    
    /* AI PANEL */
    .ai-panel {
        background-color: #111;
        border: 1px solid #444;
        padding: 20px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        line-height: 1.6;
        color: #0f0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. TITAN MATH ENGINE (PINE PORTS)
# ==========================================
class TitanMath:
    @staticmethod
    def calc_entropy(series, length, bins=16):
        """Calculates Rolling Shannon Entropy (Vectorized-ish via Apply)."""
        def _entropy(window):
            hist, _ = np.histogram(window, bins=bins, density=True)
            return scipy.stats.entropy(hist)
        
        return series.rolling(length).apply(_entropy, raw=True)

    @staticmethod
    def riemann_quantum_oscillator(df, length=252, chaos_len=55, re_thresh=1.8, psi_thresh=0.35):
        """
        Ω – Riemann–Quantum–Lyapunov–Reynolds Fusion Oscillator
        Ported from Pine Script logic.
        """
        # 1. Quantum Tunnel Probability
        # V0 = Barrier Height (StdDev), E = Particle Energy (Diff)
        v0 = df['close'].rolling(length).std()
        e = (df['close'] - df['close'].shift(length)).abs()
        atr = df['high'].sub(df['low']).rolling(14).mean() # Simple ATR approx
        
        # Kappa calculation: sqrt(2m(V0-E)) / h_bar
        # We ensure inside sqrt is positive
        inner = (v0 - e).clip(lower=0)
        kappa = np.sqrt(inner) / (atr.replace(0, 1)) # Avoid div by zero
        
        # Psi = sech^2(kappa) -> 1 / cosh^2(kappa)
        psi = 1 / (np.cosh(kappa) ** 2)
        
        # 2. Lyapunov Exponent (Simplified Rosenstein)
        # We use Log divergence of returns over chaos_len
        log_ret = np.log(df['close'] / df['close'].shift(1))
        # Rolling divergence of returns represents chaos
        lyap = log_ret.rolling(chaos_len).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        # Scale Lyapunov for visualization
        lyap_scaled = np.sin(np.pi * lyap * 100) 

        # 3. Reynolds Number (Re)
        # Re = rho * u * L / mu
        rho = df['volume']
        u = (df['close'] - df['close'].shift(1)).abs() # Velocity
        L = atr # Characteristic Length
        mu = df['close'].rolling(length).std() # Viscosity
        
        Re = (rho * u * L) / mu.replace(0, 0.0001)
        Re_Star = Re.rolling(length).mean()
        
        # 4. Fusion Omega
        # Entropy
        raw_change = df['close'].diff()
        ent = TitanMath.calc_entropy(raw_change, length=50) # Shorter length for speed in Python
        ent_max = 0.92
        
        # Omega Calculation
        # Ω = Ψ · tanh(Re/Re*) · (1 – S/Smax)
        term1 = psi
        term2 = np.tanh(Re / Re_Star.replace(0, 1))
        term3 = (1 - (ent / ent_max)).clip(lower=0)
        
        omega_mod = term1 * term2 * term3
        omega_im = omega_mod * lyap_scaled
        
        return pd.DataFrame({
            'Omega_Im': omega_im,
            'Omega_Mod': omega_mod,
            'Reynolds': Re,
            'Entropy': ent
        })

    @staticmethod
    def apex_vector(df, len_vec=14, vol_norm=55, len_sm=5):
        """
        Apex Vector [Flux + Efficiency]
        """
        # 1. Geometric Efficiency
        range_abs = df['high'] - df['low']
        body_abs = (df['close'] - df['open']).abs()
        
        # Avoid div by zero
        raw_eff = body_abs / range_abs.replace(0, 0.0001)
        efficiency = raw_eff.ewm(span=len_vec).mean()
        
        # 2. Volume Flux
        vol_avg = df['volume'].rolling(vol_norm).mean()
        vol_fact = df['volume'] / vol_avg.replace(0, 1)
        
        # 3. Vector Calculation
        direction = np.sign(df['close'] - df['open'])
        vector_raw = direction * efficiency * vol_fact
        
        # 4. Smoothing
        flux = vector_raw.ewm(span=len_sm).mean()
        
        return flux

    @staticmethod
    def rqzo_oscillator(df, N=25):
        """
        Relativistic Quantum-Zeta Oscillator (RQZO)
        Uses Riemann Zeta summation approximation.
        """
        # 1. Normalization
        norm_price = (df['close'] - df['close'].rolling(100).min()) / \
                     (df['close'].rolling(100).max() - df['close'].rolling(100).min() + 1e-10)
        
        # 2. Lorentz Factor (Gamma)
        velocity = norm_price.diff().abs()
        c = 0.05 # Terminal volatility constant
        gamma = 1 / np.sqrt(1 - (velocity.clip(upper=c*0.99)/c)**2)
        
        # 3. Riemann Summation
        # This is computationally heavy, so we vectorize the summation
        # s = sigma + i*tau
        sigma = 0.5
        # Create a time index for Tau
        tau = (np.arange(len(df)) % 100) / gamma
        
        zeta_imag = np.zeros(len(df))
        
        # Vectorized summation for harmonics 1 to N
        # Im(n^-s) = -n^-sigma * sin(tau * ln(n))
        for n in range(1, N + 1):
            amp = n ** (-sigma)
            theta = tau * np.log(n)
            zeta_imag += amp * np.sin(theta)
            
        return pd.Series(zeta_imag, index=df.index)

# ==========================================
# 3. DATA HANDLER
# ==========================================
@st.cache_data(ttl=60)
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        return None

# ==========================================
# 4. UI COMPONENTS
# ==========================================
def render_sidebar():
    st.sidebar.markdown("## ⚙️ SYSTEM CONFIG")
    ticker = st.sidebar.text_input("ASSET TICKER", value="BTC-USD")
    period = st.sidebar.selectbox("DATA RANGE", ["3mo", "6mo", "1y", "2y"], index=2)
    interval = st.sidebar.selectbox("TIMEFRAME", ["1h", "1d", "1wk"], index=1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🧠 QUANT PARAMS")
    
    # Fusion Params
    chaos_len = st.sidebar.slider("Lyapunov Length", 10, 100, 55)
    psi_thresh = st.sidebar.slider("Quantum Tunnel Prob", 0.1, 0.9, 0.35)
    
    # Apex Params
    eff_thresh = st.sidebar.slider("Apex Superconductor", 0.1, 1.0, 0.6)
    
    # AI Config
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🤖 CORTEX AI")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    return ticker, period, interval, chaos_len, psi_thresh, eff_thresh, api_key

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    ticker, period, interval, chaos_len, psi_thresh, eff_thresh, api_key = render_sidebar()
    
    # 1. Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"# TITAN // {ticker} <span style='font-size:0.5em; color:#555'>MACRO DASHBOARD</span>", unsafe_allow_html=True)
    
    # 2. Fetch Data
    df = fetch_data(ticker, period, interval)
    if df is None:
        st.error("SYSTEM ERROR: Data feed interrupted. Check ticker.")
        return

    # 3. Run Math Engine
    with st.spinner("CALCULATING QUANTUM VECTORS..."):
        # A. Fusion Oscillator
        fusion_df = TitanMath.riemann_quantum_oscillator(df, chaos_len=chaos_len, psi_thresh=psi_thresh)
        
        # B. Apex Vector
        apex_flux = TitanMath.apex_vector(df)
        
        # C. RQZO
        rqzo = TitanMath.rqzo_oscillator(df)

    # 4. Metrics Deck
    last_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    chg_pct = ((last_price - prev_price) / prev_price) * 100
    
    last_omega = fusion_df['Omega_Im'].iloc[-1]
    last_flux = apex_flux.iloc[-1]
    last_rqzo = rqzo.iloc[-1]
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("PRICE", f"${last_price:,.2f}", f"{chg_pct:.2f}%")
    with m2:
        st.metric("Ω (FUSION)", f"{last_omega:.4f}", delta_color="off")
    with m3:
        st.metric("APEX FLUX", f"{last_flux:.2f}", delta_color="normal")
    with m4:
        st.metric("RQZO ZETA", f"{last_rqzo:.2f}", delta_color="off")

    # 5. Elite Visuals (Plotly)
    st.markdown("### 📊 QUANTUM DATA VISUALIZATION")
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=("PRICE ACTION & APEX COLORING", "Ω FUSION OSCILLATOR", "APEX VECTOR FLUX", "RELATIVISTIC ZETA (RQZO)")
    )
    
    # ROW 1: Candlestick + Apex Coloring
    # Determine bar colors based on Apex Flux
    colors = np.where(apex_flux > eff_thresh, '#00E676', # Super Bull
             np.where(apex_flux < -eff_thresh, '#FF1744', # Super Bear
             '#546E7A')) # Resistive/Neutral
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price'
    ), row=1, col=1)
    
    # Overlay Colored Bars (Simulated via Bar chart on top for coloring)
    # Using simple markers for Apex State
    fig.add_trace(go.Scatter(
        x=df.index, y=df['high']*1.01,
        mode='markers',
        marker=dict(color=colors, size=4, symbol='square'),
        name='Apex State'
    ), row=1, col=1)

    # ROW 2: Fusion Oscillator (Omega)
    fig.add_trace(go.Scatter(
        x=df.index, y=fusion_df['Omega_Im'],
        line=dict(color='#FFA500', width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.1)',
        name='Ω Imag'
    ), row=2, col=1)
    
    # Add horizontal zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    # ROW 3: Apex Flux
    fig.add_trace(go.Bar(
        x=df.index, y=apex_flux,
        marker_color=colors,
        name='Apex Flux'
    ), row=3, col=1)

    # ROW 4: RQZO
    fig.add_trace(go.Scatter(
        x=df.index, y=rqzo,
        line=dict(color='#9C27B0', width=1.5),
        name='RQZO'
    ), row=4, col=1)

    # Layout Styling
    fig.update_layout(
        height=900,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(family="Roboto Mono", color="#e0e0e0"),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(showgrid=False, gridcolor='#222')
    fig.update_yaxes(showgrid=True, gridcolor='#222')
    
    st.plotly_chart(fig, use_container_width=True)

    # 6. AI INTERFACE
    st.markdown("### 🤖 CORTEX ANALYSIS")
    
    if st.button("INITIALIZE AI DIAGNOSTIC"):
        if not api_key:
            st.error("MISSING API KEY: Please inject credentials in the Sidebar.")
        else:
            client = OpenAI(api_key=api_key)
            
            # Prepare context for AI
            summary_data = f"""
            Asset: {ticker}
            Last Price: {last_price}
            
            INDICATOR READINGS (Last 5 periods):
            Omega (Fusion): {fusion_df['Omega_Im'].tail(5).tolist()}
            Apex Flux: {apex_flux.tail(5).tolist()}
            RQZO (Zeta): {rqzo.tail(5).tolist()}
            Entropy: {fusion_df['Entropy'].tail(5).tolist()}
            Reynolds Num: {fusion_df['Reynolds'].tail(5).tolist()}
            
            THRESHOLDS:
            Apex Superconductor: +/- {eff_thresh}
            Fusion Tunnel Prob: {psi_thresh}
            """
            
            system_prompt = """
            You are 'The Architect', an elite quantitative trading AI.
            Analyze the provided technical data derived from Riemann-Zeta physics and Fluid Dynamics market models.
            
            Logic:
            1. Apex Flux > Threshold = Superconductor Regime (Strong Trend).
            2. Omega Crossing Zero with High Entropy = Phase Transition (Reversal).
            3. RQZO Extremes = Time Dilation/Over-extension.
            
            Output Format:
            **MARKET STATE:** [LAMINAR | TURBULENT | SUPERCONDUCTOR]
            **BIAS:** [BULLISH | BEARISH | NEUTRAL]
            **CONFIDENCE:** [0-100%]
            **STRATEGIC BRIEF:** 2-3 sentences max. purely technical. No financial advice disclaimers.
            """
            
            with st.spinner("PROCESSING NEURAL TENSORS..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": summary_data}
                        ]
                    )
                    analysis = response.choices[0].message.content
                    st.markdown(f"<div class='ai-panel'>{analysis}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"NEURAL SYNC FAILURE: {e}")

if __name__ == "__main__":
    main()
