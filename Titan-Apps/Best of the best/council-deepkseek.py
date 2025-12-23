
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import time as time_lib
import tweepy
import warnings
warnings.filterwarnings('ignore')

# --- OpenAI Integration ---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ==========================================
# 1. COSMIC NEXUS BRANDING & THEME
# ==========================================
st.set_page_config(
    page_title="COSMIC NEXUS | Multi-Dimensional Trading Terminal",
    layout="wide",
    page_icon="🪐",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/cosmic-nexus',
        'Report a bug': "https://github.com/cosmic-nexus/issues",
        'About': "# COSMIC NEXUS v2.0\nMulti-Dimensional Trading Intelligence"
    }
)

# Cosmic Nexus CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400&display=swap');
    
    :root {
        --cosmic-bg: #0a0a1a;
        --nebula-primary: #0f0f2e;
        --nebula-secondary: #1a1a3e;
        --starlight: #2a2a5e;
        --quasar-blue: #00b4d8;
        --neutron-green: #00ff9d;
        --pulsar-purple: #9d4edd;
        --supernova-red: #ff0050;
        --blackhole: #000814;
        --event-horizon: #001d3d;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-dim: #64748b;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--cosmic-bg) 0%, var(--nebula-primary) 50%, var(--blackhole) 100%);
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-primary);
        background-attachment: fixed;
    }
    
    /* Starfield Animation */
    .starfield {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        opacity: 0.3;
    }
    
    /* Cyber Terminal Header */
    .terminal-header {
        background: linear-gradient(90deg, var(--event-horizon) 0%, transparent 100%);
        border-left: 4px solid var(--quasar-blue);
        padding: 1.5rem;
        margin-bottom: 2rem;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 0 30px rgba(0, 180, 216, 0.1);
    }
    
    /* Nebula Cards */
    .nebula-card {
        background: rgba(16, 23, 41, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(42, 42, 94, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .nebula-card:hover {
        border-color: var(--quasar-blue);
        box-shadow: 0 0 30px rgba(0, 180, 216, 0.2);
        transform: translateY(-2px);
    }
    
    /* Dimensional Portals (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(10, 10, 26, 0.8);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid var(--starlight);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 8px 16px;
        color: var(--text-dim);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--quasar-blue), var(--pulsar-purple));
        color: white;
        box-shadow: 0 0 20px rgba(157, 78, 221, 0.3);
    }
    
    /* Quantum Metrics */
    .quantum-metric {
        background: linear-gradient(135deg, var(--event-horizon), var(--nebula-secondary));
        border: 1px solid rgba(0, 180, 216, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        position: relative;
        overflow: hidden;
    }
    
    .quantum-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--quasar-blue), transparent);
    }
    
    /* Wormhole Sidebar */
    .wormhole-sidebar {
        background: rgba(10, 10, 26, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--starlight);
    }
    
    /* Pulsing Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .pulsing {
        animation: pulse 2s infinite;
    }
    
    /* Constellation Grid */
    .constellation-grid {
        background-image: 
            radial-gradient(circle at 25% 25%, rgba(0, 180, 216, 0.1) 2px, transparent 2px),
            radial-gradient(circle at 75% 75%, rgba(157, 78, 221, 0.1) 2px, transparent 2px);
        background-size: 40px 40px;
    }
    
    /* Terminal Typing Effect */
    .terminal-text {
        font-family: 'JetBrains Mono', monospace;
        color: var(--neutron-green);
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.3);
    }
    
    /* Gravitational Wave Lines */
    .gravitational-wave {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--quasar-blue), transparent);
        margin: 1rem 0;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .nebula-card { padding: 1rem; }
        .terminal-header { padding: 1rem; }
    }
</style>

<div class="starfield">
    <div style="position: absolute; width: 2px; height: 2px; background: white; border-radius: 50%; top: 20%; left: 30%; animation: twinkle 3s infinite;"></div>
    <div style="position: absolute; width: 1px; height: 1px; background: var(--quasar-blue); border-radius: 50%; top: 40%; left: 60%; animation: twinkle 2s infinite 1s;"></div>
    <div style="position: absolute; width: 3px; height: 3px; background: var(--neutron-green); border-radius: 50%; top: 60%; left: 20%; animation: twinkle 4s infinite 0.5s;"></div>
</div>

<style>
    @keyframes twinkle {
        0%, 100% { opacity: 0.2; }
        50% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. QUANTUM COMMUNICATION PROTOCOLS
# ==========================================
def quantum_transmit(protocol, *args, **kwargs):
    """Unified communication handler for all transmission protocols"""
    
    protocols = {
        'telegram': lambda t, c, m: requests.post(
            f"https://api.telegram.org/bot{t}/sendMessage",
            json={"chat_id": c, "text": m, "parse_mode": "HTML"}
        ) if t and c else None,
        
        'x_twitter': lambda k, s, at, ats, m: tweepy.Client(
            consumer_key=k, consumer_secret=s,
            access_token=at, access_token_secret=ats
        ).create_tweet(text=m) if all([k, s, at, ats]) else None,
        
        'discord': lambda w, m: requests.post(w, json={"content": m}) if w else None,
        
        'slack': lambda w, m: requests.post(w, json={"text": m}) if w else None
    }
    
    if protocol in protocols:
        try:
            result = protocols[protocol](*args, **kwargs)
            return f"✓ {protocol.upper()} transmission successful"
        except Exception as e:
            return f"✗ {protocol.upper()} transmission failed: {str(e)[:50]}"
    
    return "✗ Unknown transmission protocol"

# ==========================================
# 3. QUANTUM COMPUTATION ENGINE
# ==========================================
class QuantumEngine:
    """Multi-dimensional market analysis engine"""
    
    @staticmethod
    def temporal_smooth(series, length):
        """Quantum temporal smoothing"""
        alpha = 2 / (length + 1)
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    @staticmethod
    def gravitational_wave(src, fast, slow):
        """Gravitational wave oscillator"""
        fast_ma = src.ewm(span=fast).mean()
        slow_ma = src.ewm(span=slow).mean()
        return (fast_ma - slow_ma) / slow_ma * 100
    
    @staticmethod
    def event_horizon(high, low, close, period=20):
        """Black hole event horizon detector"""
        hl2 = (high + low) / 2
        typical = (high + low + close) / 3
        hd = high.diff()
        ld = -low.diff()
        dmp = pd.Series(np.where((hd > 0) & (hd > ld), hd, 0)).rolling(period).sum()
        dmn = pd.Series(np.where((ld > 0) & (ld > hd), ld, 0)).rolling(period).sum()
        return dmp / (dmp + dmn + 1e-10) * 100

# CORE 1: NEXUS QUANTUM FIELD
def compute_quantum_field(df, params):
    """Multi-dimensional quantum field analysis"""
    df = df.copy()
    
    # Quantum entanglement between price and volume
    price_volume_entanglement = (df['close'].pct_change() * df['volume'].pct_change()).ewm(
        span=params['qe_period']).mean()
    
    # Temporal distortion field
    temporal_distortion = QuantumEngine.gravitational_wave(
        df['close'], 
        params['temp_fast'], 
        params['temp_slow']
    )
    
    # Spacetime curvature (volatility surface)
    spacetime_curvature = df['close'].rolling(params['st_curvature']).std() / df['close'].rolling(
        params['st_curvature']).mean() * 100
    
    # Quantum flux state
    df['quantum_flux'] = price_volume_entanglement * temporal_distortion / (spacetime_curvature + 1e-10)
    
    # Entanglement states
    conditions = [
        (df['quantum_flux'] > params['superposition_thresh']),
        (df['quantum_flux'] < -params['superposition_thresh']),
        (df['quantum_flux'].abs() < params['decoherence_thresh'])
    ]
    choices = ['ENTANGLED_BULL', 'ENTANGLED_BEAR', 'DECOHERENCE']
    df['quantum_state'] = np.select(conditions, choices, default='UNCERTAINTY')
    
    # Wavefunction collapse signals
    df['collapse_bull'] = (df['quantum_state'] == 'ENTANGLED_BULL') & (
        df['quantum_flux'].shift(1) < df['quantum_flux'])
    df['collapse_bear'] = (df['quantum_state'] == 'ENTANGLED_BEAR') & (
        df['quantum_flux'].shift(1) > df['quantum_flux'])
    
    return df

# CORE 2: TEMPORAL VORTEX ANALYZER
def compute_temporal_vortex(df, params):
    """Analyze temporal anomalies and market memory"""
    df = df.copy()
    
    # Fractal time dimensions
    time_scales = [3, 5, 8, 13, 21, 34]
    fractal_dimensions = []
    
    for scale in time_scales:
        log_range = np.log(df['high'].rolling(scale).max() / df['low'].rolling(scale).min() + 1e-10)
        log_scale = np.log(scale)
        fractal_dimensions.append(log_range / (log_scale + 1e-10))
    
    df['fractal_complexity'] = pd.concat(fractal_dimensions, axis=1).mean(axis=1)
    
    # Temporal autocorrelation (market memory)
    autocorr_lags = [1, 2, 3, 5, 8]
    memory_strength = []
    
    for lag in autocorr_lags:
        autocorr = df['close'].pct_change().autocorr(lag=lag)
        memory_strength.append(autocorr if not np.isnan(autocorr) else 0)
    
    df['temporal_memory'] = np.mean(memory_strength) * 100
    
    # Vortex detection (regime changes)
    volatility_regime = df['close'].pct_change().rolling(params['vortex_window']).std() * np.sqrt(365)
    df['volatility_regime'] = np.where(volatility_regime > params['high_vol_thresh'], 'HIGH',
                                      np.where(volatility_regime < params['low_vol_thresh'], 'LOW', 'MEDIUM'))
    
    # Temporal divergence
    short_term = QuantumEngine.temporal_smooth(df['close'], params['temp_short'])
    long_term = QuantumEngine.temporal_smooth(df['close'], params['temp_long'])
    df['temporal_divergence'] = (short_term - long_term) / long_term * 100
    
    return df

# CORE 3: GRAVITATIONAL LENS
def compute_gravitational_lens(df, params):
    """Mass distortion and gravitational pull analysis"""
    df = df.copy()
    
    # Calculate gravitational mass (market capitalization proxy)
    price_mass = df['close'] * df['volume']
    df['gravitational_mass'] = price_mass.rolling(params['mass_window']).mean()
    
    # Schwarzschild radius (support/resistance boundaries)
    avg_true_range = (df['high'] - df['low']).rolling(params['atr_period']).mean()
    df['event_horizon_upper'] = df['close'] + avg_true_range * params['gravity_multiplier']
    df['event_horizon_lower'] = df['close'] - avg_true_range * params['gravity_multiplier']
    
    # Gravitational lensing (price distortion)
    df['lensing_factor'] = (df['high'] - df['low']) / df['close'].rolling(20).std()
    
    # Singularity detection (extreme events)
    z_scores = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
    df['singularity'] = np.where(z_scores.abs() > params['singularity_thresh'], True, False)
    
    # Gravitational pull indicator
    distance_from_high = (df['high'].rolling(20).max() - df['close']) / df['close']
    distance_from_low = (df['close'] - df['low'].rolling(20).min()) / df['close']
    df['gravitational_pull'] = distance_from_low - distance_from_high
    
    return df

# CORE 4: DARK MATTER FLOW
def compute_dark_matter_flow(df, params):
    """Invisible market force analysis"""
    df = df.copy()
    
    # Hidden order flow (dark pool proxy)
    wick_ratio = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['dark_flow'] = (df['volume'] * (wick_ratio - 0.5)).ewm(span=params['dark_flow_smooth']).mean()
    
    # Quantum tunneling (breakout detection)
    resistance = df['high'].rolling(params['tunnel_window']).max()
    support = df['low'].rolling(params['tunnel_window']).min()
    
    df['quantum_tunnel_bull'] = (df['close'] > resistance.shift(1)) & (df['dark_flow'] > 0)
    df['quantum_tunnel_bear'] = (df['close'] < support.shift(1)) & (df['dark_flow'] < 0)
    
    # Entropy measurement (market disorder)
    returns = df['close'].pct_change()
    df['market_entropy'] = -np.sum(returns.rolling(20).apply(
        lambda x: np.histogram(x, bins=10)[0] / len(x) * np.log(np.histogram(x, bins=10)[0] / len(x) + 1e-10)
    )) / np.log(10)
    
    # Coherence states
    df['coherence_state'] = np.where(df['market_entropy'] < params['low_entropy_thresh'], 'COHERENT',
                                    np.where(df['market_entropy'] > params['high_entropy_thresh'], 'CHAOTIC', 'TRANSITIONAL'))
    
    return df

# CORE 5: MULTIVERSE CONVERGENCE
def compute_multiverse_convergence(df, all_cores):
    """Cross-dimensional signal convergence"""
    df = df.copy()
    
    # Initialize signal counters
    bull_signals = pd.Series(0, index=df.index)
    bear_signals = pd.Series(0, index=df.index)
    
    # Quantum Field signals
    if 'collapse_bull' in df.columns:
        bull_signals += df['collapse_bull'].astype(int)
        bear_signals += df['collapse_bear'].astype(int)
    
    # Gravitational Lens signals
    if 'gravitational_pull' in df.columns:
        bull_signals += (df['gravitational_pull'] > 0.1).astype(int)
        bear_signals += (df['gravitational_pull'] < -0.1).astype(int)
    
    # Dark Matter Flow signals
    if 'quantum_tunnel_bull' in df.columns:
        bull_signals += df['quantum_tunnel_bull'].astype(int)
        bear_signals += df['quantum_tunnel_bear'].astype(int)
    
    # Temporal Vortex signals
    if 'temporal_divergence' in df.columns:
        bull_signals += (df['temporal_divergence'] > 2).astype(int)
        bear_signals += (df['temporal_divergence'] < -2).astype(int)
    
    # Calculate convergence score
    df['multiverse_convergence'] = bull_signals - bear_signals
    
    # Determine primary dimension (strongest signal)
    max_signals = bull_signals + bear_signals
    conditions = [
        bull_signals == max_signals.max(),
        bear_signals == max_signals.max(),
        (bull_signals == 0) & (bear_signals == 0)
    ]
    choices = ['QUANTUM_BULL', 'QUANTUM_BEAR', 'DIMENSIONAL_VOID']
    df['primary_dimension'] = np.select(conditions, choices, default='MULTI_DIMENSIONAL')
    
    # Calculate confidence level
    total_possible = 4  # Number of cores
    df['convergence_confidence'] = (bull_signals.abs() + bear_signals.abs()) / total_possible * 100
    
    return df

# ==========================================
# 4. INTERSTELLAR DATA WORMHOLE
# ==========================================
@st.cache_data(ttl=10, show_spinner="Opening wormhole to exchange...")
def access_wormhole(exchange_name, symbol, timeframe, limit=1000):
    """Access market data through quantum wormhole"""
    try:
        # Initialize exchange connection
        exchange_class = getattr(ccxt, exchange_name.lower())
        exchange = exchange_class({
            'enableRateLimit': True,
            'timeout': 15000,
            'options': {'defaultType': 'spot'}
        })
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(
            symbol.replace(' ', '').upper(),
            timeframe,
            limit=limit
        )
        
        # Create spacetime continuum (DataFrame)
        continuum = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert temporal coordinates
        continuum['timestamp'] = pd.to_datetime(continuum['timestamp'], unit='ms')
        continuum.set_index('timestamp', inplace=True)
        
        # Calculate additional dimensions
        continuum['returns'] = continuum['close'].pct_change()
        continuum['log_returns'] = np.log(continuum['close'] / continuum['close'].shift(1))
        
        return continuum
        
    except Exception as e:
        st.error(f"⚠️ Wormhole instability detected: {str(e)}")
        return pd.DataFrame()

# ==========================================
# 5. NEXUS TERMINAL INTERFACE
# ==========================================
# Initialize quantum state
if 'nexus_params' not in st.session_state:
    st.session_state.nexus_params = {
        # Quantum Field
        'qe_period': 14,
        'temp_fast': 12,
        'temp_slow': 26,
        'st_curvature': 20,
        'superposition_thresh': 0.8,
        'decoherence_thresh': 0.3,
        
        # Temporal Vortex
        'vortex_window': 20,
        'high_vol_thresh': 0.8,
        'low_vol_thresh': 0.2,
        'temp_short': 10,
        'temp_long': 30,
        
        # Gravitational Lens
        'mass_window': 20,
        'atr_period': 14,
        'gravity_multiplier': 2.0,
        'singularity_thresh': 2.5,
        
        # Dark Matter Flow
        'dark_flow_smooth': 8,
        'tunnel_window': 20,
        'low_entropy_thresh': 0.3,
        'high_entropy_thresh': 0.7,
        
        # Communication
        'telegram_token': '',
        'telegram_chat_id': '',
        'x_api_key': '',
        'x_api_secret': '',
        'x_access_token': '',
        'x_access_secret': '',
        'openai_key': '',
        
        # Terminal
        'auto_sync': False,
        'alert_level': 'medium',
        'theme': 'nebula'
    }

if 'quantum_logs' not in st.session_state:
    st.session_state.quantum_logs = []

# ==========================================
# 6. WORMHOLE SIDEBAR (Navigation)
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="wormhole-sidebar" style="padding: 1rem;">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: var(--quasar-blue); margin-bottom: 0.5rem;">🪐</h1>
            <h3 style="color: var(--text-primary); margin-bottom: 0.2rem;">COSMIC NEXUS</h3>
            <p style="color: var(--text-dim); font-size: 0.8rem;">Multi-Dimensional Trading Terminal</p>
            <div class="gravitational-wave"></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Stargate Configuration
    with st.expander("🌌 STARGATE CONFIGURATION", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            exchange = st.selectbox(
                "Exchange",
                ["Binance", "Kraken", "Coinbase", "Bybit", "OKX", "KuCoin"],
                index=0,
                help="Select your gateway exchange"
            )
        with col2:
            timeframe = st.selectbox(
                "Temporal Resolution",
                ["1m", "5m", "15m", "1h", "4h", "1d"],
                index=2,
                help="Choose your observation timeframe"
            )
        
        symbol = st.text_input(
            "Quantum Symbol",
            value="BTC/USDT",
            help="Format: BTC/USDT, ETH/USDT, etc."
        )
        
        st.session_state.nexus_params['auto_sync'] = st.checkbox(
            "🌀 Auto-Sync (30s)",
            value=False,
            help="Enable temporal synchronization"
        )
    
    # Quantum Parameters
    with st.expander("⚙️ QUANTUM PARAMETERS", expanded=False):
        tab1, tab2 = st.columns(2)
        
        with tab1:
            st.session_state.nexus_params['qe_period'] = st.slider(
                "Entanglement Period", 5, 50, 14
            )
            st.session_state.nexus_params['superposition_thresh'] = st.slider(
                "Superposition Threshold", 0.1, 2.0, 0.8, 0.1
            )
        
        with tab2:
            st.session_state.nexus_params['gravity_multiplier'] = st.slider(
                "Gravity Multiplier", 1.0, 4.0, 2.0, 0.1
            )
            st.session_state.nexus_params['singularity_thresh'] = st.slider(
                "Singularity Threshold", 1.5, 4.0, 2.5, 0.1
            )
    
    # Quantum Communication
    with st.expander("📡 QUANTUM COMMUNICATION", expanded=False):
        st.session_state.nexus_params['telegram_token'] = st.text_input(
            "Telegram Token",
            value=st.session_state.nexus_params['telegram_token'],
            type="password"
        )
        st.session_state.nexus_params['telegram_chat_id'] = st.text_input(
            "Chat ID",
            value=st.session_state.nexus_params['telegram_chat_id']
        )
        
        st.session_state.nexus_params['openai_key'] = st.text_input(
            "OpenAI Key",
            value=st.session_state.nexus_params['openai_key'],
            type="password"
        )
    
    # Control Panel
    st.markdown("---")
    if st.button("🚀 INITIATE QUANTUM SCAN", type="primary", use_container_width=True):
        st.session_state.quantum_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Quantum scan initiated")
        st.rerun()
    
    if st.button("🔄 FLUSH WORMHOLE CACHE", type="secondary", use_container_width=True):
        access_wormhole.clear()
        st.session_state.quantum_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Wormhole cache cleared")
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 7. NEXUS TERMINAL DASHBOARD
# ==========================================
# Terminal Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="terminal-header">
        <h1 style="color: var(--quasar-blue); margin-bottom: 0.5rem; text-align: center;">
            🪐 COSMIC NEXUS TERMINAL
        </h1>
        <p style="color: var(--text-dim); text-align: center; font-size: 0.9rem;">
            Real-time Multi-Dimensional Market Analysis • v2.0
        </p>
    </div>
    """, unsafe_allow_html=True)

# Access wormhole data
df = access_wormhole(exchange, symbol, timeframe, 500)

if df.empty:
    st.error("""
    ## ⚠️ WORMHOLE INSTABILITY DETECTED
    
    Unable to establish connection with the exchange. Please check:
    - Symbol format (e.g., BTC/USDT)
    - Internet connection
    - Exchange availability
    """)
    st.stop()

# ==========================================
# 8. QUANTUM COMPUTATION EXECUTION
# ==========================================
# Execute all quantum cores
df = compute_quantum_field(df, st.session_state.nexus_params)
df = compute_temporal_vortex(df, st.session_state.nexus_params)
df = compute_gravitational_lens(df, st.session_state.nexus_params)
df = compute_dark_matter_flow(df, st.session_state.nexus_params)
df = compute_multiverse_convergence(df, st.session_state.nexus_params)

# Get current quantum state
current_state = df.iloc[-1]
previous_state = df.iloc[-2]

# ==========================================
# 9. QUANTUM HUD (Heads Up Display)
# ==========================================
st.markdown('<div class="constellation-grid">', unsafe_allow_html=True)

# Quantum Metrics
metric_cols = st.columns(5)
with metric_cols[0]:
    st.markdown("""
    <div class="quantum-metric">
        <div style="color: var(--text-dim); font-size: 0.8rem;">QUANTUM FLUX</div>
        <div style="color: var(--quasar-blue); font-size: 1.5rem; font-weight: bold;">
            {:.4f}
        </div>
        <div style="color: var(--text-dim); font-size: 0.8rem; margin-top: 0.5rem;">
            State: <span style="color: {}">{}</span>
        </div>
    </div>
    """.format(
        current_state['quantum_flux'],
        '#00ff9d' if current_state['quantum_state'] == 'ENTANGLED_BULL' else 
        '#ff0050' if current_state['quantum_state'] == 'ENTANGLED_BEAR' else '#64748b',
        current_state['quantum_state'].replace('_', ' ')
    ), unsafe_allow_html=True)

with metric_cols[1]:
    st.markdown("""
    <div class="quantum-metric">
        <div style="color: var(--text-dim); font-size: 0.8rem;">MULTIVERSE CONVERGENCE</div>
        <div style="color: {}; font-size: 1.5rem; font-weight: bold;">
            {:+d}
        </div>
        <div style="color: var(--text-dim); font-size: 0.8rem; margin-top: 0.5rem;">
            Confidence: {:.0f}%
        </div>
    </div>
    """.format(
        '#00ff9d' if current_state['multiverse_convergence'] > 0 else '#ff0050',
        int(current_state['multiverse_convergence']),
        current_state['convergence_confidence']
    ), unsafe_allow_html=True)

with metric_cols[2]:
    st.markdown("""
    <div class="quantum-metric">
        <div style="color: var(--text-dim); font-size: 0.8rem;">TEMPORAL MEMORY</div>
        <div style="color: var(--pulsar-purple); font-size: 1.5rem; font-weight: bold;">
            {:.1f}%
        </div>
        <div style="color: var(--text-dim); font-size: 0.8rem; margin-top: 0.5rem;">
            Regime: <span style="color: #00ff9d">{}</span>
        </div>
    </div>
    """.format(
        current_state['temporal_memory'],
        current_state['volatility_regime']
    ), unsafe_allow_html=True)

with metric_cols[3]:
    st.markdown("""
    <div class="quantum-metric">
        <div style="color: var(--text-dim); font-size: 0.8rem;">GRAVITATIONAL PULL</div>
        <div style="color: {}; font-size: 1.5rem; font-weight: bold;">
            {:.3f}
        </div>
        <div style="color: var(--text-dim); font-size: 0.8rem; margin-top: 0.5rem;">
            Lensing: {:.2f}
        </div>
    </div>
    """.format(
        '#00ff9d' if current_state['gravitational_pull'] > 0 else '#ff0050',
        current_state['gravitational_pull'],
        current_state['lensing_factor']
    ), unsafe_allow_html=True)

with metric_cols[4]:
    st.markdown("""
    <div class="quantum-metric">
        <div style="color: var(--text-dim); font-size: 0.8rem;">DARK MATTER FLOW</div>
        <div style="color: {}; font-size: 1.5rem; font-weight: bold;">
            {:.2f}
        </div>
        <div style="color: var(--text-dim); font-size: 0.8rem; margin-top: 0.5rem;">
            Entropy: {:.2f}
        </div>
    </div>
    """.format(
        '#00ff9d' if current_state['dark_flow'] > 0 else '#ff0050',
        current_state['dark_flow'],
        current_state['market_entropy']
    ), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Price Display
price_col1, price_col2, price_col3 = st.columns([1, 2, 1])
with price_col2:
    price_change = ((current_state['close'] / previous_state['close']) - 1) * 100
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: rgba(16, 23, 41, 0.5); border-radius: 16px; margin: 1rem 0;">
        <div style="color: var(--text-dim); font-size: 0.9rem;">CURRENT QUANTUM STATE</div>
        <div style="color: var(--text-primary); font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">
            ${:,.2f}
        </div>
        <div style="color: {}; font-size: 1.1rem; font-weight: bold;">
            {}{:.2f}%
        </div>
    </div>
    """.format(
        '#00ff9d' if price_change > 0 else '#ff0050',
        '↗' if price_change > 0 else '↘',
        price_change
    ), unsafe_allow_html=True)

# ==========================================
# 10. DIMENSIONAL VIEWPORTS
# ==========================================
tabs = st.tabs([
    "🌌 QUANTUM FIELD", 
    "🌀 TEMPORAL VORTEX", 
    "🔭 GRAVITATIONAL LENS",
    "🌑 DARK MATTER FLOW",
    "⚡ CONVERGENCE MATRIX",
    "🤖 QUANTUM COUNCIL"
])

# Helper function for charts
def create_cosmic_chart():
    layout = go.Layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Space Grotesk', color='#e2e8f0'),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        xaxis=dict(gridcolor='rgba(42, 42, 94, 0.3)', showgrid=True),
        yaxis=dict(gridcolor='rgba(42, 42, 94, 0.3)', showgrid=True)
    )
    return layout

# Tab 1: Quantum Field
with tabs[0]:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Price candles
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Quantum State'
        ), row=1, col=1)
        
        # Quantum flux
        colors = ['#00ff9d' if x > 0 else '#ff0050' for x in df['quantum_flux']]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['quantum_flux'],
            name='Quantum Flux',
            marker_color=colors
        ), row=2, col=1)
        
        fig.update_layout(create_cosmic_chart())
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="nebula-card">
            <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">QUANTUM FIELD DIAGNOSTICS</h4>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Current State</div>
                <div style="color: {}; font-size: 1.1rem; font-weight: bold;">{}</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Flux Magnitude</div>
                <div style="color: var(--text-primary); font-size: 1.1rem;">{:.4f}</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Bull Collapse</div>
                <div style="color: {}; font-size: 1.1rem;">{}</div>
            </div>
            <div>
                <div style="color: var(--text-dim); font-size: 0.8rem;">Bear Collapse</div>
                <div style="color: {}; font-size: 1.1rem;">{}</div>
            </div>
        </div>
        """.format(
            '#00ff9d' if current_state['quantum_state'] == 'ENTANGLED_BULL' else 
            '#ff0050' if current_state['quantum_state'] == 'ENTANGLED_BEAR' else '#64748b',
            current_state['quantum_state'].replace('_', ' '),
            current_state['quantum_flux'],
            '#00ff9d' if current_state['collapse_bull'] else '#64748b',
            'ACTIVE' if current_state['collapse_bull'] else 'INACTIVE',
            '#ff0050' if current_state['collapse_bear'] else '#64748b',
            'ACTIVE' if current_state['collapse_bear'] else 'INACTIVE'
        ), unsafe_allow_html=True)

# Tab 2: Temporal Vortex
with tabs[1]:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.6, 0.4], vertical_spacing=0.05)
        
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Temporal Stream'
        ), row=1, col=1)
        
        # Fractal complexity
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['fractal_complexity'],
            name='Fractal Complexity',
            line=dict(color='#9d4edd', width=2),
            fill='tozeroy',
            fillcolor='rgba(157, 78, 221, 0.1)'
        ), row=2, col=1)
        
        fig.update_layout(create_cosmic_chart())
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="nebula-card">
            <h4 style="color: var(--pulsar-purple); margin-bottom: 1rem;">TEMPORAL VORTEX DIAGNOSTICS</h4>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Fractal Dimension</div>
                <div style="color: var(--pulsar-purple); font-size: 1.1rem; font-weight: bold;">{:.3f}</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Temporal Memory</div>
                <div style="color: var(--text-primary); font-size: 1.1rem;">{:.1f}%</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Volatility Regime</div>
                <div style="color: #00ff9d; font-size: 1.1rem;">{}</div>
            </div>
            <div>
                <div style="color: var(--text-dim); font-size: 0.8rem;">Temporal Divergence</div>
                <div style="color: {}; font-size: 1.1rem;">{:.2f}%</div>
            </div>
        </div>
        """.format(
            current_state['fractal_complexity'],
            current_state['temporal_memory'],
            current_state['volatility_regime'],
            '#00ff9d' if current_state['temporal_divergence'] > 0 else '#ff0050',
            current_state['temporal_divergence']
        ), unsafe_allow_html=True)

# Tab 3: Gravitational Lens
with tabs[2]:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = go.Figure()
        
        # Price with event horizons
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            name='Gravitational Core',
            line=dict(color='#00b4d8', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['event_horizon_upper'],
            name='Upper Event Horizon',
            line=dict(color='#00ff9d', width=1, dash='dash'),
            fillcolor='rgba(0, 255, 157, 0.1)',
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['event_horizon_lower'],
            name='Lower Event Horizon',
            line=dict(color='#ff0050', width=1, dash='dash'),
            fill='tonexty'
        ))
        
        # Mark singularities
        singularities = df[df['singularity']]
        if not singularities.empty:
            fig.add_trace(go.Scatter(
                x=singularities.index,
                y=singularities['close'],
                mode='markers',
                name='Singularity',
                marker=dict(
                    size=10,
                    color='#ff0050',
                    symbol='diamond'
                )
            ))
        
        fig.update_layout(create_cosmic_chart())
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="nebula-card">
            <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">GRAVITATIONAL LENS DIAGNOSTICS</h4>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Gravitational Mass</div>
                <div style="color: var(--text-primary); font-size: 1.1rem;">{:,.0f}</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Lensing Factor</div>
                <div style="color: #9d4edd; font-size: 1.1rem;">{:.2f}</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Singularity Detected</div>
                <div style="color: {}; font-size: 1.1rem;">{}</div>
            </div>
            <div>
                <div style="color: var(--text-dim); font-size: 0.8rem;">Gravity Strength</div>
                <div style="color: var(--text-primary); font-size: 1.1rem;">{:.3f}</div>
            </div>
        </div>
        """.format(
            current_state['gravitational_mass'],
            current_state['lensing_factor'],
            '#ff0050' if current_state['singularity'] else '#64748b',
            'DETECTED' if current_state['singularity'] else 'QUIET',
            abs(current_state['gravitational_pull'])
        ), unsafe_allow_html=True)

# Tab 4: Dark Matter Flow
with tabs[3]:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.6, 0.4], vertical_spacing=0.05)
        
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Visible Spectrum'
        ), row=1, col=1)
        
        # Dark flow with tunneling markers
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['dark_flow'],
            name='Dark Matter Flow',
            line=dict(color='#9d4edd', width=2)
        ), row=2, col=1)
        
        # Add tunneling markers
        bull_tunnels = df[df['quantum_tunnel_bull']]
        bear_tunnels = df[df['quantum_tunnel_bear']]
        
        if not bull_tunnels.empty:
            fig.add_trace(go.Scatter(
                x=bull_tunnels.index,
                y=bull_tunnels['dark_flow'],
                mode='markers',
                name='Quantum Tunnel (Bull)',
                marker=dict(size=8, color='#00ff9d', symbol='triangle-up')
            ), row=2, col=1)
        
        if not bear_tunnels.empty:
            fig.add_trace(go.Scatter(
                x=bear_tunnels.index,
                y=bear_tunnels['dark_flow'],
                mode='markers',
                name='Quantum Tunnel (Bear)',
                marker=dict(size=8, color='#ff0050', symbol='triangle-down')
            ), row=2, col=1)
        
        fig.update_layout(create_cosmic_chart())
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="nebula-card">
            <h4 style="color: var(--pulsar-purple); margin-bottom: 1rem;">DARK MATTER DIAGNOSTICS</h4>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Dark Flow</div>
                <div style="color: {}; font-size: 1.1rem; font-weight: bold;">{:.2f}</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Market Entropy</div>
                <div style="color: #00b4d8; font-size: 1.1rem;">{:.3f}</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.8rem;">Coherence State</div>
                <div style="color: {}; font-size: 1.1rem;">{}</div>
            </div>
            <div>
                <div style="color: var(--text-dim); font-size: 0.8rem;">Bull Tunnels</div>
                <div style="color: #00ff9d; font-size: 1.1rem;">{}</div>
            </div>
        </div>
        """.format(
            '#00ff9d' if current_state['dark_flow'] > 0 else '#ff0050',
            current_state['dark_flow'],
            current_state['market_entropy'],
            '#00ff9d' if current_state['coherence_state'] == 'COHERENT' else 
            '#ff0050' if current_state['coherence_state'] == 'CHAOTIC' else '#00b4d8',
            current_state['coherence_state'],
            'ACTIVE' if current_state['quantum_tunnel_bull'] else 'INACTIVE'
        ), unsafe_allow_html=True)

# Tab 5: Convergence Matrix
with tabs[4]:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create convergence radar chart
        dimensions = ['Quantum Field', 'Temporal Vortex', 'Gravitational Lens', 'Dark Matter Flow']
        
        # Calculate scores for each dimension
        scores = [
            current_state['quantum_flux'] * 10,
            current_state['fractal_complexity'] * 20,
            abs(current_state['gravitational_pull']) * 100,
            current_state['dark_flow'] * 10
        ]
        
        # Normalize scores
        scores_normalized = [min(max(s, 0), 100) for s in scores]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=scores_normalized + [scores_normalized[0]],
            theta=dimensions + [dimensions[0]],
            fill='toself',
            fillcolor='rgba(0, 180, 216, 0.3)',
            line=dict(color='#00b4d8', width=2),
            name='Dimensional Strength'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='rgba(42, 42, 94, 0.3)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(42, 42, 94, 0.3)',
                    rotation=90
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=False,
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Grotesk', color='#e2e8f0')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        convergence_score = current_state['multiverse_convergence']
        confidence = current_state['convergence_confidence']
        
        if convergence_score > 0:
            signal = "BULLISH CONVERGENCE"
            signal_color = "#00ff9d"
            emoji = "🚀"
        elif convergence_score < 0:
            signal = "BEARISH CONVERGENCE"
            signal_color = "#ff0050"
            emoji = "⚠️"
        else:
            signal = "DIMENSIONAL EQUILIBRIUM"
            signal_color = "#00b4d8"
            emoji = "⚖️"
        
        st.markdown(f"""
        <div class="nebula-card">
            <h4 style="color: {signal_color}; margin-bottom: 1rem;">CONVERGENCE MATRIX</h4>
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{emoji}</div>
                <div style="color: {signal_color}; font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">
                    {signal}
                </div>
                <div style="color: var(--text-dim); font-size: 0.9rem;">
                    Convergence Score: {convergence_score:+d}
                </div>
            </div>
            <div class="gravitational-wave"></div>
            <div style="margin-top: 1.5rem;">
                <div style="color: var(--text-dim); font-size: 0.9rem;">Confidence Level</div>
                <div style="background: rgba(42, 42, 94, 0.3); border-radius: 8px; height: 20px; margin: 0.5rem 0;">
                    <div style="background: linear-gradient(90deg, #00b4d8, #00ff9d); width: {confidence}%; 
                            height: 100%; border-radius: 8px;"></div>
                </div>
                <div style="color: var(--text-primary); text-align: right; font-size: 0.9rem;">
                    {confidence:.0f}%
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <div style="color: var(--text-dim); font-size: 0.9rem;">Primary Dimension</div>
                <div style="color: #9d4edd; font-size: 1rem; font-weight: bold;">
                    {current_state['primary_dimension'].replace('_', ' ')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Tab 6: Quantum Council
with tabs[5]:
    if OPENAI_AVAILABLE and st.session_state.nexus_params['openai_key']:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            council_member = st.selectbox(
                "Select Quantum Analyst",
                [
                    "Dr. Nova - Quantum Physicist",
                    "Commander Orion - Strategic Analyst",
                    "Professor Hawking - Temporal Expert",
                    "AI Sentinel - Neural Network",
                    "The Oracle - Intuitive Analyst"
                ]
            )
            
            analysis_depth = st.select_slider(
                "Analysis Depth",
                options=["Quantum Scan", "Deep Analysis", "Multi-Dimensional Synthesis"],
                value="Deep Analysis"
            )
            
            if st.button("🔭 REQUEST QUANTUM ANALYSIS", type="primary", use_container_width=True):
                with st.spinner("Consulting the Quantum Council..."):
                    try:
                        client = OpenAI(api_key=st.session_state.nexus_params['openai_key'])
                        
                        # Prepare quantum data for analysis
                        analysis_data = {
                            "symbol": symbol,
                            "current_price": current_state['close'],
                            "quantum_flux": current_state['quantum_flux'],
                            "quantum_state": current_state['quantum_state'],
                            "convergence_score": current_state['multiverse_convergence'],
                            "primary_dimension": current_state['primary_dimension'],
                            "volatility_regime": current_state['volatility_regime'],
                            "gravitational_pull": current_state['gravitational_pull'],
                            "dark_flow": current_state['dark_flow'],
                            "market_entropy": current_state['market_entropy'],
                            "price_change_1h": ((current_state['close'] / df['close'].iloc[-5]) - 1) * 100,
                            "volume_trend": "increasing" if current_state['volume'] > df['volume'].iloc[-10:-1].mean() else "decreasing"
                        }
                        
                        prompt = f"""
                        Quantum Analysis Request:
                        
                        ANALYST: {council_member}
                        DEPTH: {analysis_depth}
                        
                        QUANTUM DATA:
                        {analysis_data}
                        
                        Please provide:
                        1. Quantum State Assessment
                        2. Dimensional Alignment Analysis
                        3. Risk Assessment
                        4. Strategic Recommendation
                        5. Confidence Score (0-100%)
                        
                        Format: Clear, professional, with emoji indicators.
                        """
                        
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": f"You are {council_member}, a quantum market analyst. Provide precise, actionable insights."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=500,
                            temperature=0.7
                        )
                        
                        st.session_state.council_response = response.choices[0].message.content
                        
                    except Exception as e:
                        st.error(f"Quantum communication error: {str(e)}")
        
        with col2:
            st.markdown("""
            <div class="nebula-card" style="height: 100%;">
                <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">QUANTUM COUNCIL CHAMBER</h4>
            """, unsafe_allow_html=True)
            
            if 'council_response' in st.session_state:
                st.markdown(f"""
                <div style="background: rgba(16, 23, 41, 0.5); padding: 1.5rem; border-radius: 12px; border-left: 4px solid var(--quasar-blue);">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                        Analysis by {council_member.split(' - ')[0]}
                    </div>
                    <div style="color: var(--text-primary); line-height: 1.6;">
                        {st.session_state.council_response.replace('\n', '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem 1rem; color: var(--text-dim);">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🪐</div>
                    <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">Quantum Council Ready</h4>
                    <p>Select an analyst and request analysis to begin consultation.</p>
                    <p style="font-size: 0.9rem; margin-top: 2rem; color: var(--text-dim);">
                        The Quantum Council provides multi-dimensional market insights powered by advanced AI.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="nebula-card" style="text-align: center; padding: 3rem 1rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🔒</div>
            <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">Quantum Council Locked</h4>
            <p style="color: var(--text-dim);">
                To access the Quantum Council, please provide your OpenAI API key in the sidebar configuration.
            </p>
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(16, 23, 41, 0.5); border-radius: 8px;">
                <p style="color: var(--text-dim); font-size: 0.9rem;">
                    <strong>Why enable Quantum Council?</strong><br>
                    • Multi-dimensional market analysis<br>
                    • AI-powered strategic insights<br>
                    • Risk assessment and prediction<br>
                    • Automated report generation
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 11. QUANTUM LOG & ALERTS
# ==========================================
st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    <div class="nebula-card">
        <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">QUANTUM EVENT LOG</h4>
    """, unsafe_allow_html=True)
    
    # Display quantum logs
    log_container = st.empty()
    with log_container.container():
        for log in st.session_state.quantum_logs[-10:]:  # Show last 10 logs
            st.markdown(f"""
            <div style="background: rgba(16, 23, 41, 0.3); padding: 0.5rem 1rem; margin-bottom: 0.5rem; 
                     border-radius: 8px; border-left: 3px solid var(--quasar-blue); font-size: 0.9rem;">
                <span style="color: var(--text-dim);">{log.split(']')[0]}]</span>
                <span style="color: var(--text-primary);">{log.split(']')[1]}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Check for alert conditions
    alerts = []
    
    if current_state['singularity']:
        alerts.append(("SINGULARITY DETECTED", "#ff0050", "⚠️"))
    
    if abs(current_state['quantum_flux']) > st.session_state.nexus_params['superposition_thresh']:
        alerts.append(("QUANTUM FLUX SPIKE", "#00ff9d", "⚡"))
    
    if current_state['multiverse_convergence'] >= 3:
        alerts.append(("STRONG BULL CONVERGENCE", "#00ff9d", "🚀"))
    elif current_state['multiverse_convergence'] <= -3:
        alerts.append(("STRONG BEAR CONVERGENCE", "#ff0050", "🔻"))
    
    if current_state['quantum_tunnel_bull']:
        alerts.append(("QUANTUM TUNNEL BULL", "#00ff9d", "🔼"))
    elif current_state['quantum_tunnel_bear']:
        alerts.append(("QUANTUM TUNNEL BEAR", "#ff0050", "🔽"))
    
    st.markdown("""
    <div class="nebula-card">
        <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">ACTIVE ALERTS</h4>
    """, unsafe_allow_html=True)
    
    if alerts:
        for alert_text, color, emoji in alerts[:5]:  # Show max 5 alerts
            st.markdown(f"""
            <div style="background: {color}20; border: 1px solid {color}; padding: 0.8rem; 
                     margin-bottom: 0.5rem; border-radius: 8px; font-size: 0.9rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">{emoji}</span>
                    <span style="color: var(--text-primary); font-weight: 500;">{alert_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Send alerts if configured
        if st.session_state.nexus_params['telegram_token'] and alerts:
            alert_message = f"🚨 COSMIC NEXUS ALERT for {symbol}:\n"
            for alert_text, _, emoji in alerts:
                alert_message += f"{emoji} {alert_text}\n"
            alert_message += f"\nCurrent Price: ${current_state['close']:.2f}"
            
            quantum_transmit('telegram', 
                           st.session_state.nexus_params['telegram_token'],
                           st.session_state.nexus_params['telegram_chat_id'],
                           alert_message)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem; color: var(--text-dim);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">✅</div>
            <div>All systems nominal</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem;">No active alerts</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 12. TEMPORAL SYNCHRONIZATION
# ==========================================
if st.session_state.nexus_params['auto_sync']:
    current_time = time_lib.time()
    if 'last_sync' not in st.session_state:
        st.session_state.last_sync = current_time
    
    if current_time - st.session_state.last_sync > 30:  # 30 seconds
        st.session_state.last_sync = current_time
        st.session_state.quantum_logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] Temporal synchronization complete"
        )
        st.rerun()

# ==========================================
# 13. TERMINAL FOOTER
# ==========================================
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1.5rem; color: var(--text-dim); font-size: 0.8rem;">
    <div class="gravitational-wave"></div>
    <div style="margin-top: 1rem;">
        <strong>COSMIC NEXUS TERMINAL v2.0</strong> • Multi-Dimensional Trading Intelligence
    </div>
    <div style="margin-top: 0.5rem;">
        Last Updated: {} • Temporal Resolution: {} • Convergence Confidence: {:.0f}%
    </div>
    <div style="margin-top: 0.5rem; color: var(--text-dim);">
        ⚠️ This is a quantum computing simulation. Not financial advice. Trade at your own risk.
    </div>
</div>
""".format(
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    timeframe,
    current_state['convergence_confidence']
), unsafe_allow_html=True)
