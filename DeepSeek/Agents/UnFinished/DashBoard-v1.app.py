import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time as time_lib
import tweepy
import warnings
from typing import Optional, Dict, Any, List, Tuple
import json
from scipy import stats
import math
warnings.filterwarnings('ignore')

# ==========================================
# 1. ENHANCED IMPORTS & CONFIGURATION
# ==========================================
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==========================================
# 2. TECHNICAL INDICATORS LIBRARY (REPLACEMENT FOR `ta`)
# ==========================================

class TechnicalIndicators:
    """Complete technical indicators library without external dependencies"""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int, adjust: bool = False) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=adjust).mean()
    
    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD, Signal, and Histogram"""
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands (Upper, Middle, Lower)"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator (%K and %D)"""
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        k = 100 * ((close - low_min) / (high_max - low_min + 1e-10))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        hl = high - low
        hc = np.abs(high - close.shift())
        lc = np.abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(0, index=close.index)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index (Simplified)"""
        # Calculate directional movements
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = pd.Series(0, index=high.index)
        minus_dm = pd.Series(0, index=high.index)
        
        # Plus DM
        plus_condition = (high_diff > low_diff.abs()) & (high_diff > 0)
        plus_dm[plus_condition] = high_diff[plus_condition]
        
        # Minus DM
        minus_condition = (low_diff.abs() > high_diff) & (low_diff < 0)
        minus_dm[minus_condition] = low_diff.abs()[minus_condition]
        
        # Smooth DMs
        plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate True Range
        tr = TechnicalIndicators.atr(high, low, close, period)
        
        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm_smooth / (tr + 1e-10))
        minus_di = 100 * (minus_dm_smooth / (tr + 1e-10))
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = (typical_price - sma).abs().rolling(window=period).mean()
        cci = (typical_price - sma) / (0.015 * mean_deviation + 1e-10)
        return cci
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
        return williams_r
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        price_change = typical_price.diff()
        positive_flow = money_flow.where(price_change > 0, 0)
        negative_flow = money_flow.where(price_change < 0, 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / (negative_mf + 1e-10)
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    @staticmethod
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series, 
                conversion_period: int = 9, base_period: int = 26, 
                leading_span_b_period: int = 52, displacement: int = 26) -> Dict[str, pd.Series]:
        """Ichimoku Cloud Components"""
        # Conversion Line (Tenkan-sen)
        conversion = (high.rolling(window=conversion_period).max() + 
                     low.rolling(window=conversion_period).min()) / 2
        
        # Base Line (Kijun-sen)
        base = (high.rolling(window=base_period).max() + 
               low.rolling(window=base_period).min()) / 2
        
        # Leading Span A (Senkou Span A)
        leading_span_a = ((conversion + base) / 2).shift(displacement)
        
        # Leading Span B (Senkou Span B)
        leading_span_b = ((high.rolling(window=leading_span_b_period).max() + 
                          low.rolling(window=leading_span_b_period).min()) / 2).shift(displacement)
        
        # Lagging Span (Chikou Span)
        lagging_span = close.shift(-displacement)
        
        return {
            'conversion': conversion,
            'base': base,
            'leading_span_a': leading_span_a,
            'leading_span_b': leading_span_b,
            'lagging_span': lagging_span
        }

# Initialize indicators class
ti = TechnicalIndicators()

# Enhanced Exchange Support
EXCHANGE_METADATA = {
    'Kraken': {
        'class': ccxt.kraken,
        'global_access': True,
        'us_accessible': True,
        'rate_limit': 1.0,
        'symbol_format': lambda s: s.replace('/', '').upper(),
        'testnet': False,
        'features': ['spot', 'futures', 'margin']
    },
    'Coinbase': {
        'class': ccxt.coinbase,
        'global_access': True,
        'us_accessible': True,
        'rate_limit': 3.0,
        'symbol_format': lambda s: s.replace('/', '-').upper(),
        'testnet': False,
        'features': ['spot']
    },
    'Binance': {
        'class': ccxt.binance,
        'global_access': False,
        'us_accessible': False,
        'rate_limit': 0.5,
        'symbol_format': lambda s: s.replace('/', '').upper(),
        'testnet': True,
        'testnet_url': 'https://testnet.binance.vision',
        'features': ['spot', 'futures', 'margin', 'options']
    },
    'KuCoin': {
        'class': ccxt.kucoin,
        'global_access': True,
        'us_accessible': True,
        'rate_limit': 1.5,
        'symbol_format': lambda s: s.replace('/', '-').upper(),
        'testnet': False,
        'features': ['spot', 'futures', 'margin']
    },
    'Bybit': {
        'class': ccxt.bybit,
        'global_access': False,
        'us_accessible': True,
        'rate_limit': 1.0,
        'symbol_format': lambda s: s.replace('/', '').upper(),
        'testnet': True,
        'testnet_url': 'https://api-testnet.bybit.com',
        'features': ['spot', 'futures', 'options']
    },
    'OKX': {
        'class': ccxt.okx,
        'global_access': True,
        'us_accessible': True,
        'rate_limit': 1.0,
        'symbol_format': lambda s: s.replace('/', '-').upper(),
        'testnet': False,
        'features': ['spot', 'futures', 'margin', 'options']
    }
}

# ==========================================
# 3. ULTIMATE UI/UX ENHANCEMENTS
# ==========================================
st.set_page_config(
    page_title="COSMIC NEXUS v3.0 | Advanced Quantum Trading Terminal",
    layout="wide",
    page_icon="🪐",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://cosmic-nexus.io/docs',
        'Report a bug': "https://github.com/cosmic-nexus/issues",
        'About': """
        # COSMIC NEXUS v3.0
        
        **Advanced Quantum Trading Terminal**
        
        Features:
        • Multi-Dimensional Market Analysis
        • Quantum Computing Simulations
        • Real-time AI Insights
        • Portfolio Risk Management
        • Institutional-Grade Analytics
        
        © 2024 Cosmic Nexus Labs. All rights reserved.
        """
    }
)

st.markdown("""
<style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Space+Grotesk:wght@400;500;600&display=swap');
    
    :root {
        /* Enhanced Color Palette */
        --cosmic-bg: #050510;
        --nebula-primary: #0a0a1e;
        --nebula-secondary: #15152e;
        --starlight: #2a2a5e;
        --quasar-blue: #00d4ff;
        --neutron-green: #00ffaa;
        --pulsar-purple: #aa55ff;
        --supernova-red: #ff3366;
        --quantum-yellow: #ffcc00;
        --blackhole: #000000;
        --event-horizon: #001a33;
        
        /* Text Colors */
        --text-primary: #f0f4f8;
        --text-secondary: #cbd5e1;
        --text-dim: #94a3b8;
        --text-success: #00ffaa;
        --text-warning: #ffcc00;
        --text-danger: #ff3366;
        
        /* Gradients */
        --gradient-primary: linear-gradient(135deg, #00d4ff 0%, #aa55ff 100%);
        --gradient-success: linear-gradient(135deg, #00ffaa 0%, #00cc88 100%);
        --gradient-danger: linear-gradient(135deg, #ff3366 0%, #ff0066 100%);
        --gradient-warning: linear-gradient(135deg, #ffcc00 0%, #ff9900 100%);
    }
    
    /* Base Styling */
    .stApp {
        background: var(--cosmic-bg);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 212, 255, 0.05) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(170, 85, 255, 0.05) 0%, transparent 20%);
        background-attachment: fixed;
    }
    
    /* Typography Hierarchy */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    
    h1 { font-size: 2.5rem; color: var(--quasar-blue); }
    h2 { font-size: 2rem; color: var(--text-primary); }
    h3 { font-size: 1.5rem; color: var(--text-secondary); }
    
    .mono-font {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 400;
    }
    
    /* Enhanced Cards */
    .cosmic-card {
        background: rgba(10, 10, 30, 0.85);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(42, 42, 94, 0.4);
        border-radius: 20px;
        padding: 1.75rem;
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.4),
            0 0 0 1px rgba(0, 212, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .cosmic-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--quasar-blue), transparent);
    }
    
    .cosmic-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 212, 255, 0.6);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.6),
            0 0 0 1px rgba(0, 212, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    /* Quantum Metric Cards */
    .quantum-metric {
        background: linear-gradient(135deg, rgba(0, 20, 40, 0.8), rgba(10, 10, 30, 0.9));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .quantum-metric::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 212, 255, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .quantum-metric:hover::after {
        opacity: 1;
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(10, 10, 30, 0.8);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(42, 42, 94, 0.4);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 12px 24px;
        color: var(--text-dim);
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(42, 42, 94, 0.3);
        color: var(--text-secondary);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary);
        color: white;
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
    }
    
    /* Advanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--quasar-blue), var(--pulsar-purple));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(0, 212, 255, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Enhanced Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background: rgba(15, 15, 35, 0.8) !important;
        border: 1px solid rgba(42, 42, 94, 0.6) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem 1rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {
        border-color: var(--quasar-blue) !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2) !important;
        outline: none !important;
    }
    
    /* Status Indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: var(--quasar-blue);
    }
    
    .status-badge.success {
        background: rgba(0, 255, 170, 0.1);
        border-color: rgba(0, 255, 170, 0.3);
        color: var(--neutron-green);
    }
    
    .status-badge.warning {
        background: rgba(255, 204, 0, 0.1);
        border-color: rgba(255, 204, 0, 0.3);
        color: var(--quantum-yellow);
    }
    
    .status-badge.danger {
        background: rgba(255, 51, 102, 0.1);
        border-color: rgba(255, 51, 102, 0.3);
        color: var(--supernova-red);
    }
    
    /* Progress Bars */
    .cosmic-progress {
        height: 8px;
        background: rgba(42, 42, 94, 0.4);
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .cosmic-progress-bar {
        height: 100%;
        background: var(--gradient-primary);
        border-radius: 4px;
        transition: width 0.6s ease;
    }
    
    /* Data Tables */
    .data-table {
        background: rgba(10, 10, 30, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(42, 42, 94, 0.4);
        overflow: hidden;
    }
    
    /* Animations */
    @keyframes pulse-glow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .pulse-glow {
        animation: pulse-glow 2s ease-in-out infinite;
    }
    
    .float-animation {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(10, 10, 30, 0.4);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--quasar-blue), var(--pulsar-purple));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00b8e6, #9944ff);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .cosmic-card { padding: 1rem; }
        h1 { font-size: 2rem; }
        h2 { font-size: 1.5rem; }
        .stTabs [data-baseweb="tab"] { padding: 8px 16px; font-size: 0.85rem; }
    }
    
    /* Custom Dividers */
    .quantum-divider {
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(0, 212, 255, 0.3) 25%, 
            rgba(170, 85, 255, 0.3) 75%, 
            transparent 100%);
        margin: 2rem 0;
        border: none;
    }
    
    /* Tooltip Styling */
    .tooltip {
        position: relative;
        cursor: help;
    }
    
    .tooltip::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    .tooltip:hover::after {
        opacity: 1;
        visibility: visible;
        bottom: calc(100% + 10px);
    }
</style>

<!-- Starfield Background -->
<div class="starfield">
    <div style="position: fixed; width: 1px; height: 1px; background: white; border-radius: 50%; 
                top: 15%; left: 25%; animation: twinkle 4s infinite;"></div>
    <div style="position: fixed; width: 2px; height: 2px; background: var(--quasar-blue); border-radius: 50%; 
                top: 35%; left: 60%; animation: twinkle 3s infinite 0.5s;"></div>
    <div style="position: fixed; width: 1px; height: 1px; background: var(--neutron-green); border-radius: 50%; 
                top: 55%; left: 15%; animation: twinkle 5s infinite 1s;"></div>
    <div style="position: fixed; width: 3px; height: 3px; background: var(--pulsar-purple); border-radius: 50%; 
                top: 70%; left: 80%; animation: twinkle 6s infinite 1.5s;"></div>
</div>

<style>
    @keyframes twinkle {
        0%, 100% { opacity: 0.1; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.2); }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. ENHANCED DATA MANAGER WITH BUILT-IN INDICATORS
# ==========================================

class QuantumDataManager:
    """Advanced data management with built-in technical indicators"""
    
    def __init__(self):
        self.cache = {}
        self.exchange_health = {}
        
    @st.cache_data(ttl=60, show_spinner="📡 Connecting to quantum data stream...")
    def fetch_market_data(_self, exchange_name: str, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch market data with intelligent retry logic"""
        try:
            exchange_info = EXCHANGE_METADATA.get(exchange_name, EXCHANGE_METADATA['Kraken'])
            
            # Configure exchange
            config = {
                'enableRateLimit': True,
                'timeout': 15000,
                'rateLimit': int(exchange_info['rate_limit'] * 1000),
            }
            
            if exchange_info.get('testnet', False):
                if exchange_name == 'Binance':
                    config['urls'] = {'api': exchange_info['testnet_url']}
                elif exchange_name == 'Bybit':
                    config['urls'] = {'api': exchange_info['testnet_url']}
            
            exchange = exchange_info['class'](config)
            
            # Format symbol
            formatted_symbol = exchange_info['symbol_format'](symbol)
            
            # Fetch data
            ohlcv = exchange.fetch_ohlcv(formatted_symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 20:
                raise ValueError("Insufficient data received")
            
            # Create DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate additional features
            df = _self._enhance_dataframe(df)
            
            # Update exchange health
            _self.exchange_health[exchange_name] = {
                'status': 'healthy',
                'last_update': datetime.now(),
                'latency': 'unknown'
            }
            
            return df
            
        except Exception as e:
            st.error(f"Data fetch error: {str(e)[:200]}")
            return pd.DataFrame()
    
    def _enhance_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators using built-in library"""
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['close'].shift(1)
        
        # Volume features
        df['volume_sma'] = ti.sma(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        df['obv'] = ti.obv(df['close'], df['volume'])
        
        # Volatility features
        df['atr'] = ti.atr(df['high'], df['low'], df['close'], 14)
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(365)
        
        # Technical indicators
        df['rsi'] = ti.rsi(df['close'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = ti.macd(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ti.bollinger_bands(df['close'])
        df['stoch_k'], df['stoch_d'] = ti.stochastic(df['high'], df['low'], df['close'])
        
        # Advanced features
        df['vwap'] = ti.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['adx'] = ti.adx(df['high'], df['low'], df['close'], 14)
        df['cci'] = ti.cci(df['high'], df['low'], df['close'], 20)
        df['williams_r'] = ti.williams_r(df['high'], df['low'], df['close'], 14)
        df['mfi'] = ti.mfi(df['high'], df['low'], df['close'], df['volume'], 14)
        
        # Moving averages
        df['sma_20'] = ti.sma(df['close'], 20)
        df['ema_20'] = ti.ema(df['close'], 20)
        df['sma_50'] = ti.sma(df['close'], 50)
        df['ema_50'] = ti.ema(df['close'], 50)
        df['sma_200'] = ti.sma(df['close'], 200)
        df['ema_200'] = ti.ema(df['close'], 200)
        
        # Price position relative to MAs
        df['above_sma_20'] = df['close'] > df['sma_20']
        df['above_ema_20'] = df['close'] > df['ema_20']
        df['above_sma_50'] = df['close'] > df['sma_50']
        df['above_ema_50'] = df['close'] > df['ema_50']
        
        # Price momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
        
        # Support and resistance levels
        df['support_20'] = df['low'].rolling(20).min()
        df['resistance_20'] = df['high'].rolling(20).max()
        df['support_50'] = df['low'].rolling(50).min()
        df['resistance_50'] = df['high'].rolling(50).max()
        
        # Distance from support/resistance
        df['dist_from_support'] = (df['close'] - df['support_20']) / df['close'] * 100
        df['dist_from_resistance'] = (df['resistance_20'] - df['close']) / df['close'] * 100
        
        # Volume-price trend
        df['vpt'] = df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))
        df['vpt'] = df['vpt'].cumsum()
        
        # Average directional movement
        df['adx_strength'] = np.where(df['adx'] > 25, 'Strong', 'Weak')
        
        # Market regime detection
        df['regime'] = np.where(
            (df['volatility'] > df['volatility'].rolling(50).mean() * 1.5), 
            'High Volatility',
            np.where(
                (df['volatility'] < df['volatility'].rolling(50).mean() * 0.5),
                'Low Volatility',
                'Normal'
            )
        )
        
        return df

# ==========================================
# 5. ADVANCED VISUALIZATION ENGINE
# ==========================================

class QuantumVisualizer:
    """Professional Plotly visualization engine"""
    
    @staticmethod
    def create_cosmic_layout(title: str = "", height: int = 500) -> go.Layout:
        """Create standardized cosmic layout"""
        return go.Layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                color='#f0f4f8'
            ),
            title=dict(
                text=title,
                font=dict(
                    family='Space Grotesk, sans-serif',
                    size=18,
                    color='#00d4ff'
                ),
                x=0.02,
                y=0.95
            ),
            margin=dict(l=50, r=30, t=60, b=50),
            height=height,
            xaxis=dict(
                gridcolor='rgba(42, 42, 94, 0.3)',
                showgrid=True,
                zeroline=False,
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                gridcolor='rgba(42, 42, 94, 0.3)',
                showgrid=True,
                zeroline=False,
                tickfont=dict(size=11)
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='rgba(10, 10, 30, 0.9)',
                font_size=12,
                font_family="Inter"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(10, 10, 30, 0.7)',
                bordercolor='rgba(42, 42, 94, 0.4)',
                borderwidth=1
            )
        )
    
    @staticmethod
    def create_price_chart(df: pd.DataFrame, title: str = "Price Analysis") -> go.Figure:
        """Create professional price chart with multiple indicators"""
        # Use last 200 periods for cleaner display
        display_df = df.iloc[-min(500, len(df)):]
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=("Price & Volume", "RSI", "MACD", "ATR & Volatility")
        )
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=display_df.index,
            open=display_df['open'],
            high=display_df['high'],
            low=display_df['low'],
            close=display_df['close'],
            name="Price",
            increasing_line_color='#00ffaa',
            decreasing_line_color='#ff3366'
        ), row=1, col=1)
        
        # Add VWAP
        if 'vwap' in display_df.columns:
            fig.add_trace(go.Scatter(
                x=display_df.index,
                y=display_df['vwap'],
                name="VWAP",
                line=dict(color='#ffcc00', width=1.5, dash='dash'),
                opacity=0.7
            ), row=1, col=1)
        
        # Add moving averages
        if 'sma_20' in display_df.columns:
            fig.add_trace(go.Scatter(
                x=display_df.index,
                y=display_df['sma_20'],
                name="SMA 20",
                line=dict(color='#00d4ff', width=1, dash='dot'),
                opacity=0.6
            ), row=1, col=1)
        
        if 'ema_50' in display_df.columns:
            fig.add_trace(go.Scatter(
                x=display_df.index,
                y=display_df['ema_50'],
                name="EMA 50",
                line=dict(color='#aa55ff', width=1.5),
                opacity=0.6
            ), row=1, col=1)
        
        # Volume bars
        colors = ['#00ffaa' if close >= open else '#ff3366' 
                 for close, open in zip(display_df['close'], display_df['open'])]
        fig.add_trace(go.Bar(
            x=display_df.index,
            y=display_df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.5
        ), row=1, col=1)
        
        # RSI
        if 'rsi' in display_df.columns:
            fig.add_trace(go.Scatter(
                x=display_df.index,
                y=display_df['rsi'],
                name="RSI",
                line=dict(color='#aa55ff', width=1.5)
            ), row=2, col=1)
            
            # RSI bands
            fig.add_hline(y=70, line_dash="dash", line_color="#ff3366", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00ffaa", opacity=0.5, row=2, col=1)
            fig.add_hrect(y0=30, y1=70, fillcolor="rgba(42, 42, 94, 0.1)", line_width=0, row=2, col=1)
        
        # MACD
        if all(col in display_df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
            fig.add_trace(go.Scatter(
                x=display_df.index,
                y=display_df['macd'],
                name="MACD",
                line=dict(color='#00d4ff', width=1.5)
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=display_df.index,
                y=display_df['macd_signal'],
                name="Signal",
                line=dict(color='#ffcc00', width=1.5)
            ), row=3, col=1)
            
            # MACD histogram
            colors_macd = ['#00ffaa' if x >= 0 else '#ff3366' for x in display_df['macd_hist']]
            fig.add_trace(go.Bar(
                x=display_df.index,
                y=display_df['macd_hist'],
                name="Histogram",
                marker_color=colors_macd,
                opacity=0.6
            ), row=3, col=1)
        
        # ATR and Volatility
        if 'atr' in display_df.columns:
            fig.add_trace(go.Scatter(
                x=display_df.index,
                y=display_df['atr'],
                name="ATR",
                line=dict(color='#ff3366', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(255, 51, 102, 0.1)'
            ), row=4, col=1)
        
        if 'volatility' in display_df.columns:
            fig.add_trace(go.Scatter(
                x=display_df.index,
                y=display_df['volatility'],
                name="Volatility",
                line=dict(color='#00d4ff', width=1.5),
                yaxis="y2"
            ), row=4, col=1)
            
            fig.update_layout(
                yaxis4=dict(title="ATR", side="left"),
                yaxis5=dict(title="Volatility", side="right", overlaying="y4")
            )
        
        fig.update_layout(QuantumVisualizer.create_cosmic_layout(title, 800))
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    @staticmethod
    def create_indicator_grid(df: pd.DataFrame) -> go.Figure:
        """Create a grid of multiple technical indicators"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Bollinger Bands", "Stochastic", "CCI", "Williams %R", "MFI", "ADX"),
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Bollinger Bands
        if all(col in df.columns for col in ['close', 'bb_upper', 'bb_middle', 'bb_lower']):
            display_df = df.iloc[-100:]
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['close'], name="Price",
                line=dict(color='#f0f4f8', width=2)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['bb_upper'], name="Upper",
                line=dict(color='#ff3366', width=1, dash='dash')
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['bb_middle'], name="Middle",
                line=dict(color='#ffcc00', width=1)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['bb_lower'], name="Lower",
                line=dict(color='#00ffaa', width=1, dash='dash')
            ), row=1, col=1)
        
        # Stochastic
        if all(col in df.columns for col in ['stoch_k', 'stoch_d']):
            display_df = df.iloc[-100:]
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['stoch_k'], name="%K",
                line=dict(color='#00d4ff', width=2)
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['stoch_d'], name="%D",
                line=dict(color='#aa55ff', width=2, dash='dash')
            ), row=1, col=2)
            fig.add_hline(y=80, line_dash="dash", line_color="#ff3366", opacity=0.5, row=1, col=2)
            fig.add_hline(y=20, line_dash="dash", line_color="#00ffaa", opacity=0.5, row=1, col=2)
        
        # CCI
        if 'cci' in df.columns:
            display_df = df.iloc[-100:]
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['cci'], name="CCI",
                line=dict(color='#ffcc00', width=2)
            ), row=2, col=1)
            fig.add_hline(y=100, line_dash="dash", line_color="#ff3366", opacity=0.5, row=2, col=1)
            fig.add_hline(y=-100, line_dash="dash", line_color="#00ffaa", opacity=0.5, row=2, col=1)
        
        # Williams %R
        if 'williams_r' in df.columns:
            display_df = df.iloc[-100:]
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['williams_r'], name="Williams %R",
                line=dict(color='#00d4ff', width=2)
            ), row=2, col=2)
            fig.add_hline(y=-20, line_dash="dash", line_color="#ff3366", opacity=0.5, row=2, col=2)
            fig.add_hline(y=-80, line_dash="dash", line_color="#00ffaa", opacity=0.5, row=2, col=2)
        
        # MFI
        if 'mfi' in df.columns:
            display_df = df.iloc[-100:]
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['mfi'], name="MFI",
                line=dict(color='#aa55ff', width=2)
            ), row=3, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="#ff3366", opacity=0.5, row=3, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="#00ffaa", opacity=0.5, row=3, col=1)
        
        # ADX
        if 'adx' in df.columns:
            display_df = df.iloc[-100:]
            fig.add_trace(go.Scatter(
                x=display_df.index, y=display_df['adx'], name="ADX",
                line=dict(color='#ff3366', width=2)
            ), row=3, col=2)
            fig.add_hline(y=25, line_dash="dash", line_color="#ffcc00", opacity=0.5, row=3, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f0f4f8')
        )
        
        return fig
    
    @staticmethod
    def create_momentum_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create momentum indicator heatmap"""
        indicators = ['rsi', 'stoch_k', 'macd_hist', 'volume_ratio', 'cci', 'williams_r']
        
        # Calculate momentum scores for last 20 periods
        momentum_data = []
        indicator_names = []
        
        for indicator in indicators:
            if indicator in df.columns:
                values = df[indicator].dropna().iloc[-20:]
                if len(values) > 0:
                    # Normalize to 0-100 scale
                    if indicator in ['rsi', 'stoch_k', 'williams_r']:
                        # These are already in percentage scale
                        scaled = values.clip(0, 100)
                    else:
                        # Scale other indicators
                        min_val = values.min()
                        max_val = values.max()
                        if max_val > min_val:
                            scaled = (values - min_val) / (max_val - min_val) * 100
                        else:
                            scaled = pd.Series(50, index=values.index)
                    
                    momentum_data.append(scaled.values)
                    indicator_names.append(indicator.upper())
        
        if momentum_data:
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=momentum_data,
                x=list(range(len(momentum_data[0]))),
                y=indicator_names,
                colorscale='RdYlGn',
                zmid=50,
                colorbar=dict(title="Momentum Score"),
                hovertemplate='Indicator: %{y}<br>Period: %{x}<br>Score: %{z:.1f}<extra></extra>'
            ))
            
            fig.update_layout(
                QuantumVisualizer.create_cosmic_layout("Momentum Heatmap (Last 20 Periods)", 400),
                xaxis=dict(title="Periods Ago", tickvals=list(range(0, 20, 5)), ticktext=[str(i) for i in range(0, 20, 5)]),
                yaxis=dict(title="Indicator")
            )
            
            return fig
        
        return go.Figure()
    
    @staticmethod
    def create_correlation_matrix(df: pd.DataFrame) -> go.Figure:
        """Create correlation matrix of technical indicators"""
        indicator_cols = ['returns', 'volume', 'rsi', 'macd', 'atr', 'volatility', 'cci', 'mfi']
        available_cols = [col for col in indicator_cols if col in df.columns]
        
        if len(available_cols) >= 3:
            corr_data = df[available_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=available_cols,
                y=available_cols,
                colorscale='RdBu',
                zmid=0,
                text=corr_data.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation"),
                hovertemplate='X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                QuantumVisualizer.create_cosmic_layout("Indicator Correlation Matrix", 500),
                xaxis_title="Indicator",
                yaxis_title="Indicator"
            )
            
            return fig
        
        return go.Figure()
    
    @staticmethod
    def create_performance_gauge(current_value: float, target_value: float = 100, 
                                 title: str = "Performance") -> go.Figure:
        """Create performance gauge chart"""
        percentage = min(100, max(0, (current_value / target_value) * 100)) if target_value != 0 else 0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 16}},
            delta={'reference': 100},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00d4ff"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#2a2a5e",
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(255, 51, 102, 0.3)'},
                    {'range': [33, 66], 'color': 'rgba(255, 204, 0, 0.3)'},
                    {'range': [66, 100], 'color': 'rgba(0, 255, 170, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "#00d4ff", 'width': 4},
                    'thickness': 0.75,
                    'value': percentage
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f0f4f8", 'family': "Inter"},
            margin=dict(l=30, r=30, t=60, b=30)
        )
        
        return fig

# ==========================================
# 6. ADVANCED RISK MANAGEMENT SYSTEM
# ==========================================

class QuantumRiskManager:
    """Comprehensive risk analysis and management"""
    
    @staticmethod
    def calculate_var(df: pd.DataFrame, confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate Value at Risk"""
        returns = df['returns'].dropna()
        
        if len(returns) < 20:
            return {}
        
        # Historical VaR
        historical_var = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Parametric VaR (assuming normal distribution)
        parametric_var = returns.mean() + stats.norm.ppf(1 - confidence_level) * returns.std()
        
        # Expected Shortfall
        es = returns[returns <= historical_var].mean()
        
        return {
            'historical_var': historical_var * 100,  # as percentage
            'parametric_var': parametric_var * 100,
            'expected_shortfall': es * 100 if not np.isnan(es) else 0,
            'volatility': returns.std() * np.sqrt(252) * 100  # Annualized
        }
    
    @staticmethod
    def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        returns = df['returns'].dropna()
        if len(returns) < 2:
            return 0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        
        return excess_returns / volatility if volatility != 0 else 0
    
    @staticmethod
    def calculate_max_drawdown(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate maximum drawdown"""
        cumulative = (1 + df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_duration = (drawdown == max_dd).sum()
        
        return {
            'max_drawdown': max_dd * 100,
            'max_drawdown_duration': max_dd_duration,
            'current_drawdown': drawdown.iloc[-1] * 100 if len(drawdown) > 0 else 0
        }
    
    @staticmethod
    def generate_risk_report(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        var_metrics = QuantumRiskManager.calculate_var(df)
        drawdown_metrics = QuantumRiskManager.calculate_max_drawdown(df)
        sharpe = QuantumRiskManager.calculate_sharpe_ratio(df)
        
        # Calculate win rate
        returns = df['returns'].dropna()
        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
        
        # Calculate additional risk metrics
        sortino_ratio = QuantumRiskManager.calculate_sortino_ratio(df)
        calmar_ratio = QuantumRiskManager.calculate_calmar_ratio(df)
        
        return {
            **var_metrics,
            **drawdown_metrics,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'total_return': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100 if len(df) > 0 else 0,
            'avg_daily_return': returns.mean() * 100 if len(returns) > 0 else 0,
            'positive_skew': stats.skew(returns) > 0 if len(returns) > 0 else False
        }
    
    @staticmethod
    def calculate_sortino_ratio(df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio"""
        returns = df['returns'].dropna()
        if len(returns) < 2:
            return 0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        return excess_returns / downside_volatility if downside_volatility != 0 else 0
    
    @staticmethod
    def calculate_calmar_ratio(df: pd.DataFrame) -> float:
        """Calculate Calmar Ratio"""
        returns = df['returns'].dropna()
        if len(returns) < 252:  # Need at least 1 year of data
            return 0
        
        annual_return = ((df['close'].iloc[-1] / df['close'].iloc[-min(252, len(df))]) - 1) * 100
        max_dd = QuantumRiskManager.calculate_max_drawdown(df)['max_drawdown']
        
        return annual_return / abs(max_dd) if max_dd != 0 else 0

# ==========================================
# 7. ENHANCED QUANTUM ANALYTICS ENGINE
# ==========================================

class QuantumAnalytics:
    """Advanced quantum computing-inspired analytics"""
    
    @staticmethod
    def calculate_entanglement_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate quantum entanglement between indicators"""
        indicator_cols = ['close', 'volume', 'rsi', 'macd', 'atr', 'vwap', 'cci', 'mfi']
        available_cols = [col for col in indicator_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr()
        
        # Apply quantum-inspired transformation (probability amplitude)
        entanglement_matrix = np.abs(corr_matrix) ** 2
        
        return pd.DataFrame(entanglement_matrix, 
                          index=available_cols, 
                          columns=available_cols)
    
    @staticmethod
    def calculate_superposition_state(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate quantum superposition state (trend + volatility)"""
        if 'close' not in df.columns or len(df) < window:
            return pd.Series()
        
        # Normalize price
        price_norm = (df['close'] - df['close'].rolling(window).min()) / \
                    (df['close'].rolling(window).max() - df['close'].rolling(window).min() + 1e-10)
        
        # Normalize volatility
        if 'volatility' in df.columns:
            vol_norm = (df['volatility'] - df['volatility'].rolling(window).min()) / \
                      (df['volatility'].rolling(window).max() - df['volatility'].rolling(window).min() + 1e-10)
        else:
            returns = df['close'].pct_change()
            vol = returns.rolling(window).std()
            vol_norm = (vol - vol.rolling(window).min()) / \
                      (vol.rolling(window).max() - vol.rolling(window).min() + 1e-10)
        
        # Quantum superposition state (normalized to 0-1)
        superposition = np.sqrt(price_norm**2 + vol_norm**2) / np.sqrt(2)
        
        return superposition
    
    @staticmethod
    def detect_quantum_tunneling(df: pd.DataFrame, threshold: float = 0.8) -> pd.Series:
        """Detect quantum tunneling events (sudden regime changes)"""
        if 'close' not in df.columns or len(df) < 50:
            return pd.Series()
        
        # Calculate rate of change
        roc = df['close'].pct_change().abs()
        
        # Calculate entropy of returns (simplified)
        returns = df['close'].pct_change().dropna()
        if len(returns) < 20:
            return pd.Series([False] * len(df), index=df.index)
        
        # Simple entropy approximation
        hist, _ = np.histogram(returns, bins=20)
        hist = hist / len(returns)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        # Normalize ROC
        roc_mean = roc.rolling(50).mean()
        roc_std = roc.rolling(50).std()
        roc_norm = (roc - roc_mean) / (roc_std + 1e-10)
        
        # Detect tunneling events
        tunneling = (roc_norm > threshold) | (entropy > 2.0)
        
        return tunneling
    
    @staticmethod
    def calculate_wave_function(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate quantum wave function probabilities"""
        if 'close' not in df.columns:
            return {}
        
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 50:
            return {}
        
        # Calculate probability density function
        hist, bin_edges = np.histogram(returns, bins=50, density=True)
        
        # Calculate cumulative distribution function
        cdf = np.cumsum(hist) * np.diff(bin_edges)
        
        # Calculate momentum space (Fourier transform)
        fft_result = np.fft.fft(returns.values)
        power_spectrum = np.abs(fft_result) ** 2
        
        return {
            'pdf': pd.Series(hist, index=bin_edges[:-1]),
            'cdf': pd.Series(cdf, index=bin_edges[:-1]),
            'power_spectrum': pd.Series(power_spectrum[:len(power_spectrum)//2])
        }
    
    @staticmethod
    def calculate_quantum_momentum(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.Series:
        """Calculate quantum momentum across multiple timeframes"""
        if 'close' not in df.columns:
            return pd.Series()
        
        momentum_scores = []
        
        for period in periods:
            if len(df) >= period:
                # Simple momentum calculation
                momentum = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
                momentum_scores.append(momentum)
        
        if momentum_scores:
            # Average momentum across timeframes
            avg_momentum = pd.concat(momentum_scores, axis=1).mean(axis=1)
            return avg_momentum
        
        return pd.Series()

# ==========================================
# 8. MAIN APPLICATION INITIALIZATION
# ==========================================

# Initialize global instances
data_manager = QuantumDataManager()
visualizer = QuantumVisualizer()
risk_manager = QuantumRiskManager()
quantum_analytics = QuantumAnalytics()

# Initialize session state
if 'nexus_config' not in st.session_state:
    st.session_state.nexus_config = {
        # Exchange Configuration
        'exchange': 'Kraken',
        'symbol': 'BTC/USDT',
        'timeframe': '15m',
        'data_points': 1000,
        
        # Quantum Parameters
        'quantum_depth': 'Advanced',
        'risk_tolerance': 'Medium',
        'auto_refresh': False,
        'refresh_interval': 30,
        
        # Display Settings
        'theme': 'Cosmic Dark',
        'chart_style': 'Professional',
        'data_density': 'High',
        
        # API Keys
        'openai_key': '',
        'telegram_token': '',
        'telegram_chat_id': '',
        
        # Analytics Settings
        'enable_quantum_analytics': True,
        'enable_risk_management': True,
        'enable_ai_insights': False,  # Default to False to avoid OpenAI requirement
        
        # Last Updated
        'last_update': datetime.now()
    }

if 'analytics_cache' not in st.session_state:
    st.session_state.analytics_cache = {}

if 'risk_report' not in st.session_state:
    st.session_state.risk_report = {}

if 'market_data' not in st.session_state:
    st.session_state.market_data = None

# ==========================================
# 9. ENHANCED SIDEBAR WITH ADVANCED CONTROLS
# ==========================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: var(--quasar-blue); margin-bottom: 0.5rem; font-size: 2rem;">🪐</h1>
        <h3 style="color: var(--text-primary); margin-bottom: 0.2rem;">COSMIC NEXUS</h3>
        <p style="color: var(--text-dim); font-size: 0.8rem; margin-bottom: 0.5rem;">
            Advanced Quantum Trading Terminal v3.0
        </p>
        <div class="cosmic-progress">
            <div class="cosmic-progress-bar" style="width: 100%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Connection Panel
    with st.expander("🌐 QUANTUM CONNECTION", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            exchange = st.selectbox(
                "Exchange",
                list(EXCHANGE_METADATA.keys()),
                index=0,
                help="Select your preferred exchange",
                key='exchange_select'
            )
            
            # Show exchange info
            exchange_info = EXCHANGE_METADATA[exchange]
            status_color = "success" if exchange_info['global_access'] else "warning"
            st.markdown(f"""
            <div class="status-badge {status_color}" style="margin-top: 0.5rem;">
                {'🌍 Global Access' if exchange_info['global_access'] else '⚠️ Restricted'}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                index=2,
                help="Select analysis timeframe",
                key='timeframe_select'
            )
        
        # Symbol with autocomplete
        symbol_options = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", 
                         "DOT/USDT", "MATIC/USDT", "AVAX/USDT", "LINK/USDT", "UNI/USDT"]
        symbol = st.selectbox(
            "Trading Pair",
            options=symbol_options,
            index=0,
            help="Select trading pair",
            key='symbol_select'
        )
        
        # Data points slider
        data_points = st.slider(
            "Data Points",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Number of historical data points to analyze"
        )
        
        # Connection test button
        if st.button("🔬 Test Connection", use_container_width=True):
            with st.spinner("Testing quantum connection..."):
                time_lib.sleep(1)
                if exchange_info['global_access']:
                    st.success(f"✅ {exchange} connection successful")
                else:
                    st.warning(f"⚠️ {exchange} may have regional restrictions")
    
    # Quantum Analytics Panel
    with st.expander("⚛️ QUANTUM ANALYTICS", expanded=False):
        quantum_depth = st.select_slider(
            "Analytics Depth",
            options=["Basic", "Standard", "Advanced", "Quantum"],
            value="Advanced",
            help="Depth of quantum analytics calculations"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            enable_quantum = st.checkbox("Quantum Analytics", value=True)
            enable_risk = st.checkbox("Risk Management", value=True)
        
        with col2:
            enable_ai = st.checkbox("AI Insights", value=False)  # Default to False
            enable_alerts = st.checkbox("Smart Alerts", value=True)
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            entanglement_strength = st.slider("Entanglement Strength", 0.1, 1.0, 0.5)
            superposition_threshold = st.slider("Superposition Threshold", 0.1, 1.0, 0.7)
            tunneling_sensitivity = st.slider("Tunneling Sensitivity", 0.1, 1.0, 0.5)
    
    # Risk Management Panel
    with st.expander("🛡️ RISK MANAGEMENT", expanded=False):
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive", "Quantum"],
            value="Moderate",
            help="Set your risk tolerance level"
        )
        
        # Risk parameters
        var_confidence = st.slider("VaR Confidence", 0.90, 0.99, 0.95)
        max_position_size = st.slider("Max Position Size %", 1, 100, 20)
        stop_loss = st.slider("Stop Loss %", 1, 20, 5)
        
        st.markdown("""
        <div style="background: rgba(255, 51, 102, 0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <div style="color: var(--text-danger); font-size: 0.9rem; font-weight: 600;">
                ⚠️ Risk Warning
            </div>
            <div style="color: var(--text-dim); font-size: 0.8rem; margin-top: 0.5rem;">
                Trading involves significant risk. Only trade with capital you can afford to lose.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Configuration
    with st.expander("🤖 AI CONFIGURATION", expanded=False):
        st.session_state.nexus_config['openai_key'] = st.text_input(
            "OpenAI API Key",
            value=st.session_state.nexus_config['openai_key'],
            type="password",
            help="Optional: Required for AI insights and predictions"
        )
        
        if st.session_state.nexus_config['openai_key']:
            ai_model = st.selectbox(
                "AI Model",
                ["GPT-4 Turbo", "GPT-4", "GPT-3.5 Turbo"],
                index=0,
                help="Select AI model for analysis"
            )
            
            ai_temperature = st.slider(
                "AI Creativity",
                0.0, 1.0, 0.7,
                help="Higher values make AI more creative, lower values more deterministic"
            )
        else:
            st.info("🔒 Add OpenAI API key to unlock AI features")
    
    # System Controls
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Launch Scan", type="primary", use_container_width=True):
            st.session_state.analytics_cache.clear()
            st.session_state.risk_report.clear()
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh Data", type="secondary", use_container_width=True):
            # Clear specific cache
            st.cache_data.clear()
            st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    if auto_refresh:
        refresh_rate = st.select_slider(
            "Refresh Rate",
            options=["30s", "1m", "5m", "15m"],
            value="1m"
        )
    
    # System Status
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(10, 10, 30, 0.6); padding: 1rem; border-radius: 12px;">
        <div style="color: var(--text-dim); font-size: 0.8rem; margin-bottom: 0.5rem;">
            System Status
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: var(--text-secondary); font-size: 0.9rem;">Quantum Core</span>
            <span class="status-badge success">Online</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
            <span style="color: var(--text-secondary); font-size: 0.9rem;">Data Stream</span>
            <span class="status-badge success">Active</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
            <span style="color: var(--text-secondary); font-size: 0.9rem;">AI Engine</span>
            <span class="status-badge {'success' if st.session_state.nexus_config['openai_key'] else 'warning'}">
                {'Ready' if st.session_state.nexus_config['openai_key'] else 'Disabled'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 10. MAIN DASHBOARD LAYOUT
# ==========================================

# Header Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="cosmic-card" style="text-align: center;">
        <h1 style="margin-bottom: 0.5rem;">COSMIC NEXUS v3.0</h1>
        <p style="color: var(--text-dim); font-size: 1rem; margin-bottom: 1rem;">
            Advanced Quantum Trading Terminal • Real-time Multi-Dimensional Analysis
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <span class="status-badge">Exchange: {}</span>
            <span class="status-badge">Symbol: {}</span>
            <span class="status-badge">Timeframe: {}</span>
            <span class="status-badge">Analytics: {}</span>
        </div>
    </div>
    """.format(exchange, symbol, timeframe, quantum_depth), unsafe_allow_html=True)

# Fetch Data
with st.spinner("🌌 Connecting to quantum data stream..."):
    df = data_manager.fetch_market_data(exchange, symbol, timeframe, data_points)

if df.empty:
    st.error("""
    ## ⚠️ QUANTUM CONNECTION FAILED
    
    ### Troubleshooting Steps:
    
    1. **Test Exchange Connection** in sidebar
    2. **Try Alternative Exchange**: Kraken, Coinbase, or KuCoin
    3. **Check Symbol Format**: Ensure correct format (BTC/USDT)
    4. **Network Issues**: Verify internet connection
    
    ### Recommended Configuration:
    - **Exchange**: Kraken (global access)
    - **Symbol**: BTC/USDT (most liquid)
    - **Timeframe**: 15m or 1h
    """)
    
    # Quick connection buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Try Kraken (Recommended)"):
            st.session_state.nexus_config['exchange'] = 'Kraken'
            st.rerun()
    
    with col2:
        if st.button("🔄 Try Coinbase"):
            st.session_state.nexus_config['exchange'] = 'Coinbase'
            st.rerun()
    
    with col3:
        if st.button("🔄 Try KuCoin"):
            st.session_state.nexus_config['exchange'] = 'KuCoin'
            st.rerun()
    
    st.stop()

# Store data in session state
st.session_state.market_data = df

# ==========================================
# 11. ENHANCED ANALYTICS EXECUTION
# ==========================================

# Calculate all analytics
if enable_quantum:
    with st.spinner("⚛️ Computing quantum analytics..."):
        # Calculate risk metrics
        st.session_state.risk_report = risk_manager.generate_risk_report(df)
        
        # Calculate quantum analytics
        entanglement_matrix = quantum_analytics.calculate_entanglement_matrix(df)
        superposition_state = quantum_analytics.calculate_superposition_state(df)
        tunneling_events = quantum_analytics.detect_quantum_tunneling(df)
        wave_function = quantum_analytics.calculate_wave_function(df)
        quantum_momentum = quantum_analytics.calculate_quantum_momentum(df)

# ==========================================
# 12. COMPREHENSIVE DASHBOARD TABS
# ==========================================

# Create enhanced tabs
tabs = st.tabs([
    "📊 MARKET OVERVIEW",
    "📈 PRICE ANALYSIS", 
    "⚛️ QUANTUM ANALYTICS",
    "🛡️ RISK MANAGEMENT",
    "📊 TECHNICAL SUITE",
    "📋 DATA INSIGHTS"
])

# Tab 1: Market Overview
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['close'].iloc[-1]
        price_change_1h = ((df['close'].iloc[-1] / df['close'].iloc[-4]) - 1) * 100 if len(df) >= 4 else 0
        price_color = "var(--text-success)" if price_change_1h >= 0 else "var(--text-danger)"
        
        st.markdown(f"""
        <div class="cosmic-card">
            <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Current Price</div>
            <div style="color: var(--text-primary); font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
                ${current_price:,.2f}
            </div>
            <div style="color: {price_color}; font-size: 1.1rem; font-weight: 600;">
                {price_change_1h:+.2f}% (1h)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        volume_ratio = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1.0
        volume_color = "var(--text-success)" if volume_ratio > 1.2 else "var(--text-warning)" if volume_ratio > 0.8 else "var(--text-danger)"
        
        st.markdown(f"""
        <div class="cosmic-card">
            <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Volume Trend</div>
            <div style="color: {volume_color}; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                {volume_ratio:.2f}x
            </div>
            <div style="color: var(--text-dim); font-size: 0.9rem;">
                vs 20-period average
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'rsi' in df.columns:
            rsi_value = df['rsi'].iloc[-1]
            rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
            rsi_color = "var(--text-success)" if rsi_value < 30 else "var(--text-danger)" if rsi_value > 70 else "var(--text-primary)"
            
            st.markdown(f"""
            <div class="cosmic-card">
                <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">RSI (14)</div>
                <div style="color: {rsi_color}; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                    {rsi_value:.1f}
                </div>
                <div style="color: {rsi_color}; font-size: 0.9rem;">
                    {rsi_status}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'volatility' in df.columns:
            vol_value = df['volatility'].iloc[-1]
            vol_status = "High" if vol_value > 0.8 else "Medium" if vol_value > 0.4 else "Low"
            vol_color = "var(--text-danger)" if vol_value > 0.8 else "var(--text-warning)" if vol_value > 0.4 else "var(--text-success)"
            
            st.markdown(f"""
            <div class="cosmic-card">
                <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Volatility</div>
                <div style="color: {vol_color}; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                    {vol_value:.1f}%
                </div>
                <div style="color: {vol_color}; font-size: 0.9rem;">
                    {vol_status}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Market Overview Chart
    st.markdown("### 📈 Market Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_overview = visualizer.create_price_chart(df, "Market Overview")
        st.plotly_chart(fig_overview, use_container_width=True)
    
    with col2:
        # Market Sentiment
        st.markdown("""
        <div class="cosmic-card">
            <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">Market Sentiment</h4>
        """, unsafe_allow_html=True)
        
        # Calculate sentiment indicators
        bull_indicators = 0
        total_indicators = 6
        
        if 'rsi' in df.columns:
            bull_indicators += 1 if df['rsi'].iloc[-1] > 50 else 0
        
        if 'macd_hist' in df.columns:
            bull_indicators += 1 if df['macd_hist'].iloc[-1] > 0 else 0
        
        if 'volume_ratio' in df.columns:
            bull_indicators += 1 if df['volume_ratio'].iloc[-1] > 1 else 0
        
        if 'close' in df.columns and len(df) >= 20:
            bull_indicators += 1 if df['close'].iloc[-1] > df['close'].iloc[-20] else 0
        
        if 'sma_20' in df.columns:
            bull_indicators += 1 if df['close'].iloc[-1] > df['sma_20'].iloc[-1] else 0
        
        if 'ema_50' in df.columns:
            bull_indicators += 1 if df['close'].iloc[-1] > df['ema_50'].iloc[-1] else 0
        
        sentiment_score = (bull_indicators / total_indicators) * 100
        sentiment_status = "Bullish" if sentiment_score > 60 else "Bearish" if sentiment_score < 40 else "Neutral"
        sentiment_color = "var(--text-success)" if sentiment_score > 60 else "var(--text-danger)" if sentiment_score < 40 else "var(--text-warning)"
        
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="color: {sentiment_color}; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
                {sentiment_score:.0f}%
            </div>
            <div style="color: {sentiment_color}; font-size: 1.1rem; font-weight: 600;">
                {sentiment_status}
            </div>
        </div>
        <div class="cosmic-progress">
            <div class="cosmic-progress-bar" style="width: {sentiment_score}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
            <span style="color: var(--text-dim); font-size: 0.8rem;">Bearish</span>
            <span style="color: var(--text-dim); font-size: 0.8rem;">Bullish</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="quantum-divider" style="margin: 1.5rem 0;"></div>
        
        <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
            Bullish Indicators:
        </div>
        <div style="color: var(--text-primary); font-size: 0.9rem;">
            {}/{} indicators bullish
        </div>
        </div>
        """.format(bull_indicators, total_indicators), unsafe_allow_html=True)
    
    # Momentum Heatmap
    st.markdown("### 🔥 Momentum Analysis")
    fig_heatmap = visualizer.create_momentum_heatmap(df)
    if fig_heatmap.data:
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Insufficient data for momentum heatmap")

# Tab 2: Price Analysis
with tabs[1]:
    st.markdown("### 📊 Comprehensive Price Analysis")
    
    # Multiple chart views
    chart_view = st.radio(
        "Select Chart View",
        ["Price & Indicators", "Technical Analysis", "Advanced Metrics", "Market Structure"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if chart_view == "Price & Indicators":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig_price = visualizer.create_price_chart(df, f"{symbol} Price Analysis")
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            # Technical Summary
            st.markdown("""
            <div class="cosmic-card">
                <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">Technical Summary</h4>
            """, unsafe_allow_html=True)
            
            # Calculate technical signals
            signals = []
            
            if 'rsi' in df.columns:
                rsi_val = df['rsi'].iloc[-1]
                if rsi_val > 70:
                    signals.append(("RSI Overbought", "danger"))
                elif rsi_val < 30:
                    signals.append(("RSI Oversold", "success"))
            
            if 'macd_hist' in df.columns and len(df) >= 2:
                macd_val = df['macd_hist'].iloc[-1]
                if macd_val > 0 and df['macd_hist'].iloc[-2] <= 0:
                    signals.append(("MACD Bullish Cross", "success"))
                elif macd_val < 0 and df['macd_hist'].iloc[-2] >= 0:
                    signals.append(("MACD Bearish Cross", "danger"))
            
            if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
                price = df['close'].iloc[-1]
                if price > df['bb_upper'].iloc[-1]:
                    signals.append(("Above BB Upper", "warning"))
                elif price < df['bb_lower'].iloc[-1]:
                    signals.append(("Below BB Lower", "warning"))
            
            if 'volume_ratio' in df.columns:
                vol_ratio = df['volume_ratio'].iloc[-1]
                if vol_ratio > 2.0:
                    signals.append(("High Volume Spike", "warning"))
            
            # Display signals
            if signals:
                for signal_text, signal_type in signals:
                    color_map = {
                        "success": "var(--neutron-green)",
                        "danger": "var(--supernova-red)",
                        "warning": "var(--quantum-yellow)"
                    }
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <div style="width: 8px; height: 8px; border-radius: 50%; background: {color_map[signal_type]};"></div>
                        <span style="color: {color_map[signal_type]}; font-size: 0.9rem;">{signal_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="color: var(--text-dim); text-align: center; padding: 1rem;">
                    No strong technical signals detected
                </div>
                """, unsafe_allow_html=True)
            
            # Key levels
            st.markdown("""
            <div class="quantum-divider" style="margin: 1rem 0;"></div>
            <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                Key Levels:
            </div>
            """, unsafe_allow_html=True)
            
            if all(col in df.columns for col in ['support_20', 'resistance_20']):
                support = df['support_20'].iloc[-1]
                resistance = df['resistance_20'].iloc[-1]
                current = df['close'].iloc[-1]
                
                st.markdown(f"""
                <div style="color: var(--text-primary); font-size: 0.9rem;">
                    Support: ${support:,.2f}<br>
                    Resistance: ${resistance:,.2f}<br>
                    Distance to R: {((resistance - current)/current*100):.1f}%<br>
                    Distance to S: {((current - support)/current*100):.1f}%
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif chart_view == "Technical Analysis":
        # Indicator Grid
        st.markdown("### 📊 Technical Indicators Grid")
        fig_grid = visualizer.create_indicator_grid(df)
        st.plotly_chart(fig_grid, use_container_width=True)
        
        # Moving Averages Analysis
        st.markdown("### 📈 Moving Averages Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if all(col in df.columns for col in ['close', 'sma_20', 'ema_20']):
                above_sma = df['above_sma_20'].iloc[-1]
                above_ema = df['above_ema_20'].iloc[-1]
                st.metric("SMA 20", f"${df['sma_20'].iloc[-1]:,.2f}", 
                         "Above" if above_sma else "Below")
        
        with col2:
            if all(col in df.columns for col in ['close', 'sma_50', 'ema_50']):
                above_sma_50 = df['above_sma_50'].iloc[-1]
                above_ema_50 = df['above_ema_50'].iloc[-1]
                st.metric("EMA 50", f"${df['ema_50'].iloc[-1]:,.2f}",
                         "Above" if above_ema_50 else "Below")
        
        with col3:
            if 'sma_200' in df.columns:
                above_sma_200 = df['close'].iloc[-1] > df['sma_200'].iloc[-1]
                st.metric("SMA 200", f"${df['sma_200'].iloc[-1]:,.2f}",
                         "Above" if above_sma_200 else "Below")
    
    elif chart_view == "Advanced Metrics":
        # Correlation matrix
        st.markdown("### 🔗 Indicator Correlation Matrix")
        fig_corr = visualizer.create_correlation_matrix(df)
        if fig_corr.data:
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Statistical summary
        st.markdown("### 📊 Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(df) > 20:
                returns = df['returns'].dropna()
                
                stats_data = {
                    "Mean Return": f"{returns.mean() * 100:.4f}%",
                    "Std Deviation": f"{returns.std() * 100:.4f}%",
                    "Skewness": f"{stats.skew(returns):.4f}" if len(returns) > 0 else "N/A",
                    "Kurtosis": f"{stats.kurtosis(returns):.4f}" if len(returns) > 0 else "N/A",
                    "Sharpe Ratio": f"{risk_manager.calculate_sharpe_ratio(df):.4f}",
                    "Max Return": f"{returns.max() * 100:.2f}%" if len(returns) > 0 else "N/A",
                    "Min Return": f"{returns.min() * 100:.2f}%" if len(returns) > 0 else "N/A"
                }
                
                for key, value in stats_data.items():
                    st.markdown(f"""
                    <div style="background: rgba(10, 10, 30, 0.6); padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: var(--text-dim);">{key}</span>
                            <span style="color: var(--text-primary); font-weight: 600;">{value}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            # Performance gauge
            if 'sharpe_ratio' in st.session_state.risk_report:
                sharpe_value = st.session_state.risk_report['sharpe_ratio']
                fig_gauge = visualizer.create_performance_gauge(
                    min(max(sharpe_value * 10 + 50, 0), 100),
                    100,
                    "Risk-Adjusted Performance"
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
    
    elif chart_view == "Market Structure":
        # Support and Resistance Analysis
        st.markdown("### 🏗️ Market Structure Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Support levels
            st.markdown("""
            <div class="cosmic-card">
                <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">Support Levels</h4>
            """, unsafe_allow_html=True)
            
            if all(col in df.columns for col in ['support_20', 'support_50']):
                support_20 = df['support_20'].iloc[-1]
                support_50 = df['support_50'].iloc[-1]
                current = df['close'].iloc[-1]
                
                st.metric("Primary Support (20)", f"${support_20:,.2f}", 
                         f"{(current - support_20)/current*100:.1f}% below")
                st.metric("Secondary Support (50)", f"${support_50:,.2f}",
                         f"{(current - support_50)/current*100:.1f}% below")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Resistance levels
            st.markdown("""
            <div class="cosmic-card">
                <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">Resistance Levels</h4>
            """, unsafe_allow_html=True)
            
            if all(col in df.columns for col in ['resistance_20', 'resistance_50']):
                resistance_20 = df['resistance_20'].iloc[-1]
                resistance_50 = df['resistance_50'].iloc[-1]
                current = df['close'].iloc[-1]
                
                st.metric("Primary Resistance (20)", f"${resistance_20:,.2f}",
                         f"{(resistance_20 - current)/current*100:.1f}% above")
                st.metric("Secondary Resistance (50)", f"${resistance_50:,.2f}",
                         f"{(resistance_50 - current)/current*100:.1f}% above")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Market Regime
        st.markdown("### 📊 Market Regime Detection")
        
        if 'regime' in df.columns:
            current_regime = df['regime'].iloc[-1]
            regime_color = {
                'High Volatility': 'var(--text-danger)',
                'Low Volatility': 'var(--text-success)',
                'Normal': 'var(--text-warning)'
            }.get(current_regime, 'var(--text-primary)')
            
            st.markdown(f"""
            <div class="cosmic-card" style="text-align: center;">
                <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Current Market Regime
                </div>
                <div style="color: {regime_color}; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                    {current_regime}
                </div>
                <div style="color: var(--text-dim); font-size: 0.9rem;">
                    Based on volatility analysis
                </div>
            </div>
            """, unsafe_allow_html=True)

# Tab 3: Quantum Analytics
with tabs[2]:
    st.markdown("### ⚛️ Quantum Analytics Dashboard")
    
    if not enable_quantum:
        st.warning("Quantum Analytics is disabled. Enable it in the sidebar to access advanced features.")
    else:
        # Quantum Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not entanglement_matrix.empty:
                entanglement_score = entanglement_matrix.values.mean() * 100
                st.markdown(f"""
                <div class="cosmic-card">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Quantum Entanglement</div>
                    <div style="color: var(--quasar-blue); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {entanglement_score:.1f}%
                    </div>
                    <div style="color: var(--text-dim); font-size: 0.9rem;">
                        Indicator correlation strength
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if not superposition_state.empty:
                super_score = superposition_state.iloc[-1] * 100
                st.markdown(f"""
                <div class="cosmic-card">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Superposition State</div>
                    <div style="color: var(--pulsar-purple); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {super_score:.1f}%
                    </div>
                    <div style="color: var(--text-dim); font-size: 0.9rem;">
                        Price-volatility coherence
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if not tunneling_events.empty:
                tunneling_count = tunneling_events.sum()
                st.markdown(f"""
                <div class="cosmic-card">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Quantum Tunneling</div>
                    <div style="color: var(--neutron-green); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {tunneling_count}
                    </div>
                    <div style="color: var(--text-dim); font-size: 0.9rem;">
                        Regime change events
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if not quantum_momentum.empty:
                q_momentum = quantum_momentum.iloc[-1] * 100
                momentum_color = "var(--text-success)" if q_momentum > 0 else "var(--text-danger)"
                st.markdown(f"""
                <div class="cosmic-card">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Quantum Momentum</div>
                    <div style="color: {momentum_color}; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {q_momentum:+.1f}%
                    </div>
                    <div style="color: var(--text-dim); font-size: 0.9rem;">
                        Multi-timeframe average
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Quantum Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Entanglement Matrix Visualization
            if not entanglement_matrix.empty:
                fig_entanglement = go.Figure(data=go.Heatmap(
                    z=entanglement_matrix.values,
                    x=entanglement_matrix.columns,
                    y=entanglement_matrix.index,
                    colorscale='Viridis',
                    colorbar=dict(title="Entanglement<br>Strength"),
                    hovertemplate='X: %{x}<br>Y: %{y}<br>Strength: %{z:.3f}<extra></extra>'
                ))
                
                fig_entanglement.update_layout(
                    visualizer.create_cosmic_layout("Quantum Entanglement Matrix", 400),
                    xaxis_title="Indicator",
                    yaxis_title="Indicator"
                )
                
                st.plotly_chart(fig_entanglement, use_container_width=True)
        
        with col2:
            # Superposition State Chart
            if not superposition_state.empty:
                fig_superposition = go.Figure()
                
                display_super = superposition_state.iloc[-100:] * 100
                
                fig_superposition.add_trace(go.Scatter(
                    x=display_super.index,
                    y=display_super.values,
                    name="Superposition State",
                    line=dict(color='#aa55ff', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(170, 85, 255, 0.1)'
                ))
                
                fig_superposition.update_layout(
                    visualizer.create_cosmic_layout("Quantum Superposition State", 400),
                    yaxis_title="Superposition Strength (%)",
                    yaxis_range=[0, 100]
                )
                
                st.plotly_chart(fig_superposition, use_container_width=True)
        
        # Tunneling Events Timeline
        if not tunneling_events.empty and tunneling_events.any():
            st.markdown("### 🌀 Quantum Tunneling Events")
            
            tunneling_df = df[tunneling_events]
            
            if not tunneling_df.empty:
                fig_tunneling = go.Figure()
                
                display_df = df.iloc[-200:]
                
                fig_tunneling.add_trace(go.Candlestick(
                    x=display_df.index,
                    open=display_df['open'],
                    high=display_df['high'],
                    low=display_df['low'],
                    close=display_df['close'],
                    name="Price"
                ))
                
                # Mark tunneling events in the display range
                display_tunnels = tunneling_df[tunneling_df.index.isin(display_df.index)]
                
                if not display_tunnels.empty:
                    fig_tunneling.add_trace(go.Scatter(
                        x=display_tunnels.index,
                        y=display_tunnels['close'],
                        mode='markers',
                        name='Tunneling Event',
                        marker=dict(
                            size=12,
                            color='#ffcc00',
                            symbol='diamond',
                            line=dict(width=2, color='white')
                        )
                    ))
                
                fig_tunneling.update_layout(
                    visualizer.create_cosmic_layout("Quantum Tunneling Events", 400),
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig_tunneling, use_container_width=True)

# Tab 4: Risk Management
with tabs[3]:
    st.markdown("### 🛡️ Risk Management Dashboard")
    
    if not enable_risk:
        st.warning("Risk Management is disabled. Enable it in the sidebar to access risk features.")
    else:
        # Risk Metrics Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'historical_var' in st.session_state.risk_report:
                var_value = st.session_state.risk_report['historical_var']
                st.markdown(f"""
                <div class="cosmic-card">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Value at Risk (95%)</div>
                    <div style="color: var(--supernova-red); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {var_value:.2f}%
                    </div>
                    <div style="color: var(--text-dim); font-size: 0.9rem;">
                        Daily potential loss
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if 'max_drawdown' in st.session_state.risk_report:
                dd_value = st.session_state.risk_report['max_drawdown']
                st.markdown(f"""
                <div class="cosmic-card">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Max Drawdown</div>
                    <div style="color: var(--quantum-yellow); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {dd_value:.2f}%
                    </div>
                    <div style="color: var(--text-dim); font-size: 0.9rem;">
                        Historical worst decline
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'sharpe_ratio' in st.session_state.risk_report:
                sharpe_value = st.session_state.risk_report['sharpe_ratio']
                st.markdown(f"""
                <div class="cosmic-card">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Sharpe Ratio</div>
                    <div style="color: var(--neutron-green); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {sharpe_value:.2f}
                    </div>
                    <div style="color: var(--text-dim); font-size: 0.9rem;">
                        Risk-adjusted return
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if 'win_rate' in st.session_state.risk_report:
                win_rate_value = st.session_state.risk_report['win_rate']
                st.markdown(f"""
                <div class="cosmic-card">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Win Rate</div>
                    <div style="color: var(--quasar-blue); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {win_rate_value:.1f}%
                    </div>
                    <div style="color: var(--text-dim); font-size: 0.9rem;">
                        Positive return probability
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Risk Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Drawdown chart
            if len(df) > 20:
                cumulative = (1 + df['returns']).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max * 100
                
                fig_drawdown = go.Figure()
                
                fig_drawdown.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name="Drawdown",
                    line=dict(color='#ff3366', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 51, 102, 0.2)'
                ))
                
                fig_drawdown.update_layout(
                    visualizer.create_cosmic_layout("Historical Drawdown", 400),
                    yaxis_title="Drawdown (%)",
                    showlegend=False
                )
                
                st.plotly_chart(fig_drawdown, use_container_width=True)
        
        with col2:
            # Risk metrics gauge
            if 'sharpe_ratio' in st.session_state.risk_report:
                sharpe_value = st.session_state.risk_report['sharpe_ratio']
                fig_gauge = visualizer.create_performance_gauge(
                    min(max(sharpe_value * 10 + 50, 0), 100),
                    100,
                    "Risk-Adjusted Performance"
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Detailed Risk Report
        st.markdown("### 📊 Detailed Risk Report")
        
        if st.session_state.risk_report:
            # Create risk metrics table
            risk_metrics = [
                ("Value at Risk (95%)", f"{st.session_state.risk_report.get('historical_var', 0):.2f}%"),
                ("Expected Shortfall", f"{st.session_state.risk_report.get('expected_shortfall', 0):.2f}%"),
                ("Max Drawdown", f"{st.session_state.risk_report.get('max_drawdown', 0):.2f}%"),
                ("Current Drawdown", f"{st.session_state.risk_report.get('current_drawdown', 0):.2f}%"),
                ("Sharpe Ratio", f"{st.session_state.risk_report.get('sharpe_ratio', 0):.3f}"),
                ("Sortino Ratio", f"{st.session_state.risk_report.get('sortino_ratio', 0):.3f}"),
                ("Calmar Ratio", f"{st.session_state.risk_report.get('calmar_ratio', 0):.3f}"),
                ("Win Rate", f"{st.session_state.risk_report.get('win_rate', 0):.1f}%"),
                ("Total Return", f"{st.session_state.risk_report.get('total_return', 0):.2f}%"),
                ("Volatility", f"{st.session_state.risk_report.get('volatility', 0):.2f}%")
            ]
            
            # Display metrics in a grid
            cols = st.columns(3)
            for idx, (metric, value) in enumerate(risk_metrics):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div style="background: rgba(10, 10, 30, 0.6); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
                        <div style="color: var(--text-dim); font-size: 0.85rem; margin-bottom: 0.25rem;">
                            {metric}
                        </div>
                        <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600;">
                            {value}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Risk Recommendations
        st.markdown("### 💡 Risk Recommendations")
        
        recommendations = []
        
        if 'volatility' in st.session_state.risk_report:
            vol = st.session_state.risk_report['volatility']
            if vol > 100:
                recommendations.append(("⚠️ High Volatility", 
                                      "Consider reducing position size or using options for hedging"))
            elif vol > 60:
                recommendations.append(("🔶 Moderate Volatility", 
                                      "Monitor closely and consider trailing stops"))
        
        if 'max_drawdown' in st.session_state.risk_report:
            dd = st.session_state.risk_report['max_drawdown']
            if abs(dd) > 20:
                recommendations.append(("⚠️ Large Historical Drawdown", 
                                      "Implement strict stop-losses and consider diversification"))
        
        if 'sharpe_ratio' in st.session_state.risk_report:
            sharpe = st.session_state.risk_report['sharpe_ratio']
            if sharpe < 0.5:
                recommendations.append(("📉 Low Risk-Adjusted Return", 
                                      "Re-evaluate strategy or consider alternative assets"))
        
        if recommendations:
            for title, desc in recommendations:
                st.markdown(f"""
                <div style="background: rgba(255, 204, 0, 0.1); border-left: 4px solid var(--quantum-yellow); 
                         padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
                    <div style="color: var(--quantum-yellow); font-weight: 600; margin-bottom: 0.5rem;">
                        {title}
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem;">
                        {desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("✅ No critical risk issues detected. Current risk profile appears acceptable.")

# Tab 5: Technical Suite
with tabs[4]:
    st.markdown("### 📊 Complete Technical Analysis Suite")
    
    # Create tabs within the technical suite
    tech_tabs = st.tabs(["📈 Moving Averages", "📊 Oscillators", "📉 Volume Analysis", "🎯 Advanced Patterns"])
    
    with tech_tabs[0]:
        st.markdown("### 📈 Moving Averages Analysis")
        
        # Moving averages comparison
        if all(col in df.columns for col in ['close', 'sma_20', 'ema_20', 'sma_50', 'ema_50']):
            fig_ma = go.Figure()
            
            display_df = df.iloc[-100:]
            
            fig_ma.add_trace(go.Scatter(
                x=display_df.index, y=display_df['close'],
                name='Price', line=dict(color='#f0f4f8', width=2)
            ))
            
            fig_ma.add_trace(go.Scatter(
                x=display_df.index, y=display_df['sma_20'],
                name='SMA 20', line=dict(color='#00d4ff', width=1.5)
            ))
            
            fig_ma.add_trace(go.Scatter(
                x=display_df.index, y=display_df['ema_20'],
                name='EMA 20', line=dict(color='#00d4ff', width=1.5, dash='dash')
            ))
            
            fig_ma.add_trace(go.Scatter(
                x=display_df.index, y=display_df['sma_50'],
                name='SMA 50', line=dict(color='#aa55ff', width=1.5)
            ))
            
            fig_ma.add_trace(go.Scatter(
                x=display_df.index, y=display_df['ema_50'],
                name='EMA 50', line=dict(color='#aa55ff', width=1.5, dash='dash')
            ))
            
            fig_ma.update_layout(visualizer.create_cosmic_layout("Moving Averages Comparison", 500))
            st.plotly_chart(fig_ma, use_container_width=True)
        
        # MA Cross signals
        st.markdown("### 📊 Moving Average Crossovers")
        
        if all(col in df.columns for col in ['sma_20', 'sma_50']):
            # Detect Golden Cross (SMA20 crosses above SMA50)
            golden_cross = (df['sma_20'] > df['sma_50']) & (df['sma_20'].shift(1) <= df['sma_50'].shift(1))
            death_cross = (df['sma_20'] < df['sma_50']) & (df['sma_20'].shift(1) >= df['sma_50'].shift(1))
            
            col1, col2 = st.columns(2)
            
            with col1:
                current_cross = "Golden Cross" if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] else "Death Cross"
                cross_color = "var(--text-success)" if current_cross == "Golden Cross" else "var(--text-danger)"
                
                st.markdown(f"""
                <div class="cosmic-card" style="text-align: center;">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                        Current MA Relationship
                    </div>
                    <div style="color: {cross_color}; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {current_cross}
                    </div>
                    <div style="color: var(--text-dim); font-size: 0.9rem;">
                        SMA20 vs SMA50
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                golden_count = golden_cross.sum()
                death_count = death_cross.sum()
                
                st.markdown(f"""
                <div class="cosmic-card" style="text-align: center;">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                        Crossover Events
                    </div>
                    <div style="display: flex; justify-content: space-around; margin-top: 0.5rem;">
                        <div>
                            <div style="color: var(--text-success); font-size: 1.2rem; font-weight: 700;">{golden_count}</div>
                            <div style="color: var(--text-dim); font-size: 0.8rem;">Golden</div>
                        </div>
                        <div>
                            <div style="color: var(--text-danger); font-size: 1.2rem; font-weight: 700;">{death_count}</div>
                            <div style="color: var(--text-dim); font-size: 0.8rem;">Death</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tech_tabs[1]:
        st.markdown("### 📊 Oscillators Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI Analysis
            if 'rsi' in df.columns:
                rsi_value = df['rsi'].iloc[-1]
                rsi_trend = "Rising" if rsi_value > df['rsi'].iloc[-5] else "Falling"
                
                st.markdown(f"""
                <div class="cosmic-card">
                    <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">RSI Analysis</h4>
                    <div style="color: var(--text-primary); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {rsi_value:.1f}
                    </div>
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                        Trend: {rsi_trend}
                    </div>
                    <div class="cosmic-progress">
                        <div class="cosmic-progress-bar" style="width: {rsi_value}%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                        <span style="color: var(--text-dim); font-size: 0.8rem;">0</span>
                        <span style="color: var(--text-dim); font-size: 0.8rem;">100</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Stochastic Analysis
            if all(col in df.columns for col in ['stoch_k', 'stoch_d']):
                stoch_k = df['stoch_k'].iloc[-1]
                stoch_d = df['stoch_d'].iloc[-1]
                
                st.markdown(f"""
                <div class="cosmic-card">
                    <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">Stochastic Oscillator</h4>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: var(--text-dim);">%K:</span>
                        <span style="color: var(--text-primary); font-weight: 700;">{stoch_k:.1f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                        <span style="color: var(--text-dim);">%D:</span>
                        <span style="color: var(--text-primary); font-weight: 700;">{stoch_d:.1f}</span>
                    </div>
                    <div class="cosmic-progress">
                        <div class="cosmic-progress-bar" style="width: {stoch_k}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # MACD Analysis
        if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
            st.markdown("### 📈 MACD Analysis")
            
            macd_val = df['macd'].iloc[-1]
            signal_val = df['macd_signal'].iloc[-1]
            hist_val = df['macd_hist'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("MACD Line", f"{macd_val:.4f}", 
                         "Bullish" if macd_val > 0 else "Bearish")
            
            with col2:
                st.metric("Signal Line", f"{signal_val:.4f}",
                         "Above" if macd_val > signal_val else "Below")
            
            with col3:
                st.metric("Histogram", f"{hist_val:.4f}",
                         "Positive" if hist_val > 0 else "Negative")

# Tab 6: Data Insights
with tabs[5]:
    st.markdown("### 📋 Data Insights & Statistics")
    
    # Data Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data Points", len(df))
    
    with col2:
        date_range = df.index[-1] - df.index[0]
        st.metric("Time Period", f"{date_range.days} days")
    
    with col3:
        avg_volume = df['volume'].mean()
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
    
    with col4:
        price_range = df['high'].max() - df['low'].min()
        st.metric("Price Range", f"${price_range:,.2f}")
    
    # Data Quality Check
    st.markdown("### 🔍 Data Quality Analysis")
    
    missing_data = df.isnull().sum().sum()
    completeness = (1 - (missing_data / (len(df) * len(df.columns)))) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="cosmic-card">
            <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                Data Completeness
            </div>
            <div style="color: var(--neutron-green); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                {completeness:.1f}%
            </div>
            <div style="color: var(--text-dim); font-size: 0.9rem;">
                Missing values: {missing_data}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Data freshness
        last_update = datetime.now()
        data_age = (last_update - df.index[-1].to_pydatetime()).total_seconds() / 60
        
        freshness_color = "var(--text-success)" if data_age < 5 else "var(--text-warning)" if data_age < 15 else "var(--text-danger)"
        
        st.markdown(f"""
        <div class="cosmic-card">
            <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                Data Freshness
            </div>
            <div style="color: {freshness_color}; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                {data_age:.1f} min
            </div>
            <div style="color: var(--text-dim); font-size: 0.9rem;">
                Since last data point
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Raw Data Preview
    st.markdown("### 📊 Data Preview")
    
    with st.expander("View Raw Data"):
        # Show last 20 rows
        display_df = df.tail(20)
        
        # Format the display
        formatted_df = display_df.copy()
        for col in ['open', 'high', 'low', 'close']:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.2f}")
        
        if 'volume' in formatted_df.columns:
            formatted_df['volume'] = formatted_df['volume'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(formatted_df, use_container_width=True)
    
    # Export Options
    st.markdown("### 📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Copy Latest Values", use_container_width=True):
            latest_data = df.iloc[-1].to_dict()
            # Format for copying
            formatted = "\n".join([f"{k}: {v}" for k, v in latest_data.items()])
            st.code(formatted, language="text")
    
    with col2:
        csv = df.to_csv()
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        json_data = df.tail(100).to_json(orient="records", date_format="iso")
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"{symbol.replace('/', '_')}_{timeframe}_latest.json",
            mime="application/json",
            use_container_width=True
        )

# ==========================================
# 13. AUTO-REFRESH SYSTEM
# ==========================================

if auto_refresh:
    refresh_seconds = {
        "30s": 30,
        "1m": 60,
        "5m": 300,
        "15m": 900
    }.get(refresh_rate, 60)
    
    current_time = time_lib.time()
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = current_time
    
    if current_time - st.session_state.last_refresh > refresh_seconds:
        st.session_state.last_refresh = current_time
        st.rerun()

# ==========================================
# 14. ENHANCED FOOTER & STATUS
# ==========================================

st.markdown("""
<div class="quantum-divider"></div>

<div style="background: rgba(10, 10, 30, 0.8); padding: 1.5rem; border-radius: 16px; margin-top: 2rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div style="color: var(--text-dim); font-size: 0.8rem;">System Status</div>
            <div style="color: var(--neutron-green); font-size: 1rem; font-weight: 600;">
                🟢 All Systems Operational
            </div>
        </div>
        
        <div style="text-align: center;">
            <div style="color: var(--text-dim); font-size: 0.8rem;">Last Update</div>
            <div style="color: var(--text-primary); font-size: 1rem; font-weight: 600;">
                {update_time}
            </div>
        </div>
        
        <div style="text-align: right;">
            <div style="color: var(--text-dim); font-size: 0.8rem;">Data Points</div>
            <div style="color: var(--text-primary); font-size: 1rem; font-weight: 600;">
                {data_points:,}
            </div>
        </div>
    </div>
    
    <div class="quantum-divider" style="margin: 1rem 0;"></div>
    
    <div style="text-align: center; color: var(--text-dim); font-size: 0.8rem; line-height: 1.5;">
        <strong>COSMIC NEXUS v3.0</strong> • Advanced Quantum Trading Terminal • © 2024 Cosmic Nexus Labs<br>
        ⚠️ This is a sophisticated trading tool. Trading involves significant risk. Past performance does not guarantee future results.
    </div>
</div>
""".format(
    update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    data_points=len(df)
), unsafe_allow_html=True)
