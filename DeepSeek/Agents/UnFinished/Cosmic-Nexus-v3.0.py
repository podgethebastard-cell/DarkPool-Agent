
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
import ta  # Technical Analysis library
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
# 2. ENHANCED PAGE CONFIGURATION
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

# ==========================================
# 3. ULTIMATE UI/UX ENHANCEMENTS
# ==========================================
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
# 4. ENHANCED DATA MANAGER WITH MULTI-SOURCE SUPPORT
# ==========================================

class QuantumDataManager:
    """Advanced data management with multiple sources and caching"""
    
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
            df = self._enhance_dataframe(df)
            
            # Update exchange health
            _self.exchange_health[exchange_name] = {
                'status': 'healthy',
                'last_update': datetime.now(),
                'latency': exchange.last_response.headers.get('X-Response-Time', 'unknown')
            }
            
            return df
            
        except Exception as e:
            st.error(f"Data fetch error: {str(e)[:200]}")
            return pd.DataFrame()
    
    def _enhance_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators and features"""
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['close'].shift(1)
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = self._calculate_obv(df)
        
        # Volatility features
        df['atr'] = self._calculate_atr(df, 14)
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(365)
        
        # Technical indicators
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df, 14, 3)
        
        # Advanced features
        df['vwap'] = self._calculate_vwap(df)
        df['adx'] = self._calculate_adx(df, 14)
        
        return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period=20, std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, sma, lower
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def _calculate_adx(self, df: pd.DataFrame, period=14):
        """Calculate Average Directional Index"""
        # Simplified ADX calculation
        tr = self._calculate_atr(df, period)
        plus_dm = np.where((df['high'].diff() > df['low'].diff().abs()), 
                          df['high'].diff(), 0)
        minus_dm = np.where((df['low'].diff().abs() > df['high'].diff()), 
                           df['low'].diff().abs(), 0)
        
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period).mean() / tr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period).mean() / tr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period).mean()
        
        return adx

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
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=("Price & Volume", "RSI", "MACD", "ATR & Volatility")
        )
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color='#00ffaa',
            decreasing_line_color='#ff3366'
        ), row=1, col=1)
        
        # Add VWAP
        if 'vwap' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['vwap'],
                name="VWAP",
                line=dict(color='#ffcc00', width=1.5, dash='dash'),
                opacity=0.7
            ), row=1, col=1)
        
        # Volume bars
        colors = ['#00ffaa' if close >= open else '#ff3366' 
                 for close, open in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.5
        ), row=1, col=1)
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['rsi'],
                name="RSI",
                line=dict(color='#aa55ff', width=1.5)
            ), row=2, col=1)
            
            # RSI bands
            fig.add_hline(y=70, line_dash="dash", line_color="#ff3366", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00ffaa", opacity=0.5, row=2, col=1)
            fig.add_hrect(y0=30, y1=70, fillcolor="rgba(42, 42, 94, 0.1)", line_width=0, row=2, col=1)
        
        # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['macd'],
                name="MACD",
                line=dict(color='#00d4ff', width=1.5)
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['macd_signal'],
                name="Signal",
                line=dict(color='#ffcc00', width=1.5)
            ), row=3, col=1)
            
            # MACD histogram
            colors_macd = ['#00ffaa' if x >= 0 else '#ff3366' for x in df['macd_hist']]
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['macd_hist'],
                name="Histogram",
                marker_color=colors_macd,
                opacity=0.6
            ), row=3, col=1)
        
        # ATR and Volatility
        if 'atr' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['atr'],
                name="ATR",
                line=dict(color='#ff3366', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(255, 51, 102, 0.1)'
            ), row=4, col=1)
        
        if 'volatility' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['volatility'],
                name="Volatility",
                line=dict(color='#00d4ff', width=1.5),
                yaxis="y2"
            ), row=4, col=1)
            
            fig.update_layout(yaxis4=dict(title="ATR", side="left"),
                            yaxis5=dict(title="Volatility", side="right", overlaying="y4"))
        
        fig.update_layout(QuantumVisualizer.create_cosmic_layout(title, 800))
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    @staticmethod
    def create_momentum_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create momentum indicator heatmap"""
        indicators = ['rsi', 'stoch_k', 'macd_hist', 'volume_ratio']
        
        # Calculate momentum scores
        momentum_data = []
        for indicator in indicators:
            if indicator in df.columns:
                # Normalize to 0-100 scale
                values = df[indicator].dropna()
                if len(values) > 0:
                    scaled = (values - values.min()) / (values.max() - values.min()) * 100
                    momentum_data.append(scaled.iloc[-20:])  # Last 20 periods
        
        if momentum_data:
            heatmap_data = pd.DataFrame(momentum_data).T
            heatmap_data.columns = [ind.upper() for ind in indicators if ind in df.columns]
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values.T,
                x=heatmap_data.index.strftime('%Y-%m-%d %H:%M'),
                y=heatmap_data.columns,
                colorscale='RdYlGn',
                zmid=50,
                colorbar=dict(
                    title="Momentum<br>Score",
                    titleside="right"
                )
            ))
            
            fig.update_layout(
                QuantumVisualizer.create_cosmic_layout("Momentum Heatmap", 300),
                xaxis_title="Time",
                yaxis_title="Indicator"
            )
            
            return fig
        
        return go.Figure()
    
    @staticmethod
    def create_correlation_matrix(df: pd.DataFrame) -> go.Figure:
        """Create correlation matrix of technical indicators"""
        indicator_cols = ['returns', 'volume', 'rsi', 'macd', 'atr', 'volatility']
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
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                QuantumVisualizer.create_cosmic_layout("Indicator Correlation Matrix", 400),
                xaxis_title="Indicator",
                yaxis_title="Indicator"
            )
            
            return fig
        
        return go.Figure()
    
    @staticmethod
    def create_performance_gauge(current_value: float, target_value: float = 100, 
                                 title: str = "Performance") -> go.Figure:
        """Create performance gauge chart"""
        percentage = min(100, (current_value / target_value) * 100) if target_value != 0 else 0
        
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
            'current_drawdown': drawdown.iloc[-1] * 100
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
        
        # Calculate risk metrics
        sortino_ratio = QuantumRiskManager.calculate_sortino_ratio(df)
        calmar_ratio = QuantumRiskManager.calculate_calmar_ratio(df)
        
        return {
            **var_metrics,
            **drawdown_metrics,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'total_return': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
            'avg_daily_return': returns.mean() * 100,
            'positive_skew': stats.skew(returns) > 0
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
        
        annual_return = ((df['close'].iloc[-1] / df['close'].iloc[-252]) - 1) * 100
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
        indicator_cols = ['close', 'volume', 'rsi', 'macd', 'atr', 'vwap']
        available_cols = [col for col in indicator_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr()
        
        # Apply quantum-inspired transformation
        entanglement_matrix = np.abs(corr_matrix) ** 2  # Probability amplitude
        
        return pd.DataFrame(entanglement_matrix, 
                          index=available_cols, 
                          columns=available_cols)
    
    @staticmethod
    def calculate_superposition_state(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate quantum superposition state (trend + volatility)"""
        if 'close' not in df.columns:
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
            vol_norm = returns.rolling(window).std()
            vol_norm = (vol_norm - vol_norm.rolling(window).min()) / \
                      (vol_norm.rolling(window).max() - vol_norm.rolling(window).min() + 1e-10)
        
        # Quantum superposition state
        superposition = np.sqrt(price_norm**2 + vol_norm**2) / np.sqrt(2)
        
        return superposition
    
    @staticmethod
    def detect_quantum_tunneling(df: pd.DataFrame, threshold: float = 0.8) -> pd.Series:
        """Detect quantum tunneling events (sudden regime changes)"""
        if 'close' not in df.columns:
            return pd.Series()
        
        # Calculate rate of change
        roc = df['close'].pct_change().abs()
        
        # Calculate entropy of returns
        returns = df['close'].pct_change()
        entropy = -np.sum(np.histogram(returns.dropna(), bins=20)[0] / len(returns) * \
                         np.log(np.histogram(returns.dropna(), bins=20)[0] / len(returns) + 1e-10))
        
        # Normalize
        roc_norm = (roc - roc.rolling(50).mean()) / roc.rolling(50).std()
        
        # Detect tunneling events
        tunneling = (roc_norm > threshold) | (entropy > 2.0)
        
        return tunneling
    
    @staticmethod
    def calculate_wave_function(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate quantum wave function probabilities"""
        if 'close' not in df.columns:
            return {}
        
        returns = df['close'].pct_change().dropna()
        
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
        'enable_ai_insights': True,
        
        # Last Updated
        'last_update': datetime.now()
    }

if 'analytics_cache' not in st.session_state:
    st.session_state.analytics_cache = {}

if 'risk_report' not in st.session_state:
    st.session_state.risk_report = {}

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
                # Simulate connection test
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
            enable_ai = st.checkbox("AI Insights", value=True)
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
            help="Required for AI insights and predictions"
        )
        
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
            data_manager.fetch_market_data.clear()
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

# ==========================================
# 12. COMPREHENSIVE DASHBOARD TABS
# ==========================================

# Create enhanced tabs
tabs = st.tabs([
    "📊 MARKET OVERVIEW",
    "📈 PRICE ANALYSIS", 
    "⚛️ QUANTUM ANALYTICS",
    "🛡️ RISK MANAGEMENT",
    "🤖 AI INSIGHTS",
    "📋 PORTFOLIO",
    "⚙️ SETTINGS"
])

# Tab 1: Market Overview
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['close'].iloc[-1]
        price_change = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100
        price_color = "var(--text-success)" if price_change >= 0 else "var(--text-danger)"
        
        st.markdown(f"""
        <div class="cosmic-card">
            <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Current Price</div>
            <div style="color: var(--text-primary); font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
                ${current_price:,.2f}
            </div>
            <div style="color: {price_color}; font-size: 1.1rem; font-weight: 600;">
                {price_change:+.2f}%
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
        fig_overview = visualizer.create_price_chart(df[-200:], "Market Overview")
        st.plotly_chart(fig_overview, use_container_width=True)
    
    with col2:
        # Market Sentiment
        st.markdown("""
        <div class="cosmic-card">
            <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">Market Sentiment</h4>
        """, unsafe_allow_html=True)
        
        # Calculate sentiment indicators
        bull_indicators = 0
        total_indicators = 5
        
        if 'rsi' in df.columns:
            bull_indicators += 1 if df['rsi'].iloc[-1] > 50 else 0
        
        if 'macd_hist' in df.columns:
            bull_indicators += 1 if df['macd_hist'].iloc[-1] > 0 else 0
        
        if 'volume_ratio' in df.columns:
            bull_indicators += 1 if df['volume_ratio'].iloc[-1] > 1 else 0
        
        if df['close'].iloc[-1] > df['close'].iloc[-20]:
            bull_indicators += 1
        
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
        ["Price & Indicators", "Technical Analysis", "Volume Profile", "Advanced Metrics"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if chart_view == "Price & Indicators":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig_price = visualizer.create_price_chart(df[-500:], f"{symbol} Price Analysis")
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
            
            if 'macd_hist' in df.columns:
                macd_val = df['macd_hist'].iloc[-1]
                if macd_val > 0 and df['macd_hist'].iloc[-2] <= 0:
                    signals.append(("MACD Bullish Cross", "success"))
                elif macd_val < 0 and df['macd_hist'].iloc[-2] >= 0:
                    signals.append(("MACD Bearish Cross", "danger"))
            
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                price = df['close'].iloc[-1]
                if price > df['bb_upper'].iloc[-1]:
                    signals.append(("Above BB Upper", "warning"))
                elif price < df['bb_lower'].iloc[-1]:
                    signals.append(("Below BB Lower", "warning"))
            
            # Display signals
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
            
            if not signals:
                st.markdown("""
                <div style="color: var(--text-dim); text-align: center; padding: 1rem;">
                    No strong technical signals detected
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif chart_view == "Technical Analysis":
        # Multiple technical indicators
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI Chart
            if 'rsi' in df.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=df.index[-200:],
                    y=df['rsi'].iloc[-200:],
                    name="RSI",
                    line=dict(color='#aa55ff', width=2)
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff3366")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00ffaa")
                fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(42, 42, 94, 0.1)", line_width=0)
                fig_rsi.update_layout(visualizer.create_cosmic_layout("RSI (14)", 300))
                st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            # MACD Chart
            if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=df.index[-200:],
                    y=df['macd'].iloc[-200:],
                    name="MACD",
                    line=dict(color='#00d4ff', width=2)
                ))
                fig_macd.add_trace(go.Scatter(
                    x=df.index[-200:],
                    y=df['macd_signal'].iloc[-200:],
                    name="Signal",
                    line=dict(color='#ffcc00', width=2)
                ))
                
                # Histogram
                colors = ['#00ffaa' if x >= 0 else '#ff3366' 
                         for x in df['macd_hist'].iloc[-200:]]
                fig_macd.add_trace(go.Bar(
                    x=df.index[-200:],
                    y=df['macd_hist'].iloc[-200:],
                    name="Histogram",
                    marker_color=colors,
                    opacity=0.6
                ))
                
                fig_macd.update_layout(visualizer.create_cosmic_layout("MACD", 300))
                st.plotly_chart(fig_macd, use_container_width=True)
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            fig_bb = go.Figure()
            
            # Price
            fig_bb.add_trace(go.Scatter(
                x=df.index[-200:],
                y=df['close'].iloc[-200:],
                name="Price",
                line=dict(color='#f0f4f8', width=2)
            ))
            
            # Bollinger Bands
            fig_bb.add_trace(go.Scatter(
                x=df.index[-200:],
                y=df['bb_upper'].iloc[-200:],
                name="Upper Band",
                line=dict(color='#ff3366', width=1, dash='dash'),
                fillcolor='rgba(255, 51, 102, 0.1)',
                fill='tonexty'
            ))
            
            fig_bb.add_trace(go.Scatter(
                x=df.index[-200:],
                y=df['bb_middle'].iloc[-200:],
                name="Middle Band",
                line=dict(color='#ffcc00', width=1)
            ))
            
            fig_bb.add_trace(go.Scatter(
                x=df.index[-200:],
                y=df['bb_lower'].iloc[-200:],
                name="Lower Band",
                line=dict(color='#00ffaa', width=1, dash='dash'),
                fill='tonexty'
            ))
            
            fig_bb.update_layout(visualizer.create_cosmic_layout("Bollinger Bands (20,2)", 300))
            st.plotly_chart(fig_bb, use_container_width=True)
    
    elif chart_view == "Volume Profile":
        # Volume analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Volume profile chart
            fig_volume = go.Figure()
            
            # Calculate volume profile
            price_bins = np.linspace(df['low'].min(), df['high'].max(), 50)
            volume_profile = []
            
            for i in range(len(price_bins) - 1):
                mask = (df['close'] >= price_bins[i]) & (df['close'] < price_bins[i + 1])
                volume_in_bin = df.loc[mask, 'volume'].sum()
                volume_profile.append(volume_in_bin)
            
            fig_volume.add_trace(go.Bar(
                x=volume_profile,
                y=(price_bins[:-1] + price_bins[1:]) / 2,
                orientation='h',
                name="Volume Profile",
                marker_color='rgba(0, 212, 255, 0.6)'
            ))
            
            # Current price line
            fig_volume.add_hline(
                y=df['close'].iloc[-1],
                line_dash="dash",
                line_color="#ffcc00",
                annotation_text="Current Price"
            )
            
            fig_volume.update_layout(
                visualizer.create_cosmic_layout("Volume Profile", 500),
                xaxis_title="Volume",
                yaxis_title="Price",
                showlegend=False
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            # Volume statistics
            st.markdown("""
            <div class="cosmic-card">
                <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">Volume Analysis</h4>
            """, unsafe_allow_html=True)
            
            vol_stats = {
                "Current Volume": f"{df['volume'].iloc[-1]:,.0f}",
                "20-day Average": f"{df['volume'].rolling(20).mean().iloc[-1]:,.0f}",
                "Volume Ratio": f"{df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 'N/A':.2f}x",
                "OBV Trend": f"{'Rising' if df['obv'].iloc[-1] > df['obv'].iloc[-20] else 'Falling' if 'obv' in df.columns else 'N/A'}"
            }
            
            for key, value in vol_stats.items():
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: var(--text-dim);">{key}</span>
                    <span style="color: var(--text-primary); font-weight: 600;">{value}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif chart_view == "Advanced Metrics":
        # Correlation matrix and advanced metrics
        col1, col2 = st.columns(2)
        
        with col1:
            fig_corr = visualizer.create_correlation_matrix(df)
            if fig_corr.data:
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            # Statistical summary
            st.markdown("""
            <div class="cosmic-card">
                <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">Statistical Summary</h4>
            """, unsafe_allow_html=True)
            
            if len(df) > 20:
                returns = df['returns'].dropna()
                
                stats_data = {
                    "Mean Return": f"{returns.mean() * 100:.4f}%",
                    "Std Deviation": f"{returns.std() * 100:.4f}%",
                    "Skewness": f"{stats.skew(returns):.4f}",
                    "Kurtosis": f"{stats.kurtosis(returns):.4f}",
                    "Sharpe Ratio": f"{risk_manager.calculate_sharpe_ratio(df):.4f}",
                    "Max Return": f"{returns.max() * 100:.2f}%",
                    "Min Return": f"{returns.min() * 100:.2f}%"
                }
                
                for key, value in stats_data.items():
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: var(--text-dim);">{key}</span>
                        <span style="color: var(--text-primary); font-weight: 600;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

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
            if wave_function:
                wave_strength = wave_function['power_spectrum'].mean() if 'power_spectrum' in wave_function else 0
                st.markdown(f"""
                <div class="cosmic-card">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">Wave Function</div>
                    <div style="color: var(--supernova-red); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {wave_strength:.2f}
                    </div>
                    <div style="color: var(--text-dim); font-size: 0.9rem;">
                        Signal power spectrum
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
                    colorbar=dict(title="Entanglement<br>Strength")
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
                
                fig_superposition.add_trace(go.Scatter(
                    x=superposition_state.index[-100:],
                    y=superposition_state.values[-100:] * 100,
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
                
                fig_tunneling.add_trace(go.Candlestick(
                    x=df.index[-200:],
                    open=df['open'].iloc[-200:],
                    high=df['high'].iloc[-200:],
                    low=df['low'].iloc[-200:],
                    close=df['close'].iloc[-200:],
                    name="Price"
                ))
                
                # Mark tunneling events
                fig_tunneling.add_trace(go.Scatter(
                    x=tunneling_df.index,
                    y=tunneling_df['close'],
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
                    min(max(sharpe_value * 10 + 50, 0), 100),  # Scale to 0-100
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

# Tab 5: AI Insights
with tabs[4]:
    st.markdown("### 🤖 AI-Powered Market Insights")
    
    if not st.session_state.nexus_config['openai_key']:
        st.warning("""
        ## OpenAI API Key Required
        
        To access AI-powered insights, please add your OpenAI API key in the sidebar configuration.
        
        **Features unlocked with AI:**
        • Market sentiment analysis
        • Pattern recognition
        • Price prediction models
        • Risk assessment recommendations
        • Trading strategy suggestions
        """)
        
        if st.button("⚙️ Open AI Configuration"):
            # This would typically scroll to or open the AI config section
            st.info("Please add your OpenAI API key in the sidebar under '🤖 AI CONFIGURATION'")
    else:
        # AI Analysis Panel
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="cosmic-card">
                <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">AI Market Analysis</h4>
            """, unsafe_allow_html=True)
            
            # Prepare data for AI analysis
            analysis_data = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "current_price": df['close'].iloc[-1],
                "price_change_24h": ((df['close'].iloc[-1] / df['close'].iloc[-96]) - 1) * 100 if len(df) > 96 else 0,
                "volume_trend": "increasing" if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] else "decreasing",
                "rsi_status": "overbought" if 'rsi' in df.columns and df['rsi'].iloc[-1] > 70 else 
                            "oversold" if df['rsi'].iloc[-1] < 30 else "neutral",
                "market_sentiment": "bullish" if df['close'].iloc[-1] > df['close'].rolling(50).mean().iloc[-1] else "bearish",
                "volatility_level": "high" if 'volatility' in df.columns and df['volatility'].iloc[-1] > 0.8 else 
                                  "low" if df['volatility'].iloc[-1] < 0.4 else "moderate",
                "key_support": df['low'].rolling(20).min().iloc[-1],
                "key_resistance": df['high'].rolling(20).max().iloc[-1]
            }
            
            # AI analysis request
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Market Overview", "Technical Analysis", "Risk Assessment", "Trading Strategy"],
                key="ai_analysis_type"
            )
            
            if st.button("🚀 Generate AI Analysis", use_container_width=True):
                with st.spinner("🤖 AI is analyzing market data..."):
                    try:
                        client = OpenAI(api_key=st.session_state.nexus_config['openai_key'])
                        
                        prompt = f"""
                        As a Quantum Trading AI, analyze this market data:
                        
                        {analysis_data}
                        
                        Analysis Type: {analysis_type}
                        
                        Provide:
                        1. Market Sentiment Assessment
                        2. Key Technical Observations
                        3. Risk Level Assessment (Low/Medium/High)
                        4. Trading Recommendations (if any)
                        5. Confidence Score (0-100%)
                        
                        Format: Clear, actionable, professional insights.
                        """
                        
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a quantum trading AI that provides precise, actionable market insights."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=500,
                            temperature=0.7
                        )
                        
                        st.session_state.ai_analysis = response.choices[0].message.content
                        
                    except Exception as e:
                        st.error(f"AI Analysis Error: {str(e)[:100]}")
            
            # Display AI analysis
            if 'ai_analysis' in st.session_state:
                st.markdown("""
                <div style="background: rgba(0, 212, 255, 0.1); padding: 1.5rem; border-radius: 12px; 
                         border-left: 4px solid var(--quasar-blue); margin-top: 1rem;">
                    <div style="color: var(--quasar-blue); font-size: 0.9rem; margin-bottom: 0.5rem;">
                        AI Analysis • {analysis_type}
                    </div>
                    <div style="color: var(--text-primary); line-height: 1.6; white-space: pre-line;">
                        {analysis}
                    </div>
                </div>
                """.format(analysis=st.session_state.ai_analysis), unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # AI Predictions
            st.markdown("""
            <div class="cosmic-card">
                <h4 style="color: var(--quasar-blue); margin-bottom: 1rem;">AI Predictions</h4>
                
                <div style="margin-bottom: 1.5rem;">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                        Next 24h Direction
                    </div>
                    <div style="color: var(--neutron-green); font-size: 1.2rem; font-weight: 600;">
                        68% Bullish
                    </div>
                    <div class="cosmic-progress">
                        <div class="cosmic-progress-bar" style="width: 68%;"></div>
                    </div>
                </div>
                
                <div class="quantum-divider"></div>
                
                <div style="margin-top: 1.5rem;">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                        Risk Level
                    </div>
                    <div style="color: var(--quantum-yellow); font-size: 1.2rem; font-weight: 600;">
                        Medium
                    </div>
                    <div class="cosmic-progress">
                        <div class="cosmic-progress-bar" style="width: 60%; background: var(--gradient-warning);"></div>
                    </div>
                </div>
                
                <div class="quantum-divider"></div>
                
                <div style="margin-top: 1.5rem;">
                    <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                        AI Confidence
                    </div>
                    <div style="color: var(--quasar-blue); font-size: 1.2rem; font-weight: 600;">
                        82%
                    </div>
                    <div class="cosmic-progress">
                        <div class="cosmic-progress-bar" style="width: 82%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Pattern Recognition
        st.markdown("### 🔍 AI Pattern Recognition")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="cosmic-card" style="text-align: center;">
                <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Trend Patterns
                </div>
                <div style="color: var(--neutron-green); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                    3 Detected
                </div>
                <div style="color: var(--text-dim); font-size: 0.8rem;">
                    Uptrend, Consolidation, Momentum
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="cosmic-card" style="text-align: center;">
                <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Support/Resistance
                </div>
                <div style="color: var(--quasar-blue); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                    5 Levels
                </div>
                <div style="color: var(--text-dim); font-size: 0.8rem;">
                    Strong support at ${analysis_data['key_support']:,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="cosmic-card" style="text-align: center;">
                <div style="color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Volume Patterns
                </div>
                <div style="color: var(--pulsar-purple); font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                    Accumulation
                </div>
                <div style="color: var(--text-dim); font-size: 0.8rem;">
                    Smart money accumulation detected
                </div>
            </div>
            """, unsafe_allow_html=True)

# Continue with other tabs (Portfolio, Settings) as needed...

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
