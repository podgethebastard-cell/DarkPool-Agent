import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================
class Config:
    APP_NAME = "TITAN INTRADAY PRO"
    CACHE_TTL = 30
    MAX_CANDLES = 2000
    DEFAULT_SYMBOL = "BTCUSDT"
    DEFAULT_TIMEFRAME = "5m"
    INITIAL_CAPITAL = 10000
    
    # Data sources (will try in order)
    DATA_SOURCES = [
        "binance",
        "kucoin",
        "coingecko"
    ]

# =========================
# DATA FETCHER WITH MULTIPLE SOURCES
# =========================
class DataFetcher:
    """Fetch market data from multiple sources with fallbacks"""
    
    @staticmethod
    def fetch_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Fetch data from Binance"""
        try:
            # Clean symbol
            symbol = symbol.upper()
            
            # Binance interval mapping
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "4h": "4h", "1d": "1d"
            }
            
            if interval not in interval_map:
                interval = "5m"
            
            # Build URL
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, 1000)
            }
            
            # Fetch data
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                raise Exception(f"Binance API error: {response.status_code}")
            
            data = response.json()
            
            if not data:
                raise Exception("No data returned from Binance")
            
            # Parse data
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            df.set_index("timestamp", inplace=True)
            
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df[["open", "high", "low", "close", "volume"]].dropna()
            
        except Exception as e:
            st.warning(f"Binance failed: {str(e)[:100]}")
            return pd.DataFrame()
    
    @staticmethod
    def fetch_kucoin(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Fetch data from KuCoin (fallback)"""
        try:
            # KuCoin uses different format (e.g., BTC-USDT)
            if "-" not in symbol:
                symbol_formatted = f"{symbol[:-4]}-{symbol[-4:]}"
            else:
                symbol_formatted = symbol
            
            # Interval mapping
            interval_map = {
                "1m": "1min", "5m": "5min", "15m": "15min", 
                "30m": "30min", "1h": "1hour", "4h": "4hour", "1d": "1day"
            }
            
            kucoin_interval = interval_map.get(interval, "5min")
            
            url = f"https://api.kucoin.com/api/v1/market/candles"
            params = {
                "symbol": symbol_formatted,
                "type": kucoin_interval
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return pd.DataFrame()
            
            data = response.json()
            
            if data.get("code") != "200000" or not data.get("data"):
                return pd.DataFrame()
            
            # Parse KuCoin format (different order than Binance)
            klines = data["data"][:limit]
            
            df_data = []
            for kline in klines:
                df_data.append({
                    "timestamp": pd.to_datetime(kline[0], unit='s'),
                    "open": float(kline[1]),
                    "close": float(kline[2]),
                    "high": float(kline[3]),
                    "low": float(kline[4]),
                    "volume": float(kline[5])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            
            return df[["open", "high", "low", "close", "volume"]]
            
        except:
            return pd.DataFrame()
    
    @staticmethod
    def fetch_data(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch data from multiple sources with fallback"""
        # Try Binance first
        df = DataFetcher.fetch_binance(symbol, timeframe, limit)
        
        # If Binance fails, try KuCoin
        if df.empty:
            df = DataFetcher.fetch_kucoin(symbol, timeframe, limit)
        
        # If still empty, try with different symbol variations
        if df.empty:
            # Try with USDT if not present
            if not symbol.endswith("USDT"):
                symbol_with_usdt = f"{symbol}USDT"
                df = DataFetcher.fetch_binance(symbol_with_usdt, timeframe, limit)
            
            # Try without USDT if it was present
            elif symbol.endswith("USDT"):
                symbol_without_usdt = symbol[:-4]
                df = DataFetcher.fetch_binance(symbol_without_usdt, timeframe, limit)
        
        return df

# =========================
# INDICATOR CALCULATIONS (NO EXTERNAL DEPENDENCIES)
# =========================
class IndicatorCalculator:
    """Calculate all technical indicators without external dependencies"""
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators"""
        df = df.copy()
        
        if len(df) < 20:
            return df
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Additional indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['trend'] = np.where(df['close'] > df['ema_12'], 1, -1)
        
        return df

# =========================
# SIGNAL GENERATOR
# =========================
class SignalGenerator:
    """Generate trading signals"""
    
    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals"""
        df = df.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['signal_type'] = 'NEUTRAL'
        
        # MACD Crossover
        macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
        macd_sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
        
        # RSI Signals
        rsi_buy = (df['rsi'] < 30)
        rsi_sell = (df['rsi'] > 70)
        
        # Bollinger Band Signals
        bb_buy = (df['close'] < df['bb_lower'])
        bb_sell = (df['close'] > df['bb_upper'])
        
        # Moving Average Crossover
        ma_buy = (df['ema_12'] > df['ema_26']) & (df['ema_12'].shift() <= df['ema_26'].shift())
        ma_sell = (df['ema_12'] < df['ema_26']) & (df['ema_12'].shift() >= df['ema_26'].shift())
        
        # Combine signals
        df.loc[(macd_buy | rsi_buy | bb_buy | ma_buy), 'signal'] = 1
        df.loc[(macd_sell | rsi_sell | bb_sell | ma_sell), 'signal'] = -1
        
        # Determine signal type
        df['signal_type'] = np.where(
            df['signal'] == 1, 'BUY',
            np.where(df['signal'] == -1, 'SELL', 'NEUTRAL')
        )
        
        # Strong signals
        strong_buy = (df['signal'] == 1) & (df['rsi'] < 30) & (df['volume_ratio'] > 1.5)
        strong_sell = (df['signal'] == -1) & (df['rsi'] > 70) & (df['volume_ratio'] > 1.5)
        
        df.loc[strong_buy, 'signal_type'] = 'STRONG_BUY'
        df.loc[strong_sell, 'signal_type'] = 'STRONG_SELL'
        
        return df

# =========================
# BACKTESTING ENGINE
# =========================
class Backtester:
    """Simple backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
    
    def run(self, df: pd.DataFrame) -> Dict:
        """Run backtest"""
        if 'signal' not in df.columns:
            return {}
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        
        for i in range(1, len(df)):
            price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            # Exit conditions
            if position > 0 and price <= entry_price * 0.98:  # 2% stop loss
                pnl = (price - entry_price) * position
                capital += pnl
                trades.append({
                    'type': 'LONG',
                    'entry': entry_price,
                    'exit': price,
                    'pnl': pnl,
                    'pnl_pct': (price / entry_price - 1) * 100
                })
                position = 0
            
            # Entry conditions
            if position == 0 and signal != 0:
                if signal == 1:  # Buy
                    position = capital * 0.1 / price  # 10% position
                    entry_price = price
                elif signal == -1:  # Sell
                    position = -capital * 0.1 / price  # 10% short
                    entry_price = price
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'trades': trades
        }

# =========================
# STREAMLIT APP
# =========================
def main():
    # Page config
    st.set_page_config(
        layout="wide",
        page_title="TITAN INTRADAY PRO",
        page_icon="🚀"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background: linear-gradient(45deg, #4361EE, #3A0CA3);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #3A0CA3, #4361EE);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 TITAN INTRADAY PRO</h1>', unsafe_allow_html=True)
    st.caption("Advanced Trading Platform • Real-Time Analytics • Professional Signals")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/bitcoin--v1.png", width=80)
        
        st.header("⚙️ Market Configuration")
        
        # Symbol input with examples
        symbol = st.text_input(
            "Symbol",
            value=Config.DEFAULT_SYMBOL,
            help="Examples: BTCUSDT, ETHUSDT, BNBUSDT"
        )
        
        # Timeframe
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=1
        )
        
        # Candles
        candles = st.slider("Historical Candles", 100, Config.MAX_CANDLES, 500)
        
        # Load button
        if st.button("📥 Load Market Data", type="primary", use_container_width=True):
            st.session_state.load_data = True
            st.session_state.symbol = symbol
            st.session_state.timeframe = timeframe
            st.session_state.candles = candles
    
    # Initialize session state
    if 'load_data' not in st.session_state:
        st.session_state.load_data = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'signals' not in st.session_state:
        st.session_state.signals = None
    if 'performance' not in st.session_state:
        st.session_state.performance = None
    
    # Main content area
    if st.session_state.load_data:
        load_market_data()
    else:
        show_welcome_screen()
    
    # Footer
    st.divider()
    st.caption("⚠️ Educational Use Only • Not Financial Advice")

def show_welcome_screen():
    """Show welcome screen when no data loaded"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 50px 0;'>
            <h2>🚀 Welcome to TITAN INTRADAY PRO</h2>
            <p style='font-size: 18px; color: #888; margin: 20px 0;'>
                Professional Trading Platform with Advanced Analytics
            </p>
            <div style='background: rgba(25, 25, 35, 0.8); padding: 30px; border-radius: 10px; margin: 20px 0;'>
                <h3>📊 Get Started:</h3>
                <ol style='text-align: left; margin-left: 20px;'>
                    <li>Enter a symbol (e.g., <strong>BTCUSDT</strong>)</li>
                    <li>Select a timeframe</li>
                    <li>Choose number of historical candles</li>
                    <li>Click "Load Market Data"</li>
                </ol>
            </div>
            <p style='color: #666; margin-top: 30px;'>
                <strong>Supported symbols:</strong> BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT, XRPUSDT
            </p>
            <p style='color: #888; font-size: 14px;'>
                Note: Always use the format <strong>SYMBOLUSDT</strong> (e.g., BTCUSDT not BTC-USD)
            </p>
        </div>
        """, unsafe_allow_html=True)

def load_market_data():
    """Load and process market data"""
    with st.spinner("Loading market data..."):
        try:
            # Fetch data
            df = DataFetcher.fetch_data(
                st.session_state.symbol,
                st.session_state.timeframe,
                st.session_state.candles
            )
            
            if df.empty:
                st.error("""
                ❌ Could not fetch market data. Please check:
                1. **Symbol format**: Use BTCUSDT (not BTC-USD or BTC/USD)
                2. **Internet connection**: Ensure you're connected to the internet
                3. **Symbol availability**: Try a different symbol like ETHUSDT
                
                **Common working symbols:**
                - BTCUSDT (Bitcoin)
                - ETHUSDT (Ethereum)
                - BNBUSDT (Binance Coin)
                - ADAUSDT (Cardano)
                - SOLUSDT (Solana)
                """)
                return
            
            # Calculate indicators
            df = IndicatorCalculator.calculate_all(df)
            
            # Generate signals
            df = SignalGenerator.generate(df)
            
            # Run backtest
            backtester = Backtester(Config.INITIAL_CAPITAL)
            performance = backtester.run(df)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.signals = df[df['signal'] != 0]
            st.session_state.performance = performance
            
            # Display success
            st.success(f"✅ Successfully loaded {len(df)} candles for {st.session_state.symbol}")
            
            # Display data
            display_market_data()
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Try using BTCUSDT or ETHUSDT as the symbol")

def display_market_data():
    """Display the loaded market data"""
    df = st.session_state.df
    signals = st.session_state.signals
    
    # Current Market Overview
    st.subheader("📈 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['close'].iloc[-1]
        price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        st.metric("Current Price", f"${current_price:,.2f}", f"{price_change:+.2f}%")
    
    with col2:
        current_signal = df['signal_type'].iloc[-1]
        if "BUY" in current_signal:
            st.success(f"**{current_signal}**")
        elif "SELL" in current_signal:
            st.error(f"**{current_signal}**")
        else:
            st.info(f"**{current_signal}**")
    
    with col3:
        rsi_value = df['rsi'].iloc[-1]
        rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
        st.metric("RSI", f"{rsi_value:.1f}", rsi_status)
    
    with col4:
        volume_ratio = df['volume_ratio'].iloc[-1]
        st.metric("Volume", f"{volume_ratio:.2f}x")
    
    # Performance Metrics
    if st.session_state.performance:
        perf = st.session_state.performance
        st.subheader("📊 Performance Metrics")
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Return", f"${perf.get('total_pnl', 0):,.2f}")
        with cols[1]:
            st.metric("Win Rate", f"{perf.get('win_rate', 0):.1f}%")
        with cols[2]:
            st.metric("Total Trades", f"{perf.get('total_trades', 0)}")
        with cols[3]:
            st.metric("Winning Trades", f"{perf.get('winning_trades', 0)}")
    
    # Chart Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Price Chart", "📈 Indicators", "📋 Recent Data"])
    
    with tab1:
        display_price_chart(df, signals)
    
    with tab2:
        display_indicators(df)
    
    with tab3:
        display_recent_data(df)

def display_price_chart(df: pd.DataFrame, signals: pd.DataFrame):
    """Display price chart with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & Signals', 'Volume', 'RSI')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    fig.add_trace(
        go.Scatter(x=df.index, y=df['ema_12'], line=dict(color='orange', width=1), name='EMA 12'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['ema_26'], line=dict(color='red', width=1), name='EMA 26'),
        row=1, col=1
    )
    
    # Signals
    if not signals.empty:
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['close'],
                    mode='markers',
                    name='Buy',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['close'],
                    mode='markers',
                    name='Sell',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=1, col=1
            )
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # Volume SMA
    fig.add_trace(
        go.Scatter(x=df.index, y=df['volume_sma'], line=dict(color='yellow', width=1), name='Volume MA'),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['rsi'], line=dict(color='purple', width=2), name='RSI'),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_indicators(df: pd.DataFrame):
    """Display technical indicators"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Trend Indicators")
        
        indicators = {
            "MACD": f"{df['macd'].iloc[-1]:.4f}",
            "MACD Signal": f"{df['macd_signal'].iloc[-1]:.4f}",
            "MACD Histogram": f"{df['macd_hist'].iloc[-1]:.4f}",
            "Trend": "Bullish" if df['trend'].iloc[-1] > 0 else "Bearish"
        }
        
        for name, value in indicators.items():
            st.metric(name, value)
    
    with col2:
        st.subheader("📈 Momentum Indicators")
        
        indicators = {
            "RSI": f"{df['rsi'].iloc[-1]:.1f}",
            "Stochastic %K": f"{df['stoch_k'].iloc[-1]:.1f}",
            "Stochastic %D": f"{df['stoch_d'].iloc[-1]:.1f}",
            "Volatility": f"{df['volatility'].iloc[-1]:.2%}"
        }
        
        for name, value in indicators.items():
            st.metric(name, value)
    
    # Bollinger Bands Position
    st.subheader("📉 Bollinger Bands")
    col3, col4 = st.columns(2)
    
    with col3:
        bb_position = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / \
                     (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
        st.metric("BB Position", f"{bb_position:.2%}")
        st.metric("ATR", f"{df['atr'].iloc[-1]:.2f}")
    
    with col4:
        st.metric("Upper Band", f"${df['bb_upper'].iloc[-1]:,.2f}")
        st.metric("Lower Band", f"${df['bb_lower'].iloc[-1]:,.2f}")

def display_recent_data(df: pd.DataFrame):
    """Display recent market data"""
    st.subheader("📋 Recent Market Data")
    
    # Show last 20 rows
    display_df = df.tail(20).copy()
    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')
    
    # Format columns
    format_dict = {
        'open': '${:,.2f}',
        'high': '${:,.2f}',
        'low': '${:,.2f}',
        'close': '${:,.2f}',
        'volume': '{:,.0f}',
        'rsi': '{:.1f}',
        'macd': '{:.4f}'
    }
    
    st.dataframe(
        display_df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'signal_type']]
        .style.format(format_dict),
        use_container_width=True,
        height=400
    )

if __name__ == "__main__":
    main()
