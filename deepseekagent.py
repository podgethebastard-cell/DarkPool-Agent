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
    DEFAULT_SYMBOL = "XBTUSD"  # Kraken uses XBT for Bitcoin, XBTUSD
    DEFAULT_TIMEFRAME = "5"
    INITIAL_CAPITAL = 10000
    
    # Kraken API endpoints
    KRAKEN_BASE_URL = "https://api.kraken.com/0/public"
    
    # Common Kraken symbols mapping
    SYMBOL_MAP = {
        "BTCUSD": "XBTUSD",
        "XBTUSD": "XBTUSD",
        "ETHUSD": "ETHUSD", 
        "ETH": "ETHUSD",
        "BTC": "XBTUSD",
        "XBT": "XBTUSD",
        "SOLUSD": "SOLUSD",
        "ADAUSD": "ADAUSD",
        "DOTUSD": "DOTUSD",
        "DOGEUSD": "XDGUSD",  # Dogecoin on Kraken
        "LTCUSD": "LTCUSD",
        "XRPUSD": "XRPUSD",
        "MATICUSD": "MATICUSD",
        "AVAXUSD": "AVAXUSD",
        "LINKUSD": "LINKUSD"
    }

# =========================
# KRAKEN DATA FETCHER
# =========================
class KrakenDataFetcher:
    """Fetch market data from Kraken API"""
    
    @staticmethod
    def map_symbol_to_kraken(symbol: str) -> str:
        """Map common symbols to Kraken format"""
        symbol = symbol.upper().strip()
        
        # If it's already a Kraken symbol (starts with X or ends with USD)
        if symbol in ["XBTUSD", "ETHUSD", "SOLUSD", "ADAUSD", "DOTUSD", 
                     "XDGUSD", "LTCUSD", "XRPUSD", "MATICUSD", "AVAXUSD", "LINKUSD"]:
            return symbol
        
        # Map common symbols
        if symbol in Config.SYMBOL_MAP:
            return Config.SYMBOL_MAP[symbol]
        
        # Try to parse common formats
        if symbol.endswith("USD"):
            base = symbol[:-3]
            # Add X prefix for Bitcoin
            if base == "BTC":
                return "XBTUSD"
            return symbol
        
        # If no USD suffix, add it
        if symbol in ["BTC", "XBT"]:
            return "XBTUSD"
        elif symbol in ["ETH", "SOL", "ADA", "DOT", "DOGE", "LTC", "XRP", "MATIC", "AVAX", "LINK"]:
            return f"{symbol}USD"
        
        return symbol  # Return as-is, will fail if invalid
    
    @staticmethod
    def map_timeframe_to_kraken(tf: str) -> str:
        """Map timeframe to Kraken interval codes"""
        tf_map = {
            "1m": "1",
            "5m": "5", 
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "4h": "240",
            "1d": "1440",
            "1w": "10080",
            "2w": "21600"
        }
        return tf_map.get(tf, "5")  # Default to 5 minutes
    
    @staticmethod
    def fetch_ohlcv(symbol: str, timeframe: str = "5", limit: int = 720) -> pd.DataFrame:
        """Fetch OHLCV data from Kraken"""
        try:
            # Map symbol and timeframe
            kraken_symbol = KrakenDataFetcher.map_symbol_to_kraken(symbol)
            kraken_interval = KrakenDataFetcher.map_timeframe_to_kraken(timeframe)
            
            # Kraken API endpoint
            url = f"{Config.KRAKEN_BASE_URL}/OHLC"
            params = {
                "pair": kraken_symbol,
                "interval": kraken_interval,
                "since": int((datetime.now() - timedelta(days=30)).timestamp())  # Last 30 days
            }
            
            # Make request
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                st.error(f"Kraken API error: HTTP {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            
            if data.get("error"):
                error_msg = data["error"][0] if data["error"] else "Unknown error"
                st.error(f"Kraken API error: {error_msg}")
                st.info(f"Available Kraken USD pairs: XBTUSD, ETHUSD, SOLUSD, ADAUSD, DOTUSD, XDGUSD (DOGE), LTCUSD, XRPUSD, MATICUSD, AVAXUSD, LINKUSD")
                return pd.DataFrame()
            
            # Extract the data
            result_key = list(data["result"].keys())[0] if data["result"] else None
            if not result_key or result_key == "last":
                st.error("No data returned from Kraken")
                return pd.DataFrame()
            
            ohlc_data = data["result"][result_key]
            
            if not ohlc_data:
                st.error("Empty data from Kraken")
                return pd.DataFrame()
            
            # Parse Kraken OHLC format
            # Kraken returns: [time, open, high, low, close, vwap, volume, count]
            df = pd.DataFrame(ohlc_data, columns=[
                "timestamp", "open", "high", "low", "close", 
                "vwap", "volume", "count"
            ])
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
            df.set_index("timestamp", inplace=True)
            
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Limit to requested number of candles
            df = df.tail(min(limit, len(df)))
            
            # Sort by timestamp (just in case)
            df.sort_index(inplace=True)
            
            return df[["open", "high", "low", "close", "volume"]].dropna()
            
        except Exception as e:
            st.error(f"Error fetching from Kraken: {str(e)[:200]}")
            return pd.DataFrame()
    
    @staticmethod
    def get_ticker_info(symbol: str) -> Dict:
        """Get current ticker information"""
        try:
            kraken_symbol = KrakenDataFetcher.map_symbol_to_kraken(symbol)
            
            url = f"{Config.KRAKEN_BASE_URL}/Ticker"
            params = {"pair": kraken_symbol}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("error"):
                return {}
            
            result_key = list(data["result"].keys())[0]
            ticker_data = data["result"][result_key]
            
            return {
                "bid": float(ticker_data.get("b", [0])[0]),
                "ask": float(ticker_data.get("a", [0])[0]),
                "last": float(ticker_data.get("c", [0])[0]),
                "volume": float(ticker_data.get("v", [0])[0]),
                "high": float(ticker_data.get("h", [0])[0]),
                "low": float(ticker_data.get("l", [0])[0]),
                "open": float(ticker_data.get("o", 0))
            }
            
        except:
            return {}

# =========================
# INDICATOR CALCULATOR
# =========================
class IndicatorCalculator:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators"""
        df = df.copy()
        
        if len(df) < 20:
            return df
        
        # Price returns
        df['returns'] = df['close'].pct_change()
        
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
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # VWAP (using Kraken's vwap if available, otherwise calculate)
        if 'vwap' in df.columns:
            df['vwap'] = df['vwap'].astype(float)
        else:
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Additional indicators
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['trend'] = np.where(df['close'] > df['ema_12'], 1, -1)
        df['momentum'] = df['close'].pct_change(10)
        
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
        rsi_buy = (df['rsi'] < 30) & (df['rsi'].shift() >= 30)
        rsi_sell = (df['rsi'] > 70) & (df['rsi'].shift() <= 70)
        
        # Bollinger Band Signals
        bb_buy = (df['close'] < df['bb_lower']) & (df['close'].shift() >= df['bb_lower'].shift())
        bb_sell = (df['close'] > df['bb_upper']) & (df['close'].shift() <= df['bb_upper'].shift())
        
        # Moving Average Crossover
        ma_buy = (df['ema_12'] > df['ema_26']) & (df['ema_12'].shift() <= df['ema_26'].shift())
        ma_sell = (df['ema_12'] < df['ema_26']) & (df['ema_12'].shift() >= df['ema_26'].shift())
        
        # Volume Confirmation
        volume_spike = df['volume_ratio'] > 1.5
        
        # Combine signals
        buy_conditions = (macd_buy | rsi_buy | bb_buy | ma_buy) & volume_spike
        sell_conditions = (macd_sell | rsi_sell | bb_sell | ma_sell) & volume_spike
        
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Signal type
        df['signal_type'] = np.where(
            df['signal'] == 1, 'BUY',
            np.where(df['signal'] == -1, 'SELL', 'NEUTRAL')
        )
        
        # Strong signals
        strong_buy = (df['signal'] == 1) & (df['rsi'] < 25) & (df['volume_ratio'] > 2)
        strong_sell = (df['signal'] == -1) & (df['rsi'] > 75) & (df['volume_ratio'] > 2)
        
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
        equity_curve = []
        
        for i in range(1, len(df)):
            price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            # Current equity
            current_equity = capital + (position * price)
            equity_curve.append(current_equity)
            
            # Exit conditions (trailing stop)
            if position > 0:  # Long position
                stop_loss = entry_price * 0.98  # 2% stop loss
                
                if price <= stop_loss:
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
            
            elif position < 0:  # Short position
                stop_loss = entry_price * 1.02  # 2% stop loss
                
                if price >= stop_loss:
                    pnl = (entry_price - price) * abs(position)
                    capital += pnl
                    trades.append({
                        'type': 'SHORT',
                        'entry': entry_price,
                        'exit': price,
                        'pnl': pnl,
                        'pnl_pct': (entry_price / price - 1) * 100
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
        
        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': ((capital - self.initial_capital) / self.initial_capital) * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'trades': trades[-10:] if trades else []  # Last 10 trades
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
    .symbol-box {
        background: rgba(25, 25, 35, 0.8);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4CC9F0;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 TITAN INTRADAY PRO (KRAKEN)</h1>', unsafe_allow_html=True)
    st.caption("Advanced Trading Platform • Kraken API • Real-Time Analytics")
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Kraken_logo.svg/2560px-Kraken_logo.svg.png", width=120)
        
        st.header("⚙️ Kraken Configuration")
        
        # Symbol input with auto-suggest
        symbol = st.selectbox(
            "Select Symbol",
            options=[
                "XBTUSD (Bitcoin)",
                "ETHUSD (Ethereum)",
                "SOLUSD (Solana)",
                "ADAUSD (Cardano)",
                "DOTUSD (Polkadot)",
                "XDGUSD (Dogecoin)",
                "LTCUSD (Litecoin)",
                "XRPUSD (Ripple)",
                "MATICUSD (Polygon)",
                "AVAXUSD (Avalanche)",
                "LINKUSD (Chainlink)"
            ],
            index=0,
            help="Select a Kraken USD trading pair"
        )
        
        # Extract symbol code
        selected_symbol = symbol.split(" ")[0]
        
        # Timeframe
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=1,
            help="Chart timeframe"
        )
        
        # Candles
        candles = st.slider("Historical Candles", 100, Config.MAX_CANDLES, 500)
        
        # Load button
        if st.button("📥 Load Market Data", type="primary", use_container_width=True):
            with st.spinner(f"Loading {selected_symbol} data from Kraken..."):
                load_market_data(selected_symbol, timeframe, candles)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'signals' not in st.session_state:
        st.session_state.signals = None
    if 'performance' not in st.session_state:
        st.session_state.performance = None
    if 'symbol' not in st.session_state:
        st.session_state.symbol = Config.DEFAULT_SYMBOL
    
    # Display data if loaded
    if st.session_state.df is not None:
        display_market_data()
    else:
        show_welcome_screen()
    
    # Footer
    st.divider()
    st.caption("⚠️ Educational Use Only • Not Financial Advice • Powered by Kraken API")

def show_welcome_screen():
    """Show welcome screen when no data loaded"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 50px 0;'>
            <h2>🚀 Welcome to TITAN INTRADAY PRO</h2>
            <p style='font-size: 18px; color: #888; margin: 20px 0;'>
                Professional Trading Platform Powered by Kraken API
            </p>
            
            <div class="symbol-box">
                <h3>📊 Available Kraken USD Pairs:</h3>
                <ul style='text-align: left;'>
                    <li><strong>XBTUSD</strong> - Bitcoin (BTC)</li>
                    <li><strong>ETHUSD</strong> - Ethereum (ETH)</li>
                    <li><strong>SOLUSD</strong> - Solana (SOL)</li>
                    <li><strong>ADAUSD</strong> - Cardano (ADA)</li>
                    <li><strong>DOTUSD</strong> - Polkadot (DOT)</li>
                    <li><strong>XDGUSD</strong> - Dogecoin (DOGE)</li>
                    <li><strong>LTCUSD</strong> - Litecoin (LTC)</li>
                    <li><strong>XRPUSD</strong> - Ripple (XRP)</li>
                    <li><strong>MATICUSD</strong> - Polygon (MATIC)</li>
                    <li><strong>AVAXUSD</strong> - Avalanche (AVAX)</li>
                    <li><strong>LINKUSD</strong> - Chainlink (LINK)</li>
                </ul>
            </div>
            
            <p style='color: #666; margin-top: 30px;'>
                <strong>📈 How to start:</strong>
                <br>1. Select a symbol from the sidebar
                <br>2. Choose timeframe
                <br>3. Click "Load Market Data"
            </p>
            
            <div style='background: rgba(0, 200, 83, 0.1); padding: 15px; border-radius: 10px; margin-top: 20px;'>
                <p style='color: #00C853;'>
                    ✅ <strong>Kraken API is reliable and doesn't require API keys for public data!</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def load_market_data(symbol: str, timeframe: str, candles: int):
    """Load and process market data"""
    try:
        # Show status
        status_text = st.empty()
        status_text.info(f"Fetching {symbol} data from Kraken...")
        
        # Fetch data
        df = KrakenDataFetcher.fetch_ohlcv(symbol, timeframe, candles)
        
        if df.empty:
            st.error(f"❌ No data returned for {symbol}. Please try a different symbol.")
            return
        
        status_text.info("Calculating indicators...")
        
        # Calculate indicators
        df = IndicatorCalculator.calculate_all(df)
        
        status_text.info("Generating trading signals...")
        
        # Generate signals
        df = SignalGenerator.generate(df)
        
        status_text.info("Running backtest...")
        
        # Run backtest
        backtester = Backtester(Config.INITIAL_CAPITAL)
        performance = backtester.run(df)
        
        # Store in session state
        st.session_state.df = df
        st.session_state.signals = df[df['signal'] != 0]
        st.session_state.performance = performance
        st.session_state.symbol = symbol
        
        status_text.empty()
        st.success(f"✅ Successfully loaded {len(df)} candles for {symbol}")
        
        # Also get current ticker info
        ticker_info = KrakenDataFetcher.get_ticker_info(symbol)
        if ticker_info:
            st.session_state.ticker_info = ticker_info
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Try using XBTUSD or ETHUSD as the symbol")

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
        st.metric(
            "Current Price", 
            f"${current_price:,.2f}", 
            f"{price_change:+.2f}%"
        )
    
    with col2:
        current_signal = df['signal_type'].iloc[-1]
        signal_color = "green" if "BUY" in current_signal else "red" if "SELL" in current_signal else "gray"
        st.markdown(f"""
        <div style='background: rgba({'0, 200, 83' if 'BUY' in current_signal else '255, 61, 0' if 'SELL' in current_signal else '100, 100, 100'}, 0.2); 
                    padding: 15px; border-radius: 10px; border-left: 4px solid {signal_color};'>
            <h3 style='margin: 0; color: {signal_color};'>Current Signal: {current_signal}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        rsi_value = df['rsi'].iloc[-1]
        rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
        rsi_color = "green" if rsi_value < 30 else "red" if rsi_value > 70 else "gray"
        st.metric("RSI", f"{rsi_value:.1f}", rsi_status)
    
    with col4:
        volume_ratio = df['volume_ratio'].iloc[-1]
        volume_status = "High" if volume_ratio > 1.5 else "Normal" if volume_ratio > 0.5 else "Low"
        st.metric("Volume", f"{volume_ratio:.2f}x", volume_status)
    
    # Performance Metrics
    if st.session_state.performance:
        perf = st.session_state.performance
        st.subheader("📊 Performance Metrics")
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Return", f"{perf.get('total_return', 0):.2f}%")
        with cols[1]:
            st.metric("Win Rate", f"{perf.get('win_rate', 0):.1f}%")
        with cols[2]:
            st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
        with cols[3]:
            st.metric("Total Trades", f"{perf.get('total_trades', 0)}")
    
    # Chart Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Price Chart", "📈 Indicators", "📋 Data & Signals"])
    
    with tab1:
        display_price_chart(df, signals)
    
    with tab2:
        display_indicators(df)
    
    with tab3:
        display_data_and_signals(df)

def display_price_chart(df: pd.DataFrame, signals: pd.DataFrame):
    """Display price chart with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & Trading Signals', 'Volume', 'RSI & MACD')
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
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['bb_upper'], line=dict(color='gray', width=1, dash='dash'), 
                  name='BB Upper', showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['bb_lower'], line=dict(color='gray', width=1, dash='dash'), 
                  fill='tonexty', name='BB Lower', showlegend=False),
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
                    name='Buy Signal',
                    marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='darkgreen'))
                ),
                row=1, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='darkred'))
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
    
    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df['macd'], line=dict(color='blue', width=2), name='MACD'),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['macd_signal'], line=dict(color='orange', width=2), name='Signal'),
        row=3, col=1
    )
    
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
            "ADX": "N/A",  # Not calculated in simple version
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
            "Momentum (10)": f"{df['momentum'].iloc[-1]:.2%}",
            "Volatility": f"{df['volatility'].iloc[-1]:.2%}"
        }
        
        for name, value in indicators.items():
            st.metric(name, value)
    
    # Support/Resistance Levels
    st.subheader("📉 Support & Resistance")
    col3, col4 = st.columns(2)
    
    with col3:
        bb_position = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / \
                     (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
        st.metric("BB Position", f"{bb_position:.2%}")
        st.metric("ATR", f"{df['atr'].iloc[-1]:.2f}")
    
    with col4:
        st.metric("Upper Band", f"${df['bb_upper'].iloc[-1]:,.2f}")
        st.metric("Lower Band", f"${df['bb_lower'].iloc[-1]:,.2f}")

def display_data_and_signals(df: pd.DataFrame):
    """Display recent data and signals"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Recent Market Data")
        
        # Show last 10 rows
        display_df = df.tail(10).copy()
        display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')
        
        format_dict = {
            'open': '${:,.2f}',
            'high': '${:,.2f}',
            'low': '${:,.2f}',
            'close': '${:,.2f}',
            'volume': '{:,.0f}',
            'rsi': '{:.1f}'
        }
        
        st.dataframe(
            display_df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'signal_type']]
            .style.format(format_dict),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.subheader("🎯 Recent Trading Signals")
        
        # Get recent signals
        signals_df = df[df['signal'] != 0].tail(10).copy()
        
        if not signals_df.empty:
            signals_df.index = signals_df.index.strftime('%Y-%m-%d %H:%M')
            
            # Color function for signals
            def color_signal(val):
                if 'BUY' in val:
                    return 'color: green; font-weight: bold'
                elif 'SELL' in val:
                    return 'color: red; font-weight: bold'
                return 'color: gray'
            
            st.dataframe(
                signals_df[['close', 'rsi', 'signal_type']]
                .style.format({'close': '${:,.2f}', 'rsi': '{:.1f}'})
                .applymap(color_signal, subset=['signal_type']),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No trading signals generated yet")
        
        # Performance summary
        if st.session_state.performance:
            perf = st.session_state.performance
            if perf.get('trades'):
                st.subheader("💰 Recent Trades")
                trades_df = pd.DataFrame(perf['trades'])
                st.dataframe(
                    trades_df.tail(5).style.format({
                        'entry': '${:,.2f}',
                        'exit': '${:,.2f}',
                        'pnl': '${:,.2f}',
                        'pnl_pct': '{:.2f}%'
                    }),
                    use_container_width=True,
                    height=250
                )

if __name__ == "__main__":
    main()
