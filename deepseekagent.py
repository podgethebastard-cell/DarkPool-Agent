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
# TELEGRAM BROADCASTER
# =========================
class TelegramBroadcaster:
    """Telegram broadcast functionality"""
    
    def __init__(self):
        self.base_url = "https://api.telegram.org/bot"
    
    def send_message(self, bot_token: str, chat_id: str, message: str, 
                    parse_mode: str = "HTML", disable_notification: bool = False) -> Dict:
        """Send message to Telegram"""
        try:
            if not bot_token or not chat_id:
                return {"success": False, "error": "Bot token or chat ID missing"}
            
            url = f"{self.base_url}{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification
            }
            
            response = requests.post(url, json=payload, timeout=10)
            result = response.json()
            
            if result.get("ok"):
                return {"success": True, "message_id": result["result"]["message_id"]}
            else:
                return {"success": False, "error": result.get("description", "Unknown error")}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def send_signal_alert(self, bot_token: str, chat_id: str, signal_data: Dict) -> Dict:
        """Send formatted signal alert to Telegram"""
        symbol = signal_data.get('symbol', 'Unknown')
        signal = signal_data.get('signal', 'NEUTRAL')
        price = signal_data.get('price', 0)
        rsi = signal_data.get('rsi', 0)
        volume_ratio = signal_data.get('volume_ratio', 0)
        confidence = signal_data.get('confidence', 0)
        
        # Create emoji based on signal
        if "BUY" in signal:
            emoji = "🟢"
            action = "BUY"
        elif "SELL" in signal:
            emoji = "🔴"
            action = "SELL"
        else:
            emoji = "⚪"
            action = "NEUTRAL"
        
        # Create message with HTML formatting
        message = f"""
{emoji} <b>TITAN INTRADAY PRO SIGNAL</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Signal:</b> <b>{signal}</b>
<b>Price:</b> ${price:,.2f}
<b>RSI:</b> {rsi:.1f}
<b>Volume Ratio:</b> {volume_ratio:.2f}x
<b>Confidence:</b> {confidence:.0%}

<b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(bot_token, chat_id, message)
    
    def send_performance_report(self, bot_token: str, chat_id: str, performance_data: Dict) -> Dict:
        """Send performance report to Telegram"""
        total_return = performance_data.get('total_return', 0)
        win_rate = performance_data.get('win_rate', 0)
        total_trades = performance_data.get('total_trades', 0)
        sharpe = performance_data.get('sharpe_ratio', 0)
        
        message = f"""
📊 <b>TITAN PERFORMANCE REPORT</b> 📊

<b>Total Return:</b> {total_return:+.2f}%
<b>Win Rate:</b> {win_rate:.1f}%
<b>Total Trades:</b> {total_trades}
<b>Sharpe Ratio:</b> {sharpe:.2f}

<b>Report Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        return self.send_message(bot_token, chat_id, message)
    
    def test_connection(self, bot_token: str, chat_id: str) -> Dict:
        """Test Telegram connection"""
        try:
            if not bot_token or not chat_id:
                return {"success": False, "error": "Bot token or chat ID missing"}
            
            url = f"{self.base_url}{bot_token}/getMe"
            response = requests.get(url, timeout=10)
            bot_info = response.json()
            
            if not bot_info.get("ok"):
                return {"success": False, "error": "Invalid bot token"}
            
            # Try to send a test message
            test_message = "✅ TITAN INTRADAY PRO - Connection Test Successful!"
            test_result = self.send_message(bot_token, chat_id, test_message)
            
            if test_result["success"]:
                return {"success": True, "bot_username": bot_info["result"]["username"]}
            else:
                return {"success": False, "error": f"Can't send to chat: {test_result.get('error')}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

# =========================
# KRAKEN DATA FETCHER
# =========================
class KrakenDataFetcher:
    """Fetch market data from Kraken API"""
    
    @staticmethod
    def map_symbol_to_kraken(symbol: str) -> str:
        """Map common symbols to Kraken format"""
        symbol = symbol.upper().strip()
        
        # If it's already a Kraken symbol
        if symbol in Config.SYMBOL_MAP.values():
            return symbol
        
        # Map common symbols
        if symbol in Config.SYMBOL_MAP:
            return Config.SYMBOL_MAP[symbol]
        
        # Try to parse common formats
        if symbol.endswith("USD"):
            base = symbol[:-3]
            if base == "BTC":
                return "XBTUSD"
            return symbol
        
        # If no USD suffix, add it
        if symbol in ["BTC", "XBT"]:
            return "XBTUSD"
        elif symbol in ["ETH", "SOL", "ADA", "DOT", "DOGE", "LTC", "XRP", "MATIC", "AVAX", "LINK"]:
            return f"{symbol}USD"
        
        return symbol
    
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
        return tf_map.get(tf, "5")
    
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
                "interval": kraken_interval
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
                return pd.DataFrame()
            
            # Extract the data
            result_keys = list(data["result"].keys())
            if not result_keys or result_keys[0] == "last":
                st.error("No data returned from Kraken")
                return pd.DataFrame()
            
            result_key = result_keys[0]
            ohlc_data = data["result"][result_key]
            
            if not ohlc_data:
                st.error("Empty data from Kraken")
                return pd.DataFrame()
            
            # Parse Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
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
            df.sort_index(inplace=True)
            
            return df[["open", "high", "low", "close", "volume"]].dropna()
            
        except Exception as e:
            st.error(f"Error fetching from Kraken: {str(e)}")
            return pd.DataFrame()

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
            # Create basic columns with NaN
            for col in ['returns', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 
                       'macd_signal', 'macd_hist', 'rsi', 'bb_middle', 'bb_upper', 
                       'bb_lower', 'atr', 'volume_sma', 'volume_ratio', 'vwap', 
                       'stoch_k', 'stoch_d', 'volatility', 'momentum']:
                df[col] = np.nan
            df['trend'] = 0
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
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum().replace(0, np.nan)
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        range_14 = high_14 - low_14
        df['stoch_k'] = 100 * ((df['close'] - low_14) / range_14.replace(0, np.nan))
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
        
        # Check if we have enough data for indicators
        if len(df) < 20:
            return df
        
        # Check if required columns exist
        required_cols = ['macd', 'macd_signal', 'rsi', 'bb_lower', 'bb_upper', 'ema_12', 'ema_26', 'volume_ratio']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return df
        
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
    
    @staticmethod
    def get_current_signal_data(df: pd.DataFrame, symbol: str) -> Dict:
        """Get current signal data for broadcasting"""
        if len(df) == 0:
            return {}
        
        latest = df.iloc[-1]
        
        return {
            'symbol': symbol,
            'signal': latest.get('signal_type', 'NEUTRAL'),
            'price': latest.get('close', 0),
            'rsi': latest.get('rsi', 50),
            'volume_ratio': latest.get('volume_ratio', 1),
            'confidence': min(abs(latest.get('rsi', 50) - 50) / 50, 1.0),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# =========================
# BACKTESTING ENGINE
# =========================
class Backtester:
    """Simple backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
    
    def run(self, df: pd.DataFrame) -> Dict:
        """Run backtest"""
        if 'signal' not in df.columns or len(df) < 20:
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'trades': []
            }
        
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
                        'pnl_pct': (price / entry_price - 1) * 100 if entry_price > 0 else 0
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
                        'pnl_pct': (entry_price / price - 1) * 100 if price > 0 else 0
                    })
                    position = 0
            
            # Entry conditions
            if position == 0 and signal != 0:
                if signal == 1:  # Buy
                    position = capital * 0.1 / price if price > 0 else 0
                    entry_price = price
                elif signal == -1:  # Sell
                    position = -capital * 0.1 / price if price > 0 else 0
                    entry_price = price
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        total_return = ((capital - self.initial_capital) / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'trades': trades[-10:] if trades else []
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
    .telegram-success {
        background: rgba(0, 200, 83, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #00C853;
        margin: 10px 0;
    }
    .telegram-error {
        background: rgba(255, 61, 0, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #FF3D00;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 TITAN INTRADAY PRO (KRAKEN)</h1>', unsafe_allow_html=True)
    st.caption("Advanced Trading Platform • Kraken API • Telegram Broadcast")
    
    # Initialize Telegram broadcaster
    telegram = TelegramBroadcaster()
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Kraken_logo.svg/2560px-Kraken_logo.svg.png", width=120)
        
        # Market Configuration
        st.header("⚙️ Kraken Configuration")
        
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
        
        selected_symbol = symbol.split(" ")[0]
        
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=1,
            help="Chart timeframe"
        )
        
        candles = st.slider("Historical Candles", 100, Config.MAX_CANDLES, 500)
        
        # Load button
        if st.button("📥 Load Market Data", type="primary", use_container_width=True):
            with st.spinner(f"Loading {selected_symbol} data from Kraken..."):
                load_market_data(selected_symbol, timeframe, candles)
        
        # Telegram Configuration
        st.header("📡 Telegram Broadcast")
        
        with st.expander("Telegram Settings", expanded=True):
            bot_token = st.text_input(
                "Bot Token",
                type="password",
                help="Get this from @BotFather on Telegram"
            )
            
            chat_id = st.text_input(
                "Chat ID",
                help="Channel/Group ID (use @getidsbot to find)"
            )
            
            # Store in session state
            st.session_state.telegram_bot_token = bot_token
            st.session_state.telegram_chat_id = chat_id
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔗 Test Connection", use_container_width=True):
                    if bot_token and chat_id:
                        with st.spinner("Testing Telegram connection..."):
                            result = telegram.test_connection(bot_token, chat_id)
                            if result["success"]:
                                st.markdown(f"""
                                <div class="telegram-success">
                                    ✅ Connected to @{result['bot_username']}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="telegram-error">
                                    ❌ Connection failed: {result.get('error', 'Unknown error')}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("Please enter Bot Token and Chat ID")
            
            with col2:
                if st.button("🔄 Auto Broadcast", use_container_width=True):
                    if 'df' in st.session_state and st.session_state.df is not None:
                        if bot_token and chat_id:
                            signal_data = SignalGenerator.get_current_signal_data(
                                st.session_state.df, 
                                st.session_state.symbol
                            )
                            with st.spinner("Sending signal to Telegram..."):
                                result = telegram.send_signal_alert(bot_token, chat_id, signal_data)
                                if result["success"]:
                                    st.markdown(f"""
                                    <div class="telegram-success">
                                        ✅ Signal broadcasted successfully!
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="telegram-error">
                                        ❌ Broadcast failed: {result.get('error', 'Unknown error')}
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.warning("Please configure Telegram settings first")
                    else:
                        st.warning("Please load market data first")
            
            # Broadcast options
            st.subheader("Broadcast Options")
            
            auto_broadcast = st.checkbox(
                "Auto-broadcast signals",
                value=False,
                help="Automatically send signals when they occur"
            )
            
            broadcast_performance = st.checkbox(
                "Broadcast performance reports",
                value=False,
                help="Send daily performance summaries"
            )
            
            if auto_broadcast:
                st.info("Auto-broadcast will send signals when BUY/SELL signals are detected")
            
            if broadcast_performance:
                st.info("Performance reports will be sent daily at market close")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'signals' not in st.session_state:
        st.session_state.signals = None
    if 'performance' not in st.session_state:
        st.session_state.performance = None
    if 'symbol' not in st.session_state:
        st.session_state.symbol = Config.DEFAULT_SYMBOL
    if 'telegram_bot_token' not in st.session_state:
        st.session_state.telegram_bot_token = ""
    if 'telegram_chat_id' not in st.session_state:
        st.session_state.telegram_chat_id = ""
    
    # Display data if loaded
    if st.session_state.df is not None:
        display_market_data(telegram)
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
                Professional Trading Platform with Telegram Broadcast
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
            
            <div class="symbol-box">
                <h3>📡 Telegram Setup:</h3>
                <ol style='text-align: left;'>
                    <li>Create a bot via @BotFather on Telegram</li>
                    <li>Get your bot token</li>
                    <li>Add bot to your channel/group</li>
                    <li>Get chat ID using @getidsbot</li>
                    <li>Configure in sidebar</li>
                </ol>
            </div>
            
            <p style='color: #666; margin-top: 30px;'>
                <strong>📈 How to start:</strong>
                <br>1. Select a symbol from the sidebar
                <br>2. Configure Telegram (optional)
                <br>3. Click "Load Market Data"
            </p>
        </div>
        """, unsafe_allow_html=True)

def load_market_data(symbol: str, timeframe: str, candles: int):
    """Load and process market data"""
    try:
        status_text = st.empty()
        status_text.info(f"Fetching {symbol} data from Kraken...")
        
        # Fetch data
        df = KrakenDataFetcher.fetch_ohlcv(symbol, timeframe, candles)
        
        if df.empty:
            st.error(f"❌ No data returned for {symbol}. Please try a different symbol.")
            return
        
        if len(df) < 20:
            st.warning(f"⚠️ Only {len(df)} candles loaded. Some indicators may not work properly.")
        
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
        
        # Auto-broadcast if enabled and Telegram is configured
        if (st.session_state.get('telegram_bot_token') and 
            st.session_state.get('telegram_chat_id') and
            'signal_type' in df.columns):
            
            current_signal = df['signal_type'].iloc[-1]
            if current_signal in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
                status_text.info("Broadcasting signal to Telegram...")
                telegram = TelegramBroadcaster()
                signal_data = SignalGenerator.get_current_signal_data(df, symbol)
                telegram.send_signal_alert(
                    st.session_state.telegram_bot_token,
                    st.session_state.telegram_chat_id,
                    signal_data
                )
        
        status_text.empty()
        st.success(f"✅ Successfully loaded {len(df)} candles for {symbol}")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Try using XBTUSD or ETHUSD as the symbol")

def display_market_data(telegram: TelegramBroadcaster):
    """Display the loaded market data"""
    df = st.session_state.df
    signals = st.session_state.signals
    
    # Current Market Overview
    st.subheader("📈 Market Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if len(df) > 1:
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
            st.metric(
                "Current Price", 
                f"${current_price:,.2f}", 
                f"{price_change:+.2f}%"
            )
        else:
            st.metric("Current Price", f"${df['close'].iloc[-1]:,.2f}" if len(df) > 0 else "$0.00")
    
    with col2:
        current_signal = df['signal_type'].iloc[-1] if 'signal_type' in df.columns else 'NEUTRAL'
        signal_color = "green" if "BUY" in current_signal else "red" if "SELL" in current_signal else "gray"
        st.markdown(f"""
        <div style='background: rgba({'0, 200, 83' if 'BUY' in current_signal else '255, 61, 0' if 'SELL' in current_signal else '100, 100, 100'}, 0.2); 
                    padding: 15px; border-radius: 10px; border-left: 4px solid {signal_color};'>
            <h3 style='margin: 0; color: {signal_color};'>Signal: {current_signal}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]):
            rsi_value = df['rsi'].iloc[-1]
            rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
            st.metric("RSI", f"{rsi_value:.1f}", rsi_status)
        else:
            st.metric("RSI", "N/A")
    
    with col4:
        if 'volume_ratio' in df.columns and not pd.isna(df['volume_ratio'].iloc[-1]):
            volume_ratio = df['volume_ratio'].iloc[-1]
            volume_status = "High" if volume_ratio > 1.5 else "Normal" if volume_ratio > 0.5 else "Low"
            st.metric("Volume", f"{volume_ratio:.2f}x", volume_status)
        else:
            st.metric("Volume", "N/A")
    
    with col5:
        # Telegram Broadcast Button
        if (st.session_state.get('telegram_bot_token') and 
            st.session_state.get('telegram_chat_id')):
            
            if st.button("📤 Broadcast Signal", use_container_width=True):
                signal_data = SignalGenerator.get_current_signal_data(df, st.session_state.symbol)
                with st.spinner("Sending to Telegram..."):
                    result = telegram.send_signal_alert(
                        st.session_state.telegram_bot_token,
                        st.session_state.telegram_chat_id,
                        signal_data
                    )
                    if result["success"]:
                        st.markdown(f"""
                        <div class="telegram-success">
                            ✅ Signal broadcasted successfully!
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="telegram-error">
                            ❌ Broadcast failed: {result.get('error', 'Unknown error')}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Configure Telegram to broadcast")
    
    # Performance Metrics
    if st.session_state.performance:
        perf = st.session_state.performance
        st.subheader("📊 Performance Metrics")
        
        cols = st.columns(6)
        with cols[0]:
            st.metric("Total Return", f"{perf.get('total_return', 0):.2f}%")
        with cols[1]:
            st.metric("Win Rate", f"{perf.get('win_rate', 0):.1f}%")
        with cols[2]:
            st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
        with cols[3]:
            st.metric("Total Trades", f"{perf.get('total_trades', 0)}")
        with cols[4]:
            st.metric("Winning Trades", f"{perf.get('winning_trades', 0)}")
        with cols[5]:
            # Broadcast Performance Button
            if (st.session_state.get('telegram_bot_token') and 
                st.session_state.get('telegram_chat_id')):
                
                if st.button("📊 Broadcast Report", use_container_width=True):
                    with st.spinner("Sending performance report..."):
                        result = telegram.send_performance_report(
                            st.session_state.telegram_bot_token,
                            st.session_state.telegram_chat_id,
                            perf
                        )
                        if result["success"]:
                            st.markdown(f"""
                            <div class="telegram-success">
                                ✅ Performance report sent!
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="telegram-error">
                                ❌ Report failed: {result.get('error', 'Unknown error')}
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chart Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Price Chart", "📈 Indicators", "📋 Data & Signals"])
    
    with tab1:
        display_price_chart(df, signals)
    
    with tab2:
        display_indicators(df)
    
    with tab3:
        display_data_and_signals(df, telegram)

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
    if 'ema_12' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ema_12'], line=dict(color='orange', width=1), name='EMA 12'),
            row=1, col=1
        )
    if 'ema_26' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ema_26'], line=dict(color='red', width=1), name='EMA 26'),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
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
    if signals is not None and not signals.empty:
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
    if 'volume_sma' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['volume_sma'], line=dict(color='yellow', width=1), name='Volume MA'),
            row=2, col=1
        )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], line=dict(color='purple', width=2), name='RSI'),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
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
        
        # Helper function to safely get values
        def get_value(col, default="N/A", formatter=lambda x: f"{x:.4f}"):
            if col in df.columns and not pd.isna(df[col].iloc[-1]):
                return formatter(df[col].iloc[-1])
            return default
        
        indicators = {
            "MACD": get_value('macd'),
            "MACD Signal": get_value('macd_signal'),
            "MACD Histogram": get_value('macd_hist'),
            "Trend": "Bullish" if ('trend' in df.columns and df['trend'].iloc[-1] > 0) else "Bearish" if ('trend' in df.columns and df['trend'].iloc[-1] < 0) else "N/A"
        }
        
        for name, value in indicators.items():
            st.metric(name, value)
    
    with col2:
        st.subheader("📈 Momentum Indicators")
        
        indicators = {
            "RSI": get_value('rsi', formatter=lambda x: f"{x:.1f}"),
            "Stochastic %K": get_value('stoch_k', formatter=lambda x: f"{x:.1f}"),
            "Stochastic %D": get_value('stoch_d', formatter=lambda x: f"{x:.1f}"),
            "Momentum (10)": get_value('momentum', formatter=lambda x: f"{x:.2%}"),
            "Volatility": get_value('volatility', formatter=lambda x: f"{x:.2%}")
        }
        
        for name, value in indicators.items():
            st.metric(name, value)

def display_data_and_signals(df: pd.DataFrame, telegram: TelegramBroadcaster):
    """Display recent data and signals"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Recent Market Data")
        
        # Show last 10 rows
        display_df = df.tail(10).copy()
        display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')
        
        # Create format dictionary
        format_dict = {}
        for col in ['open', 'high', 'low', 'close', 'volume', 'rsi']:
            if col in display_df.columns:
                if col == 'rsi':
                    format_dict[col] = '{:.1f}'
                elif col == 'volume':
                    format_dict[col] = '{:,.0f}'
                else:
                    format_dict[col] = '${:,.2f}'
        
        # Select columns that exist
        columns_to_show = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'signal_type']
        existing_cols = [col for col in columns_to_show if col in display_df.columns]
        
        st.dataframe(
            display_df[existing_cols]
            .style.format(format_dict),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.subheader("🎯 Recent Trading Signals")
        
        # Get recent signals
        if 'signal' in df.columns:
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
                
                # Broadcast individual signal button
                if (st.session_state.get('telegram_bot_token') and 
                    st.session_state.get('telegram_chat_id')):
                    
                    with st.expander("Broadcast Selected Signal"):
                        selected_index = st.selectbox(
                            "Select signal to broadcast",
                            range(len(signals_df))
                        )
                        
                        if st.button("📤 Broadcast This Signal"):
                            selected_signal = signals_df.iloc[selected_index]
                            signal_data = {
                                'symbol': st.session_state.symbol,
                                'signal': selected_signal.get('signal_type', 'NEUTRAL'),
                                'price': selected_signal.get('close', 0),
                                'rsi': selected_signal.get('rsi', 0),
                                'volume_ratio': selected_signal.get('volume_ratio', 1),
                                'confidence': min(abs(selected_signal.get('rsi', 50) - 50) / 50, 1.0),
                                'timestamp': selected_signal.name
                            }
                            
                            with st.spinner("Broadcasting..."):
                                result = telegram.send_signal_alert(
                                    st.session_state.telegram_bot_token,
                                    st.session_state.telegram_chat_id,
                                    signal_data
                                )
                                if result["success"]:
                                    st.success("✅ Signal broadcasted!")
                                else:
                                    st.error(f"❌ Failed: {result.get('error')}")
            else:
                st.info("No trading signals generated yet")
        else:
            st.info("Signal column not available")

if __name__ == "__main__":
    main()
