import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================
CONFIG = {
    'app_name': 'TITAN INTRADAY PRO',
    'cache_ttl': 30,
    'max_candles': 2000,
    'db_path': 'titan_trading.db',
    'default_symbol': 'BTCUSDT',
    'default_timeframe': '5m',
    'initial_capital': 10000,
    'max_position_size': 0.1,
    'min_position_size': 0.01,
}

# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# DATABASE MANAGER
# =========================
class DatabaseManager:
    def __init__(self, db_path: str = CONFIG['db_path']):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                signal_type TEXT,
                price REAL,
                confidence REAL,
                processed INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                entry_time DATETIME,
                exit_time DATETIME,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                size REAL,
                pnl REAL,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_signal(self, signal: Dict) -> int:
        """Save signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (timestamp, symbol, signal_type, price, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            signal['timestamp'],
            signal['symbol'],
            signal['signal_type'],
            signal['price'],
            signal['confidence']
        ))
        
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return signal_id
    
    def get_recent_signals(self, symbol: str, limit: int = 20) -> List[Dict]:
        """Get recent signals for a symbol"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM signals 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (symbol, limit))
        
        signals = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return signals

# =========================
# MARKET DATA MANAGER
# =========================
class MarketDataManager:
    def __init__(self):
        self.cache = {}
    
    def fetch_ohlcv_binance(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data from Binance API"""
        try:
            # Clean symbol format
            symbol = symbol.upper().replace("-", "").replace("/", "")
            if not symbol.endswith("USDT"):
                symbol += "USDT"
            
            # Map intervals
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "4h": "4h", "1d": "1d"
            }
            
            binance_interval = interval_map.get(interval, "5m")
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": binance_interval,
                "limit": min(limit, 1000)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            df.set_index("timestamp", inplace=True)
            
            # Convert to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df[["open", "high", "low", "close", "volume"]].dropna()
        
        except Exception as e:
            logger.error(f"Error fetching data from Binance: {e}")
            return pd.DataFrame()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Main method to fetch OHLCV data"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        # Check cache
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < CONFIG['cache_ttl']:
                return data.copy()
        
        # Fetch from Binance
        df = self.fetch_ohlcv_binance(symbol, timeframe, limit)
        
        # Cache the result
        if not df.empty:
            self.cache[cache_key] = (datetime.now(), df.copy())
        
        return df

# =========================
# TECHNICAL INDICATOR ENGINE (PURE PYTHON)
# =========================
class IndicatorEngine:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive set of technical indicators"""
        df = df.copy()
        
        # Ensure we have enough data
        if len(df) < 50:
            return df
        
        # Price-based calculations
        df['returns'] = df['close'].pct_change()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum().replace(0, 1)
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14).replace(0, 1))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Parabolic SAR (simplified)
        df['sar'] = IndicatorEngine._calculate_sar(df)
        
        # ADX (simplified)
        df['adx'] = IndicatorEngine._calculate_adx(df)
        
        # Additional indicators
        df = IndicatorEngine._calculate_additional_indicators(df)
        
        return df
    
    @staticmethod
    def _calculate_sar(df: pd.DataFrame) -> pd.Series:
        """Calculate Parabolic SAR (simplified)"""
        if len(df) < 2:
            return pd.Series(np.nan, index=df.index)
        
        sar = np.zeros(len(df))
        trend = np.ones(len(df))  # 1 for uptrend, -1 for downtrend
        af = 0.02
        ep = df['high'].iloc[0]  # extreme point
        
        sar[0] = df['low'].iloc[0]
        
        for i in range(1, len(df)):
            if trend[i-1] == 1:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                
                if df['low'].iloc[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep
                    af = 0.02
                    ep = df['low'].iloc[i]
                else:
                    trend[i] = 1
                    if df['high'].iloc[i] > ep:
                        ep = df['high'].iloc[i]
                        af = min(af + 0.02, 0.2)
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                
                if df['high'].iloc[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep
                    af = 0.02
                    ep = df['high'].iloc[i]
                else:
                    trend[i] = -1
                    if df['low'].iloc[i] < ep:
                        ep = df['low'].iloc[i]
                        af = min(af + 0.02, 0.2)
        
        return pd.Series(sar, index=df.index)
    
    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (simplified)"""
        if len(df) < period * 2:
            return pd.Series(np.zeros(len(df)), index=df.index)
        
        # True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth the values
        tr_smooth = tr.rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / tr_smooth.replace(0, 1))
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / tr_smooth.replace(0, 1))
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(0)
    
    @staticmethod
    def _calculate_additional_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional indicators"""
        # Price position
        df['price_position'] = (df['close'] - df['low'].rolling(20).min()) / \
                               (df['high'].rolling(20).max() - df['low'].rolling(20).min()).replace(0, 1)
        
        # Trend detection
        df['trend'] = np.where(df['close'] > df['ema_12'], 1, -1)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Price momentum
        df['momentum'] = df['close'].pct_change(periods=10)
        
        # Volume momentum
        df['volume_momentum'] = df['volume'].pct_change(periods=5)
        
        # Support/Resistance levels (simplified)
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        
        return df

# =========================
# SIGNAL GENERATION ENGINE
# =========================
class SignalEngine:
    def __init__(self):
        self.signals = []
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using multiple strategies"""
        df = df.copy()
        
        # Initialize signal columns
        df['signal'] = 0
        df['signal_strength'] = 0
        df['signal_type'] = 'NEUTRAL'
        
        # Strategy 1: MACD Crossover
        macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
        macd_sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
        
        # Strategy 2: RSI Oversold/Overbought
        rsi_oversold = (df['rsi'] < 30) & (df['close'] < df['bb_lower'])
        rsi_overbought = (df['rsi'] > 70) & (df['close'] > df['bb_upper'])
        
        # Strategy 3: Moving Average Crossover
        ma_cross_buy = (df['ema_12'] > df['ema_26']) & (df['ema_12'].shift() <= df['ema_26'].shift())
        ma_cross_sell = (df['ema_12'] < df['ema_26']) & (df['ema_12'].shift() >= df['ema_26'].shift())
        
        # Strategy 4: Bollinger Band Bounce
        bb_bounce_buy = (df['close'] < df['bb_lower']) & (df['close'].shift() >= df['bb_lower'].shift())
        bb_bounce_sell = (df['close'] > df['bb_upper']) & (df['close'].shift() <= df['bb_upper'].shift())
        
        # Strategy 5: Volume Spike
        volume_spike = df['volume'] > (df['volume_sma'] * 1.5)
        
        # Strategy 6: Price Breakout
        price_breakout = df['close'] > df['resistance'].shift()
        price_breakdown = df['close'] < df['support'].shift()
        
        # Combine signals
        buy_conditions = (
            (macd_buy * 0.25) + 
            (rsi_oversold * 0.20) + 
            (ma_cross_buy * 0.20) + 
            (bb_bounce_buy * 0.15) + 
            (volume_spike * 0.10) +
            (price_breakout * 0.10)
        )
        
        sell_conditions = (
            (macd_sell * 0.25) + 
            (rsi_overbought * 0.20) + 
            (ma_cross_sell * 0.20) + 
            (bb_bounce_sell * 0.15) + 
            (volume_spike * 0.10) +
            (price_breakdown * 0.10)
        )
        
        # Generate final signals
        df['signal'] = np.where(
            buy_conditions > 0.4, 1,
            np.where(sell_conditions > 0.4, -1, 0)
        )
        
        # Signal strength
        df['signal_strength'] = np.where(
            df['signal'] == 1, buy_conditions,
            np.where(df['signal'] == -1, sell_conditions, 0)
        )
        
        # Signal type
        df['signal_type'] = np.where(
            df['signal'] == 1, 'BUY',
            np.where(df['signal'] == -1, 'SELL', 'NEUTRAL')
        )
        
        # Strong signals
        strong_buy = (df['signal'] == 1) & (df['signal_strength'] > 0.6)
        strong_sell = (df['signal'] == -1) & (df['signal_strength'] > 0.6)
        
        df.loc[strong_buy, 'signal_type'] = 'STRONG_BUY'
        df.loc[strong_sell, 'signal_type'] = 'STRONG_SELL'
        
        return df
    
    def get_current_signal(self, df: pd.DataFrame) -> Dict:
        """Get the most recent signal"""
        if len(df) == 0 or 'signal_type' not in df.columns:
            return {
                'signal': 'NEUTRAL',
                'strength': 0,
                'price': 0,
                'timestamp': datetime.now(),
                'confidence': 0
            }
        
        latest = df.iloc[-1]
        
        return {
            'signal': latest['signal_type'],
            'strength': latest.get('signal_strength', 0),
            'price': latest['close'],
            'timestamp': latest.name,
            'confidence': min(abs(latest.get('rsi', 50) - 50) / 50, 1.0)
        }

# =========================
# BACKTESTING ENGINE
# =========================
class BacktestingEngine:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
    
    def run_backtest(self, df: pd.DataFrame, commission: float = 0.001) -> Dict:
        """Run backtest on historical data"""
        if 'signal' not in df.columns:
            return {}
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            current_signal = df['signal'].iloc[i]
            
            # Calculate current equity
            current_equity = capital + (position * current_price)
            equity_curve.append(current_equity)
            
            # Exit conditions (simple trailing stop)
            if position > 0:  # Long position
                stop_loss = entry_price * 0.95  # 5% stop loss
                
                if current_price <= stop_loss:
                    pnl = (current_price - entry_price) * position
                    capital += pnl - (position * entry_price * commission)
                    trades.append({
                        'entry_time': df.index[i-1],
                        'exit_time': df.index[i],
                        'side': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': (current_price / entry_price - 1) * 100
                    })
                    position = 0
                    entry_price = 0
            
            elif position < 0:  # Short position
                stop_loss = entry_price * 1.05  # 5% stop loss
                
                if current_price >= stop_loss:
                    pnl = (entry_price - current_price) * abs(position)
                    capital += pnl - (abs(position) * entry_price * commission)
                    trades.append({
                        'entry_time': df.index[i-1],
                        'exit_time': df.index[i],
                        'side': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': (entry_price / current_price - 1) * 100
                    })
                    position = 0
                    entry_price = 0
            
            # Entry conditions
            if position == 0 and current_signal != 0:
                if current_signal == 1:  # Buy signal
                    position = capital * 0.1 / current_price  # 10% of capital
                    entry_price = current_price
                elif current_signal == -1:  # Sell signal
                    position = -capital * 0.1 / current_price  # 10% short
                    entry_price = current_price
        
        # Calculate performance metrics
        results = self._calculate_performance(capital, trades, equity_curve)
        results['trades'] = trades
        
        return results
    
    def _calculate_performance(self, final_capital: float, trades: List, equity_curve: List) -> Dict:
        """Calculate performance metrics"""
        if not trades:
            return {
                'initial_capital': self.initial_capital,
                'final_capital': final_capital,
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0
            }
        
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        
        total_return_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (len(winning_trades) / len(trades_df)) * 100
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl_pct'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
        
        profit_factor = abs(winning_trades['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) \
            if len(trades_df[trades_df['pnl'] < 0]) > 0 and trades_df[trades_df['pnl'] < 0]['pnl'].sum() != 0 else 0
        
        # Sharpe Ratio
        returns = pd.Series(equity_curve).pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max.replace(0, 1)
        max_drawdown = drawdown.min() * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown
        }

# =========================
# RISK MANAGEMENT ENGINE
# =========================
class RiskManagementEngine:
    def __init__(self, max_risk_per_trade: float = 0.02):
        self.max_risk_per_trade = max_risk_per_trade
    
    def calculate_position_size(self, capital: float, entry_price: float, 
                              stop_loss: float) -> Dict:
        """Calculate optimal position size"""
        if capital <= 0 or entry_price <= 0:
            return {'size': 0, 'risk_amount': 0}
        
        risk_amount = capital * self.max_risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return {'size': 0, 'risk_amount': 0}
        
        position_size = risk_amount / stop_distance
        
        # Apply constraints
        max_position = capital * CONFIG['max_position_size'] / entry_price
        min_position = capital * CONFIG['min_position_size'] / entry_price
        
        position_size = max(min(position_size, max_position), min_position)
        
        # Calculate actual risk
        actual_risk = position_size * stop_distance
        risk_percent = (actual_risk / capital) * 100
        
        return {
            'size': position_size,
            'risk_amount': actual_risk,
            'risk_percent': risk_percent,
            'stop_distance': stop_distance,
            'stop_percent': (stop_distance / entry_price) * 100
        }

# =========================
# STREAMLIT APPLICATION
# =========================
class TitanTradingApp:
    def __init__(self):
        self.setup_page()
        self.init_components()
        self.init_session_state()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            layout="wide",
            page_title="TITAN INTRADAY PRO",
            page_icon="🚀",
            initial_sidebar_state="expanded"
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
        .metric-card {
            background: rgba(25, 25, 35, 0.8);
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #4CC9F0;
            margin-bottom: 10px;
        }
        .signal-buy {
            background: rgba(0, 200, 83, 0.2);
            border-left: 4px solid #00C853;
        }
        .signal-sell {
            background: rgba(255, 61, 0, 0.2);
            border-left: 4px solid #FF3D00;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(25, 25, 35, 0.7);
            border-radius: 4px 4px 0px 0px;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(45deg, #4361EE, #3A0CA3);
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
    
    def init_components(self):
        """Initialize all components"""
        self.db = DatabaseManager()
        self.market_data = MarketDataManager()
        self.indicators = IndicatorEngine()
        self.signal_engine = SignalEngine()
        self.backtester = BacktestingEngine()
        self.risk_manager = RiskManagementEngine()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'signals' not in st.session_state:
            st.session_state.signals = None
        if 'performance' not in st.session_state:
            st.session_state.performance = None
        if 'symbol' not in st.session_state:
            st.session_state.symbol = CONFIG['default_symbol']
        if 'timeframe' not in st.session_state:
            st.session_state.timeframe = CONFIG['default_timeframe']
        if 'candles' not in st.session_state:
            st.session_state.candles = 500
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🚀 TITAN INTRADAY PRO</h1>', unsafe_allow_html=True)
        st.caption("Advanced Trading Platform • Real-Time Analytics • AI-Powered Signals")
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/bitcoin--v1.png", width=80)
            
            # Market Configuration
            with st.expander("⚙️ Market Configuration", expanded=True):
                # Symbol input with common examples
                symbol = st.text_input(
                    "Symbol (e.g., BTCUSDT, ETHUSDT)",
                    value=CONFIG['default_symbol'],
                    help="Use Binance format without dashes or slashes"
                )
                
                # Timeframe selector
                timeframe = st.selectbox(
                    "Timeframe",
                    ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                    index=1,
                    help="Select chart timeframe"
                )
                
                # Candles slider
                candles = st.slider(
                    "Historical Candles",
                    50, 2000, 500,
                    help="Number of historical candles to fetch"
                )
                
                # Load button with better styling
                if st.button("📥 Load Market Data", use_container_width=True, type="primary"):
                    with st.spinner("Loading market data..."):
                        try:
                            self.load_market_data(symbol, timeframe, candles)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            logger.error(f"Load error: {e}")
            
            # Strategy Settings
            with st.expander("🎯 Trading Strategy", expanded=False):
                strategy = st.selectbox(
                    "Strategy Type",
                    ["Multi-Indicator", "Trend Following", "Mean Reversion", "Breakout"]
                )
                
                # Indicator toggles
                use_rsi = st.checkbox("Use RSI", True)
                use_macd = st.checkbox("Use MACD", True)
                use_bb = st.checkbox("Use Bollinger Bands", True)
                use_volume = st.checkbox("Volume Confirmation", True)
                
                # RSI thresholds
                rsi_overbought = st.slider("RSI Overbought", 65, 85, 70)
                rsi_oversold = st.slider("RSI Oversold", 15, 35, 30)
            
            # Risk Management
            with st.expander("🛡️ Risk Management", expanded=False):
                risk_per_trade = st.slider(
                    "Risk per Trade %",
                    0.1, 5.0, 2.0, 0.1,
                    help="Percentage of capital to risk per trade"
                )
                self.risk_manager.max_risk_per_trade = risk_per_trade / 100
                
                stop_loss_type = st.selectbox(
                    "Stop Loss Type",
                    ["Fixed %", "ATR-based", "Trailing"]
                )
                
                if stop_loss_type == "Fixed %":
                    stop_loss_pct = st.slider("Stop Loss %", 0.5, 10.0, 2.0, 0.5)
                elif stop_loss_type == "ATR-based":
                    atr_multiplier = st.slider("ATR Multiplier", 1.0, 5.0, 2.0, 0.5)
            
            # System Controls
            with st.expander("⚡ System", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refresh", use_container_width=True):
                        st.cache_data.clear()
                        if st.session_state.data_loaded:
                            self.load_market_data(
                                st.session_state.symbol,
                                st.session_state.timeframe,
                                st.session_state.candles
                            )
                        else:
                            st.rerun()
                
                with col2:
                    if st.button("🗑️ Clear Cache", use_container_width=True):
                        st.session_state.clear()
                        st.cache_data.clear()
                        st.rerun()
                
                # Debug toggle
                debug_mode = st.checkbox("Debug Mode", False)
                if debug_mode:
                    st.info("Debug mode enabled")
    
    def load_market_data(self, symbol: str, timeframe: str, candles: int):
        """Load and process market data"""
        try:
            # Validate symbol
            symbol = symbol.strip().upper()
            if not symbol:
                st.error("Please enter a symbol")
                return
            
            # Remove any invalid characters
            symbol = ''.join(c for c in symbol if c.isalnum())
            
            # Fetch data
            df = self.market_data.fetch_ohlcv(symbol, timeframe, candles)
            
            if df.empty:
                st.error(f"❌ No data returned for {symbol}. Please check the symbol format.")
                logger.error(f"No data for symbol: {symbol}")
                return
            
            if len(df) < 50:
                st.warning(f"⚠️ Only {len(df)} candles loaded. Some indicators may not work properly.")
            
            # Calculate indicators
            df = self.indicators.calculate_indicators(df)
            
            # Generate signals
            df = self.signal_engine.generate_signals(df)
            
            # Run backtest
            performance = self.backtester.run_backtest(df)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.signals = df[df['signal'] != 0]
            st.session_state.performance = performance
            st.session_state.symbol = symbol
            st.session_state.timeframe = timeframe
            st.session_state.candles = candles
            st.session_state.data_loaded = True
            
            # Save current signal to database
            current_signal = self.signal_engine.get_current_signal(df)
            if current_signal['signal'] != 'NEUTRAL':
                signal_data = {
                    'timestamp': current_signal['timestamp'],
                    'symbol': symbol,
                    'signal_type': current_signal['signal'],
                    'price': current_signal['price'],
                    'confidence': current_signal['confidence']
                }
                try:
                    self.db.save_signal(signal_data)
                except Exception as e:
                    logger.error(f"Failed to save signal: {e}")
            
            st.success(f"✅ Loaded {len(df)} candles for {symbol}")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            logger.error(f"Data loading error: {e}", exc_info=True)
    
    def render_dashboard(self):
        """Render main dashboard"""
        if not st.session_state.data_loaded or st.session_state.df is None:
            self.render_welcome_screen()
            return
        
        df = st.session_state.df
        
        # Performance Metrics
        self.render_performance_metrics()
        
        # Current Market Status
        self.render_market_overview(df)
        
        # Main Chart Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Price Chart", 
            "📈 Indicators", 
            "📉 Backtest Results", 
            "📋 Signal History"
        ])
        
        with tab1:
            self.render_price_chart(df)
        
        with tab2:
            self.render_indicator_panels(df)
        
        with tab3:
            self.render_backtest_results()
        
        with tab4:
            self.render_signal_history()
    
    def render_welcome_screen(self):
        """Render welcome screen when no data loaded"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 50px 0;'>
                <h2>🚀 Welcome to TITAN INTRADAY PRO</h2>
                <p style='font-size: 18px; color: #888; margin: 20px 0;'>
                    Advanced Trading Platform with Real-Time Analytics
                </p>
                <div style='background: rgba(25, 25, 35, 0.8); padding: 30px; border-radius: 10px; margin: 20px 0;'>
                    <h3>📊 Get Started:</h3>
                    <ol style='text-align: left; margin-left: 20px;'>
                        <li>Enter a symbol (e.g., BTCUSDT, ETHUSDT)</li>
                        <li>Select a timeframe</li>
                        <li>Choose number of historical candles</li>
                        <li>Click "Load Market Data"</li>
                    </ol>
                </div>
                <p style='color: #666; margin-top: 30px;'>
                    Supported symbols: BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, etc.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_performance_metrics(self):
        """Render performance metrics"""
        st.subheader("📊 Performance Dashboard")
        
        if st.session_state.performance:
            perf = st.session_state.performance
            
            cols = st.columns(6)
            metrics = [
                ("💰 Total Return", f"{perf.get('total_return_pct', 0):.2f}%"),
                ("📈 Win Rate", f"{perf.get('win_rate', 0):.1f}%"),
                ("⚖️ Profit Factor", f"{perf.get('profit_factor', 0):.2f}"),
                ("📉 Max Drawdown", f"{perf.get('max_drawdown_pct', 0):.2f}%"),
                ("🎯 Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}"),
                ("🔢 Total Trades", f"{perf.get('total_trades', 0)}")
            ]
            
            for col, (label, value) in zip(cols, metrics):
                with col:
                    st.metric(label, value)
    
    def render_market_overview(self, df: pd.DataFrame):
        """Render market overview metrics"""
        st.subheader("📈 Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
            price_change = ((current_price - prev_price) / prev_price) * 100
            st.metric(
                "Current Price",
                f"${current_price:,.2f}",
                f"{price_change:+.2f}%"
            )
        
        with col2:
            current_signal = self.signal_engine.get_current_signal(df)
            signal_display = current_signal['signal']
            
            if "BUY" in signal_display:
                st.success(f"**{signal_display}**")
            elif "SELL" in signal_display:
                st.error(f"**{signal_display}**")
            else:
                st.info(f"**{signal_display}**")
        
        with col3:
            volume_ratio = df['volume_ratio'].iloc[-1]
            st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
        
        with col4:
            rsi_value = df['rsi'].iloc[-1]
            rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
            st.metric("RSI", f"{rsi_value:.1f}", rsi_status)
    
    def render_price_chart(self, df: pd.DataFrame):
        """Render interactive price chart"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price & Indicators', 'Volume', 'RSI')
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
        
        # Moving averages
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
        signals = st.session_state.signals
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
                        name='Sell Signal',
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
    
    def render_indicator_panels(self, df: pd.DataFrame):
        """Render indicator panels"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Trend Indicators")
            
            trend_data = {
                "MACD": f"{df['macd'].iloc[-1]:.4f}",
                "MACD Signal": f"{df['macd_signal'].iloc[-1]:.4f}",
                "MACD Hist": f"{df['macd_hist'].iloc[-1]:.4f}",
                "ADX": f"{df['adx'].iloc[-1]:.2f}",
                "Trend": "Bullish" if df['trend'].iloc[-1] > 0 else "Bearish"
            }
            
            for indicator, value in trend_data.items():
                st.metric(indicator, value)
        
        with col2:
            st.subheader("📈 Momentum Indicators")
            
            momentum_data = {
                "RSI": f"{df['rsi'].iloc[-1]:.2f}",
                "Stoch %K": f"{df['stoch_k'].iloc[-1]:.2f}",
                "Stoch %D": f"{df['stoch_d'].iloc[-1]:.2f}",
                "Price Momentum": f"{df['momentum'].iloc[-1]:.2%}"
            }
            
            for indicator, value in momentum_data.items():
                st.metric(indicator, value)
        
        # Volatility & Volume
        st.subheader("📉 Volatility & Volume")
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric("ATR", f"{df['atr'].iloc[-1]:.2f}")
            st.metric("Volatility", f"{df['volatility'].iloc[-1]:.2%}")
        
        with col4:
            bb_position = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / \
                         (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
            st.metric("BB Position", f"{bb_position:.2%}")
            st.metric("Volume Momentum", f"{df['volume_momentum'].iloc[-1]:.2%}")
    
    def render_backtest_results(self):
        """Render backtest results"""
        if not st.session_state.performance:
            st.info("No backtest results available")
            return
        
        perf = st.session_state.performance
        
        # Trade Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Trade Statistics")
            
            trade_stats = {
                "Total Trades": perf.get('total_trades', 0),
                "Winning Trades": perf.get('winning_trades', 0),
                "Win Rate": f"{perf.get('win_rate', 0):.1f}%",
                "Avg Win %": f"{perf.get('avg_win_pct', 0):.2f}%",
                "Avg Loss %": f"{perf.get('avg_loss_pct', 0):.2f}%"
            }
            
            for key, value in trade_stats.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("📈 Performance Metrics")
            
            perf_metrics = {
                "Total Return": f"{perf.get('total_return_pct', 0):.2f}%",
                "Profit Factor": f"{perf.get('profit_factor', 0):.2f}",
                "Sharpe Ratio": f"{perf.get('sharpe_ratio', 0):.2f}",
                "Max Drawdown": f"{perf.get('max_drawdown_pct', 0):.2f}%"
            }
            
            for key, value in perf_metrics.items():
                st.metric(key, value)
        
        # Recent trades if available
        if 'trades' in perf and perf['trades']:
            st.subheader("📋 Recent Trades")
            trades_df = pd.DataFrame(perf['trades'])
            
            # Format the dataframe
            display_df = trades_df.tail(10).copy()
            if not display_df.empty:
                display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(
                    display_df.style.format({
                        'entry_price': '${:.2f}',
                        'exit_price': '${:.2f}',
                        'pnl': '${:.2f}',
                        'pnl_pct': '{:.2f}%'
                    }),
                    use_container_width=True
                )
    
    def render_signal_history(self):
        """Render signal history"""
        # Current signal
        if st.session_state.df is not None:
            current_signal = self.signal_engine.get_current_signal(st.session_state.df)
            
            st.subheader("🎯 Current Signal")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                signal_display = current_signal['signal']
                if "BUY" in signal_display:
                    st.success(f"**{signal_display}**")
                elif "SELL" in signal_display:
                    st.error(f"**{signal_display}**")
                else:
                    st.info(f"**{signal_display}**")
            
            with col2:
                st.metric("Price", f"${current_signal['price']:,.2f}")
            
            with col3:
                st.metric("Confidence", f"{current_signal['confidence']:.0%}")
        
        # Signal History
        st.subheader("📜 Signal History")
        
        try:
            recent_signals = self.db.get_recent_signals(st.session_state.symbol, 20)
            
            if recent_signals:
                signals_df = pd.DataFrame(recent_signals)
                signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Color coding function
                def color_row(row):
                    if "BUY" in row['signal_type']:
                        return ['background-color: rgba(0, 200, 83, 0.1)'] * len(row)
                    elif "SELL" in row['signal_type']:
                        return ['background-color: rgba(255, 61, 0, 0.1)'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    signals_df.style.apply(color_row, axis=1).format({
                        'price': '${:.2f}',
                        'confidence': '{:.0%}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No signal history available")
        
        except Exception as e:
            st.warning(f"Could not load signal history: {e}")
    
    def render_footer(self):
        """Render application footer"""
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "✅ Loaded" if st.session_state.data_loaded else "⏳ Waiting"
            st.caption(f"{status} | {st.session_state.symbol}")
        
        with col2:
            if st.session_state.df is not None:
                last_update = st.session_state.df.index[-1].strftime('%Y-%m-%d %H:%M')
                st.caption(f"📅 Last Data: {last_update}")
        
        with col3:
            st.caption("⚠️ Educational Use Only • Not Financial Advice")
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        self.render_dashboard()
        self.render_footer()

# =========================
# RUN THE APPLICATION
# =========================
if __name__ == "__main__":
    # Initialize and run the app
    app = TitanTradingApp()
    app.run()
