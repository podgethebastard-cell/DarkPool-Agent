import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import threading
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
warnings.filterwarnings('ignore')

# =========================
# FALLBACK IMPORTS WITH GRACEFUL DEGRADATION
# =========================
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("yfinance not installed. Using alternative data source.")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    st.warning("pandas_ta not installed. Some indicators may be limited.")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not installed. ML features disabled.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Try to import websocket but provide fallback
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    st.info("WebSocket not available. Real-time features will use polling.")

# =========================
# CONFIGURATION
# =========================
CONFIG = {
    'app_name': 'TITAN INTRADAY PRO',
    'cache_ttl': 30,
    'max_candles': 2000,
    'db_path': 'titan_trading.db',
    'default_symbol': 'BTC-USD',
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
# ENUMS & DATA CLASSES
# =========================
from enum import Enum

class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

# =========================
# DATABASE MANAGER (SIMPLIFIED)
# =========================
class DatabaseManager:
    def __init__(self, db_path: str = CONFIG['db_path']):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Signals table
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
        
        # Trades table
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
# MARKET DATA MANAGER (WITH YFINANCE FALLBACK)
# =========================
class MarketDataManager:
    def __init__(self):
        self.cache = {}
    
    def fetch_ohlcv_yfinance(self, symbol: str, interval: str, period: str = "60d") -> pd.DataFrame:
        """Fetch OHLCV data using yfinance"""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not available")
        
        # Map intervals
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d"
        }
        
        yf_interval = interval_map.get(interval, "5m")
        
        # Adjust period based on interval
        if yf_interval in ["1m", "5m"]:
            period = "7d"
        elif yf_interval in ["15m", "30m"]:
            period = "60d"
        elif yf_interval == "1h":
            period = "730d"
        else:
            period = "max"
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=yf_interval)
            
            if df.empty:
                # Try alternative symbol format
                if "-" in symbol:
                    alt_symbol = symbol.replace("-", "")
                else:
                    alt_symbol = f"{symbol}-USD"
                
                ticker = yf.Ticker(alt_symbol)
                df = ticker.history(period=period, interval=yf_interval)
            
            # Rename columns to standard format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            df.index.name = 'timestamp'
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data with yfinance: {e}")
            return pd.DataFrame()
    
    def fetch_ohlcv_binance(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data from Binance API"""
        try:
            # Convert symbol format
            binance_symbol = symbol.replace("-", "")
            if not binance_symbol.endswith("USDT"):
                binance_symbol += "USDT"
            
            # Map intervals
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "4h": "4h", "1d": "1d"
            }
            
            binance_interval = interval_map.get(interval, "5m")
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": binance_symbol,
                "interval": binance_interval,
                "limit": limit
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
                df[col] = df[col].astype(float)
            
            return df[["open", "high", "low", "close", "volume"]]
        
        except Exception as e:
            logger.error(f"Error fetching data from Binance: {e}")
            # Fallback to yfinance
            if YFINANCE_AVAILABLE:
                return self.fetch_ohlcv_yfinance(symbol, interval)
            return pd.DataFrame()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Main method to fetch OHLCV data with fallbacks"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        # Check cache
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < CONFIG['cache_ttl']:
                return data.copy()
        
        # Try Binance first
        df = self.fetch_ohlcv_binance(symbol, timeframe, limit)
        
        # If empty, try yfinance
        if df.empty and YFINANCE_AVAILABLE:
            df = self.fetch_ohlcv_yfinance(symbol, timeframe)
        
        # Cache the result
        if not df.empty:
            self.cache[cache_key] = (datetime.now(), df.copy())
        
        return df

# =========================
# TECHNICAL INDICATOR ENGINE
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
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # VWAP (daily)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Parabolic SAR (simplified)
        df['sar'] = IndicatorEngine._calculate_sar(df)
        
        # ADX (simplified)
        df['adx'] = IndicatorEngine._calculate_adx(df)
        
        # Custom indicators
        df = IndicatorEngine._calculate_custom_indicators(df)
        
        return df
    
    @staticmethod
    def _calculate_sar(df: pd.DataFrame) -> pd.Series:
        """Calculate Parabolic SAR (simplified)"""
        sar = pd.Series(np.nan, index=df.index)
        
        if len(df) < 2:
            return sar
        
        # Start values
        sar.iloc[0] = df['low'].iloc[0]
        trend = 1  # 1 for uptrend, -1 for downtrend
        af = 0.02  # acceleration factor
        ep = df['high'].iloc[0]  # extreme point
        
        for i in range(1, len(df)):
            if trend == 1:
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                
                if df['low'].iloc[i] < sar.iloc[i]:
                    trend = -1
                    sar.iloc[i] = ep
                    af = 0.02
                    ep = df['low'].iloc[i]
                else:
                    if df['high'].iloc[i] > ep:
                        ep = df['high'].iloc[i]
                        af = min(af + 0.02, 0.2)
            
            else:  # downtrend
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                
                if df['high'].iloc[i] > sar.iloc[i]:
                    trend = 1
                    sar.iloc[i] = ep
                    af = 0.02
                    ep = df['high'].iloc[i]
                else:
                    if df['low'].iloc[i] < ep:
                        ep = df['low'].iloc[i]
                        af = min(af + 0.02, 0.2)
        
        return sar
    
    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (simplified)"""
        if len(df) < period * 2:
            return pd.Series(np.nan, index=df.index)
        
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
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / tr_smooth)
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def _calculate_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom proprietary indicators"""
        # Supertrend (simplified)
        df['hl2'] = (df['high'] + df['low']) / 2
        df['basic_upper'] = df['hl2'] + (df['atr'] * 2)
        df['basic_lower'] = df['hl2'] - (df['atr'] * 2)
        
        # Trend detection
        df['trend'] = np.where(df['close'] > df['ema_12'], 1, -1)
        
        # Volatility ratio
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Price position in Bollinger Band
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI momentum
        df['rsi_momentum'] = df['rsi'].diff(5)
        
        return df
    
    @staticmethod
    def detect_divergences(df: pd.DataFrame) -> pd.DataFrame:
        """Detect bullish and bearish divergences"""
        df = df.copy()
        
        # Find peaks and troughs
        df['peak'] = (df['close'].shift(1) < df['close']) & (df['close'].shift(-1) < df['close'])
        df['trough'] = (df['close'].shift(1) > df['close']) & (df['close'].shift(-1) > df['close'])
        
        # RSI divergences
        df['rsi_peak'] = (df['rsi'].shift(1) < df['rsi']) & (df['rsi'].shift(-1) < df['rsi'])
        df['rsi_trough'] = (df['rsi'].shift(1) > df['rsi']) & (df['rsi'].shift(-1) > df['rsi'])
        
        # Detect divergences
        df['bearish_divergence'] = False
        df['bullish_divergence'] = False
        
        for i in range(20, len(df)):
            # Bearish divergence (price makes higher high, RSI makes lower high)
            if df['peak'].iloc[i] and df['rsi_peak'].iloc[i-10:i].any():
                df.loc[df.index[i], 'bearish_divergence'] = True
            
            # Bullish divergence (price makes lower low, RSI makes higher low)
            if df['trough'].iloc[i] and df['rsi_trough'].iloc[i-10:i].any():
                df.loc[df.index[i], 'bullish_divergence'] = True
        
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
        
        # Strategy 1: Trend Following (MACD + EMA)
        macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
        macd_sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
        
        # Strategy 2: Mean Reversion (RSI + Bollinger)
        rsi_oversold = (df['rsi'] < 30) & (df['close'] < df['bb_lower'])
        rsi_overbought = (df['rsi'] > 70) & (df['close'] > df['bb_upper'])
        
        # Strategy 3: Breakout (ATR based)
        atr_breakout = df['close'] > (df['high'].rolling(20).max().shift() + df['atr'])
        atr_breakdown = df['close'] < (df['low'].rolling(20).min().shift() - df['atr'])
        
        # Strategy 4: Moving Average Crossover
        ma_cross_buy = (df['ema_12'] > df['ema_26']) & (df['ema_12'].shift() <= df['ema_26'].shift())
        ma_cross_sell = (df['ema_12'] < df['ema_26']) & (df['ema_12'].shift() >= df['ema_26'].shift())
        
        # Strategy 5: Volume Confirmation
        volume_spike = df['volume'] > (df['volume'].rolling(20).mean() * 1.5)
        
        # Combine signals with weights
        buy_signals = (
            (macd_buy * 0.3) + 
            (rsi_oversold * 0.2) + 
            (atr_breakout * 0.2) + 
            (ma_cross_buy * 0.2) + 
            (volume_spike * 0.1)
        )
        
        sell_signals = (
            (macd_sell * 0.3) + 
            (rsi_overbought * 0.2) + 
            (atr_breakdown * 0.2) + 
            (ma_cross_sell * 0.2) + 
            (volume_spike * 0.1)
        )
        
        # Apply divergences as filters
        df = IndicatorEngine.detect_divergences(df)
        buy_signals = buy_signals & ~df['bearish_divergence']
        sell_signals = sell_signals & ~df['bullish_divergence']
        
        # Set signals
        df.loc[buy_signals, 'signal'] = 1
        df.loc[sell_signals, 'signal'] = -1
        
        # Calculate signal strength
        df['signal_strength'] = abs(buy_signals.astype(int) * 0.5 + sell_signals.astype(int) * -0.5)
        
        # Determine signal type
        df['signal_type'] = np.where(
            df['signal'] == 1, 'BUY',
            np.where(df['signal'] == -1, 'SELL', 'NEUTRAL')
        )
        
        # For strong signals
        strong_buy = buy_signals & (df['rsi'] < 40) & (df['volume_ratio'] > 1.5)
        strong_sell = sell_signals & (df['rsi'] > 60) & (df['volume_ratio'] > 1.5)
        
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
                'timestamp': datetime.now()
            }
        
        latest = df.iloc[-1]
        
        return {
            'signal': latest['signal_type'],
            'strength': latest.get('signal_strength', 0),
            'price': latest['close'],
            'timestamp': latest.name if hasattr(latest, 'name') else datetime.now(),
            'confidence': min(abs(latest.get('rsi', 50) - 50) / 50, 1.0)
        }

# =========================
# BACKTESTING ENGINE
# =========================
class BacktestingEngine:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.trades = []
    
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
            
            # Exit conditions
            if position > 0:  # Long position
                # Simple trailing stop (10% from entry)
                stop_loss = entry_price * 0.9
                
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
            
            elif position < 0:  # Short position
                stop_loss = entry_price * 1.1
                
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
            
            # Entry conditions
            if position == 0 and current_signal != 0:
                if current_signal == 1:  # Buy signal
                    position = capital * 0.1 / current_price  # 10% of capital
                    entry_price = current_price
                
                elif current_signal == -1:  # Sell signal
                    position = -capital * 0.1 / current_price  # 10% short
                    entry_price = current_price
        
        # Calculate performance metrics
        if trades:
            trades_df = pd.DataFrame(trades)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            
            total_return_pct = ((capital - self.initial_capital) / self.initial_capital) * 100
            win_rate = (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
            avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl_pct'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            
            profit_factor = abs(winning_trades['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
            
            # Sharpe Ratio (simplified)
            returns = pd.Series(equity_curve).pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            else:
                sharpe_ratio = 0
            
            # Maximum Drawdown
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            return {
                'initial_capital': self.initial_capital,
                'final_capital': capital,
                'total_return_pct': total_return_pct,
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'win_rate': win_rate,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'trades': trades_df.to_dict('records')
            }
        
        return {}

# =========================
# RISK MANAGEMENT ENGINE
# =========================
class RiskManagementEngine:
    def __init__(self, max_risk_per_trade: float = 0.02):
        self.max_risk_per_trade = max_risk_per_trade
    
    def calculate_position_size(self, capital: float, entry_price: float, 
                              stop_loss: float) -> Dict:
        """Calculate optimal position size"""
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
# MAIN STREAMLIT APPLICATION
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
                symbol = st.text_input("Symbol", CONFIG['default_symbol'])
                timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=1)
                candles = st.slider("Historical Candles", 100, 2000, 500)
                
                if st.button("📥 Load Market Data", use_container_width=True, type="primary"):
                    with st.spinner("Loading data..."):
                        self.load_market_data(symbol, timeframe, candles)
            
            # Strategy Settings
            with st.expander("🎯 Trading Strategy", expanded=False):
                strategy_type = st.selectbox(
                    "Strategy Type",
                    ["Trend Following", "Mean Reversion", "Breakout", "Multi-Indicator"]
                )
                
                use_rsi = st.checkbox("Use RSI", True)
                use_macd = st.checkbox("Use MACD", True)
                use_volume = st.checkbox("Volume Confirmation", True)
                
                rsi_overbought = st.slider("RSI Overbought", 70, 90, 70)
                rsi_oversold = st.slider("RSI Oversold", 10, 30, 30)
            
            # Risk Management
            with st.expander("🛡️ Risk Management", expanded=False):
                risk_per_trade = st.slider("Risk per Trade %", 0.1, 5.0, 1.0, 0.1)
                self.risk_manager.max_risk_per_trade = risk_per_trade / 100
                
                stop_loss_type = st.selectbox("Stop Loss", ["ATR-based", "Fixed %", "Trailing"])
                
                if stop_loss_type == "ATR-based":
                    atr_multiplier = st.slider("ATR Multiplier", 1.0, 5.0, 2.0, 0.5)
                elif stop_loss_type == "Fixed %":
                    stop_loss_pct = st.slider("Stop Loss %", 0.5, 10.0, 2.0, 0.5)
            
            # System Controls
            with st.expander("⚡ System", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refresh", use_container_width=True):
                        st.cache_data.clear()
                        st.rerun()
                with col2:
                    if st.button("🗑️ Clear", use_container_width=True):
                        st.session_state.clear()
                        st.rerun()
    
    def load_market_data(self, symbol: str, timeframe: str, candles: int):
        """Load and process market data"""
        try:
            # Fetch data
            df = self.market_data.fetch_ohlcv(symbol, timeframe, candles)
            
            if df.empty:
                st.error("❌ Failed to load market data. Check symbol and connection.")
                return
            
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
            st.session_state.data_loaded = True
            
            # Save to database
            current_signal = self.signal_engine.get_current_signal(df)
            if current_signal['signal'] != 'NEUTRAL':
                signal_data = {
                    'timestamp': current_signal['timestamp'],
                    'symbol': symbol,
                    'signal_type': current_signal['signal'],
                    'price': current_signal['price'],
                    'confidence': current_signal['confidence']
                }
                self.db.save_signal(signal_data)
            
            st.success(f"✅ Loaded {len(df)} candles for {symbol}")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            logger.error(f"Data loading error: {e}")
    
    def render_dashboard(self):
        """Render main dashboard"""
        if not st.session_state.data_loaded or st.session_state.df is None:
            st.info("👈 Load market data from the sidebar to begin")
            return
        
        df = st.session_state.df
        signals = st.session_state.signals
        
        # Performance Metrics
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
        
        # Current Market Status
        st.subheader("📈 Market Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df['close'].iloc[-1]
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            st.metric("Current Price", f"${current_price:,.2f}", f"{price_change:+.2f}%")
        
        with col2:
            current_signal = self.signal_engine.get_current_signal(df)
            signal_color = "green" if "BUY" in current_signal['signal'] else "red" if "SELL" in current_signal['signal'] else "gray"
            st.metric("Current Signal", current_signal['signal'])
        
        with col3:
            volume_ratio = df['volume_ratio'].iloc[-1]
            st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
        
        with col4:
            rsi_value = df['rsi'].iloc[-1]
            rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
            st.metric("RSI", f"{rsi_value:.1f}", rsi_status)
        
        # Main Chart Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Price Chart", 
            "📈 Indicators", 
            "📉 Backtest", 
            "📋 Signals"
        ])
        
        with tab1:
            self.render_price_chart(df, signals)
        
        with tab2:
            self.render_indicator_panels(df)
        
        with tab3:
            self.render_backtest_results()
        
        with tab4:
            self.render_signal_history()
    
    def render_price_chart(self, df: pd.DataFrame, signals: pd.DataFrame):
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
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_50'], line=dict(color='blue', width=1), name='SMA 50'),
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
        
        # Buy/Sell signals
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
        
        # Volume MA
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
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
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
            st.subheader("Trend Indicators")
            
            trend_indicators = {
                "MACD": df['macd'].iloc[-1],
                "MACD Signal": df['macd_signal'].iloc[-1],
                "MACD Hist": df['macd_hist'].iloc[-1],
                "ADX": df['adx'].iloc[-1] if 'adx' in df.columns else 0,
                "Trend": "Bullish" if df['trend'].iloc[-1] == 1 else "Bearish"
            }
            
            for indicator, value in trend_indicators.items():
                st.metric(indicator, f"{value:.4f}" if isinstance(value, float) else value)
        
        with col2:
            st.subheader("Momentum Indicators")
            
            momentum_indicators = {
                "RSI": df['rsi'].iloc[-1],
                "Stoch %K": df['stoch_k'].iloc[-1],
                "Stoch %D": df['stoch_d'].iloc[-1],
                "RSI Momentum": df['rsi_momentum'].iloc[-1] if 'rsi_momentum' in df.columns else 0
            }
            
            for indicator, value in momentum_indicators.items():
                st.metric(indicator, f"{value:.2f}")
        
        # Volatility Indicators
        st.subheader("Volatility & Volume")
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric("ATR", f"{df['atr'].iloc[-1]:.2f}")
            st.metric("Volatility", f"{df['volatility'].iloc[-1]:.2%}" if 'volatility' in df.columns else "N/A")
        
        with col4:
            bb_position = df['bb_position'].iloc[-1] if 'bb_position' in df.columns else 0.5
            st.metric("BB Position", f"{bb_position:.2%}")
            st.metric("Volume Ratio", f"{df['volume_ratio'].iloc[-1]:.2f}x")
    
    def render_backtest_results(self):
        """Render backtest results"""
        if not st.session_state.performance:
            st.info("No backtest results available")
            return
        
        perf = st.session_state.performance
        
        # Equity Curve
        if 'trades' in perf and perf['trades']:
            trades_df = pd.DataFrame(perf['trades'])
            
            if 'pnl' in trades_df.columns:
                fig = go.Figure()
                
                # Cumulative P&L
                fig.add_trace(go.Scatter(
                    x=trades_df['exit_time'],
                    y=trades_df['pnl'].cumsum(),
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='green', width=2)
                ))
                
                # Individual trades
                winning_trades = trades_df[trades_df['pnl'] > 0]
                losing_trades = trades_df[trades_df['pnl'] < 0]
                
                if not winning_trades.empty:
                    fig.add_trace(go.Scatter(
                        x=winning_trades['exit_time'],
                        y=winning_trades['pnl'].cumsum(),
                        mode='markers',
                        name='Winning Trades',
                        marker=dict(color='green', size=8)
                    ))
                
                if not losing_trades.empty:
                    fig.add_trace(go.Scatter(
                        x=losing_trades['exit_time'],
                        y=losing_trades['pnl'].cumsum(),
                        mode='markers',
                        name='Losing Trades',
                        marker=dict(color='red', size=8)
                    ))
                
                fig.update_layout(
                    title='Equity Curve',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Trade Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trade Statistics")
            
            trade_stats = {
                "Total Trades": perf.get('total_trades', 0),
                "Winning Trades": perf.get('winning_trades', 0),
                "Losing Trades": perf.get('total_trades', 0) - perf.get('winning_trades', 0),
                "Win Rate": f"{perf.get('win_rate', 0):.1f}%",
                "Avg Win %": f"{perf.get('avg_win_pct', 0):.2f}%",
                "Avg Loss %": f"{perf.get('avg_loss_pct', 0):.2f}%"
            }
            
            for key, value in trade_stats.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("Performance Metrics")
            
            perf_metrics = {
                "Total Return": f"{perf.get('total_return_pct', 0):.2f}%",
                "Profit Factor": f"{perf.get('profit_factor', 0):.2f}",
                "Sharpe Ratio": f"{perf.get('sharpe_ratio', 0):.2f}",
                "Max Drawdown": f"{perf.get('max_drawdown_pct', 0):.2f}%"
            }
            
            for key, value in perf_metrics.items():
                st.metric(key, value)
        
        # Recent Trades Table
        if 'trades' in perf and perf['trades']:
            st.subheader("Recent Trades")
            trades_df = pd.DataFrame(perf['trades'])
            st.dataframe(
                trades_df.tail(10).style.format({
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
            
            st.subheader("Current Signal")
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
        
        # Signal History from Database
        st.subheader("Signal History")
        
        try:
            recent_signals = self.db.get_recent_signals(st.session_state.symbol, 20)
            
            if recent_signals:
                signals_df = pd.DataFrame(recent_signals)
                signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
                
                # Color code signals
                def color_signal(val):
                    if "BUY" in val:
                        return 'color: green'
                    elif "SELL" in val:
                        return 'color: red'
                    return 'color: gray'
                
                st.dataframe(
                    signals_df.style.applymap(color_signal, subset=['signal_type']).format({
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
            st.caption(f"📊 Data: {st.session_state.symbol if st.session_state.data_loaded else 'Not loaded'}")
        
        with col2:
            if st.session_state.df is not None:
                last_update = st.session_state.df.index[-1].strftime('%Y-%m-%d %H:%M')
                st.caption(f"⏰ Last Data: {last_update}")
        
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
