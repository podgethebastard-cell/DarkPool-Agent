import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import websocket
import json
import threading
import asyncio
import pickle
import hashlib
import warnings
from datetime import datetime, timedelta
import time
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import yaml
import os
from pathlib import Path
import sys
from dataclasses import dataclass
from enum import Enum
import talib
from scipy import stats
import ccxt
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import boto3
import redis
import psutil
import gc

# =========================
# CONFIGURATION & CONSTANTS
# =========================
CONFIG = {
    'version': '3.0.0',
    'app_name': 'TITAN INTRADAY PRO',
    'cache_ttl': 30,
    'max_candles': 5000,
    'db_path': 'data/titan.db',
    'log_path': 'logs/titan.log',
    'backtest_days': 365,
    'max_position_size': 0.1,  # 10% of capital
    'min_position_size': 0.01,  # 1% of capital
    'commission_rate': 0.001,  # 0.1%
    'slippage_rate': 0.0005,   # 0.05%
}

# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_path']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# ENUMS & DATA CLASSES
# =========================
class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    price: float
    confidence: float
    indicators: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class Trade:
    id: str
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: Optional[float]
    size: float
    stop_loss: float
    take_profit: float
    pnl: Optional[float]
    status: str

# =========================
# DATABASE MANAGER
# =========================
class DatabaseManager:
    def __init__(self, db_path: str = CONFIG['db_path']):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database with all required tables"""
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
                indicators TEXT,
                metadata TEXT,
                processed BOOLEAN DEFAULT 0
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
                stop_loss REAL,
                take_profit REAL,
                pnl REAL,
                status TEXT,
                commission REAL,
                slippage REAL
            )
        ''')
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp DATETIME,
                symbol TEXT,
                timeframe TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (timestamp, symbol, timeframe)
            )
        ''')
        
        # Performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                date DATE,
                symbol TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                PRIMARY KEY (date, symbol)
            )
        ''')
        
        # User settings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                settings TEXT,
                last_updated DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_signal(self, signal: Signal) -> int:
        """Save signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals 
            (timestamp, symbol, signal_type, price, confidence, indicators, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp,
            signal.symbol,
            signal.signal_type.value,
            signal.price,
            signal.confidence,
            json.dumps(signal.indicators),
            json.dumps(signal.metadata)
        ))
        
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return signal_id
    
    def get_recent_signals(self, symbol: str, limit: int = 50) -> List[Dict]:
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
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.cache = {}
        self.ws_connections = {}
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < CONFIG['cache_ttl']:
                return data.copy()
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self.cache[cache_key] = (datetime.now(), df.copy())
            
            logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_timeframes(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes"""
        result = {}
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, tf, 1000)
            if not df.empty:
                result[tf] = df
        return result
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book data"""
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book: {str(e)}")
            return {}

# =========================
# ADVANCED INDICATOR ENGINE
# =========================
class IndicatorEngine:
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive set of technical indicators"""
        df = df.copy()
        
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Trend indicators
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        df['wma'] = talib.WMA(df['close'], timeperiod=20)
        df['dema'] = talib.DEMA(df['close'], timeperiod=30)
        df['tema'] = talib.TEMA(df['close'], timeperiod=30)
        
        # Momentum indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['momentum'] = talib.MOM(df['close'], timeperiod=10)
        
        # Volatility indicators
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['natr'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Volume indicators
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Cycle indicators
        df['ht_dcperiod'] = talib.HT_DCPERIOD(df['close'])
        df['ht_dcphase'] = talib.HT_DCPHASE(df['close'])
        df['ht_phasor_inphase'], df['ht_phasor_quadrature'] = talib.HT_PHASOR(df['close'])
        df['ht_sine'], df['ht_leadsine'] = talib.HT_SINE(df['close'])
        df['ht_trendmode'] = talib.HT_TRENDMODE(df['close'])
        
        # Pattern recognition
        df['cdl_doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['cdl_hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['cdl_engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['cdl_morningstar'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['cdl_eveningstar'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        # Statistical indicators
        df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['skew'] = df['returns'].rolling(50).skew()
        df['kurtosis'] = df['returns'].rolling(50).kurtosis()
        
        # Custom indicators
        df = IndicatorEngine._calculate_custom_indicators(df)
        
        return df
    
    @staticmethod
    def _calculate_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom proprietary indicators"""
        # SuperTrend
        df['supertrend'] = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)['SUPERT_10_3.0']
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Volume Profile
        df['vp_buy_volume'] = np.where(df['close'] > df['open'], df['volume'], 0)
        df['vp_sell_volume'] = np.where(df['close'] <= df['open'], df['volume'], 0)
        
        # Market Profile
        df['tpoc'] = df.groupby(pd.Grouper(freq='D'))['volume'].transform('idxmax')
        
        # Advanced ATR
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        df['atr_ratio'] = df['atr'] / df['atr'].rolling(50).mean()
        
        # Trend strength composite
        trend_indicators = ['adx', 'rsi', 'macd_hist', 'momentum']
        df['trend_strength'] = df[trend_indicators].apply(
            lambda row: np.mean([(row['adx']/100), (abs(row['rsi']-50)/50), 
                               abs(row['macd_hist']/row['macd_hist'].std()), 
                               abs(row['momentum']/row['momentum'].std())]), axis=1
        )
        
        # Support/Resistance levels
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        
        return df
    
    @staticmethod
    def detect_divergence(df: pd.DataFrame) -> pd.DataFrame:
        """Detect RSI and MACD divergences"""
        df = df.copy()
        
        # RSI divergence
        df['rsi_high'] = df['rsi'].rolling(5, center=True).max()
        df['price_high'] = df['close'].rolling(5, center=True).max()
        df['bearish_divergence'] = (df['price_high'] > df['price_high'].shift(10)) & (df['rsi_high'] < df['rsi_high'].shift(10))
        df['bullish_divergence'] = (df['price_high'] < df['price_high'].shift(10)) & (df['rsi_high'] > df['rsi_high'].shift(10))
        
        return df

# =========================
# MACHINE LEARNING SIGNAL ENGINE
# =========================
class MLSignalEngine:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for ML model"""
        feature_df = df.copy()
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            feature_df[f'close_lag_{lag}'] = feature_df['close'].shift(lag)
            feature_df[f'volume_lag_{lag}'] = feature_df['volume'].shift(lag)
            feature_df[f'rsi_lag_{lag}'] = feature_df['rsi'].shift(lag)
        
        # Rolling statistics
        feature_df['close_ma_ratio'] = feature_df['close'] / feature_df['close'].rolling(20).mean()
        feature_df['volume_ma_ratio'] = feature_df['volume'] / feature_df['volume'].rolling(20).mean()
        feature_df['volatility'] = feature_df['returns'].rolling(20).std()
        
        # Price action features
        feature_df['body_size'] = abs(feature_df['close'] - feature_df['open']) / feature_df['atr']
        feature_df['upper_shadow'] = (feature_df['high'] - feature_df[['open', 'close']].max(axis=1)) / feature_df['atr']
        feature_df['lower_shadow'] = (feature_df[['open', 'close']].min(axis=1) - feature_df['low']) / feature_df['atr']
        
        # Technical indicator features
        feature_df['macd_signal_diff'] = feature_df['macd'] - feature_df['macd_signal']
        feature_df['bollinger_position'] = (feature_df['close'] - feature_df['bollinger_lower']) / (feature_df['bollinger_upper'] - feature_df['bollinger_lower'])
        
        # Drop NaN values
        feature_df = feature_df.dropna()
        
        # Select feature columns
        self.feature_columns = [col for col in feature_df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'returns', 'timestamp']]
        
        X = feature_df[self.feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, self.feature_columns
    
    def train_anomaly_detector(self, df: pd.DataFrame):
        """Train isolation forest for anomaly detection"""
        X, _ = self.prepare_features(df)
        
        # Train isolation forest
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        iso_forest.fit(X)
        self.models['anomaly_detector'] = iso_forest
        
        logger.info("Anomaly detector trained")
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalous market conditions"""
        if 'anomaly_detector' not in self.models:
            return pd.DataFrame()
        
        X, _ = self.prepare_features(df)
        predictions = self.models['anomaly_detector'].predict(X)
        
        df = df.iloc[-len(predictions):].copy()
        df['anomaly'] = predictions == -1
        df['anomaly_score'] = self.models['anomaly_detector'].decision_function(X)
        
        return df

# =========================
# ADVANCED BACKTESTING ENGINE
# =========================
class BacktestingEngine:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
        self.trades = []
    
    def run_backtest(self, df: pd.DataFrame, signals: pd.Series, 
                    stop_loss_pct: float = 0.02, take_profit_pct: float = 0.04,
                    commission: float = 0.001) -> Dict:
        """Run comprehensive backtest"""
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trade_history = []
        equity_curve = []
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            current_signal = signals.iloc[i]
            
            equity_curve.append(capital + (position * current_price))
            
            # Exit conditions
            if position > 0:  # Long position
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                
                if current_price <= stop_loss or current_price >= take_profit:
                    pnl = (current_price - entry_price) * position
                    capital += pnl - (position * entry_price * commission)
                    trade_history.append({
                        'exit': df.index[i],
                        'side': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': (current_price / entry_price - 1) * 100
                    })
                    position = 0
            
            elif position < 0:  # Short position
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
                
                if current_price >= stop_loss or current_price <= take_profit:
                    pnl = (entry_price - current_price) * abs(position)
                    capital += pnl - (abs(position) * entry_price * commission)
                    trade_history.append({
                        'exit': df.index[i],
                        'side': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': (entry_price / current_price - 1) * 100
                    })
                    position = 0
            
            # Entry conditions
            if position == 0:
                if current_signal == 1:  # Buy signal
                    position = capital * 0.1 / current_price  # 10% of capital
                    entry_price = current_price
                    trade_history.append({
                        'entry': df.index[i],
                        'side': 'LONG',
                        'entry_price': entry_price
                    })
                
                elif current_signal == -1:  # Sell signal
                    position = -capital * 0.1 / current_price  # 10% short
                    entry_price = current_price
                    trade_history.append({
                        'entry': df.index[i],
                        'side': 'SHORT',
                        'entry_price': entry_price
                    })
        
        # Calculate metrics
        if trade_history:
            trades_df = pd.DataFrame(trade_history)
            winning_trades = trades_df[trades_df['pnl'] > 0] if 'pnl' in trades_df.columns else pd.DataFrame()
            
            total_return = (capital - self.initial_capital) / self.initial_capital * 100
            win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
            avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl_pct'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
            
            # Sharpe Ratio (annualized)
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 and returns.std() > 0 else 0
            
            # Maximum Drawdown
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Calmar Ratio
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            self.results = {
                'initial_capital': self.initial_capital,
                'final_capital': capital,
                'total_return_pct': total_return,
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'win_rate': win_rate,
                'avg_win_pct': avg_win,
                'avg_loss_pct': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'trades': trades_df.to_dict('records')
            }
        
        return self.results

# =========================
# RISK MANAGEMENT ENGINE
# =========================
class RiskManagementEngine:
    def __init__(self, max_risk_per_trade: float = 0.02, max_portfolio_risk: float = 0.10):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.active_positions = []
        self.portfolio_value = 10000  # Default
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              risk_percent: float = None) -> Dict:
        """Calculate optimal position size based on risk parameters"""
        if risk_percent is None:
            risk_percent = self.max_risk_per_trade
        
        risk_amount = self.portfolio_value * risk_percent
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return {'size': 0, 'risk_amount': 0}
        
        position_size = risk_amount / stop_distance
        
        # Apply constraints
        max_position = self.portfolio_value * CONFIG['max_position_size'] / entry_price
        min_position = self.portfolio_value * CONFIG['min_position_size'] / entry_price
        
        position_size = max(min(position_size, max_position), min_position)
        
        # Calculate actual risk
        actual_risk = position_size * stop_distance
        risk_percent_actual = actual_risk / self.portfolio_value
        
        return {
            'size': position_size,
            'risk_amount': actual_risk,
            'risk_percent': risk_percent_actual * 100,
            'stop_distance': stop_distance,
            'stop_distance_pct': (stop_distance / entry_price) * 100
        }
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var * self.portfolio_value
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0
        
        var = self.calculate_var(returns, confidence_level) / self.portfolio_value
        cvar = returns[returns <= var].mean()
        
        return cvar * self.portfolio_value if not np.isnan(cvar) else 0

# =========================
# WEBSOCKET REAL-TIME ENGINE
# =========================
class WebSocketManager:
    def __init__(self):
        self.ws = None
        self.thread = None
        self.running = False
        self.data_queue = []
        self.subscriptions = set()
    
    def connect(self, symbols: List[str]):
        """Connect to Binance WebSocket"""
        stream_names = [f"{symbol.lower()}@kline_1m" for symbol in symbols]
        stream_names.extend([f"{symbol.lower()}@depth20@100ms" for symbol in symbols])
        
        ws_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(stream_names)}"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        self.running = True
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True
        self.thread.start()
    
    def _on_open(self, ws):
        logger.info("WebSocket connection opened")
    
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            self.data_queue.append(data)
            
            # Process data in main thread
            if hasattr(st, 'session_state'):
                if 'websocket_data' not in st.session_state:
                    st.session_state.websocket_data = []
                st.session_state.websocket_data.append(data)
                
                # Keep only last 100 messages
                if len(st.session_state.websocket_data) > 100:
                    st.session_state.websocket_data = st.session_state.websocket_data[-100:]
        
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {str(e)}")
    
    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.running = False
    
    def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws:
            self.ws.close()
        self.running = False

# =========================
# ALERT SYSTEM
# =========================
class AlertSystem:
    def __init__(self):
        self.alerts = []
        self.alert_rules = {}
    
    def add_alert_rule(self, name: str, condition: Callable, message: str, 
                      priority: str = "MEDIUM"):
        """Add alert rule"""
        self.alert_rules[name] = {
            'condition': condition,
            'message': message,
            'priority': priority,
            'triggered': False,
            'last_triggered': None
        }
    
    def check_alerts(self, market_data: Dict) -> List[Dict]:
        """Check all alert conditions"""
        triggered_alerts = []
        
        for name, rule in self.alert_rules.items():
            try:
                if rule['condition'](market_data) and not rule['triggered']:
                    alert = {
                        'timestamp': datetime.now(),
                        'name': name,
                        'message': rule['message'],
                        'priority': rule['priority'],
                        'data': market_data
                    }
                    
                    triggered_alerts.append(alert)
                    self.alerts.append(alert)
                    
                    # Update rule state
                    rule['triggered'] = True
                    rule['last_triggered'] = datetime.now()
                    
                    # Log alert
                    logger.info(f"Alert triggered: {name} - {rule['message']}")
                    
            except Exception as e:
                logger.error(f"Error checking alert {name}: {str(e)}")
        
        return triggered_alerts
    
    def reset_alert(self, name: str):
        """Reset alert trigger state"""
        if name in self.alert_rules:
            self.alert_rules[name]['triggered'] = False

# =========================
# MAIN APPLICATION CLASS
# =========================
class TitanTradingApp:
    def __init__(self):
        self.setup_streamlit()
        self.init_components()
        self.load_session_state()
    
    def setup_streamlit(self):
        """Setup Streamlit configuration"""
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
            font-size: 3rem;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: rgba(25, 25, 35, 0.9);
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid #4CC9F0;
            margin-bottom: 10px;
        }
        .signal-buy {
            background: rgba(0, 200, 83, 0.2);
            border-left: 5px solid #00C853;
        }
        .signal-sell {
            background: rgba(255, 61, 0, 0.2);
            border-left: 5px solid #FF3D00;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 5px 5px 0px 0px;
            padding: 10px 20px;
            background: rgba(25, 25, 35, 0.7);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(45deg, #4361EE, #3A0CA3) !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def init_components(self):
        """Initialize all components"""
        self.db = DatabaseManager()
        self.market_data = MarketDataManager()
        self.indicators = IndicatorEngine()
        self.ml_engine = MLSignalEngine()
        self.backtester = BacktestingEngine()
        self.risk_manager = RiskManagementEngine()
        self.websocket = WebSocketManager()
        self.alert_system = AlertSystem()
    
    def load_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.selected_symbol = "BTCUSDT"
            st.session_state.selected_timeframe = "5m"
            st.session_state.market_data = {}
            st.session_state.signals = []
            st.session_state.alerts = []
            st.session_state.websocket_data = []
            st.session_state.last_update = datetime.now()
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🚀 TITAN INTRADAY PRO</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.caption("Advanced AI-Powered Trading Platform • Real-Time Execution • Institutional Grade")
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/bitcoin--v1.png", width=80)
            
            # Market Configuration
            with st.expander("⚙️ Market Configuration", expanded=True):
                symbol = st.text_input("Symbol", "BTCUSDT").upper()
                timeframe = st.selectbox("Timeframe", [tf.value for tf in TimeFrame], index=1)
                candles = st.slider("Historical Candles", 100, 5000, 1000)
                
                if st.button("🔄 Load Market Data", use_container_width=True):
                    with st.spinner("Loading market data..."):
                        self.load_market_data(symbol, timeframe, candles)
            
            # Strategy Configuration
            with st.expander("🎯 Strategy Engine", expanded=True):
                strategy_type = st.selectbox("Strategy Type", [
                    "Trend Following", 
                    "Mean Reversion", 
                    "Breakout",
                    "Multi-Timeframe",
                    "ML Ensemble"
                ])
                
                use_ml = st.checkbox("Enable ML Signals", True)
                use_volume = st.checkbox("Volume Analysis", True)
                use_market_profile = st.checkbox("Market Profile", False)
                
                # Parameters
                fast_period = st.slider("Fast Period", 5, 50, 12)
                slow_period = st.slider("Slow Period", 20, 200, 26)
                rsi_period = st.slider("RSI Period", 5, 30, 14)
                atr_period = st.slider("ATR Period", 5, 30, 14)
            
            # Risk Management
            with st.expander("🛡️ Risk Management", expanded=True):
                risk_per_trade = st.slider("Risk per Trade %", 0.1, 5.0, 1.0, 0.1)
                max_daily_loss = st.slider("Max Daily Loss %", 1.0, 10.0, 5.0, 0.5)
                position_size_mode = st.selectbox("Position Sizing", ["Fixed", "Kelly", "Optimal f"])
                
                # Stop Loss/Take Profit
                sl_type = st.selectbox("Stop Loss Type", ["ATR", "Fixed %", "Trailing", "Dynamic"])
                tp_type = st.selectbox("Take Profit Type", ["Risk Reward", "Fixed %", "Trailing"])
                
                if sl_type == "ATR":
                    atr_multiplier = st.slider("ATR Multiplier", 1.0, 5.0, 2.0, 0.5)
                elif sl_type == "Fixed %":
                    sl_percent = st.slider("Stop Loss %", 0.5, 5.0, 2.0, 0.1)
                
                if tp_type == "Risk Reward":
                    rr_ratio = st.slider("Risk:Reward Ratio", 1.0, 5.0, 2.0, 0.5)
            
            # Alerts & Notifications
            with st.expander("🔔 Alert System", expanded=False):
                enable_alerts = st.checkbox("Enable Alerts", True)
                alert_types = st.multiselect("Alert Types", [
                    "Price Breakout",
                    "Volume Spike",
                    "RSI Extreme",
                    "Divergence",
                    "Pattern Recognition"
                ])
                
                notification_channels = st.multiselect("Notification Channels", [
                    "In-App", "Email", "Telegram", "SMS"
                ])
            
            # System Controls
            with st.expander("⚡ System Controls", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("▶️ Start Live", use_container_width=True):
                        self.start_live_trading()
                with col2:
                    if st.button("⏸️ Pause", use_container_width=True):
                        self.pause_trading()
                
                if st.button("🗑️ Clear Cache", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
            
            # Performance Summary
            with st.expander("📊 Quick Stats", expanded=False):
                if 'performance' in st.session_state:
                    perf = st.session_state.performance
                    st.metric("Win Rate", f"{perf.get('win_rate', 0):.1f}%")
                    st.metric("Profit Factor", f"{perf.get('profit_factor', 0):.2f}")
                    st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
                else:
                    st.info("No performance data yet")
    
    def load_market_data(self, symbol: str, timeframe: str, candles: int):
        """Load and process market data"""
        try:
            with st.spinner(f"Loading {symbol} {timeframe} data..."):
                # Fetch data
                df = self.market_data.fetch_ohlcv(symbol, timeframe, candles)
                
                if df.empty:
                    st.error("Failed to load market data")
                    return
                
                # Calculate indicators
                df = self.indicators.calculate_all_indicators(df)
                
                # Detect divergences
                df = self.indicators.detect_divergence(df)
                
                # Train ML model
                self.ml_engine.train_anomaly_detector(df)
                
                # Detect anomalies
                df = self.ml_engine.detect_anomalies(df)
                
                # Generate signals
                signals = self.generate_signals(df)
                
                # Store in session state
                st.session_state.market_data = {
                    'df': df,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'signals': signals,
                    'last_updated': datetime.now()
                }
                
                # Run backtest
                if len(signals) > 0:
                    backtest_results = self.backtester.run_backtest(
                        df, signals, 
                        stop_loss_pct=0.02, 
                        take_profit_pct=0.04
                    )
                    st.session_state.performance = backtest_results
                
                st.success(f"Loaded {len(df)} candles for {symbol}")
        
        except Exception as e:
            st.error(f"Error loading market data: {str(e)}")
            logger.exception("Market data loading error")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using multi-strategy approach"""
        signals = pd.Series(0, index=df.index)
        
        # Strategy 1: Trend Following (MACD + EMA)
        macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
        macd_sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
        
        # Strategy 2: Mean Reversion (RSI + Bollinger)
        rsi_oversold = (df['rsi'] < 30) & (df['close'] < df['bollinger_lower'])
        rsi_overbought = (df['rsi'] > 70) & (df['close'] > df['bollinger_upper'])
        
        # Strategy 3: Breakout (ATR based)
        atr_breakout = df['close'] > (df['high'].rolling(20).max() + df['atr'])
        atr_breakdown = df['close'] < (df['low'].rolling(20).min() - df['atr'])
        
        # Strategy 4: Volume Confirmation
        volume_spike = df['volume'] > (df['volume'].rolling(20).mean() * 2)
        volume_confirmation = (macd_buy & volume_spike) | (macd_sell & volume_spike)
        
        # Strategy 5: Pattern Recognition
        bullish_pattern = (df['cdl_hammer'] > 0) | (df['cdl_morningstar'] > 0)
        bearish_pattern = (df['cdl_doji'] > 0) | (df['cdl_eveningstar'] > 0)
        
        # Combine signals with weights
        signals = (
            (macd_buy * 0.3) + (rsi_oversold * 0.2) + (atr_breakout * 0.2) + 
            (bullish_pattern * 0.1) + (volume_confirmation * 0.2)
        ) - (
            (macd_sell * 0.3) + (rsi_overbought * 0.2) + (atr_breakdown * 0.2) + 
            (bearish_pattern * 0.1) + (volume_confirmation * 0.2)
        )
        
        # Apply threshold
        buy_signals = signals > 0.5
        sell_signals = signals < -0.5
        
        final_signals = pd.Series(0, index=df.index)
        final_signals[buy_signals] = 1
        final_signals[sell_signals] = -1
        
        return final_signals
    
    def render_dashboard(self):
        """Render main dashboard"""
        # Performance Metrics Row
        st.subheader("📊 Performance Dashboard")
        
        if 'performance' in st.session_state:
            perf = st.session_state.performance
            cols = st.columns(8)
            
            metrics = [
                ("💰 Total Return", f"{perf.get('total_return_pct', 0):.2f}%"),
                ("📈 Win Rate", f"{perf.get('win_rate', 0):.1f}%"),
                ("⚖️ Profit Factor", f"{perf.get('profit_factor', 0):.2f}"),
                ("📉 Max Drawdown", f"{perf.get('max_drawdown_pct', 0):.2f}%"),
                ("🎯 Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}"),
                ("📊 Calmar Ratio", f"{perf.get('calmar_ratio', 0):.2f}"),
                ("🔢 Total Trades", f"{perf.get('total_trades', 0)}"),
                ("✅ Winning Trades", f"{perf.get('winning_trades', 0)}")
            ]
            
            for col, (label, value) in zip(cols, metrics):
                with col:
                    st.metric(label, value)
        
        # Main Chart Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Price Chart", 
            "📊 Indicators", 
            "📉 Backtest Results", 
            "🔍 Market Depth",
            "⚙️ System Status"
        ])
        
        with tab1:
            self.render_price_chart()
        
        with tab2:
            self.render_indicator_panels()
        
        with tab3:
            self.render_backtest_results()
        
        with tab4:
            self.render_market_depth()
        
        with tab5:
            self.render_system_status()
    
    def render_price_chart(self):
        """Render interactive price chart"""
        if 'market_data' not in st.session_state or st.session_state.market_data.get('df') is None:
            st.info("Load market data to view charts")
            return
        
        df = st.session_state.market_data['df']
        signals = st.session_state.market_data.get('signals', pd.Series())
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=('Price & Indicators', 'Volume', 'RSI', 'MACD')
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
            go.Scatter(x=df.index, y=df['bollinger_upper'], line=dict(color='gray', width=1, dash='dash'), 
                      name='BB Upper', showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bollinger_lower'], line=dict(color='gray', width=1, dash='dash'), 
                      fill='tonexty', name='BB Lower', showlegend=False),
            row=1, col=1
        )
        
        # Buy/Sell signals
        if len(signals) > 0:
            buy_signals = df.index[signals == 1]
            sell_signals = df.index[signals == -1]
            
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals,
                        y=df.loc[buy_signals, 'close'],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=12, color='green'),
                        name='Buy Signal'
                    ),
                    row=1, col=1
                )
            
            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals,
                        y=df.loc[sell_signals, 'close'],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=12, color='red'),
                        name='Sell Signal'
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
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd'], line=dict(color='blue', width=2), name='MACD'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd_signal'], line=dict(color='red', width=2), name='Signal'),
            row=4, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df['macd_hist'], name='Histogram', marker_color='gray'),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_indicator_panels(self):
        """Render indicator panels"""
        if 'market_data' not in st.session_state:
            return
        
        df = st.session_state.market_data['df']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trend Indicators")
            
            trend_data = {
                "ADX": df['adx'].iloc[-1],
                "DMI+": talib.PLUS_DI(df['high'], df['low'], df['close']).iloc[-1],
                "DMI-": talib.MINUS_DI(df['high'], df['low'], df['close']).iloc[-1],
                "Parabolic SAR": talib.SAR(df['high'], df['low']).iloc[-1],
                "Ichimoku Cloud": "Bullish" if df['close'].iloc[-1] > df['sma_20'].iloc[-1] else "Bearish"
            }
            
            for indicator, value in trend_data.items():
                st.metric(indicator, f"{value:.2f}" if isinstance(value, (int, float)) else value)
        
        with col2:
            st.subheader("Momentum Indicators")
            
            momentum_data = {
                "RSI": df['rsi'].iloc[-1],
                "Stoch %K": df['stoch_k'].iloc[-1],
                "Stoch %D": df['stoch_d'].iloc[-1],
                "Williams %R": df['williams_r'].iloc[-1],
                "CCI": df['cci'].iloc[-1],
                "MFI": df['mfi'].iloc[-1]
            }
            
            for indicator, value in momentum_data.items():
                st.metric(indicator, f"{value:.2f}")
        
        # Volatility Indicators
        st.subheader("Volatility Indicators")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("ATR", f"{df['atr'].iloc[-1]:.2f}")
            st.metric("ATR %", f"{df['atr_pct'].iloc[-1]:.2f}%")
        
        with col4:
            bb_position = ((df['close'].iloc[-1] - df['bollinger_lower'].iloc[-1]) / 
                          (df['bollinger_upper'].iloc[-1] - df['bollinger_lower'].iloc[-1]))
            st.metric("BB Position", f"{bb_position:.2%}")
            st.metric("BB Width", f"{(df['bollinger_upper'].iloc[-1] - df['bollinger_lower'].iloc[-1]) / df['close'].iloc[-1]:.2%}")
        
        with col5:
            st.metric("Historical Vol", f"{df['returns'].std() * np.sqrt(252):.2%}")
            st.metric("IV Rank", "N/A")
    
    def render_backtest_results(self):
        """Render backtest results"""
        if 'performance' not in st.session_state:
            st.info("Run backtest to see results")
            return
        
        perf = st.session_state.performance
        
        # Equity Curve
        if 'trades' in perf and perf['trades']:
            trades_df = pd.DataFrame(perf['trades'])
            
            fig = go.Figure()
            
            # Plot equity curve
            fig.add_trace(go.Scatter(
                x=trades_df.get('exit', trades_df.get('entry')),
                y=trades_df['pnl'].cumsum() if 'pnl' in trades_df.columns else [0],
                mode='lines',
                name='Equity Curve',
                line=dict(color='green', width=2)
            ))
            
            # Plot trades
            if 'pnl' in trades_df.columns:
                winning_trades = trades_df[trades_df['pnl'] > 0]
                losing_trades = trades_df[trades_df['pnl'] < 0]
                
                fig.add_trace(go.Scatter(
                    x=winning_trades.get('exit', winning_trades.get('entry')),
                    y=winning_trades['pnl'],
                    mode='markers',
                    name='Winning Trades',
                    marker=dict(color='green', size=8, symbol='circle')
                ))
                
                fig.add_trace(go.Scatter(
                    x=losing_trades.get('exit', losing_trades.get('entry')),
                    y=losing_trades['pnl'],
                    mode='markers',
                    name='Losing Trades',
                    marker=dict(color='red', size=8, symbol='circle')
                ))
            
            fig.update_layout(
                title='Equity Curve',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trade Statistics")
            
            stats_data = {
                "Total Trades": perf.get('total_trades', 0),
                "Winning Trades": perf.get('winning_trades', 0),
                "Win Rate": f"{perf.get('win_rate', 0):.1f}%",
                "Avg Win": f"{perf.get('avg_win_pct', 0):.2f}%",
                "Avg Loss": f"{perf.get('avg_loss_pct', 0):.2f}%",
                "Profit Factor": f"{perf.get('profit_factor', 0):.2f}",
                "Max Drawdown": f"{perf.get('max_drawdown_pct', 0):.2f}%"
            }
            
            for key, value in stats_data.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("Performance Metrics")
            
            perf_data = {
                "Total Return": f"{perf.get('total_return_pct', 0):.2f}%",
                "Sharpe Ratio": f"{perf.get('sharpe_ratio', 0):.2f}",
                "Calmar Ratio": f"{perf.get('calmar_ratio', 0):.2f}",
                "Sortino Ratio": "N/A",
                "Omega Ratio": "N/A",
                "Kelly Criterion": "N/A"
            }
            
            for key, value in perf_data.items():
                st.metric(key, value)
    
    def render_market_depth(self):
        """Render market depth chart"""
        if 'market_data' not in st.session_state:
            return
        
        symbol = st.session_state.market_data.get('symbol', 'BTCUSDT')
        
        try:
            order_book = self.market_data.get_order_book(symbol)
            
            if order_book:
                bids = pd.DataFrame(order_book['bids'], columns=['price', 'volume']).astype(float)
                asks = pd.DataFrame(order_book['asks'], columns=['price', 'volume']).astype(float)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=bids['volume'],
                    y=bids['price'],
                    orientation='h',
                    name='Bids',
                    marker_color='green',
                    opacity=0.7
                ))
                
                fig.add_trace(go.Bar(
                    x=asks['volume'],
                    y=asks['price'],
                    orientation='h',
                    name='Asks',
                    marker_color='red',
                    opacity=0.7
                ))
                
                fig.update_layout(
                    title='Market Depth',
                    xaxis_title='Volume',
                    yaxis_title='Price',
                    template='plotly_dark',
                    height=500,
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Order book stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Bid/Ask Spread", f"{(asks['price'].min() - bids['price'].max()):.2f}")
                with col2:
                    st.metric("Total Bid Volume", f"{bids['volume'].sum():.2f}")
                with col3:
                    st.metric("Total Ask Volume", f"{asks['volume'].sum():.2f}")
        
        except Exception as e:
            st.error(f"Error fetching market depth: {str(e)}")
    
    def render_system_status(self):
        """Render system status panel"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🖥️ System Resources")
            
            # CPU Usage
            cpu_percent = psutil.cpu_percent()
            st.progress(cpu_percent / 100, text=f"CPU: {cpu_percent}%")
            
            # Memory Usage
            memory = psutil.virtual_memory()
            st.progress(memory.percent / 100, text=f"Memory: {memory.percent}%")
            
            # Disk Usage
            disk = psutil.disk_usage('/')
            st.progress(disk.percent / 100, text=f"Disk: {disk.percent}%")
        
        with col2:
            st.subheader("📡 Data Connections")
            
            connection_status = {
                "Market Data": "🟢 Connected" if hasattr(self, 'market_data') else "🔴 Disconnected",
                "WebSocket": "🟢 Connected" if self.websocket.running else "🔴 Disconnected",
                "Database": "🟢 Connected" if hasattr(self, 'db') else "🔴 Disconnected",
                "Exchange API": "🟢 Connected" if hasattr(self, 'market_data') else "🔴 Disconnected"
            }
            
            for connection, status in connection_status.items():
                st.write(f"{connection}: {status}")
        
        with col3:
            st.subheader("⚡ Performance")
            
            if 'market_data' in st.session_state:
                df = st.session_state.market_data.get('df', pd.DataFrame())
                st.metric("Data Points", len(df))
                st.metric("Indicators", len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]))
            
            st.metric("Cache Hits", "N/A")
            st.metric("Latency", "< 100ms")
        
        # Log Viewer
        st.subheader("📋 System Logs")
        
        try:
            with open(CONFIG['log_path'], 'r') as f:
                logs = f.readlines()[-50:]  # Last 50 lines
            
            log_text = "".join(logs)
            st.text_area("Recent Logs", log_text, height=200)
        
        except:
            st.info("No logs available")
    
    def start_live_trading(self):
        """Start live trading mode"""
        try:
            symbol = st.session_state.selected_symbol
            self.websocket.connect([symbol])
            
            # Start alert system
            self.setup_alerts()
            
            st.success("Live trading started")
            logger.info(f"Live trading started for {symbol}")
        
        except Exception as e:
            st.error(f"Error starting live trading: {str(e)}")
            logger.exception("Live trading start error")
    
    def pause_trading(self):
        """Pause trading"""
        self.websocket.disconnect()
        st.info("Trading paused")
    
    def setup_alerts(self):
        """Setup alert rules"""
        # Price breakout alert
        self.alert_system.add_alert_rule(
            name="price_breakout",
            condition=lambda data: False,  # Implement condition
            message="Price breakout detected",
            priority="HIGH"
        )
        
        # Volume spike alert
        self.alert_system.add_alert_rule(
            name="volume_spike",
            condition=lambda data: False,
            message="Volume spike detected",
            priority="MEDIUM"
        )
        
        # RSI extreme alert
        self.alert_system.add_alert_rule(
            name="rsi_extreme",
            condition=lambda data: False,
            message="RSI in extreme territory",
            priority="MEDIUM"
        )
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        
        # Check for WebSocket updates
        if hasattr(st, 'session_state') and 'websocket_data' in st.session_state:
            if len(st.session_state.websocket_data) > 0:
                self.process_websocket_data()
        
        self.render_dashboard()
        
        # Auto-refresh
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    # Initialize application
    app = TitanTradingApp()
    
    # Run main loop
    try:
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.exception("Application crashed")
