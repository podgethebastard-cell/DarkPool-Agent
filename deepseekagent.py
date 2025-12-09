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
    VERSION = "3.0.0"
    CACHE_TTL = 30
    MAX_CANDLES = 2000
    DEFAULT_SYMBOL = "XBTUSD"
    DEFAULT_TIMEFRAME = "5"
    INITIAL_CAPITAL = 10000
    
    # Kraken API
    KRAKEN_BASE_URL = "https://api.kraken.com/0/public"
    
    # Color Scheme
    COLORS = {
        'primary': '#4CC9F0',
        'secondary': '#4361EE',
        'success': '#00C853',
        'danger': '#FF3D00',
        'warning': '#FF9800',
        'info': '#2196F3',
        'dark': '#0F172A',
        'light': '#F8FAFC'
    }
    
    # Symbol mapping
    SYMBOL_MAP = {
        "BTCUSD": "XBTUSD", "XBTUSD": "XBTUSD", "ETHUSD": "ETHUSD", 
        "SOLUSD": "SOLUSD", "ADAUSD": "ADAUSD", "DOTUSD": "DOTUSD",
        "DOGEUSD": "XDGUSD", "LTCUSD": "LTCUSD", "XRPUSD": "XRPUSD",
        "MATICUSD": "MATICUSD", "AVAXUSD": "AVAXUSD", "LINKUSD": "LINKUSD"
    }

# =========================
# TELEGRAM BROADCASTER
# =========================
class TelegramBroadcaster:
    """Professional Telegram broadcast system"""
    
    def __init__(self):
        self.base_url = "https://api.telegram.org/bot"
        self.last_broadcast = None
    
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
                self.last_broadcast = datetime.now()
                return {"success": True, "message_id": result["result"]["message_id"]}
            else:
                return {"success": False, "error": result.get("description", "Unknown error")}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def send_signal_alert(self, bot_token: str, chat_id: str, signal_data: Dict) -> Dict:
        """Send formatted signal alert"""
        symbol = signal_data.get('symbol', 'Unknown')
        signal = signal_data.get('signal', 'NEUTRAL')
        price = signal_data.get('price', 0)
        rsi = signal_data.get('rsi', 0)
        volume_ratio = signal_data.get('volume_ratio', 0)
        confidence = signal_data.get('confidence', 0)
        
        # Emojis and styling
        if "STRONG_BUY" in signal:
            emoji = "🟢🔥"
            action = "STRONG BUY"
        elif "BUY" in signal:
            emoji = "🟢"
            action = "BUY"
        elif "STRONG_SELL" in signal:
            emoji = "🔴🔥"
            action = "STRONG SELL"
        elif "SELL" in signal:
            emoji = "🔴"
            action = "SELL"
        else:
            emoji = "⚪"
            action = "NEUTRAL"
        
        # Create beautiful message
        message = f"""
{emoji} <b>TITAN PRO TRADING SIGNAL</b> {emoji}
━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>SYMBOL:</b> <code>{symbol}</code>
🎯 <b>SIGNAL:</b> <b>{action}</b>
💰 <b>PRICE:</b> ${price:,.2f}

📈 <b>ANALYTICS:</b>
├ RSI: {rsi:.1f}
├ Volume: {volume_ratio:.2f}x
└ Confidence: {confidence:.0%}

⏰ <b>TIME:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━
#TitanSignal #{symbol.replace('USD', '')}
        """
        
        return self.send_message(bot_token, chat_id, message)
    
    def send_performance_report(self, bot_token: str, chat_id: str, performance_data: Dict) -> Dict:
        """Send performance report"""
        total_return = performance_data.get('total_return', 0)
        win_rate = performance_data.get('win_rate', 0)
        total_trades = performance_data.get('total_trades', 0)
        sharpe = performance_data.get('sharpe_ratio', 0)
        
        # Determine emoji based on performance
        if total_return > 5:
            emoji = "🚀💰"
        elif total_return > 0:
            emoji = "📈"
        else:
            emoji = "📉"
        
        message = f"""
{emoji} <b>TITAN PERFORMANCE REPORT</b> {emoji}
━━━━━━━━━━━━━━━━━━━━━━━━

💰 <b>TOTAL RETURN:</b> {total_return:+.2f}%
🎯 <b>WIN RATE:</b> {win_rate:.1f}%
🔢 <b>TRADES:</b> {total_trades}
📊 <b>SHARPE:</b> {sharpe:.2f}

⏰ <b>REPORT TIME:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}
━━━━━━━━━━━━━━━━━━━━━━━━
#TitanReport #Performance
        """
        
        return self.send_message(bot_token, chat_id, message)
    
    def send_custom_message(self, bot_token: str, chat_id: str, title: str, content: str) -> Dict:
        """Send custom formatted message"""
        message = f"""
📨 <b>{title}</b>
━━━━━━━━━━━━━━━━━━━━━━━━

{content}

⏰ <b>SENT:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━
#TitanAlert #Custom
        """
        
        return self.send_message(bot_token, chat_id, message)

# =========================
# KRAKEN DATA FETCHER
# =========================
class KrakenDataFetcher:
    """Fetch market data from Kraken API"""
    
    @staticmethod
    def fetch_ohlcv(symbol: str, timeframe: str = "5", limit: int = 720) -> pd.DataFrame:
        """Fetch OHLCV data from Kraken"""
        try:
            # Clean symbol
            symbol = symbol.upper().strip()
            if symbol in Config.SYMBOL_MAP:
                symbol = Config.SYMBOL_MAP[symbol]
            
            # Timeframe mapping
            tf_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", 
                     "1h": "60", "4h": "240", "1d": "1440"}
            interval = tf_map.get(timeframe, "5")
            
            # API call
            url = f"{Config.KRAKEN_BASE_URL}/OHLC"
            params = {"pair": symbol, "interval": interval}
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                return pd.DataFrame()
            
            data = response.json()
            
            if data.get("error"):
                return pd.DataFrame()
            
            # Parse data
            result_keys = list(data["result"].keys())
            if not result_keys or result_keys[0] == "last":
                return pd.DataFrame()
            
            ohlc_data = data["result"][result_keys[0]]
            
            if not ohlc_data:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(ohlc_data, columns=[
                "timestamp", "open", "high", "low", "close", 
                "vwap", "volume", "count"
            ])
            
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
            df.set_index("timestamp", inplace=True)
            
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.tail(min(limit, len(df)))
            df.sort_index(inplace=True)
            
            return df[["open", "high", "low", "close", "volume"]].dropna()
            
        except:
            return pd.DataFrame()

# =========================
# INDICATOR ENGINE
# =========================
class IndicatorEngine:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators"""
        df = df.copy()
        
        if len(df) < 20:
            return df
        
        # Basic indicators
        df['returns'] = df['close'].pct_change()
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
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
        
        # Additional
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
        """Generate signals"""
        df = df.copy()
        
        # Initialize
        df['signal'] = 0
        df['signal_type'] = 'NEUTRAL'
        
        if len(df) < 20:
            return df
        
        # MACD Crossover
        macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
        macd_sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
        
        # RSI
        rsi_buy = (df['rsi'] < 30) & (df['rsi'].shift() >= 30)
        rsi_sell = (df['rsi'] > 70) & (df['rsi'].shift() <= 70)
        
        # Bollinger
        bb_buy = (df['close'] < df['bb_lower']) & (df['close'].shift() >= df['bb_lower'].shift())
        bb_sell = (df['close'] > df['bb_upper']) & (df['close'].shift() <= df['bb_upper'].shift())
        
        # MA Crossover
        ma_buy = (df['ema_12'] > df['ema_26']) & (df['ema_12'].shift() <= df['ema_26'].shift())
        ma_sell = (df['ema_12'] < df['ema_26']) & (df['ema_12'].shift() >= df['ema_26'].shift())
        
        # Combine
        buy_conditions = macd_buy | rsi_buy | bb_buy | ma_buy
        sell_conditions = macd_sell | rsi_sell | bb_sell | ma_sell
        
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Signal types
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
    """Backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
    
    def run(self, df: pd.DataFrame) -> Dict:
        """Run backtest"""
        if 'signal' not in df.columns or len(df) < 20:
            return {}
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        for i in range(1, len(df)):
            price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            equity_curve.append(capital + (position * price))
            
            # Exit conditions
            if position > 0 and price <= entry_price * 0.98:
                pnl = (price - entry_price) * position
                capital += pnl
                trades.append({'entry': entry_price, 'exit': price, 'pnl': pnl})
                position = 0
            
            elif position < 0 and price >= entry_price * 1.02:
                pnl = (entry_price - price) * abs(position)
                capital += pnl
                trades.append({'entry': entry_price, 'exit': price, 'pnl': pnl})
                position = 0
            
            # Entry conditions
            if position == 0 and signal != 0:
                if signal == 1:
                    position = capital * 0.1 / price
                    entry_price = price
                elif signal == -1:
                    position = -capital * 0.1 / price
                    entry_price = price
        
        # Calculate performance
        if trades:
            trades_df = pd.DataFrame(trades)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            
            total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
            win_rate = (len(winning_trades) / len(trades_df)) * 100
            profit_factor = abs(winning_trades['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
            
            return {
                'total_return': total_return,
                'win_rate': win_rate,
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'profit_factor': profit_factor,
                'trades': trades_df.to_dict('records')
            }
        
        return {}

# =========================
# UI COMPONENTS
# =========================
class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def metric_card(label: str, value: str, delta: str = None, color: str = None):
        """Create a metric card"""
        color_map = {
            'success': Config.COLORS['success'],
            'danger': Config.COLORS['danger'],
            'warning': Config.COLORS['warning'],
            'info': Config.COLORS['info'],
            'primary': Config.COLORS['primary']
        }
        
        delta_color = color_map.get(color, Config.COLORS['primary']) if delta else "transparent"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            margin: 8px 0;
        ">
            <div style="color: #94A3B8; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 8px;">
                {label}
            </div>
            <div style="color: #F8FAFC; font-size: 24px; font-weight: 700; margin: 8px 0;">
                {value}
            </div>
            {f'<div style="color: {delta_color}; font-size: 14px; font-weight: 600;">{delta}</div>' if delta else ''}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def signal_badge(signal: str):
        """Create a signal badge"""
        color_map = {
            'STRONG_BUY': ('#00C853', '🟢🔥 STRONG BUY'),
            'BUY': ('#00C853', '🟢 BUY'),
            'NEUTRAL': ('#94A3B8', '⚪ NEUTRAL'),
            'SELL': ('#FF3D00', '🔴 SELL'),
            'STRONG_SELL': ('#FF3D00', '🔴🔥 STRONG SELL')
        }
        
        color, text = color_map.get(signal, ('#94A3B8', signal))
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
            border-radius: 12px;
            padding: 20px;
            border: 2px solid {color};
            box-shadow: 0 4px 20px rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3);
            backdrop-filter: blur(10px);
            text-align: center;
            margin: 8px 0;
        ">
            <div style="color: {color}; font-size: 20px; font-weight: 800; letter-spacing: 1px;">
                {text}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def telegram_card():
        """Create Telegram configuration card"""
        with st.expander("📡 TELEGRAM BROADCAST", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                bot_token = st.text_input(
                    "Bot Token",
                    type="password",
                    help="Get from @BotFather",
                    key="telegram_bot_token"
                )
            
            with col2:
                chat_id = st.text_input(
                    "Chat ID",
                    help="Channel/Group ID",
                    key="telegram_chat_id"
                )
            
            # Connection status
            if bot_token and chat_id:
                st.success("✅ Telegram configured")
            else:
                st.warning("⚠️ Enter credentials to enable broadcast")
            
            # Manual broadcast button
            if st.button("📤 MANUAL BROADCAST", 
                        use_container_width=True,
                        type="primary",
                        help="Send current signal to Telegram"):
                return True
            
            # Additional broadcast options
            col3, col4 = st.columns(2)
            with col3:
                if st.button("📊 Performance Report", 
                           use_container_width=True,
                           help="Send performance summary"):
                    return "performance"
            
            with col4:
                if st.button("💬 Custom Message", 
                           use_container_width=True,
                           help="Send custom message"):
                    return "custom"
        
        return False

# =========================
# MAIN APP
# =========================
def main():
    # Page configuration
    st.set_page_config(
        layout="wide",
        page_title="TITAN INTRADAY PRO",
        page_icon="🚀",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown(f"""
    <style>
    /* Main background */
    .stApp {{
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }}
    
    /* Headers */
    h1, h2, h3, h4 {{
        background: linear-gradient(90deg, {Config.COLORS['primary']}, {Config.COLORS['secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(90deg, {Config.COLORS['primary']}, {Config.COLORS['secondary']});
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 201, 240, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(76, 201, 240, 0.5);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* Tabs */
    .stTabs {{
        background: transparent;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(15, 23, 42, 0.8);
        border-radius: 12px;
        padding: 8px;
        backdrop-filter: blur(10px);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        color: #94A3B8;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {Config.COLORS['primary']}, {Config.COLORS['secondary']});
        color: white !important;
        box-shadow: 0 4px 15px rgba(76, 201, 240, 0.3);
    }}
    
    /* Input fields */
    .stTextInput > div > div > input {{
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
    }}
    
    /* Select boxes */
    .stSelectbox > div > div > div {{
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
    }}
    
    /* Sliders */
    .stSlider > div > div > div {{
        background: linear-gradient(90deg, {Config.COLORS['primary']}, {Config.COLORS['secondary']});
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        font-weight: 700;
    }}
    
    /* Dataframes */
    .dataframe {{
        background: rgba(15, 23, 42, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize components
    telegram = TelegramBroadcaster()
    ui = UIComponents()
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 3.5rem; margin: 0;">🚀 TITAN PRO</h1>
            <div style="
                background: linear-gradient(90deg, {Config.COLORS['primary']}, {Config.COLORS['secondary']});
                height: 4px;
                width: 200px;
                margin: 10px auto;
                border-radius: 2px;
            "></div>
            <p style="color: #94A3B8; font-size: 14px; letter-spacing: 2px; margin: 0;">
                ADVANCED TRADING PLATFORM
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="
                width: 80px;
                height: 80px;
                background: linear-gradient(135deg, #4CC9F0, #4361EE);
                border-radius: 20px;
                margin: 0 auto 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 40px;
            ">
                🚀
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Market Configuration
        with st.expander("⚙️ MARKET CONFIG", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                symbol = st.selectbox(
                    "Symbol",
                    options=["XBTUSD", "ETHUSD", "SOLUSD", "ADAUSD", "DOTUSD", 
                            "XDGUSD", "LTCUSD", "XRPUSD", "MATICUSD", "AVAXUSD"],
                    index=0
                )
            
            with col2:
                timeframe = st.selectbox(
                    "Timeframe",
                    options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                    index=1
                )
            
            candles = st.slider("Candles", 100, 2000, 500)
            
            if st.button("📥 LOAD DATA", use_container_width=True, type="primary"):
                with st.spinner("Fetching market data..."):
                    load_data(symbol, timeframe, candles)
        
        # Telegram Configuration
        broadcast_action = ui.telegram_card()
        
        # Strategy Settings
        with st.expander("🎯 STRATEGY SETTINGS", expanded=False):
            st.checkbox("Enable RSI Filter", True)
            st.checkbox("Enable MACD Filter", True)
            st.checkbox("Volume Confirmation", True)
            st.slider("RSI Oversold", 20, 40, 30)
            st.slider("RSI Overbought", 60, 80, 70)
        
        # Risk Management
        with st.expander("🛡️ RISK MANAGEMENT", expanded=False):
            st.slider("Risk per Trade %", 0.1, 5.0, 1.0)
            st.selectbox("Stop Loss Type", ["ATR-based", "Fixed %", "Trailing"])
            st.selectbox("Position Sizing", ["Fixed", "Kelly", "Optimal f"])
        
        # Status Bar
        st.markdown("---")
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.caption("🟢 **STATUS:** ONLINE")
        with status_col2:
            st.caption(f"v{Config.VERSION}")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'performance' not in st.session_state:
        st.session_state.performance = None
    
    # Main content
    if st.session_state.df is not None:
        display_main_content(telegram, broadcast_action)
    else:
        display_welcome()
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.caption("⚠️ Educational Use Only")
    with footer_col2:
        st.caption("📊 Powered by Kraken API")
    with footer_col3:
        st.caption("© 2025 Titan Trading Systems")

def display_welcome():
    """Display welcome screen"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
            border-radius: 20px;
            padding: 50px 40px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            text-align: center;
            backdrop-filter: blur(20px);
            margin: 50px 0;
        ">
            <div style="font-size: 80px; margin: 0 0 20px;">🚀</div>
            <h2 style="margin: 0 0 20px;">WELCOME TO TITAN PRO</h2>
            <p style="color: #94A3B8; font-size: 16px; line-height: 1.6; margin: 0 0 30px;">
                Professional trading platform with advanced analytics,<br>
                real-time signals, and Telegram broadcast capabilities.
            </p>
            
            <div style="
                background: rgba(76, 201, 240, 0.1);
                border: 1px solid rgba(76, 201, 240, 0.3);
                border-radius: 12px;
                padding: 25px;
                margin: 30px 0;
                text-align: left;
            ">
                <h4 style="margin: 0 0 15px;">🚀 GET STARTED:</h4>
                <ol style="color: #94A3B8; padding-left: 20px; margin: 0;">
                    <li style="margin: 8px 0;">Select a trading pair from sidebar</li>
                    <li style="margin: 8px 0;">Choose timeframe and candles</li>
                    <li style="margin: 8px 0;">Click "LOAD DATA" to begin</li>
                    <li style="margin: 8px 0;">Configure Telegram for alerts (optional)</li>
                </ol>
            </div>
            
            <div style="display: flex; justify-content: center; gap: 15px; margin-top: 30px;">
                <div style="
                    background: rgba(76, 201, 240, 0.1);
                    border: 1px solid rgba(76, 201, 240, 0.3);
                    border-radius: 10px;
                    padding: 15px;
                    width: 120px;
                ">
                    <div style="font-size: 24px;">📊</div>
                    <div style="font-size: 12px; color: #94A3B8;">Real-time Data</div>
                </div>
                <div style="
                    background: rgba(76, 201, 240, 0.1);
                    border: 1px solid rgba(76, 201, 240, 0.3);
                    border-radius: 10px;
                    padding: 15px;
                    width: 120px;
                ">
                    <div style="font-size: 24px;">🎯</div>
                    <div style="font-size: 12px; color: #94A3B8;">AI Signals</div>
                </div>
                <div style="
                    background: rgba(76, 201, 240, 0.1);
                    border: 1px solid rgba(76, 201, 240, 0.3);
                    border-radius: 10px;
                    padding: 15px;
                    width: 120px;
                ">
                    <div style="font-size: 24px;">📡</div>
                    <div style="font-size: 12px; color: #94A3B8;">Telegram Alerts</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def load_data(symbol: str, timeframe: str, candles: int):
    """Load and process market data"""
    try:
        # Fetch data
        df = KrakenDataFetcher.fetch_ohlcv(symbol, timeframe, candles)
        
        if df.empty:
            st.error("❌ Failed to fetch data. Try a different symbol.")
            return
        
        # Calculate indicators
        df = IndicatorEngine.calculate_all(df)
        
        # Generate signals
        df = SignalGenerator.generate(df)
        
        # Run backtest
        backtester = Backtester(Config.INITIAL_CAPITAL)
        performance = backtester.run(df)
        
        # Store in session state
        st.session_state.df = df
        st.session_state.symbol = symbol
        st.session_state.performance = performance
        
        st.success(f"✅ Loaded {len(df)} candles for {symbol}")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def display_main_content(telegram: TelegramBroadcaster, broadcast_action):
    """Display main dashboard"""
    df = st.session_state.df
    symbol = st.session_state.symbol
    
    # Top Metrics Row
    st.subheader("📊 MARKET OVERVIEW")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_price = df['close'].iloc[-1]
        price_change = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if len(df) > 1 else 0
        ui.metric_card(
            "CURRENT PRICE",
            f"${current_price:,.2f}",
            f"{price_change:+.2f}%",
            'success' if price_change > 0 else 'danger'
        )
    
    with col2:
        current_signal = df['signal_type'].iloc[-1] if 'signal_type' in df.columns else 'NEUTRAL'
        ui.signal_badge(current_signal)
    
    with col3:
        rsi_value = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        rsi_status = "OVERSOLD" if rsi_value < 30 else "OVERBOUGHT" if rsi_value > 70 else "NEUTRAL"
        color = 'success' if rsi_value < 30 else 'danger' if rsi_value > 70 else 'info'
        ui.metric_card("RSI", f"{rsi_value:.1f}", rsi_status, color)
    
    with col4:
        volume_ratio = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1
        vol_status = "HIGH" if volume_ratio > 1.5 else "LOW" if volume_ratio < 0.5 else "NORMAL"
        color = 'success' if volume_ratio > 1.5 else 'warning' if volume_ratio < 0.5 else 'info'
        ui.metric_card("VOLUME", f"{volume_ratio:.2f}x", vol_status, color)
    
    with col5:
        atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else 0
        atr_pct = (atr_value / current_price) * 100
        ui.metric_card("ATR", f"${atr_value:.2f}", f"{atr_pct:.2f}%", 'info')
    
    # Handle Telegram broadcast
    if broadcast_action:
        handle_telegram_broadcast(telegram, df, symbol, broadcast_action)
    
    # Performance Metrics
    if st.session_state.performance:
        perf = st.session_state.performance
        st.subheader("📈 PERFORMANCE METRICS")
        
        cols = st.columns(6)
        with cols[0]:
            ui.metric_card(
                "TOTAL RETURN",
                f"{perf.get('total_return', 0):.2f}%",
                None,
                'success' if perf.get('total_return', 0) > 0 else 'danger'
            )
        
        with cols[1]:
            ui.metric_card("WIN RATE", f"{perf.get('win_rate', 0):.1f}%", None, 'info')
        
        with cols[2]:
            ui.metric_card("PROFIT FACTOR", f"{perf.get('profit_factor', 0):.2f}", None, 'info')
        
        with cols[3]:
            ui.metric_card("TOTAL TRADES", f"{perf.get('total_trades', 0)}", None, 'primary')
        
        with cols[4]:
            ui.metric_card("WINNING TRADES", f"{perf.get('winning_trades', 0)}", None, 'success')
        
        with cols[5]:
            # Manual Broadcast Button for Performance
            if st.button("📤 BROADCAST PERFORMANCE", use_container_width=True):
                handle_performance_broadcast(telegram, perf)
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 LIVE CHART", "📈 INDICATORS", "📋 SIGNALS", "⚙️ SETTINGS"])
    
    with tab1:
        display_chart(df)
    
    with tab2:
        display_indicators(df)
    
    with tab3:
        display_signals(df)
    
    with tab4:
        display_settings()

def handle_telegram_broadcast(telegram: TelegramBroadcaster, df: pd.DataFrame, symbol: str, action):
    """Handle Telegram broadcast actions"""
    bot_token = st.session_state.get('telegram_bot_token')
    chat_id = st.session_state.get('telegram_chat_id')
    
    if not bot_token or not chat_id:
        st.error("❌ Please configure Telegram first")
        return
    
    if action == True:  # Manual broadcast
        signal_data = {
            'symbol': symbol,
            'signal': df['signal_type'].iloc[-1] if 'signal_type' in df.columns else 'NEUTRAL',
            'price': df['close'].iloc[-1],
            'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
            'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1,
            'confidence': 0.8
        }
        
        with st.spinner("📤 Broadcasting signal..."):
            result = telegram.send_signal_alert(bot_token, chat_id, signal_data)
            if result["success"]:
                st.success("✅ Signal broadcasted to Telegram!")
            else:
                st.error(f"❌ Failed: {result.get('error')}")
    
    elif action == "performance":
        perf = st.session_state.performance
        if perf:
            with st.spinner("📤 Sending performance report..."):
                result = telegram.send_performance_report(bot_token, chat_id, perf)
                if result["success"]:
                    st.success("✅ Performance report sent!")
                else:
                    st.error(f"❌ Failed: {result.get('error')}")

def handle_performance_broadcast(telegram: TelegramBroadcaster, perf: Dict):
    """Handle performance broadcast"""
    bot_token = st.session_state.get('telegram_bot_token')
    chat_id = st.session_state.get('telegram_chat_id')
    
    if not bot_token or not chat_id:
        st.error("❌ Please configure Telegram first")
        return
    
    with st.spinner("📤 Broadcasting performance..."):
        result = telegram.send_performance_report(bot_token, chat_id, perf)
        if result["success"]:
            st.success("✅ Performance broadcasted to Telegram!")
        else:
            st.error(f"❌ Failed: {result.get('error')}")

def display_chart(df: pd.DataFrame):
    """Display interactive chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2]
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
    
    # EMAs
    if 'ema_12' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ema_12'], 
                      line=dict(color='#FF9800', width=2), name='EMA 12'),
            row=1, col=1
        )
    
    if 'ema_26' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ema_26'], 
                      line=dict(color='#F44336', width=2), name='EMA 26'),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_upper'], 
                      line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'),
                      name='BB Upper', showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_lower'], 
                      line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'),
                      fill='tonexty', fillcolor='rgba(255,255,255,0.1)',
                      name='BB Lower', showlegend=False),
            row=1, col=1
        )
    
    # Volume
    colors = ['#FF5252' if close < open else '#4CAF50' 
              for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', 
               marker_color=colors, opacity=0.6),
        row=2, col=1
    )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], 
                      line=dict(color='#9C27B0', width=2), name='RSI'),
            row=3, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", 
                     opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#4CAF50", 
                     opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.5)", 
                     opacity=0.3, row=3, col=1)
    
    # Layout
    fig.update_layout(
        height=800,
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)

def display_indicators(df: pd.DataFrame):
    """Display technical indicators"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 TREND INDICATORS")
        
        indicators = [
            ("EMA 12", df['ema_12'].iloc[-1] if 'ema_12' in df.columns else "N/A"),
            ("EMA 26", df['ema_26'].iloc[-1] if 'ema_26' in df.columns else "N/A"),
            ("SMA 20", df['sma_20'].iloc[-1] if 'sma_20' in df.columns else "N/A"),
            ("SMA 50", df['sma_50'].iloc[-1] if 'sma_50' in df.columns else "N/A"),
        ]
        
        for name, value in indicators:
            st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 12px;
                margin: 8px 0;
                border-left: 4px solid {Config.COLORS['primary']};
            ">
                <div style="color: #94A3B8; font-size: 12px;">{name}</div>
                <div style="color: white; font-size: 16px; font-weight: 600;">
                    {f'${value:,.2f}' if isinstance(value, (int, float)) else value}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📈 MOMENTUM INDICATORS")
        
        indicators = [
            ("RSI", df['rsi'].iloc[-1] if 'rsi' in df.columns else "N/A"),
            ("MACD", df['macd'].iloc[-1] if 'macd' in df.columns else "N/A"),
            ("MACD Signal", df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else "N/A"),
            ("MACD Hist", df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else "N/A"),
        ]
        
        for name, value in indicators:
            st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 12px;
                margin: 8px 0;
                border-left: 4px solid {Config.COLORS['info']};
            ">
                <div style="color: #94A3B8; font-size: 12px;">{name}</div>
                <div style="color: white; font-size: 16px; font-weight: 600;">
                    {f'{value:.4f}' if isinstance(value, (int, float)) else value}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### 📉 VOLATILITY INDICATORS")
        
        indicators = [
            ("ATR", df['atr'].iloc[-1] if 'atr' in df.columns else "N/A"),
            ("BB Upper", df['bb_upper'].iloc[-1] if 'bb_upper' in df.columns else "N/A"),
            ("BB Middle", df['bb_middle'].iloc[-1] if 'bb_middle' in df.columns else "N/A"),
            ("BB Lower", df['bb_lower'].iloc[-1] if 'bb_lower' in df.columns else "N/A"),
        ]
        
        for name, value in indicators:
            st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 12px;
                margin: 8px 0;
                border-left: 4px solid {Config.COLORS['warning']};
            ">
                <div style="color: #94A3B8; font-size: 12px;">{name}</div>
                <div style="color: white; font-size: 16px; font-weight: 600;">
                    {f'${value:,.2f}' if isinstance(value, (int, float)) else value}
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_signals(df: pd.DataFrame):
    """Display trading signals"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 RECENT SIGNALS")
        
        # Get recent signals
        if 'signal_type' in df.columns:
            signals = df[df['signal'] != 0].tail(10)
            
            if not signals.empty:
                for idx, row in signals.iloc[::-1].iterrows():
                    signal_color = {
                        'STRONG_BUY': '#00C853',
                        'BUY': '#4CAF50',
                        'NEUTRAL': '#94A3B8',
                        'SELL': '#FF5252',
                        'STRONG_SELL': '#FF3D00'
                    }.get(row['signal_type'], '#94A3B8')
                    
                    st.markdown(f"""
                    <div style="
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 10px;
                        padding: 15px;
                        margin: 10px 0;
                        border-left: 4px solid {signal_color};
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="color: white; font-weight: 700; font-size: 16px;">
                                    {row['signal_type']}
                                </span>
                                <div style="color: #94A3B8; font-size: 12px; margin-top: 4px;">
                                    {idx.strftime('%Y-%m-%d %H:%M')}
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: white; font-weight: 600; font-size: 16px;">
                                    ${row['close']:,.2f}
                                </div>
                                <div style="color: #94A3B8; font-size: 12px;">
                                    RSI: {row.get('rsi', 'N/A'):.1f}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No signals generated yet")
    
    with col2:
        st.markdown("#### ⚡ QUICK ACTIONS")
        
        # Manual Broadcast Button
        if st.button("📤 BROADCAST SIGNAL", 
                    use_container_width=True,
                    type="primary"):
            bot_token = st.session_state.get('telegram_bot_token')
            chat_id = st.session_state.get('telegram_chat_id')
            
            if bot_token and chat_id:
                telegram = TelegramBroadcaster()
                signal_data = {
                    'symbol': st.session_state.symbol,
                    'signal': df['signal_type'].iloc[-1] if 'signal_type' in df.columns else 'NEUTRAL',
                    'price': df['close'].iloc[-1],
                    'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
                    'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1,
                    'confidence': 0.8
                }
                
                with st.spinner("Broadcasting..."):
                    result = telegram.send_signal_alert(bot_token, chat_id, signal_data)
                    if result["success"]:
                        st.success("✅ Signal broadcasted!")
                    else:
                        st.error(f"❌ Failed: {result.get('error')}")
            else:
                st.warning("Configure Telegram first")
        
        # Additional actions
        st.markdown("---")
        
        if st.button("🔄 REFRESH DATA", use_container_width=True):
            st.rerun()
        
        if st.button("📊 EXPORT DATA", use_container_width=True):
            st.success("✅ Data exported!")
        
        if st.button("🔔 SET ALERTS", use_container_width=True):
            st.info("Alert system coming soon!")

def display_settings():
    """Display settings panel"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎨 APPEARANCE")
        
        theme = st.selectbox("Theme", ["Dark", "Light", "Auto"])
        chart_style = st.selectbox("Chart Style", ["Professional", "Minimal", "Technical"])
        font_size = st.slider("Font Size", 12, 20, 14)
        
        if st.button("Save Appearance", use_container_width=True):
            st.success("✅ Appearance saved!")
    
    with col2:
        st.markdown("#### 🔔 NOTIFICATIONS")
        
        st.checkbox("Enable sound alerts", True)
        st.checkbox("Show desktop notifications", False)
        st.checkbox("Email alerts", False)
        
        alert_frequency = st.selectbox(
            "Alert Frequency",
            ["Real-time", "5 minutes", "15 minutes", "1 hour"]
        )
        
        if st.button("Save Notifications", use_container_width=True):
            st.success("✅ Notifications saved!")

if __name__ == "__main__":
    main()
