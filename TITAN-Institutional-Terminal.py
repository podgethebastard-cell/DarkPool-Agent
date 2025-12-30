import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import requests
import sqlite3
import time
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import threading

# ==========================================
# 0. CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="TITAN | Institutional Terminal",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Bloomberg Terminal" Aesthetic
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .metric-card { background: #111; border: 1px solid #333; padding: 15px; border-radius: 4px; }
    h1, h2, h3 { color: #FFA500; font-weight: 900; letter-spacing: -1px; }
    .stButton>button { background-color: #333; color: #FFA500; border: 1px solid #FFA500; text-transform: uppercase; }
    .stButton>button:hover { background-color: #FFA500; color: black; }
    div[data-testid="stExpander"] { background-color: #0a0a0a; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

DB_NAME = "titan_signals.db"

# ==========================================
# 1. DATABASE LAYER (PERSISTENCE)
# ==========================================
class DatabaseManager:
    def __init__(self, db_name=DB_NAME):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ticker TEXT,
                strategy TEXT,
                signal TEXT,
                price REAL,
                ai_analysis TEXT
            )
        """)
        self.conn.commit()

    def save_signal(self, ticker, strategy, signal, price, ai_analysis):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO signals (ticker, strategy, signal, price, ai_analysis) VALUES (?, ?, ?, ?, ?)",
                       (ticker, strategy, signal, price, ai_analysis))
        self.conn.commit()

    def get_history(self):
        return pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 50", self.conn)

# Initialize DB
db = DatabaseManager()

# ==========================================
# 2. ADVANCED MATH & INDICATORS LIB
# ==========================================
class QuantitativeLib:
    """Institutional Grade Indicator Calculation Engine"""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
        
        # ATR (Volatility)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Ichimoku Cloud (Complex)
        nine_period_high = df['High'].rolling(window=9).max()
        nine_period_low = df['Low'].rolling(window=9).min()
        df['Tenkan_sen'] = (nine_period_high + nine_period_low) / 2
        
        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()
        df['Kijun_sen'] = (period26_high + period26_low) / 2
        
        df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
        
        period52_high = df['High'].rolling(window=52).max()
        period52_low = df['Low'].rolling(window=52).min()
        df['Senkou_Span_B'] = ((period52_high + period52_low) / 2).shift(26)
        
        df['Chikou_Span'] = df['Close'].shift(-26)
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Volume VWAP approximation
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

        return df

# ==========================================
# 3. PATTERN RECOGNITION ENGINE
# ==========================================
class PatternRecognition:
    @staticmethod
    def identify_patterns(df: pd.DataFrame) -> List[str]:
        patterns = []
        if len(df) < 5: return patterns
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. Doji
        body_size = abs(curr['Open'] - curr['Close'])
        total_size = curr['High'] - curr['Low']
        if total_size > 0 and (body_size / total_size) < 0.1:
            patterns.append("Doji (Indecision)")
            
        # 2. Bullish Engulfing
        if (prev['Close'] < prev['Open']) and (curr['Close'] > curr['Open']):
            if (curr['Open'] < prev['Close']) and (curr['Close'] > prev['Open']):
                patterns.append("Bullish Engulfing")

        # 3. Hammer
        lower_shadow = min(curr['Open'], curr['Close']) - curr['Low']
        if total_size > 0 and (lower_shadow / total_size) > 0.6 and body_size < (0.2 * total_size):
            patterns.append("Hammer (Reversal)")
            
        return patterns

# ==========================================
# 4. STRATEGY INTERFACE & IMPLEMENTATIONS
# ==========================================
class Strategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Tuple[str, float, List[str]]:
        pass

class TrendFollowingStrategy(Strategy):
    def analyze(self, df):
        curr = df.iloc[-1]
        signal = "NEUTRAL"
        score = 0
        reasons = []
        
        # Trend check
        if curr['SMA_50'] > curr['SMA_200']:
            score += 1
            reasons.append("Golden Cross Structure (Bullish)")
        elif curr['SMA_50'] < curr['SMA_200']:
            score -= 1
            reasons.append("Death Cross Structure (Bearish)")
            
        # Ichimoku Confirmation
        if curr['Close'] > curr['Senkou_Span_A'] and curr['Close'] > curr['Senkou_Span_B']:
            score += 1
            reasons.append("Price Above Ichimoku Cloud")
        
        # MACD
        if curr['MACD'] > curr['MACD_Signal']:
            score += 1
            reasons.append("MACD Momentum Positive")
            
        if score >= 2: signal = "STRONG BUY"
        elif score == 1: signal = "BUY"
        elif score == -1: signal = "SELL"
        elif score <= -2: signal = "STRONG SELL"
        
        return signal, score, reasons

class MeanReversionStrategy(Strategy):
    def analyze(self, df):
        curr = df.iloc[-1]
        signal = "NEUTRAL"
        score = 0
        reasons = []
        
        # RSI Extremes
        if curr['RSI'] < 30:
            score += 2
            reasons.append(f"RSI Deep Oversold ({curr['RSI']:.1f})")
        elif curr['RSI'] > 70:
            score -= 2
            reasons.append(f"RSI Deep Overbought ({curr['RSI']:.1f})")
            
        # Bollinger Bands
        if curr['Close'] < curr['BB_Lower']:
            score += 1
            reasons.append("Price below Lower BB (Statistical Discount)")
        elif curr['Close'] > curr['BB_Upper']:
            score -= 1
            reasons.append("Price above Upper BB (Statistical Premium)")
            
        if score >= 2: signal = "LONG REVERSAL"
        elif score <= -2: signal = "SHORT REVERSAL"
        
        return signal, score, reasons

# ==========================================
# 5. BACKTESTING ENGINE
# ==========================================
class Backtester:
    def __init__(self, df, strategy_logic):
        self.df = df
        self.logic = strategy_logic
        
    def run(self):
        """Vectorized Backtest Simulation"""
        df = self.df.copy()
        df['Signal'] = 0
        df['Strategy_Return'] = 0.0
        
        # Simple Logic Simulation for performance
        # Buy when RSI < 30, Sell when RSI > 70 (Example)
        df.loc[df['RSI'] < 30, 'Signal'] = 1
        df.loc[df['RSI'] > 70, 'Signal'] = -1
        
        df['Market_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)
        
        df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
        
        total_return = df['Cumulative_Strategy'].iloc[-1] - 1
        sharpe = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252)
        
        # Max Drawdown
        cum_ret = df['Cumulative_Strategy']
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Equity Curve": df['Cumulative_Strategy']
        }

# ==========================================
# 6. AI & TELEGRAM HANDLERS
# ==========================================
class Broadcaster:
    @staticmethod
    def generate_ai_message(ticker, signal, reasons, patterns, context):
        api_key = st.secrets.get("openai", {}).get("api_key")
        if not api_key: return "⚠️ OPENAI KEY MISSING"
        
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Act as a Wall Street Quant Analyst. Write a Telegram alert for {ticker}.
        
        DATA:
        - Signal: {signal}
        - Technical Reasons: {', '.join(reasons)}
        - Candlestick Patterns: {', '.join(patterns)}
        - Price: {context['Close']:.2f}
        - ATR (Volatility): {context['ATR']:.2f}
        - Ichimoku Status: {'Bullish' if context['Close'] > context['Senkou_Span_A'] else 'Bearish'}
        
        INSTRUCTIONS:
        1. Professional, concise, authoritative tone.
        2. Calculate a Stop Loss at Price - (1.5 * ATR).
        3. Calculate a Take Profit at Price + (3 * ATR).
        4. Use emojis.
        5. DO NOT mention you are an AI.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI Error: {e}"

    @staticmethod
    def send_telegram(message):
        token = st.secrets.get("telegram", {}).get("bot_token")
        cid = st.secrets.get("telegram", {}).get("channel_id")
        if not token or not cid: return False, "Missing Credentials"
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            res = requests.post(url, json={"chat_id": cid, "text": message, "parse_mode": "Markdown"})
            return res.status_code == 200, res.text
        except Exception as e:
            return False, str(e)

# ==========================================
# 7. MAIN APP LOGIC & UI
# ==========================================
def main():
    st.sidebar.title("TITAN TERMINAL")
    st.sidebar.markdown("---")
    
    # --- Sidebar Inputs ---
    ticker = st.sidebar.text_input("SYMBOL", "BTC-USD").upper()
    timeframe = st.sidebar.selectbox("TIMEFRAME", ["15m", "1h", "4h", "1d", "1wk"], index=3)
    strategy_select = st.sidebar.selectbox("STRATEGY MODEL", ["Trend Following", "Mean Reversion"])
    
    # --- Main Data Loading ---
    period_map = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "2y", "1wk": "5y"}
    
    if ticker:
        with st.spinner(f"Connecting to Exchange for {ticker}..."):
            df = yf.download(ticker, period=period_map[timeframe], interval=timeframe, progress=False)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # 1. Process Data
            df = QuantitativeLib.add_all_indicators(df)
            patterns = PatternRecognition.identify_patterns(df)
            
            # 2. Run Strategy
            if strategy_select == "Trend Following":
                strat = TrendFollowingStrategy()
            else:
                strat = MeanReversionStrategy()
            
            signal, score, reasons = strat.analyze(df)
            
            # 3. UI Layout
            tab1, tab2, tab3, tab4 = st.tabs(["📊 DASHBOARD", "🧪 BACKTEST", "🤖 AI BROADCAST", "💾 HISTORY"])
            
            # --- TAB 1: DASHBOARD ---
            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                curr_price = df['Close'].iloc[-1]
                col1.metric("PRICE", f"{curr_price:.2f}", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")
                col2.metric("SIGNAL", signal, delta_color="off")
                col3.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
                col4.metric("VOLATILITY (ATR)", f"{df['ATR'].iloc[-1]:.2f}")
                
                # Charts
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                
                # Main Candle Stick
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
                
                # Indicators
                fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_Span_A'], fill=None, mode='lines', line_color='rgba(0,255,0,0.3)', name='Senkou A'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_Span_B'], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.3)', name='Cloud'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(dash='dot', color='gray'), name='BB Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(dash='dot', color='gray'), name='BB Lower'), row=1, col=1)
                
                # Subplot Indicators
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist', marker_color='teal'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='white')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
                
                fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Pattern Display
                if patterns:
                    st.warning(f"Detected Patterns: {', '.join(patterns)}")
            
            # --- TAB 2: BACKTEST ---
            with tab2:
                st.subheader("Strategy Validation (Vectorized Simulation)")
                if st.button("Run Backtest Simulation"):
                    bt = Backtester(df, strategy_select)
                    results = bt.run()
                    
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Est. Total Return", f"{results['Total Return']*100:.2f}%")
                    b2.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")
                    b3.metric("Max Drawdown", f"{results['Max Drawdown']*100:.2f}%")
                    
                    st.area_chart(results['Equity Curve'])
                    st.caption("Note: This is a simplified vector backtest for indicative purposes only.")
            
            # --- TAB 3: AI BROADCAST ---
            with tab3:
                st.subheader("📢 Institutional Broadcast Center")
                
                col_gen, col_send = st.columns(2)
                
                if 'msg_cache' not in st.session_state: st.session_state['msg_cache'] = ""
                
                with col_gen:
                    if st.button("GENERATE ALPHA REPORT"):
                        with st.spinner("Analyzing Market Microstructure..."):
                            context = df.iloc[-1].to_dict()
                            msg = Broadcaster.generate_ai_message(ticker, signal, reasons, patterns, context)
                            st.session_state['msg_cache'] = msg
                
                with col_send:
                    if st.session_state['msg_cache']:
                        st.text_area("Preview", st.session_state['msg_cache'], height=300)
                        if st.button("CONFIRM BROADCAST"):
                            success, log = Broadcaster.send_telegram(st.session_state['msg_cache'])
                            if success:
                                st.success("Broadcast Sent!")
                                db.save_signal(ticker, strategy_select, signal, curr_price, st.session_state['msg_cache'])
                            else:
                                st.error(f"Failed: {log}")
            
            # --- TAB 4: HISTORY ---
            with tab4:
                st.subheader("Signal Database")
                history = db.get_history()
                st.dataframe(history, use_container_width=True)

if __name__ == "__main__":
    main()
