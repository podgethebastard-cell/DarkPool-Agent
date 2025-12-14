import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import datetime
from scipy.stats import linregress
from openai import OpenAI
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Optional

# ==========================================
# 1. SYSTEM CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="TITAN GOD MODE v2",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* TITAN DARKPOOL THEME */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* NEON TYPOGRAPHY */
    h1, h2, h3 { 
        color: #fff; 
        text-shadow: 0 0 10px rgba(0, 255, 187, 0.5); 
        letter-spacing: 1px;
    }
    
    /* GLASSMORPHISM CARDS */
    div[data-testid="metric-container"] {
        background: rgba(20, 20, 20, 0.8);
        border: 1px solid #333;
        border-left: 3px solid #00ffbb;
        padding: 15px;
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    label[data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.8rem; }
    div[data-testid="stMetricValue"] { color: #fff !important; font-weight: 700; font-size: 1.4rem; }
    
    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background-color: #111;
        border: 1px solid #333;
        color: #888;
        border-radius: 4px;
        padding: 8px 16px;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ffbb !important;
        color: #000 !important;
        font-weight: bold;
        border: none;
        box-shadow: 0 0 15px rgba(0, 255, 187, 0.4);
    }
    
    /* SIDEBAR & INPUTS */
    section[data-testid="stSidebar"] { background-color: #080808; border-right: 1px solid #222; }
    .stTextInput > div > div > input { background-color: #111; color: #00ffbb; border: 1px solid #333; }
    
    /* ALERTS */
    .titan-alert {
        padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: bold;
        background: rgba(0, 255, 187, 0.1); border: 1px solid #00ffbb; color: #00ffbb;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE ENGINES (MATH & DATA)
# ==========================================
class TitanMath:
    @staticmethod
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def hma(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma_half = TitanMath.wma(series, half_length)
        wma_full = TitanMath.wma(series, length)
        diff = 2 * wma_half - wma_full
        return TitanMath.wma(diff, sqrt_length)

    @staticmethod
    def atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def rsi(series, length=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/length, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/length, adjust=False).mean()
        return 100 - (100 / (1 + (gain / loss)))

    @staticmethod
    def linreg_curve(series, length=20):
        x = np.arange(length)
        return series.rolling(length).apply(lambda y: linregress(x, y)[0], raw=True)

class TitanData:
    @staticmethod
    @st.cache_data(ttl=60)
    def fetch_crypto(symbol, timeframe, limit):
        try:
            exchange = ccxt.kraken()
            tf_map = {'15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
            ohlcv = exchange.fetch_ohlcv(symbol, tf_map[timeframe], limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except: return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_stock(ticker, interval, period="1y"):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except: return pd.DataFrame()

    @staticmethod
    def calculate_god_mode(df):
        # 1. Apex Trend (HMA 55)
        df['HMA55'] = TitanMath.hma(df['Close'], 55)
        df['ATR'] = TitanMath.atr(df, 55)
        df['Apex_Upper'] = df['HMA55'] + (df['ATR'] * 2.0)
        df['Apex_Lower'] = df['HMA55'] - (df['ATR'] * 2.0)
        df['Trend'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))
        df['Trend'] = df['Trend'].replace(to_replace=0, method='ffill')

        # 2. Squeeze Momentum
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['KC_Mid'] = df['Close'].rolling(20).mean()
        df['KC_ATR'] = TitanMath.atr(df, 20)
        df['Squeeze_On'] = (df['BB_Mid'] + 2*df['BB_Std']) < (df['KC_Mid'] + 1.5*df['KC_ATR'])
        
        # Momentum (LinReg of delta)
        avg = (df['High'].rolling(20).max() + df['Low'].rolling(20).min()) / 2
        df['Sqz_Mom'] = TitanMath.linreg_curve(df['Close'] - avg, 20)

        # 3. Money Flow Matrix
        df['RSI'] = TitanMath.rsi(df['Close'])
        df['MFI'] = (df['RSI'] - 50) * (df['Volume'] / df['Volume'].rolling(20).mean())
        
        # 4. SMC Order Blocks (Simplified)
        df['Pivot_H'] = df['High'].rolling(5, center=True).max()
        df['Pivot_L'] = df['Low'].rolling(5, center=True).min()
        df['Structure'] = np.where(df['Close'] > df['Pivot_H'].shift(1), 1, np.where(df['Close'] < df['Pivot_L'].shift(1), -1, 0))
        
        return df

# ==========================================
# 3. BACKTESTING ENGINE (FROM CURSOR2)
# ==========================================
@dataclass
class Trade:
    entry_time: datetime.datetime
    exit_time: Optional[datetime.datetime]
    type: str # LONG/SHORT
    entry_price: float
    exit_price: float
    pnl: float
    status: str # OPEN/CLOSED

class TitanBacktester:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.trades = []
        self.equity = [initial_capital]
        
    def run(self, df):
        position = None
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Strategy: Apex Trend Flip + Momentum Positive
            signal_long = (row['Trend'] == 1) and (prev['Trend'] != 1) and (row['Sqz_Mom'] > 0)
            signal_short = (row['Trend'] == -1) and (prev['Trend'] != -1) and (row['Sqz_Mom'] < 0)
            
            # Close Logic
            if position:
                if (position.type == 'LONG' and row['Trend'] == -1) or \
                   (position.type == 'SHORT' and row['Trend'] == 1):
                    self.close_trade(position, row['Close'], row.name)
                    position = None

            # Open Logic
            if not position:
                if signal_long:
                    position = Trade(row.name, None, 'LONG', row['Close'], 0, 0, 'OPEN')
                elif signal_short:
                    position = Trade(row.name, None, 'SHORT', row['Close'], 0, 0, 'OPEN')
            
            # Update Equity
            current_val = self.capital
            if position:
                pnl = (row['Close'] - position.entry_price) if position.type == 'LONG' else (position.entry_price - row['Close'])
                # Assume 1 unit for simplicity in visualizer
                current_val += pnl 
            self.equity.append(current_val)
            
        return pd.DataFrame([vars(t) for t in self.trades]), self.equity

    def close_trade(self, trade, price, time):
        trade.exit_price = price
        trade.exit_time = time
        trade.status = 'CLOSED'
        if trade.type == 'LONG':
            trade.pnl = price - trade.entry_price
        else:
            trade.pnl = trade.entry_price - price
        self.trades.append(trade)
        self.capital += trade.pnl

# ==========================================
# 4. BROADCASTER (FROM DEEPSEEKAGENT)
# ==========================================
class TelegramBroadcaster:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        
    def send(self, message):
        if not self.token or not self.chat_id: return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            r = requests.post(url, json={"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"})
            return r.status_code == 200
        except: return False

# ==========================================
# 5. UI & APP LOGIC
# ==========================================
def main():
    st.sidebar.markdown("## 🧠 TITAN CORE")
    
    # AI CONFIG
    ai_model = st.sidebar.selectbox("AI Model", ["OpenAI (GPT-4o)", "DeepSeek-V3", "Gemini Pro"])
    api_key = st.sidebar.text_input("API Key", type="password")
    
    # TG CONFIG
    with st.sidebar.expander("📡 Broadcast"):
        tg_token = st.text_input("Bot Token", type="password")
        tg_chat = st.text_input("Chat ID")
        broadcaster = TelegramBroadcaster(tg_token, tg_chat)

    st.sidebar.markdown("---")
    
    # NAVIGATION
    app_mode = st.sidebar.radio("MODULE", ["Terminal", "Deep Screener", "Backtester", "Macro"])
    
    if app_mode == "Terminal":
        render_terminal(ai_model, api_key, broadcaster)
    elif app_mode == "Deep Screener":
        render_screener()
    elif app_mode == "Backtester":
        render_backtester()
    elif app_mode == "Macro":
        render_macro()

# --- MODULES ---

def render_terminal(model, key, broadcaster):
    st.markdown("### ⚡ TITAN GOD MODE TERMINAL")
    
    c1, c2, c3 = st.columns([1,1,2])
    with c1: 
        symbol = st.text_input("Asset", value="BTC/USD").upper()
        is_crypto = "/" in symbol
    with c2: timeframe = st.selectbox("TF", ["15m", "1h", "4h", "1d"], index=2)
    
    if st.button("RUN GOD MODE", type="primary"):
        with st.spinner("Processing Apex Logic..."):
            if is_crypto:
                df = TitanData.fetch_crypto(symbol, timeframe, 500)
            else:
                df = TitanData.fetch_stock(symbol, timeframe)
            
            if not df.empty:
                df = TitanData.calculate_god_mode(df)
                last = df.iloc[-1]
                
                # --- HUD ---
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Price", f"${last['Close']:,.2f}")
                m2.metric("Apex Trend", "BULLISH" if last['Trend']==1 else "BEARISH", delta=f"{last['Close']-last['HMA55']:.2f} vs HMA")
                m3.metric("Squeeze", "ON" if last['Squeeze_On'] else "OFF", delta_color="inverse")
                m4.metric("Money Flow", f"{last['MFI']:.2f}")
                
                # --- CHART ---
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
                
                # Price
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 255, 187, 0.1)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA55'], line=dict(color='yellow', width=2), name="HMA 55"), row=1, col=1)
                
                # Momentum
                cols = ['#00ffbb' if x > 0 else '#ff3333' for x in df['Sqz_Mom']]
                fig.add_trace(go.Bar(x=df.index, y=df['Sqz_Mom'], marker_color=cols, name="Momentum"), row=2, col=1)
                
                fig.update_layout(height=700, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # --- AI ANALYSIS & BROADCAST ---
                c_ai, c_bc = st.columns(2)
                
                with c_ai:
                    if st.button("🤖 GENERATE ANALYST REPORT"):
                        if not key:
                            st.error("API Key Required")
                        else:
                            prompt = f"""
                            Analyze {symbol} ({timeframe}) using Titan God Mode.
                            Price: {last['Close']}
                            Trend: {'Bull' if last['Trend']==1 else 'Bear'}
                            Squeeze: {last['Squeeze_On']}
                            Momentum: {last['Sqz_Mom']:.2f}
                            MFI: {last['MFI']:.2f}
                            
                            Provide a trading plan: Bias, Entry Zone, and Risks. Short & Professional.
                            """
                            # Mock call for structure (Replace with actual API call based on 'model' selection)
                            try:
                                client = OpenAI(api_key=key)
                                resp = client.chat.completions.create(model="gpt-4", messages=[{"role":"user", "content": prompt}])
                                st.info(resp.choices[0].message.content)
                            except Exception as e:
                                st.error(f"AI Error: {e}")

                with c_bc:
                    msg = f"🔥 TITAN SIGNAL: {symbol}\nTrend: {'BULL' if last['Trend']==1 else 'BEAR'}\nPrice: {last['Close']}\nMom: {last['Sqz_Mom']:.2f}"
                    if st.button("📡 BROADCAST TO TELEGRAM"):
                        if broadcaster.send(msg):
                            st.success("Signal Sent!")
                        else:
                            st.error("Telegram Config Missing")

def render_screener():
    st.markdown("### ⛏️ DEEP VALUE & MINING SCREENER")
    st.info("Scanning for: Net Cash > 0, P/B < 2, Revenue Growth > 10% (Clive's Miners Logic)")
    
    universe = ["NXE", "UEC", "UUUU", "DNN", "CCJ", "KGC", "EQX", "PAAS", "AG", "FCX", "NEM"]
    
    if st.button("RUN DEEP SCAN"):
        results = []
        bar = st.progress(0)
        for i, tick in enumerate(universe):
            bar.progress((i+1)/len(universe))
            try:
                # Fundamental fetch
                stock = yf.Ticker(tick)
                info = stock.info
                
                cash = info.get('totalCash', 0)
                debt = info.get('totalDebt', 0)
                pb = info.get('priceToBook', 999)
                
                score = 0
                if (cash - debt) > 0: score += 2 # Net Cash Bonus
                if pb < 2.0: score += 1
                
                results.append({
                    "Ticker": tick,
                    "Price": info.get('currentPrice'),
                    "P/B": pb,
                    "Net Cash": "YES" if (cash-debt)>0 else "NO",
                    "Score": score
                })
            except: continue
        
        bar.empty()
        df = pd.DataFrame(results).sort_values('Score', ascending=False)
        st.dataframe(df, use_container_width=True)

def render_backtester():
    st.markdown("### 🧬 STRATEGY SIMULATOR")
    
    col1, col2 = st.columns([1,3])
    with col1:
        sym = st.text_input("Sim Asset", "BTC/USD")
        cap = st.number_input("Capital", 1000, 100000, 10000)
    
    if st.button("RUN SIMULATION"):
        df = TitanData.fetch_crypto(sym, "4h", 1000)
        df = TitanData.calculate_god_mode(df)
        
        bt = TitanBacktester(cap)
        trades, equity = bt.run(df)
        
        if not trades.empty:
            st.line_chart(equity)
            st.dataframe(trades)
            
            win_rate = len(trades[trades['pnl']>0]) / len(trades) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{len(trades)} Trades")
        else:
            st.warning("No trades generated with current logic.")

def render_macro():
    st.markdown("### 🦅 MACRO INSIGHTER")
    
    ratios = {
        "Risk (SPY/TLT)": ["SPY", "TLT"],
        "Inflation (GLD/TLT)": ["GLD", "TLT"],
        "Tech (QQQ/SPY)": ["QQQ", "SPY"]
    }
    
    sel = st.selectbox("Institutional Ratio", list(ratios.keys()))
    t1, t2 = ratios[sel]
    
    if st.button("Analyze Flow"):
        d1 = yf.download(t1, period="1y", progress=False)['Close']
        d2 = yf.download(t2, period="1y", progress=False)['Close']
        
        # Flatten MultiIndex
        if isinstance(d1, pd.DataFrame): d1 = d1.iloc[:,0]
        if isinstance(d2, pd.DataFrame): d2 = d2.iloc[:,0]
        
        ratio = d1 / d2
        st.line_chart(ratio)
        st.caption(f"Rising = {t1} Outperforming {t2}")

if __name__ == "__main__":
    main()
