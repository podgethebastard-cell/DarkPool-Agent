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
# 1. SYSTEM CONFIGURATION & DARKPOOL CSS
# ==========================================
st.set_page_config(
    page_title="TITAN OMNI v4.0",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* TITAN DARKPOOL THEME v4 */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* NEON HEADERS */
    h1, h2, h3 { 
        color: #fff; 
        text-shadow: 0 0 15px rgba(0, 255, 187, 0.6); 
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* HUD METRIC CARDS */
    div[data-testid="metric-container"] {
        background: rgba(15, 15, 15, 0.9);
        border: 1px solid #333;
        border-left: 3px solid #00ffbb;
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-left-color: #00b887;
    }
    label[data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.85rem; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { color: #fff !important; font-weight: 700; font-size: 1.5rem; }
    
    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background-color: #0a0a0a;
        border: 1px solid #333;
        color: #666;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ffbb !important;
        color: #000 !important;
        border: none;
        box-shadow: 0 0 15px rgba(0, 255, 187, 0.4);
    }
    
    /* SIDEBAR & INPUTS */
    section[data-testid="stSidebar"] { background-color: #020202; border-right: 1px solid #222; }
    .stTextInput > div > div > input { background-color: #111; color: #00ffbb; border: 1px solid #333; }
    .stSelectbox > div > div > div { background-color: #111; color: #fff; border: 1px solid #333; }
    .stMultiSelect > div > div > div { background-color: #111; color: #fff; }
    
    /* BUTTONS */
    button[kind="primary"] {
        background: linear-gradient(90deg, #00ffbb, #008f6b);
        color: #000;
        border: none;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 0.5rem 1rem;
    }
    button[kind="secondary"] { background: #111; color: #ccc; border: 1px solid #333; }
    
    /* ALERTS */
    .titan-alert {
        padding: 12px; border-radius: 6px; margin-bottom: 15px; font-weight: bold;
        background: rgba(0, 255, 187, 0.05); border: 1px solid #00ffbb; color: #00ffbb;
        text-align: center; letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. TITAN MATH ENGINE (The Brain)
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
    def rma(series, length):
        return series.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def atr(df, length=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return TitanMath.rma(tr, length)

    @staticmethod
    def rsi(series, length=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/length, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/length, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def adx(df, length=14):
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = -df['Low'].diff().clip(upper=0)
        plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
        minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
        
        tr = TitanMath.atr(df, length)
        plus_di = 100 * TitanMath.rma(pd.Series(plus_dm), length) / tr
        minus_di = 100 * TitanMath.rma(pd.Series(minus_dm), length) / tr
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return TitanMath.rma(dx, length)

    @staticmethod
    def wavetrend(df, chlen=10, avg=21):
        """From ApexCryptoSMCScout"""
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ap.ewm(span=chlen).mean()
        d = (ap - esa).abs().ewm(span=chlen).mean()
        ci = (ap - esa) / (0.015 * d)
        tci = ci.ewm(span=avg).mean()
        return tci

    @staticmethod
    def linreg_slope(series, length=20):
        x = np.arange(length)
        return series.rolling(length).apply(lambda y: linregress(x, y)[0], raw=True)

# ==========================================
# 3. DATA & LOGIC LAYER
# ==========================================
class TitanData:
    @staticmethod
    @st.cache_data(ttl=60)
    def fetch_market_data(symbol, interval, limit=500, source="Crypto (CCXT)"):
        try:
            if source == "Crypto (CCXT)":
                exchange = ccxt.kraken()
                # Map timeframe
                tf_map = {'15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
                ohlcv = exchange.fetch_ohlcv(symbol, tf_map[interval], limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                # Yahoo Finance
                df = yf.download(symbol, period="2y", interval=interval, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception as e:
            return pd.DataFrame()

    @staticmethod
    def calculate_strategies(df, params, active_strats):
        """
        The Consensus Engine: Runs selected strategies and aggregates a 'God Score'.
        """
        if df.empty: return df
        
        # --- BASE INDICATORS ---
        df['ATR'] = TitanMath.atr(df, 14)
        df['RSI'] = TitanMath.rsi(df['Close'], 14)
        df['ADX'] = TitanMath.adx(df, 14)
        
        # --- 1. APEX TREND (HMA) ---
        if "Apex Trend" in active_strats:
            df['HMA'] = TitanMath.hma(df['Close'], params['hma_len'])
            df['Apex_Upper'] = df['HMA'] + (df['ATR'] * params['atr_mult'])
            df['Apex_Lower'] = df['HMA'] - (df['ATR'] * params['atr_mult'])
            df['Sig_Apex'] = np.where(df['Close'] > df['Apex_Upper'], 1, np.where(df['Close'] < df['Apex_Lower'], -1, 0))
        else:
            df['Sig_Apex'] = 0

        # --- 2. DARK VECTOR (Structure Trail) ---
        if "Dark Vector" in active_strats:
            amp = params['amp']
            df['LL'] = df['Low'].rolling(amp).min()
            df['HH'] = df['High'].rolling(amp).max()
            # Simplified Vector Logic for vectorization speed
            df['Sig_Vector'] = np.where(df['Close'] > df['HH'].shift(1), 1, np.where(df['Close'] < df['LL'].shift(1), -1, 0))
        else:
            df['Sig_Vector'] = 0

        # --- 3. WAVETREND (Momentum) ---
        if "WaveTrend" in active_strats:
            df['WT'] = TitanMath.wavetrend(df)
            df['Sig_WT'] = np.where(df['WT'] > 60, -1, np.where(df['WT'] < -60, 1, 0)) # Contrarian signals at extremes
        else:
            df['Sig_WT'] = 0

        # --- 4. SQUEEZE (Volatility) ---
        if "Squeeze" in active_strats:
            bb_mid = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            kc_atr = TitanMath.atr(df, 20)
            df['Squeeze_On'] = (bb_mid + 2*bb_std) < (bb_mid + 1.5*kc_atr)
            # Breakout signal
            df['Sig_Sqz'] = np.where((df['Squeeze_On'].shift(1)) & (~df['Squeeze_On']), np.sign(df['Close'].diff()), 0)
        else:
            df['Sig_Sqz'] = 0

        # --- 5. GANN HILO ---
        if "Gann HiLo" in active_strats:
            gl = params['gann_len']
            df['Gann_H'] = df['High'].rolling(gl).mean()
            df['Gann_L'] = df['Low'].rolling(gl).mean()
            df['Sig_Gann'] = np.where(df['Close'] > df['Gann_H'].shift(1), 1, np.where(df['Close'] < df['Gann_L'].shift(1), -1, 0))
        else:
            df['Sig_Gann'] = 0

        # --- GOD SCORE AGGREGATION ---
        # Sum active signals
        df['God_Score'] = df['Sig_Apex'] + df['Sig_Vector'] + df['Sig_WT'] + df['Sig_Gann'] + df['Sig_Sqz']
        
        # --- SMC DETECTION (Always On for Charts) ---
        df['Pivot_H'] = df['High'].rolling(5, center=True).max()
        df['Pivot_L'] = df['Low'].rolling(5, center=True).min()
        df['FVG_Bull'] = (df['Low'] > df['High'].shift(2))
        
        return df

# ==========================================
# 4. BACKTESTING ENGINE (Advanced)
# ==========================================
@dataclass
class Trade:
    entry_time: datetime.datetime
    type: str
    entry_price: float
    exit_price: float
    pnl: float
    reason: str

class TitanBacktester:
    def __init__(self, initial_capital):
        self.capital = initial_capital
        self.trades = []
        self.equity = [initial_capital]
        self.position = None 

    def run(self, df, threshold=1):
        # Event-driven loop
        for i in range(50, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Entry Logic: God Score crosses threshold
            long_sig = (row['God_Score'] >= threshold) and (prev['God_Score'] < threshold)
            short_sig = (row['God_Score'] <= -threshold) and (prev['God_Score'] > -threshold)
            
            # Exit Logic
            if self.position:
                if self.position['type'] == 'LONG':
                    if row['God_Score'] < 0: # Trend broken
                        self.close(row, 'Trend Rev')
                elif self.position['type'] == 'SHORT':
                    if row['God_Score'] > 0:
                        self.close(row, 'Trend Rev')
            
            # Open Logic
            if not self.position:
                if long_sig: self.open(row, 'LONG')
                elif short_sig: self.open(row, 'SHORT')
            
            # Mark to Market
            val = self.capital
            if self.position:
                pnl = (row['Close'] - self.position['entry']) if self.position['type'] == 'LONG' else (self.position['entry'] - row['Close'])
                val += pnl * (self.capital * 0.1 / row['Close']) # 10% sizing
            self.equity.append(val)
            
        return self.get_stats()

    def open(self, row, type):
        self.position = {'type': type, 'entry': row['Close'], 'time': row.name}

    def close(self, row, reason):
        pnl = (row['Close'] - self.position['entry']) if self.position['type'] == 'LONG' else (self.position['entry'] - row['Close'])
        # Simplified: PnL is raw price diff * size (1 unit for ease)
        realized_pnl = pnl * (self.capital * 0.1 / self.position['entry']) 
        self.capital += realized_pnl
        self.trades.append(Trade(self.position['time'], self.position['type'], self.position['entry'], row['Close'], realized_pnl, reason))
        self.position = None

    def get_stats(self):
        if not self.trades: return None
        df_t = pd.DataFrame([vars(t) for t in self.trades])
        equity_s = pd.Series(self.equity)
        
        # Metrics
        total_ret = (self.equity[-1] - self.equity[0]) / self.equity[0]
        win_rate = len(df_t[df_t['pnl'] > 0]) / len(df_t)
        
        # Sharpe
        returns = equity_s.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        
        # Drawdown
        roll_max = equity_s.cummax()
        drawdown = (equity_s - roll_max) / roll_max
        max_dd = drawdown.min()
        
        return {
            "trades": df_t,
            "equity": equity_s,
            "total_return": total_ret,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "max_dd": max_dd
        }

# ==========================================
# 5. UI COMPONENTS
# ==========================================
def render_terminal(api_key, model, strategies, params):
    st.markdown("### ⚡ TITAN GOD MODE: TERMINAL")
    
    c1, c2, c3 = st.columns([1,1,2])
    with c1: symbol = st.text_input("Asset", "BTC/USD").upper()
    with c2: timeframe = st.selectbox("TF", ["15m", "1h", "4h", "1d"], index=2)
    with c3: source = st.selectbox("Source", ["Crypto (CCXT)", "Stocks (Yahoo)"])
    
    if st.button("🚀 INITIATE SYSTEM", type="primary"):
        with st.spinner("Processing Consensus Engine..."):
            df = TitanData.fetch_market_data(symbol, timeframe, 1000, source)
            
            if not df.empty:
                df = TitanData.calculate_strategies(df, params, strategies)
                last = df.iloc[-1]
                
                # --- HUD ---
                score_col = "normal" if last['God_Score'] > 0 else "inverse"
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Price", f"${last['Close']:,.2f}")
                m2.metric("God Score", f"{last['God_Score']:.0f}", f"{len(strategies)} Active Engines", delta_color="off")
                m3.metric("Trend State", "BULLISH" if last['God_Score'] > 0 else "BEARISH", delta_color=score_col)
                m4.metric("Volatility", f"{last['ATR']:.2f}", f"RSI: {last['RSI']:.1f}")
                
                # --- ALERTS ---
                if last['FVG_Bull']: st.markdown('<div class="titan-alert">🏛️ BULLISH FVG DETECTED</div>', unsafe_allow_html=True)
                
                # --- CHART ---
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                
                # Main Price
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                
                # Add Active Overlays
                if "Apex Trend" in strategies:
                    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 255, 187, 0.1)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
                
                # Add Signals
                buys = df[df['God_Score'] >= params['threshold']]
                sells = df[df['God_Score'] <= -params['threshold']]
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', color='#00ffbb', size=10), name="Buy"), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', color='#ff1155', size=10), name="Sell"), row=1, col=1)

                # Subplot
                fig.add_trace(go.Bar(x=df.index, y=df['God_Score'], name="God Score"), row=2, col=1)
                
                fig.update_layout(height=700, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # --- BACKTEST OVERLAY ---
                with st.expander("🧬 LIVE STRATEGY SIMULATION"):
                    bt = TitanBacktester(10000)
                    stats = bt.run(df, params['threshold'])
                    if stats:
                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric("Net Profit", f"{stats['total_return']*100:.2f}%")
                        b2.metric("Win Rate", f"{stats['win_rate']*100:.1f}%")
                        b3.metric("Sharpe Ratio", f"{stats['sharpe']:.2f}")
                        b4.metric("Max Drawdown", f"{stats['max_dd']*100:.2f}%")
                        st.line_chart(stats['equity'])
                        st.dataframe(stats['trades'])
                    else:
                        st.warning("No trades found with current parameters.")

            else:
                st.error("Invalid Symbol or API Error.")

def render_screener():
    st.markdown("### 🧪 QUANT FACTOR SCREENER")
    
    mode = st.radio("Screening Logic", ["Clive's Deep Value (Mining)", "Titan Factor Growth (Tech)"])
    
    if mode == "Clive's Deep Value (Mining)":
        universe = ["NXE", "UEC", "UUUU", "DNN", "CCJ", "KGC", "EQX", "PAAS", "FCX", "NEM"]
        st.info("Searching for: Net Cash > 0, P/B < 2, Revenue Growth > 10% (Miners)")
    else:
        universe = ["AAPL", "MSFT", "NVDA", "AMD", "PLTR", "SNOW", "DDOG", "NET", "TSLA"]
        st.info("Searching for: Z-Score Value, Momentum, Quality > 50th Percentile")
        
    if st.button("RUN DEEP SCAN"):
        results = []
        bar = st.progress(0)
        
        for i, tic in enumerate(universe):
            bar.progress((i+1)/len(universe))
            try:
                stock = yf.Ticker(tic)
                info = stock.info
                if not info: continue
                
                # Fundamental Data
                data = {
                    'Ticker': tic,
                    'Price': info.get('currentPrice'),
                    'PE': info.get('forwardPE', 999),
                    'PB': info.get('priceToBook', 999),
                    'RevGrowth': info.get('revenueGrowth', 0),
                    'Cash': info.get('totalCash', 0),
                    'Debt': info.get('totalDebt', 0),
                    'Margins': info.get('profitMargins', 0)
                }
                
                # Scoring Logic
                score = 0
                if mode == "Clive's Deep Value (Mining)":
                    net_cash = (data['Cash'] - data['Debt']) > 0
                    if net_cash: score += 2
                    if data['PB'] < 2.0: score += 1
                    if data['RevGrowth'] > 0.10: score += 1
                    data['Net Cash'] = "YES" if net_cash else "NO"
                else:
                    # Factor Logic
                    if data['PE'] < 40: score += 1
                    if data['RevGrowth'] > 0.20: score += 2
                    if data['Margins'] > 0.20: score += 1
                
                data['Score'] = score
                results.append(data)
                
            except: continue
            
        bar.empty()
        st.dataframe(pd.DataFrame(results).sort_values('Score', ascending=False), use_container_width=True)

def render_math_lab():
    st.markdown("### 🧮 TITAN MATH LAB")
    sym = st.text_input("Analyze Distribution", "SPY")
    
    if st.button("CALCULATE STATISTICS"):
        df = TitanData.fetch_stock(sym, "1d", "2y")
        if not df.empty:
            returns = df['Close'].pct_change().dropna()
            
            c1, c2 = st.columns(2)
            with c1:
                # Distribution
                fig = go.Figure(data=[go.Histogram(x=returns, nbinsx=50, marker_color='#00ffbb')])
                fig.update_layout(title="Return Distribution", template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                # LinReg Slope
                slope = linregress(np.arange(len(df)), df['Close'])[0]
                st.metric("Linear Regression Slope", f"{slope:.4f}")
                st.metric("Skewness", f"{returns.skew():.4f}")
                st.metric("Kurtosis", f"{returns.kurtosis():.4f}")
                
            st.caption("Fat tails indicate higher probability of extreme events (Black Swans).")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    # --- SIDEBAR SETTINGS ---
    st.sidebar.title("TITAN OMNI")
    
    # 1. AI CONFIG
    with st.sidebar.expander("🤖 AI Brain"):
        ai_model = st.selectbox("Model", ["OpenAI GPT-4o", "DeepSeek-V3", "Gemini Pro"])
        api_key = st.text_input("API Key", type="password")

    # 2. STRATEGY SELECTOR
    st.sidebar.subheader("⚔️ STRATEGY SELECTOR")
    available_strats = ["Apex Trend", "Dark Vector", "WaveTrend", "Squeeze", "Gann HiLo"]
    active_strats = st.sidebar.multiselect("Active Engines", available_strats, default=["Apex Trend", "Squeeze"])
    
    # 3. PARAMETERS
    with st.sidebar.expander("🎛️ Tuner"):
        params = {
            'threshold': st.slider("Signal Threshold (Confluence)", 1, len(active_strats), 2),
            'hma_len': st.number_input("HMA Length", 10, 200, 55),
            'atr_mult': st.number_input("ATR Multiplier", 1.0, 5.0, 1.5),
            'amp': st.number_input("Vector Amplitude", 2, 50, 10),
            'gann_len': st.number_input("Gann Length", 2, 20, 3)
        }

    # 4. NAVIGATION
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("MODULE", ["Terminal", "Screener", "Math Lab"])

    # --- RENDER ---
    if app_mode == "Terminal":
        render_terminal(api_key, ai_model, active_strats, params)
    elif app_mode == "Screener":
        render_screener()
    elif app_mode == "Math Lab":
        render_math_lab()

if __name__ == "__main__":
    main()
