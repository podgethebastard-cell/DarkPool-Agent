import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Crypto Strategy Lab (Native Pandas)")

st.title("🛡️ Modular Crypto Trading Strategy Lab")
st.markdown("""
This tool allows you to combine up to **10 different trading strategies** using **pure Pandas** (no extra TA libraries).
If you select multiple strategies, the app uses a **Consensus Mechanism**:
- **BUY:** All active strategies must agree to Buy (or neutral), with at least one Buy.
- **SELL:** All active strategies must agree to Sell (or neutral), with at least one Sell.
- **CONFLICT:** If one strategy says BUY and another says SELL, the combined signal is **NEUTRAL (Hold)**.
""")

# ==========================================
# 2. DATA FETCHING ENGINE
# ==========================================
@st.cache_data(ttl=300)
def fetch_crypto_data(symbol, timeframe, limit=500):
    try:
        # Using Kraken as it usually allows public OHLCV access without API keys reliably
        exchange = ccxt.kraken() 
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ==========================================
# 3. STRATEGY LIBRARY (Manual Pandas Implementations)
# ==========================================

def strat_sma_crossover(df, fast_len, slow_len):
    col_name = f'Signal_SMA_{fast_len}_{slow_len}'
    
    # Manual SMA using rolling
    df[f'SMA_{fast_len}'] = df['close'].rolling(window=fast_len).mean()
    df[f'SMA_{slow_len}'] = df['close'].rolling(window=slow_len).mean()
    
    df[col_name] = 0
    # Buy: Fast > Slow
    df.loc[(df[f'SMA_{fast_len}'] > df[f'SMA_{slow_len}']), col_name] = 1
    # Sell: Fast < Slow
    df.loc[(df[f'SMA_{fast_len}'] < df[f'SMA_{slow_len}']), col_name] = -1
    return df, col_name

def strat_rsi(df, length, overbought, oversold):
    col_name = f'Signal_RSI_{length}'
    
    # Manual RSI Calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Use exponential moving average for Wilder's smoothing (standard RSI)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df[col_name] = 0
    df.loc[df['RSI'] < oversold, col_name] = 1
    df.loc[df['RSI'] > overbought, col_name] = -1
    return df, col_name

def strat_bollinger(df, length, std_dev):
    col_name = f'Signal_BB_{length}'
    
    # Manual Bollinger Bands
    sma = df['close'].rolling(window=length).mean()
    std = df['close'].rolling(window=length).std()
    
    df['BBU'] = sma + (std * std_dev)
    df['BBL'] = sma - (std * std_dev)
    
    df[col_name] = 0
    df.loc[df['close'] < df['BBL'], col_name] = 1 # Buy (Mean Reversion)
    df.loc[df['close'] > df['BBU'], col_name] = -1 # Sell
    return df, col_name

def strat_macd(df, fast, slow, signal):
    col_name = f'Signal_MACD'
    
    # Manual MACD
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    df['MACD_Line'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
    
    df[col_name] = 0
    df.loc[df['MACD_Line'] > df['MACD_Signal'], col_name] = 1
    df.loc[df['MACD_Line'] < df['MACD_Signal'], col_name] = -1
    return df, col_name

def strat_donchian(df, length):
    col_name = f'Signal_Donchian_{length}'
    
    # Manual Donchian
    df['Donchian_High'] = df['high'].rolling(window=length).max()
    df['Donchian_Low'] = df['low'].rolling(window=length).min()
    
    df[col_name] = 0
    # Breakout logic (Close > Previous High)
    df.loc[df['close'] > df['Donchian_High'].shift(1), col_name] = 1
    df.loc[df['close'] < df['Donchian_Low'].shift(1), col_name] = -1
    return df, col_name

def strat_vwap(df):
    col_name = 'Signal_VWAP'
    
    # Manual VWAP (Cumulative since start of loaded data for simplicity)
    # Standard VWAP resets daily, but for a continuous stream, we use the rolling window or cumulative sum
    v = df['volume']
    tp = (df['high'] + df['low'] + df['close']) / 3
    # We will use a rolling VWAP approximation if the data is long, 
    # but here we do a cumulative sum over the fetched dataframe (e.g. 500 candles)
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    
    df[col_name] = 0
    df.loc[df['close'] < df['VWAP'], col_name] = 1 # Undervalued
    df.loc[df['close'] > df['VWAP'], col_name] = -1 # Overvalued
    return df, col_name

def strat_z_score(df, length, thresh):
    col_name = 'Signal_ZScore'
    
    # Manual Z-Score
    mean = df['close'].rolling(window=length).mean()
    std = df['close'].rolling(window=length).std()
    df['Z_Score'] = (df['close'] - mean) / std
    
    df[col_name] = 0
    df.loc[df['Z_Score'] < -thresh, col_name] = 1 # Revert to mean (Buy)
    df.loc[df['Z_Score'] > thresh, col_name] = -1 # Revert to mean (Sell)
    return df, col_name

def strat_momentum_roc(df, length):
    col_name = 'Signal_ROC'
    
    # Manual ROC: ((Current - N_ago) / N_ago) * 100
    df['ROC'] = ((df['close'] - df['close'].shift(length)) / df['close'].shift(length)) * 100
    
    df[col_name] = 0
    df.loc[df['ROC'] > 0, col_name] = 1
    df.loc[df['ROC'] < 0, col_name] = -1
    return df, col_name

def strat_grid_sim(df, grid_count, grid_range_pct):
    col_name = 'Signal_Grid'
    # Simplified Grid: Assumes a central pivot based on recent SMA
    pivot = df['close'].rolling(window=50).mean().fillna(df['close'])
    upper_bound = pivot * (1 + grid_range_pct)
    lower_bound = pivot * (1 - grid_range_pct)
    
    df[col_name] = 0
    df.loc[df['close'] <= lower_bound, col_name] = 1
    df.loc[df['close'] >= upper_bound, col_name] = -1
    
    df['Grid_Upper'] = upper_bound
    df['Grid_Lower'] = lower_bound
    return df, col_name

def strat_dca(df):
    col_name = 'Signal_DCA'
    # DCA is always buying
    df[col_name] = 1
    return df, col_name

# ==========================================
# 4. SIDEBAR INPUTS
# ==========================================
with st.sidebar:
    st.header("1. Asset Settings")
    symbol = st.text_input("Symbol", "BTC/USD")
    timeframe = st.selectbox("Timeframe", ['1d', '4h', '1h', '15m'], index=0)
    
    st.header("2. Strategy Selector")
    # Multiselect to allow combinations
    strategies = st.multiselect(
        "Activate Strategies",
        ["SMA Crossover", "RSI", "Bollinger Bands", "MACD", "Donchian Channels", 
         "VWAP", "Z-Score", "Momentum (ROC)", "Grid Trading", "DCA"]
    )
    
    st.header("3. Parameters")
    
    # Dynamic params based on selection
    params = {}
    if "SMA Crossover" in strategies:
        st.subheader("SMA Settings")
        params['sma_fast'] = st.number_input("SMA Fast", 10, 100, 50)
        params['sma_slow'] = st.number_input("SMA Slow", 50, 300, 200)
        
    if "RSI" in strategies:
        st.subheader("RSI Settings")
        params['rsi_len'] = st.number_input("RSI Length", 5, 50, 14)
        params['rsi_ob'] = st.number_input("RSI Overbought", 50, 100, 70)
        params['rsi_os'] = st.number_input("RSI Oversold", 0, 50, 30)

    if "Bollinger Bands" in strategies:
        st.subheader("Bollinger Settings")
        params['bb_len'] = st.number_input("BB Length", 5, 50, 20)
        params['bb_std'] = st.number_input("BB Std Dev", 1.0, 5.0, 2.0)
        
    if "MACD" in strategies:
        st.subheader("MACD Settings")
        params['macd_fast'] = st.number_input("MACD Fast", 5, 50, 12)
        params['macd_slow'] = st.number_input("MACD Slow", 20, 100, 26)
        params['macd_sig'] = st.number_input("MACD Signal", 2, 50, 9)

    if "Donchian Channels" in strategies:
        st.subheader("Donchian Settings")
        params['donchian_len'] = st.number_input("Donchian Length", 5, 100, 20)
        
    if "Z-Score" in strategies:
        st.subheader("Z-Score Settings")
        params['z_len'] = st.number_input("Z-Score Length", 10, 100, 20)
        params['z_thresh'] = st.number_input("Z-Score Threshold", 1.0, 5.0, 2.0)
        
    if "Momentum (ROC)" in strategies:
        st.subheader("Momentum Settings")
        params['roc_len'] = st.number_input("ROC Length", 5, 50, 10)
        
    if "Grid Trading" in strategies:
        st.subheader("Grid Settings")
        params['grid_range'] = st.number_input("Grid Range % (Decimal)", 0.01, 0.5, 0.05)

# ==========================================
# 5. MAIN LOGIC
# ==========================================

# 1. Fetch Data
data = fetch_crypto_data(symbol, timeframe)

if data is None:
    st.error(f"Could not fetch data for {symbol}. It might be invalid or the Kraken API is busy.")
else:
    # 2. Run Active Strategies
    active_signal_cols = []
    
    if "SMA Crossover" in strategies:
        data, col = strat_sma_crossover(data, params['sma_fast'], params['sma_slow'])
        active_signal_cols.append(col)
        
    if "RSI" in strategies:
        data, col = strat_rsi(data, params['rsi_len'], params['rsi_ob'], params['rsi_os'])
        active_signal_cols.append(col)
        
    if "Bollinger Bands" in strategies:
        data, col = strat_bollinger(data, params['bb_len'], params['bb_std'])
        active_signal_cols.append(col)
        
    if "MACD" in strategies:
        data, col = strat_macd(data, params['macd_fast'], params['macd_slow'], params['macd_sig'])
        active_signal_cols.append(col)
        
    if "Donchian Channels" in strategies:
        data, col = strat_donchian(data, params['donchian_len'])
        active_signal_cols.append(col)
        
    if "VWAP" in strategies:
        data, col = strat_vwap(data)
        active_signal_cols.append(col)
        
    if "Z-Score" in strategies:
        data, col = strat_z_score(data, params['z_len'], params['z_thresh'])
        active_signal_cols.append(col)
        
    if "Momentum (ROC)" in strategies:
        data, col = strat_momentum_roc(data, params['roc_len'])
        active_signal_cols.append(col)
        
    if "Grid Trading" in strategies:
        data, col = strat_grid_sim(data, 10, params['grid_range'])
        active_signal_cols.append(col)
        
    if "DCA" in strategies:
        data, col = strat_dca(data)
        active_signal_cols.append(col)

    # 3. Combine Signals
    if active_signal_cols:
        # Sum all signal columns
        data['Total_Signal'] = data[active_signal_cols].sum(axis=1)
        
        # Conflict Check:
        # If any strategy says Buy (1) AND any strategy says Sell (-1) in the same row, result is 0.
        
        def resolve_conflict(row):
            signals = [row[c] for c in active_signal_cols]
            has_buy = 1 in signals
            has_sell = -1 in signals
            
            if has_buy and has_sell:
                return 0 # CONFLICT -> NEUTRAL
            elif has_buy:
                return 1
            elif has_sell:
                return -1
            else:
                return 0
        
        data['Final_Signal'] = data.apply(resolve_conflict, axis=1)
    else:
        data['Final_Signal'] = 0

    # ==========================================
    # 6. VISUALIZATION
    # ==========================================
    
    # Split layout: Chart on top, dataframe below
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"Price Chart: {symbol}")
        
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data['timestamp'], open=data['open'], high=data['high'],
            low=data['low'], close=data['close'], name='Price'
        ))
        
        # Add Technical Indicators to chart if selected
        if "SMA Crossover" in strategies:
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data[f'SMA_{params["sma_fast"]}'], line=dict(color='orange', width=1), name='SMA Fast'))
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data[f'SMA_{params["sma_slow"]}'], line=dict(color='blue', width=1), name='SMA Slow'))
            
        if "Bollinger Bands" in strategies:
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data['BBU'], line=dict(color='gray', width=1, dash='dot'), name='Bands Upper'))
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data['BBL'], line=dict(color='gray', width=1, dash='dot'), name='Bands Lower'))

        if "Grid Trading" in strategies:
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data['Grid_Upper'], line=dict(color='red', width=1, dash='dash'), name='Grid Sell Zone'))
            fig.add_trace(go.Scatter(x=data['timestamp'], y=data['Grid_Lower'], line=dict(color='green', width=1, dash='dash'), name='Grid Buy Zone'))

        # Plot Buy/Sell Signals
        buy_signals = data[data['Final_Signal'] == 1]
        sell_signals = data[data['Final_Signal'] == -1]
        
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals['timestamp'], y=buy_signals['low']*0.99,
                mode='markers', marker=dict(symbol='triangle-up', color='green', size=12),
                name='Combined BUY'
            ))
        
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals['timestamp'], y=sell_signals['high']*1.01,
                mode='markers', marker=dict(symbol='triangle-down', color='red', size=12),
                name='Combined SELL'
            ))
        
        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Signal Log")
        st.markdown("Recent signals generated by the combination:")
        
        # Filter only rows where signal != 0
        sig_log = data[data['Final_Signal'] != 0][['timestamp', 'close', 'Final_Signal']].tail(10)
        
        def color_signal(val):
            color = 'green' if val == 1 else 'red'
            return f'color: {color}; font-weight: bold'
        
        if not sig_log.empty:
            # Simple custom display since style.applymap can be tricky in some st versions
            st.dataframe(sig_log)
        else:
            st.write("No signals generated in current view.")

    # ==========================================
    # 7. RAW DATA INSPECTION
    # ==========================================
    with st.expander("Detailed Data View"):
        st.dataframe(data)

    # ==========================================
    # 8. CONFLICT MONITOR
    # ==========================================
    if len(strategies) > 1:
        st.divider()
        st.warning("⚠️ Conflict Monitor Active")
        st.write("If you see gaps where you expect trades, it is likely because your selected strategies are conflicting.")
        
        if active_signal_cols:
            data['Has_Buy'] = (data[active_signal_cols] == 1).any(axis=1)
            data['Has_Sell'] = (data[active_signal_cols] == -1).any(axis=1)
            conflicts = data[data['Has_Buy'] & data['Has_Sell']]
            
            st.metric("Conflicting Signals Detected (Neutralized)", len(conflicts))
