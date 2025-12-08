import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import uuid

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="DarkPool AI Analyst",
    page_icon="xx",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 10px; color: white; }
    .stAlert { background-color: #1e1e1e; color: white; border: 1px solid #444; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #262730; border-bottom: 2px solid #00E676; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS (NATIVE MATH EXTENDED)
# ==========================================
class NativeIndicators:
    @staticmethod
    def sma(series, length):
        return series.rolling(window=length).mean()

    @staticmethod
    def ema(series, length):
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def rma(series, length):
        return series.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def hma(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wmaf = NativeIndicators.wma(series, half_length)
        wmas = NativeIndicators.wma(series, length)
        diff = 2 * wmaf - wmas
        return NativeIndicators.wma(diff, sqrt_length)

    @staticmethod
    def stdev(series, length):
        # FIXED: Added missing stdev function to prevent AttributeError
        return series.rolling(window=length).std()

    @staticmethod
    def rsi(series, length=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = NativeIndicators.rma(gain, length)
        avg_loss = NativeIndicators.rma(loss, length)
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def mfi(high, low, close, volume, length=14):
        tp = (high + low + close) / 3
        raw_money_flow = tp * volume
        flow_pos = np.where(tp > tp.shift(1), raw_money_flow, 0)
        flow_neg = np.where(tp < tp.shift(1), raw_money_flow, 0)
        mf_pos_sum = pd.Series(flow_pos).rolling(length).sum()
        mf_neg_sum = pd.Series(flow_neg).rolling(length).sum()
        mfi_ratio = mf_pos_sum / mf_neg_sum
        return 100 - (100 / (1 + mfi_ratio))

    @staticmethod
    def adx(high, low, close, length=14, smooth=14):
        h_l = high - low
        h_pc = (high - close.shift(1)).abs()
        l_pc = (low - close.shift(1)).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        up = high - high.shift(1)
        down = low.shift(1) - low
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        tr_smooth = NativeIndicators.rma(tr, length)
        plus_dm_smooth = NativeIndicators.rma(plus_dm, length)
        minus_dm_smooth = NativeIndicators.rma(minus_dm, length)
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = NativeIndicators.rma(dx, smooth)
        return plus_di, minus_di, adx

ta = NativeIndicators()

# ==========================================
# 3. INDICATOR LOGIC 
# ==========================================
def calculate_money_flow(df, length=14, smooth=3):
    close = df['Close']
    volume = df['Volume']
    rsi_source = ta.rsi(close, length=length) - 50
    mf_vol = volume / ta.sma(volume, length=length)
    money_flow = ta.ema(rsi_source * mf_vol, length=smooth)
    return money_flow

def calculate_apex_trend(df, len_main=55, mult=1.5):
    close = df['Close']
    high = df['High']
    low = df['Low']
    baseline = ta.hma(close, length=len_main)
    h_l = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low - close.shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    atr = ta.rma(tr, len_main)
    upper = baseline + (atr * mult)
    lower = baseline - (atr * mult)
    trend_list = []
    close_vals = close.values
    upper_vals = upper.values
    lower_vals = lower.values
    curr = 0
    for i in range(len(df)):
        c = close_vals[i]
        u = upper_vals[i]
        l = lower_vals[i]
        if np.isnan(u):
            trend_list.append(0)
            continue
        if c > u: curr = 1
        elif c < l: curr = -1
        trend_list.append(curr)
    df['Apex_Trend'] = trend_list
    df['Apex_Upper'] = upper
    df['Apex_Lower'] = lower
    return df

def calculate_evwm(df, length=21, vol_smooth=5, mult=2.0):
    close = df['Close']
    volume = df['Volume']
    baseline = ta.hma(close, length=length)
    h_l = df['High'] - df['Low']
    tr = pd.concat([h_l, (df['High']-close.shift(1)).abs(), (df['Low']-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = ta.rma(tr, length)
    elasticity = (close - baseline) / atr
    rvol = volume / ta.sma(volume, length)
    smooth_rvol = ta.sma(rvol, vol_smooth)
    final_force = np.sqrt(smooth_rvol)
    evwm = elasticity * final_force
    band_basis = ta.sma(evwm, length*2)
    band_dev = ta.stdev(evwm, length*2) * mult 
    df['EVWM'] = evwm
    df['EVWM_Upper'] = band_basis + band_dev
    df['EVWM_Lower'] = band_basis - band_dev
    return df

def calculate_sr(df, period=10):
    high = df['High']
    low = df['Low']
    df['Pivot_High'] = high.rolling(period*2+1, center=True).max()
    df['Pivot_Low'] = low.rolling(period*2+1, center=True).min()
    is_pivot_high = (high == df['Pivot_High'])
    is_pivot_low = (low == df['Pivot_Low'])
    res = df[is_pivot_high]['High'].tail(3).values
    sup = df[is_pivot_low]['Low'].tail(3).values
    return res, sup

def calculate_darkpool_macd(df, fast=12, slow=26, sig=9, vol_len=20, use_real_vol=True):
    close = df['Close']
    volume = df['Volume']
    fast_ma = ta.ema(close, fast)
    slow_ma = ta.ema(close, slow)
    macd = fast_ma - slow_ma
    signal = ta.ema(macd, sig)
    hist = macd - signal
    avg_vol = ta.sma(volume, vol_len)
    rvol = np.where(avg_vol > 0, volume / avg_vol, 1.0)
    final_hist = (hist * rvol) if use_real_vol else hist
    df['DP_MACD'] = macd
    df['DP_Signal'] = signal
    df['DP_Hist'] = final_hist
    return df

def calculate_my_adx(df, len_adx=14, len_smooth=14):
    pdi, mdi, adx = ta.adx(df['High'], df['Low'], df['Close'], length=len_adx, smooth=len_smooth)
    df['ADX'] = adx
    df['PlusDI'] = pdi
    df['MinusDI'] = mdi
    return df

def calculate_rsi_divergence(df, length=14, lookback=5):
    rsi = ta.rsi(df['Close'], length)
    df['RSI'] = rsi
    piv_high = rsi.rolling(lookback*2+1, center=True).max()
    piv_low = rsi.rolling(lookback*2+1, center=True).min()
    bull_divs = [np.nan] * len(df)
    bear_divs = [np.nan] * len(df)
    last_pl_price = np.nan
    last_pl_rsi = np.nan
    last_ph_price = np.nan
    last_ph_rsi = np.nan
    
    for i in range(lookback*2, len(df)):
        curr_idx = i - lookback
        if rsi.iloc[curr_idx] == piv_low.iloc[curr_idx]:
            curr_pl_price = df['Low'].iloc[curr_idx]
            curr_pl_rsi = rsi.iloc[curr_idx]
            if not np.isnan(last_pl_price):
                if curr_pl_price < last_pl_price and curr_pl_rsi > last_pl_rsi:
                    bull_divs[curr_idx] = curr_pl_rsi 
            last_pl_price = curr_pl_price
            last_pl_rsi = curr_pl_rsi

        if rsi.iloc[curr_idx] == piv_high.iloc[curr_idx]:
            curr_ph_price = df['High'].iloc[curr_idx]
            curr_ph_rsi = rsi.iloc[curr_idx]
            if not np.isnan(last_ph_price):
                if curr_ph_price > last_ph_price and curr_ph_rsi < last_ph_rsi:
                    bear_divs[curr_idx] = curr_ph_rsi 
            last_ph_price = curr_ph_price
            last_ph_rsi = curr_ph_rsi

    df['Bull_Div'] = bull_divs
    df['Bear_Div'] = bear_divs
    return df

def calculate_gann_hilo(df, length=3):
    high = df['High']
    low = df['Low']
    close = df['Close']
    sma_high = high.rolling(length).mean()
    sma_low = low.rolling(length).mean()
    activator = [np.nan] * len(df)
    trend = [0] * len(df) 
    curr_trend = 1
    curr_act = sma_low.iloc[0] if not np.isnan(sma_low.iloc[0]) else low.iloc[0]
    
    for i in range(len(df)):
        c = close.iloc[i]
        sh = sma_high.iloc[i]
        sl = sma_low.iloc[i]
        if np.isnan(sh) or np.isnan(sl):
            activator[i] = np.nan
            continue
        prev_act = curr_act 
        if curr_trend == 1:
            if c < prev_act:
                curr_trend = -1
                curr_act = sh
            else:
                curr_act = sl
        else:
            if c > prev_act:
                curr_trend = 1
                curr_act = sl
            else:
                curr_act = sh
        activator[i] = curr_act
        trend[i] = curr_trend
    df['Gann_Activator'] = activator
    df['Gann_Trend'] = trend
    return df

def calculate_advanced_volume(df, len_cmf=20, len_mfi=14, len_vrsi=14, len_rvol=20):
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    mfm = ((2 * close - low - high) / (high - low)).replace([np.inf, -np.inf], 0)
    mf_vol = volume * mfm
    cmf = mf_vol.rolling(len_cmf).sum() / volume.rolling(len_cmf).sum()
    mfi_val = ta.mfi(high, low, close, volume, len_mfi)
    vrsi = ta.rsi(volume, len_vrsi)
    rvol = volume / volume.rolling(len_rvol).mean()
    df['CMF'] = cmf
    df['MFI'] = mfi_val
    df['Vol_RSI'] = vrsi
    df['RVOL'] = rvol
    return df

# ==========================================
# 4. MACRO DATA
# ==========================================
@st.cache_data(ttl=3600)
def get_macro_data():
    tickers = {
        'BTC': 'BTC-USD', 'SPX': '^GSPC', 'TLT': 'TLT', 
        'HYG': 'HYG', 'VIX': '^VIX', 'US10Y': '^TNX', 'US02Y': '^IRX'
    }
    try:
        data = yf.download(list(tickers.values()), period="1y", interval="1d")['Close']
        data = data.ffill().dropna()
        df = pd.DataFrame(index=data.index)
        def get_col(key): return data[tickers[key]] if tickers[key] in data.columns else pd.Series(0, index=data.index)
        df['SPY_TLT'] = get_col('SPX') / get_col('TLT')
        df['HYG_TLT'] = get_col('HYG') / get_col('TLT')
        df['BTC_SPX'] = get_col('BTC') / get_col('SPX')
        df['YIELD_CURVE'] = get_col('US10Y') - get_col('US02Y') 
        df['VIX'] = get_col('VIX')
        def get_regime(series, invert=False):
            e20 = ta.ema(series, 20)
            e50 = ta.ema(series, 50)
            pct = (e20 - e50) / e50 * 100
            sig = np.where(pct > 0.05, 1, np.where(pct < -0.05, -1, 0))
            return sig * -1 if invert else sig
        score = 0
        score += get_regime(df['SPY_TLT'])
        score += get_regime(df['HYG_TLT'])
        score += get_regime(df['BTC_SPX'])
        score += get_regime(df['VIX'], invert=True)
        score += np.where(df['YIELD_CURVE'] > 0, 1, -1)
        return score[-1] if hasattr(score[-1], "item") else score[-1], df
    except:
        return 0, None

# ==========================================
# 5. DATA FETCHING (FIXED KEYERROR & MAPPING)
# ==========================================
TICKERS = [
    "SGLN", "SSLN", "SPLT", "SPDM", 
    "SILG", "GJGB", "ESGP", "URJP", 
    "COPP", "SPGP"
]

@st.cache_data(ttl=900)
def get_ticker_data(symbol):
    yf_symbol = f"{symbol}.L"
    
    # Use auto_adjust=False to get standard OHLC, avoiding some multi-index issues
    df = yf.download(yf_symbol, period="1y", interval="1d", auto_adjust=False)
    
    # FIXED: Robust MultiIndex Flattener
    if isinstance(df.columns, pd.MultiIndex):
        # If columns are (Price, Ticker), we want level 0 (Price)
        # If columns are (Ticker, Price), we want level 1 (Price)
        
        # Check if 'Close' is in level 0
        if 'Close' in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        # Check if 'Close' is in level 1
        elif df.columns.nlevels > 1 and 'Close' in df.columns.get_level_values(1):
            df.columns = df.columns.get_level_values(1)
            
    return df

# ==========================================
# 6. MAIN APP
# ==========================================
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key: st.session_state['openai_api_key'] = api_key
    st.markdown("---")
    st.info("System Status: Online")

macro_score, _ = get_macro_data()
risk_state = "RISK ON" if macro_score >= 2 else ("RISK OFF" if macro_score <= -2 else "NEUTRAL")
risk_color = "green" if macro_score >= 2 else ("red" if macro_score <= -2 else "gray")

st.markdown(f"""
<div style="background-color: #262730; padding: 10px; border-radius: 5px; border-left: 5px solid {risk_color}; margin-bottom: 20px;">
    <h3>🌍 Global Macro Score: {int(macro_score)} ({risk_state})</h3>
</div>
""", unsafe_allow_html=True)

# --- CREATE TABS FOR EACH TICKER ---
tabs = st.tabs(TICKERS)

for i, (tab, ticker_name) in enumerate(zip(tabs, TICKERS)):
    with tab:
        # FIXED: MAPPING LOGIC FOR TRADINGVIEW
        # We manually attach "LSE:" because "SGLN" defaults to US OTC (Surgline)
        # But we do NOT show this prefix in the Tab Title (user request)
        tv_sym = f"LSE:{ticker_name}"
        unique_tv_id = f"tv_chart_{i}"
        
        components.html(f"""
        <div class="tradingview-widget-container">
          <div id="{unique_tv_id}"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{"width": "100%", "height": 450, "symbol": "{tv_sym}", "interval": "D", "theme": "dark", "container_id": "{unique_tv_id}"}});
          </script>
        </div>
        """, height=450)

        # FETCH & CALCULATE
        df = get_ticker_data(ticker_name)
        
        if df.empty:
            st.warning(f"No data available for {ticker_name}.")
        else:
            # Squeeze single-column DFs to Series (Safety Check)
            for col in ['Open','High','Low','Close','Volume']:
                if col in df.columns and isinstance(df[col], pd.DataFrame): 
                    df[col] = df[col].squeeze()

            # INDICATORS
            df = calculate_apex_trend(df)
            df = calculate_evwm(df)
            df = calculate_money_flow(df)
            df = calculate_darkpool_macd(df)  
            df = calculate_my_adx(df)         
            df = calculate_rsi_divergence(df)
            df = calculate_gann_hilo(df)      
            df = calculate_advanced_volume(df)
            res, sup = calculate_sr(df) 

            # PLOTLY (7 Rows)
            fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                row_heights=[0.3, 0.1, 0.1, 0.12, 0.12, 0.12, 0.14],
                                subplot_titles=("Price, Apex Trend, Gann & Liquidity", "Money Flow Matrix", "EVWM", 
                                                "DarkPool MACD", "ADX Optimized", "RSI & Divergence", 
                                                "Adv. Volume (CMF)"))

            # R1: Price
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Upper'], line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Apex_Lower'], fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)', line=dict(width=0), name="Apex Cloud"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Gann_Activator'], line=dict(color='orange', width=2), name="Gann Activator"), row=1, col=1)
            for r in res: fig.add_hline(y=r, line=dict(color='red', dash='dash'), row=1, col=1)
            for s in sup: fig.add_hline(y=s, line=dict(color='green', dash='dash'), row=1, col=1)
            
            # R2: MF
            cols_mf = ['green' if x > 0 else 'red' for x in df['Money_Flow']]
            fig.add_trace(go.Bar(x=df.index, y=df['Money_Flow'], marker_color=cols_mf, name="Money Flow"), row=2, col=1)

            # R3: EVWM
            fig.add_trace(go.Scatter(x=df.index, y=df['EVWM'], line_color='white', name="EVWM"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EVWM_Upper'], line=dict(color='gray', dash='dot'), showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EVWM_Lower'], line=dict(color='gray', dash='dot'), showlegend=False), row=3, col=1)

            # R4: MACD
            hist_cols = []
            for j in range(len(df)):
                h = df['DP_Hist'].iloc[j]
                prev_h = df['DP_Hist'].iloc[j-1] if j > 0 else 0
                if h >= 0: hist_cols.append("#00E5FF" if h > prev_h else "#2979FF") 
                else: hist_cols.append("#FF5252" if h < prev_h else "#FFC400")
            fig.add_trace(go.Bar(x=df.index, y=df['DP_Hist'], marker_color=hist_cols, name="MACD Hist"), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['DP_MACD'], line_color='#00E676', name="MACD Line"), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['DP_Signal'], line_color='#FFFF00', name="Signal"), row=4, col=1)

            # R5: ADX
            fig.add_trace(go.Scatter(x=df.index, y=df['PlusDI'], line_color='blue', name="+DI"), row=5, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MinusDI'], line_color='gray', name="-DI"), row=5, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line_color='fuchsia', name="ADX"), row=5, col=1)
            fig.add_hline(y=25, line=dict(color='white', dash='dot'), row=5, col=1)

            # R6: RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line_color='black', name="RSI"), row=6, col=1)
            fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=6, col=1)
            fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=6, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Bull_Div'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name="Bull Div"), row=6, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Bear_Div'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name="Bear Div"), row=6, col=1)
            
            # R7: CMF
            cmf_cols = ['#00E676' if x > 0 else '#FF5252' for x in df['CMF']]
            fig.add_trace(go.Bar(x=df.index, y=df['CMF'], marker_color=cmf_cols, name="CMF"), row=7, col=1)
            fig.add_hline(y=0, line=dict(color='gray'), row=7, col=1)

            fig.update_layout(height=1600, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}")

            # AI REPORT
            st.markdown("### 🤖 AI Analyst Report")
            if st.button(f"Generate Report for {ticker_name}", key=f"btn_{i}"):
                 if not st.session_state.get('openai_api_key'):
                     st.warning("Needs API Key")
                 else:
                     with st.spinner(f"Analyzing {ticker_name}..."):
                         last = df.iloc[-1]
                         gann_state = "BULL" if last['Gann_Trend'] == 1 else "BEAR"
                         prompt = f"""
                         Analyze {ticker_name}. 
                         MACRO: {macro_score} ({risk_state}).
                         TREND: Apex={"Bull" if last['Apex_Trend']==1 else "Bear"}. Gann HiLo={gann_state}.
                         MOMENTUM: MACD Hist={last['DP_Hist']:.4f}. ADX={last['ADX']:.1f}. RSI={last['RSI']:.1f}.
                         VOLUME: CMF={last['CMF']:.3f}. RVOL={last['RVOL']:.2f}. MFI={last['MFI']:.1f}.
                         Verdict (Bull/Bear/Neutral)? <150 words.
                         """
                         try:
                             client = OpenAI(api_key=st.session_state['openai_api_key'])
                             resp = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}])
                             st.success(resp.choices[0].message.content)
                         except Exception as e:
                             st.error(str(e))
