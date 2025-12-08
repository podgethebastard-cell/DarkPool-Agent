import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import requests
import time
from openai import OpenAI # REQUIRES: pip install openai

# ==========================================
# 1. PAGE CONFIG & TERMINAL CSS
# ==========================================
st.set_page_config(page_title="Titan Scalper [Terminal]", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    /* MAIN BACKGROUND */
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* BLOCK PADDING */
    .block-container { padding-top: 0rem; padding-bottom: 5rem; }
    
    /* EXCELLENT HEADER */
    .titan-header {
        background: linear-gradient(180deg, #111 0%, #050505 100%);
        border-bottom: 1px solid #333;
        padding: 2rem 1rem;
        text-align: center;
        margin-bottom: 2rem;
        border-top: 3px solid #00ffbb;
    }
    .titan-title {
        font-size: 3rem;
        font-weight: 900;
        color: #fff;
        letter-spacing: 4px;
        margin: 0;
        text-shadow: 0 0 20px rgba(0, 255, 187, 0.3);
    }
    .titan-subtitle {
        font-size: 1rem;
        color: #666;
        letter-spacing: 2px;
        margin-top: 0.5rem;
        text-transform: uppercase;
    }
    .titan-badge {
        background: #00ffbb;
        color: #000;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
        vertical-align: middle;
        margin-left: 10px;
    }

    /* TITAN METRIC CARD CSS */
    .titan-card {
        background: #0f0f0f;
        border: 1px solid #222;
        border-left: 4px solid #555;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .titan-card h4 { margin: 0; font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .titan-card h2 { margin: 5px 0 0 0; font-size: 1.8rem; font-weight: 700; color: #fff; }
    .titan-card .sub { font-size: 0.7rem; color: #555; margin-top: 5px; }
    
    /* STATUS COLORS */
    .border-bull { border-left-color: #00ffbb !important; }
    .border-bear { border-left-color: #ff1155 !important; }
    .text-bull { color: #00ffbb !important; }
    .text-bear { color: #ff1155 !important; }
    .text-white { color: #fff !important; }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #080808; border-right: 1px solid #222; }
    
    /* AI ANALYSIS BOX */
    .ai-box {
        background: #0a0a0a;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
        border-left: 3px solid #7d00ff;
    }
    
    /* TOASTS & ALERTS */
    div[data-testid="stToast"] { background-color: #1a1a1a; border: 1px solid #333; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("⚡ TITAN CONFIG")
st.sidebar.caption("v5.0 | AI COMMANDER")
st.sidebar.markdown("---")

# Market Data
st.sidebar.subheader("MARKET FEED")
symbol = st.sidebar.text_input("Symbol (Kraken)", value="BTC/USD") 
timeframe = st.sidebar.selectbox("Timeframe", options=['1m', '5m', '15m'], index=1)
limit = st.sidebar.slider("Candles", min_value=200, max_value=1500, value=500)

st.sidebar.markdown("---")

# Strategies
st.sidebar.subheader("LOGIC ENGINE")
with st.sidebar.expander("Titan Settings", expanded=True):
    amplitude = st.number_input("Sensitivity", min_value=1, value=5)
    channel_dev = st.number_input("Deviation", min_value=1.0, value=3.0, step=0.1)
    hma_len = st.number_input("HMA Length", min_value=1, value=50)
    use_hma_filter = st.checkbox("HMA Filter", value=False)

with st.sidebar.expander("Money Flow Matrix"):
    mf_len = st.number_input("MF Length", value=14)
    hyper_long = st.number_input("Hyper Long", value=25)
    hyper_short = st.number_input("Hyper Short", value=13)

with st.sidebar.expander("Advanced Volume"):
    vol_metric = st.selectbox("Sub-Chart Metric", ["CMF", "MFI", "Volume RSI", "RVOL", "Vol Osc"])
    vol_len = st.number_input("Volume Length", value=20)

st.sidebar.markdown("---")

# --- CREDENTIALS HANDLING (SECRETS vs MANUAL) ---
st.sidebar.subheader("SYSTEM CREDENTIALS")

# 1. TELEGRAM
tg_on = st.sidebar.checkbox("Telegram Active", value=True)
try:
    tg_token = st.secrets["TELEGRAM_TOKEN"]
    tg_chat = st.secrets["TELEGRAM_CHAT_ID"]
    tg_secrets = True
except:
    tg_secrets = False

if tg_secrets:
    bot_token = tg_token
    chat_id = tg_chat
    st.sidebar.success("🔹 Telegram: Connected (Secrets)")
else:
    bot_token = st.sidebar.text_input("Bot Token", type="password")
    chat_id = st.sidebar.text_input("Chat ID")

# 2. OPENAI
try:
    ai_key = st.secrets["OPENAI_API_KEY"]
    ai_secrets = True
    st.sidebar.success("🔹 OpenAI: Connected (Secrets)")
except:
    ai_secrets = False

if not ai_secrets:
    ai_key = st.sidebar.text_input("OpenAI API Key", type="password")

test_btn = st.sidebar.button("Test Connections")

if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = None

# ==========================================
# 3. UTILITIES & CALCS
# ==========================================
def send_telegram_msg(token, chat, msg):
    if not token or not chat: 
        st.error("Missing Token or Chat ID")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat, "text": msg, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except Exception as e: 
        st.error(f"Connection Error: {e}")
        return False

# --- AI ANALYSIS FUNCTION ---
def get_ai_analysis(df_summary, symbol, tf):
    if not ai_key:
        return "⚠️ Please provide an OpenAI API Key."
    
    client = OpenAI(api_key=ai_key)
    
    prompt = f"""
    You are the TITAN SCALPER AI, a professional institutional crypto trader.
    Analyze this market snapshot for {symbol} on the {tf} timeframe.
    
    DATA SNAPSHOT:
    - Current Price: {df_summary['price']}
    - Trend: {df_summary['trend']}
    - Titan Stop: {df_summary['stop']}
    - Money Flow Matrix: {df_summary['mf']} ({'Inflow' if df_summary['mf'] > 0 else 'Outflow'})
    - HyperWave Momentum: {df_summary['hw']}
    - Relative Volume: {df_summary['rvol']}x
    - Institutional Trend (HMA): {df_summary['inst_trend']}
    
    INSTRUCTIONS:
    1. Provide a concise, bulleted assessment of the immediate setup.
    2. Identify if there is a confluence of factors (e.g., Price > HMA AND Money Flow > 0).
    3. Give a clear bias: BULLISH, BEARISH, or NEUTRAL/WAIT.
    4. Keep it under 100 words. Use emojis.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-3.5-turbo if you prefer speed/cost
            messages=[{"role": "system", "content": "You are a sniper scalper."}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

if test_btn:
    t_succ = send_telegram_msg(bot_token, chat_id, "🔥 **TITAN SYSTEM ONLINE**\nTelegram Connected.")
    if t_succ: st.toast("Telegram OK", icon="✅")
    if ai_key: st.toast("OpenAI Key Detected", icon="🤖")

# Math Helpers
def weighted_ma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calc_hma_full(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = weighted_ma(series, half_length)
    wma_full = weighted_ma(series, length)
    diff = 2 * wma_half - wma_full
    return weighted_ma(diff, sqrt_length)

def calculate_rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_mfi(high, low, close, volume, length):
    tp = (high + low + close) / 3
    rmf = tp * volume
    pos = rmf.where(tp > tp.shift(1), 0).rolling(length).sum()
    neg = rmf.where(tp < tp.shift(1), 0).rolling(length).sum()
    return 100 - (100 / (1 + (pos / neg)))

# ==========================================
# 4. MASTER ENGINE
# ==========================================
@st.cache_data(ttl=5) 
def get_data(symbol, timeframe, limit):
    try:
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except: return pd.DataFrame()

def run_titan_engine(df):
    # Titan Vector
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['dev'] = (df['tr'].rolling(100).mean() / 2) * channel_dev
    df['hma'] = calc_hma_full(df['close'], hma_len)

    df['ll'] = df['low'].rolling(amplitude).min()
    df['hh'] = df['high'].rolling(amplitude).max()
    
    trend = np.zeros(len(df)); stop = np.zeros(len(df))
    curr_trend = 0; curr_stop = df['close'].iloc[0]
    curr_max_l = 0.0; curr_min_h = 0.0

    for i in range(amplitude, len(df)):
        c = df['close'].iloc[i]; l = df['ll'].iloc[i]; h = df['hh'].iloc[i]
        dev = df['dev'].iloc[i] if not np.isnan(df['dev'].iloc[i]) else 0
        curr_max_l = max(l, curr_max_l)
        curr_min_h = min(h, curr_min_h) if curr_min_h != 0 else h

        if curr_trend == 0: 
            if c < curr_max_l: curr_trend = 1; curr_min_h = h
            else: 
                curr_min_h = min(curr_min_h, h)
                if l < curr_max_l: curr_max_l = l
        else:
            if c > curr_min_h: curr_trend = 0; curr_max_l = l
            else: 
                curr_max_l = max(curr_max_l, l)
                if h > curr_min_h: curr_min_h = h
        
        if curr_trend == 0:
            s = l + dev
            if s < curr_stop: s = curr_stop
            curr_stop = s
        else:
            s = h - dev
            if s > curr_stop and curr_stop != 0: s = curr_stop
            elif curr_stop == 0: s = h - dev
            curr_stop = s
            
        trend[i] = curr_trend
        stop[i] = curr_stop

    df['trend'] = trend; df['trend_stop'] = stop
    df['is_bull'] = df['trend'] == 0
    
    # Signals
    df['bull_flip'] = (df['is_bull']) & (~df['is_bull'].shift(1).fillna(False).astype(bool))
    df['bear_flip'] = (~df['is_bull']) & (df['is_bull'].shift(1).fillna(True).astype(bool))
    filter_buy = ~use_hma_filter | (df['close'] > df['hma'])
    filter_sell = ~use_hma_filter | (df['close'] < df['hma'])
    df['buy_signal'] = df['bull_flip'] & filter_buy
    df['sell_signal'] = df['bear_flip'] & filter_sell

    # Money Flow Matrix
    rsi_src = calculate_rsi(df['close'], mf_len) - 50
    mf_vol = df['volume'] / df['volume'].rolling(mf_len).mean()
    df['money_flow'] = (rsi_src * mf_vol).fillna(0).ewm(span=3).mean()
    
    pc = df['close'].diff()
    ss = pc.ewm(span=hyper_long).mean().ewm(span=hyper_short).mean()
    ss_abs = abs(pc).ewm(span=hyper_long).mean().ewm(span=hyper_short).mean()
    df['hyper_wave'] = np.where(ss_abs != 0, (100 * (ss / ss_abs)) / 2, 0)

    # Advanced Volume
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['cmf'] = (mfm.fillna(0) * df['volume']).rolling(vol_len).sum() / df['volume'].rolling(vol_len).sum()
    df['mfi'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'], vol_len)
    df['vol_rsi'] = calculate_rsi(df['volume'], 14)
    df['rvol'] = df['volume'] / df['volume'].rolling(vol_len).mean()
    v_s = df['volume'].rolling(14).mean(); v_l = df['volume'].rolling(28).mean()
    df['vol_osc'] = np.where(v_l != 0, 100 * (v_s - v_l) / v_l, 0)

    return df

# ==========================================
# 5. UI EXECUTION
# ==========================================
# --- HEADER ---
st.markdown("""
<div class="titan-header">
    <h1 class="titan-title">TITAN SCALPER <span style="color:#00ffbb">TERMINAL</span></h1>
    <div class="titan-subtitle">INSTITUTIONAL ORDER FLOW & MOMENTUM ENGINE <span class="titan-badge">LIVE</span></div>
</div>
""", unsafe_allow_html=True)

def render_tv(sym):
    s = f"KRAKEN:{sym.replace('/','')}"
    components.html(f"""
    <div class="tradingview-widget-container"><div id="tv"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script>new TradingView.widget({{
      "width": "100%", "height": 450, "symbol": "{s}", "interval": "5",
      "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en",
      "toolbar_bg": "#f1f3f6", "enable_publishing": false, "hide_side_toolbar": false,
      "container_id": "tv"
    }});</script></div>
    """, height=460)

render_tv(symbol)

df = get_data(symbol, timeframe, limit)

if not df.empty:
    df = run_titan_engine(df)
    last = df.iloc[-1]
    
    # --- BROADCASTING ---
    if tg_on and bot_token and chat_id:
        if (last['buy_signal'] or last['sell_signal']) and st.session_state.last_signal_time != last['timestamp']:
            is_buy = last['buy_signal']
            direction = "LONG" if is_buy else "SHORT"
            icon = "🟢" if is_buy else "🔴"
            risk = abs(last['close'] - last['trend_stop'])
            target = last['close'] + (risk * 1.5) if is_buy else last['close'] - (risk * 1.5)
            
            msg = f"""🔥 *TITAN SIGNAL: {symbol} ({timeframe})*
{icon} DIRECTION: *{direction}*
🚪 ENTRY: `${last['close']:,.2f}`
🛑 STOP LOSS: `${last['trend_stop']:,.2f}`
🎯 TARGET: `${target:,.2f}`
🌊 Trend: {'BULLISH' if last['is_bull'] else 'BEARISH'}
📊 Momentum: {'POSITIVE' if last['hyper_wave'] > 0 else 'NEGATIVE'}
💰 Money Flow: {'INFLOW' if last['money_flow'] > 5 else 'OUTFLOW' if last['money_flow'] < -5 else 'NEUTRAL'}
💀 Institutional Trend: {'MACRO BULL' if last['close'] > last['hma'] else 'MACRO BEAR'}
⚠️ _Not financial advice. DYOR._
#DarkPool #Titan #Crypto"""
            
            success = send_telegram_msg(bot_token, chat_id, msg)
            if success:
                st.session_state.last_signal_time = last['timestamp']
                st.toast(f"Broadcast: {direction}", icon="🚀")

    # --- HUD METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    
    trend_cls = "border-bull" if last['is_bull'] else "border-bear"
    trend_txt_cls = "text-bull" if last['is_bull'] else "text-bear"
    trend_lbl = "BULLISH" if last['is_bull'] else "BEARISH"
    mf_val = last['money_flow']
    mf_cls = "text-bull" if mf_val > 0 else "text-bear"
    rvol_cls = "text-bull" if last['rvol'] > 1.5 else "text-bear" if last['rvol'] < 0.8 else "text-white"
    
    with c1:
        st.markdown(f"""<div class="titan-card {trend_cls}"><h4>Price</h4><h2>${last['close']:,.2f}</h2>
        <div class="sub">Trend: <span class="{trend_txt_cls}"><b>{trend_lbl}</b></span></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="titan-card {trend_cls}"><h4>Titan Stop</h4><h2>${last['trend_stop']:,.2f}</h2>
        <div class="sub">Risk Mgmt System</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="titan-card"><h4>Money Flow Matrix</h4><h2 class="{mf_cls}">{mf_val:.2f}</h2>
        <div class="sub">Institutional Pressure</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="titan-card"><h4>Relative Volume</h4><h2 class="{rvol_cls}">{last['rvol']:.2f}x</h2>
        <div class="sub">Anomaly Detection</div></div>""", unsafe_allow_html=True)

    # --- AI ANALYSIS BUTTON ---
    if st.button("🤖 GENERATE AI REPORT", use_container_width=True):
        with st.spinner("Titan AI is analyzing market structure..."):
            summary = {
                'price': last['close'],
                'trend': trend_lbl,
                'stop': last['trend_stop'],
                'mf': last['money_flow'],
                'hw': last['hyper_wave'],
                'rvol': last['rvol'],
                'inst_trend': 'BULL' if last['close'] > last['hma'] else 'BEAR'
            }
            ai_report = get_ai_analysis(summary, symbol, timeframe)
            st.markdown(f"""<div class="ai-box"><h3>🤖 TITAN AI ASSESSMENT</h3>{ai_report}</div>""", unsafe_allow_html=True)

    # --- TRI-PANE CHART ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.20, 0.25],
        subplot_titles=("Price Action", "Money Flow Matrix", f"Advanced Volume ({vol_metric})"))
    
    # 1. Price
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    b_stop = df['trend_stop'].where(df['is_bull'], np.nan).replace(0, np.nan)
    s_stop = df['trend_stop'].where(~df['is_bull'], np.nan).replace(0, np.nan)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=b_stop, mode='lines', line=dict(color='#00ffbb', width=2), name='Bull Stop'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=s_stop, mode='lines', line=dict(color='#ff1155', width=2), name='Bear Stop'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hma'], mode='lines', line=dict(color='gray', dash='dot'), name='HMA'), row=1, col=1)
    
    buys = df[df['buy_signal']]
    if not buys.empty: fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['low']*0.999, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00ffbb'), name='BUY'), row=1, col=1)
    sells = df[df['sell_signal']]
    if not sells.empty: fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['high']*1.001, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff1155'), name='SELL'), row=1, col=1)

    # 2. Matrix
    col_mf = np.where(df['money_flow'] >= 0, '#00ffbb', '#ff1155')
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['money_flow'], marker_color=col_mf, name="Money Flow"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hyper_wave'], mode='lines', line=dict(color='yellow', width=1), name="HyperWave"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#333", row=2, col=1)

    # 3. Volume
    v_data = df['cmf']
    if vol_metric == "MFI": v_data = df['mfi']
    elif vol_metric == "Volume RSI": v_data = df['vol_rsi']
    elif vol_metric == "RVOL": v_data = df['rvol']
    elif vol_metric == "Vol Osc": v_data = df['vol_osc']
    
    is_osc = vol_metric in ["MFI", "Volume RSI"]
    col_vol = np.where(v_data >= (50 if is_osc else 0), '#00ffbb', '#ff1155')
    
    if vol_metric == "RVOL" or vol_metric == "CMF":
         fig.add_trace(go.Bar(x=df['timestamp'], y=v_data, marker_color=col_vol, name=vol_metric), row=3, col=1)
    else:
         fig.add_trace(go.Scatter(x=df['timestamp'], y=v_data, mode='lines', line=dict(color='#00ffbb'), name=vol_metric), row=3, col=1)
         if is_osc:
             fig.add_hline(y=80, line_dash="dot", line_color="red", row=3, col=1)
             fig.add_hline(y=20, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(height=900, paper_bgcolor='#050505', plot_bgcolor='#050505', font=dict(color="#aaa"), showlegend=False, xaxis_rangeslider_visible=False)
    fig.update_yaxes(gridcolor="#222", autorange=True, fixedrange=False)
    fig.update_xaxes(gridcolor="#222")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Awaiting Data Stream...")
