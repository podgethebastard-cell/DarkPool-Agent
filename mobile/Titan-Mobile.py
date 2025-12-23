import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import requests
from openai import OpenAI

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Titan Mobile v2", 
    layout="centered", 
    page_icon="⚡", 
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM CSS (DPC ARCHITECTURE)
# ==========================================
st.markdown("""
    <style>
    /* --- CORE THEME --- */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto Mono', monospace; 
        background-color: #050505;
        color: #e0e0e0;
    }
    
    /* HIDE STREAMLIT CHROME */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* INPUTS */
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        background-color: #111;
        color: #00ffbb;
        border: 1px solid #333;
    }

    /* BUTTONS */
    div.stButton > button:first-child {
        width: 100%;
        height: 3.5em;
        font-weight: bold; 
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: #111;
        border: 1px solid #333;
        color: #ccc;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        border-color: #00ffbb;
        color: #00ffbb;
        box-shadow: 0 0 10px rgba(0, 255, 187, 0.2);
    }
    div.stButton > button:active {
        background: #00ffbb;
        color: #000;
    }

    /* --- TITAN CARDS --- */
    .titan-card {
        background: #0a0a0a;
        border: 1px solid #222;
        border-left: 3px solid #555;
        padding: 12px 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
    }
    .titan-card::after {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.02), transparent);
        pointer-events: none;
    }

    .titan-card h4 { 
        margin: 0; 
        font-size: 0.65rem; 
        color: #666; 
        text-transform: uppercase; 
        letter-spacing: 1.5px;
        font-weight: 700;
    }
    .titan-card h2 { 
        margin: 4px 0 0 0; 
        font-size: 1.3rem; 
        font-weight: 700; 
        color: #fff; 
        font-family: 'Roboto Mono', monospace;
    } 
    .titan-card .sub { 
        font-size: 0.7rem; 
        color: #444; 
        margin-top: 4px; 
    }
    
    /* STATUS MODIFIERS */
    .border-bull { border-left-color: #00E676 !important; } /* Apex Neon Green */
    .border-bear { border-left-color: #FF1744 !important; } /* Apex Neon Red */
    .border-chop { border-left-color: #546E7A !important; } /* Apex Resistive */
    
    .text-bull { color: #00E676 !important; text-shadow: 0 0 10px rgba(0, 230, 118, 0.3); }
    .text-bear { color: #FF1744 !important; text-shadow: 0 0 10px rgba(255, 23, 68, 0.3); }
    .text-chop { color: #546E7A !important; }
    .text-white { color: #fff !important; }

    /* AI TERMINAL */
    .ai-box {
        background: #080808;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
        border-left: 2px solid #7d00ff;
        font-size: 0.85rem;
        line-height: 1.4;
        color: #ccc;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e0e0e;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #666;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a1a;
        color: #fff;
        border-bottom: 2px solid #00ffbb;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. UTILITIES & MATH ENGINE
# ==========================================

def fmt_price(val):
    if val is None or np.isnan(val): return "0.00"
    if val < 1.0: return f"{val:.6f}"
    elif val < 10.0: return f"{val:.4f}"
    else: return f"{val:,.2f}"

def send_telegram_msg(token, chat, msg):
    if not token or not chat: return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
        return True
    except: return False

def get_ai_analysis(summary_text, ai_key):
    if not ai_key: return "⚠️ Missing API Key in Config."
    try:
        client = OpenAI(api_key=ai_key)
        prompt = f"""
        Role: Apex Quantitative System.
        Analyze this crypto setup based on Titan/Apex indicators:
        {summary_text}
        
        Task: Provide a high-precision trading assessment.
        Output Format:
        • BIAS: [BULLISH / BEARISH / NEUTRAL]
        • SETUP: [Brief description of Flux/Structure]
        • RISK: [High/Med/Low]
        • TACTIC: [Limit Entry / Market / Wait]
        Keep total response under 60 words. No fluff.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a quantitative trading assistant."}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# --- CORE MATH FUNCTIONS ---
def weighted_ma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calc_hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = weighted_ma(series, half_length)
    wma_full = weighted_ma(series, length)
    diff = 2 * wma_half - wma_full
    return weighted_ma(diff, sqrt_length)

def rma(series, length):
    """Running Moving Average (Pine Script default for ATR)"""
    return series.ewm(alpha=1/length, adjust=False).mean()

def calculate_atr(df, length):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return rma(true_range, length)

# ==========================================
# 4. TITAN + APEX DATA ENGINE
# ==========================================
@st.cache_data(ttl=15) 
def fetch_market_data(symbol, timeframe, limit):
    try:
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

def run_apex_engine(df, hma_len, atr_mult, flux_len, flux_smooth):
    if df.empty: return df

    # --- 1. APEX VECTOR (FLUX) LOGIC ---
    # Concept: Efficiency = Body / Range. Flux = Dir * Eff * RelVol
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    # Avoid div by zero
    df['efficiency'] = np.where(df['range'] == 0, 0, df['body'] / df['range'])
    
    # Volume Factor
    df['vol_avg'] = df['volume'].rolling(55).mean()
    df['vol_fact'] = np.where(df['vol_avg'] == 0, 1.0, df['volume'] / df['vol_avg'])
    
    # Direction & Raw Vector
    df['direction'] = np.sign(df['close'] - df['open'])
    df['vector_raw'] = df['direction'] * df['efficiency'] * df['vol_fact']
    
    # Smoothed Flux
    df['apex_flux'] = df['vector_raw'].ewm(span=flux_smooth).mean()
    
    # State Classification (Superconductor vs Resistive)
    super_thresh = 0.60 
    resist_thresh = 0.30
    
    conditions = [
        (df['apex_flux'] > super_thresh), # Super Bull
        (df['apex_flux'] < -super_thresh), # Super Bear
        (df['apex_flux'].abs() < resist_thresh) # Resistive
    ]
    choices = [2, -2, 0] # 2=SuperBull, -2=SuperBear, 0=Resistive, 1/-1=Heat
    df['flux_state_code'] = np.select(conditions, choices, default=np.where(df['apex_flux']>0, 1, -1))

    # --- 2. APEX TREND (HMA CLOUD) LOGIC ---
    df['hma_base'] = calc_hma(df['close'], hma_len)
    df['atr'] = calculate_atr(df, hma_len)
    df['trend_upper'] = df['hma_base'] + (df['atr'] * atr_mult)
    df['trend_lower'] = df['hma_base'] - (df['atr'] * atr_mult)
    
    # Trend Determination
    # 1 = Bull, -1 = Bear
    # Vectorized check: if close > upper -> 1, if close < lower -> -1, else hold previous
    # Since Pandas vectorization with "hold previous" is hard, we use a loop or forward fill logic
    # Fast approach: assign triggers then ffill
    
    df['trend_trigger'] = 0
    df.loc[df['close'] > df['trend_upper'], 'trend_trigger'] = 1
    df.loc[df['close'] < df['trend_lower'], 'trend_trigger'] = -1
    
    # Replace 0 with NaN for ffill, then fillna(0) for start
    df['apex_trend'] = df['trend_trigger'].replace(0, np.nan).ffill().fillna(0).astype(int)

    # --- 3. SMC: FVG DETECTION ---
    # Bullish FVG: Low > High[2]
    # Bearish FVG: High < Low[2]
    df['fvg_bull'] = (df['low'] > df['high'].shift(2)) & ((df['low'] - df['high'].shift(2)) > (df['atr']*0.2))
    df['fvg_bear'] = (df['high'] < df['low'].shift(2)) & ((df['low'].shift(2) - df['high']) > (df['atr']*0.2))

    # --- 4. TRAILING STOP (STAIRCASE) ---
    # Custom iteration for ratcheting stop based on Apex Trend
    # If Trend Bull: Stop is Max(PrevStop, Close - ATR*Multiplier)
    # If Trend Bear: Stop is Min(PrevStop, Close + ATR*Multiplier)
    
    stop_arr = np.zeros(len(df))
    trend_arr = df['apex_trend'].values
    close_arr = df['close'].values
    atr_arr = df['atr'].values
    stop_mult = 2.0
    
    curr_stop = close_arr[0]
    
    for i in range(1, len(df)):
        t = trend_arr[i]
        prev_t = trend_arr[i-1]
        c = close_arr[i]
        a = atr_arr[i]
        
        if t == 1:
            # Bullish Logic
            if prev_t != 1: # Trend Switch
                curr_stop = c - (a * stop_mult)
            else:
                curr_stop = max(curr_stop, c - (a * stop_mult))
        elif t == -1:
            # Bearish Logic
            if prev_t != -1: # Trend Switch
                curr_stop = c + (a * stop_mult)
            else:
                curr_stop = min(curr_stop, c + (a * stop_mult))
        else:
            curr_stop = c # Neutral
            
        stop_arr[i] = curr_stop
        
    df['apex_stop'] = stop_arr

    return df

# ==========================================
# 5. UI LAYOUT & MAIN APP
# ==========================================

if 'last_run' not in st.session_state:
    st.session_state.last_run = None

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["⚡ APEX TERMINAL", "🔬 DEEP DIVE", "⚙️ CONFIG"])

# --- CONFIG TAB ---
with tab3:
    st.header("System Parameters")
    
    with st.expander("📡 Data Feed", expanded=True):
        col_s1, col_s2 = st.columns(2)
        with col_s1: symbol = st.text_input("Symbol", value="BTC/USD")
        with col_s2: timeframe = st.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '4h'], index=2)
        limit = st.slider("Lookback", 200, 1000, 500)
        
    with st.expander("🧠 Apex Algo Settings"):
        hma_len = st.number_input("Trend Length (HMA)", 10, 200, 55)
        atr_mult = st.number_input("Cloud Multiplier", 1.0, 5.0, 1.5, 0.1)
        flux_len = st.number_input("Flux Length", 5, 50, 14)
        flux_smooth = st.number_input("Flux Smooth", 2, 20, 5)

    with st.expander("🔑 API Keys", expanded=True):
        tg_active = st.checkbox("Enable Telegram", value=False)
        bot_token = st.text_input("Bot Token", type="password")
        chat_id = st.text_input("Chat ID")
        
        try:
            sec_ai = st.secrets["OPENAI_API_KEY"]
        except:
            sec_ai = ""
        ai_key = st.text_input("OpenAI Key", value=sec_ai, type="password")

# --- FETCH & CALC ---
df = fetch_market_data(symbol, timeframe, limit)

if not df.empty:
    df = run_apex_engine(df, hma_len, atr_mult, flux_len, flux_smooth)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Helpers for display
    is_bull = last['apex_trend'] == 1
    
    # Trend Strings
    trend_str = "BULLISH" if is_bull else ("BEARISH" if last['apex_trend'] == -1 else "NEUTRAL")
    trend_color = "border-bull" if is_bull else "border-bear"
    trend_txt = "text-bull" if is_bull else "text-bear"
    
    # Flux Strings
    flux_val = last['apex_flux']
    if last['flux_state_code'] == 2:
        flux_lbl = "SUPERCONDUCTOR"
        flux_cls = "text-bull"
    elif last['flux_state_code'] == -2:
        flux_lbl = "SUPERCONDUCTOR"
        flux_cls = "text-bear"
    elif last['flux_state_code'] == 0:
        flux_lbl = "RESISTIVE / CHOP"
        flux_cls = "text-chop"
    else:
        flux_lbl = "HIGH HEAT"
        flux_cls = "text-white"

    # FVG/SMC Check
    smc_event = "NONE"
    smc_color = "text-chop"
    if last['fvg_bull']: 
        smc_event = "BULLISH FVG"
        smc_color = "text-bull"
    elif last['fvg_bear']:
        smc_event = "BEARISH FVG"
        smc_color = "text-bear"
    elif last['apex_trend'] != prev['apex_trend']:
        smc_event = "TREND FLIP"
        smc_color = trend_txt
        
    # --- TAB 1: TERMINAL ---
    with tab1:
        # 1. HEADER
        st.caption(f"APEX PROTOCOL | {symbol} | {timeframe.upper()}")
        
        # 2. METRIC GRID
        c1, c2 = st.columns(2)
        
        with c1:
            # PRICE CARD
            st.markdown(f"""
            <div class="titan-card {trend_color}">
                <h4>Market Price</h4>
                <h2>${last['close']:,.2f}</h2>
                <div class="sub">Trend: <span class="{trend_txt}"><b>{trend_str}</b></span></div>
            </div>
            """, unsafe_allow_html=True)
            
            # SMC CARD
            st.markdown(f"""
            <div class="titan-card">
                <h4>SMC Structure</h4>
                <h2 class="{smc_color}">{smc_event}</h2>
                <div class="sub">Last Detected Event</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            # STOP CARD
            dist = abs(last['close'] - last['apex_stop']) / last['close'] * 100
            st.markdown(f"""
            <div class="titan-card {trend_color}">
                <h4>Trailing Stop</h4>
                <h2>${last['apex_stop']:,.2f}</h2>
                <div class="sub">Risk Distance: {dist:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # FLUX CARD
            st.markdown(f"""
            <div class="titan-card">
                <h4>Apex Flux</h4>
                <h2 class="{flux_cls}">{flux_val:.2f}</h2>
                <div class="sub">{flux_lbl}</div>
            </div>
            """, unsafe_allow_html=True)

        # 3. ACTION BUTTONS
        st.markdown("---")
        b1, b2 = st.columns(2)
        
        with b1:
            if st.button("🔥 GENERATE SIGNAL"):
                # Logic for Signal
                direction = "LONG" if is_bull else "SHORT"
                icon = "🟢" if is_bull else "🔴"
                entry = last['close']
                stop = last['apex_stop']
                risk = abs(entry - stop)
                target = entry + (risk * 2) if is_bull else entry - (risk * 2)
                
                msg = f"""
*APEX SIGNAL DETECTED* {icon}
Sym: {symbol} [{timeframe}]
Dir: *{direction}*
Flux: {flux_lbl} ({flux_val:.2f})
-------------------
Entry: `{fmt_price(entry)}`
Stop:  `{fmt_price(stop)}`
Target: `{fmt_price(target)}` (2R)
-------------------
#Apex #Titan #Crypto
"""
                st.code(msg, language="markdown")
                if tg_active:
                    if send_telegram_msg(bot_token, chat_id, msg):
                        st.success("Sent to Telegram")
                    else:
                        st.error("Telegram Failed")
        
        with b2:
            if st.button("🧠 AI TACTIC"):
                with st.spinner("Processing Apex Data..."):
                    summary = f"""
                    Symbol: {symbol}
                    Price: {last['close']}
                    Trend: {trend_str}
                    Apex Flux: {flux_val:.2f} ({flux_lbl})
                    Trailing Stop: {last['apex_stop']}
                    Recent Structure: {smc_event}
                    Vol Factor: {last['vol_fact']:.2f}
                    """
                    analysis = get_ai_analysis(summary, ai_key)
                    st.markdown(f'<div class="ai-box">{analysis}</div>', unsafe_allow_html=True)

    # --- TAB 2: DEEP DIVE ---
    with tab2:
        # TradingView Widget
        tv_sym = f"KRAKEN:{symbol.replace('/','')}"
        components.html(f"""
        <div class="tradingview-widget-container">
          <div id="tradingview_123"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {{
            "width": "100%",
            "height": 400,
            "symbol": "{tv_sym}",
            "interval": "5",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "hide_side_toolbar": false,
            "container_id": "tradingview_123"
          }}
          );
          </script>
        </div>
        """, height=410)
        
        st.subheader("Titan/Apex Technicals")
        
        # Plotly Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # CANDLES
        fig.add_trace(go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], 
            low=df['low'], close=df['close'], name='Price'
        ), row=1, col=1)
        
        # HMA CLOUD
        # We fill between Upper and Lower
        # Trick: use Scatter with fill='tonexty'
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['trend_upper'], 
            mode='lines', line=dict(width=0), showlegend=False
        ), row=1, col=1)
        
        cloud_color = 'rgba(0, 230, 118, 0.1)' if is_bull else 'rgba(255, 23, 68, 0.1)'
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['trend_lower'], 
            mode='lines', line=dict(width=0), fill='tonexty', 
            fillcolor=cloud_color, showlegend=False
        ), row=1, col=1)
        
        # TRAILING STOP
        stop_col = '#00E676' if is_bull else '#FF1744'
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['apex_stop'],
            mode='lines', line=dict(color=stop_col, width=2),
            name="Apex Stop"
        ), row=1, col=1)
        
        # APEX FLUX HISTOGRAM
        # Color logic for bars
        colors = []
        for c in df['flux_state_code']:
            if c == 2: colors.append('#00E676') # Super Bull
            elif c == -2: colors.append('#FF1744') # Super Bear
            elif c == 0: colors.append('#546E7A') # Resistive
            else: colors.append('#FFD600') # Heat
            
        fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['apex_flux'],
            marker_color=colors, name="Flux Vector"
        ), row=2, col=1)
        
        # Threshold Lines
        fig.add_hline(y=0.6, line_dash="dot", line_color="#333", row=2, col=1)
        fig.add_hline(y=-0.6, line_dash="dot", line_color="#333", row=2, col=1)
        
        # Layout
        fig.update_layout(
            height=600, 
            margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor='#000',
            plot_bgcolor='#050505',
            xaxis_rangeslider_visible=False,
            showlegend=False,
            font=dict(family="Roboto Mono", color="#888")
        )
        fig.update_xaxes(showgrid=False, gridcolor='#222')
        fig.update_yaxes(showgrid=True, gridcolor='#222')
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Awaiting Configuration... Please check symbols or API connection.")
