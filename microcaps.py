import streamlit as st
import pandas as pd
import yfinance as yf
import openai
from datetime import datetime, date, timedelta
import io
import xlsxwriter
import requests
import numpy as np

# ------------------------------------------------------------------
# CONFIGURATION & SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="AI Micro-Cap Mining Scout", layout="wide")

st.title("💎 AI Micro-Cap Mining Scout")
st.markdown("""
**Focus: High-Risk / High-Reward Micro-Caps ($10M - $500M)**
This agent hunts for "Tenbaggers" by analyzing:
1.  **Cash Runway:** Does the company have cash to drill, or is a dilutive raise coming?
2.  **Cash Backing:** Is the stock trading close to its cash value? (Downside protection).
3.  **Explosive Technicals:** Micro-caps move on news. We scan for volume spikes.
""")

# ------------------------------------------------------------------
# SECRETS & SIDEBAR
# ------------------------------------------------------------------
st.sidebar.header("Configuration")

# 1. OpenAI API Key
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("OpenAI Key: Loaded")
else:
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# 2. Telegram Integration
use_telegram = False
if "TELEGRAM_TOKEN" in st.secrets and "TELEGRAM_CHAT_ID" in st.secrets:
    tele_token = st.secrets["TELEGRAM_TOKEN"]
    tele_chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    use_telegram = True
    st.sidebar.success("Telegram: Connected")
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Telegram Setup (Optional)")
    tele_token = st.sidebar.text_input("Bot Token", type="password")
    tele_chat_id = st.sidebar.text_input("Chat ID")
    use_telegram = bool(tele_token and tele_chat_id)

# ------------------------------------------------------------------
# 1. MICRO-CAP UNIVERSE (Explorers & Developers)
# ------------------------------------------------------------------
# Focus: TSX-V (Venture), CSE, and ASX Small Caps
MICRO_UNIVERSE = [
    # Gold/Silver Juniors (Canada/US)
    "ABRA.V", "AMX.V", "DSV.TO", "WM.TO", "GWM.V", "SBB.TO", "KRR.TO", "PGM.V", "LIO.V",
    "NFG.V", "VZLA.V", "SGD.V", "RVG.V", "DEF.V",
    # Uranium Juniors (High Volatility)
    "ISO.V", "CUR.V", "SYH.V", "LAM.TO", "AAZ.V", "CVV.V", "FCU.TO",
    # Lithium/Battery Metals (ASX/Canada)
    "PMET.TO", "CRE.TO", "LLI.AX", "GL1.AX", "AZS.AX", "VUL.AX", "INR.AX", "SYA.AX",
    # Copper/Base Metals
    "ALS.TO", "FIL.TO", "NGQ.TO", "WAR.AX", "RXM.AX", "ADN.AX"
]

# ------------------------------------------------------------------
# 2. TECHNICAL ANALYSIS ENGINE (Micro-Cap Optimized)
# ------------------------------------------------------------------
def calculate_technicals(hist):
    """
    Calculates RSI, Volatility, and Volume Spikes.
    """
    if hist is None or len(hist) < 50:
        return {
            "RSI_14": None, "Trend": "Insufficient Data", 
            "Volatility": None, "Volume_Spike": False
        }
    
    # 1. RSI (14-day)
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # 2. Trend (Simple SMA check for micro caps)
    sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
    current_price = hist['Close'].iloc[-1]
    trend = "Bullish" if current_price > sma_50 else "Bearish"
        
    # 3. Volatility (Annualized) - Crucial for Micros
    daily_returns = hist['Close'].pct_change()
    volatility = daily_returns.std() * np.sqrt(252)
    
    # 4. Volume Spike Detection (Is something happening?)
    avg_vol_20 = hist['Volume'].rolling(window=20).mean().iloc[-1]
    current_vol = hist['Volume'].iloc[-1]
    # If today's volume is 2x the average, it's a spike (news leak?)
    vol_spike = current_vol > (2 * avg_vol_20)

    return {
        "RSI_14": round(current_rsi, 2),
        "Trend": trend,
        "Volatility": f"{round(volatility*100, 1)}%",
        "Volume_Spike": vol_spike
    }

# ------------------------------------------------------------------
# 3. DATA FETCHING (MICRO CAP OPTIMIZED)
# ------------------------------------------------------------------
@st.cache_data(ttl=3600) 
def get_micro_financials(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # MICRO CAP FILTER: $10M to $500M
        market_cap = info.get('marketCap', 0)
        if market_cap < 10_000_000: return None  # Too illiquid/Penny stock risk
        if market_cap > 500_000_000: return None # Graduated to Mid-Cap

        # Extraction
        data = {
            "ticker": ticker_symbol,
            "name": info.get('longName', ticker_symbol),
            "country": info.get('country', 'Unknown'),
            "sector": "Mining Junior",
            "market_cap": market_cap,
            "current_price": info.get('currentPrice', 0),
            
            # Survival Metrics
            "total_cash": info.get('totalCash', 0),
            "total_debt": info.get('totalDebt', 0),
            "book_value": info.get('bookValue', 0),
            "price_to_book": info.get('priceToBook', 999),
            
            # Ownership (Important for micros)
            "insider_ownership": info.get('heldPercentInsiders', 0),
        }
        
        # 1. CASH BACKING RATIO
        # (Total Cash / Market Cap). 
        # If this is 0.5, then 50% of the share price is backed by cash. Very safe.
        if market_cap > 0:
            data["cash_backing"] = data["total_cash"] / market_cap
        else:
            data["cash_backing"] = 0
            
        # 2. ENTERPRISE VALUE (MCap + Debt - Cash)
        # Low EV means cheap acquisition target
        data["enterprise_value"] = market_cap + data["total_debt"] - data["total_cash"]

        return data
    except Exception as e:
        return None

def get_price_history(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="6mo") # 6 months is enough for micros
        return hist
    except:
        return None

def get_price_at_date(hist, target_date_str):
    target_date = pd.to_datetime(target_date_str).tz_localize(None)
    if hist is None or hist.empty: return None
    hist.index = hist.index.tz_localize(None)
    hist = hist.sort_index()
    try:
        idx = hist.index.get_indexer([target_date], method='nearest')[0]
        return hist.iloc[idx]['Close']
    except:
        return hist['Close'].iloc[-1]

# ------------------------------------------------------------------
# 4. MICRO SCREENING LOGIC
# ------------------------------------------------------------------
def run_micro_screen(universe):
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    total = len(universe)
    for i, ticker in enumerate(universe):
        status_text.text(f"Scouting: {ticker} ({i+1}/{total})")
        progress_bar.progress((i + 1) / total)
        
        # 1. Fundamentals
        data = get_micro_financials(ticker)
        if not data: continue
            
        # 2. Technicals
        hist = get_price_history(data['ticker']) 
        technicals = calculate_technicals(hist)
        data.update(technicals)

        # 3. MICRO CAP FILTERS
        
        cash_ratio = data['cash_backing'] # 0.0 to 1.0+
        insider = data['insider_ownership']
        pb = data['price_to_book'] or 999
        vol_spike = data['Volume_Spike']
        
        # List A: The "Cash Box" (Trading near cash value)
        # Safe play: If Cash > 20% of Mcap
        l_cash = (cash_ratio > 0.20)
        
        # List B: Insider Conviction
        # Insiders own > 10%
        l_insider = (insider > 0.10)
        
        # List C: Deep Value (Trading under Book Value)
        # Market hates it, but assets are there
        l_value = (pb < 1.0)
        
        # List D: Action (Volume Spike + Bullish)
        l_action = vol_spike and (data['Trend'] == "Bullish")

        # Keep if matches ANY criteria
        if any([l_cash, l_insider, l_value, l_action]):
            # Calc Performance
            curr = data['current_price']
            # Just take 3 month performance for micros (moves fast)
            p_start = hist['Close'].iloc[0] if (hist is not None and not hist.empty) else curr
            perf = ((curr - p_start) / p_start)

            row = data.copy()
            row.update({
                "perf_6mo": perf,
                "matched_criteria": [k for k,v in zip(["CashRich","HighInsider","DeepValue","VolSpike"], [l_cash,l_insider,l_value,l_action]) if v]
            })
            results.append(row)
            
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

# ------------------------------------------------------------------
# 5. AI SPECULATOR ANALYST
# ------------------------------------------------------------------
def analyze_micro_with_ai(row, api_key):
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    Act as a Micro-Cap Speculator (High Risk Tolerance). Analyze this junior miner.
    
    [PROFILE]
    Name: {row['name']} ({row['ticker']})
    Market Cap: ${row['market_cap'] / 1e6:.1f}M
    Cash Backing: {row['cash_backing']*100:.1f}% of Market Cap is CASH.
    Insider Ownership: {row['insider_ownership']*100:.1f}%
    
    [TECHNICALS]
    Trend: {row['Trend']} | Volatility: {row['Volatility']}
    Volume Spike Today: {row['Volume_Spike']}
    
    OUTPUT REQUIREMENTS (Separated by "|"):
    1. VERDICT: "Speculative Buy", "Watchlist", or "Avoid".
    2. CASH RUNWAY: Comment on if they need to raise money soon (Dilution Risk).
    3. BLUE SKY: What is the dream scenario? (e.g. "Next major district discovery").
    4. RED FLAG: The single biggest danger.
    
    Example:
    Speculative Buy | Solid cash position, no raise needed for 12 months. | Could define a new lithium district in Quebec. | Low liquidity, hard to exit position.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a micro-cap speculator."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content
        parts = content.split('|')
        if len(parts) < 4: return ["Watchlist", "AI Error", "AI Error", "AI Error"]
        return [p.strip() for p in parts[:4]]
    except Exception as e:
        return ["Error", f"API Error: {str(e)}", "", ""]

# ------------------------------------------------------------------
# 6. TELEGRAM SENDER
# ------------------------------------------------------------------
def send_telegram_package(token, chat_id, text, excel_buffer, filename):
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        )
        excel_buffer.seek(0)
        requests.post(
            f"https://api.telegram.org/bot{token}/sendDocument",
            data={"chat_id": chat_id, "caption": "💎 Micro-Cap Scout Report"},
            files={"document": (filename, excel_buffer, "application/vnd.ms-excel")}
        )
        return True
    except Exception as e:
        st.error(f"Telegram Error: {e}")
        return False

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if "micro_done" not in st.session_state:
    st.session_state.micro_done = False
if "micro_df" not in st.session_state:
    st.session_state.micro_df = None
if "micro_excel" not in st.session_state:
    st.session_state.micro_excel = None

if st.button("🔎 Scout for Tenbaggers"):
    if not api_key:
        st.error("Please provide OpenAI API Key.")
    else:
        st.subheader("1. Scouting Micro-Cap Universe...")
        
        df = run_micro_screen(MICRO_UNIVERSE)
        
        if df.empty:
            st.warning("No micro-caps matched criteria. (Check market hours or data availability).")
        else:
            # Sort: Prioritize "Volume Spikes" (Action) then "Cash Backing" (Safety)
            df = df.sort_values(by=['Volume_Spike', 'cash_backing'], ascending=False)
            
            # Select Top 10
            final_df = df.head(10).reset_index(drop=True)
            
            st.success(f"Found {len(final_df)} Micro-Cap Opportunities.")
            
            # AI Analysis
            st.subheader("2. Assessing Drill Potential...")
            prog = st.progress(0)
            for i, idx in enumerate(final_df.index):
                row = final_df.loc[idx]
                insights = analyze_micro_with_ai(row, api_key)
                final_df.loc[idx, 'AI_Verdict'] = insights[0]
                final_df.loc[idx, 'Cash_Runway'] = insights[1]
                final_df.loc[idx, 'Blue_Sky'] = insights[2]
                final_df.loc[idx, 'Red_Flag'] = insights[3]
                prog.progress((i+1)/len(final_df))
            prog.empty()
            
            st.session_state.micro_df = final_df
            st.session_state.micro_done = True
            
            # Create Excel
            output_df = pd.DataFrame()
            output_df['Ticker'] = final_df['ticker']
            output_df['Name'] = final_df['name']
            output_df['Tags'] = final_df['matched_criteria'].apply(lambda x: ", ".join(x))
            output_df['Market Cap'] = final_df['market_cap']
            output_df['Cash % of MCap'] = final_df['cash_backing']
            output_df['Insider %'] = final_df['insider_ownership']
            output_df['Vol Spike'] = final_df['Volume_Spike']
            output_df['AI Verdict'] = final_df['AI_Verdict']
            output_df['Dilution Risk'] = final_df['Cash_Runway']
            output_df['Blue Sky'] = final_df['Blue_Sky']
            output_df['Red Flag'] = final_df['Red_Flag']

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                output_df.to_excel(writer, index=False, sheet_name="Micro Caps")
                # Format Percentage Columns
                workbook = writer.book
                worksheet = writer.sheets['Micro Caps']
                pct_fmt = workbook.add_format({'num_format': '0.0%'})
                worksheet.set_column('E:F', 12, pct_fmt) # Cash% and Insider%

            buffer.seek(0)
            st.session_state.micro_excel = buffer.getvalue()

# ------------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------------
if st.session_state.micro_done and st.session_state.micro_df is not None:
    final_df = st.session_state.micro_df
    
    st.write("### 🧨 Top High-Risk Micro Picks")
    
    # Custom display for Micros
    for i, row in final_df.iterrows():
        with st.expander(f"{row['ticker']} - {row['name']} ({row['AI_Verdict']})"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Market Cap", f"${row['market_cap']/1e6:.1f}M")
            c2.metric("Cash Backing", f"{row['cash_backing']*100:.1f}%", help="% of share price backed by hard cash.")
            c3.metric("Volume Spike", "YES" if row['Volume_Spike'] else "No")
            st.write(f"**Dream Scenario:** {row['Blue_Sky']}")
            st.write(f"**Risk:** {row['Red_Flag']}")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        fname = f"MicroCap_Scout_{date.today()}.xlsx"
        st.download_button("📥 Download Recon Report", data=st.session_state.micro_excel, file_name=fname)

    with col2:
        if use_telegram:
            if st.button("📡 Broadcast Speculative Alert"):
                st.info("Transmitting...")
                top_stock = final_df.iloc[0]
                
                signal_msg = f"""
🧨 **MICRO-CAP SPEC ALERT** 🧨

**Target:** {top_stock['name']} ({top_stock['ticker']})
**Verdict:** {top_stock['AI_Verdict']}

**Why we like it:**
• Market Cap: ${top_stock['market_cap']/1e6:.1f}M
• Cash Backing: {top_stock['cash_backing']*100:.1f}%
• Insider Ownership: {top_stock['insider_ownership']*100:.1f}%

**Blue Sky:** {top_stock['Blue_Sky']}
**Caution:** {top_stock['Red_Flag']}
"""
                send_buffer = io.BytesIO(st.session_state.micro_excel)
                if send_telegram_package(tele_token, tele_chat_id, signal_msg, send_buffer, "MicroCap_Report.xlsx"):
                    st.success("✅ Alert Sent!")
                else:
                    st.error("❌ Transmission Failed.")

st.markdown("---")
st.caption("Disclaimer: Micro-cap stocks are extremely volatile and can lose 100% of value. 'Cash Backing' is based on last reported financials.")
