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
st.set_page_config(page_title="AI Junior Mining Analyst", layout="wide")

st.title("⛏️ AI Junior Mining & Resource Analyst")
st.markdown("""
**Sector Specialist:** This agent focuses exclusively on **Junior & Mid-Tier Miners**.
It adapts standard financial screening to resource-specific metrics:
1.  **Survival Filters:** Focus on Balance Sheet health (Cash/Debt) over P/E.
2.  **Technicals:** Volatility and Momentum scanning for breakout drills.
3.  **AI Geologist:** Analyzes jurisdiction, commodity exposure, and drill potential.
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
# 1. MINING UNIVERSE (High-Potential Juniors/Mid-Tiers)
# ------------------------------------------------------------------
# Focus: TSX-V (Canada), ASX (Australia), NYSE American
MINING_UNIVERSE = [
    # Uranium (High Beta)
    "NXE", "UEC", "UUUU", "DNN", "PDN.AX", "BOE.AX", "GLO.TO",
    # Lithium / Battery Metals
    "LAC", "SGML", "PLS.AX", "CXO.AX", "SYA.AX", "LTR.AX", "PMET.TO", "CRE.TO",
    # Gold / Silver Juniors
    "KGC", "EQX", "NGD", "SILV", "MAG", "SVM", "GREG.L", "CMM.AX", "PRU.AX", "WAF.AX",
    # Copper / Base Metals
    "ERO", "IVN.TO", "HBM", "CAM.TO", "FM.TO", "ALS.TO", "SFR.AX", "29M.AX",
    # Rare Earths / Strategic
    "MP", "LYC.AX", "ARU.AX", "ASM.AX"
]

# ------------------------------------------------------------------
# 2. TECHNICAL ANALYSIS ENGINE
# ------------------------------------------------------------------
def calculate_technicals(hist):
    """
    Calculates RSI, SMA50, SMA200, and Volatility.
    """
    if hist is None or len(hist) < 200:
        return {
            "RSI_14": None, "SMA_50": None, "SMA_200": None, 
            "Trend": "Insufficient Data", "Volatility": None
        }
    
    # 1. RSI (14-day)
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # 2. Moving Averages
    sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
    sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
    current_price = hist['Close'].iloc[-1]
    
    # 3. Trend Identification
    if current_price > sma_50 > sma_200:
        trend = "Strong Bullish"
    elif current_price < sma_50 < sma_200:
        trend = "Strong Bearish"
    elif sma_50 > sma_200:
        trend = "Bullish (Golden Cross)"
    else:
        trend = "Bearish (Death Cross)"
        
    # 4. Volatility (Annualized)
    daily_returns = hist['Close'].pct_change()
    volatility = daily_returns.std() * np.sqrt(252)

    return {
        "RSI_14": round(current_rsi, 2),
        "SMA_50": round(sma_50, 2),
        "SMA_200": round(sma_200, 2),
        "Trend": trend,
        "Volatility": f"{round(volatility*100, 1)}%"
    }

# ------------------------------------------------------------------
# 3. DATA FETCHING (MINING OPTIMIZED)
# ------------------------------------------------------------------
@st.cache_data(ttl=3600) 
def get_mining_financials(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # MINING FILTER: Market Cap
        # Juniors can be small. Range: $50M to $10B (Mid-Cap)
        market_cap = info.get('marketCap', 0)
        if market_cap < 50_000_000: return None # Filter ultra-micro caps
        if market_cap > 15_000_000_000: return None # Filter Majors (BHP, Rio)

        # Extraction
        data = {
            "ticker": ticker_symbol,
            "name": info.get('longName', ticker_symbol),
            "country": info.get('country', 'Unknown'),
            "sector": "Mining/Materials",
            "market_cap": market_cap,
            "current_price": info.get('currentPrice', 0),
            
            # Fundamentals (Critical for Miners)
            "price_to_book": info.get('priceToBook', 999),
            "debt_to_equity": info.get('debtToEquity', 999),
            "total_cash": info.get('totalCash', 0),
            "total_debt": info.get('totalDebt', 0),
            "quick_ratio": info.get('quickRatio', 0), # Liquidity check
            
            # Growth/Valuation (Often N/A for juniors)
            "revenue_growth": info.get('revenueGrowth', 0),
            "gross_margins": info.get('grossMargins', 0),
            
            # Trading
            "avg_volume": info.get('averageVolume', 0)
        }
        
        # Net Cash Calculation (Cash - Debt)
        # Positive Net Cash is huge for juniors (no dilution risk)
        try:
            net_cash = data["total_cash"] - data["total_debt"]
            data["net_cash_pos"] = net_cash > 0
        except:
            data["net_cash_pos"] = False

        return data
    except Exception as e:
        return None

def get_price_history(ticker_symbol):
    """Fetches 1 year of history."""
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y") 
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
# 4. MINING SCREENING LOGIC
# ------------------------------------------------------------------
def run_mining_screen(universe):
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    total = len(universe)
    for i, ticker in enumerate(universe):
        status_text.text(f"Assaying: {ticker} ({i+1}/{total})")
        progress_bar.progress((i + 1) / total)
        
        # 1. Fundamentals
        data = get_mining_financials(ticker)
        if not data: continue
            
        # 2. Technicals
        hist = get_price_history(data['ticker']) 
        technicals = calculate_technicals(hist)
        data.update(technicals)

        # 3. Junior Mining Filters
        # We don't use P/E. We use Solvency and Book Value.
        
        pb = data['price_to_book'] or 999
        de = data['debt_to_equity'] or 999
        quick = data['quick_ratio'] or 0
        mkt_cap = data['market_cap']
        
        # List 1: Deep Value Junior
        # Trading near book value + Solvent
        l1 = (pb < 1.5) and (quick > 1.0)
        
        # List 2: Cash Rich (Exploration upside without dilution)
        l2 = data['net_cash_pos'] and (mkt_cap < 2_000_000_000)
        
        # List 3: Momentum Play (Breakout)
        l3 = (data['Trend'] == "Strong Bullish") and (data['RSI_14'] < 80)
        
        # List 4: Mid-Tier Growth (Revenue > 0)
        l4 = (data['revenue_growth'] > 0.10) and (mkt_cap > 1_000_000_000)

        if any([l1, l2, l3, l4]):
            p_jan = get_price_at_date(hist, "2025-01-01")
            p_apr = get_price_at_date(hist, "2025-04-01")
            curr = data['current_price']
            
            perf_jan = ((curr - p_jan) / p_jan) if p_jan else 0
            
            row = data.copy()
            row.update({
                "price_jan25": p_jan,
                "perf_since_jan": perf_jan,
                "matched_criteria": [k for k,v in zip(["Value","Cash","Momentum","Growth"], [l1,l2,l3,l4]) if v]
            })
            results.append(row)
            
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

# ------------------------------------------------------------------
# 5. AI GEOLOGIST ANALYST
# ------------------------------------------------------------------
def analyze_miner_with_ai(row, api_key):
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    Act as a specialized Mining Investment Analyst. Analyze this resource company.
    
    [PROFILE]
    Name: {row['name']} ({row['ticker']})
    Market Cap: ${row['market_cap'] / 1e6:.0f}M
    Country: {row['country']}
    
    [FINANCIAL HEALTH]
    Price/Book: {row['price_to_book']}
    Debt/Equity: {row['debt_to_equity']}
    Net Cash Positive: {row['net_cash_pos']}
    Quick Ratio: {row['quick_ratio']}
    
    [TECHNICALS]
    Trend: {row['Trend']} | RSI: {row['RSI_14']}
    Volatility: {row['Volatility']}
    
    OUTPUT REQUIREMENTS (Separated by "|"):
    1. VERDICT: Buy (Speculative), Buy (Value), Hold, or Sell.
    2. ASSET QUALITY: Comment on the commodity (Gold/Lithium/etc) and Jurisdictional Risk.
    3. CATALYSTS: Comment on drill results, DFS/PFS studies, or M&A potential.
    4. RISK: One key specific risk (Dilution, Permit, Geopolitical).
    
    Example:
    Buy (Speculative) | High-grade uranium in Athabasca, safe jurisdiction. | Awaiting winter drill results in Q3. | Risk of share dilution to fund capex.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a senior mining analyst."},
                      {"role": "user", "content": prompt}],
            temperature=0.6
        )
        content = response.choices[0].message.content
        parts = content.split('|')
        if len(parts) < 4: return ["Hold", "AI Error", "AI Error", "AI Error"]
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
            data={"chat_id": chat_id, "caption": "⛏️ Mining Analysis Report"},
            files={"document": (filename, excel_buffer, "application/vnd.ms-excel")}
        )
        return True
    except Exception as e:
        st.error(f"Telegram Error: {e}")
        return False

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if "mining_done" not in st.session_state:
    st.session_state.mining_done = False
if "mining_df" not in st.session_state:
    st.session_state.mining_df = None
if "mining_excel" not in st.session_state:
    st.session_state.mining_excel = None

if st.button("⚒️ Start Drill Program (Analysis)"):
    if not api_key:
        st.error("Please provide OpenAI API Key.")
    else:
        st.subheader("1. Assaying Samples (Screening)...")
        
        df = run_mining_screen(MINING_UNIVERSE)
        
        if df.empty:
            st.warning("No miners matched the criteria.")
        else:
            # Rank by Technical Strength + Value
            # Prefer stocks with Momentum (perf) but decent Book Value
            df = df.sort_values(by='perf_since_jan', ascending=False)
            
            # Select Top 10
            final_df = df.head(10).reset_index(drop=True)
            
            st.success(f"Identified {len(final_df)} High-Grade Targets.")
            
            # AI Analysis
            st.subheader("2. Geologist Review...")
            prog = st.progress(0)
            for i, idx in enumerate(final_df.index):
                row = final_df.loc[idx]
                insights = analyze_miner_with_ai(row, api_key)
                final_df.loc[idx, 'AI_Verdict'] = insights[0]
                final_df.loc[idx, 'Asset_Jurisdiction'] = insights[1]
                final_df.loc[idx, 'Catalysts'] = insights[2]
                final_df.loc[idx, 'Key_Risk'] = insights[3]
                prog.progress((i+1)/len(final_df))
            prog.empty()
            
            st.session_state.mining_df = final_df
            st.session_state.mining_done = True
            
            # Create Excel
            output_df = pd.DataFrame()
            output_df['Ticker'] = final_df['ticker']
            output_df['Name'] = final_df['name']
            output_df['Criteria'] = final_df['matched_criteria'].apply(lambda x: ", ".join(x))
            output_df['Market Cap'] = final_df['market_cap']
            output_df['Price/Book'] = final_df['price_to_book']
            output_df['Net Cash +'] = final_df['net_cash_pos']
            output_df['RSI (14)'] = final_df['RSI_14']
            output_df['Verdict'] = final_df['AI_Verdict']
            output_df['Assets'] = final_df['Asset_Jurisdiction']
            output_df['Catalysts'] = final_df['Catalysts']
            output_df['Risks'] = final_df['Key_Risk']

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                output_df.to_excel(writer, index=False, sheet_name="Mining Picks")
            buffer.seek(0)
            st.session_state.mining_excel = buffer.getvalue()

# ------------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------------
if st.session_state.mining_done and st.session_state.mining_df is not None:
    final_df = st.session_state.mining_df
    
    st.write("### 💎 Top Mineral Picks")
    st.dataframe(final_df[['ticker', 'name', 'market_cap', 'AI_Verdict', 'Asset_Jurisdiction']])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fname = f"Mining_Picks_{date.today()}.xlsx"
        st.download_button("📥 Download Drill Report", data=st.session_state.mining_excel, file_name=fname)

    with col2:
        if use_telegram:
            if st.button("📡 Broadcast Alert to Telegram"):
                st.info("Transmitting...")
                top_stock = final_df.iloc[0]
                
                signal_msg = f"""
⛏️ **MINING ALERT** ⛏️

**Top Target:** {top_stock['name']} ({top_stock['ticker']})
**Verdict:** {top_stock['AI_Verdict']}

**Fundamentals:**
• Market Cap: ${top_stock['market_cap']/1e6:.1f}M
• P/B Ratio: {top_stock['price_to_book']}
• Net Cash Positive: {top_stock['net_cash_pos']}

**Catalyst:** {top_stock['Catalysts']}
"""
                send_buffer = io.BytesIO(st.session_state.mining_excel)
                if send_telegram_package(tele_token, tele_chat_id, signal_msg, send_buffer, "Mining_Report.xlsx"):
                    st.success("✅ Alert Sent!")
                else:
                    st.error("❌ Transmission Failed.")

st.markdown("---")
st.caption("Disclaimer: Junior mining is high risk. 'Net Cash' is estimated from latest filings. Always verify drill results manually.")
