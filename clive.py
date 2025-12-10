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
st.set_page_config(page_title="AI Institutional Analyst", layout="wide")

st.title("🤖 AI Institutional Investment Analyst")
st.markdown("""
**Deep-Dive Version:** This agent performs institutional-grade analysis by combining:
1.  **Raw Fundamental Extraction:** 3-Year CAGR, granular margins, and balance sheet health.
2.  **Technical Analysis Engine:** RSI, Moving Averages (50/200), and Volatility checks.
3.  **Generative AI reasoning:** Synthesizes quant data into a professional buy-side memo.
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
# 1. UNIVERSE DEFINITION
# ------------------------------------------------------------------
GLOBAL_UNIVERSE = [
    # North America
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "JNJ", "PG", "LLY", "AVGO",
    # Europe
    "NESN.SW", "NOVN.SW", "ROG.SW", "ASML.AS", "MC.PA", "OR.PA", "SAP.DE", "SIE.DE", "AIR.PA",
    "AZN.L", "HSBA.L", "SHEL.L", "EQNR.OL", "NOVO-B.CO", "LIN",
    # Japan
    "7203.T", "6758.T", "8035.T", "9984.T", "6861.T", "6098.T", "4063.T",
    # Hong Kong / China
    "0700.HK", "09988.HK", "03690.HK", "01299.HK", "00941.HK", "BYDDF",
    # Singapore
    "D05.SI", "O39.SI", "U11.SI",
    # Australasia
    "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WDS.AX",
    # South Africa
    "NPN.JO", "FSR.JO", "SBK.JO", "SOL.JO"
]

# ------------------------------------------------------------------
# 2. TECHNICAL ANALYSIS ENGINE
# ------------------------------------------------------------------
def calculate_technicals(hist):
    """
    Calculates RSI, SMA50, SMA200, and Volatility from price history.
    Expects a DataFrame with a 'Close' column.
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
        trend = "Bullish (Golden Cross active)"
    else:
        trend = "Bearish (Death Cross active)"
        
    # 4. Volatility (Annualized Standard Deviation of daily returns)
    daily_returns = hist['Close'].pct_change()
    volatility = daily_returns.std() * np.sqrt(252) # Annualized

    return {
        "RSI_14": round(current_rsi, 2),
        "SMA_50": round(sma_50, 2),
        "SMA_200": round(sma_200, 2),
        "Trend": trend,
        "Volatility": f"{round(volatility*100, 1)}%"
    }

# ------------------------------------------------------------------
# 3. DATA FETCHING (FIXED CACHING)
# ------------------------------------------------------------------
@st.cache_data(ttl=3600) 
def get_deep_financial_data(ticker_symbol):
    """
    Fetches fundamental data. 
    CRITICAL FIX: Does NOT return the 'stock_obj' (yfinance Ticker) 
    because it is not serializable.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # 1. Basic Screening Filter
        market_cap = info.get('marketCap', 0)
        if market_cap < 2_000_000_000: return None # Skip small caps
        
        # 2. Fetch Raw Statements for Deep Analysis
        try:
            fin = stock.financials
            cash = stock.cashflow
            has_stmts = True if (not fin.empty) else False
        except:
            fin, cash = pd.DataFrame(), pd.DataFrame()
            has_stmts = False

        # 3. Calculate CAGR (3-Year Sales)
        sales_cagr_3y = 0.0
        if has_stmts and "Total Revenue" in fin.index:
            try:
                cols = sorted(fin.columns)
                if len(cols) >= 4:
                    rev_now = fin[cols[-1]].loc["Total Revenue"]
                    rev_3yr = fin[cols[-4]].loc["Total Revenue"]
                    sales_cagr_3y = (rev_now / rev_3yr)**(1/3) - 1
                elif len(cols) >= 2:
                    rev_now = fin[cols[-1]].loc["Total Revenue"]
                    rev_prev = fin[cols[0]].loc["Total Revenue"]
                    sales_cagr_3y = (rev_now / rev_prev) - 1
            except:
                sales_cagr_3y = info.get('revenueGrowth', 0)
        else:
             sales_cagr_3y = info.get('revenueGrowth', 0)

        # 4. Build Data Dictionary (ONLY SERIALIZABLE TYPES)
        data = {
            "ticker": ticker_symbol,
            "name": info.get('longName', ticker_symbol),
            "country": info.get('country', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "market_cap": market_cap,
            "forward_pe": info.get('forwardPE', 0),
            "revenue_cagr_3y": sales_cagr_3y,
            "eps_growth": info.get('earningsGrowth', 0),
            "debt_to_equity": info.get('debtToEquity', 999),
            "price_to_sales": info.get('priceToSalesTrailing12Months', 999),
            "dividend_yield": info.get('dividendYield', 0),
            "price_to_book": info.get('priceToBook', 999),
            "peg_ratio": info.get('pegRatio', 999),
            "profit_margin": info.get('profitMargins', 0),
            "employees": info.get('fullTimeEmployees', 0),
            "avg_volume": info.get('averageVolume', 0),
            # REMOVED "stock_obj": stock  <-- THIS CAUSED THE ERROR
        }

        # FCF Calculation
        try:
            fcf = info.get('freeCashflow', None)
            if fcf is None and has_stmts and "Free Cash Flow" in cash.index:
                 fcf = cash.iloc[0]["Free Cash Flow"] 
            
            if fcf and fcf > 0:
                data["p_fcf"] = market_cap / fcf
                data["fcf_share"] = fcf / info.get('sharesOutstanding', 1)
            else:
                data["p_fcf"] = 999
                data["fcf_share"] = 0
        except:
            data["p_fcf"] = 999
            data["fcf_share"] = 0

        data["insider_percent"] = info.get('heldPercentInsiders', 0)

        return data
    except Exception as e:
        return None

def get_price_history(ticker_symbol):
    """Fetches 1 year of history for calculations. Creates new Ticker obj."""
    try:
        stock = yf.Ticker(ticker_symbol) # Instantiate locally
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
# 4. SCREENING LOGIC
# ------------------------------------------------------------------
def run_screening(universe):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    total = len(universe)
    for i, ticker in enumerate(universe):
        status_text.text(f"Deep Scanning: {ticker} ({i+1}/{total})")
        progress_bar.progress((i + 1) / total)
        
        # 1. Get Fundamentals (Cached)
        data = get_deep_financial_data(ticker)
        if not data: continue
            
        # 2. Get Price History & Technicals (Live)
        # We pass the string ticker, NOT a stock object
        hist = get_price_history(data['ticker']) 
        technicals = calculate_technicals(hist)
        
        # Merge Technicals into Data
        data.update(technicals)

        # 3. Apply Filters
        pe = data['forward_pe'] or 999
        rev_growth = data['revenue_cagr_3y'] or -1.0
        eps_growth = data['eps_growth'] or -1.0
        de = data['debt_to_equity'] or 999
        ps = data['price_to_sales'] or 999
        insider = data['insider_percent'] or 0
        div = data['dividend_yield'] or 0
        pb = data['price_to_book'] or 999
        peg = data['peg_ratio'] or 999
        margin = data['profit_margin'] or 0
        p_fcf = data['p_fcf'] or 999
        mkt_cap = data['market_cap']

        l1 = (pe < 25) and (rev_growth > 0.05)
        l2 = (eps_growth > 0.25) and (de < 15)
        l3 = (ps < 4) and (insider > 0.50)
        l4 = (div > 0.04) and (pb < 1.5)
        l5 = (peg < 2) and (margin > 0.20)
        l6 = (p_fcf < 20) and (div > 0.02)
        l7 = (2e9 <= mkt_cap <= 20e9) and (pe < 20) and (eps_growth > 0.15)

        if any([l1, l2, l3, l4, l5, l6, l7]):
            p_jan = get_price_at_date(hist, "2025-01-01")
            p_apr = get_price_at_date(hist, "2025-04-01")
            p_jun = get_price_at_date(hist, "2025-06-20")
            p_dec24 = get_price_at_date(hist, "2024-12-31")
            
            curr = p_jun if p_jun else 0
            
            perf_jan = ((curr - p_jan) / p_jan) if p_jan else 0
            perf_apr = ((curr - p_apr) / p_apr) if p_apr else 0

            row = data.copy()
            row.update({
                "price_dec24": p_dec24,
                "price_jan25": p_jan,
                "price_apr25": p_apr,
                "price_jun25": p_jun,
                "perf_since_jan": perf_jan,
                "perf_since_apr": perf_apr
            })
            results.append(row)
            
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

# ------------------------------------------------------------------
# 5. AI ANALYST (DEEP DIVE)
# ------------------------------------------------------------------
def analyze_with_ai_deep(row, api_key):
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    Act as a Senior Portfolio Manager. Analyze this stock deeply.
    
    [PROFILE]
    Name: {row['name']} ({row['ticker']})
    Region: {row['country']} | Industry: {row['industry']}
    
    [VALUATION]
    Forward P/E: {row['forward_pe']} | PEG: {row['peg_ratio']} | P/B: {row['price_to_book']}
    P/FCF: {row['p_fcf']} | Div Yield: {row['dividend_yield']*100:.2f}%
    
    [GROWTH & HEALTH]
    Rev CAGR (3Y): {row['revenue_cagr_3y']*100:.2f}% | EPS Growth (1Y): {row['eps_growth']*100:.2f}%
    Profit Margin: {row['profit_margin']*100:.2f}% | Debt/Equity: {row['debt_to_equity']}
    
    [TECHNICALS]
    Price Trend: {row['Trend']}
    RSI (14): {row['RSI_14']} (Overbought > 70, Oversold < 30)
    Volatility (Ann): {row['Volatility']}
    
    OUTPUT REQUIREMENTS:
    Provide 4 sections separated STRICTLY by the pipe symbol "|".
    1. VERDICT: Buy, Hold, or Sell (with a short justification).
    2. FUNDAMENTAL THESIS: 2 sentences on valuation/growth balance.
    3. TECHNICAL VIEW: 1 sentence interpreting RSI and Trend.
    4. RISKS/CATALYSTS: 1 sentence on key upcoming risks or AI exposure.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a concise, high-level investment analyst."},
                      {"role": "user", "content": prompt}],
            temperature=0.5
        )
        content = response.choices[0].message.content
        parts = content.split('|')
        if len(parts) < 4: return ["Hold", "AI Parsing Error", "AI Parsing Error", "AI Parsing Error"]
        return [p.strip() for p in parts[:4]]
    except Exception as e:
        return ["Error", f"API Error: {str(e)}", "", ""]

# ------------------------------------------------------------------
# 6. TELEGRAM SENDER
# ------------------------------------------------------------------
def send_telegram_package(token, chat_id, text, excel_buffer, filename):
    try:
        # 1. Text
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        )
        # 2. File
        excel_buffer.seek(0)
        requests.post(
            f"https://api.telegram.org/bot{token}/sendDocument",
            data={"chat_id": chat_id, "caption": "📈 Deep-Dive Analysis File"},
            files={"document": (filename, excel_buffer, "application/vnd.ms-excel")}
        )
        return True
    except Exception as e:
        st.error(f"Telegram Error: {e}")
        return False

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "final_df" not in st.session_state:
    st.session_state.final_df = None
if "excel_data" not in st.session_state:
    st.session_state.excel_data = None

if st.button("🚀 Run Institutional Analysis"):
    if not api_key:
        st.error("Please provide OpenAI API Key.")
    else:
        st.subheader("1. Data Extraction & Quantitative Screening")
        
        df = run_screening(GLOBAL_UNIVERSE)
        
        if df.empty:
            st.warning("Strict filters returned 0 stocks. Try loosening criteria in code.")
        else:
            # Ranking Logic
            top_jan = df.sort_values(by='perf_since_jan', ascending=False).head(15)
            top_apr = df.sort_values(by='perf_since_apr', ascending=False).head(15)
            shortlist = pd.concat([top_jan, top_apr]).drop_duplicates(subset='ticker')
            shortlist = shortlist.sort_values(by='revenue_cagr_3y', ascending=False)
            
            final_rows = []
            seen = set()
            for _, row in shortlist.iterrows():
                k = (row['country'], row['industry'])
                if k not in seen:
                    final_rows.append(row)
                    seen.add(k)
                if len(final_rows) >= 10: break
            
            final_df = pd.DataFrame(final_rows)
            
            st.success(f"Identified {len(final_df)} High-Conviction Candidates.")
            
            # AI Analysis Loop
            st.subheader("2. Generating Analyst Memos...")
            prog = st.progress(0)
            for i, idx in enumerate(final_df.index):
                row = final_df.loc[idx]
                insights = analyze_with_ai_deep(row, api_key)
                final_df.loc[idx, 'AI_Verdict'] = insights[0]
                final_df.loc[idx, 'AI_Thesis'] = insights[1]
                final_df.loc[idx, 'AI_Technical_View'] = insights[2]
                final_df.loc[idx, 'AI_Risks'] = insights[3]
                prog.progress((i+1)/len(final_df))
            prog.empty()
            
            # Store in Session State
            st.session_state.final_df = final_df
            st.session_state.analysis_done = True
            
            # Create Excel for Session
            output_df = pd.DataFrame()
            output_df['Ticker'] = final_df['ticker']
            output_df['Name'] = final_df['name']
            output_df['Country'] = final_df['country']
            output_df['Industry'] = final_df['industry']
            output_df['Price (Current)'] = final_df['price_jun25']
            output_df['Perf YTD'] = final_df['perf_since_jan']
            output_df['P/E (Fwd)'] = final_df['forward_pe']
            output_df['RSI (14)'] = final_df['RSI_14']
            output_df['Trend Status'] = final_df['Trend']
            output_df['Analyst Verdict'] = final_df['AI_Verdict']
            output_df['Fundamental Thesis'] = final_df['AI_Thesis']
            output_df['Risks'] = final_df['AI_Risks']

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                output_df.to_excel(writer, index=False, sheet_name="Deep Analysis")
            buffer.seek(0)
            st.session_state.excel_data = buffer.getvalue()

# ------------------------------------------------------------------
# DISPLAY RESULTS & SIGNALS
# ------------------------------------------------------------------
if st.session_state.analysis_done and st.session_state.final_df is not None:
    final_df = st.session_state.final_df
    
    st.write("### 🏆 Top Institutional Picks")
    st.dataframe(final_df[['ticker', 'name', 'country', 'forward_pe', 'RSI_14', 'Trend', 'AI_Verdict']])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fname = f"Deep_Analysis_{date.today()}.xlsx"
        st.download_button("📥 Download Excel Report", data=st.session_state.excel_data, file_name=fname)

    with col2:
        if use_telegram:
            # TELEGRAM SIGNAL BUTTON
            if st.button("📡 Broadcast Signal to Telegram"):
                st.info("Sending Signal...")
                top_stock = final_df.iloc[0]
                
                signal_msg = f"""
🚨 **DARK POOL AGENT SIGNAL** 🚨

**Top Pick:** {top_stock['name']} ({top_stock['ticker']})
**Verdict:** {top_stock['AI_Verdict']}

**Technical Profile:**
• Price: {top_stock['price_jun25']:.2f}
• RSI (14): {top_stock['RSI_14']}
• Trend: {top_stock['Trend']}

**Thesis:** {top_stock['AI_Thesis']}
"""
                send_buffer = io.BytesIO(st.session_state.excel_data)
                
                if send_telegram_package(tele_token, tele_chat_id, signal_msg, send_buffer, "Signal_Report.xlsx"):
                    st.success("✅ Signal Broadcasted!")
                else:
                    st.error("❌ Failed to broadcast.")
        else:
             st.warning("Configure Telegram secrets to enable Signals.")

st.markdown("---")
st.caption("Disclaimer: This tool provides technical and fundamental data for informational purposes.")
