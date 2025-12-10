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
# 3. DATA FETCHING (ENHANCED)
# ------------------------------------------------------------------
@st.cache_data(ttl=3600) # Cache data for 1 hour to speed up re-runs
def get_deep_financial_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # 1. Basic Screening Filter
        market_cap = info.get('marketCap', 0)
        if market_cap < 2_000_000_000: return None # Skip small caps
        
        # 2. Fetch Raw Statements for Deep Analysis
        # Note: yfinance financials calls can be slow, so we handle errors gracefully
        try:
            fin = stock.financials
            bal = stock.balance_sheet
            cash = stock.cashflow
            has_stmts = True if (not fin.empty) else False
        except:
            fin, bal, cash = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            has_stmts = False

        # 3. Calculate CAGR (3-Year Sales) if statements exist
        sales_cagr_3y = 0.0
        if has_stmts and "Total Revenue" in fin.index:
            try:
                # Get columns (dates) sorted ascending
                cols = sorted(fin.columns)
                if len(cols) >= 4:
                    rev_now = fin[cols[-1]].loc["Total Revenue"]
                    rev_3yr = fin[cols[-4]].loc["Total Revenue"]
                    sales_cagr_3y = (rev_now / rev_3yr)**(1/3) - 1
                elif len(cols) >= 2:
                    # Fallback to 1 year if 3 years not available
                    rev_now = fin[cols[-1]].loc["Total Revenue"]
                    rev_prev = fin[cols[0]].loc["Total Revenue"]
                    sales_cagr_3y = (rev_now / rev_prev) - 1
            except:
                sales_cagr_3y = info.get('revenueGrowth', 0)
        else:
             sales_cagr_3y = info.get('revenueGrowth', 0)

        # 4. Build Data Dictionary
        data = {
            "ticker": ticker_symbol,
            "name": info.get('longName', ticker_symbol),
            "country": info.get('country', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "market_cap": market_cap,
            "forward_pe": info.get('forwardPE', 0),
            "trailing_pe": info.get('trailingPE', 0),
            "revenue_growth_1y": info.get('revenueGrowth', 0),
            "revenue_cagr_3y": sales_cagr_3y,
            "eps_growth": info.get('earningsGrowth', 0),
            "debt_to_equity": info.get('debtToEquity', 999),
            "price_to_sales": info.get('priceToSalesTrailing12Months', 999),
            "dividend_yield": info.get('dividendYield', 0),
            "price_to_book": info.get('priceToBook', 999),
            "peg_ratio": info.get('pegRatio', 999),
            "profit_margin": info.get('profitMargins', 0),
            "beta": info.get('beta', 1.0),
            "employees": info.get('fullTimeEmployees', 0),
            "avg_volume": info.get('averageVolume', 0),
            "stock_obj": stock
        }

        # FCF Calculation
        try:
            fcf = info.get('freeCashflow', None)
            if fcf is None and has_stmts and "Free Cash Flow" in cash.index:
                 fcf = cash.iloc[0]["Free Cash Flow"] # Most recent
            
            if fcf and fcf > 0:
                data["p_fcf"] = market_cap / fcf
                data["fcf_share"] = fcf / info.get('sharesOutstanding', 1)
            else:
                data["p_fcf"] = 999
                data["fcf_share"] = 0
        except:
            data["p_fcf"] = 999
            data["fcf_share"] = 0

        # Insider Placeholder (Rarely available in free API)
        data["insider_percent"] = info.get('heldPercentInsiders', 0)

        return data
    except Exception as e:
        return None

def get_price_history(stock_obj):
    """Fetches 1 year of history for calculations"""
    try:
        # Fetch slightly more than needed to ensure SMA200 is valid
        hist = stock_obj.history(period="1y") 
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
        
        # 1. Get Fundamentals
        data = get_deep_financial_data(ticker)
        if not data: continue
            
        # 2. Get Price History & Technicals
        hist = get_price_history(data['stock_obj'])
        technicals = calculate_technicals(hist)
        
        # Merge Technicals into Data
        data.update(technicals)

        # 3. Apply Filters
        # Use safe defaults for None values
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

        # List Logic
        l1 = (pe < 25) and (rev_growth > 0.05)
        l2 = (eps_growth > 0.25) and (de < 15) # Relaxed DE slightly for realism
        l3 = (ps < 4) and (insider > 0.50) # Relaxed Insider for realism
        l4 = (div > 0.04) and (pb < 1.5) # Relaxed PB slightly
        l5 = (peg < 2) and (margin > 0.20)
        l6 = (p_fcf < 20) and (div > 0.02) # Relaxed P/FCF
        l7 = (2e9 <= mkt_cap <= 20e9) and (pe < 20) and (eps_growth > 0.15)

        if any([l1, l2, l3, l4, l5, l6, l7]):
            # Calculate Performance Dates
            p_jan = get_price_at_date(hist, "2025-01-01")
            p_apr = get_price_at_date(hist, "2025-04-01")
            p_jun = get_price_at_date(hist, "2025-06-20")
            p_dec24 = get_price_at_date(hist, "2024-12-31")
            
            curr = p_jun if p_jun else 0
            
            perf_jan = ((curr - p_jan) / p_jan) if p_jan else 0
            perf_apr = ((curr - p_apr) / p_apr) if p_apr else 0

            # Add to results
            row = data.copy()
            del row['stock_obj']
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
    
    # Construct a rich prompt with Technicals + Fundamentals
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
    
    Example:
    Buy (Undervalued) | Strong cash flows justify the low P/E. | Trend is bullish with RSI neutral, suggesting entry. | Risk of regulation in EU.
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
if st.button("🚀 Run Institutional Analysis"):
    if not api_key:
        st.error("Please provide OpenAI API Key.")
    else:
        st.subheader("1. Data Extraction & Quantitative Screening")
        
        # 1. Run Screen
        df = run_screening(GLOBAL_UNIVERSE)
        
        if df.empty:
            st.warning("Strict filters returned 0 stocks. Try loosening criteria in code.")
        else:
            # 2. Ranking Logic
            # Rank Jan Performance
            top_jan = df.sort_values(by='perf_since_jan', ascending=False).head(15)
            # Rank Apr Performance
            top_apr = df.sort_values(by='perf_since_apr', ascending=False).head(15)
            
            shortlist = pd.concat([top_jan, top_apr]).drop_duplicates(subset='ticker')
            
            # Final Sort by Growth Potential (Rev CAGR)
            shortlist = shortlist.sort_values(by='revenue_cagr_3y', ascending=False)
            
            # Select Unique 10
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
            st.dataframe(final_df[['ticker', 'name', 'country', 'forward_pe', 'RSI_14', 'Trend']])
            
            # 3. AI Analysis Loop
            st.subheader("2. Generating Analyst Memos...")
            ai_data = []
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
            
            # 4. Excel Formatting (Detailed)
            st.subheader("3. Finalizing Report")
            
            output_df = pd.DataFrame()
            # Info
            output_df['Ticker'] = final_df['ticker']
            output_df['Name'] = final_df['name']
            output_df['Country'] = final_df['country']
            output_df['Industry'] = final_df['industry']
            
            # Prices
            output_df['Price (Current)'] = final_df['price_jun25']
            output_df['Price (Jan 1)'] = final_df['price_jan25']
            output_df['Perf YTD'] = final_df['perf_since_jan']
            
            # Fundamentals
            output_df['P/E (Fwd)'] = final_df['forward_pe']
            output_df['PEG'] = final_df['peg_ratio']
            output_df['P/S'] = final_df['price_to_sales']
            output_df['P/FCF'] = final_df['p_fcf']
            output_df['Rev CAGR (3Y)'] = final_df['revenue_cagr_3y']
            output_df['Margin'] = final_df['profit_margin']
            output_df['Debt/Eq'] = final_df['debt_to_equity']
            
            # Technicals
            output_df['RSI (14)'] = final_df['RSI_14']
            output_df['Trend Status'] = final_df['Trend']
            output_df['Volatility'] = final_df['Volatility']
            output_df['SMA 50'] = final_df['SMA_50']
            output_df['SMA 200'] = final_df['SMA_200']
            
            # AI Insights
            output_df['Analyst Verdict'] = final_df['AI_Verdict']
            output_df['Fundamental Thesis'] = final_df['AI_Thesis']
            output_df['Technical Commentary'] = final_df['AI_Technical_View']
            output_df['Risks & AI Exposure'] = final_df['AI_Risks']

            # Buffer
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                output_df.to_excel(writer, index=False, sheet_name="Deep Analysis")
                wb = writer.book
                ws = writer.sheets["Deep Analysis"]
                
                # Formatting
                fmt_pct = wb.add_format({'num_format': '0.00%'})
                fmt_curr = wb.add_format({'num_format': '$#,##0.00'})
                
                # Apply formats
                ws.set_column('E:F', 12, fmt_curr) # Prices
                ws.set_column('G:G', 10, fmt_pct)  # Perf
                ws.set_column('L:M', 10, fmt_pct)  # Growth/Margin
                ws.set_column('P:P', 25)           # Trend Status
                ws.set_column('T:W', 40)           # AI Text
            
            buffer.seek(0)
            file_data = buffer.getvalue()
            fname = f"Deep_Analysis_{date.today()}.xlsx"
            
            # 5. Delivery
            if use_telegram:
                st.info("Sending encrypted report to Telegram...")
                top_pick = final_df.iloc[0]['name']
                msg = f"🔔 **Institutional Alert**\n\nAnalysis complete for {len(final_df)} stocks.\n🔥 **Top High-Conviction Pick:** {top_pick}\n\nMetrics:\n- RSI: {final_df.iloc[0]['RSI_14']}\n- Trend: {final_df.iloc[0]['Trend']}\n- AI Verdict: {final_df.iloc[0]['AI_Verdict']}\n\nFull report attached."
                
                send_buffer = io.BytesIO(file_data)
                if send_telegram_package(tele_token, tele_chat_id, msg, send_buffer, fname):
                    st.success("Telegram Sent!")
            
            st.download_button("📥 Download Institutional Report", data=file_data, file_name=fname)
            
st.markdown("---")
st.caption("Disclaimer: This tool provides technical and fundamental data for informational purposes. It is not financial advice. Technical indicators (RSI, SMA) are calculated on 1-year daily closing data.")
