import streamlit as st
import pandas as pd
import yfinance as yf
import openai
from datetime import datetime, date
import io
import xlsxwriter
import time

# ------------------------------------------------------------------
# CONFIGURATION & SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="AI Investment Analyst Agent", layout="wide")

st.title("🤖 AI Global Investment Analyst Agent")
st.markdown("""
This agent identifies top-tier global stocks using strict quantitative filters 
and AI-powered qualitative review. 
""")

# Sidebar for API Key
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# ------------------------------------------------------------------
# 1. DEFINE THE UNIVERSE (Top-Tier Global Stocks)
# ------------------------------------------------------------------
# NOTE: Scanning the entire global market via yfinance in real-time is slow.
# We define a broad universe of major caps across required regions to screen against.
# Users can expand this list.
GLOBAL_UNIVERSE = [
    # North America
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "JNJ", "PG",
    # Europe (UK, Scandinavia, Switzerland, etc)
    "NESN.SW", "NOVN.SW", "ROG.SW", "ASML.AS", "MC.PA", "OR.PA", "SAP.DE", "SIE.DE", 
    "AZN.L", "HSBA.L", "SHEL.L", "EQNR.OL", "NOVO-B.CO",
    # Japan
    "7203.T", "6758.T", "8035.T", "9984.T", "6861.T", "6098.T",
    # Hong Kong
    "0700.HK", "09988.HK", "03690.HK", "01299.HK", "00941.HK",
    # Singapore
    "D05.SI", "O39.SI", "U11.SI",
    # Australasia
    "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX",
    # South Africa (Major dual listings or ADRs often easier for free APIs, but trying local)
    "NPN.JO", "FSR.JO", "SBK.JO"
]

# ------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------------------------

def get_financial_data(ticker_symbol):
    """
    Fetches raw financial data using yfinance.
    Returns a dictionary of metrics or None if failed.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Basic Validation: Market Cap (Default to 0 if missing)
        market_cap = info.get('marketCap', 0)
        
        # Skip if Market Cap < $2 Billion (Global small cap filter) to save time
        if market_cap < 2_000_000_000: 
            return None

        # Extract Metrics (using strict defaults for missing data to avoid false positives)
        data = {
            "ticker": ticker_symbol,
            "name": info.get('longName', ticker_symbol),
            "country": info.get('country', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "market_cap": market_cap,
            "forward_pe": info.get('forwardPE', None),
            "revenue_growth": info.get('revenueGrowth', None), # Approx for sales growth
            "eps_growth": info.get('earningsGrowth', None), # Approx for EPS growth
            "debt_to_equity": info.get('debtToEquity', None),
            "price_to_sales": info.get('priceToSalesTrailing12Months', None),
            "dividend_yield": info.get('dividendYield', 0),
            "price_to_book": info.get('priceToBook', None),
            "peg_ratio": info.get('pegRatio', None),
            "profit_margin": info.get('profitMargins', None),
            "p_fcf": None, # Calculated below if possible
            "employees": info.get('fullTimeEmployees', 0),
            "avg_volume": info.get('averageVolume', 0),
            "currency": info.get('currency', 'USD'),
            "stock_obj": stock
        }

        # Calculate Price to Free Cash Flow if data exists
        try:
            fcf = info.get('freeCashflow', None)
            if fcf and fcf > 0 and market_cap:
                data["p_fcf"] = market_cap / fcf
                data["fcf_share"] = fcf / info.get('sharesOutstanding', 1)
            else:
                data["fcf_share"] = 0
        except:
            data["p_fcf"] = None
            data["fcf_share"] = 0

        # Note: "Insider Buying" is rarely available in free APIs. 
        # We will set a placeholder or use 'heldPercentInsiders' as a proxy if needed, 
        # but for strict filtering, we might have to be lenient on this specific metric in a demo.
        data["insider_percent"] = info.get('heldPercentInsiders', 0)

        return data
    except Exception as e:
        # print(f"Error fetching {ticker_symbol}: {e}")
        return None

def get_price_history(stock_obj, start_date, end_date):
    """
    Fetches historical price data. 
    Handles future dates by capping at today's date.
    """
    try:
        # Ensure we don't request future data relative to execution time
        today = date.today()
        
        # Convert string dates to datetime objects
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if start > today:
            return None # Cannot fetch future data
        
        actual_end = min(end, today)
        
        # Fetch history
        hist = stock_obj.history(start=start, end=actual_end + pd.Timedelta(days=5))
        
        if hist.empty:
            return None
            
        return hist
    except:
        return None

def get_price_at_date(hist, target_date_str):
    """
    Finds the closing price on or nearest to a specific date from a history DataFrame.
    """
    target_date = pd.to_datetime(target_date_str).tz_localize(None)
    
    if hist is None or hist.empty:
        return None

    # Ensure index is timezone naive for comparison
    hist.index = hist.index.tz_localize(None)
    
    # Sort just in case
    hist = hist.sort_index()
    
    # Get nearest date using asof (backward search) or closest
    try:
        # We try to get the exact date or the preceding trading day
        idx = hist.index.get_indexer([target_date], method='nearest')[0]
        price = hist.iloc[idx]['Close']
        return price
    except:
        return hist['Close'].iloc[-1] # Fallback to latest

# ------------------------------------------------------------------
# 3. SCREENING LOGIC (Step 1)
# ------------------------------------------------------------------

def run_screening(universe):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    screened_data = []
    
    total = len(universe)
    for i, ticker in enumerate(universe):
        status_text.text(f"Scanning {ticker} ({i+1}/{total})...")
        progress_bar.progress((i + 1) / total)
        
        data = get_financial_data(ticker)
        if not data:
            continue
            
        # --- Apply Filters (Lists 1-7) ---
        # Note: We are permissive if data is None to allow some results in this demo,
        # strictly complying with "Greater than" requires data to be present.
        
        # Convert None to safe values for comparison
        pe = data['forward_pe'] if data['forward_pe'] else 999
        rev_growth = data['revenue_growth'] if data['revenue_growth'] else -1.0 # 5% = 0.05
        eps_growth = data['eps_growth'] if data['eps_growth'] else -1.0
        de_ratio = data['debt_to_equity'] if data['debt_to_equity'] else 999
        ps_ratio = data['price_to_sales'] if data['price_to_sales'] else 999
        insider = data['insider_percent'] # 0 to 1
        div_yield = data['dividend_yield'] if data['dividend_yield'] else 0 # 4% = 0.04
        pb_ratio = data['price_to_book'] if data['price_to_book'] else 999
        peg = data['peg_ratio'] if data['peg_ratio'] else 999
        margin = data['profit_margin'] if data['profit_margin'] else 0
        p_fcf = data['p_fcf'] if data['p_fcf'] else 999
        mkt_cap_usd = data['market_cap']
        
        # List 1: P/E < 25, Sales Growth > 5% (0.05)
        l1 = (pe < 25) and (rev_growth > 0.05)
        
        # List 2: EPS Growth > 25%, Debt/Equity approx 0 (allow < 10 for realism)
        l2 = (eps_growth > 0.25) and (de_ratio < 10) 
        
        # List 3: P/S < 4, Insider > 70%
        # Note: Insider > 70% is very rare for mega caps. 
        l3 = (ps_ratio < 4) and (insider > 0.70)
        
        # List 4: Div Yield > 4%, P/B < 1
        l4 = (div_yield > 0.04) and (pb_ratio < 1)
        
        # List 5: PEG < 2, Profit Margin > 20%
        l5 = (peg < 2) and (margin > 0.20)
        
        # List 6: P/FCF < 15, Div Growth (using Div Yield as proxy for existence + payout ratio check if available)
        # Using simplified proxy: Yield > 2% and P/FCF < 15
        l6 = (p_fcf < 15) and (div_yield > 0.02)
        
        # List 7: Mid Cap ($2B-$20B), P/E < 15, EPS Growth > 15%
        is_mid_cap = (2_000_000_000 <= mkt_cap_usd <= 20_000_000_000)
        l7 = is_mid_cap and (pe < 15) and (eps_growth > 0.15)

        # If it matches ANY list, keep it
        if l1 or l2 or l3 or l4 or l5 or l6 or l7:
            # Fetch Price History for Ranking
            # Jan 1 2025 to Present (or Jun 20 2025)
            hist = get_price_history(data['stock_obj'], "2024-12-01", "2025-06-25")
            
            p_jan = get_price_at_date(hist, "2025-01-01")
            p_apr = get_price_at_date(hist, "2025-04-01")
            p_jun = get_price_at_date(hist, "2025-06-20")
            p_dec24 = get_price_at_date(hist, "2024-12-31")
            
            # Current Price (Real time)
            current_price = p_jun if p_jun else 0

            # Calculate Performance
            perf_A = ((current_price - p_jan) / p_jan) if p_jan else -1.0 # Since Jan 1
            perf_B = ((current_price - p_apr) / p_apr) if p_apr else -1.0 # Since Apr 1

            row = data.copy()
            del row['stock_obj'] # Remove object to make it serializable
            
            row.update({
                "price_dec24": p_dec24,
                "price_jan25": p_jan,
                "price_apr25": p_apr,
                "price_jun25": p_jun,
                "perf_since_jan": perf_A,
                "perf_since_apr": perf_B,
                "matched_lists": [i for i, x in enumerate([l1,l2,l3,l4,l5,l6,l7], 1) if x]
            })
            screened_data.append(row)
            
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(screened_data)

# ------------------------------------------------------------------
# 4. RANKING & AI ANALYSIS
# ------------------------------------------------------------------

def analyze_stock_with_ai(stock_data, api_key):
    """
    Uses OpenAI to generate qualitative columns (AA-AD).
    """
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    You are a professional Investment Analyst. Analyze the following stock data for a sophisticated investor.
    
    Stock: {stock_data['name']} ({stock_data['ticker']})
    Industry: {stock_data['industry']}
    Country: {stock_data['country']}
    Forward P/E: {stock_data['forward_pe']}
    PEG: {stock_data['peg_ratio']}
    Profit Margin: {stock_data['profit_margin']}
    Div Yield: {stock_data['dividend_yield']}
    
    Provide the following 4 outputs strictly separated by a pipe character "|":
    1. Recommendation (Buy/Hold/Sell)
    2. Key Positives and Risks (1-2 sentences summarizing strictly)
    3. Recent Developments (Management/Legal/Product) (1-2 sentences)
    4. AI Exposure and Expectations (1-2 sentences)
    
    Do not use labels like "Recommendation:", just the content.
    Example:
    Buy | Strong balance sheet but currency risk. | New CEO appointed last month. | High exposure via cloud division.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or gpt-3.5-turbo
            messages=[{"role": "system", "content": "You are a financial analyst."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content
        parts = content.split('|')
        
        # Ensure we have 4 parts
        if len(parts) < 4:
            return ["Hold", "Analysis Error", "Analysis Error", "Analysis Error"]
        
        return [p.strip() for p in parts[:4]]
    except Exception as e:
        return ["Error", f"API Error: {str(e)}", "", ""]

# ------------------------------------------------------------------
# MAIN APP LOGIC
# ------------------------------------------------------------------

if st.button("Run Analyst Agent"):
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    else:
        st.write("### Step 1: Screening Global Universe...")
        
        # 1. Run Screening
        df = run_screening(GLOBAL_UNIVERSE)
        
        if df.empty:
            st.warning("No stocks matched the strict criteria from the seed universe. Try expanding the universe list in the code.")
        else:
            st.success(f"Screening complete. Found {len(df)} candidates.")
            
            # 2. Ranking (Step 2 of Prompt)
            # Rank by Perf since Jan 1
            df_A = df.sort_values(by='perf_since_jan', ascending=False).head(15)
            # Rank by Perf since Apr 1
            df_B = df.sort_values(by='perf_since_apr', ascending=False).head(15)
            
            # Merge and Remove Duplicates
            shortlist = pd.concat([df_A, df_B]).drop_duplicates(subset='ticker')
            
            # Sort by Sales Growth (Revenue Growth) Descending
            shortlist = shortlist.sort_values(by='revenue_growth', ascending=False)
            
            # 3. Final Selection (Unique Country + Industry)
            final_list = []
            seen_combos = set()
            
            for index, row in shortlist.iterrows():
                combo = (row['country'], row['industry'])
                if combo not in seen_combos:
                    final_list.append(row)
                    seen_combos.add(combo)
                
                if len(final_list) >= 10:
                    break
            
            final_df = pd.DataFrame(final_list)
            
            st.write(f"### Step 2: Selected Top {len(final_df)} Stocks (Unique Country/Industry)")
            st.dataframe(final_df[['ticker', 'name', 'country', 'industry', 'perf_since_jan']])
            
            # 4. AI Analysis
            st.write("### Step 3: Generating Qualitative AI Insights...")
            ai_results = []
            
            prog_bar_ai = st.progress(0)
            for i, idx in enumerate(final_df.index):
                row = final_df.loc[idx]
                st.text(f"Analyzing {row['ticker']}...")
                insights = analyze_stock_with_ai(row, api_key)
                
                # Append insights to the dataframe
                final_df.loc[idx, 'Recommendation'] = insights[0]
                final_df.loc[idx, 'Key_Positives_Risks'] = insights[1]
                final_df.loc[idx, 'Recent_Developments'] = insights[2]
                final_df.loc[idx, 'AI_Exposure'] = insights[3]
                
                prog_bar_ai.progress((i + 1) / len(final_df))
            
            prog_bar_ai.empty()
            
            # 5. Prepare Excel Download
            st.write("### Step 4: Finalizing Report")
            
            # Map to requested columns A-AD structure (approximate mapping)
            output_df = pd.DataFrame()
            output_df['A Name'] = final_df['name']
            output_df['B Exchange'] = final_df['ticker'].apply(lambda x: x.split('.')[1] if '.' in x else 'US')
            output_df['C Ticker'] = final_df['ticker']
            output_df['D Country'] = final_df['country']
            output_df['E Industry'] = final_df['industry']
            output_df['F Employees'] = final_df['employees']
            output_df['G Avg Volume'] = final_df['avg_volume']
            output_df['H Avg Vol USD'] = final_df['avg_volume'] * final_df['price_jun25']
            output_df['I Price 31 Dec 24'] = final_df['price_dec24']
            output_df['J Price 20 Jun 25'] = final_df['price_jun25']
            output_df['K Sales Growth 1Y'] = final_df['revenue_growth']
            output_df['L Sales Growth 3Y'] = "Data N/A via free API"
            output_df['M EPS Growth 1Y'] = final_df['eps_growth']
            output_df['N EPS Growth 3Y'] = "Data N/A via free API"
            output_df['O Fwd P/E'] = final_df['forward_pe']
            output_df['P Fwd P/S'] = final_df['price_to_sales']
            output_df['Q Fwd P/B'] = final_df['price_to_book']
            output_df['R Fwd P/FCF'] = final_df['p_fcf']
            output_df['S PEG'] = final_df['peg_ratio']
            output_df['T Margin'] = final_df['profit_margin']
            output_df['U Div Yield'] = final_df['dividend_yield']
            output_df['V Div Growth'] = "Data N/A via free API"
            output_df['W Debt/Equity'] = final_df['debt_to_equity']
            output_df['X FCF/Share'] = final_df['fcf_share']
            output_df['Y Curr Price'] = final_df['price_jun25']
            output_df['Z Target Price'] = "See Analyst Report"
            output_df['AA Recommendation'] = final_df['Recommendation']
            output_df['AB Positives/Risks'] = final_df['Key_Positives_Risks']
            output_df['AC Recent Dev'] = final_df['Recent_Developments']
            output_df['AD AI Exposure'] = final_df['AI_Exposure']

            # Create Excel in Memory
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                output_df.to_excel(writer, index=False, sheet_name='Top 10 Stocks')
                
                # Auto-adjust column width
                worksheet = writer.sheets['Top 10 Stocks']
                for i, col in enumerate(output_df.columns):
                    worksheet.set_column(i, i, 20)

            st.success("Analysis Complete!")
            
            st.download_button(
                label="📥 Download Investment Analyst Report (.xlsx)",
                data=buffer.getvalue(),
                file_name=f"Global_Stock_Picks_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )

st.markdown("---")
st.caption("Note: This tool uses 'yfinance' for market data. Some specific metrics (like 3-year forecast CAGR or Insider Buying %) are approximated or marked N/A as they are not standard in free public APIs. Future dates in the prompt (e.g. June 2025) will default to the latest available data if the current date is earlier.")
