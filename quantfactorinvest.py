import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import io
import time
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & CONTEXT (DECEMBER 2025)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid Smart Beta Agent (Dec 2025)", 
    page_icon="🧬", 
    layout="wide"
)

# Initialize OpenAI
api_key = st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

st.title("🧬 Hybrid Smart Beta & 7-List Agent")
st.markdown("""
**System Architecture:**
1.  **Screening Layer:** Applies the **7-List Protocol** (Strict Filters).
2.  **Factor Layer:** Calculates **Smart Beta Z-Scores** (Value, Quality, Volatility, Momentum).
3.  **AI Layer:** Generates qualitative narratives using **Real News**.
""")

# -----------------------------------------------------------------------------
# 2. DATA FETCHING (INC. NEWS & TECHNICALS)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_dec2025_data(tickers):
    data_list = []
    progress_text = st.empty()
    bar = st.progress(0)
    
    target_dates = {"Dec31_24": "2024-12-31", "Apr01_25": "2025-04-01", "Jun20_25": "2025-06-20"}
    
    for i, ticker in enumerate(tickers):
        progress_text.text(f"Hybrid Scan: {ticker} ({i+1}/{len(tickers)})")
        bar.progress((i+1)/len(tickers))
        time.sleep(0.1) 
        
        try:
            stock = yf.Ticker(ticker)
            
            # --- Fundamentals ---
            try:
                info = stock.info
                if not info: continue
            except: continue
                
            # --- History & Technicals ---
            try:
                hist = stock.history(start="2024-10-01", end=datetime.now().strftime("%Y-%m-%d"))
            except:
                hist = pd.DataFrame()

            # Price Checks
            def get_close(target_date_str):
                if hist.empty: return np.nan
                try:
                    ts = pd.to_datetime(target_date_str)
                    idx = hist.index.get_indexer([ts], method='nearest')[0]
                    found_date = hist.index[idx]
                    if abs((found_date - ts).days) > 5: return np.nan
                    return hist.iloc[idx]['Close']
                except: return np.nan

            p_dec24 = get_close(target_dates["Dec31_24"])
            p_apr25 = get_close(target_dates["Apr01_25"])
            p_jun25 = get_close(target_dates["Jun20_25"])
            p_curr = info.get('currentPrice', hist.iloc[-1]['Close'] if not hist.empty else np.nan)
            
            # Volatility (Factor)
            volatility = np.nan
            if not hist.empty:
                rets = hist['Close'].pct_change().dropna()
                volatility = rets.std() * np.sqrt(252) # Annualized

            # RSI
            rsi_val = np.nan
            if not hist.empty and len(hist) > 15:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi_val = 100 - (100 / (1 + rs.iloc[-1]))

            # Metrics
            mkt_cap = info.get('marketCap', np.nan)
            fwd_pe = info.get('forwardPE', np.nan)
            roe = info.get('returnOnEquity', np.nan) # Quality Factor
            rev_growth = info.get('revenueGrowth', np.nan)
            eps_growth = info.get('earningsGrowth', np.nan)
            debt_equity = info.get('debtToEquity', np.nan)
            p_book = info.get('priceToBook', np.nan) # Value Factor
            div_yield = info.get('dividendYield', 0)
            held_insiders = info.get('heldPercentInsiders', 0)
            profit_margin = info.get('profitMargins', np.nan)
            peg = info.get('pegRatio', np.nan)
            p_sales = info.get('priceToSalesTrailing12Months', np.nan)
            
            fcf = info.get('freeCashflow', np.nan)
            p_fcf = (mkt_cap / fcf) if (fcf and mkt_cap) else np.nan
            
            # News
            news_titles = []
            try:
                raw_news = stock.news
                if raw_news: news_titles = [n['title'] for n in raw_news[:3]]
            except: news_titles = ["No recent news."]

            row = {
                'Ticker': ticker,
                'Name': info.get('longName', ticker),
                'Exchange': info.get('exchange', 'Unknown'),
                'Country': info.get('country', 'Unknown'),
                'Industry': info.get('industry', 'Unknown'),
                'Employees': info.get('fullTimeEmployees', 'N/A'),
                'Market_Cap': mkt_cap,
                'Price_Dec24': p_dec24,
                'Price_Apr25': p_apr25,
                'Price_Jun25': p_jun25,
                'Price_Current': p_curr,
                'RSI': rsi_val,
                'Volatility': volatility, # Factor
                'Vol_Avg': info.get('averageVolume', 0),
                'Fwd_PE': fwd_pe,         # Value Factor
                'ROE': roe,               # Quality Factor
                'Rev_Growth': rev_growth,
                'EPS_Growth': eps_growth,
                'Debt_Equity': debt_equity,
                'P_S': p_sales,
                'Insider_Pct': held_insiders,
                'Div_Yield': div_yield,
                'P_B': p_book,
                'PEG': peg,
                'Margin': profit_margin,
                'P_FCF': p_fcf,
                'Target_Price': info.get('targetMeanPrice', np.nan),
                'Recent_News': "; ".join(news_titles)
            }
            data_list.append(row)
            
        except Exception as e: continue

    bar.empty()
    progress_text.empty()
    return pd.DataFrame(data_list)

# -----------------------------------------------------------------------------
# 3. SMART BETA ENGINE (THE FACTOR LOGIC)
# -----------------------------------------------------------------------------

def calculate_factor_scores(df):
    """
    Calculates Z-Scores for Value, Quality, Volatility, and Momentum.
    Returns the DF with a 'Factor_Score' column.
    """
    if df.empty: return df
    
    scored = df.copy()
    
    # 1. Normalize Metrics (Z-Score)
    # Value: Low PE is better (-1)
    scored['Z_Value'] = (scored['Fwd_PE'] - scored['Fwd_PE'].mean()) / scored['Fwd_PE'].std() * -1
    
    # Quality: High ROE is better (+1)
    # Fill NaN ROE with median to prevent crash
    scored['ROE'] = scored['ROE'].fillna(scored['ROE'].median())
    scored['Z_Quality'] = (scored['ROE'] - scored['ROE'].mean()) / scored['ROE'].std()
    
    # Volatility: Low Vol is better (-1)
    scored['Volatility'] = scored['Volatility'].fillna(scored['Volatility'].median())
    scored['Z_LowVol'] = (scored['Volatility'] - scored['Volatility'].mean()) / scored['Volatility'].std() * -1
    
    # Momentum: High Jan-Current Return is better (+1)
    # We calculate return first
    if 'Price_Dec24' in scored.columns:
        scored['Ret_YTD'] = (scored['Price_Current'] - scored['Price_Dec24']) / scored['Price_Dec24']
        scored['Z_Mom'] = (scored['Ret_YTD'] - scored['Ret_YTD'].mean()) / scored['Ret_YTD'].std()
    else:
        scored['Z_Mom'] = 0

    # 2. Composite Score (Equal Weight)
    scored['Factor_Score'] = (
        scored['Z_Value'].fillna(0) + 
        scored['Z_Quality'].fillna(0) + 
        scored['Z_LowVol'].fillna(0) + 
        scored['Z_Mom'].fillna(0)
    ) / 4
    
    return scored

# -----------------------------------------------------------------------------
# 4. STRICT SCREENING LOGIC (7-LIST)
# -----------------------------------------------------------------------------

def apply_screens(df):
    if df.empty: return {}, pd.DataFrame()
    
    large_cap = df[df['Market_Cap'] >= 10e9].copy()
    lists = {}
    
    # List 1: Value + Growth
    l1 = large_cap[(large_cap['Fwd_PE'] < 25) & (large_cap['Rev_Growth'] > 0.05)].copy()
    l1['Source'] = 'List 1'
    lists['List 1'] = l1
    
    # List 2: Zero Debt + High EPS
    l2 = large_cap[(large_cap['EPS_Growth'] > 0.25) & (large_cap['Debt_Equity'] < 10.0)].copy()
    l2['Source'] = 'List 2'
    lists['List 2'] = l2
    
    # List 3: Insider
    l3 = large_cap[(large_cap['P_S'] < 4) & (large_cap['Insider_Pct'] > 0.30)].copy()
    l3['Source'] = 'List 3'
    lists['List 3'] = l3
    
    # List 4: Value Income
    l4 = large_cap[(large_cap['Div_Yield'] > 0.04) & (large_cap['P_B'] < 1.0)].copy()
    l4['Source'] = 'List 4'
    lists['List 4'] = l4
    
    # List 5: Quality
    l5 = large_cap[(large_cap['PEG'] < 2.0) & (large_cap['Margin'] > 0.20)].copy()
    l5['Source'] = 'List 5'
    lists['List 5'] = l5
    
    # List 6: Cash Cows
    l6 = large_cap[(large_cap['P_FCF'] < 15) & (large_cap['Div_Yield'] > 0.02)].copy()
    l6['Source'] = 'List 6'
    lists['List 6'] = l6
    
    # List 7: Mid Cap
    mask_mid = (df['Market_Cap'] >= 2e9) & (df['Market_Cap'] < 20e9)
    l7 = df[mask_mid & (df['Fwd_PE'] < 15) & (df['EPS_Growth'] > 0.15)].copy()
    l7['Source'] = 'List 7'
    lists['List 7'] = l7
    
    all_hits = pd.concat(lists.values()).drop_duplicates(subset=['Ticker'])
    return lists, all_hits

# -----------------------------------------------------------------------------
# 5. RANKING & FINALIZATION
# -----------------------------------------------------------------------------

def rank_and_finalize(df):
    if df.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # 1. Apply Factor Scoring (Smart Beta) to the Survivors
    df_scored = calculate_factor_scores(df)
    
    # 2. Rank by Performance (as per 7-List rules)
    df_scored['Ret_Jan25'] = (df_scored['Price_Current'] - df_scored['Price_Dec24']) / df_scored['Price_Dec24']
    df_scored['Ret_Apr25'] = (df_scored['Price_Current'] - df_scored['Price_Apr25']) / df_scored['Price_Apr25']
    
    short_a = df_scored.sort_values(by='Ret_Jan25', ascending=False).head(15)
    short_b = df_scored.sort_values(by='Ret_Apr25', ascending=False).head(15)
    
    combined = pd.concat([short_a, short_b]).drop_duplicates(subset=['Ticker'])
    
    # 3. Final Sort by Sales Growth
    combined = combined.sort_values(by='Rev_Growth', ascending=False)
    
    # Diversity Filter
    final_10 = []
    seen = set()
    for idx, row in combined.iterrows():
        key = (row['Country'], row['Industry'])
        if key not in seen:
            final_10.append(row)
            seen.add(key)
        if len(final_10) >= 10: break
            
    return short_a, short_b, pd.DataFrame(final_10)

# -----------------------------------------------------------------------------
# 6. AI AGENT
# -----------------------------------------------------------------------------

def get_ai_narrative(row):
    if not client: return ("AI N/A", "AI N/A", "AI N/A", "AI N/A")
    
    prompt = f"""
    Date: Dec 11, 2025.
    Analyze {row['Name']} ({row['Ticker']}).
    
    Metrics:
    - Smart Beta Factor Score: {row.get('Factor_Score', 0):.2f} (High is good)
    - RSI: {row.get('RSI', 50):.1f}
    - News: {row['Recent_News']}
    
    Task: Write 4 concise Excel cell outputs.
    1. REC: Buy/Hold/Sell.
    2. RISKS: Positives/Risks (News based).
    3. DEV: Developments (News based).
    4. AI: AI Strategy.
    
    Format: REC: [Tx] RISKS: [Tx] DEV: [Tx] AI: [Tx]
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}]
        )
        txt = resp.choices[0].message.content
        rec = txt.split("REC:")[1].split("RISKS:")[0].strip() if "REC:" in txt else "Hold"
        risk = txt.split("RISKS:")[1].split("DEV:")[0].strip() if "RISKS:" in txt else "N/A"
        dev = txt.split("DEV:")[1].split("AI:")[0].strip() if "DEV:" in txt else "N/A"
        ai = txt.split("AI:")[1].strip() if "AI:" in txt else "N/A"
        return rec, risk, dev, ai
    except: return "Err", "Err", "Err", "Err"

# -----------------------------------------------------------------------------
# 7. UI
# -----------------------------------------------------------------------------

default_univ = """
AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, BRK-B, LLY, V, 
UNH, XOM, JNJ, MA, PG, HD, COST, ABBV, MRK, KO, PEP, AVGO,
AZN.L, HSBA.L, SHEL.L, ULVR.L, BP.L, RIO.L, GSK.L,
NESN.SW, ROG.SW, NOVN.SW, UBSG.SW, OR.PA, MC.PA, TTE.PA,
SIE.DE, SAP.DE, ALV.DE, BMW.DE,
7203.T, 6758.T, 9984.T, 8035.T,
0700.HK, 0941.HK, 1299.HK,
DBS.SI, U11.SI,
BHP.AX, CBA.AX, CSL.AX
"""

st.sidebar.header("Configuration")
tickers_raw = st.sidebar.text_area("Universe (Dec 2025)", default_univ, height=300)
tickers = [t.strip() for t in tickers_raw.replace("\n", ",").split(",") if t.strip()]

if st.button("🚀 RUN HYBRID AGENT"):
    if not tickers:
        st.error("No tickers.")
    else:
        st.info(f"Scanning {len(tickers)} companies...")
        df_raw = fetch_dec2025_data(tickers)
        
        if df_raw.empty:
            st.error("No data found.")
        else:
            # 1. Screen
            st.info("Applying 7-List Screens...")
            lists_dict, all_candidates = apply_screens(df_raw)
            
            # 2. Factor Score & Rank
            st.info("Calculating Smart Beta Z-Scores & Ranking...")
            short_a, short_b, final_10 = rank_and_finalize(all_candidates)
            
            if final_10.empty:
                st.warning("Strict screens found no matches. Relax constraints or add tickers.")
            else:
                st.success(f"Done. {len(final_10)} finalists selected.")
                
                # Show Result with Factor Score
                st.dataframe(final_10[['Name', 'Country', 'Factor_Score', 'Fwd_PE', 'Rev_Growth']]
                             .style.background_gradient(subset=['Factor_Score'], cmap='Greens'))

                # Generate Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    wb = writer.book
                    fmt_head = wb.add_format({'bold': True, 'bg_color': '#DCE6F1'})
                    
                    final_rows = []
                    bar_ai = st.progress(0)
                    for i, (idx, row) in enumerate(final_10.iterrows()):
                        bar_ai.progress((i+1)/len(final_10))
                        rec, risks, dev, ai_strat = get_ai_narrative(row)
                        
                        final_rows.append({
                            'A_Name': row['Name'], 'B_Ticker': row['Ticker'],
                            'C_Factor_Score': row.get('Factor_Score', 0), # Added Back
                            'D_RSI': row.get('RSI', 0),
                            'E_Industry': row['Industry'], 'F_Country': row['Country'],
                            'G_Fwd_PE': row['Fwd_PE'], 'H_Sales_Growth': row['Rev_Growth'],
                            'I_Price_Current': row['Price_Current'],
                            'AA_Rec': rec, 'AB_Risks': risks, 'AC_Dev': dev, 'AD_AI': ai_strat
                        })
                    
                    df_final = pd.DataFrame(final_rows)
                    df_final.to_excel(writer, sheet_name='Final_Smart_Beta', index=False)
                    writer.sheets['Final_Smart_Beta'].set_row(0, 20, fmt_head)
                    
                    short_a.to_excel(writer, sheet_name='Shortlist_A', index=False)
                    
                output.seek(0)
                st.download_button("📥 Download Hybrid Report", output, "Hybrid_Smart_Beta.xlsx")
