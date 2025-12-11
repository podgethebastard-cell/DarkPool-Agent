import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import io
import time
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. SYSTEM CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Global 1000 Auto-Scanner", page_icon="🌍", layout="wide")

api_key = st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

st.title("🌍 Global 1000 Market Scanner (Dec 2025)")
st.markdown("""
**Scale:** Massive (~1000 Tickers)
**Targets:** S&P 500 (US), FTSE 100 (UK), DAX 40 (DE), CAC 40 (FR), Nikkei 225 (JP), Top 50 (HK/SG/AU).
**Operation:** 1. Dynamic Scraping. 2. Bulk Price Fetch. 3. Fundamental Loop. 4. Factor Rank.
""")

# -----------------------------------------------------------------------------
# 2. UNIVERSE GENERATION (DYNAMIC + HARDCODED)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def get_global_1000():
    """
    Constructs a universe of approx 1000 global liquid stocks.
    """
    tickers = []
    status = st.empty()
    
    # 1. US: S&P 500 (Dynamic Scrape)
    try:
        status.text("Scraping S&P 500 from Wikipedia...")
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers.extend(sp500['Symbol'].tolist())
    except:
        # Fallback if Wiki fails
        tickers.extend(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V"]) 

    # 2. UK: FTSE 100 (Dynamic Scrape)
    try:
        status.text("Scraping FTSE 100...")
        ftse = pd.read_html('https://en.wikipedia.org/wiki/FTSE_100_Index')[3]
        # Yahoo requires .L for London
        uk_syms = [x + ".L" for x in ftse['Ticker'].tolist()] 
        tickers.extend(uk_syms)
    except:
        tickers.extend(["HSBA.L", "SHEL.L", "BP.L", "AZN.L", "ULVR.L", "DGE.L"])

    # 3. Germany: DAX 40 (Dynamic Scrape)
    try:
        status.text("Scraping DAX 40...")
        dax = pd.read_html('https://en.wikipedia.org/wiki/DAX')[3]
        # Yahoo requires .DE
        de_syms = [x + ".DE" for x in dax['Ticker'].tolist()]
        tickers.extend(de_syms)
    except:
        tickers.extend(["SIE.DE", "SAP.DE", "ALV.DE", "BMW.DE", "VOW3.DE"])
        
    # 4. France: CAC 40 (Dynamic Scrape)
    try:
        status.text("Scraping CAC 40...")
        cac = pd.read_html('https://en.wikipedia.org/wiki/CAC_40')[3]
        # Yahoo requires .PA
        fr_syms = [x + ".PA" for x in cac['Ticker'].tolist()]
        tickers.extend(fr_syms)
    except:
         tickers.extend(["MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA"])

    # 5. ASIA / PACIFIC (Hardcoded Top ~150 Liquid)
    # Scraping Asian indices is unreliable due to format changes, hardcoding the titans is safer.
    status.text("Adding Asia/Pacific Titans...")
    
    # Japan (Nikkei/Topix Leaders)
    jp = [
        "7203.T", "6758.T", "9984.T", "8035.T", "7974.T", "9432.T", "8306.T", "6861.T", "6098.T", "4063.T",
        "6501.T", "7267.T", "8316.T", "8001.T", "8031.T", "4502.T", "8766.T", "3382.T", "6902.T", "6273.T",
        "7741.T", "6367.T", "4503.T", "4568.T", "6954.T", "7201.T", "6981.T", "4901.T", "8411.T", "8053.T"
    ]
    # Hong Kong
    hk = [
        "0700.HK", "0941.HK", "1299.HK", "9988.HK", "0388.HK", "0005.HK", "3690.HK", "1810.HK", "1211.HK",
        "2318.HK", "2020.HK", "0011.HK", "0001.HK", "0016.HK", "0027.HK", "0066.HK", "0883.HK", "1088.HK"
    ]
    # Singapore
    sg = ["DBS.SI", "U11.SI", "Z74.SI", "C52.SI", "S68.SI", "Y92.SI", "A17U.SI"]
    # Australia
    au = [
        "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "ANZ.AX", "WBC.AX", "FMG.AX", "WES.AX", "MQG.AX", "WOW.AX",
        "TLS.AX", "RIO.AX", "GMG.AX", "WDS.AX", "STO.AX", "ALL.AX", "SUN.AX", "QBE.AX", "SCG.AX", "RMD.AX"
    ]
    
    tickers.extend(jp + hk + sg + au)
    
    # Deduplicate and Clean
    tickers = list(set(tickers))
    # Remove any None or empty strings
    tickers = [x for x in tickers if isinstance(x, str)]
    
    status.empty()
    return tickers

# -----------------------------------------------------------------------------
# 3. MASSIVE DATA FETCHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def scan_global_market(tickers):
    """
    Optimized for scale:
    1. Bulk download price history (Fast).
    2. Loop for Fundamentals (Necessary Slowness).
    """
    data_list = []
    
    # 1. Bulk History Download (The Speed Hack)
    st.write(f"📉 Bulk downloading price history for {len(tickers)} stocks...")
    # Download last 6 months to cover Dec 2024 to June 2025
    bulk_hist = yf.download(tickers, start="2024-12-01", end=datetime.now().strftime("%Y-%m-%d"), progress=True)
    
    # Handle the multi-index columns from bulk download
    # If only 1 ticker, it's not multi-index, but with 1000 it is.
    is_multi = isinstance(bulk_hist.columns, pd.MultiIndex)
    
    # 2. Fundamental Loop
    st.write("🔍 Extracting Fundamental Data (This takes time)...")
    bar = st.progress(0)
    status_text = st.empty()
    
    target_dates = {"Dec31_24": "2024-12-31", "Apr01_25": "2025-04-01"}
    
    start_time = time.time()
    
    for i, ticker in enumerate(tickers):
        # Update progress
        elapsed = time.time() - start_time
        rate = (i+1) / elapsed if elapsed > 0 else 0
        remaining = (len(tickers) - (i+1)) / rate if rate > 0 else 0
        status_text.text(f"Scanning {ticker} ({i+1}/{len(tickers)}) | Est. Rem: {remaining/60:.1f} min")
        bar.progress((i+1)/len(tickers))
        
        try:
            # A. GET PRICES FROM BULK DATA
            # We avoid making a network call here by using the bulk df
            try:
                if is_multi:
                    # Access via MultiIndex (Adj Close or Close)
                    series = bulk_hist['Close'][ticker]
                else:
                    series = bulk_hist['Close']
            except:
                # If ticker failed in bulk download, skip
                continue
                
            # Helper to find date in series
            def get_price_from_series(s, date_str):
                try:
                    ts = pd.to_datetime(date_str)
                    # Get index of nearest date
                    idx = s.index.get_indexer([ts], method='nearest')[0]
                    if abs((s.index[idx] - ts).days) > 5: return np.nan
                    return s.iloc[idx]
                except: return np.nan

            p_dec = get_price_from_series(series, target_dates['Dec31_24'])
            p_apr = get_price_from_series(series, target_dates['Apr01_25'])
            p_cur = series.iloc[-1] if not series.empty else np.nan
            
            # RSI Calc (Vectorized on the series)
            rsi = np.nan
            if len(series) > 15:
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs.iloc[-1]))
                
            # Vol Calc
            vol = series.pct_change().std() * np.sqrt(252)

            # B. GET FUNDAMENTALS (Network Call - The Bottleneck)
            # We must be gentle or Yahoo bans us
            stock = yf.Ticker(ticker)
            
            try:
                # Info fetch
                info = stock.info 
                if not info: continue
            except:
                continue

            # Filtering: Skip Tiny Caps (<2B) to speed up "Quality" list processing
            mkt_cap = info.get('marketCap', 0)
            if mkt_cap < 2e9: continue

            # Extract Ratios
            fwd_pe = info.get('forwardPE', np.nan)
            
            # If P/E is missing or crazy, skip (Optimization)
            if pd.isna(fwd_pe) or fwd_pe > 200: 
                # Keep if it might fit "Growth" list, else maybe skip? 
                # We'll keep it to be safe.
                pass

            data_list.append({
                'Ticker': ticker,
                'Name': info.get('longName', ticker),
                'Country': info.get('country', 'Unknown'),
                'Industry': info.get('industry', 'Unknown'),
                'Market_Cap': mkt_cap,
                'Price_Dec24': p_dec, 'Price_Apr25': p_apr, 'Price_Current': p_cur,
                'RSI': rsi, 'Volatility': vol,
                'Fwd_PE': fwd_pe,
                'Rev_Growth': info.get('revenueGrowth', np.nan),
                'EPS_Growth': info.get('earningsGrowth', np.nan),
                'Debt_Equity': info.get('debtToEquity', np.nan),
                'P_S': info.get('priceToSalesTrailing12Months', np.nan),
                'Insider_Pct': info.get('heldPercentInsiders', 0),
                'Div_Yield': info.get('dividendYield', 0),
                'P_B': info.get('priceToBook', np.nan),
                'PEG': info.get('pegRatio', np.nan),
                'Margin': info.get('profitMargins', np.nan),
                'P_FCF': (mkt_cap / info.get('freeCashflow')) if info.get('freeCashflow') else np.nan
            })
            
        except Exception as e:
            # print(f"Error on {ticker}: {e}")
            continue
            
    status_text.empty()
    bar.empty()
    return pd.DataFrame(data_list)

# -----------------------------------------------------------------------------
# 4. LOGIC ENGINE
# -----------------------------------------------------------------------------
def process_results(df):
    if df.empty: return {}, pd.DataFrame(), pd.DataFrame()
    
    # 1. SCORING
    s = df.copy()
    s['Z_Val'] = (s['Fwd_PE'] - s['Fwd_PE'].mean()) / s['Fwd_PE'].std() * -1
    s['Z_Qual'] = (s['Margin'] - s['Margin'].mean()) / s['Margin'].std()
    s['Factor_Score'] = (s['Z_Val'].fillna(0) + s['Z_Qual'].fillna(0))/2
    
    # 2. LISTS
    lists = {}
    lists['L1_ValueGrowth'] = s[(s['Fwd_PE'] < 25) & (s['Rev_Growth'] > 0.05)]
    lists['L2_ZeroDebt'] = s[(s['EPS_Growth'] > 0.25) & (s['Debt_Equity'] < 10)]
    lists['L3_Insider'] = s[(s['P_S'] < 4) & (s['Insider_Pct'] > 0.20)]
    lists['L4_DeepValue'] = s[(s['Div_Yield'] > 0.04) & (s['P_B'] < 1)]
    lists['L5_Quality'] = s[(s['PEG'] < 2) & (s['Margin'] > 0.20)]
    lists['L6_CashCow'] = s[(s['P_FCF'] < 15)]
    lists['L7_MidCap'] = s[(s['Market_Cap'] < 20e9) & (s['Fwd_PE'] < 15)]
    
    all_hits = pd.concat(lists.values()).drop_duplicates(subset=['Ticker'])
    
    # 3. RANKING
    if all_hits.empty: return {}, pd.DataFrame(), pd.DataFrame()
    
    all_hits['Ret_Jan'] = (all_hits['Price_Current'] - all_hits['Price_Dec24'])/all_hits['Price_Dec24']
    all_hits['Ret_Apr'] = (all_hits['Price_Current'] - all_hits['Price_Apr25'])/all_hits['Price_Apr25']
    
    # Shortlists
    short_a = all_hits.sort_values('Ret_Jan', ascending=False).head(15)
    short_b = all_hits.sort_values('Ret_Apr', ascending=False).head(15)
    
    # Final Combine
    final = pd.concat([short_a, short_b]).drop_duplicates(subset=['Ticker'])
    final = final.sort_values('Rev_Growth', ascending=False)
    
    # Diversity
    final_10 = []
    seen = set()
    for i, r in final.iterrows():
        k = (r['Country'], r['Industry'])
        if k not in seen:
            final_10.append(r)
            seen.add(k)
        if len(final_10) >= 10: break
            
    return lists, short_a, pd.DataFrame(final_10)

# -----------------------------------------------------------------------------
# 5. EXECUTION UI
# -----------------------------------------------------------------------------
st.subheader("Global 1000 Setup")
st.write("Initializing universe...")

# Load Universe Immediately
univ = get_global_1000()
st.success(f"Universe Loaded: {len(univ)} Tickers (US, EU, UK, JP, APAC).")

if st.button("🚀 LAUNCH GLOBAL 1000 SCAN"):
    
    # 1. Scan
    df_results = scan_global_market(univ)
    
    if df_results.empty:
        st.error("Scan returned no data.")
    else:
        st.success(f"Scanned {len(df_results)} stocks successfully.")
        
        # 2. Process
        lists, short_a, final_10 = process_results(df_results)
        
        if final_10.empty:
            st.warning("No stocks passed the strict filters.")
        else:
            st.balloons()
            st.subheader("🏆 The Final 10 Titans")
            st.dataframe(final_10[['Name', 'Country', 'Factor_Score', 'Fwd_PE', 'Rev_Growth', 'RSI']])
            
            # 3. Excel Report
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                # Main
                rows = []
                st.write("Generating AI Reports...")
                prog = st.progress(0)
                
                for i, (idx, row) in enumerate(final_10.iterrows()):
                    prog.progress((i+1)/len(final_10))
                    
                    # AI Analysis
                    rec, risk, dev, strat = "N/A", "N/A", "N/A", "N/A"
                    if client:
                        try:
                            # We fetch news on the fly for just the final 10 to save API calls earlier
                            t = yf.Ticker(row['Ticker'])
                            news = " ".join([n['title'] for n in t.news[:2]]) if t.news else "No news"
                            
                            p = f"Stock: {row['Ticker']}. News: {news}. RSI: {row['RSI']}. Write 4 concise excel cells: REC, RISKS, DEV, AI."
                            resp = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content":p}])
                            txt = resp.choices[0].message.content
                            
                            rec = txt.split("REC:")[1].split("RISKS:")[0] if "REC:" in txt else "Hold"
                            risk = txt.split("RISKS:")[1].split("DEV:")[0] if "RISKS:" in txt else "N/A"
                            dev = txt.split("DEV:")[1].split("AI:")[0] if "DEV:" in txt else "N/A"
                            strat = txt.split("AI:")[1] if "AI:" in txt else "N/A"
                        except: pass
                        
                    rows.append({
                        'Ticker': row['Ticker'], 'Name': row['Name'], 'Country': row['Country'],
                        'Factor_Score': row['Factor_Score'], 'PE': row['Fwd_PE'],
                        'AI_Rec': rec, 'AI_Risks': risk, 'AI_Dev': dev, 'AI_Strat': strat
                    })
                    
                pd.DataFrame(rows).to_excel(writer, sheet_name='Final_List', index=False)
                
                # Lists
                for k, v in lists.items():
                    if not v.empty: v.to_excel(writer, sheet_name=k, index=False)
                    
            out.seek(0)
            st.download_button("📥 Download Titan Report", out, "Global_1000_Scan.xlsx")
