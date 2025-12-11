import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
import io
import time
import logging

# -----------------------------------------------------------------------------
# CONFIGURATION & LOGGING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ECVS Screener", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data
def get_index_tickers(index_name):
    """
    Fetches tickers for major indices from Wikipedia and formats them for yfinance.
    """
    tickers = []
    try:
        if index_name == "S&P 500":
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            df = pd.read_html(url)[0]
            tickers = df['Symbol'].tolist()
            # Clean: BRK.B -> BRK-B
            tickers = [t.replace('.', '-') for t in tickers]

        elif index_name == "Nasdaq 100":
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            # Wikipedia table structure varies; look for table with 'Ticker' column
            tables = pd.read_html(url)
            for t in tables:
                if 'Ticker' in t.columns:
                    tickers = t['Ticker'].tolist()
                    break
                elif 'Symbol' in t.columns:
                    tickers = t['Symbol'].tolist()
                    break

        elif index_name == "Dow Jones":
            url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
            tables = pd.read_html(url)
            for t in tables:
                if 'Symbol' in t.columns:
                    tickers = t['Symbol'].tolist()
                    break

        elif index_name == "FTSE 100":
            url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
            tables = pd.read_html(url)
            for t in tables:
                if 'Ticker' in t.columns:
                    # FTSE tickers on Wikipedia usually lack the .L suffix required for Yahoo
                    raw_tickers = t['Ticker'].tolist()
                    tickers = [f"{x}.L" if not x.endswith('.') else f"{x}L" for x in raw_tickers]
                    break

        elif index_name == "DAX 40":
            url = 'https://en.wikipedia.org/wiki/DAX'
            tables = pd.read_html(url)
            for t in tables:
                if 'Ticker' in t.columns:
                    raw_tickers = t['Ticker'].tolist()
                    # DAX tickers usually need .DE
                    tickers = [f"{x}.DE" for x in raw_tickers]
                    break

        return tickers

    except Exception as e:
        st.error(f"Failed to fetch list for {index_name}: {e}")
        return []

def fetch_alpha_vantage_financials(ticker: str, api_key: str):
    """Fetch company overview from Alpha Vantage if key is provided."""
    if not api_key:
        return {}
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}'
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logging.warning(f"AlphaVantage request failed for {ticker}: {e}")
    return {}

def safe_get(d, *keys, default=None):
    """Safely retrieve nested dictionary keys."""
    for k in keys:
        if d is None:
            return default
        if isinstance(d, dict) and k in d:
            val = d[k]
            if val == 'None': return default
            return val
    return default

# -----------------------------------------------------------------------------
# CORE METRICS & SCORING LOGIC
# -----------------------------------------------------------------------------

def compute_prices_dynamic(hist: pd.DataFrame) -> dict:
    """Dynamically determine Start-of-Year, 3-Month-Ago, and Current prices."""
    today = datetime.date.today()
    start_of_year = datetime.date(today.year, 1, 1)
    three_months_ago = today - datetime.timedelta(days=90)
    
    def get_price(target_date):
        target_ts = pd.Timestamp(target_date)
        if hist.empty:
            return None
        try:
            idx = hist.index.get_indexer([target_ts], method='nearest')[0]
            if idx == -1: return None
            return float(hist.iloc[idx]['Close'])
        except:
            return None

    return {
        'p_start_year': get_price(start_of_year),
        'p_3mo_ago': get_price(three_months_ago),
        'p_current': float(hist['Close'].iloc[-1]) if not hist.empty else None
    }

def compute_simple_metrics(info: dict, overview_av: dict) -> dict:
    metrics = {}
    metrics['sector'] = safe_get(info, 'sector')
    metrics['industry'] = safe_get(info, 'industry')
    metrics['country'] = safe_get(info, 'country')
    metrics['employees'] = safe_get(info, 'fullTimeEmployees')
    
    metrics['marketCap'] = safe_get(info, 'marketCap')
    metrics['forwardPE'] = safe_get(info, 'forwardPE')
    metrics['trailingPE'] = safe_get(info, 'trailingPE')
    metrics['priceToSales'] = safe_get(info, 'priceToSalesTrailing12Months')
    metrics['priceToBook'] = safe_get(info, 'priceToBook')
    metrics['pegRatio'] = safe_get(info, 'pegRatio')
    metrics['forwardEps'] = safe_get(info, 'forwardEps')
    metrics['dividendYield'] = safe_get(info, 'dividendYield')
    
    metrics['freeCashflow'] = safe_get(info, 'freeCashflow')
    metrics['ebitda'] = safe_get(info, 'ebitda')
    metrics['debtToEquity'] = safe_get(info, 'debtToEquity')
    metrics['profitMargins'] = safe_get(info, 'profitMargins')
    metrics['returnOnEquity'] = safe_get(info, 'returnOnEquity')
    metrics['revenueGrowth'] = safe_get(info, 'revenueGrowth')
    
    if overview_av:
        metrics['AV_PE'] = overview_av.get('PERatio')
        metrics['AV_EPS'] = overview_av.get('EPS')
        if metrics['marketCap'] is None and overview_av.get('MarketCapitalization'):
            try:
                metrics['marketCap'] = int(overview_av.get('MarketCapitalization'))
            except: pass
                
    return metrics

def passes_list_filters(metrics: dict, params: dict) -> dict:
    out = {f'list{i}': False for i in range(1, 8)}
    mc = metrics.get('marketCap')
    if mc is None: return out

    # List 1: Value
    try: out['list1'] = (metrics.get('forwardPE') is not None and metrics.get('forwardPE') < 25)
    except: pass

    # List 2: Quality
    try: out['list2'] = (metrics.get('returnOnEquity') is not None and metrics.get('returnOnEquity') > 0) and \
                       (metrics.get('debtToEquity') is not None and metrics.get('debtToEquity') < 10) 
    except: pass

    # List 3: P/S
    out['list3'] = (metrics.get('priceToSales') is not None and metrics.get('priceToSales') < 4)

    # List 4: Div + Value
    out['list4'] = (metrics.get('dividendYield') is not None and metrics.get('dividendYield') > 0.04) and \
                   (metrics.get('priceToBook') is not None and metrics.get('priceToBook') < 1)

    # List 5: GARP
    out['list5'] = (metrics.get('pegRatio') is not None and metrics.get('pegRatio') < 2) and \
                   (metrics.get('profitMargins') is not None and metrics.get('profitMargins') > 0.20)

    # List 6: FCF Yield
    pfcf = None
    if metrics.get('freeCashflow') and mc:
        try: pfcf = mc / metrics.get('freeCashflow')
        except: pass
    out['list6'] = (pfcf is not None and pfcf < 15)

    # List 7: Market Cap
    out['list7'] = (params['min_cap'] <= mc <= params['max_cap'])

    return out

def compute_ecvs_score(row: dict) -> float:
    score = 0.0
    weight = 0.0

    # 1. VALUE (25%)
    val_components = []
    if row.get('forwardPE') and row['forwardPE'] > 0:
        val_components.append(max(0, 25 - row['forwardPE']) / 25)
    if row.get('priceToSales') and row['priceToSales'] > 0:
        val_components.append(max(0, 4 - row['priceToSales']) / 4)
    if row.get('pegRatio') and row['pegRatio'] > 0:
        val_components.append(max(0, 2 - row['pegRatio']) / 2)
    
    if val_components:
        score += (np.mean(val_components) * 100) * 0.25
        weight += 0.25

    # 2. QUALITY (20%)
    if row.get('returnOnEquity'):
        score += min(1, max(0, row['returnOnEquity'] / 0.20)) * 100 * 0.20
        weight += 0.20

    # 3. PROFITABILITY (15%)
    if row.get('profitMargins') is not None:
        score += min(1, max(0, row['profitMargins'] / 0.20)) * 100 * 0.15
        weight += 0.15

    # 4. CASH FLOW (15%)
    if row.get('freeCashflow') and row.get('marketCap'):
        try:
            fcf_yield = row['freeCashflow'] / row['marketCap']
            score += min(1, max(0, fcf_yield / 0.05)) * 100 * 0.15
            weight += 0.15
        except: pass

    # 5. RISK (10%)
    if row.get('debtToEquity') is not None:
        score += max(0, 100 - (row['debtToEquity'] / 2)) * 0.10
        weight += 0.10

    # 6. GROWTH (10%)
    if row.get('forwardEps') and row.get('trailingPE'):
        try:
            growth = min(1, abs(row['forwardEps']) / (row['trailingPE'] if row['trailingPE'] > 0 else 1)) * 100
            score += growth * 0.10
            weight += 0.10
        except: pass

    # 7. THEMATIC (5%)
    if row.get('industry'):
        ind = str(row['industry']).lower()
        if any(x in ind for x in ['semiconductor', 'software', 'ai', 'data', 'technology']):
            score += 100 * 0.05
            weight += 0.05
        else:
            weight += 0.05

    if weight > 0:
        return score / weight
    return 0

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------

def process_universe(tickers, av_key, params, progress_bar, status_text):
    results = []
    total = len(tickers)
    
    for i, t in enumerate(tickers):
        t = t.strip()
        # Ensure upper case, but keep suffix case sensitive if needed (though Yahoo is usually upper)
        t = t.upper() 
        if not t: continue
        
        status_text.text(f"Processing {i+1}/{total}: {t}")
        progress_bar.progress((i + 1) / total)
        
        try:
            tk = yf.Ticker(t)
            info = tk.info
            hist = tk.history(period="1y") 
            
            av_data = fetch_alpha_vantage_financials(t, av_key)
            
            prices = compute_prices_dynamic(hist)
            metrics = compute_simple_metrics(info, av_data)
            metrics.update(prices)
            metrics['ticker'] = t
            
            lists = passes_list_filters(metrics, params)
            metrics['lists'] = lists
            metrics['ecvs'] = compute_ecvs_score(metrics)
            
            if metrics['p_current'] and metrics['p_start_year']:
                metrics['YTD_Perf'] = (metrics['p_current'] / metrics['p_start_year']) - 1
            else: metrics['YTD_Perf'] = None
                
            if metrics['p_current'] and metrics['p_3mo_ago']:
                metrics['3Mo_Perf'] = (metrics['p_current'] / metrics['p_3mo_ago']) - 1
            else: metrics['3Mo_Perf'] = None
                
            for k, v in lists.items():
                metrics[k] = v
                
            results.append(metrics)
            # Gentle rate limiting
            time.sleep(0.1) 
            
        except Exception as e:
            logging.error(f"Error processing {t}: {e}")
            
    return pd.DataFrame(results)

def main():
    st.title("🚀 ECVS Multi-Market Screener")
    st.markdown("""
    **Automated Screener**: Filters stocks through 7 fundamental lists, calculates momentum, and computes the composite **ECVS Score**.
    """)
    
    with st.sidebar:
        st.header("Settings")
        av_api_key = st.text_input("Alpha Vantage API Key (Optional)", type="password")
        
        st.subheader("Universe Selection")
        
        # Expanded options
        universe_options = [
            "Manual Input", 
            "S&P 500 (US)", 
            "Nasdaq 100 (US)", 
            "Dow Jones (US)", 
            "FTSE 100 (UK)", 
            "DAX 40 (Germany)"
        ]
        universe_mode = st.selectbox("Source:", universe_options)
        
        tickers = []
        if universe_mode == "Manual Input":
            ticker_input = st.text_area("Enter Tickers (comma separated)", value="", placeholder="e.g. AAPL, MSFT, SHEL.L, SAP.DE", height=150)
        else:
            st.info(f"ℹ️ Selected {universe_mode}. This will fetch current constituents from Wikipedia.")
            
        st.subheader("Thresholds")
        min_cap_b = st.number_input("Min Market Cap ($B)", value=5.0, step=1.0)
        max_cap_b = st.number_input("List 7 Max Cap ($B)", value=50.0, step=5.0)
        
        run_btn = st.button("Run Screener", type="primary")

    if run_btn:
        # Determine Ticker List
        if universe_mode == "Manual Input":
            if ticker_input.strip():
                tickers = [x.strip() for x in ticker_input.split(',')]
            else:
                st.warning("Please enter tickers or select a Market Index.")
                return
        else:
            # Map selection to index name
            index_map = {
                "S&P 500 (US)": "S&P 500",
                "Nasdaq 100 (US)": "Nasdaq 100",
                "Dow Jones (US)": "Dow Jones",
                "FTSE 100 (UK)": "FTSE 100",
                "DAX 40 (Germany)": "DAX 40"
            }
            target_index = index_map[universe_mode]
            
            with st.spinner(f"Fetching {target_index} constituents..."):
                tickers = get_index_tickers(target_index)
                if tickers:
                    st.success(f"Loaded {len(tickers)} tickers from {target_index}.")
                else:
                    st.error("Failed to load tickers. Please try again or use Manual Input.")
                    return

        params = {
            'min_cap': min_cap_b * 1e9, 
            'max_cap': max_cap_b * 1e9
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        df = process_universe(tickers, av_api_key, params, progress_bar, status_text)
        
        status_text.text("Processing Complete!")
        progress_bar.empty()
        
        if not df.empty:
            list_cols = [f'list{i}' for i in range(1,8)]
            df['Matches_Any'] = df[list_cols].any(axis=1)
            
            numeric_cols = ['ecvs', 'marketCap', 'forwardPE', 'YTD_Perf', '3Mo_Perf']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.sort_values(by='ecvs', ascending=False)
            
            display_cols = ['ticker', 'ecvs', 'marketCap', 'sector', 'forwardPE', 'YTD_Perf', '3Mo_Perf', 'Matches_Any'] + list_cols
            display_cols = [c for c in display_cols if c in df.columns]

            tab1, tab2, tab3 = st.tabs(["🏆 Final Ranking", "📂 Detailed Data", "📥 Export"])
            
            with tab1:
                st.subheader("Top Ranked Candidates (by ECVS Score)")
                
                styled_df = df[display_cols].style.format({
                    'ecvs': '{:.1f}',
                    'marketCap': '${:,.0f}',
                    'forwardPE': '{:.1f}',
                    'YTD_Perf': '{:.1%}',
                    '3Mo_Perf': '{:.1%}'
                }, na_rep="-")
                
                try: styled_df = styled_df.background_gradient(subset=['ecvs'], cmap='Greens')
                except: pass
                    
                st.dataframe(styled_df, use_container_width=True)
            
            with tab2:
                st.subheader("Full Data Table")
                st.dataframe(df)

            with tab3:
                st.subheader("Download Excel Report")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='All_Results', index=False)
                    df[df['Matches_Any']].to_excel(writer, sheet_name='Filtered_Matches', index=False)
                    pd.DataFrame([{'Date': str(datetime.date.today()), 'Source': f'yfinance ({universe_mode})'}]).to_excel(writer, sheet_name='Metadata', index=False)
                
                st.download_button(
                    label="Download .xlsx Workbook",
                    data=output.getvalue(),
                    file_name=f"ecvs_screener_{universe_mode.split()[0]}_{datetime.date.today()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.error("No data found.")

if __name__ == "__main__":
    main()
