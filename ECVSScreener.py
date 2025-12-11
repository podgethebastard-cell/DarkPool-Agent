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
            # convert 'None' string to None type if necessary
            if val == 'None': return default
            return val
    return default

# -----------------------------------------------------------------------------
# CORE METRICS & SCORING LOGIC
# -----------------------------------------------------------------------------

def compute_prices_dynamic(hist: pd.DataFrame) -> dict:
    """
    Dynamically determine Start-of-Year, 3-Month-Ago, and Current prices.
    Uses today's date context (Dec 2025).
    """
    today = datetime.date.today()
    start_of_year = datetime.date(today.year, 1, 1)
    three_months_ago = today - datetime.timedelta(days=90)
    
    # Helper to find closest price
    def get_price(target_date):
        target_ts = pd.Timestamp(target_date)
        if hist.empty:
            return None
        try:
            # Get index location of nearest date
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
    # Identity
    metrics['sector'] = safe_get(info, 'sector')
    metrics['industry'] = safe_get(info, 'industry')
    metrics['country'] = safe_get(info, 'country')
    metrics['employees'] = safe_get(info, 'fullTimeEmployees')
    
    # Valuation
    metrics['marketCap'] = safe_get(info, 'marketCap')
    metrics['forwardPE'] = safe_get(info, 'forwardPE')
    metrics['trailingPE'] = safe_get(info, 'trailingPE')
    metrics['priceToSales'] = safe_get(info, 'priceToSalesTrailing12Months')
    metrics['priceToBook'] = safe_get(info, 'priceToBook')
    metrics['pegRatio'] = safe_get(info, 'pegRatio')
    metrics['forwardEps'] = safe_get(info, 'forwardEps')
    metrics['dividendYield'] = safe_get(info, 'dividendYield')
    
    # Financials
    metrics['freeCashflow'] = safe_get(info, 'freeCashflow')
    metrics['ebitda'] = safe_get(info, 'ebitda')
    metrics['debtToEquity'] = safe_get(info, 'debtToEquity')
    metrics['profitMargins'] = safe_get(info, 'profitMargins')
    metrics['returnOnEquity'] = safe_get(info, 'returnOnEquity')
    metrics['revenueGrowth'] = safe_get(info, 'revenueGrowth')
    
    # Alpha Vantage Verification Data
    if overview_av:
        metrics['AV_PE'] = overview_av.get('PERatio')
        metrics['AV_EPS'] = overview_av.get('EPS')
        # Use AV market cap as fallback if yfinance is missing
        if metrics['marketCap'] is None and overview_av.get('MarketCapitalization'):
            try:
                metrics['marketCap'] = int(overview_av.get('MarketCapitalization'))
            except:
                pass
                
    return metrics

def passes_list_filters(metrics: dict, params: dict) -> dict:
    """Return booleans for the 7 screening lists."""
    out = {f'list{i}': False for i in range(1, 8)}
    
    mc = metrics.get('marketCap')
    if mc is None:
        return out

    # List 1: Value Trap / Standard Value
    try:
        out['list1'] = (metrics.get('forwardPE') is not None and metrics.get('forwardPE') < 25)
    except: pass

    # List 2: Clean Balance Sheet Quality
    try:
        out['list2'] = (metrics.get('returnOnEquity') is not None and metrics.get('returnOnEquity') > 0) and \
                       (metrics.get('debtToEquity') is not None and metrics.get('debtToEquity') < 10) 
    except: pass

    # List 3: Low P/S
    out['list3'] = (metrics.get('priceToSales') is not None and metrics.get('priceToSales') < 4)

    # List 4: Dividend + Value
    out['list4'] = (metrics.get('dividendYield') is not None and metrics.get('dividendYield') > 0.04) and \
                   (metrics.get('priceToBook') is not None and metrics.get('priceToBook') < 1)

    # List 5: GARP (PEG < 2, Margins > 20%)
    out['list5'] = (metrics.get('pegRatio') is not None and metrics.get('pegRatio') < 2) and \
                   (metrics.get('profitMargins') is not None and metrics.get('profitMargins') > 0.20)

    # List 6: FCF Yield
    pfcf = None
    if metrics.get('freeCashflow') and mc:
        try:
            pfcf = mc / metrics.get('freeCashflow')
        except: pass
    out['list6'] = (pfcf is not None and pfcf < 15)

    # List 7: Market Cap Constraints
    out['list7'] = (params['min_cap'] <= mc <= params['max_cap'])

    return out

def compute_ecvs_score(row: dict) -> float:
    """Compute the ECVS composite score (0-100)."""
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

    # 2. QUALITY (20%) - ROE
    if row.get('returnOnEquity'):
        roe_score = min(1, max(0, row['returnOnEquity'] / 0.20)) * 100
        score += roe_score * 0.20
        weight += 0.20

    # 3. PROFITABILITY (15%)
    if row.get('profitMargins') is not None:
        pm_score = min(1, max(0, row['profitMargins'] / 0.20)) * 100
        score += pm_score * 0.15
        weight += 0.15

    # 4. CASH FLOW (15%)
    if row.get('freeCashflow') and row.get('marketCap'):
        try:
            fcf_yield = row['freeCashflow'] / row['marketCap']
            fcf_score = min(1, max(0, fcf_yield / 0.05)) * 100
            score += fcf_score * 0.15
            weight += 0.15
        except: pass

    # 5. RISK (10%) - Debt/Equity
    if row.get('debtToEquity') is not None:
        risk_score = max(0, 100 - (row['debtToEquity'] / 2)) 
        score += risk_score * 0.10
        weight += 0.10

    # 6. GROWTH (10%) - Forward EPS vs Trailing PE Proxy
    if row.get('forwardEps') and row.get('trailingPE'):
        growth_proxy = 0
        try:
            growth_proxy = min(1, abs(row['forwardEps']) / (row['trailingPE'] if row['trailingPE'] > 0 else 1)) * 100
        except: pass
        score += growth_proxy * 0.10
        weight += 0.10

    # 7. AI/THEMATIC BOOST (5%)
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
        t = t.strip().upper()
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
            
            # Performance Calc
            if metrics['p_current'] and metrics['p_start_year']:
                metrics['YTD_Perf'] = (metrics['p_current'] / metrics['p_start_year']) - 1
            else:
                metrics['YTD_Perf'] = None
                
            if metrics['p_current'] and metrics['p_3mo_ago']:
                metrics['3Mo_Perf'] = (metrics['p_current'] / metrics['p_3mo_ago']) - 1
            else:
                metrics['3Mo_Perf'] = None
                
            for k, v in lists.items():
                metrics[k] = v
                
            results.append(metrics)
            time.sleep(0.2)
            
        except Exception as e:
            logging.error(f"Error processing {t}: {e}")
            
    return pd.DataFrame(results)

def main():
    st.title("🚀 ECVS Multi-Symbol Screener")
    st.markdown("""
    **Automated Screener**: Filters stocks through 7 fundamental lists, calculates momentum, and computes the composite **ECVS Score**.
    """)
    
    with st.sidebar:
        st.header("Settings")
        av_api_key = st.text_input("Alpha Vantage API Key (Optional)", type="password")
        
        st.subheader("Universe")
        default_tickers = "AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, BRK-B, V, JNJ"
        ticker_input = st.text_area("Enter Tickers (comma separated)", value=default_tickers, height=150)
        
        st.subheader("Thresholds")
        min_cap_b = st.number_input("Min Market Cap ($B)", value=10.0, step=1.0)
        max_cap_b = st.number_input("List 7 Max Cap ($B)", value=50.0, step=5.0)
        
        run_btn = st.button("Run Screener", type="primary")

    if run_btn:
        tickers = [x.strip() for x in ticker_input.split(',')]
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
            
            # Ensure numeric types for sorting/formatting
            numeric_cols = ['ecvs', 'marketCap', 'forwardPE', 'YTD_Perf', '3Mo_Perf']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.sort_values(by='ecvs', ascending=False)
            
            display_cols = ['ticker', 'ecvs', 'marketCap', 'sector', 'forwardPE', 'YTD_Perf', '3Mo_Perf', 'Matches_Any'] + list_cols
            
            # Filter valid columns only
            display_cols = [c for c in display_cols if c in df.columns]

            tab1, tab2, tab3 = st.tabs(["🏆 Final Ranking", "📂 Detailed Data", "📥 Export"])
            
            with tab1:
                st.subheader("Top Ranked Candidates (by ECVS Score)")
                
                # --- FIX: USE NA_REP TO HANDLE NONE/NAN VALUES SAFELY ---
                styled_df = df[display_cols].style.format({
                    'ecvs': '{:.1f}',
                    'marketCap': '${:,.0f}',
                    'forwardPE': '{:.1f}',
                    'YTD_Perf': '{:.1%}',
                    '3Mo_Perf': '{:.1%}'
                }, na_rep="-")
                
                # Apply gradient only if data exists to avoid errors
                try:
                    styled_df = styled_df.background_gradient(subset=['ecvs'], cmap='Greens')
                except:
                    pass
                    
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
                    pd.DataFrame([{'Date': str(datetime.date.today()), 'Source': 'yfinance + AlphaVantage'}]).to_excel(writer, sheet_name='Metadata', index=False)
                
                st.download_button(
                    label="Download .xlsx Workbook",
                    data=output.getvalue(),
                    file_name=f"ecvs_screener_results_{datetime.date.today()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.error("No data found. Please check your tickers.")

if __name__ == "__main__":
    main()
