import streamlit as st
import pandas as pd
import yfinance as yf
import openai
from datetime import date
import io
import xlsxwriter  # noqa: F401 (used by pandas ExcelWriter engine)
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

# --- DEAL ROOM UX THEME (Unique UX Twist) ---
st.markdown("""
<style>
.deal-card{
  padding: 1.0rem 1.1rem;
  border-radius: 16px;
  border: 1px solid rgba(120,120,120,.25);
  background: rgba(255,255,255,.03);
  margin-bottom: 0.9rem;
}
.pill{
  display:inline-block;
  padding:.18rem .6rem;
  border-radius: 999px;
  border: 1px solid rgba(120,120,120,.25);
  font-size: .80rem;
  opacity: .95;
  margin-right: .35rem;
}
.small{
  font-size: .85rem;
  opacity: .85;
}
</style>
""", unsafe_allow_html=True)

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

# Model choice (more robust than hardcoding)
st.sidebar.markdown("---")
st.sidebar.subheader("AI Model")
model_name = st.sidebar.text_input("Model name", value="gpt-4o-mini")

# Deal Room persona
st.sidebar.markdown("---")
st.sidebar.subheader("Deal Room Mode")
persona = st.sidebar.selectbox(
    "Committee Persona",
    ["Balanced PM", "Value Hawk", "Growth Hunter", "Dividend Steward", "Risk-First CIO"],
    index=0
)

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
# PORTFOLIO COMMITTEE STATE (Unique UX Twist)
# ------------------------------------------------------------------
if "committee_votes" not in st.session_state:
    st.session_state.committee_votes = {}  # ticker -> {"vote": "...", "note": str}
if "watch_notes" not in st.session_state:
    st.session_state.watch_notes = {}  # ticker -> note

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
def calculate_technicals(hist: pd.DataFrame):
    """
    Calculates RSI, SMA50, SMA200, and Volatility from price history.
    Expects a DataFrame with a 'Close' column.
    """
    if hist is None or len(hist) < 200 or "Close" not in hist.columns:
        return {
            "RSI_14": None, "SMA_50": None, "SMA_200": None,
            "Trend": "Insufficient Data", "Volatility": None
        }

    # 1. RSI (14-day)
    delta = hist["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

    # 2. Moving Averages
    sma_50 = float(hist["Close"].rolling(window=50).mean().iloc[-1])
    sma_200 = float(hist["Close"].rolling(window=200).mean().iloc[-1])
    current_price = float(hist["Close"].iloc[-1])

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
    daily_returns = hist["Close"].pct_change()
    volatility = float(daily_returns.std() * np.sqrt(252))

    return {
        "RSI_14": round(current_rsi, 2) if current_rsi is not None else None,
        "SMA_50": round(sma_50, 2),
        "SMA_200": round(sma_200, 2),
        "Trend": trend,
        "Volatility": f"{round(volatility * 100, 1)}%"
    }

# ------------------------------------------------------------------
# 2B. CONVICTION SCORING (Unique UX Twist)
# ------------------------------------------------------------------
def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def compute_conviction_score(row: dict) -> int:
    """
    Returns 0-100 conviction score using rough institutional heuristics.
    (Not investment advice — purely UX scoring.)
    """
    rev = float(row.get("revenue_cagr_3y") or 0)
    eps = float(row.get("eps_growth") or 0)
    margin = float(row.get("profit_margin") or 0)

    pe = float(row.get("forward_pe") or 999)
    pfcf = float(row.get("p_fcf") or 999)
    peg = float(row.get("peg_ratio") or 999)

    de = float(row.get("debt_to_equity") or 999)

    rsi = row.get("RSI_14")
    trend = (row.get("Trend") or "").lower()

    growth_score = _clamp((rev / 0.20) * 0.5 + (eps / 0.40) * 0.5)
    quality_score = _clamp(margin / 0.30)

    val_score = (
        _clamp((25 - pe) / 25) * 0.45 +
        _clamp((20 - pfcf) / 20) * 0.45 +
        _clamp((2 - peg) / 2) * 0.10
    )
    balance_score = _clamp((50 - de) / 50)

    if rsi is None or pd.isna(rsi):
        rsi_score = 0.5
    else:
        rsi_score = _clamp(1 - (abs(float(rsi) - 52) / 35))

    trend_score = 0.5
    if "strong bullish" in trend:
        trend_score = 1.0
    elif "golden cross" in trend:
        trend_score = 0.8
    elif "strong bearish" in trend:
        trend_score = 0.2
    elif "bearish" in trend:
        trend_score = 0.35

    score_0_1 = (
        0.28 * growth_score +
        0.18 * quality_score +
        0.24 * val_score +
        0.14 * balance_score +
        0.08 * rsi_score +
        0.08 * trend_score
    )
    return int(round(100 * _clamp(score_0_1)))

def conviction_label(score: int) -> str:
    if score >= 80:
        return "🔥 High Conviction"
    if score >= 65:
        return "✅ Strong"
    if score >= 50:
        return "🟡 Interesting"
    return "🧊 Speculative"

# ------------------------------------------------------------------
# 3. DATA FETCHING (CACHED, SERIALIZABLE)
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_deep_financial_data(ticker_symbol: str):
    """
    Fetches fundamental data.
    Avoid returning non-serializable objects (like yfinance Ticker).
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info or {}

        # 1. Basic Screening Filter
        market_cap = info.get("marketCap", 0) or 0
        if market_cap < 2_000_000_000:
            return None  # Skip small caps

        # 2. Fetch Raw Statements for Deep Analysis
        try:
            fin = stock.financials
            cash = stock.cashflow
            has_stmts = True if (fin is not None and not fin.empty) else False
        except Exception:
            fin, cash = pd.DataFrame(), pd.DataFrame()
            has_stmts = False

        # 3. Calculate CAGR (3-Year Sales)
        sales_cagr_3y = 0.0
        if has_stmts and "Total Revenue" in fin.index:
            try:
                cols = sorted(fin.columns)
                if len(cols) >= 4:
                    rev_now = float(fin[cols[-1]].loc["Total Revenue"])
                    rev_3yr = float(fin[cols[-4]].loc["Total Revenue"])
                    sales_cagr_3y = (rev_now / rev_3yr) ** (1 / 3) - 1
                elif len(cols) >= 2:
                    rev_now = float(fin[cols[-1]].loc["Total Revenue"])
                    rev_prev = float(fin[cols[0]].loc["Total Revenue"])
                    sales_cagr_3y = (rev_now / rev_prev) - 1
            except Exception:
                sales_cagr_3y = info.get("revenueGrowth", 0) or 0
        else:
            sales_cagr_3y = info.get("revenueGrowth", 0) or 0

        # 4. Build Data Dictionary (ONLY SERIALIZABLE TYPES)
        data = {
            "ticker": ticker_symbol,
            "name": info.get("longName", ticker_symbol),
            "country": info.get("country", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": market_cap,
            "forward_pe": info.get("forwardPE", 0) or 0,
            "revenue_cagr_3y": float(sales_cagr_3y) if sales_cagr_3y is not None else 0,
            "eps_growth": info.get("earningsGrowth", 0) or 0,
            "debt_to_equity": info.get("debtToEquity", 999) or 999,
            "price_to_sales": info.get("priceToSalesTrailing12Months", 999) or 999,
            "dividend_yield": info.get("dividendYield", 0) or 0,
            "price_to_book": info.get("priceToBook", 999) or 999,
            "peg_ratio": info.get("pegRatio", 999) or 999,
            "profit_margin": info.get("profitMargins", 0) or 0,
            "employees": info.get("fullTimeEmployees", 0) or 0,
            "avg_volume": info.get("averageVolume", 0) or 0,
        }

        # FCF Calculation
        try:
            fcf = info.get("freeCashflow", None)
            if (fcf is None) and has_stmts and ("Free Cash Flow" in cash.index):
                # yfinance cashflow orientation can vary; try best-effort
                try:
                    fcf = float(cash.loc["Free Cash Flow"].iloc[0])
                except Exception:
                    fcf = None

            if fcf and fcf > 0:
                data["p_fcf"] = market_cap / float(fcf)
                shares = info.get("sharesOutstanding", 1) or 1
                data["fcf_share"] = float(fcf) / float(shares)
            else:
                data["p_fcf"] = 999
                data["fcf_share"] = 0
        except Exception:
            data["p_fcf"] = 999
            data["fcf_share"] = 0

        data["insider_percent"] = info.get("heldPercentInsiders", 0) or 0

        return data
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_price_history(ticker_symbol: str):
    """Fetches 1 year of history for calculations (cached for speed)."""
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        return hist
    except Exception:
        return None

def get_price_at_date(hist: pd.DataFrame, target_date_str: str):
    target_date = pd.to_datetime(target_date_str).tz_localize(None)
    if hist is None or hist.empty:
        return None
    h = hist.copy()
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    h = h.sort_index()
    try:
        idx = h.index.get_indexer([target_date], method="nearest")[0]
        return float(h.iloc[idx]["Close"])
    except Exception:
        return float(h["Close"].iloc[-1])

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
        if not data:
            continue

        # 2. Get Price History & Technicals
        hist = get_price_history(data["ticker"])
        technicals = calculate_technicals(hist)
        data.update(technicals)

        # 3. Apply Filters
        pe = data["forward_pe"] or 999
        rev_growth = data["revenue_cagr_3y"] or -1.0
        eps_growth = data["eps_growth"] or -1.0
        de = data["debt_to_equity"] or 999
        ps = data["price_to_sales"] or 999
        insider = data["insider_percent"] or 0
        div = data["dividend_yield"] or 0
        pb = data["price_to_book"] or 999
        peg = data["peg_ratio"] or 999
        margin = data["profit_margin"] or 0
        p_fcf = data["p_fcf"] or 999
        mkt_cap = data["market_cap"] or 0

        l1 = (pe < 25) and (rev_growth > 0.05)
        l2 = (eps_growth > 0.25) and (de < 15)
        l3 = (ps < 4) and (insider > 0.50)
        l4 = (div > 0.04) and (pb < 1.5)
        l5 = (peg < 2) and (margin > 0.20)
        l6 = (p_fcf < 20) and (div > 0.02)
        l7 = (2e9 <= mkt_cap <= 20e9) and (pe < 20) and (eps_growth > 0.15)

        if any([l1, l2, l3, l4, l5, l6, l7]):
            # NOTE: you used 2025 dates in the original prototype. Keeping as-is.
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
def analyze_with_ai_deep(row, api_key, model_name: str, persona: str):
    client = openai.OpenAI(api_key=api_key)

    prompt = f"""
Act as a {persona}. Analyze this stock deeply and concisely.

[PROFILE]
Name: {row.get('name', '')} ({row.get('ticker', '')})
Region: {row.get('country', '')} | Industry: {row.get('industry', '')}

[VALUATION]
Forward P/E: {row.get('forward_pe', '')} | PEG: {row.get('peg_ratio', '')} | P/B: {row.get('price_to_book', '')}
P/FCF: {row.get('p_fcf', '')} | Div Yield: {(row.get('dividend_yield', 0) * 100):.2f}%

[GROWTH & HEALTH]
Rev CAGR (3Y): {(row.get('revenue_cagr_3y', 0) * 100):.2f}% | EPS Growth (1Y): {(row.get('eps_growth', 0) * 100):.2f}%
Profit Margin: {(row.get('profit_margin', 0) * 100):.2f}% | Debt/Equity: {row.get('debt_to_equity', '')}

[TECHNICALS]
Price Trend: {row.get('Trend', '')}
RSI (14): {row.get('RSI_14', '')} (Overbought > 70, Oversold < 30)
Volatility (Ann): {row.get('Volatility', '')}

OUTPUT REQUIREMENTS:
Provide 4 sections separated STRICTLY by the pipe symbol "|".
1. VERDICT: Buy, Hold, or Sell (short justification).
2. FUNDAMENTAL THESIS: 2 sentences on valuation/growth/quality balance.
3. TECHNICAL VIEW: 1 sentence interpreting RSI + Trend.
4. RISKS/CATALYSTS: 1 sentence on key upcoming risks or catalysts.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a concise, high-level investment analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5
        )
        content = response.choices[0].message.content or ""
        parts = content.split("|")
        if len(parts) < 4:
            return ["Hold", "AI Parsing Error", "AI Parsing Error", "AI Parsing Error"]
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
            data={"chat_id": chat_id, "caption": "📈 Deep-Dive Analysis File"},
            files={"document": (filename, excel_buffer, "application/vnd.ms-excel")}
        )
        return True
    except Exception as e:
        st.error(f"Telegram Error: {e}")
        return False

# ------------------------------------------------------------------
# MAIN EXECUTION STATE
# ------------------------------------------------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "final_df" not in st.session_state:
    st.session_state.final_df = None
if "excel_data" not in st.session_state:
    st.session_state.excel_data = None

# ------------------------------------------------------------------
# RUN BUTTON
# ------------------------------------------------------------------
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
            top_jan = df.sort_values(by="perf_since_jan", ascending=False).head(15)
            top_apr = df.sort_values(by="perf_since_apr", ascending=False).head(15)
            shortlist = pd.concat([top_jan, top_apr]).drop_duplicates(subset="ticker")
            shortlist = shortlist.sort_values(by="revenue_cagr_3y", ascending=False)

            # Diversity filter: 1 per (country, industry) bucket
            final_rows, seen = [], set()
            for _, row in shortlist.iterrows():
                k = (row["country"], row["industry"])
                if k not in seen:
                    final_rows.append(row)
                    seen.add(k)
                if len(final_rows) >= 10:
                    break

            final_df = pd.DataFrame(final_rows)

            st.success(f"Identified {len(final_df)} High-Conviction Candidates.")

            # AI Analysis Loop
            st.subheader("2. Generating Analyst Memos...")
            prog = st.progress(0)
            for i, idx in enumerate(final_df.index):
                row = final_df.loc[idx]
                insights = analyze_with_ai_deep(row, api_key, model_name=model_name, persona=persona)
                final_df.loc[idx, "AI_Verdict"] = insights[0]
                final_df.loc[idx, "AI_Thesis"] = insights[1]
                final_df.loc[idx, "AI_Technical_View"] = insights[2]
                final_df.loc[idx, "AI_Risks"] = insights[3]
                final_df.loc[idx, "Conviction"] = compute_conviction_score(row.to_dict())
                prog.progress((i + 1) / len(final_df))
            prog.empty()

            # Store in Session State
            st.session_state.final_df = final_df
            st.session_state.analysis_done = True

            # Create Excel for Session
            output_df = pd.DataFrame()
            output_df["Ticker"] = final_df["ticker"]
            output_df["Name"] = final_df["name"]
            output_df["Country"] = final_df["country"]
            output_df["Industry"] = final_df["industry"]
            output_df["Price (Current)"] = final_df["price_jun25"]
            output_df["Perf YTD"] = final_df["perf_since_jan"]
            output_df["P/E (Fwd)"] = final_df["forward_pe"]
            output_df["P/FCF"] = final_df["p_fcf"]
            output_df["RSI (14)"] = final_df["RSI_14"]
            output_df["Trend Status"] = final_df["Trend"]
            output_df["Conviction (0-100)"] = final_df.get("Conviction", "")
            output_df["Analyst Verdict"] = final_df["AI_Verdict"]
            output_df["Fundamental Thesis"] = final_df["AI_Thesis"]
            output_df["Risks"] = final_df["AI_Risks"]

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                output_df.to_excel(writer, index=False, sheet_name="Deep Analysis")
            buffer.seek(0)
            st.session_state.excel_data = buffer.getvalue()

# ------------------------------------------------------------------
# DISPLAY RESULTS & SIGNALS (Deal Room UX Twist)
# ------------------------------------------------------------------
if st.session_state.analysis_done and st.session_state.final_df is not None:
    final_df = st.session_state.final_df

    st.write("### 🏛️ Portfolio Committee — Deal Room")

    tab_deal, tab_table, tab_watch = st.tabs(["🃏 Deal Room", "📊 Table View", "👀 Watchlist"])

    with tab_table:
        cols = ["ticker", "name", "country", "forward_pe", "RSI_14", "Trend", "AI_Verdict"]
        if "Conviction" in final_df.columns:
            cols = cols[:3] + ["Conviction"] + cols[3:]
        st.dataframe(final_df[cols], use_container_width=True)

    with tab_deal:
        st.caption("Vote like an investment committee: ✅ Approve, 👀 Watch, ❌ Pass. Notes + Watchlist build automatically.")

        for idx in final_df.index:
            row = final_df.loc[idx]
            t = row["ticker"]
            name = row.get("name", t)

            # sparkline (last 90 closes)
            hist = get_price_history(t)
            spark = None
            if hist is not None and not hist.empty and "Close" in hist.columns:
                spark = hist["Close"].tail(90)

            score = compute_conviction_score(row.to_dict())
            label = conviction_label(score)

            current_vote = st.session_state.committee_votes.get(t, {}).get("vote", "—")

            st.markdown(f"""
            <div class="deal-card">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <span class="pill"><b>{t}</b></span>
                  <span class="pill">{row.get("country","")}</span>
                  <span class="pill">{row.get("industry","")}</span>
                  <span class="pill">Vote: <b>{current_vote}</b></span>
                </div>
                <div class="pill">{label} · <b>{score}/100</b></div>
              </div>
              <div class="small" style="margin-top:.35rem;"><b>{name}</b></div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1.2, 1.6, 1.2])

            with c1:
                st.metric("Forward P/E", f"{row.get('forward_pe', 0):.2f}" if row.get("forward_pe") else "—")
                st.metric("P/FCF", f"{row.get('p_fcf', 0):.1f}" if row.get("p_fcf") else "—")
                st.metric("Rev CAGR (3Y)", f"{(row.get('revenue_cagr_3y',0)*100):.1f}%")
                st.metric("Margin", f"{(row.get('profit_margin',0)*100):.1f}%")
                st.progress(score / 100)

            with c2:
                if spark is not None and len(spark) > 5:
                    st.line_chart(spark)
                st.write("**AI Verdict**:", row.get("AI_Verdict", "—"))
                st.write("**Thesis**:", row.get("AI_Thesis", "—"))
                st.write("**Technical**:", row.get("AI_Technical_View", "—"))
                st.write("**Risks/Catalysts**:", row.get("AI_Risks", "—"))

            with c3:
                b1, b2, b3 = st.columns(3)

                def _toast(msg, icon=None):
                    # st.toast exists in newer streamlit; fail gracefully
                    try:
                        st.toast(msg, icon=icon)
                    except Exception:
                        st.info(msg)

                with b1:
                    if st.button("✅ Approve", key=f"approve_{t}"):
                        st.session_state.committee_votes[t] = {"vote": "Approve", "note": st.session_state.watch_notes.get(t, "")}
                        _toast(f"{t} approved ✅", icon="✅")
                with b2:
                    if st.button("👀 Watch", key=f"watch_{t}"):
                        st.session_state.committee_votes[t] = {"vote": "Watch", "note": st.session_state.watch_notes.get(t, "")}
                        _toast(f"{t} added to watchlist 👀", icon="👀")
                with b3:
                    if st.button("❌ Pass", key=f"pass_{t}"):
                        st.session_state.committee_votes[t] = {"vote": "Pass", "note": st.session_state.watch_notes.get(t, "")}
                        _toast(f"{t} passed ❌", icon="❌")

                st.session_state.watch_notes[t] = st.text_area(
                    "Committee note (optional)",
                    value=st.session_state.watch_notes.get(t, ""),
                    key=f"note_{t}",
                    height=80,
                    placeholder="Why are we voting this way?"
                )
                if t in st.session_state.committee_votes:
                    st.session_state.committee_votes[t]["note"] = st.session_state.watch_notes[t]

            st.markdown("---")

    with tab_watch:
        votes = st.session_state.committee_votes
        if not votes:
            st.info("No committee votes yet. Go to Deal Room and vote ✅/👀/❌.")
        else:
            rows = []
            for t, meta in votes.items():
                if meta.get("vote") in ("Approve", "Watch"):
                    sub = final_df[final_df["ticker"] == t]
                    if not sub.empty:
                        r = sub.iloc[0].to_dict()
                        rows.append({
                            "Ticker": t,
                            "Vote": meta.get("vote"),
                            "Conviction": compute_conviction_score(r),
                            "Verdict": r.get("AI_Verdict", ""),
                            "Country": r.get("country", ""),
                            "Industry": r.get("industry", ""),
                            "Note": meta.get("note", "")
                        })
            watch_df = pd.DataFrame(rows)
            if not watch_df.empty:
                watch_df = watch_df.sort_values(["Vote", "Conviction"], ascending=[True, False])

            if watch_df.empty:
                st.warning("You voted, but nothing is in Approve/Watch yet.")
            else:
                st.dataframe(watch_df, use_container_width=True)

                wbuf = io.BytesIO()
                with pd.ExcelWriter(wbuf, engine="xlsxwriter") as writer:
                    watch_df.to_excel(writer, index=False, sheet_name="Watchlist")
                wbuf.seek(0)

                st.download_button(
                    "📥 Download Committee Watchlist",
                    data=wbuf.getvalue(),
                    file_name=f"Committee_Watchlist_{date.today()}.xlsx"
                )

    # Downloads + Telegram remain below (outside tabs)
    st.markdown("### 📦 Exports & Signals")
    col1, col2 = st.columns([1, 1])

    with col1:
        fname = f"Deep_Analysis_{date.today()}.xlsx"
        st.download_button("📥 Download Excel Report", data=st.session_state.excel_data, file_name=fname)

    with col2:
        if use_telegram:
            if st.button("📡 Broadcast Signal to Telegram"):
                st.info("Sending Signal...")

                # Choose best: Approved first by conviction, otherwise top row
                top_stock = None
                approved = [t for t, v in st.session_state.committee_votes.items() if v.get("vote") == "Approve"]
                if approved:
                    sub = final_df[final_df["ticker"].isin(approved)].copy()
                    sub["Conviction"] = sub.apply(lambda r: compute_conviction_score(r.to_dict()), axis=1)
                    sub = sub.sort_values("Conviction", ascending=False)
                    top_stock = sub.iloc[0]
                else:
                    top_stock = final_df.iloc[0]

                signal_msg = f"""
🚨 **DARK POOL AGENT SIGNAL** 🚨

**Top Pick:** {top_stock.get('name','')} ({top_stock.get('ticker','')})
**Verdict:** {top_stock.get('AI_Verdict','')}

**Deal Room:**
• Conviction: {compute_conviction_score(top_stock.to_dict())}/100
• RSI (14): {top_stock.get('RSI_14','—')}
• Trend: {top_stock.get('Trend','—')}

**Thesis:** {top_stock.get('AI_Thesis','—')}
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
