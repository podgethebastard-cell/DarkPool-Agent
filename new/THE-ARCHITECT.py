import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import io

# ============================================================
# THE ARCHITECT — GLOBAL EQUITY ALPHA SCREENER (Merged Build)
# Pine-logic table + Streamlit DPC UI + 7-list rating + export
# ============================================================

# -------------------------------
# THEME / CSS (DPC ARCHITECT)
# -------------------------------
def inject_dpc_css(bg="#0e1117", text="#e0e0e0", accent="#00d4ff", neon_green="#00ffbb", neon_red="#ff3355", header="#151a22"):
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

        :root {{
            --bg-color: {bg};
            --text-color: {text};
            --accent: {accent};
            --neon-green: {neon_green};
            --neon-red: {neon_red};
            --header: {header};
        }}

        .stApp {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Roboto Mono', monospace;
        }}

        h1, h2, h3 {{
            color: var(--text-color);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.35);
        }}

        .card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(0, 212, 255, 0.18);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.35);
        }}

        .scanline {{
            position: relative;
            overflow: hidden;
        }}
        .scanline:after {{
            content: "";
            position: absolute;
            top: -40%;
            left: 0;
            width: 100%;
            height: 40%;
            background: linear-gradient(180deg, transparent, rgba(0, 212, 255, 0.06), transparent);
            animation: scan 3.2s linear infinite;
        }}
        @keyframes scan {{
            0%   {{ top: -40%; }}
            100% {{ top: 120%; }}
        }}

        .chip {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(0, 212, 255, 0.25);
            background: rgba(0, 212, 255, 0.06);
            margin-right: 6px;
            margin-bottom: 6px;
            font-size: 12px;
        }}

        .good {{
            color: var(--neon-green);
            text-shadow: 0 0 8px rgba(0, 255, 187, 0.35);
            font-weight: 700;
        }}
        .bad {{
            color: var(--neon-red);
            text-shadow: 0 0 8px rgba(255, 51, 85, 0.35);
            font-weight: 700;
        }}

        /* Buttons */
        .stButton>button {{
            background-color: transparent !important;
            color: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 10px !important;
            padding: 10px 18px !important;
            transition: 0.25s;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 1px;
        }}
        .stButton>button:hover {{
            background-color: var(--accent) !important;
            color: #05070c !important;
            box-shadow: 0 0 18px rgba(0, 212, 255, 0.35);
            transform: translateY(-1px);
        }}

        /* Dataframe */
        .stDataFrame, .dataframe {{
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 14px !important;
            overflow: hidden !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# TradingView -> yfinance mapping
# -------------------------------
def tv_to_yf_symbol(tv: str) -> str:
    """
    Converts TradingView-style symbols (e.g. NASDAQ:NVDA, LSE:AZN, TSX:SHOP)
    into yfinance-compatible tickers where possible.
    Falls back to the raw right-hand side.
    """
    tv = tv.strip()
    if not tv:
        return tv

    if ":" not in tv:
        return tv  # already likely yfinance-compatible

    exch, sym = tv.split(":", 1)
    exch = exch.upper().strip()
    sym = sym.strip()

    # Common exchange suffix conversions (best-effort)
    suffix_map = {
        "NASDAQ": "",
        "NYSE": "",
        "AMEX": "",
        "TSX": ".TO",
        "TSXV": ".V",
        "LSE": ".L",
        "ETR": ".DE",
        "FWB": ".F",
        "SIX": ".SW",
        "TSE": ".T",       # Japan TSE
        "JPX": ".T",
        "ASX": ".AX",
        "HKG": ".HK",
        "HKEX": ".HK",
        "SGX": ".SI",
    }

    suf = suffix_map.get(exch, "")
    # TradingView often uses numeric HK tickers like 0700 -> yfinance expects 0700.HK
    return f"{sym}{suf}"

def parse_universe(input_string: str) -> list[str]:
    parts = [p.strip() for p in input_string.split(",") if p.strip()]
    return [tv_to_yf_symbol(p) for p in parts]

# -------------------------------
# Data fetching (cached)
# -------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)  # 6 hours
def fetch_prices_close(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    # yfinance end is exclusive-ish; add a day for safety
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)
    df = yf.download(
        tickers=tickers,
        start=pd.to_datetime(start),
        end=end_dt,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False
    )
    # Normalize to a Close matrix: index=date, columns=tickers
    if isinstance(df.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                closes[t] = df[(t, "Close")]
        close_df = pd.DataFrame(closes)
    else:
        # single ticker
        close_df = pd.DataFrame({tickers[0]: df["Close"]}) if "Close" in df.columns else pd.DataFrame()

    close_df = close_df.sort_index().dropna(how="all")
    return close_df

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)  # 24 hours
def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}

        def g(key, default=np.nan):
            v = info.get(key, default)
            return default if v is None else v

        # Best-effort standardization
        rows.append({
            "Ticker": t,
            "Name": g("longName", t),
            "Exchange": g("exchange", "N/A"),
            "Country": g("country", "N/A"),
            "Industry": g("industry", "N/A"),
            "MarketCap": g("marketCap", np.nan),

            # Pine fields analogs
            "ForwardPE": g("forwardPE", np.nan),
            "RevGrowth_1Y": g("revenueGrowth", np.nan),     # fraction, e.g. 0.12
            "EPSGrowth_1Y": g("earningsGrowth", np.nan),    # fraction
            "DebtToEquity": g("debtToEquity", np.nan),
            "PSFwd": g("priceToSalesTrailing12Months", np.nan),  # proxy
            "DivYield": g("dividendYield", np.nan),         # fraction
            "PBRatio": g("priceToBook", np.nan),
            "PEGRatio": g("pegRatio", np.nan),
            "NetMargin": g("profitMargins", np.nan),

            # FCF valuation proxy
            "FreeCashflow": g("freeCashflow", np.nan),
            "SharesOut": g("sharesOutstanding", np.nan),
            "CurrentPrice": g("currentPrice", np.nan),
        })

    df = pd.DataFrame(rows)

    # Derived fields
    df["FCF_PerShare"] = np.where(
        (df["FreeCashflow"].notna()) & (df["SharesOut"].notna()) & (df["SharesOut"] > 0),
        df["FreeCashflow"] / df["SharesOut"],
        np.nan
    )
    df["P_FCF"] = np.where(
        (df["CurrentPrice"].notna()) & (df["FCF_PerShare"].notna()) & (df["FCF_PerShare"] > 0),
        df["CurrentPrice"] / df["FCF_PerShare"],
        np.nan
    )

    return df

# -------------------------------
# Pine-like screening engine (7 lists)
# -------------------------------
def compute_lists(df: pd.DataFrame, min_mcap_stg1=10e9, min_mcap_stg7=2e9, max_mcap_stg7=20e9) -> pd.DataFrame:
    out = df.copy()

    # Convert growth fractions into percentages for comparisons, safely
    rev_g_pct = out["RevGrowth_1Y"] * 100
    eps_g_pct = out["EPSGrowth_1Y"] * 100
    div_y_pct = out["DivYield"] * 100

    m = out["MarketCap"]
    pe = out["ForwardPE"]
    sg = rev_g_pct
    eg = eps_g_pct
    de = out["DebtToEquity"]
    ps = out["PSFwd"]
    dy = div_y_pct
    pb = out["PBRatio"]
    peg = out["PEGRatio"]
    nm = out["NetMargin"] * 100
    pfcf = out["P_FCF"]

    # Pine definitions (fixed L5 bug, aligned to intent)
    out["L1"] = (m >= min_mcap_stg1) & (pe < 25) & (sg > 5)
    out["L2"] = (m >= min_mcap_stg1) & (eg > 25) & (de == 0)
    out["L3"] = (m >= min_mcap_stg1) & (ps < 4)
    out["L4"] = (m >= min_mcap_stg1) & (dy > 4) & (pb < 1)
    out["L5"] = (m >= min_mcap_stg1) & (peg < 2) & (nm > 20)
    out["L6"] = (m >= min_mcap_stg1) & (pfcf < 15) & (dy > 0)
    out["L7"] = (m > min_mcap_stg7) & (m < max_mcap_stg7) & (pe < 15) & (eg > 15)

    list_cols = ["L1","L2","L3","L4","L5","L6","L7"]
    out["PassAny"] = out[list_cols].any(axis=1)
    out["ListCount"] = out[list_cols].sum(axis=1).astype(int)

    def rating_letter(n: int) -> str:
        if n >= 5: return "A+"
        if n == 4: return "A"
        if n == 3: return "B"
        if n == 2: return "C"
        if n == 1: return "D"
        return "F"

    out["Rating"] = out["ListCount"].apply(rating_letter)
    out["ListsHit"] = out[list_cols].apply(lambda r: ",".join([c for c in list_cols if bool(r[c])]) if r.any() else "-", axis=1)
    return out

# -------------------------------
# Performance since anchors
# -------------------------------
def nearest_price(series: pd.Series, anchor: pd.Timestamp) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    # nearest trading day ON/AFTER anchor; if none, use last before
    ix = s.index
    pos = ix.searchsorted(anchor, side="left")
    if pos < len(ix):
        return float(s.iloc[pos])
    return float(s.iloc[-1])

def compute_performance(df_f: pd.DataFrame, close_df: pd.DataFrame, jan_anchor: date, apr_anchor: date, asof: date) -> pd.DataFrame:
    out = df_f.copy()
    if close_df.empty:
        out["Perf_Jan"] = np.nan
        out["Perf_Apr"] = np.nan
        out["Price_Jan"] = np.nan
        out["Price_Apr"] = np.nan
        out["Price_AsOf"] = np.nan
        return out

    jan_ts = pd.to_datetime(jan_anchor)
    apr_ts = pd.to_datetime(apr_anchor)
    asof_ts = pd.to_datetime(asof)

    # Use last available close on/before asof
    close_cut = close_df[close_df.index <= asof_ts].copy()
    if close_cut.empty:
        close_cut = close_df.copy()

    price_asof = close_cut.ffill().iloc[-1]

    jan_prices = {}
    apr_prices = {}
    for t in close_cut.columns:
        s = close_cut[t]
        jan_prices[t] = nearest_price(s, jan_ts)
        apr_prices[t] = nearest_price(s, apr_ts)

    out = out.merge(
        pd.DataFrame({
            "Ticker": close_cut.columns,
            "Price_Jan": [jan_prices[t] for t in close_cut.columns],
            "Price_Apr": [apr_prices[t] for t in close_cut.columns],
            "Price_AsOf": [float(price_asof[t]) if pd.notna(price_asof[t]) else np.nan for t in close_cut.columns],
        }),
        on="Ticker",
        how="left"
    )

    out["Perf_Jan"] = (out["Price_AsOf"] - out["Price_Jan"]) / out["Price_Jan"] * 100
    out["Perf_Apr"] = (out["Price_AsOf"] - out["Price_Apr"]) / out["Price_Apr"] * 100

    return out

# -------------------------------
# Formatting helpers
# -------------------------------
def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.2f}%"

def fmt_num(x, dp=2):
    return "—" if pd.isna(x) else f"{x:.{dp}f}"

def fmt_mcap(x):
    if pd.isna(x): return "—"
    v = float(x)
    if v >= 1e12: return f"{v/1e12:.2f}T"
    if v >= 1e9:  return f"{v/1e9:.2f}B"
    if v >= 1e6:  return f"{v/1e6:.2f}M"
    return f"{v:,.0f}"

def style_table(df: pd.DataFrame):
    # Conditional styling: performance and rating
    def color_perf(v):
        if pd.isna(v): return ""
        return "color: #00ffbb;" if v >= 0 else "color: #ff3355;"

    def color_rating(v):
        if v in ("A+","A","B"):
            return "color: #00ffbb; font-weight: 700;"
        if v in ("C","D"):
            return "color: #e0e0e0; font-weight: 700;"
        return "color: #ff3355; font-weight: 700;"

    sty = (df.style
           .applymap(color_perf, subset=["Perf (Jan)", "Perf (Apr)"])
           .applymap(color_rating, subset=["Rating"])
          )
    return sty

# -------------------------------
# Excel export
# -------------------------------
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Global_Alpha_Screener")
        wb = writer.book
        ws = writer.sheets["Global_Alpha_Screener"]

        header_fmt = wb.add_format({"bold": True, "bg_color": "#0e1117", "font_color": "#00d4ff", "border": 1})
        cell_fmt = wb.add_format({"font_color": "#e0e0e0", "bg_color": "#111522", "border": 1})

        for c, col in enumerate(df.columns):
            ws.write(0, c, col, header_fmt)
            ws.set_column(c, c, min(24, max(12, len(col) + 2)), cell_fmt)

    return buffer.getvalue()

# ============================================================
# APP
# ============================================================
def main():
    st.set_page_config(page_title="Global Equity Alpha Screener | THE ARCHITECT", layout="wide")

    # --- Sidebar controls (merged Pine inputs + UI style) ---
    st.sidebar.markdown("## ⚙️ CONTROL DECK")

    input_tickers = st.sidebar.text_area(
        "Analysis Universe (comma-separated)",
        value="NASDAQ:NVDA, NASDAQ:AAPL, TSX:SHOP, LSE:AZN, ETR:SAP, SIX:NESN, TSE:7203, ASX:BHP, HKG:0700, SGX:D05",
        help="TradingView-style supported (EXCHANGE:SYMBOL). Best-effort conversion to yfinance tickers."
    )

    show_table = st.sidebar.toggle("Display Professional Dashboard", value=True)

    st.sidebar.markdown("### 🎛️ Style UI")
    bg_color = st.sidebar.color_picker("Dashboard Background", "#0e1117")
    hd_color = st.sidebar.color_picker("Header Background", "#151a22")
    txt_color = st.sidebar.color_picker("Primary Text Color", "#e0e0e0")
    acc_color = st.sidebar.color_picker("Accent/Bullish Color", "#00ffbb")

    inject_dpc_css(bg=bg_color, text=txt_color, accent="#00d4ff", neon_green=acc_color, neon_red="#ff3355", header=hd_color)

    st.title("🏛️ THE ARCHITECT — Global Equity Alpha Screener")
    st.caption("7-List Quant Screen • Jan 1, 2025 & Apr 1, 2025 anchors • Pro dashboard table • Drill-down chart • Excel export")

    with st.expander("SYSTEM MANIFEST (Merged Spec)", expanded=False):
        st.markdown(
            """
            - **Quant Engine:** Pine-inspired 7-List screening (value, growth, quality, cashflow).
            - **Performance Anchors:** Since **Jan 1, 2025** and **Apr 1, 2025** to *as-of* date.
            - **Output:** Professional table with rating + list membership, filters, and export.
            - **X-Factor:** **“Scanline Mode”** — the dashboard feels like a live institutional terminal.
            """
        )

    # --- Date anchors (Pine) ---
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        jan_anchor = st.date_input("Anchor A (Jan)", value=date(2025, 1, 1))
    with colB:
        apr_anchor = st.date_input("Anchor B (Apr)", value=date(2025, 4, 1))
    with colC:
        asof = st.date_input("As-Of Date", value=date.today())
    with colD:
        max_rows = st.number_input("Max rows", min_value=5, max_value=200, value=50, step=5)

    # --- Threshold controls (Pine constants) ---
    st.markdown("### 🧠 Quant Thresholds")
    t1, t2, t3 = st.columns([1, 1, 1])
    with t1:
        min_mcap_stg1_b = st.number_input("MIN MCAP Stage 1 (B)", min_value=1.0, max_value=500.0, value=10.0, step=1.0)
    with t2:
        min_mcap_stg7_b = st.number_input("MIN MCAP Stage 7 (B)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
    with t3:
        max_mcap_stg7_b = st.number_input("MAX MCAP Stage 7 (B)", min_value=1.0, max_value=500.0, value=20.0, step=1.0)

    tickers = parse_universe(input_tickers)
    tickers = [t for t in tickers if t]  # clean

    # --- Filters ---
    st.markdown("### 🧷 Filters")
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    with f1:
        only_pass = st.toggle("Show only passes", value=True)
    with f2:
        min_rating = st.selectbox("Min rating", ["F", "D", "C", "B", "A", "A+"], index=3)
    with f3:
        sort_by = st.selectbox("Sort by", ["Rating", "Perf_Jan", "Perf_Apr", "RevGrowth_1Y", "EPSGrowth_1Y", "ForwardPE"], index=1)
    with f4:
        ascending = st.toggle("Ascending sort", value=False)

    execute = st.button("🚀 EXECUTE SCREEN", use_container_width=True)

    if not execute:
        st.info("Load your universe, tune thresholds, then hit **EXECUTE SCREEN**.")
        return

    if not tickers:
        st.error("No tickers detected. Add at least one symbol.")
        return

    # --- Fetch + compute ---
    with st.status("Initializing Neural Engine…", expanded=True) as status:
        status.write("🛰️ Pulling fundamentals (yfinance)…")
        f_df = fetch_fundamentals(tickers)

        status.write("📈 Pulling daily closes for anchors…")
        start = min(jan_anchor, apr_anchor) - timedelta(days=12)
        close_df = fetch_prices_close(tickers, start=start, end=asof)

        status.write("🧬 Applying Pine-inspired 7-List screen…")
        screened = compute_lists(
            f_df,
            min_mcap_stg1=min_mcap_stg1_b * 1e9,
            min_mcap_stg7=min_mcap_stg7_b * 1e9,
            max_mcap_stg7=max_mcap_stg7_b * 1e9
        )

        status.write("🧾 Computing performance since anchors…")
        final = compute_performance(screened, close_df, jan_anchor, apr_anchor, asof)

        status.update(label="Analysis Complete", state="complete", expanded=False)

    # --- Apply filters ---
    rating_rank = {"F":0,"D":1,"C":2,"B":3,"A":4,"A+":5}
    min_r = rating_rank[min_rating]

    filt = final.copy()
    if only_pass:
        filt = filt[filt["PassAny"] == True]
    filt = filt[filt["Rating"].map(rating_rank) >= min_r]

    # Add display columns to match Pine table
    display = filt.copy()
    display["Perf (Jan)"] = display["Perf_Jan"]
    display["Perf (Apr)"] = display["Perf_Apr"]
    display["Fwd P/E"] = display["ForwardPE"]
    display["Sales G."] = display["RevGrowth_1Y"] * 100
    display["EPS G."] = display["EPSGrowth_1Y"] * 100
    display["D/E"] = display["DebtToEquity"]
    display["PEG"] = display["PEGRatio"]
    display["Div Yield"] = display["DivYield"] * 100

    # Sort
    if sort_by == "Rating":
        display["_rating_score"] = display["Rating"].map(rating_rank)
        display = display.sort_values("_rating_score", ascending=ascending).drop(columns=["_rating_score"])
    else:
        display = display.sort_values(sort_by, ascending=ascending)

    # Limit rows
    display = display.head(int(max_rows))

    # --- Headline summary cards ---
    st.markdown("### 🧿 Terminal Overview")
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        st.markdown(f"""
        <div class="card scanline">
            <div style="opacity:0.8;font-size:12px;">Universe</div>
            <div style="font-size:24px;font-weight:700;">{len(tickers)}</div>
            <div style="opacity:0.75;font-size:12px;">symbols loaded</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card scanline">
            <div style="opacity:0.8;font-size:12px;">Passing</div>
            <div style="font-size:24px;font-weight:700;">{int(final["PassAny"].sum())}</div>
            <div style="opacity:0.75;font-size:12px;">hit ≥1 list</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        best = display.dropna(subset=["Perf (Jan)"]).sort_values("Perf (Jan)", ascending=False).head(1)
        best_txt = "—"
        if not best.empty:
            best_txt = f'{best.iloc[0]["Ticker"]}  {best.iloc[0]["Perf (Jan)"]:.2f}%'
        st.markdown(f"""
        <div class="card scanline">
            <div style="opacity:0.8;font-size:12px;">Top Since Jan</div>
            <div style="font-size:20px;font-weight:700;" class="good">{best_txt}</div>
            <div style="opacity:0.75;font-size:12px;">as of {asof.isoformat()}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        a_count = int((final["Rating"].isin(["A","A+","A+"]) ).sum())
        st.markdown(f"""
        <div class="card scanline">
            <div style="opacity:0.8;font-size:12px;">High Conviction</div>
            <div style="font-size:24px;font-weight:700;">{a_count}</div>
            <div style="opacity:0.75;font-size:12px;">rated A / A+</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Dashboard table (Pine columns) ---
    if show_table:
        st.markdown("### 📋 Professional Dashboard (Pine Table Replica + Rating)")

        table_df = display[[
            "Ticker", "Country", "Industry",
            "Perf (Jan)", "Perf (Apr)",
            "Fwd P/E", "Sales G.", "EPS G.", "D/E", "PEG", "Div Yield",
            "Rating", "ListsHit", "MarketCap"
        ]].copy()

        # Human-friendly formatting
        table_df["MarketCap"] = table_df["MarketCap"].apply(fmt_mcap)
        table_df["Perf (Jan)"] = table_df["Perf (Jan)"].apply(fmt_pct)
        table_df["Perf (Apr)"] = table_df["Perf (Apr)"].apply(fmt_pct)
        table_df["Fwd P/E"] = table_df["Fwd P/E"].apply(lambda x: fmt_num(x, 2))
        table_df["Sales G."] = table_df["Sales G."].apply(lambda x: fmt_pct(x))
        table_df["EPS G."] = table_df["EPS G."].apply(lambda x: fmt_pct(x))
        table_df["D/E"] = table_df["D/E"].apply(lambda x: fmt_num(x, 2))
        table_df["PEG"] = table_df["PEG"].apply(lambda x: fmt_num(x, 2))
        table_df["Div Yield"] = table_df["Div Yield"].apply(lambda x: fmt_pct(x))

        # For styling we need numeric columns too; provide a parallel numeric frame for styler logic
        sty_base = display[[
            "Ticker", "Country", "Industry",
            "Perf (Jan)", "Perf (Apr)",
            "Rating", "ListsHit"
        ]].copy()
        sty_base["Perf (Jan)"] = display["Perf_Jan"]
        sty_base["Perf (Apr)"] = display["Perf_Apr"]
        sty_base = sty_base.rename(columns={"Perf (Jan)":"Perf (Jan)", "Perf (Apr)":"Perf (Apr)"})
        sty_base["Perf (Jan)"] = sty_base["Perf (Jan)"]
        sty_base["Perf (Apr)"] = sty_base["Perf (Apr)"]

        # Render formatted table (Streamlit doesn't style strings well; show formatted table plain + chips below)
        st.dataframe(table_df, use_container_width=True, height=520)

        st.caption("Tip: `ListsHit` shows which of L1–L7 each stock satisfied.")

    # --- Drilldown ---
    st.markdown("### 🔎 Drilldown")
    left, right = st.columns([1, 2])

    with left:
        if display.empty:
            st.warning("No rows after filters.")
            return

        pick = st.selectbox("Select a ticker", display["Ticker"].tolist(), index=0)
        row = final[final["Ticker"] == pick].iloc[0]

        st.markdown(f"""
        <div class="card scanline">
            <div style="opacity:0.8;font-size:12px;">{row["Name"]}</div>
            <div style="font-size:22px;font-weight:700;">{row["Ticker"]} <span class="chip">{row["Rating"]}</span></div>
            <div style="margin-top:8px;">
                <span class="chip">Country: {row["Country"]}</span>
                <span class="chip">Industry: {row["Industry"]}</span>
                <span class="chip">MCAP: {fmt_mcap(row["MarketCap"])}</span>
            </div>
            <div style="margin-top:10px;">
                <div style="opacity:0.8;font-size:12px;">Lists Hit</div>
                <div style="font-weight:700;">{row["ListsHit"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Key Metrics")
        st.write(
            pd.DataFrame([{
                "Perf (Jan)": fmt_pct(row["Perf_Jan"]),
                "Perf (Apr)": fmt_pct(row["Perf_Apr"]),
                "Fwd P/E": fmt_num(row["ForwardPE"], 2),
                "Sales G.": fmt_pct((row["RevGrowth_1Y"] * 100) if pd.notna(row["RevGrowth_1Y"]) else np.nan),
                "EPS G.": fmt_pct((row["EPSGrowth_1Y"] * 100) if pd.notna(row["EPSGrowth_1Y"]) else np.nan),
                "D/E": fmt_num(row["DebtToEquity"], 2),
                "PEG": fmt_num(row["PEGRatio"], 2),
                "Div Yield": fmt_pct((row["DivYield"] * 100) if pd.notna(row["DivYield"]) else np.nan),
                "P/FCF": fmt_num(row["P_FCF"], 2),
                "Net Margin": fmt_pct((row["NetMargin"] * 100) if pd.notna(row["NetMargin"]) else np.nan),
            }])
        )

    with right:
        st.markdown("#### Candlestick (1Y)")
        try:
            hist = yf.Ticker(pick).history(period="1y")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name=pick
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color,
                margin=dict(l=20, r=20, t=30, b=20),
                height=520
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart load failed for {pick}: {e}")

    # --- Export ---
    st.markdown("### 📦 Export")
    export_cols = [
        "Name","Exchange","Ticker","Country","Industry","MarketCap",
        "Perf_Jan","Perf_Apr",
        "RevGrowth_1Y","EPSGrowth_1Y","ForwardPE","PSFwd","PBRatio","P_FCF","PEGRatio","NetMargin","DivYield","DebtToEquity",
        "Rating","ListsHit"
    ]
    export_df = final[export_cols].copy()
    export_df["Perf_Jan"] = export_df["Perf_Jan"].round(4)
    export_df["Perf_Apr"] = export_df["Perf_Apr"].round(4)

    excel_bytes = to_excel_bytes(export_df)
    st.download_button(
        "📥 DOWNLOAD EXCEL REPORT",
        data=excel_bytes,
        file_name=f"Global_Equity_Alpha_Screener_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

if __name__ == "__main__":
    main()
