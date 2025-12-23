import streamlit as st
import pandas as pd

# ============================================================
# 1) GLOBAL TICKER CATALOG (curated, editable, expandable)
#    - Covers: North America, Europe/UK/Scandinavia, Switzerland,
#      South Africa, Japan, Hong Kong, Singapore, Australasia
#    - Includes: sector + industry + region + exchange + country
# ============================================================

def build_ticker_db() -> pd.DataFrame:
    rows = [
        # -------------------- NORTH AMERICA (US/CA) --------------------
        # Tech / Semis
        ("AAPL","Apple","North America","NASDAQ","USA","Technology","Consumer Electronics"),
        ("MSFT","Microsoft","North America","NASDAQ","USA","Technology","Software"),
        ("NVDA","NVIDIA","North America","NASDAQ","USA","Technology","Semiconductors"),
        ("AVGO","Broadcom","North America","NASDAQ","USA","Technology","Semiconductors"),
        ("AMD","AMD","North America","NASDAQ","USA","Technology","Semiconductors"),
        ("TSM","TSMC ADR","North America","NYSE","USA","Technology","Semiconductors"),  # ADR (HQ not NA) - tag below
        ("ORCL","Oracle","North America","NYSE","USA","Technology","Software"),
        ("ADBE","Adobe","North America","NASDAQ","USA","Technology","Software"),
        ("CRM","Salesforce","North America","NYSE","USA","Technology","Software"),
        ("INTU","Intuit","North America","NASDAQ","USA","Technology","Software"),
        ("NOW","ServiceNow","North America","NYSE","USA","Technology","Software"),
        ("PANW","Palo Alto Networks","North America","NASDAQ","USA","Technology","Cybersecurity"),
        ("CRWD","CrowdStrike","North America","NASDAQ","USA","Technology","Cybersecurity"),
        ("SNOW","Snowflake","North America","NYSE","USA","Technology","Data Infrastructure"),

        # Comm Services / Internet
        ("GOOGL","Alphabet","North America","NASDAQ","USA","Communication Services","Internet"),
        ("META","Meta Platforms","North America","NASDAQ","USA","Communication Services","Social Media"),
        ("NFLX","Netflix","North America","NASDAQ","USA","Communication Services","Streaming"),
        ("DIS","Disney","North America","NYSE","USA","Communication Services","Media"),

        # Consumer
        ("AMZN","Amazon","North America","NASDAQ","USA","Consumer Discretionary","E-commerce"),
        ("TSLA","Tesla","North America","NASDAQ","USA","Consumer Discretionary","Automobiles"),
        ("NKE","Nike","North America","NYSE","USA","Consumer Discretionary","Apparel"),
        ("HD","Home Depot","North America","NYSE","USA","Consumer Discretionary","Retail"),
        ("MCD","McDonald's","North America","NYSE","USA","Consumer Discretionary","Restaurants"),
        ("PG","Procter & Gamble","North America","NYSE","USA","Consumer Staples","Household Products"),
        ("KO","Coca-Cola","North America","NYSE","USA","Consumer Staples","Beverages"),
        ("PEP","PepsiCo","North America","NASDAQ","USA","Consumer Staples","Beverages"),
        ("COST","Costco","North America","NASDAQ","USA","Consumer Staples","Retail"),

        # Financials
        ("BRK-B","Berkshire Hathaway","North America","NYSE","USA","Financials","Conglomerate"),
        ("JPM","JPMorgan Chase","North America","NYSE","USA","Financials","Banking"),
        ("BAC","Bank of America","North America","NYSE","USA","Financials","Banking"),
        ("GS","Goldman Sachs","North America","NYSE","USA","Financials","Investment Banking"),
        ("MS","Morgan Stanley","North America","NYSE","USA","Financials","Investment Banking"),
        ("V","Visa","North America","NYSE","USA","Financials","Payments"),
        ("MA","Mastercard","North America","NYSE","USA","Financials","Payments"),

        # Healthcare
        ("LLY","Eli Lilly","North America","NYSE","USA","Healthcare","Pharmaceuticals"),
        ("JNJ","Johnson & Johnson","North America","NYSE","USA","Healthcare","Pharma & Devices"),
        ("PFE","Pfizer","North America","NYSE","USA","Healthcare","Pharmaceuticals"),
        ("MRK","Merck","North America","NYSE","USA","Healthcare","Pharmaceuticals"),
        ("UNH","UnitedHealth","North America","NYSE","USA","Healthcare","Managed Care"),
        ("ABT","Abbott","North America","NYSE","USA","Healthcare","Medical Devices"),
        ("TMO","Thermo Fisher","North America","NYSE","USA","Healthcare","Life Sciences Tools"),

        # Industrials / Energy / Materials
        ("CAT","Caterpillar","North America","NYSE","USA","Industrials","Machinery"),
        ("GE","GE Aerospace","North America","NYSE","USA","Industrials","Aerospace"),
        ("BA","Boeing","North America","NYSE","USA","Industrials","Aerospace"),
        ("XOM","Exxon Mobil","North America","NYSE","USA","Energy","Oil & Gas"),
        ("CVX","Chevron","North America","NYSE","USA","Energy","Oil & Gas"),
        ("SLB","Schlumberger","North America","NYSE","USA","Energy","Oilfield Services"),

        # Canada (TSX)
        ("SHOP.TO","Shopify","North America","TSX","Canada","Technology","E-commerce Software"),
        ("RY.TO","Royal Bank of Canada","North America","TSX","Canada","Financials","Banking"),
        ("TD.TO","Toronto-Dominion","North America","TSX","Canada","Financials","Banking"),
        ("ENB.TO","Enbridge","North America","TSX","Canada","Energy","Midstream"),
        ("CNQ.TO","Canadian Natural Resources","North America","TSX","Canada","Energy","Oil & Gas"),
        ("BNS.TO","Scotiabank","North America","TSX","Canada","Financials","Banking"),

        # -------------------- EUROPE / UK / SCANDINAVIA --------------------
        # Netherlands / France / Germany / etc.
        ("ASML.AS","ASML","Europe","Euronext Amsterdam","Netherlands","Technology","Semiconductors"),
        ("AIR.PA","Airbus","Europe","Euronext Paris","France","Industrials","Aerospace"),
        ("MC.PA","LVMH","Europe","Euronext Paris","France","Consumer Discretionary","Luxury"),
        ("OR.PA","L'Oréal","Europe","Euronext Paris","France","Consumer Staples","Personal Care"),
        ("SAN.PA","Sanofi","Europe","Euronext Paris","France","Healthcare","Pharmaceuticals"),
        ("SAP.DE","SAP","Europe","XETRA","Germany","Technology","Software"),
        ("SIE.DE","Siemens","Europe","XETRA","Germany","Industrials","Automation"),
        ("DTE.DE","Deutsche Telekom","Europe","XETRA","Germany","Communication Services","Telecom"),
        ("RHM.DE","Rheinmetall","Europe","XETRA","Germany","Industrials","Defense"),
        ("NVO.CO","Novo Nordisk","Europe","Copenhagen","Denmark","Healthcare","Pharmaceuticals"),
        ("DSV.CO","DSV","Europe","Copenhagen","Denmark","Industrials","Logistics"),
        ("NESN.SW","Nestlé","Switzerland","SIX","Switzerland","Consumer Staples","Food"),
        ("ROG.SW","Roche","Switzerland","SIX","Switzerland","Healthcare","Pharmaceuticals"),
        ("NOVN.SW","Novartis","Switzerland","SIX","Switzerland","Healthcare","Pharmaceuticals"),
        ("UBSG.SW","UBS","Switzerland","SIX","Switzerland","Financials","Banking"),
        ("CFR.SW","Richemont","Switzerland","SIX","Switzerland","Consumer Discretionary","Luxury"),
        ("ABBN.SW","ABB","Switzerland","SIX","Switzerland","Industrials","Electrification"),

        # UK (LSE)
        ("AZN.L","AstraZeneca","Europe","LSE","UK","Healthcare","Pharmaceuticals"),
        ("SHEL.L","Shell","Europe","LSE","UK","Energy","Oil & Gas"),
        ("ULVR.L","Unilever","Europe","LSE","UK","Consumer Staples","Household Products"),
        ("HSBA.L","HSBC","Europe","LSE","UK","Financials","Banking"),
        ("RIO.L","Rio Tinto","Europe","LSE","UK","Materials","Mining"),
        ("BP.L","BP","Europe","LSE","UK","Energy","Oil & Gas"),

        # Scandinavia extras
        ("EQNR.OL","Equinor","Europe","Oslo","Norway","Energy","Oil & Gas"),
        ("VOLV-B.ST","Volvo","Europe","Stockholm","Sweden","Industrials","Commercial Vehicles"),
        ("ERIC-B.ST","Ericsson","Europe","Stockholm","Sweden","Technology","Telecom Equipment"),

        # -------------------- SOUTH AFRICA (JSE) --------------------
        ("NPN.JO","Naspers","South Africa","JSE","South Africa","Consumer Discretionary","Internet Holdings"),
        ("FSR.JO","FirstRand","South Africa","JSE","South Africa","Financials","Banking"),
        ("SBK.JO","Standard Bank","South Africa","JSE","South Africa","Financials","Banking"),
        ("SOL.JO","Sasol","South Africa","JSE","South Africa","Energy","Chemicals & Energy"),
        ("ANG.JO","Anglo American","South Africa","JSE","South Africa","Materials","Mining"),

        # -------------------- JAPAN (TSE) --------------------
        ("7203.T","Toyota","Japan","TSE","Japan","Consumer Discretionary","Automobiles"),
        ("6758.T","Sony","Japan","TSE","Japan","Communication Services","Entertainment & Electronics"),
        ("9984.T","SoftBank","Japan","TSE","Japan","Communication Services","Telecom & Holdings"),
        ("8306.T","Mitsubishi UFJ","Japan","TSE","Japan","Financials","Banking"),
        ("8035.T","Tokyo Electron","Japan","TSE","Japan","Technology","Semiconductor Equipment"),
        ("9432.T","NTT","Japan","TSE","Japan","Communication Services","Telecom"),
        ("4502.T","Takeda","Japan","TSE","Japan","Healthcare","Pharmaceuticals"),

        # -------------------- HONG KONG (HKEX) --------------------
        ("0700.HK","Tencent","Hong Kong","HKEX","Hong Kong","Communication Services","Internet"),
        ("9988.HK","Alibaba","Hong Kong","HKEX","Hong Kong","Consumer Discretionary","E-commerce"),
        ("3690.HK","Meituan","Hong Kong","HKEX","Hong Kong","Consumer Discretionary","Local Services"),
        ("0005.HK","HSBC (HK)","Hong Kong","HKEX","Hong Kong","Financials","Banking"),
        ("1299.HK","AIA","Hong Kong","HKEX","Hong Kong","Financials","Insurance"),

        # -------------------- SINGAPORE (SGX) --------------------
        ("D05.SI","DBS","Singapore","SGX","Singapore","Financials","Banking"),
        ("O39.SI","OCBC","Singapore","SGX","Singapore","Financials","Banking"),
        ("U11.SI","UOB","Singapore","SGX","Singapore","Financials","Banking"),
        ("Z74.SI","Singtel","Singapore","SGX","Singapore","Communication Services","Telecom"),
        ("C6L.SI","Singapore Airlines","Singapore","SGX","Singapore","Industrials","Airlines"),

        # -------------------- AUSTRALASIA (ASX / NZX) --------------------
        ("BHP.AX","BHP","Australasia","ASX","Australia","Materials","Mining"),
        ("CBA.AX","Commonwealth Bank","Australasia","ASX","Australia","Financials","Banking"),
        ("CSL.AX","CSL","Australasia","ASX","Australia","Healthcare","Biotech"),
        ("WBC.AX","Westpac","Australasia","ASX","Australia","Financials","Banking"),
        ("NAB.AX","National Australia Bank","Australasia","ASX","Australia","Financials","Banking"),
        ("RIO.AX","Rio Tinto (AU)","Australasia","ASX","Australia","Materials","Mining"),
    ]

    df = pd.DataFrame(rows, columns=["ticker","name","region","exchange","country","sector","industry"])

    # Add a strict HQ eligibility flag you can enforce in your screener:
    # (Example: ADRs can be flagged if you want HQ rules.)
    df["hq_eligible"] = True
    df.loc[df["ticker"].isin(["TSM"]), "hq_eligible"] = False  # ADR example (HQ not in allowed geo)
    return df


# ============================================================
# 2) USEFUL PRESET UNIVERSES (many options)
# ============================================================
def build_presets(df: pd.DataFrame) -> dict:
    # helpers
    def tickers_where(**kwargs):
        d = df.copy()
        for k,v in kwargs.items():
            d = d[d[k].isin(v if isinstance(v, list) else [v])]
        return d["ticker"].tolist()

    presets = {
        # Broad region baskets
        "North America — Core": tickers_where(region="North America"),
        "Europe — Core": tickers_where(region="Europe"),
        "Switzerland — Core": tickers_where(region="Switzerland"),
        "Japan — Core": tickers_where(region="Japan"),
        "Hong Kong — Core": tickers_where(region="Hong Kong"),
        "Singapore — Core": tickers_where(region="Singapore"),
        "Australasia — Core": tickers_where(region="Australasia"),
        "South Africa — Core": tickers_where(region="South Africa"),

        # Sector baskets (global)
        "Global — AI & Semiconductors": tickers_where(industry=["Semiconductors","Semiconductor Equipment"]),
        "Global — Software Leaders": tickers_where(industry=["Software","Cybersecurity","Data Infrastructure"]),
        "Global — Healthcare Mega": tickers_where(sector="Healthcare"),
        "Global — Banks & Payments": tickers_where(industry=["Banking","Payments","Insurance"]),
        "Global — Energy Majors": tickers_where(sector="Energy"),
        "Global — Materials & Mining": tickers_where(sector="Materials"),
        "Global — Consumer Staples Defensive": tickers_where(sector="Consumer Staples"),
        "Global — Luxury": tickers_where(industry="Luxury"),

        # Thematic mixes (practical “watchlists”)
        "Defense / Aerospace": tickers_where(industry=["Defense","Aerospace"]),
        "Big Tech Basket": ["AAPL","MSFT","NVDA","GOOGL","META","AMZN"],
        "Dividend-ish Blue Chips": ["KO","PEP","PG","ULVR.L","NESN.SW","ENB.TO"],

        "Custom (start empty)": [],
    }
    # remove empties safely
    return {k: v for k, v in presets.items() if isinstance(v, list)}

# ============================================================
# 3) POWER UNIVERSE BUILDER W/ MANY OPTIONS + CSV EXTENSION
# ============================================================
def ticker_label(row: pd.Series) -> str:
    return f"{row['ticker']} — {row['name']} ({row['exchange']}, {row['country']}) • {row['sector']} / {row['industry']}"

def build_label_maps(df: pd.DataFrame):
    labels = df.apply(ticker_label, axis=1).tolist()
    label_to_ticker = {ticker_label(r): r["ticker"] for _, r in df.iterrows()}
    ticker_to_label = {v: k for k, v in label_to_ticker.items()}
    return labels, label_to_ticker, ticker_to_label

def universe_builder_sidebar() -> tuple[list[str], pd.DataFrame, dict]:
    TICKER_DB = build_ticker_db()

    # Optional: let user upload a CSV to extend/override catalog
    # CSV columns expected: ticker,name,region,exchange,country,sector,industry,hq_eligible(optional)
    with st.sidebar:
        st.subheader("Universe Catalog")
        up = st.file_uploader("Upload ticker catalog CSV (optional)", type=["csv"])
        if up is not None:
            add = pd.read_csv(up)
            # minimal validation
            required = {"ticker","name","region","exchange","country","sector","industry"}
            if required.issubset(set(add.columns)):
                if "hq_eligible" not in add.columns:
                    add["hq_eligible"] = True
                # merge: uploaded rows override same ticker
                TICKER_DB = pd.concat([TICKER_DB, add], ignore_index=True)
                TICKER_DB = TICKER_DB.drop_duplicates(subset=["ticker"], keep="last")
            else:
                st.warning("CSV missing required columns: ticker,name,region,exchange,country,sector,industry")

    PRESETS = build_presets(TICKER_DB)

    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = set()
    if "active_preset" not in st.session_state:
        st.session_state.active_preset = "Custom (start empty)"

    with st.sidebar:
        st.subheader("Universe Builder")

        # --- Preset apply/clear ---
        preset = st.selectbox("Preset Universe", list(PRESETS.keys()),
                              index=list(PRESETS.keys()).index(st.session_state.active_preset)
                              if st.session_state.active_preset in PRESETS else 0)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Apply Preset"):
                st.session_state.selected_tickers = set(PRESETS[preset])
                st.session_state.active_preset = preset
        with c2:
            if st.button("Clear"):
                st.session_state.selected_tickers = set()
                st.session_state.active_preset = "Custom (start empty)"

        st.divider()

        # --- hard eligibility toggle ---
        enforce_hq = st.checkbox("Enforce HQ-eligible only", value=True)
        df = TICKER_DB[TICKER_DB["hq_eligible"]] if enforce_hq else TICKER_DB.copy()

        # --- filters ---
        regions = ["All"] + sorted(df["region"].dropna().unique().tolist())
        region = st.selectbox("Region", regions, index=0)
        if region != "All":
            df = df[df["region"] == region]

        exchanges = ["All"] + sorted(df["exchange"].dropna().unique().tolist())
        exchange = st.selectbox("Exchange", exchanges, index=0)
        if exchange != "All":
            df = df[df["exchange"] == exchange]

        sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
        sector = st.selectbox("Sector", sectors, index=0)
        if sector != "All":
            df = df[df["sector"] == sector]

        industries = ["All"] + sorted(df["industry"].dropna().unique().tolist())
        industry = st.selectbox("Industry", industries, index=0)
        if industry != "All":
            df = df[df["industry"] == industry]

        # --- search ---
        q = st.text_input("Search", "")
        if q.strip():
            s = q.strip().lower()
            df = df[
                df["ticker"].str.lower().str.contains(s)
                | df["name"].str.lower().str.contains(s)
                | df["country"].str.lower().str.contains(s)
                | df["industry"].str.lower().str.contains(s)
                | df["sector"].str.lower().str.contains(s)
            ]

        labels, label_to_ticker, ticker_to_label = build_label_maps(df)

        # --- add/remove all in current filter ---
        a, b = st.columns(2)
        with a:
            if st.button("Add all (filtered)"):
                st.session_state.selected_tickers |= set(df["ticker"].tolist())
        with b:
            if st.button("Remove all (filtered)"):
                st.session_state.selected_tickers -= set(df["ticker"].tolist())

        # --- multiselect with persistence for current view ---
        default_labels = [ticker_to_label[t] for t in st.session_state.selected_tickers if t in ticker_to_label]
        selected_labels = st.multiselect("Tickers (filtered list)", options=labels, default=default_labels)

        selected_in_filter = {label_to_ticker[lbl] for lbl in selected_labels}
        tickers_in_filter = set(df["ticker"].tolist())
        st.session_state.selected_tickers = (st.session_state.selected_tickers - tickers_in_filter) | selected_in_filter

        st.caption(f"Selected: {len(st.session_state.selected_tickers)} tickers")

    selected = sorted(st.session_state.selected_tickers)
    return selected, TICKER_DB, PRESETS


# =======================
# USAGE IN YOUR APP:
# =======================
# selected_tickers, ticker_db, presets = universe_builder_sidebar()
# st.write("Selected tickers:", selected_tickers)
