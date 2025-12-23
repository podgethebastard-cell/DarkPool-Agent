# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, Callable, List, Tuple, Optional
from datetime import datetime, timezone, timedelta

# ----------------------------
# CONFIG / UX THEME
# ----------------------------
def inject_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
      :root { --bg:#0e1117; --txt:#e0e0e0; --acc:#00d4ff; --card:rgba(255,255,255,0.03); }
      .stApp { background-color: var(--bg); color: var(--txt); font-family: 'Roboto Mono', monospace; }
      h1,h2,h3 { letter-spacing: 1px; text-transform: uppercase; }
      .card { background: var(--card); border: 1px solid rgba(0,212,255,0.18); padding: 14px; border-radius: 12px; }
      .small { opacity: 0.8; font-size: 12px; }
      /* Custom dataframe styling */
      .stDataFrame { border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# ROBUST MATH UTILITIES
# ----------------------------
def winsorize(s: pd.Series, p: float = 0.02) -> pd.Series:
    if s.dropna().empty:
        return s
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)

def robust_z(s: pd.Series, eps: float = 1e-9) -> pd.Series:
    """Median/MAD robust z-score."""
    x = s.astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if np.isnan(mad) or mad < eps:
        return (x - med) * 0.0
    return (x - med) / (1.4826 * mad + eps)

def softmax(x: np.ndarray, t: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(t, 1e-9)
    x = x - np.nanmax(x)
    e = np.exp(x)
    return e / (np.nansum(e) + 1e-12)

def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Spearman is more robust to outliers than Pearson for factor signals.
    return df.corr(method="spearman", min_periods=max(5, int(df.shape[0] * 0.25)))

def uniqueness_penalty(corr: pd.DataFrame, floor: float = 0.35) -> pd.Series:
    """
    Penalize crowded signals:
    - For each signal, compute average absolute correlation to others.
    - Convert to penalty in [floor, 1].
    """
    if corr.empty:
        return pd.Series(dtype=float)
    avg_abs = corr.abs().replace(1.0, np.nan).mean(axis=1).fillna(0.0)
    pen = 1.0 - avg_abs.clip(0, 1)
    return pen.clip(lower=floor, upper=1.0)

def confidence_from_coverage(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Confidence = fraction of required columns present (not null) per row."""
    if not cols:
        return pd.Series(1.0, index=df.index)
    present = df[cols].notna().mean(axis=1)
    return present.clip(0.0, 1.0)

# ----------------------------
# DATA ADAPTER (YFINANCE LIVE)
# ----------------------------
@dataclass
class DataBundle:
    df: pd.DataFrame
    # sources could be expanded for audit logs

class DataAdapter:
    """
    Production-ready Adapter using yfinance.
    Fetches:
    1. Fundamentals via .info
    2. Technicals via .history (batch download)
    """
    def load(self, tickers: List[str]) -> DataBundle:
        if not tickers:
            return DataBundle(pd.DataFrame())

        # 1. Batch fetch price history for Momentum/Vol
        # Fetch 1 year + buffer
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
        
        try:
            hist_data = yf.download(tickers, start=start_date, group_by='ticker', progress=False, threads=True)
        except Exception as e:
            st.error(f"Data fetch error: {e}")
            return DataBundle(pd.DataFrame())

        # 2. Iterate tickers to build the Master DataFrame
        rows = []
        
        # We need a Tickers object to get info efficiently (though serial is safer for 'info')
        yf_tickers = yf.Tickers(" ".join(tickers))

        for t in tickers:
            row = {"ticker": t}
            
            # --- A. FUNDAMENTALS (Info) ---
            try:
                # Accessing the specific ticker object
                info = yf_tickers.tickers[t].info
                
                # Descriptors
                row["name"] = info.get("shortName", t)
                row["industry"] = info.get("industry", "N/A")
                row["sector"] = info.get("sector", "N/A")
                row["country"] = info.get("country", "N/A")
                row["mcap_usd"] = info.get("marketCap", np.nan)
                
                # Value
                row["fwd_pe"] = info.get("forwardPE", np.nan)
                row["fwd_ps"] = info.get("priceToSalesTrailing12Months", np.nan) # approx
                row["peg"] = info.get("pegRatio", np.nan)
                row["price_to_book"] = info.get("priceToBook", np.nan)
                
                # Quality
                row["profit_margin"] = info.get("profitMargins", np.nan)
                row["gross_margin"] = info.get("grossMargins", np.nan)
                row["op_margin"] = info.get("operatingMargins", np.nan)
                row["roic"] = info.get("returnOnEquity", np.nan) # Proxy if ROIC missing
                row["debt_to_equity"] = info.get("debtToEquity", np.nan)
                
                # Income
                row["div_yield"] = info.get("dividendYield", 0.0)
                # Growth (approx)
                row["rev_growth"] = info.get("revenueGrowth", np.nan)
                row["eps_growth"] = info.get("earningsGrowth", np.nan)
                
                # Beta
                row["beta"] = info.get("beta", np.nan)

            except Exception:
                # If info fails, we leave cols as NaN
                pass

            # --- B. TECHNICALS (History) ---
            try:
                # Handle single ticker vs multi-ticker structure of yf.download
                if len(tickers) == 1:
                    bars = hist_data
                else:
                    bars = hist_data[t]
                
                if not bars.empty:
                    # Clean Close
                    close = bars["Close"]
                    
                    # Momentum (Returns)
                    # We grab the last available price
                    current_price = close.iloc[-1]
                    row["price_now"] = current_price
                    
                    # Periodic Returns
                    # approximate trading days: 1m=21, 3m=63, 6m=126, 12m=252
                    row["ret_1m"] = close.pct_change(21).iloc[-1]
                    row["ret_3m"] = close.pct_change(63).iloc[-1]
                    row["ret_6m"] = close.pct_change(126).iloc[-1]
                    row["ret_12m"] = close.pct_change(252).iloc[-1]
                    
                    # Volatility (90d annualized)
                    # std dev of daily returns * sqrt(252)
                    daily_rets = close.pct_change()
                    row["vol_90d"] = daily_rets.tail(63).std() * np.sqrt(252)
                    
                    # Max Drawdown (1y)
                    # Rolling max
                    roll_max = close.rolling(252, min_periods=100).max()
                    dd = (close / roll_max) - 1.0
                    row["maxdd_1y"] = dd.iloc[-1]
                    
            except Exception:
                pass

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("ticker")
        
        return DataBundle(df=df)

# ----------------------------
# STRATEGY PLUGINS
# ----------------------------
@dataclass
class StrategySpec:
    key: str
    label: str
    enabled: bool = True
    weight: float = 1.0
    params: Dict = None

StrategyFn = Callable[[pd.DataFrame, Dict], Tuple[pd.Series, pd.Series]]

def strat_value(df: pd.DataFrame, p: Dict) -> Tuple[pd.Series, pd.Series]:
    w = p.get("winsor", 0.02)
    inv = p.get("invert", True)
    metrics = ["fwd_pe", "peg", "price_to_book"] # Updated to match YF keys
    sigs = []
    for m in metrics:
        if m in df.columns:
            s = winsorize(df[m], w)
            z = robust_z(s)
            sigs.append(-z if inv else z)
        else:
            sigs.append(pd.Series(0, index=df.index))
            
    signal = pd.concat(sigs, axis=1).mean(axis=1)
    conf = confidence_from_coverage(df, metrics)
    return signal, conf

def strat_quality(df: pd.DataFrame, p: Dict) -> Tuple[pd.Series, pd.Series]:
    w = p.get("winsor", 0.02)
    metrics = ["roic", "gross_margin", "op_margin", "profit_margin"]
    sigs = []
    for m in metrics:
        if m in df.columns:
            s = winsorize(df[m], w)
            sigs.append(robust_z(s))
        else:
            sigs.append(pd.Series(0, index=df.index))

    signal = pd.concat(sigs, axis=1).mean(axis=1)
    conf = confidence_from_coverage(df, metrics)
    return signal, conf

def strat_momentum(df: pd.DataFrame, p: Dict) -> Tuple[pd.Series, pd.Series]:
    w = p.get("winsor", 0.02)
    crash_penalty = p.get("crash_penalty", 0.5)
    metrics = ["ret_1m", "ret_3m", "ret_6m", "ret_12m"]
    
    sigs = []
    for m in metrics:
        if m in df.columns:
            sigs.append(robust_z(winsorize(df[m], w)))
    
    if not sigs:
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index)

    sig = pd.concat(sigs, axis=1).mean(axis=1)
    
    # Drawdown penalty
    if "maxdd_1y" in df.columns:
        dd = robust_z(winsorize(df["maxdd_1y"], w))
        signal = sig - crash_penalty * (dd * -1) # dd is negative, so * -1 to make deep dd a high positive number to subtract? 
        # Wait, robust_z of DD:
        # DDs are -0.10, -0.20. Median might be -0.15.
        # If I have -0.50 (crash), Z will be negative (below median).
        # We want to penalize crash. So we want to SUBTRACT "badness".
        # Actually simplest is: Deep DD = Low Raw Value.
        # Robust Z of Deep DD = Negative Z.
        # Signal = Momentum (High Z) + 1.0 * DD (Negative Z).
        # So we simply Add DD Z-score?
        # Let's stick to the Architect's specific logic: 
        # "Deep DD should hurt score".
        # If DD Z-score is -3 (bad), adding it reduces score.
        signal = sig + (crash_penalty * dd) 
    else:
        signal = sig

    conf = confidence_from_coverage(df, metrics + ["maxdd_1y"])
    return signal, conf

def strat_lowvol(df: pd.DataFrame, p: Dict) -> Tuple[pd.Series, pd.Series]:
    w = p.get("winsor", 0.02)
    metrics = ["vol_90d", "beta", "maxdd_1y"]
    
    # We want LOW vol, LOW beta, LOW drawdown (closest to 0)
    # Vol: High = High Z. Invert.
    # Beta: High = High Z. Invert.
    # DD: Deep (-0.5) = Low Z. Shallow (-0.01) = High Z.
    # So actually High Z for DD is GOOD (Stability).
    
    components = []
    if "vol_90d" in df.columns:
        components.append(-1 * robust_z(winsorize(df["vol_90d"], w)))
    if "beta" in df.columns:
        components.append(-1 * robust_z(winsorize(df["beta"], w)))
    if "maxdd_1y" in df.columns:
        # maxdd is negative. Closer to 0 is better (higher).
        # Robust Z of maxdd: -0.01 is > -0.50. So Higher Z is better.
        components.append(robust_z(winsorize(df["maxdd_1y"], w)))
        
    if not components:
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index)

    signal = pd.concat(components, axis=1).mean(axis=1)
    conf = confidence_from_coverage(df, metrics)
    return signal, conf

def strat_income(df: pd.DataFrame, p: Dict) -> Tuple[pd.Series, pd.Series]:
    w = p.get("winsor", 0.02)
    metrics = ["div_yield"] # YF info usually lacks growth forecasts reliable enough
    
    if "div_yield" not in df.columns:
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index)

    z_y = robust_z(winsorize(df["div_yield"], w))
    signal = z_y 
    conf = confidence_from_coverage(df, metrics)
    return signal, conf

STRATEGIES: Dict[str, Tuple[str, StrategyFn, Dict]] = {
    "value": ("Value Composite", strat_value, {"winsor": 0.02, "invert": True}),
    "quality": ("Quality Composite", strat_quality, {"winsor": 0.02}),
    "momentum": ("Momentum + Crash Penalty", strat_momentum, {"winsor": 0.02, "crash_penalty": 1.0}),
    "lowvol": ("Low Vol / Defensive", strat_lowvol, {"winsor": 0.02}),
    "income": ("Carry / Income", strat_income, {"winsor": 0.02}),
}

# ----------------------------
# ALPHA CONSTELLATION ENSEMBLE
# ----------------------------
def run_ensemble(df: pd.DataFrame, specs: List[StrategySpec], crowd_floor: float, temp: float) -> pd.DataFrame:
    signals = {}
    confs = {}

    for s in specs:
        if not s.enabled:
            continue
        label, fn, defaults = STRATEGIES[s.key]
        params = dict(defaults)
        if s.params:
            params.update(s.params)
        sig, conf = fn(df, params)
        signals[s.key] = sig
        confs[s.key] = conf

    sig_df = pd.DataFrame(signals)
    conf_df = pd.DataFrame(confs)

    if sig_df.empty:
        return pd.DataFrame(index=df.index)

    # Redundancy/crowding control
    corr = corr_matrix(sig_df)
    uniq = uniqueness_penalty(corr, floor=crowd_floor)

    # Strategy weights
    w_raw = np.array([sp.weight for sp in specs if sp.enabled], dtype=float)
    keys = [sp.key for sp in specs if sp.enabled]
    w = softmax(w_raw, t=temp)
    w_s = pd.Series(w, index=keys)

    # Weighted sum
    score = pd.Series(0.0, index=df.index)
    for k in keys:
        s_k = sig_df[k].fillna(0)
        c_k = conf_df[k].fillna(0)
        u_k = uniq.get(k, 1.0)
        score = score + w_s[k] * s_k * c_k * u_k

    out = df.copy()
    out["alpha_score"] = score
    out["alpha_rank"] = out["alpha_score"].rank(ascending=False, method="min")
    
    for k in keys:
        out[f"sig_{k}"] = sig_df[k]
        out[f"conf_{k}"] = conf_df[k]
        
    return out.sort_values("alpha_score", ascending=False)

# ----------------------------
# UI
# ----------------------------
def main():
    st.set_page_config(page_title="THE ARCHITECT — Alpha Constellation", layout="wide")
    inject_css()

    st.title("🏛️ THE ARCHITECT — ALPHA CONSTELLATION")
    st.caption("Live Institutional Multi-Factor Screener [YFinance Powered]")

    with st.sidebar:
        st.subheader("Universe")
        default_tickers = "AAPL, MSFT, NVDA, GOOGL, META, TSLA, AMZN, JPM, V, WMT, PG, KO, JNJ, UNH, XOM"
        tickers_raw = st.text_area("Tickers (comma-separated)", default_tickers, height=120)
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

        st.divider()
        st.subheader("Ensemble Controls")
        crowd_floor = st.slider("Uniqueness penalty floor", 0.10, 0.90, 0.35, 0.05)
        temp = st.slider("Weight softmax temperature", 0.2, 3.0, 1.0, 0.1)

        st.divider()
        st.subheader("Strategies")
        specs: List[StrategySpec] = []
        for key, (label, _, defaults) in STRATEGIES.items():
            with st.expander(label, expanded=False):
                enabled = st.checkbox("Enabled", value=True, key=f"en_{key}")
                weight = st.slider("Weight", 0.0, 5.0, 1.0, 0.1, key=f"w_{key}")
                params = {}
                if "winsor" in defaults:
                    params["winsor"] = st.slider("Winsorization (%)", 0.0, 0.10, float(defaults["winsor"]), 0.01, key=f"win_{key}")
                if key == "momentum":
                    params["crash_penalty"] = st.slider("Crash penalty", 0.0, 2.0, float(defaults["crash_penalty"]), 0.1, key=f"cp_{key}")
                
                specs.append(StrategySpec(key=key, label=label, enabled=enabled, weight=weight, params=params))

        st.divider()
        run_btn = st.button("EXECUTE ALPHA SEQUENCE")

    if not run_btn:
        st.info("Awaiting Execution Command.")
        return

    with st.spinner("Fetching Live Market Data..."):
        adapter = DataAdapter()
        bundle = adapter.load(tickers)
        df = bundle.df

    if df.empty:
        st.error("Data Fetch Failed. Check tickers or connection.")
        return

    # Run engine
    result = run_ensemble(df, specs, crowd_floor=crowd_floor, temp=temp)

    # Metrics
    st.subheader("Market Recon")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe Size", len(tickers))
    c2.metric("Active Strategies", sum(s.enabled for s in specs))
    best_stock = result.index[0] if not result.empty else "N/A"
    c3.metric("Top Alpha Pick", best_stock)
    c4.metric("UTC Time", datetime.now(timezone.utc).strftime("%H:%M:%S"))

    # Main Table
    st.subheader("Alpha Rankings")
    
    # Dynamic column selection based on data availability
    base_cols = ["alpha_rank", "alpha_score", "price_now", "industry", "fwd_pe", "ret_3m"]
    # Filter base_cols to only those in result
    final_cols = [c for c in base_cols if c in result.columns]
    
    st.dataframe(
        result[final_cols].style.background_gradient(subset=["alpha_score"], cmap="viridis"),
        use_container_width=True
    )

    # Detail View
    st.subheader("Signal Decomposition")
    st.caption("Standardized Z-Scores (Higher = Stronger Signal)")
    sig_cols = [c for c in result.columns if c.startswith("sig_")]
    st.dataframe(result[sig_cols].head(15), use_container_width=True)

    # Correlation Matrix
    with st.expander("Strategy Correlations (Crowding Check)"):
        enabled_keys = [s.key for s in specs if s.enabled]
        if len(enabled_keys) > 1:
            sig_df = result[[f"sig_{k}" for k in enabled_keys]].rename(columns={f"sig_{k}": k for k in enabled_keys})
            corr = corr_matrix(sig_df)
            st.dataframe(corr.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1), use_container_width=True)
        else:
            st.write("Need >1 strategy for correlation analysis.")

if __name__ == "__main__":
    main()
