# app.py
import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable, List, Tuple, Optional
from datetime import datetime, timezone

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
# DATA ADAPTER (PLUG-IN READY)
# ----------------------------
@dataclass
class DataBundle:
    df: pd.DataFrame
    sources: pd.DataFrame  # tidy log: ticker, metric, value, source, asof

class DataAdapter:
    """
    Replace this with your real multi-source adapter.
    The engine expects a wide df with rows=tickers and columns=metrics.
    """
    def load(self, tickers: List[str]) -> DataBundle:
        # ---- MOCK SHAPE ONLY (no fake financials in production) ----
        # Here we generate a skeleton with NaNs so the engine can run.
        cols = [
            "ticker","name","exchange","country","industry","mcap_usd","adv_shares","adv_usd",
            "close_2024_12_31","close_2025_04_01","close_2025_06_20","price_now",
            # forward/forecast
            "fwd_pe","fwd_ps","fwd_pb","fwd_p_fcf","peg",
            "sales_g_1y","sales_g_3y","eps_g_1y","eps_g_3y",
            "profit_margin","div_yield","div_g_forecast","debt_to_equity",
            "fcfps_ntm","target_price","reco",
            # quality + risk
            "roic","gross_margin","op_margin","fcf_margin",
            "vol_90d","maxdd_1y","beta",
            # momentum
            "ret_1m","ret_3m","ret_6m","ret_12m"
        ]
        df = pd.DataFrame({"ticker": tickers})
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df["name"] = df["ticker"]
        df["exchange"] = "N/A"
        df["country"] = "N/A"
        df["industry"] = "N/A"

        sources = pd.DataFrame(columns=["ticker","metric","value","source","asof","notes"])
        return DataBundle(df=df.set_index("ticker"), sources=sources)

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
# returns (signal, confidence)

def strat_value(df: pd.DataFrame, p: Dict) -> Tuple[pd.Series, pd.Series]:
    w = p.get("winsor", 0.02)
    inv = p.get("invert", True)
    metrics = ["fwd_pe","fwd_ps","fwd_p_fcf","peg"]
    sigs = []
    for m in metrics:
        s = winsorize(df[m], w)
        z = robust_z(s)
        sigs.append(-z if inv else z)
    signal = pd.concat(sigs, axis=1).mean(axis=1)
    conf = confidence_from_coverage(df, metrics)
    return signal, conf

def strat_quality(df: pd.DataFrame, p: Dict) -> Tuple[pd.Series, pd.Series]:
    w = p.get("winsor", 0.02)
    metrics = ["roic","gross_margin","op_margin","fcf_margin","profit_margin"]
    sigs = []
    for m in metrics:
        s = winsorize(df[m], w)
        sigs.append(robust_z(s))
    signal = pd.concat(sigs, axis=1).mean(axis=1)
    conf = confidence_from_coverage(df, metrics)
    return signal, conf

def strat_momentum(df: pd.DataFrame, p: Dict) -> Tuple[pd.Series, pd.Series]:
    w = p.get("winsor", 0.02)
    crash_penalty = p.get("crash_penalty", 0.5)
    metrics = ["ret_1m","ret_3m","ret_6m","ret_12m"]
    sig = pd.concat([robust_z(winsorize(df[m], w)) for m in metrics], axis=1).mean(axis=1)
    # penalize names with nasty drawdowns
    dd = robust_z(winsorize(df["maxdd_1y"], w))
    signal = sig - crash_penalty * dd  # dd is negative for worse drawdowns after robust z
    conf = confidence_from_coverage(df, metrics + ["maxdd_1y"])
    return signal, conf

def strat_lowvol(df: pd.DataFrame, p: Dict) -> Tuple[pd.Series, pd.Series]:
    w = p.get("winsor", 0.02)
    metrics = ["vol_90d","beta","maxdd_1y"]
    z_vol = robust_z(winsorize(df["vol_90d"], w))
    z_beta = robust_z(winsorize(df["beta"], w))
    z_dd = robust_z(winsorize(df["maxdd_1y"], w))
    # lower is better => invert
    signal = (-z_vol) + (-z_beta) + (-z_dd)
    signal = signal / 3.0
    conf = confidence_from_coverage(df, metrics)
    return signal, conf

def strat_income(df: pd.DataFrame, p: Dict) -> Tuple[pd.Series, pd.Series]:
    w = p.get("winsor", 0.02)
    metrics = ["div_yield","div_g_forecast"]
    z_y = robust_z(winsorize(df["div_yield"], w))
    z_g = robust_z(winsorize(df["div_g_forecast"], w))
    signal = (z_y + z_g) / 2.0
    conf = confidence_from_coverage(df, metrics)
    return signal, conf

STRATEGIES: Dict[str, Tuple[str, StrategyFn, Dict]] = {
    "value": ("Value Composite", strat_value, {"winsor": 0.02, "invert": True}),
    "quality": ("Quality Composite", strat_quality, {"winsor": 0.02}),
    "momentum": ("Momentum + Crash Penalty", strat_momentum, {"winsor": 0.02, "crash_penalty": 0.5}),
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

    # If nothing enabled, return empty
    if sig_df.empty:
        return pd.DataFrame(index=df.index)

    # Redundancy/crowding control
    corr = corr_matrix(sig_df)
    uniq = uniqueness_penalty(corr, floor=crowd_floor)

    # Strategy weights (user-set -> normalized by softmax so it’s stable)
    w_raw = np.array([sp.weight for sp in specs if sp.enabled], dtype=float)
    keys = [sp.key for sp in specs if sp.enabled]
    w = softmax(w_raw, t=temp)
    w_s = pd.Series(w, index=keys)

    # Weighted sum with uniqueness + confidence
    score = pd.Series(0.0, index=df.index)
    for k in keys:
        # per-stock confidence, per-strategy uniqueness
        score = score + w_s[k] * sig_df[k] * conf_df[k] * uniq.get(k, 1.0)

    out = df.copy()
    out["alpha_score"] = score
    out["alpha_rank"] = out["alpha_score"].rank(ascending=False, method="min")
    # keep signals for explainability
    for k in keys:
        out[f"sig_{k}"] = sig_df[k]
        out[f"conf_{k}"] = conf_df[k]
    return out.sort_values("alpha_score", ascending=False)

# ----------------------------
# UI
# ----------------------------
def main():
    st.set_page_config(page_title="THE ARCHITECT — Alpha Constellation Lab", layout="wide")
    inject_css()

    st.title("🏛️ THE ARCHITECT — ALPHA CONSTELLATION LAB")
    st.caption("Multi-strategy ensemble screener with robust math, crowding control, and full customization.")

    with st.sidebar:
        st.subheader("Universe")
        tickers_raw = st.text_area("Tickers (comma-separated)", "AAPL,MSFT,NVDA,ASML,7203.T,0700.HK", height=120)
        tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]

        st.divider()
        st.subheader("Ensemble Controls")
        crowd_floor = st.slider("Uniqueness penalty floor (crowding control)", 0.10, 0.90, 0.35, 0.05)
        temp = st.slider("Weight softmax temperature (lower = sharper)", 0.2, 3.0, 1.0, 0.1)

        st.divider()
        st.subheader("Strategies (toggle + weights + params)")
        specs: List[StrategySpec] = []
        for key, (label, _, defaults) in STRATEGIES.items():
            with st.expander(label, expanded=(key in ["value","quality","momentum"])):
                enabled = st.checkbox("Enabled", value=True, key=f"en_{key}")
                weight = st.slider("Weight", 0.0, 5.0, 1.0, 0.1, key=f"w_{key}")
                params = {}
                if "winsor" in defaults:
                    params["winsor"] = st.slider("Winsorization (%)", 0.0, 0.10, float(defaults["winsor"]), 0.01, key=f"win_{key}")
                if key == "momentum":
                    params["crash_penalty"] = st.slider("Crash penalty", 0.0, 2.0, float(defaults["crash_penalty"]), 0.1, key=f"cp_{key}")
                if key == "value":
                    params["invert"] = st.checkbox("Invert valuation (cheaper = higher score)", value=True, key=f"inv_{key}")

                specs.append(StrategySpec(key=key, label=label, enabled=enabled, weight=weight, params=params))

        st.divider()
        run_btn = st.button("EXECUTE ALPHA CONSTELLATION")

    if not run_btn:
        st.info("Configure strategies in the sidebar, then run the ensemble.")
        return

    # Load data
    adapter = DataAdapter()
    bundle = adapter.load(tickers)
    df = bundle.df

    # Run engine
    result = run_ensemble(df, specs, crowd_floor=crowd_floor, temp=temp)

    st.subheader("Dashboard")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Tickers")
        st.metric("Count", len(tickers))
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Enabled strategies")
        st.metric("Count", sum(s.enabled for s in specs))
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("As-of (UTC)")
        st.metric("Timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        st.markdown('</div>', unsafe_allow_html=True)

    if result.empty:
        st.error("No strategies enabled or insufficient data.")
        return

    # Explainability view
    st.subheader("Top Results (Explainable)")
    show_cols = ["name","exchange","country","industry","alpha_score","alpha_rank"]
    sig_cols = [c for c in result.columns if c.startswith("sig_")]
    conf_cols = [c for c in result.columns if c.startswith("conf_")]
    preview_cols = [c for c in show_cols if c in result.columns] + sig_cols + conf_cols
    st.dataframe(result[preview_cols].head(25), use_container_width=True)

    st.subheader("Signal Crowding (Strategy Correlations)")
    enabled_keys = [s.key for s in specs if s.enabled]
    sig_df = result[[f"sig_{k}" for k in enabled_keys]].rename(columns={f"sig_{k}": k for k in enabled_keys})
    corr = corr_matrix(sig_df)
    st.dataframe(corr, use_container_width=True)

    st.caption("Next step: connect the DataAdapter to real, cross-verified sources and enforce your 2-source rule there.")

if __name__ == "__main__":
    main()
