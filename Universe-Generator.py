#!/usr/bin/env python3
"""
Universe-Generator.py (Mixed Crypto + Equities) -> SQLite
Resilient build: retries/backoff + Binance mirrors + graceful crypto skip.

WHAT IT DOES
- Crypto: Binance exchangeInfo (public) with mirror fallbacks + retry/backoff.
- Equities/ETFs: Alpaca /v2/assets (requires ALPACA_API_KEY/ALPACA_API_SECRET).
- Writes into SQLite table matching Titan Mobile app schema:
    asset_class, market, symbol, display_name, provider, provider_symbol, sector, market_cap, updated_at

RUN
  python Universe-Generator.py --db titan_universe.sqlite --table ticker_universe --equities 700 --crypto 400

ENV
  ALPACA_API_KEY
  ALPACA_API_SECRET
  ALPACA_TRADING_BASE_URL (optional; default https://api.alpaca.markets)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

DEFAULT_DB = "titan_universe.sqlite"
DEFAULT_TABLE = "ticker_universe"


# =========================
# Time / Helpers
# =========================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def http_get_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 25,
    retries: int = 5,
) -> Any:
    """
    Resilient GET -> JSON:
    - Adds a real-ish User-Agent (helps with some WAFs)
    - Retries with exponential backoff + jitter
    - If non-200, raises HTTPError with a snippet of the response body for debugging
    """
    headers = headers or {}
    headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code != 200:
                raise requests.HTTPError(
                    f"HTTP {r.status_code} for {url} | body={r.text[:200]}",
                    response=r,
                )
            return r.json()
        except Exception as e:
            last_err = e
            sleep_s = min(8.0, (2 ** (attempt - 1)) * 0.5) + random.random() * 0.25
            time.sleep(sleep_s)

    # exhausted retries
    raise last_err  # type: ignore[misc]


# =========================
# SQLite
# =========================
def ensure_db(db_path: str, table: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                asset_class TEXT,
                market TEXT,
                symbol TEXT NOT NULL,
                display_name TEXT,
                provider TEXT,
                provider_symbol TEXT,
                sector TEXT,
                market_cap REAL,
                updated_at TEXT
            );
            """
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_symbol ON {table}(symbol);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_asset_market ON {table}(asset_class, market);")
        con.commit()
    finally:
        con.close()


def wipe_table(db_path: str, table: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(f"DELETE FROM {table};")
        con.commit()
    finally:
        con.close()


def insert_rows(db_path: str, table: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    cols = [
        "asset_class",
        "market",
        "symbol",
        "display_name",
        "provider",
        "provider_symbol",
        "sector",
        "market_cap",
        "updated_at",
    ]

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        placeholders = ",".join(["?"] * len(cols))
        values = []
        for r in rows:
            values.append([r.get(c) for c in cols])

        cur.executemany(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
            values,
        )
        con.commit()
    finally:
        con.close()


# =========================
# Crypto Universe (Binance) with fallbacks
# =========================
def fetch_crypto_binance_usdt_pairs(limit: int = 400) -> List[Dict[str, Any]]:
    """
    Pulls Binance symbols and selects spot-tradable USDT pairs.
    Tries multiple endpoints/mirrors. If all fail, returns [] without crashing.
    """

    endpoints = [
        "https://api.binance.com/api/v3/exchangeInfo",
        "https://data-api.binance.vision/api/v3/exchangeInfo",  # official mirror (often works)
        "https://api1.binance.com/api/v3/exchangeInfo",
        "https://api2.binance.com/api/v3/exchangeInfo",
        "https://api3.binance.com/api/v3/exchangeInfo",
    ]

    js = None
    last_err: Optional[Exception] = None

    # Primary: exchangeInfo (best filtering)
    for url in endpoints:
        try:
            js = http_get_json(url)
            if js and isinstance(js, dict) and "symbols" in js:
                break
        except Exception as e:
            last_err = e
            js = None

    # Secondary: ticker/24hr (sometimes allowed where exchangeInfo isn't)
    if js is None:
        try:
            tickers = http_get_json("https://api.binance.com/api/v3/ticker/24hr")
            symbols = [{"symbol": t.get("symbol", ""), "status": "TRADING"} for t in tickers if "symbol" in t]
            js = {"symbols": symbols}
        except Exception as e:
            print(f"[Crypto] Binance blocked/unreachable. Skipping crypto. Last error: {last_err or e}")
            return []

    symbols = js.get("symbols", []) or []
    bad_fragments = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT", "HALFUSDT", "HEDGEUSDT")
    out: List[Dict[str, Any]] = []

    for s in symbols:
        sym = (s.get("symbol") or "").strip()
        if not sym or not sym.endswith("USDT"):
            continue
        if any(sym.endswith(x) for x in bad_fragments):
            continue

        status = s.get("status", "TRADING")
        if status != "TRADING":
            continue

        # exchangeInfo provides baseAsset; fallback doesn't
        base = (s.get("baseAsset") or sym.replace("USDT", "")).strip()

        # In exchangeInfo we can filter spot; in fallback we cannot, so be permissive.
        # Only include if looks like a typical base symbol.
        if not base or len(base) > 15:
            continue

        out.append(
            dict(
                asset_class="Crypto",
                market="Binance Spot",
                symbol=base,          # user-facing label
                display_name=base,
                provider="binance",
                provider_symbol=sym,  # API-native e.g. BTCUSDT
                sector=None,
                market_cap=None,
                updated_at=now_utc_iso(),
            )
        )

    # Prioritize majors if present, then alphabetic.
    majors = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "ATOM", "LTC", "BCH"]
    out_sorted = sorted(out, key=lambda r: (0 if r["symbol"] in majors else 1, r["symbol"]))
    # Dedup base symbols (keep first)
    seen = set()
    dedup = []
    for r in out_sorted:
        if r["symbol"] in seen:
            continue
        seen.add(r["symbol"])
        dedup.append(r)
    return dedup[:limit]


# =========================
# Equities Universe (Alpaca)
# =========================
def fetch_equities_alpaca(limit: int = 700, include_etfs: bool = True) -> List[Dict[str, Any]]:
    """
    Pulls Alpaca assets list (active, tradable). Requires keys.
    We do not assume market cap/sector here (add later via Polygon enrichment).
    """
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in env.")

    base = os.getenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets").rstrip("/")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    js = http_get_json(f"{base}/v2/assets", headers=headers, params={"status": "active"})
    assets = js if isinstance(js, list) else []

    rows: List[Dict[str, Any]] = []
    for a in assets:
        if not a.get("tradable", False):
            continue

        # Alpaca uses class "us_equity" for stocks/ETFs. We ignore options.
        if a.get("class") != "us_equity":
            continue

        symbol = (a.get("symbol") or "").strip()
        if not symbol:
            continue

        exchange = (a.get("exchange") or "US").strip()
        name = (a.get("name") or symbol).strip()

        # Lightweight ETF heuristic based on name (can be improved via reference enrichment)
        asset_class = "ETFs" if (include_etfs and "ETF" in name.upper()) else "Equities"

        rows.append(
            dict(
                asset_class=asset_class,
                market=f"US ({exchange})",
                symbol=symbol,
                display_name=name,
                provider="alpaca",
                provider_symbol=symbol,
                sector=None,
                market_cap=None,
                updated_at=now_utc_iso(),
            )
        )

    # Prefer common tickers: shorter first, then alphabetic
    rows_sorted = sorted(rows, key=lambda r: (len(r["symbol"]), r["symbol"]))
    return rows_sorted[:limit]


# =========================
# Main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--equities", type=int, default=700)
    ap.add_argument("--crypto", type=int, default=400)
    ap.add_argument("--no-wipe", action="store_true", help="Do not clear the table before inserting.")
    args = ap.parse_args()

    ensure_db(args.db, args.table)
    if not args.no_wipe:
        wipe_table(args.db, args.table)

    all_rows: List[Dict[str, Any]] = []

    # Crypto (public) — do not hard fail if blocked
    print(f"[Universe] Fetching crypto from Binance (USDT pairs) limit={args.crypto} …")
    try:
        crypto_rows = fetch_crypto_binance_usdt_pairs(limit=args.crypto)
    except Exception as e:
        print(f"[Universe] Crypto fetch failed — continuing without crypto. Error: {e}")
        crypto_rows = []
    all_rows.extend(crypto_rows)
    print(f"[Universe] Crypto rows: {len(crypto_rows)}")

    # Equities (requires keys)
    print(f"[Universe] Fetching equities/ETFs from Alpaca limit={args.equities} …")
    eq_rows = fetch_equities_alpaca(limit=args.equities, include_etfs=True)
    all_rows.extend(eq_rows)
    print(f"[Universe] Equities/ETFs rows: {len(eq_rows)}")

    insert_rows(args.db, args.table, all_rows)
    print(f"[Universe] Inserted total rows: {len(all_rows)} -> {args.db} ({args.table})")
    print("[Universe] Done.")


if __name__ == "__main__":
    main()
