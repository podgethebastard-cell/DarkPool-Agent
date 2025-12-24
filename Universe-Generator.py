#!/usr/bin/env python3
"""
Universe Generator (Mixed Crypto + Equities) -> SQLite

- Crypto: Binance exchangeInfo (public)
- Equities/ETFs: Alpaca /v2/assets (requires keys)
- Output schema matches Titan Mobile app expectations:
    asset_class, market, symbol, display_name, provider, provider_symbol, sector, market_cap, updated_at

Run:
  python tools/universe_generator.py --db titan_universe.sqlite --table ticker_universe --equities 700 --crypto 400

Env:
  ALPACA_API_KEY
  ALPACA_API_SECRET
  (optional) POLYGON_API_KEY for later enrichment (not required here)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


DEFAULT_DB = "titan_universe.sqlite"
DEFAULT_TABLE = "ticker_universe"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def http_get_json(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, timeout: int = 25):
    r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


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


def fetch_crypto_binance_usdt_pairs(limit: int = 400) -> List[Dict[str, Any]]:
    """
    Pulls Binance symbols and selects spot-tradable USDT pairs.
    Excludes common leveraged tokens & oddities for a cleaner universe.
    """
    js = http_get_json("https://api.binance.com/api/v3/exchangeInfo")
    symbols = js.get("symbols", []) or []

    bad_fragments = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT", "HALFUSDT", "HEDGEUSDT")
    out = []

    for s in symbols:
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("isSpotTradingAllowed") is not True:
            continue
        sym = s.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if any(sym.endswith(x) for x in bad_fragments):
            continue

        base = s.get("baseAsset", sym.replace("USDT", ""))
        out.append(
            dict(
                asset_class="Crypto",
                market="Binance Spot",
                symbol=base,                 # user-facing ticker
                display_name=base,
                provider="binance",
                provider_symbol=sym,         # API-native symbol (e.g., BTCUSDT)
                sector=None,
                market_cap=None,
                updated_at=now_utc_iso(),
            )
        )

    # Stable ordering: prioritize well-known majors first if present, then alphabetic
    majors = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "ATOM", "LTC", "BCH"]
    out_sorted = sorted(out, key=lambda r: (0 if r["symbol"] in majors else 1, r["symbol"]))
    return out_sorted[:limit]


def fetch_equities_alpaca(limit: int = 700, include_etfs: bool = True) -> List[Dict[str, Any]]:
    """
    Pulls Alpaca assets list (active, tradable). Requires keys.
    We do not assume market cap/sector here (that’s enrichment).
    """
    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_API_SECRET", "")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in env.")

    # Trading API (assets) endpoint
    # Alpaca typically uses: https://paper-api.alpaca.markets or https://api.alpaca.markets
    # assets are accessible via trading base; using live by default:
    base = os.getenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")

    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    js = http_get_json(f"{base.rstrip('/')}/v2/assets", headers=headers, params={"status": "active"})
    # js is a list
    assets = js if isinstance(js, list) else []

    rows: List[Dict[str, Any]] = []
    for a in assets:
        if not a.get("tradable", False):
            continue
        if a.get("class") not in ("us_equity", "us_option"):
            # we only want equities/etfs; options excluded by class filter anyway
            continue
        symbol = a.get("symbol")
        if not symbol:
            continue
        exchange = a.get("exchange") or "US"
        name = a.get("name") or symbol
        # Try to distinguish ETF if Alpaca flags it (not always)
        # If you want strict ETF detection later, enrich via Polygon.
        asset_class = "ETFs" if (include_etfs and "ETF" in (name or "").upper()) else "Equities"

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

    # Stable ordering: prefer shorter, common tickers first, then alphabetic
    rows_sorted = sorted(rows, key=lambda r: (len(r["symbol"]), r["symbol"]))
    return rows_sorted[:limit]


def main():
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

    # Crypto (public)
    print(f"[Universe] Fetching crypto from Binance (USDT pairs) limit={args.crypto} …")
    crypto_rows = fetch_crypto_binance_usdt_pairs(limit=args.crypto)
    all_rows.extend(crypto_rows)
    print(f"[Universe] Crypto rows: {len(crypto_rows)}")

    # Equities (requires keys)
    print(f"[Universe] Fetching equities/ETFs from Alpaca limit={args.equities} …")
    eq_rows = fetch_equities_alpaca(limit=args.equities, include_etfs=True)
    all_rows.extend(eq_rows)
    print(f"[Universe] Equities/ETFs rows: {len(eq_rows)}")

    # Insert
    insert_rows(args.db, args.table, all_rows)
    print(f"[Universe] Inserted total rows: {len(all_rows)} -> {args.db} ({args.table})")
    print("[Universe] Done.")


if __name__ == "__main__":
    main()
