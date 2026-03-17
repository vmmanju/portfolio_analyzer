"""Universe loader for NSE Top 100.

Responsibilities:
- Fetch NIFTY 100 constituents from official NSE CSV when available.
- Fallback to a small set or yfinance if fetching fails.
- Provide `load_nse_top_100()` which returns list[dict].
"""
from typing import List, Dict, Optional
import csv
import io

import requests


NSE_TOP100_CSV_URL = "https://www1.nseindia.com/content/indices/ind_nifty100list.csv"
NSE_TOP100_JSON_URL = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100"

# If set to a truthy value, perform a quick validation of fetched symbols using yfinance.
# Validation is optional because it requires extra network calls and the `yfinance` package.
VALIDATE_WITH_YFINANCE = bool(("UNIVERSE_VALIDATE_WITH_YFINANCE" in __import__("os").environ) and __import__("os").environ.get("UNIVERSE_VALIDATE_WITH_YFINANCE") not in ("0", "false", "False"))


def _fetch_nse_csv(url: str) -> Optional[str]:
    """Return CSV text or None on failure."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/vnd.ms-excel,application/octet-stream",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200 and resp.text:
            return resp.text
    except Exception:
        return None
    return None


def load_nse_top_100() -> List[Dict]:
    """Load NIFTY 100 list and return list of dicts with `symbol`, `name`, `sector`.

    Returns:
        list of {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": None}
    """
    records: List[Dict] = []

    # 1) Try NSE JSON API (preferred)
    try:
        json_text = _fetch_nse_csv(NSE_TOP100_JSON_URL)
        if json_text:
            import json as _json

            try:
                j = _json.loads(json_text)
                # JSON structure may vary; look for list under common keys
                items = None
                if isinstance(j, dict):
                    # Common key in NSE JSON is 'data' or 'symbols' or 'response'
                    for k in ("data", "symbols", "members", "constituents"):
                        if k in j and isinstance(j[k], list):
                            items = j[k]
                            break
                    # Some endpoints return a list directly
                    if items is None:
                        # try to discover list values
                        for v in j.values():
                            if isinstance(v, list):
                                items = v
                                break
                elif isinstance(j, list):
                    items = j

                if items:
                    for r in items:
                        # try common field names
                        sym = None
                        name = None
                        sector = None
                        if isinstance(r, dict):
                            sym = r.get("symbol") or r.get("SC_CODE") or r.get("code") or r.get("identifier")
                            name = r.get("companyName") or r.get("company") or r.get("name")
                            sector = r.get("sector") or r.get("industry")
                        if not sym:
                            continue
                        sym = str(sym).strip()
                        if not sym.endswith(".NS"):
                            sym = f"{sym}.NS"
                        records.append({"symbol": sym, "name": (name or "").strip(), "sector": sector})
            except Exception:
                records = []
    except Exception:
        records = []

    # 2) Fallback: try CSV if JSON returned nothing
    if not records:
        csv_text = _fetch_nse_csv(NSE_TOP100_CSV_URL)
        if csv_text:
            try:
                f = io.StringIO(csv_text)
                reader = csv.DictReader(f)
                for r in reader:
                    # Many NSE CSVs use 'Symbol' and 'Company Name' or similar headers
                    sym = r.get("Symbol") or r.get("SYMBOL") or r.get("symbol")
                    name = r.get("Company Name") or r.get("Company_Name") or r.get("Company") or r.get("company") or r.get("Name")
                    if not sym:
                        continue
                    sym = sym.strip()
                    if not sym.endswith(".NS"):
                        sym = f"{sym}.NS"
                    records.append({"symbol": sym, "name": (name or "").strip(), "sector": None})
            except Exception:
                records = []

    # 3) Optional: validate via yfinance if requested (reduces false tickers)
    if records and VALIDATE_WITH_YFINANCE:
        try:
            import yfinance as yf

            validated: List[Dict] = []
            for item in records:
                sym = item["symbol"]
                try:
                    t = yf.Ticker(sym)
                    hist = t.history(period="1d")
                    if hist is not None and not hist.empty:
                        validated.append(item)
                except Exception:
                    # if validation fails, skip symbol
                    continue
            if validated:
                records = validated
        except Exception:
            # yfinance not available or failed — keep original records
            pass

    return records


def sync_universe_with_db(stock_list: List[Dict]) -> None:
    """Insert stocks from `stock_list` into DB if missing.

    - Do not delete existing stocks.
    - Do not overwrite sector unless empty.
    Prints summary: inserted, already_existing
    """
    from app.database import get_db_context
    from app.models import Stock
    from sqlalchemy import select

    if not stock_list:
        print("No stocks to sync")
        return

    inserted = 0
    already = 0
    with get_db_context() as db:
        for item in stock_list:
            sym = item.get("symbol")
            if not sym:
                continue
            existing = db.execute(select(Stock).where(Stock.symbol == sym)).scalar_one_or_none()
            if existing is not None:
                already += 1
                # update sector only if empty
                if not existing.sector and item.get("sector"):
                    existing.sector = item.get("sector")
                continue
            db.add(
                Stock(symbol=sym, name=item.get("name"), sector=item.get("sector"))
            )
            inserted += 1

    print(f"Universe sync: inserted={inserted}, already_existing={already}")
