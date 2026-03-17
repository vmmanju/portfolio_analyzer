"""
Phase 2: Data ingestion layer.

Fetches historical daily price data via yfinance and stores it in PostgreSQL.
Idempotent, incremental, no duplicates.
"""

import sys
from pathlib import Path

# Ensure project root is on path when run as script (e.g. python services/data_fetcher.py)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.database import get_db_context
from app.models import Price, Stock


# --- Pure fetch (no DB) ---


def fetch_price_data(
    symbol: str,
    start_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from yfinance.
    No DB writes. Returns DataFrame with columns: date, open, high, low, close, volume.
    Dates are Python date, no timezone, sorted ascending.
    """
    if start_date is not None:
        start_str = start_date.isoformat()
        end_str = (date.today() + timedelta(days=1)).isoformat()
        data = yf.download(
            symbol,
            start=start_str,
            end=end_str,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    else:
        # Default to 2 years instead of 'max' to stay under Neon free tier limits
        start_str = (date.today() - timedelta(days=365*2)).isoformat()
        end_str = (date.today() + timedelta(days=1)).isoformat()
        data = yf.download(
            symbol,
            start=start_str,
            end=end_str,
            progress=False,
            auto_adjust=True,
            threads=False,
        )

    if data is None or data.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    # Flatten MultiIndex columns if present (e.g. when multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0).str.lower()
    else:
        data.columns = [c.lower() for c in data.columns]

    # Normalize column names (yfinance: Open, High, Low, Close, Volume)
    col_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    data = data.rename(columns=col_map)
    required = ["open", "high", "low", "close", "volume"]
    for c in required:
        if c not in data.columns:
            data[c] = None
    data = data[required].copy()

    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data["date"] = data.index.date
    data = data[["date", "open", "high", "low", "close", "volume"]]
    data = data.sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"]).dt.date

    return data


# --- DB helpers ---


def get_last_price_date(stock_id: int) -> Optional[date]:
    """Return the latest stored date for the stock, or None if no prices."""
    with get_db_context() as db:
        stmt = (
            select(Price.date)
            .where(Price.stock_id == stock_id)
            .order_by(Price.date.desc())
            .limit(1)
        )
        row = db.execute(stmt).scalar_one_or_none()
        return row


def _to_date(d) -> date:
    """Normalize to Python date (handles pandas Timestamp, datetime, date)."""
    try:
        return d.date()
    except AttributeError:
        pass
    return d if isinstance(d, date) else date.fromisoformat(str(d))


def _existing_dates_for_stock(db, stock_id: int, dates: list[date]) -> set[date]:
    """Return set of dates that already exist for this stock_id."""
    if not dates:
        return set()
    stmt = select(Price.date).where(
        Price.stock_id == stock_id,
        Price.date.in_(dates),
    )
    return set(db.execute(stmt).scalars().all())


def store_price_data(stock_id: int, df: pd.DataFrame) -> tuple[int, int]:
    """
    Insert new price rows only. Skips duplicates (Unique(stock_id, date)).
    Uses bulk insert. Commits once per stock. Returns (inserted_count, skipped_count).
    """
    if df is None or df.empty:
        return 0, 0

    df = df.dropna(subset=["date"])
    if df.empty:
        return 0, 0

    dates = [_to_date(d) for d in df["date"].tolist()]

    with get_db_context() as db:
        existing = _existing_dates_for_stock(db, stock_id, dates)
        to_insert = df[~df["date"].apply(lambda d: _to_date(d) in existing)]
        if to_insert.empty:
            return 0, len(dates)

        rows = []
        for _, r in to_insert.iterrows():
            d = _to_date(r["date"])
            if d in existing:
                continue
            rows.append(
                {
                    "stock_id": stock_id,
                    "date": d,
                    "open": float(r["open"]) if pd.notna(r["open"]) else None,
                    "high": float(r["high"]) if pd.notna(r["high"]) else None,
                    "low": float(r["low"]) if pd.notna(r["low"]) else None,
                    "close": float(r["close"]) if pd.notna(r["close"]) else None,
                    "volume": float(r["volume"]) if pd.notna(r["volume"]) else None,
                }
            )
            existing.add(d)

        if not rows:
            return 0, len(dates)

        try:
            db.add_all([Price(**row) for row in rows])
            # commit happens on exiting get_db_context
        except IntegrityError:
            db.rollback()
            inserted = 0
            for row in rows:
                try:
                    with get_db_context() as inner_db:
                        inner_db.add(Price(**row))
                    inserted += 1
                except IntegrityError:
                    pass
            return inserted, len(dates) - inserted
        except SQLAlchemyError:
            db.rollback()
            raise

    return len(rows), len(dates) - len(rows)


def update_stock_prices(symbol: str) -> tuple[int, int]:
    """
    Fetch Stock from DB, get last stored date, fetch new data from next day,
    store only new rows. Returns (inserted, skipped). Safe to run repeatedly.
    """
    with get_db_context() as db:
        row = db.execute(select(Stock.id).where(Stock.symbol == symbol)).first()
    if row is None:
        print(f"  Skip {symbol}: not in DB")
        return 0, 0

    stock_id = row[0]
    last_date = get_last_price_date(stock_id)
    if last_date is not None:
        start_date = last_date + timedelta(days=1)
        if start_date > date.today():
            print(f"  {symbol}: already up to date")
            return 0, 0
    else:
        start_date = None

    try:
        df = fetch_price_data(symbol, start_date=start_date)
    except Exception as e:
        print(f"  {symbol}: fetch failed - {e}")
        return 0, 0

    if df.empty:
        print(f"  {symbol}: no new data")
        return 0, 0

    try:
        inserted, skipped = store_price_data(stock_id, df)
        print(f"  {symbol}: inserted={inserted}, skipped={skipped}")
        return inserted, skipped
    except SQLAlchemyError as e:
        print(f"  {symbol}: store failed - {e}")
        return 0, 0


def update_all_stocks() -> tuple[int, int, int]:
    """
    Update prices for all stocks in DB. Fails per stock, not entire batch.
    Returns (stocks_processed, total_inserted, total_skipped).
    """
    # Backwards compatible: call the new update_selected_stocks with None
    return update_selected_stocks(None)


def update_selected_stocks(symbols: Optional[list[str]]) -> tuple[int, int, int]:
    """Update prices for given list of `symbols` (formatted SYMBOL.NS).

    If `symbols` is None, update all stocks in DB.
    Returns: (stocks_processed, total_inserted, total_skipped)
    """
    if symbols is None:
        with get_db_context() as db:
            rows = db.execute(select(Stock.id, Stock.symbol)).all()
        if not rows:
            print("No stocks in DB. Run bootstrap_stocks first.")
            return 0, 0, 0
        symbols = [r[1] for r in rows]

    if not symbols:
        print("No symbols provided for update.")
        return 0, 0, 0

    print(f"Updating prices for {len(symbols)} stock(s)...")
    total_inserted = 0
    total_skipped = 0
    processed = 0
    for symbol in symbols:
        inserted, skipped = update_stock_prices(symbol)
        total_inserted += inserted
        total_skipped += skipped
        processed += 1

    return processed, total_inserted, total_skipped


# --- Sector data population ---


# Hardcoded fallback for common NSE-100 stocks (used if yfinance fails)
_NSE_SECTOR_FALLBACK: dict[str, str] = {
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "BPCL.NS": "Energy",
    "IOC.NS": "Energy", "GAIL.NS": "Energy", "HINDPETRO.NS": "Energy",
    "TCS.NS": "Technology", "INFY.NS": "Technology", "WIPRO.NS": "Technology",
    "HCLTECH.NS": "Technology", "TECHM.NS": "Technology", "LTIM.NS": "Technology",
    "PERSISTENT.NS": "Technology", "COFORGE.NS": "Technology", "MPHASIS.NS": "Technology",
    "HDFCBANK.NS": "Financial Services", "ICICIBANK.NS": "Financial Services",
    "KOTAKBANK.NS": "Financial Services", "AXISBANK.NS": "Financial Services",
    "SBIN.NS": "Financial Services", "BAJFINANCE.NS": "Financial Services",
    "BAJAJFINSV.NS": "Financial Services", "INDUSINDBK.NS": "Financial Services",
    "BANKBARODA.NS": "Financial Services", "PNB.NS": "Financial Services",
    "HDFCLIFE.NS": "Financial Services", "SBILIFE.NS": "Financial Services",
    "ICICIPRULI.NS": "Financial Services",
    "SUNPHARMA.NS": "Healthcare", "DRREDDY.NS": "Healthcare",
    "CIPLA.NS": "Healthcare", "DIVISLAB.NS": "Healthcare",
    "APOLLOHOSP.NS": "Healthcare", "BIOCON.NS": "Healthcare",
    "LUPIN.NS": "Healthcare", "TORNTPHARM.NS": "Healthcare",
    "TATAMOTORS.NS": "Automobile", "M&M.NS": "Automobile",
    "MARUTI.NS": "Automobile", "BAJAJ-AUTO.NS": "Automobile",
    "HEROMOTOCO.NS": "Automobile", "EICHERMOT.NS": "Automobile",
    "TVSMOTOR.NS": "Automobile", "TATASTEEL.NS": "Metals & Mining",
    "HINDALCO.NS": "Metals & Mining", "JSWSTEEL.NS": "Metals & Mining",
    "VEDL.NS": "Metals & Mining", "COALINDIA.NS": "Metals & Mining",
    "NMDC.NS": "Metals & Mining",
    "ITC.NS": "Consumer Staples", "HINDUNILVR.NS": "Consumer Staples",
    "NESTLEIND.NS": "Consumer Staples", "BRITANNIA.NS": "Consumer Staples",
    "DABUR.NS": "Consumer Staples", "GODREJCP.NS": "Consumer Staples",
    "MARICO.NS": "Consumer Staples", "COLPAL.NS": "Consumer Staples",
    "TATACONSUM.NS": "Consumer Staples",
    "TITAN.NS": "Consumer Discretionary", "TRENT.NS": "Consumer Discretionary",
    "LT.NS": "Industrials", "ADANIENT.NS": "Industrials",
    "ADANIPORTS.NS": "Industrials", "SIEMENS.NS": "Industrials",
    "ABB.NS": "Industrials", "HAL.NS": "Industrials", "BEL.NS": "Industrials",
    "NTPC.NS": "Utilities", "POWERGRID.NS": "Utilities",
    "TATAPOWER.NS": "Utilities", "ADANIGREEN.NS": "Utilities",
    "ULTRACEMCO.NS": "Materials", "SHREECEM.NS": "Materials",
    "AMBUJACEM.NS": "Materials", "GRASIM.NS": "Materials",
    "PIDILITIND.NS": "Materials", "UPL.NS": "Materials",
    "BHARTIARTL.NS": "Communication Services", "IDEA.NS": "Communication Services",
    "ASIANPAINT.NS": "Materials",
    "DMART.NS": "Consumer Staples",
    "ZOMATO.NS": "Consumer Discretionary",
    "NAUKRI.NS": "Communication Services",
}


def populate_sectors() -> tuple[int, int]:
    """Populate sector data for all stocks where sector is NULL.

    Uses yfinance Ticker.info to fetch sector. Falls back to
    _NSE_SECTOR_FALLBACK for known symbols.

    Returns (updated_count, failed_count).
    """
    import time

    with get_db_context() as db:
        rows = db.execute(
            select(Stock.id, Stock.symbol)
            .where(Stock.sector.is_(None))
        ).all()

    if not rows:
        print("All stocks already have sector data.")
        return 0, 0

    print(f"Populating sectors for {len(rows)} stocks...")
    updated = 0
    failed = 0

    for stock_id, symbol in rows:
        sector = None

        # Try yfinance first
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get("sector") or info.get("industry")
            time.sleep(0.3)  # Rate-limit-safe delay
        except Exception as e:
            print(f"  {symbol}: yfinance lookup failed - {e}")

        # Fallback to hardcoded mapping
        if not sector:
            sector = _NSE_SECTOR_FALLBACK.get(symbol)

        if sector:
            try:
                with get_db_context() as db:
                    stock = db.execute(
                        select(Stock).where(Stock.id == stock_id)
                    ).scalar_one_or_none()
                    if stock:
                        stock.sector = sector
                updated += 1
                print(f"  {symbol}: {sector}")
            except Exception as e:
                print(f"  {symbol}: DB update failed - {e}")
                failed += 1
        else:
            print(f"  {symbol}: no sector found")
            failed += 1

    print(f"Sector population: updated={updated}, failed={failed}")
    return updated, failed


# --- Bootstrap ---


def bootstrap_stocks(stock_list: list[dict]) -> int:
    """
    Insert stocks only if symbol not already present. Do not overwrite.
    Each dict: {"symbol": "...", "name": "...", "sector": "..."}.
    Returns count of newly inserted stocks.
    """
    if not stock_list:
        return 0

    inserted = 0
    with get_db_context() as db:
        for item in stock_list:
            symbol = item.get("symbol")
            if not symbol:
                continue
            existing = db.execute(select(Stock).where(Stock.symbol == symbol)).scalar_one_or_none()
            if existing is not None:
                continue
            db.add(
                Stock(
                    symbol=symbol,
                    name=item.get("name"),
                    sector=item.get("sector"),
                )
            )
            inserted += 1
    return inserted


# --- CLI ---


def _run_cli() -> None:
    """Bootstrap sample stocks and run update_all_stocks. Print validation checklist."""
    SAMPLE_STOCKS = [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy"},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "sector": "Technology"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Financials"},
        {"symbol": "INFY.NS", "name": "Infosys", "sector": "Technology"},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Financials"},
    ]

    print("Bootstrapping sample stocks...")
    added = bootstrap_stocks(SAMPLE_STOCKS)
    print(f"  New stocks added: {added}")

    stocks_processed, total_inserted, total_skipped = update_all_stocks()

    print()
    print("--- Validation checklist ---")
    print(f"  Total stocks processed: {stocks_processed}")
    print(f"  Total rows inserted:    {total_inserted}")
    print(f"  Total skipped:          {total_skipped}")
    print("---")


if __name__ == "__main__":
    _run_cli()
