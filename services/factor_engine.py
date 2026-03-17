"""
Phase 3: Factor engine (quant core).

Pulls historical prices from DB, computes momentum and volatility factors,
normalizes cross-sectionally (z-scores), and upserts into the factors table.
Idempotent and re-runnable.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import math
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_db_context
from app.models import Factor, Price, Stock


TRADING_DAYS_6M = 126
TRADING_DAYS_12M = 252
TRADING_DAYS_VOL = 252
MIN_STOCKS_FOR_ZSCORE = 5


def load_price_data(stock_id: int) -> pd.DataFrame:
    """
    Query DB for historical close prices for one stock.
    Returns DataFrame with columns: date, close, sorted by date ascending.
    """
    with get_db_context() as db:
        stmt = (
            select(Price.date, Price.close)
            .where(Price.stock_id == stock_id)
            .order_by(Price.date)
        )
        rows = db.execute(stmt).all()

    if not rows:
        return pd.DataFrame(columns=["date", "close"])

    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def compute_stock_factors(stock_id: int) -> pd.DataFrame:
    """
    Load price data, compute daily returns, rolling momentum (6m, 12m),
    and rolling annualized volatility. Returns DataFrame with columns:
    date, momentum_6m, momentum_12m, volatility (raw, not normalized).
    """
    df = load_price_data(stock_id)
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=["date", "momentum_6m", "momentum_12m", "volatility"])

    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)

    # Daily returns: r_t = close_t / close_t-1 - 1
    daily_ret = close / close.shift(1) - 1.0

    # Momentum: (price_today / price_n_days_ago) - 1
    momentum_6m = (close / close.shift(TRADING_DAYS_6M)) - 1.0
    momentum_12m = (close / close.shift(TRADING_DAYS_12M)) - 1.0

    # Rolling 252-day std of daily returns, annualized
    rolling_std = daily_ret.rolling(TRADING_DAYS_VOL, min_periods=TRADING_DAYS_VOL).std()
    volatility = rolling_std * math.sqrt(252)

    out = pd.DataFrame({
        "date": df["date"],
        "momentum_6m": momentum_6m,
        "momentum_12m": momentum_12m,
        "volatility": volatility,
    })
    return out.dropna(subset=["momentum_6m", "momentum_12m", "volatility"]).reset_index(drop=True)


def compute_all_raw_factors(symbols: list[str] | None = None) -> dict[int, pd.DataFrame]:
    """
    Loop through all stocks (or provided `symbols`) and compute raw factor DataFrames.
    If `symbols` is provided, only compute factors for those symbols.
    Returns dict: stock_id -> DataFrame with date, momentum_6m, momentum_12m, volatility.
    """
    with get_db_context() as db:
        if symbols:
            stmt = select(Stock.id, Stock.symbol).where(Stock.symbol.in_(symbols))
            rows = db.execute(stmt).all()
        else:
            rows = db.execute(select(Stock.id)).all()
    stock_ids = [r[0] for r in rows]

    result: dict[int, pd.DataFrame] = {}
    for sid in stock_ids:
        result[sid] = compute_stock_factors(sid)
    return result


def normalize_cross_sectionally(
    raw_factor_dict: dict[int, pd.DataFrame],
) -> list[dict[str, Any]]:
    """
    For each date, gather factor values across stocks, compute cross-sectional
    z-scores. Skip date if fewer than MIN_STOCKS_FOR_ZSCORE stocks have valid data.
    Returns list of records for DB: stock_id, date, value_score, quality_score,
    growth_score, momentum_score, volatility_score.
    """
    rows: list[dict[str, Any]] = []
    for stock_id, df in raw_factor_dict.items():
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "date": r["date"],
                "stock_id": stock_id,
                "momentum_6m": float(r["momentum_6m"]),
                "momentum_12m": float(r["momentum_12m"]),
                "volatility": float(r["volatility"]),
            })

    if not rows:
        return []

    panel = pd.DataFrame(rows)
    records: list[dict[str, Any]] = []

    for d, grp in panel.groupby("date", sort=True):
        g = grp.dropna(subset=["momentum_6m", "momentum_12m", "volatility"])
        if len(g) < MIN_STOCKS_FOR_ZSCORE:
            continue

        m6 = g["momentum_6m"].astype(float)
        m12 = g["momentum_12m"].astype(float)
        vol = g["volatility"].astype(float)

        mean_6, std_6 = m6.mean(), m6.std()
        mean_12, std_12 = m12.mean(), m12.std()
        mean_v, std_v = vol.mean(), vol.std()

        z_6m = (m6 - mean_6) / std_6 if std_6 and std_6 > 0 else pd.Series(0.0, index=g.index)
        z_12m = (m12 - mean_12) / std_12 if std_12 and std_12 > 0 else pd.Series(0.0, index=g.index)
        z_vol = (vol - mean_v) / std_v if std_v and std_v > 0 else pd.Series(0.0, index=g.index)

        momentum_score = (z_6m + z_12m) / 2.0
        volatility_score = -1.0 * z_vol

        d_date = d if isinstance(d, date) else (d.date() if hasattr(d, "date") else d)
        for i in range(len(g)):
            records.append({
                "stock_id": int(g.iloc[i]["stock_id"]),
                "date": d_date,
                "value_score": 0.0,
                "quality_score": 0.0,
                "growth_score": 0.0,
                "momentum_score": float(momentum_score.iloc[i]),
                "volatility_score": float(volatility_score.iloc[i]),
            })

    return records


def store_factors(records: list[dict[str, Any]]) -> tuple[int, int]:
    """
    Upsert factor records into DB. Respects Unique(stock_id, date).
    If record exists -> update; if not -> insert. Commit per batch.
    Returns (rows_inserted, rows_updated) approximate; PostgreSQL ON CONFLICT
    does not distinguish in one call, so we return (total_upserted, 0) or
    count total and report as single batch.
    """
    if not records:
        return 0, 0

    batch_size = 10000
    upserted_count = 0
    with get_db_context() as db:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            insert_stmt = pg_insert(Factor).values(batch)
            stmt = insert_stmt.on_conflict_do_update(
                index_elements=["stock_id", "date"],
                set_={
                    Factor.value_score: insert_stmt.excluded.value_score,
                    Factor.quality_score: insert_stmt.excluded.quality_score,
                    Factor.growth_score: insert_stmt.excluded.growth_score,
                    Factor.momentum_score: insert_stmt.excluded.momentum_score,
                    Factor.volatility_score: insert_stmt.excluded.volatility_score,
                },
            )
            db.execute(stmt)
            upserted_count += len(batch)
            db.commit() # Commit after each batch to avoid long transactions
    return upserted_count, 0


def run_factor_engine(selected_symbols: list[str] | None = None) -> None:
    """
    High-level pipeline: compute raw factors -> normalize -> store.
    Prints summary and validation (mean ≈ 0, std ≈ 1 for latest date).
    """
    if selected_symbols:
        print(f"Computing raw factors for selected {len(selected_symbols)} stocks...")
    else:
        print("Computing raw factors for all stocks...")
    raw = compute_all_raw_factors(symbols=selected_symbols)
    stocks_with_data = sum(1 for df in raw.values() if not df.empty)
    total_raw_rows = sum(len(df) for df in raw.values())
    print(f"  Stocks with factor data: {stocks_with_data}")
    print(f"  Total raw factor rows:   {total_raw_rows}")

    print("Normalizing cross-sectionally (z-scores)...")
    records = normalize_cross_sectionally(raw)
    dates_computed = len({r["date"] for r in records}) if records else 0
    print(f"  Dates with valid z-scores: {dates_computed}")
    print(f"  Records to upsert:         {len(records)}")

    if not records:
        print("No factor records to store. Exiting.")
        return

    print("Upserting factors to DB...")
    try:
        upserted, _ = store_factors(records)
        print(f"  Rows upserted: {upserted}")
    except SQLAlchemyError as e:
        print(f"  Store failed: {e}")
        raise

    print()
    print("--- Summary ---")
    print(f"  Stocks processed:  {stocks_with_data}")
    print(f"  Dates computed:    {dates_computed}")
    print(f"  Rows inserted/updated: {upserted}")
    print("---")

    # Validation: for latest date, mean momentum_score ≈ 0, mean volatility_score ≈ 0, std ≈ 1
    latest_date = max(r["date"] for r in records)
    latest_records = [r for r in records if r["date"] == latest_date]
    if latest_records:
        mom = [r["momentum_score"] for r in latest_records]
        vol = [r["volatility_score"] for r in latest_records]
        n = len(latest_records)
        print()
        print("--- Validation (latest date) ---")
        print(f"  Date:              {latest_date}")
        print(f"  Stocks:            {n}")
        print(f"  momentum_score:    mean={sum(mom)/n:.4f}, std={pd.Series(mom).std():.4f}")
        print(f"  volatility_score:  mean={sum(vol)/n:.4f}, std={pd.Series(vol).std():.4f}")
        print("  (Expect mean ≈ 0, std ≈ 1)")
        print("---")
    print("Factor engine run complete.")


if __name__ == "__main__":
    run_factor_engine()
