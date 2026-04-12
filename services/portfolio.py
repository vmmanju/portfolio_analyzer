"""
Phase 5: Portfolio construction engine.

Builds portfolios from latest ranking data using configurable strategies,
stores allocations in portfolio_allocations. Idempotent and re-runnable.
No backtesting, scheduler, or API.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from datetime import date
from typing import Any, Optional

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_db_context
from app.models import Factor, PortfolioAllocation, Score, Stock


# Strategy identifiers for DB and display
STRATEGY_EQUAL_WEIGHT = "equal_weight"
LEGACY_STRATEGY_EQUAL_WEIGHT = "equal_weight_top_n"
STRATEGY_INVERSE_VOL = "inverse_volatility"

# Inverse-vol: use volatility_score (z-scored) as proxy when raw vol not stored.
# Weight ∝ 1 / (volatility_score + offset); offset keeps denominator positive.
VOLATILITY_ZSCORE_OFFSET = 2.0


def is_equal_weight_strategy(strategy: str | None) -> bool:
    """Return True for the canonical equal-weight strategy or its legacy alias."""
    return strategy in {STRATEGY_EQUAL_WEIGHT, LEGACY_STRATEGY_EQUAL_WEIGHT}


def get_latest_scoring_date() -> Optional[date]:
    """Return the most recent date in the scores table, or None if empty."""
    with get_db_context() as db:
        stmt = select(Score.date).order_by(Score.date.desc()).limit(1)
        row = db.execute(stmt).scalars().first()
    return row


def load_ranked_stocks(as_of_date: date, selected_symbols: list[str] | None = None, sector: str | None = None, data_cache: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Join scores, stocks, and factors for the given date.
    Returns DataFrame with stock_id, symbol, sector, composite_score, rank, volatility_score,
    sorted by rank ascending.
    """
    if data_cache is not None:
        # Filter cache
        mask = (data_cache["date"] == as_of_date)
        df = data_cache.loc[mask].copy()
        if selected_symbols:
            df = df[df["symbol"].isin(selected_symbols)]
        if sector:
            df = df[df["sector"].str.contains(sector, case=False, na=False)]
        return df.sort_values("rank").reset_index(drop=True)

    with get_db_context() as db:
        stmt = (
            select(
                Score.stock_id,
                Stock.symbol,
                Stock.sector,
                Score.composite_score,
                Score.rank,
                Factor.volatility_score,
            )
            .join(Stock, Stock.id == Score.stock_id)
            .outerjoin(
                Factor,
                (Factor.stock_id == Score.stock_id) & (Factor.date == Score.date),
            )
            .where(Score.date == as_of_date)
        )

        if sector:
            # Case-insensitive partial match for sector
            stmt = stmt.where(Stock.sector.ilike(f"%{sector}%"))

        if selected_symbols:
            # Map symbols -> ids and filter
            ids = [r[0] for r in db.execute(select(Stock.id).where(Stock.symbol.in_(selected_symbols))).all()]
            if ids:
                stmt = stmt.where(Score.stock_id.in_(ids))
            else:
                rows = []
                stmt = None

        if stmt is not None:
            stmt = stmt.order_by(Score.rank)
            rows = db.execute(stmt).all()

    if not rows:
        return pd.DataFrame(
            columns=["stock_id", "symbol", "sector", "composite_score", "rank", "volatility_score"]
        )

    df = pd.DataFrame(
        rows,
        columns=["stock_id", "symbol", "sector", "composite_score", "rank", "volatility_score"],
    )
    return df.sort_values("rank").reset_index(drop=True)


def construct_equal_weight_portfolio(
    as_of_date: date,
    top_n: int = 20,
    selected_symbols: list[str] | None = None,
    data_cache: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Select top_n ranked stocks and assign equal weight 1/n.
    Returns DataFrame with columns stock_id, weight. Validates sum(weights) ≈ 1.
    """
    df = load_ranked_stocks(as_of_date, selected_symbols=selected_symbols, data_cache=data_cache)
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "weight"])

    top = df.head(top_n)
    n = len(top)
    if n == 0:
        return pd.DataFrame(columns=["stock_id", "weight"])

    weight = 1.0 / n
    out = top[["stock_id"]].copy()
    out["weight"] = weight
    return out


def construct_inverse_vol_portfolio(
    as_of_date: date,
    top_n: int = 20,
    selected_symbols: list[str] | None = None,
    data_cache: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Select top_n ranked stocks. Weight ∝ 1 / (volatility_score + offset), then normalize.
    Stocks with missing volatility_score are skipped. Returns DataFrame with stock_id, weight.
    """
    df = load_ranked_stocks(as_of_date, selected_symbols=selected_symbols, data_cache=data_cache)
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "weight"])

    top = df.head(top_n)
    # Drop rows with missing volatility_score
    has_vol = top["volatility_score"].notna()
    top = top.loc[has_vol].copy()
    if top.empty:
        return pd.DataFrame(columns=["stock_id", "weight"])

    # Inverse volatility proxy: 1 / (volatility_score + offset)
    vol_proxy = (top["volatility_score"].astype(float) + VOLATILITY_ZSCORE_OFFSET).clip(lower=1e-8)
    inv_vol = 1.0 / vol_proxy
    total = inv_vol.sum()
    if total <= 0:
        return pd.DataFrame(columns=["stock_id", "weight"])

    out = top[["stock_id"]].copy()
    out["weight"] = (inv_vol / total).values
    return out


def store_portfolio(
    as_of_date: date,
    allocation_df: pd.DataFrame,
    strategy_type: str,
) -> tuple[int, int]:
    """
    Replace all allocations for (date, strategy_type) with the given allocation.
    Deletes existing rows for that date+strategy, then inserts new rows.
    Returns (rows_inserted, rows_deleted_as_updated).
    """
    if allocation_df.empty or "stock_id" not in allocation_df.columns or "weight" not in allocation_df.columns:
        return 0, 0

    with get_db_context() as db:
        deleted = db.execute(
            delete(PortfolioAllocation).where(
                PortfolioAllocation.date == as_of_date,
                PortfolioAllocation.strategy_type == strategy_type,
            )
        )
        deleted_count = deleted.rowcount if deleted.rowcount is not None else 0

        records: list[dict[str, Any]] = []
        for _, r in allocation_df.iterrows():
            records.append({
                "date": as_of_date,
                "stock_id": int(r["stock_id"]),
                "weight": float(r["weight"]),
                "strategy_type": strategy_type,
            })

        if records:
            db.add_all([PortfolioAllocation(**rec) for rec in records])
        inserted_count = len(records)

    return inserted_count, deleted_count


def run_portfolio_construction() -> None:
    """
    Get latest scoring date; build Equal Weight and Inverse Volatility portfolios;
    store both; print summary and validation (weight sum, min/max, top 5) per strategy.
    """
    as_of_date = get_latest_scoring_date()
    if as_of_date is None:
        print("No scoring date found. Run scoring engine first.")
        return

    print(f"Latest scoring date: {as_of_date}")
    ranked = load_ranked_stocks(as_of_date)
    print()

    # --- Equal Weight ---
    print("Strategy A: Equal Weight Top N")
    eq_df = construct_equal_weight_portfolio(as_of_date, top_n=20)
    if eq_df.empty:
        print("  No allocations (no ranked stocks).")
    else:
        eq_inserted, eq_deleted = store_portfolio(as_of_date, eq_df, STRATEGY_EQUAL_WEIGHT)
        total_weight = eq_df["weight"].sum()
        print(f"  Stocks:     {len(eq_df)}")
        print(f"  Sum(weight): {total_weight:.6f}")
        print(f"  Min weight: {eq_df['weight'].min():.6f}")
        print(f"  Max weight: {eq_df['weight'].max():.6f}")
        print(f"  Stored:     inserted={eq_inserted}, replaced={eq_deleted}")

        top5 = eq_df.head(5).merge(
            ranked[["stock_id", "symbol"]].drop_duplicates(),
            on="stock_id",
            how="left",
        )
        print("  Top 5 allocations:")
        for _, r in top5.iterrows():
            sym = r.get("symbol", "")
            print(f"    {str(sym):20s}  weight={r['weight']:.4f}")
    print()

    # --- Inverse Volatility ---
    print("Strategy B: Inverse Volatility Weight")
    iv_df = construct_inverse_vol_portfolio(as_of_date, top_n=20)
    if iv_df.empty:
        print("  No allocations (no ranked stocks or missing volatility).")
    else:
        iv_inserted, iv_deleted = store_portfolio(as_of_date, iv_df, STRATEGY_INVERSE_VOL)
        total_weight = iv_df["weight"].sum()
        print(f"  Stocks:     {len(iv_df)}")
        print(f"  Sum(weight): {total_weight:.6f}")
        print(f"  Min weight: {iv_df['weight'].min():.6f}")
        print(f"  Max weight: {iv_df['weight'].max():.6f}")
        print(f"  Stored:     inserted={iv_inserted}, replaced={iv_deleted}")

        top5 = iv_df.head(5).merge(
            ranked[["stock_id", "symbol"]].drop_duplicates(),
            on="stock_id",
            how="left",
        )
        print("  Top 5 allocations:")
        for _, r in top5.iterrows():
            sym = r.get("symbol", "")
            print(f"    {str(sym):20s}  weight={r['weight']:.4f}")
    print()

    print("--- Validation ---")
    for name, df in [("Equal Weight", eq_df), ("Inverse Vol", iv_df)]:
        if df.empty:
            print(f"  {name}: no data")
        else:
            s = df["weight"].sum()
            print(f"  {name}: sum(weights) = {s:.6f}  (expect 1.0)")
    print("---")
    print("Portfolio construction complete.")


def load_ranked_stocks_batch(
    as_of_dates: list[date], 
    selected_symbols: list[str] | None = None
) -> pd.DataFrame:
    """Batch version to fetch rankings for multiple dates in one query."""
    if not as_of_dates:
        return pd.DataFrame()

    with get_db_context() as db:
        stmt = (
            select(
                Score.date,
                Score.stock_id,
                Stock.symbol,
                Stock.sector,
                Score.composite_score,
                Score.rank,
                Factor.volatility_score,
            )
            .join(Stock, Stock.id == Score.stock_id)
            .outerjoin(
                Factor,
                (Factor.stock_id == Score.stock_id) & (Factor.date == Score.date),
            )
            .where(Score.date.in_(as_of_dates))
        )

        if selected_symbols:
            # Map symbols -> ids just once
            ids_stmt = select(Stock.id).where(Stock.symbol.in_(selected_symbols))
            ids = [r[0] for r in db.execute(ids_stmt).all()]
            if ids:
                stmt = stmt.where(Score.stock_id.in_(ids))
            else:
                return pd.DataFrame()

        rows = db.execute(stmt).all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["date", "stock_id", "symbol", "sector", "composite_score", "rank", "volatility_score"],
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


if __name__ == "__main__":
    run_portfolio_construction()
