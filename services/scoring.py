"""
Phase 4: Composite scoring and ranking.

Reads normalized factor data from DB, computes weighted composite score,
ranks stocks cross-sectionally per date, and upserts into the scores table.
Idempotent and re-runnable. No portfolio logic.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_db_context
from app.models import Factor, Score, Stock


# Configurable weights for composite score (volatility not included; used later for weighting)
FACTOR_WEIGHTS = {
    "quality": 0.30,
    "growth": 0.25,
    "momentum": 0.25,
    "value": 0.20,
}


def load_factor_data(selected_symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Query all rows from factors table.
    Returns DataFrame with columns: stock_id, date, value_score, quality_score,
    growth_score, momentum_score, volatility_score, sorted by date.
    """
    with get_db_context() as db:
        if selected_symbols:
            # map symbols to ids
            ids = [r[0] for r in db.execute(select(Stock.id).where(Stock.symbol.in_(selected_symbols))).all()]
            if ids:
                stmt = (
                    select(
                        Factor.stock_id,
                        Factor.date,
                        Factor.value_score,
                        Factor.quality_score,
                        Factor.growth_score,
                        Factor.momentum_score,
                        Factor.volatility_score,
                    )
                    .where(Factor.stock_id.in_(ids))
                    .order_by(Factor.date)
                )
            else:
                rows = []
                stmt = None
        else:
            stmt = (
                select(
                    Factor.stock_id,
                    Factor.date,
                    Factor.value_score,
                    Factor.quality_score,
                    Factor.growth_score,
                    Factor.momentum_score,
                    Factor.volatility_score,
                )
                .order_by(Factor.date)
            )
        rows = db.execute(stmt).all() if stmt is not None else []

    if not rows:
        return pd.DataFrame(
            columns=[
                "stock_id",
                "date",
                "value_score",
                "quality_score",
                "growth_score",
                "momentum_score",
                "volatility_score",
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "stock_id",
            "date",
            "value_score",
            "quality_score",
            "growth_score",
            "momentum_score",
            "volatility_score",
        ],
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite_score for each row using FACTOR_WEIGHTS.
    Formula: quality*0.30 + growth*0.25 + momentum*0.25 + value*0.20.
    Volatility is not included. Returns DataFrame with composite_score column added.
    """
    if df.empty:
        df = df.copy()
        df["composite_score"] = pd.Series(dtype=float)
        return df

    out = df.copy()
    out["composite_score"] = (
        FACTOR_WEIGHTS["quality"] * out["quality_score"].fillna(0)
        + FACTOR_WEIGHTS["growth"] * out["growth_score"].fillna(0)
        + FACTOR_WEIGHTS["momentum"] * out["momentum_score"].fillna(0)
        + FACTOR_WEIGHTS["value"] * out["value_score"].fillna(0)
    )
    return out


def rank_cross_sectionally(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each date, rank stocks descending by composite_score.
    Rank 1 = highest score. Uses dense ranking (no gaps).
    Adds column: rank. Returns updated DataFrame.
    """
    if df.empty or "composite_score" not in df.columns:
        df = df.copy()
        df["rank"] = pd.Series(dtype="Int64")
        return df

    out = df.copy()
    out["rank"] = (
        out.groupby("date", sort=True)["composite_score"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    return out


def store_scores(df: pd.DataFrame) -> tuple[int, int]:
    """
    Upsert (stock_id, date, composite_score, rank) into scores table.
    If record exists -> update; else -> insert. One batch commit.
    Respects Unique(stock_id, date). Returns (rows_upserted, 0).
    """
    if df.empty or "composite_score" not in df.columns or "rank" not in df.columns:
        return 0, 0

    def _to_date(d) -> date:
        if isinstance(d, date):
            return d
        if hasattr(d, "date") and callable(getattr(d, "date")):
            return d.date()
        return pd.Timestamp(d).date()

    records: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        rank_val = r["rank"]
        if pd.isna(rank_val) or (isinstance(rank_val, float) and rank_val != rank_val):
            rank_val = None
        else:
            rank_val = int(rank_val)
        records.append({
            "stock_id": int(r["stock_id"]),
            "date": _to_date(r["date"]),
            "composite_score": float(r["composite_score"]) if pd.notna(r["composite_score"]) else None,
            "rank": rank_val,
        })

    if not records:
        return 0, 0

    batch_size = 10000
    upserted_count = 0
    with get_db_context() as db:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            insert_stmt = pg_insert(Score).values(batch)
            stmt = insert_stmt.on_conflict_do_update(
                index_elements=["stock_id", "date"],
                set_={
                    Score.composite_score: insert_stmt.excluded.composite_score,
                    Score.rank: insert_stmt.excluded.rank,
                },
            )
            db.execute(stmt)
            upserted_count += len(batch)
            db.commit() # Commit after each batch
    return upserted_count, 0


def run_scoring_engine(selected_symbols: list[str] | None = None) -> None:
    """
    Pipeline: load factors -> compute composite scores -> rank per date -> store.
    Prints summary and validation (top 10 for latest date, mean/std composite).
    """
    print("Loading factor data...")
    df = load_factor_data(selected_symbols=selected_symbols)
    if df.empty:
        print("No factor data. Run factor_engine first.")
        return

    print(f"  Rows loaded: {len(df)}")

    print("Computing composite scores...")
    df = compute_composite_scores(df)

    print("Ranking cross-sectionally...")
    df = rank_cross_sectionally(df)

    print("Storing scores...")
    try:
        upserted, _ = store_scores(df)
        print(f"  Rows upserted: {upserted}")
    except SQLAlchemyError as e:
        print(f"  Store failed: {e}")
        raise

    # Summary
    dates_count = df["date"].nunique()
    latest_date = df["date"].max()
    print()
    print("--- Summary ---")
    print(f"  Total rows processed:  {len(df)}")
    print(f"  Rows inserted/updated: {upserted}")
    print(f"  Latest date scored:    {latest_date}")
    print("---")

    # Validation: top 10 for latest date with symbol; mean and std composite
    latest = df[df["date"] == latest_date].copy()
    if latest.empty:
        print("No data for latest date.")
        return

    with get_db_context() as db:
        rows = (
            db.execute(
                select(Stock.symbol, Score.composite_score, Score.rank)
                .join(Score, Score.stock_id == Stock.id)
                .where(Score.date == latest_date)
                .order_by(Score.rank)
                .limit(10)
            )
            .all()
        )

    print()
    print("--- Top 10 (latest date) ---")
    print(f"  Date: {latest_date}")
    for r in rows:
        print(f"    {r[0]:20s}  composite_score={r[1]:.4f}  rank={r[2]}")
    print("---")

    comp = latest["composite_score"].astype(float)
    mean_c, std_c = comp.mean(), comp.std()
    print()
    print("--- Validation ---")
    print(f"  Mean composite_score:  {mean_c:.4f}  (expect ≈ 0 if factors are z-scored)")
    print(f"  Std  composite_score:  {std_c:.4f}")
    print("---")
    print("Scoring engine run complete.")


if __name__ == "__main__":
    run_scoring_engine()
