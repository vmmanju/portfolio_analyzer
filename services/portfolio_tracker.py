"""Helpers for persisted portfolio tracker positions and live valuation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import delete, func, select

from app.database import get_db_context
from app.models import PortfolioTrackerPosition, Price, Stock


TRACKER_COLUMNS = ["symbol", "invested_amount", "quantity"]


def normalize_tracker_positions(rows: Iterable[Dict[str, Any]] | pd.DataFrame | None) -> List[Dict[str, float | str]]:
    """Normalize UI-edited tracker rows into validated tracker records."""
    if rows is None:
        return []

    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame(list(rows))

    if df.empty:
        return []

    for col in TRACKER_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[TRACKER_COLUMNS].copy()
    df["symbol"] = df["symbol"].fillna("").astype(str).str.strip().str.upper()
    df["invested_amount"] = pd.to_numeric(df["invested_amount"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df[(df["symbol"] != "") & df["invested_amount"].notna() & df["quantity"].notna()]
    df = df[(df["invested_amount"] >= 0) & (df["quantity"] > 0)]
    df = df.drop_duplicates(subset=["symbol"], keep="last")
    if df.empty:
        return []
    return df.sort_values("symbol").to_dict("records")


def load_tracked_positions(user_id: Optional[int] = None) -> List[Dict[str, float | str]]:
    """Load persisted portfolio tracker rows for one user."""
    with get_db_context() as db:
        stmt = (
            select(
                PortfolioTrackerPosition.symbol,
                PortfolioTrackerPosition.invested_amount,
                PortfolioTrackerPosition.quantity,
            )
            .order_by(PortfolioTrackerPosition.symbol)
        )
        if user_id is None:
            stmt = stmt.where(PortfolioTrackerPosition.user_id.is_(None))
        else:
            stmt = stmt.where(PortfolioTrackerPosition.user_id == user_id)
        rows = db.execute(stmt).all()

    return [
        {
            "symbol": row.symbol,
            "invested_amount": float(row.invested_amount or 0.0),
            "quantity": float(row.quantity or 0.0),
        }
        for row in rows
    ]


def save_tracked_positions(
    rows: Iterable[Dict[str, Any]] | pd.DataFrame | None,
    user_id: Optional[int] = None,
) -> List[Dict[str, float | str]]:
    """Replace the tracked positions for a user with the provided normalized rows."""
    positions = normalize_tracker_positions(rows)
    symbols = [row["symbol"] for row in positions]

    with get_db_context() as db:
        delete_stmt = delete(PortfolioTrackerPosition)
        if user_id is None:
            delete_stmt = delete_stmt.where(PortfolioTrackerPosition.user_id.is_(None))
        else:
            delete_stmt = delete_stmt.where(PortfolioTrackerPosition.user_id == user_id)
        db.execute(delete_stmt)

        stock_map: Dict[str, int] = {}
        if symbols:
            stock_rows = db.execute(
                select(Stock.id, Stock.symbol).where(Stock.symbol.in_(symbols))
            ).all()
            stock_map = {symbol: stock_id for stock_id, symbol in stock_rows}

        for row in positions:
            db.add(
                PortfolioTrackerPosition(
                    user_id=user_id,
                    stock_id=stock_map.get(str(row["symbol"])),
                    symbol=str(row["symbol"]),
                    invested_amount=float(row["invested_amount"]),
                    quantity=float(row["quantity"]),
                )
            )

    return positions


def _load_latest_prices(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    if not symbols:
        return {}

    latest_price_subquery = (
        select(
            Price.stock_id.label("stock_id"),
            func.max(Price.date).label("latest_date"),
        )
        .group_by(Price.stock_id)
        .subquery()
    )

    with get_db_context() as db:
        rows = db.execute(
            select(Stock.symbol, Price.close, Price.date)
            .join(latest_price_subquery, latest_price_subquery.c.stock_id == Stock.id)
            .join(
                Price,
                (Price.stock_id == latest_price_subquery.c.stock_id)
                & (Price.date == latest_price_subquery.c.latest_date),
            )
            .where(Stock.symbol.in_(symbols))
        ).all()

    return {
        symbol: {"current_price": float(close) if close is not None else None, "price_date": price_date}
        for symbol, close, price_date in rows
    }


def build_tracker_snapshot(
    rows: Iterable[Dict[str, Any]] | pd.DataFrame | None = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a valuation snapshot using the latest available closing prices."""
    positions = normalize_tracker_positions(rows if rows is not None else load_tracked_positions(user_id=user_id))
    if not positions:
        empty_df = pd.DataFrame(
            columns=[
                "Symbol",
                "Invested Amount",
                "Quantity",
                "Average Cost",
                "Current Price",
                "Current Amount",
                "P&L",
                "P&L %",
                "Price Date",
            ]
        )
        return {"positions_df": empty_df, "summary": {}, "missing_prices": []}

    price_map = _load_latest_prices([str(row["symbol"]) for row in positions])
    display_rows: List[Dict[str, Any]] = []
    missing_prices: List[str] = []

    for row in positions:
        symbol = str(row["symbol"])
        invested = float(row["invested_amount"])
        quantity = float(row["quantity"])
        price_info = price_map.get(symbol, {})
        current_price = price_info.get("current_price")
        current_amount = current_price * quantity if current_price is not None else None
        pnl = current_amount - invested if current_amount is not None else None
        pnl_pct = (pnl / invested) if (pnl is not None and invested > 0) else None
        if current_price is None:
            missing_prices.append(symbol)

        display_rows.append(
            {
                "Symbol": symbol,
                "Invested Amount": invested,
                "Quantity": quantity,
                "Average Cost": invested / quantity if quantity > 0 else None,
                "Current Price": current_price,
                "Current Amount": current_amount,
                "P&L": pnl,
                "P&L %": pnl_pct,
                "Price Date": price_info.get("price_date"),
            }
        )

    positions_df = pd.DataFrame(display_rows).sort_values("Symbol").reset_index(drop=True)
    total_invested = float(positions_df["Invested Amount"].fillna(0).sum())
    total_current = float(positions_df["Current Amount"].fillna(0).sum())
    total_pnl = total_current - total_invested
    priced_positions = int(positions_df["Current Price"].notna().sum())
    total_positions = int(len(positions_df))
    latest_price_date = None
    if positions_df["Price Date"].notna().any():
        latest_price_date = positions_df["Price Date"].dropna().max()

    summary = {
        "total_invested": total_invested,
        "total_current": total_current,
        "total_pnl": total_pnl,
        "total_pnl_pct": (total_pnl / total_invested) if total_invested > 0 else None,
        "priced_positions": priced_positions,
        "total_positions": total_positions,
        "latest_price_date": latest_price_date,
    }
    return {"positions_df": positions_df, "summary": summary, "missing_prices": missing_prices}


__all__ = [
    "TRACKER_COLUMNS",
    "normalize_tracker_positions",
    "load_tracked_positions",
    "save_tracked_positions",
    "build_tracker_snapshot",
]
