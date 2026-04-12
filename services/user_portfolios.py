"""DB persistence helpers for user-defined portfolios.

Provides simple CRUD helpers to save/load portfolio definitions created in the
Streamlit UI into the database (`user_portfolios` and `user_portfolio_stocks`).

Also provides `refresh_portfolio_allocation` to compute and store today's
PortfolioAllocation for a specific user-defined portfolio, using that portfolio's
own stock list and strategy — so allocation data is always user-controlled and
updated daily on demand.
"""

from datetime import date
from typing import List, Dict, Any, Optional

from sqlalchemy import select, delete

from app.database import get_db_context
from app.models import (
    UserPortfolio as DBUserPortfolio,
    UserPortfolioStock as DBUserPortfolioStock,
    Stock,
)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def save_user_portfolio(defn: Dict[str, Any], user_id: Optional[int] = None) -> int:
    """Save or update a single portfolio definition.

    defn: {name, symbols, strategy, regime_mode, top_n}
    user_id: owning user's DB id (None for backward compat).
    Returns the DB id of the saved portfolio.
    """
    name = (defn.get("name") or "").strip()
    if not name:
        raise ValueError("Portfolio must have a name")

    symbols: List[str] = defn.get("symbols") or []

    with get_db_context() as db:
        # Scope lookup by user_id
        stmt = select(DBUserPortfolio).where(DBUserPortfolio.name == name)
        if user_id is not None:
            stmt = stmt.where(DBUserPortfolio.user_id == user_id)
        existing = db.execute(stmt).scalars().first()

        if existing:
            existing.strategy_type = defn.get("strategy")
            existing.regime_mode = defn.get("regime_mode")
            existing.top_n = int(defn.get("top_n") or 0)

            # Replace stocks
            db.execute(
                delete(DBUserPortfolioStock).where(
                    DBUserPortfolioStock.portfolio_id == existing.id
                )
            )

            if symbols:
                rows = db.execute(
                    select(Stock.id, Stock.symbol).where(Stock.symbol.in_(symbols))
                ).all()
                mapping = {r[1]: r[0] for r in rows}
                for sym in symbols:
                    sid = mapping.get(sym)
                    db.add(
                        DBUserPortfolioStock(
                            portfolio_id=existing.id, stock_id=sid, symbol=sym
                        )
                    )
            return int(existing.id)

        # create new
        new = DBUserPortfolio(
            name=name,
            user_id=user_id,
            strategy_type=defn.get("strategy"),
            regime_mode=defn.get("regime_mode"),
            top_n=int(defn.get("top_n") or 0),
        )
        db.add(new)
        db.flush()  # get id

        if symbols:
            rows = db.execute(
                select(Stock.id, Stock.symbol).where(Stock.symbol.in_(symbols))
            ).all()
            mapping = {r[1]: r[0] for r in rows}
            for sym in symbols:
                sid = mapping.get(sym)
                db.add(
                    DBUserPortfolioStock(
                        portfolio_id=new.id, stock_id=sid, symbol=sym
                    )
                )

        return int(new.id)


def save_portfolios(defs: List[Dict[str, Any]], user_id: Optional[int] = None) -> Dict[str, int]:
    """Save multiple portfolio definitions. Returns mapping name->id."""
    out: Dict[str, int] = {}
    for d in defs:
        pid = save_user_portfolio(d, user_id=user_id)
        out[d.get("name")] = pid
    return out


def load_user_portfolios(user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load saved portfolio definitions from DB and return list of dicts.

    Each dict has keys: name, symbols, strategy, regime_mode, top_n
    Optimized to use a single query with joined loading of stocks to avoid N+1 slow down.
    """
    from sqlalchemy.orm import joinedload
    
    out: List[Dict[str, Any]] = []
    with get_db_context() as db:
        stmt = (
            select(DBUserPortfolio)
            .options(joinedload(DBUserPortfolio.stocks))
            .order_by(DBUserPortfolio.name)
        )
        if user_id is not None:
            stmt = stmt.where(DBUserPortfolio.user_id == user_id)
        
        rows = db.execute(stmt).unique().scalars().all()
        for p in rows:
            symbols = [s.symbol for s in p.stocks]
            out.append(
                {
                    "name": p.name,
                    "symbols": symbols,
                    "strategy": p.strategy_type,
                    "regime_mode": p.regime_mode,
                    "top_n": int(p.top_n) if p.top_n is not None else 0,
                }
            )
    return out


def delete_user_portfolio_by_name(name: str, user_id: Optional[int] = None) -> int:
    """Delete portfolio (and its stocks) by name, scoped to user. Returns 0/1."""
    with get_db_context() as db:
        stmt = select(DBUserPortfolio.id).where(DBUserPortfolio.name == name)
        if user_id is not None:
            stmt = stmt.where(DBUserPortfolio.user_id == user_id)
        row = db.execute(stmt).scalars().first()
        if not row:
            return 0
        db.execute(
            delete(DBUserPortfolioStock).where(DBUserPortfolioStock.portfolio_id == row)
        )
        db.execute(delete(DBUserPortfolio).where(DBUserPortfolio.id == row))
    return 1


# ---------------------------------------------------------------------------
# Daily allocation refresh
# ---------------------------------------------------------------------------

def refresh_portfolio_allocation(
    defn: Dict[str, Any],
    as_of_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Compute and store today's PortfolioAllocation for a user-defined portfolio.

    Uses only the stocks defined in `defn["symbols"]` and the strategy/top_n
    set by the user.  The result is written to the `portfolio_allocations` table
    keyed by (date, strategy_type = portfolio name) so each user portfolio has
    its own isolated allocation row set.

    Args:
        defn: portfolio definition dict {name, symbols, strategy, regime_mode, top_n}
        as_of_date: date to compute for; defaults to the latest scoring date.

    Returns:
        dict with keys:
            "date"             – date used
            "inserted"         – rows written to DB
            "deleted"          – old rows replaced
            "allocation_df"    – pandas DataFrame with symbol, weight columns
            "error"            – str or None
    """
    # Lazy import to avoid circular deps / slow startup at module import time
    import pandas as pd
    from services.portfolio import (
        get_latest_scoring_date,
        construct_equal_weight_portfolio,
        construct_inverse_vol_portfolio,
        is_equal_weight_strategy,
        load_ranked_stocks,
        store_portfolio,
        STRATEGY_EQUAL_WEIGHT,
        STRATEGY_INVERSE_VOL,
    )

    name: str = (defn.get("name") or "").strip()
    symbols: List[str] = defn.get("symbols") or []
    strategy: str = defn.get("strategy") or STRATEGY_EQUAL_WEIGHT
    top_n: int = int(defn.get("top_n") or 20)

    if not name:
        return {"date": None, "inserted": 0, "deleted": 0, "allocation_df": pd.DataFrame(), "error": "Portfolio has no name"}

    if as_of_date is None:
        as_of_date = get_latest_scoring_date()
    if as_of_date is None:
        return {
            "date": None,
            "inserted": 0,
            "deleted": 0,
            "allocation_df": pd.DataFrame(),
            "error": "No scoring date found in DB. Run the scoring engine first.",
        }

    selected = symbols if symbols else None
    try:
        if strategy == "custom" and "weights" in defn:
            custom_weights = defn.get("weights", {})
            ranked_all = load_ranked_stocks(as_of_date, selected_symbols=selected)
            if not ranked_all.empty:
                alloc_list = []
                for _, r in ranked_all.iterrows():
                    sym = r["symbol"]
                    if sym in custom_weights:
                        alloc_list.append({"stock_id": r["stock_id"], "weight": float(custom_weights[sym])})
                alloc_df = pd.DataFrame(alloc_list)
                if not alloc_df.empty:
                    tot = alloc_df["weight"].sum()
                    if tot > 0:
                        alloc_df["weight"] = alloc_df["weight"] / tot
            else:
                alloc_df = pd.DataFrame()
        elif is_equal_weight_strategy(strategy):
            alloc_df = construct_equal_weight_portfolio(
                as_of_date, top_n=top_n, selected_symbols=selected
            )
        else:
            alloc_df = construct_inverse_vol_portfolio(
                as_of_date, top_n=top_n, selected_symbols=selected
            )
    except Exception as exc:
        return {
            "date": as_of_date,
            "inserted": 0,
            "deleted": 0,
            "allocation_df": pd.DataFrame(),
            "error": str(exc),
        }

    if alloc_df.empty:
        return {
            "date": as_of_date,
            "inserted": 0,
            "deleted": 0,
            "allocation_df": pd.DataFrame(),
            "error": "No scored stocks found for the selected symbols and date.",
        }

    # Use portfolio name as the strategy_type key so each user portfolio is
    # stored independently in portfolio_allocations.
    strategy_key = f"user::{name}"
    inserted, deleted = store_portfolio(as_of_date, alloc_df, strategy_key)

    # Enrich with symbol names for display
    ranked = load_ranked_stocks(as_of_date, selected_symbols=selected)
    if not ranked.empty and "stock_id" in alloc_df.columns:
        merged = alloc_df.merge(
            ranked[["stock_id", "symbol", "composite_score", "volatility_score"]],
            on="stock_id",
            how="left",
        )
        display_df = merged[
            ["stock_id", "symbol", "weight", "composite_score", "volatility_score"]
        ].sort_values("weight", ascending=False).reset_index(drop=True)
    else:
        display_df = alloc_df.copy()

    return {
        "date": as_of_date,
        "inserted": inserted,
        "deleted": deleted,
        "allocation_df": display_df,
        "error": None,
    }


__all__ = [
    "save_user_portfolio",
    "save_portfolios",
    "load_user_portfolios",
    "delete_user_portfolio_by_name",
    "refresh_portfolio_allocation",
]
