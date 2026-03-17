"""services/sector_analytics.py

Sector-level analytics for the Portfolio Comparison Lab.

Provides:
- Sector mapping from DB (Stock.sector)
- Sector relative performance (CAGR, Sharpe, Max DD per sector)
- Portfolio sector exposure breakdown
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging
import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.database import get_db_context
from app.models import Price, Score, Stock

_logger = logging.getLogger(__name__)

ANNUALIZE = 252


# ─────────────────────────────────────────────────────────────────────────────
# Sector Mapping
# ─────────────────────────────────────────────────────────────────────────────

def get_sector_map() -> Dict[int, str]:
    """Return {stock_id: sector} for all stocks that have a sector assigned."""
    with get_db_context() as db:
        rows = db.execute(
            select(Stock.id, Stock.sector).where(Stock.sector.isnot(None))
        ).all()
    return {r[0]: r[1] for r in rows}


def get_symbol_sector_map() -> Dict[str, str]:
    """Return {symbol: sector} for all stocks that have a sector assigned."""
    with get_db_context() as db:
        rows = db.execute(
            select(Stock.symbol, Stock.sector).where(Stock.sector.isnot(None))
        ).all()
    return {r[0]: r[1] for r in rows}


def get_stocks_by_sector() -> Dict[str, List[Dict[str, Any]]]:
    """Return {sector: [{stock_id, symbol, name}, ...]} grouped by sector."""
    with get_db_context() as db:
        rows = db.execute(
            select(Stock.id, Stock.symbol, Stock.name, Stock.sector)
            .where(Stock.sector.isnot(None))
            .order_by(Stock.sector, Stock.symbol)
        ).all()
    result: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        sector = r[3]
        if sector not in result:
            result[sector] = []
        result[sector].append({
            "stock_id": r[0],
            "symbol": r[1],
            "name": r[2],
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sector Relative Performance
# ─────────────────────────────────────────────────────────────────────────────

def compute_sector_relative_performance(
    start_date: date,
    end_date: date,
    scoring_date: Optional[date] = None,
) -> pd.DataFrame:
    """Compute per-sector performance metrics over [start_date, end_date].

    For each sector, builds an equal-weight portfolio of all sector members
    and computes CAGR, Sharpe, Max Drawdown, and average composite score.

    Returns DataFrame with columns:
        sector, n_stocks, cagr, sharpe, max_drawdown, volatility, avg_score
    Sorted by CAGR descending.
    """
    sector_map = get_sector_map()
    if not sector_map:
        return pd.DataFrame(columns=[
            "sector", "n_stocks", "cagr", "sharpe", "max_drawdown",
            "volatility", "avg_score"
        ])

    # Group stock IDs by sector
    sector_stocks: Dict[str, List[int]] = {}
    for sid, sector in sector_map.items():
        if sector not in sector_stocks:
            sector_stocks[sector] = []
        sector_stocks[sector].append(sid)

    # Load all prices in range
    all_stock_ids = list(sector_map.keys())
    with get_db_context() as db:
        stmt = (
            select(Price.date, Price.stock_id, Price.close)
            .where(
                Price.stock_id.in_(all_stock_ids),
                Price.date >= start_date,
                Price.date <= end_date,
            )
            .order_by(Price.date)
        )
        price_rows = db.execute(stmt).all()

    if not price_rows:
        return pd.DataFrame(columns=[
            "sector", "n_stocks", "cagr", "sharpe", "max_drawdown",
            "volatility", "avg_score"
        ])

    pdf = pd.DataFrame(price_rows, columns=["date", "stock_id", "close"])
    pdf["date"] = pd.to_datetime(pdf["date"])
    prices_wide = pdf.pivot(index="date", columns="stock_id", values="close").sort_index()
    returns = prices_wide.pct_change().dropna(how="all")

    # Load composite scores if scoring_date provided
    score_map: Dict[int, float] = {}
    if scoring_date is not None:
        with get_db_context() as db:
            score_rows = db.execute(
                select(Score.stock_id, Score.composite_score)
                .where(Score.date == scoring_date)
            ).all()
        score_map = {r[0]: float(r[1]) if r[1] is not None else 0.0 for r in score_rows}

    results: List[Dict[str, Any]] = []
    for sector, stock_ids in sector_stocks.items():
        # Filter to stocks that have return data
        valid_ids = [s for s in stock_ids if s in returns.columns]
        if not valid_ids or len(valid_ids) < 1:
            continue

        # Equal-weight sector return
        sector_ret = returns[valid_ids].mean(axis=1).dropna()
        if sector_ret.empty or len(sector_ret) < 5:
            continue

        n_days = len(sector_ret)
        n_years = n_days / ANNUALIZE

        cum = (1 + sector_ret).cumprod()
        total_ret = float(cum.iloc[-1] - 1.0) if n_days > 0 else 0.0
        cagr = (1 + total_ret) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0
        ann_vol = float(sector_ret.std(ddof=0) * math.sqrt(ANNUALIZE)) if n_days > 1 else 0.0
        sharpe = float(sector_ret.mean() * ANNUALIZE) / ann_vol if ann_vol > 0 else 0.0

        run_max = cum.cummax()
        dd = (cum - run_max) / run_max
        max_dd = float(dd.min()) if n_days > 0 else 0.0

        # Average composite score for this sector
        sector_scores = [score_map.get(s, 0.0) for s in valid_ids if s in score_map]
        avg_score = float(np.mean(sector_scores)) if sector_scores else 0.0

        results.append({
            "sector": sector,
            "n_stocks": len(valid_ids),
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "volatility": ann_vol,
            "avg_score": avg_score,
        })

    if not results:
        return pd.DataFrame(columns=[
            "sector", "n_stocks", "cagr", "sharpe", "max_drawdown",
            "volatility", "avg_score"
        ])

    df = pd.DataFrame(results)
    return df.sort_values("cagr", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Sector Exposure
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_sector_exposure(
    weights: Dict[int, float],
) -> pd.DataFrame:
    """Compute sector allocation breakdown for a portfolio.

    Parameters
    ----------
    weights : dict mapping stock_id -> weight

    Returns
    -------
    DataFrame with columns: sector, weight, n_stocks, pct
    Sorted by weight descending.
    """
    if not weights:
        return pd.DataFrame(columns=["sector", "weight", "n_stocks", "pct"])

    sector_map = get_sector_map()

    sector_weights: Dict[str, float] = {}
    sector_counts: Dict[str, int] = {}
    unclassified_weight = 0.0
    unclassified_count = 0

    for sid, w in weights.items():
        sector = sector_map.get(sid)
        if sector:
            sector_weights[sector] = sector_weights.get(sector, 0.0) + w
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        else:
            unclassified_weight += w
            unclassified_count += 1

    rows = []
    total_w = sum(weights.values())
    for sector in sorted(sector_weights.keys(), key=lambda s: -sector_weights[s]):
        rows.append({
            "sector": sector,
            "weight": sector_weights[sector],
            "n_stocks": sector_counts[sector],
            "pct": sector_weights[sector] / total_w * 100 if total_w > 0 else 0.0,
        })

    if unclassified_count > 0:
        rows.append({
            "sector": "Unclassified",
            "weight": unclassified_weight,
            "n_stocks": unclassified_count,
            "pct": unclassified_weight / total_w * 100 if total_w > 0 else 0.0,
        })

    return pd.DataFrame(rows)
