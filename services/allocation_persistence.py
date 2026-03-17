"""services/allocation_persistence.py

Persistence layer for monthly portfolio allocations and metrics snapshots.

PURPOSE
-------
Every time a monthly rebalance runs (hybrid auto-portfolio or any user strategy),
this service:

1. Writes one MonthlyAllocation row per stock in the portfolio.
2. Writes one PortfolioMetrics row summarising performance + stability.

Both operations are INSERT-only (no overwrites of historical records).
If a record for the same (portfolio_type, portfolio_name, rebalance_date, stock_id)
already exists, it is silently skipped (idempotent via INSERT … ON CONFLICT DO NOTHING
for SQLite/PostgreSQL, or via pre-check for other drivers).

FUTURE EXPANSION
----------------
The portfolio_name column supports multi-user scenarios: pass the user's name or ID
as portfolio_name to namespace their allocations separately from the system auto-hybrid.

PUBLIC API
----------
    save_monthly_allocation(portfolio_type, rebalance_date, weights, ...)
    save_portfolio_metrics(portfolio_type, rebalance_date, metrics, stability, ...)
    load_monthly_allocations(portfolio_type, portfolio_name, start_date, end_date)
    load_portfolio_metrics_history(portfolio_type, portfolio_name)
"""

import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

from app.database import get_db_context
from app.models import MonthlyAllocation, PortfolioMetrics

_logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_monthly_allocation(
    portfolio_type: str,
    rebalance_date: date,
    weights: Dict[int, float],          # stock_id -> weight
    user_id: Optional[int] = None,
    portfolio_name: Optional[str] = None,
    forced: bool = False,
    optimal_n_used: Optional[int] = None,
) -> int:
    """Persist stock weights for a monthly rebalance.

    Parameters
    ----------
    portfolio_type  : Strategy identifier, e.g. 'auto_diversified_hybrid'.
    rebalance_date  : The month-end date on which this allocation was built.
    weights         : Dict mapping stock_id (int) → weight (float, sums to ~1).
    portfolio_name  : Optional human label (for user portfolios / multi-user).
    forced          : True if this rebalance bypassed the monthly guard.

    Returns
    -------
    int : Number of rows newly inserted (0 if all already existed).
    """
    if not weights:
        _logger.warning("save_monthly_allocation called with empty weights — skipping.")
        return 0

    inserted = 0
    try:
        with get_db_context() as db:
            for stock_id, weight in weights.items():
                row = MonthlyAllocation(
                    user_id=user_id,
                    portfolio_type=portfolio_type,
                    portfolio_name=portfolio_name,
                    rebalance_date=rebalance_date,
                    stock_id=int(stock_id),
                    weight=float(weight),
                    forced=forced,
                    optimal_n_used=optimal_n_used,
                    created_at=datetime.utcnow(),
                )
                try:
                    db.add(row)
                    db.flush()   # catch constraint violations early
                    inserted += 1
                except IntegrityError:
                    db.rollback()
                    _logger.debug(
                        "Allocation already exists for (%s, %s, %s, %s) — skipping.",
                        portfolio_type, portfolio_name, rebalance_date, stock_id,
                    )
    except Exception as exc:
        _logger.error("save_monthly_allocation failed: %s", exc)
        raise

    _logger.info(
        "save_monthly_allocation: %d/%d rows inserted (%s | %s | %s)",
        inserted, len(weights), portfolio_type, portfolio_name, rebalance_date,
    )
    return inserted


def save_portfolio_metrics(
    portfolio_type: str,
    rebalance_date: date,
    metrics: Dict[str, Any],
    stability: Optional[Dict[str, Any]] = None,
    rating: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
    portfolio_name: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    forced: bool = False,
) -> bool:
    """Persist a performance + stability snapshot for one rebalance.

    Parameters
    ----------
    portfolio_type  : Strategy identifier.
    rebalance_date  : Month-end date for this snapshot.
    metrics         : Dict from _compute_metrics_from_returns / backtest summary.
                      Keys: CAGR, Volatility, Sharpe, Max Drawdown, Calmar,
                            Sortino, Total Return.
    stability       : Dict from compute_rolling_stability().
    rating          : Dict from rate_portfolios()[name].
    portfolio_name  : Optional human label.
    context         : Optional dict with n_stocks, expected_volatility,
                      expected_sharpe, avg_pairwise_corr.
    forced          : Whether this was a forced rebalance.

    Returns
    -------
    bool : True if row was inserted, False if it already existed.
    """
    stab = stability or {}
    rat  = rating or {}
    ctx  = context or {}
    comps = stab.get("components", {})

    def _f(d: dict, *keys) -> Optional[float]:
        """Try multiple key spellings and return float or None."""
        for k in keys:
            v = d.get(k)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return None

    row = PortfolioMetrics(
        user_id=user_id,
        portfolio_type=portfolio_type,
        portfolio_name=portfolio_name,
        rebalance_date=rebalance_date,
        forced=forced,
        created_at=datetime.utcnow(),
        # Performance
        cagr=_f(metrics, "CAGR"),
        volatility=_f(metrics, "Volatility"),
        sharpe=_f(metrics, "Sharpe"),
        max_drawdown=_f(metrics, "Max Drawdown"),
        calmar=_f(metrics, "Calmar"),
        sortino=_f(metrics, "Sortino"),
        total_return=_f(metrics, "Total Return"),
        # Stability
        stability_score=_f(stab, "stability_score"),
        stability_grade=stab.get("grade"),
        sharpe_consistency_score=_f(comps, "Sharpe Consistency (25%)"),
        drawdown_stability_score=_f(comps, "Drawdown Stability (25%)"),
        turnover_score=_f(comps, "Turnover Penalty (20%)"),
        corr_stability_score=_f(comps, "Correlation Stability (20%)"),
        regime_robustness_score=_f(comps, "Regime Robustness (10%)"),
        # Rating
        composite_rating=_f(rat, "rating_score"),
        composite_grade=rat.get("grade"),
        # Construction context
        n_stocks=int(ctx["n_stocks"]) if ctx.get("n_stocks") is not None else None,
        expected_volatility=_f(ctx, "expected_volatility"),
        expected_sharpe=_f(ctx, "expected_sharpe"),
        avg_pairwise_corr=_f(ctx, "avg_pairwise_corr"),
    )

    try:
        with get_db_context() as db:
            db.add(row)
            _logger.info(
                "save_portfolio_metrics: inserted snapshot (%s | %s | %s)  "
                "Sharpe=%.3f  Stability=%.1f",
                portfolio_type, portfolio_name, rebalance_date,
                row.sharpe or 0, row.stability_score or 0,
            )
            db.commit()
        return True
    except IntegrityError:
        _logger.info(
            "Metrics snapshot already exists for (%s, %s, %s) — skipping.",
            portfolio_type, portfolio_name, rebalance_date,
        )
        return False
    except Exception as exc:
        _logger.error("save_portfolio_metrics failed: %s", exc)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Read helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_monthly_allocations(
    portfolio_type: str,
    portfolio_name: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Load historical monthly allocations as a DataFrame.

    Returns columns: rebalance_date, stock_id, weight, portfolio_name, forced, created_at.
    Sorted by rebalance_date ASC, weight DESC.
    """
    try:
        with get_db_context() as db:
            q = select(MonthlyAllocation).where(
                MonthlyAllocation.portfolio_type == portfolio_type
            )
            if portfolio_name is not None:
                q = q.where(MonthlyAllocation.portfolio_name == portfolio_name)
            if start_date:
                q = q.where(MonthlyAllocation.rebalance_date >= start_date)
            if end_date:
                q = q.where(MonthlyAllocation.rebalance_date <= end_date)
            q = q.order_by(MonthlyAllocation.rebalance_date, MonthlyAllocation.weight.desc())
            rows = db.execute(q).scalars().all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "rebalance_date": r.rebalance_date,
                "stock_id": r.stock_id,
                "weight": r.weight,
                "portfolio_name": r.portfolio_name,
                "forced": r.forced,
                "created_at": r.created_at,
            }
            for r in rows
        ])
    except Exception as exc:
        _logger.error("load_monthly_allocations failed: %s", exc)
        return pd.DataFrame()


def load_portfolio_metrics_history(
    portfolio_type: str,
    portfolio_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load all historical metric snapshots for a portfolio strategy.

    Returns a DataFrame sorted by rebalance_date ASC with all metric columns.
    """
    try:
        with get_db_context() as db:
            q = select(PortfolioMetrics).where(
                PortfolioMetrics.portfolio_type == portfolio_type
            )
            if portfolio_name is not None:
                q = q.where(PortfolioMetrics.portfolio_name == portfolio_name)
            q = q.order_by(PortfolioMetrics.rebalance_date)
            rows = db.execute(q).scalars().all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "rebalance_date":       r.rebalance_date,
                "portfolio_name":       r.portfolio_name,
                "cagr":                 r.cagr,
                "volatility":           r.volatility,
                "sharpe":               r.sharpe,
                "max_drawdown":         r.max_drawdown,
                "calmar":               r.calmar,
                "sortino":              r.sortino,
                "total_return":         r.total_return,
                "stability_score":      r.stability_score,
                "stability_grade":      r.stability_grade,
                "composite_rating":     r.composite_rating,
                "composite_grade":      r.composite_grade,
                "n_stocks":             r.n_stocks,
                "expected_volatility":  r.expected_volatility,
                "expected_sharpe":      r.expected_sharpe,
                "avg_pairwise_corr":    r.avg_pairwise_corr,
                "forced":               r.forced,
                "created_at":           r.created_at,
            }
            for r in rows
        ])
    except Exception as exc:
        _logger.error("load_portfolio_metrics_history failed: %s", exc)
        return pd.DataFrame()


def get_latest_metrics_snapshot(
    portfolio_type: str,
    portfolio_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the most recent PortfolioMetrics record as a dict, or None."""
    df = load_portfolio_metrics_history(portfolio_type, portfolio_name)
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


__all__ = [
    "save_monthly_allocation",
    "save_portfolio_metrics",
    "load_monthly_allocations",
    "load_portfolio_metrics_history",
    "get_latest_metrics_snapshot",
]
