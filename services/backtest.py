"""
Phase 6: Backtesting engine.

Simulates monthly-rebalanced portfolios using historical scores and prices.
No external backtest libraries; transparent, educational logic.

Look-ahead bias avoidance:
- Rebalance dates are month-ends from the prices table only.
- For each rebalance we use get_latest_score_date_on_or_before(reb_date): only
  scores with date <= rebalance date are used to construct the portfolio.
- Period returns use only prices in [start_date, end_date]; no future prices.
- Transaction cost is applied at rebalance using old vs new weights known at that time.
- Volatility targeting (optional): leverage_t uses rolling vol through t-1 only (lagged).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import math
import bisect
from functools import lru_cache
import pandas as pd
import numpy as np
from sqlalchemy import select
from datetime import date, timedelta
from typing import Any, Optional

from app.database import get_db_context
from app.models import Price, Score

from services.portfolio import (
    construct_equal_weight_portfolio,
    construct_inverse_vol_portfolio,
    is_equal_weight_strategy,
    load_ranked_stocks,
    load_ranked_stocks_batch,
)


# Transaction cost: 0.2% applied to turnover (sum of absolute weight changes)
TRANSACTION_COST_RATE = 0.002

# Volatility targeting (optional): scale exposure to target annual vol
TARGET_VOL_DEFAULT = 0.15
ROLLING_VOL_WINDOW = 20
LEVERAGE_MIN = 0.5
LEVERAGE_MAX = 1.5

# Strategy names (must match portfolio construction)
STRATEGY_EQUAL_WEIGHT = "equal_weight"
STRATEGY_INVERSE_VOL = "inverse_volatility"


def get_latest_score_date_on_or_before(as_of_date: date) -> Optional[date]:
    """
    Return the most recent score date that is <= as_of_date.
    Used to avoid look-ahead bias: we only use information available at rebalance.
    """
    with get_db_context() as db:
        stmt = (
            select(Score.date)
            .where(Score.date <= as_of_date)
            .order_by(Score.date.desc())
            .limit(1)
        )
        row = db.execute(stmt).scalars().first()
    return row


@lru_cache(maxsize=4)
def _get_all_rebalance_dates_cache(
    latest_price_date: Optional[date],
    latest_score_date: Optional[date],
) -> tuple[list[date], tuple[date, ...]]:
    """Cache the expensive full-history date scan used to derive month-end rebalances."""
    with get_db_context() as db:
        price_dates = db.execute(select(Price.date).distinct()).scalars().all()
        score_dates = tuple(sorted(db.execute(select(Score.date).distinct()).scalars().all()))

    if not price_dates or not score_dates:
        return [], score_dates

    df = pd.DataFrame({"date": pd.to_datetime(price_dates)})
    month_ends = (
        df.groupby([df["date"].dt.year, df["date"].dt.month])["date"]
        .max()
        .sort_values()
        .dt.date
        .tolist()
    )
    return month_ends, score_dates


def get_rebalance_dates(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[date]:
    """
    Return sorted list of month-end trading dates from prices table.
    Only includes dates for which at least one score exists on or before that date.
    start_date: only return rebalance dates on or after this date.
    end_date: only return rebalance dates on or before this date.
    """
    with get_db_context() as db:
        latest_price_date = db.execute(select(Price.date).order_by(Price.date.desc()).limit(1)).scalars().first()
        latest_score_date = db.execute(select(Score.date).order_by(Score.date.desc()).limit(1)).scalars().first()

    month_ends, score_dates = _get_all_rebalance_dates_cache(latest_price_date, latest_score_date)
    if not month_ends or not score_dates:
        return []

    # Filter: keep only rebalance dates where we have a score on or before that date
    rebalance_dates = []
    for d in month_ends:
        # Efficiently find if there is a score date <= d
        idx = bisect.bisect_right(score_dates, d)
        if idx > 0:
            if start_date is not None and d < start_date:
                continue
            if end_date is not None and d > end_date:
                continue
            rebalance_dates.append(d)
    
    return sorted(rebalance_dates)


def construct_portfolio_for_date(
    rebalance_date: date,
    strategy: str,
    top_n: int = 20,
    selected_symbols: list[str] | None = None,
    custom_weights: dict[str, float] | None = None,
    score_date_cache: dict[date, Optional[date]] | None = None,
    data_cache: pd.DataFrame | None = None,
) -> dict[int, float]:
    """
    Build portfolio weights for a rebalance date using only information available then.
    Uses latest score date on or before rebalance_date (no look-ahead).
    Returns dict stock_id -> weight (sum = 1).
    """
    score_date = score_date_cache.get(rebalance_date) if score_date_cache is not None else get_latest_score_date_on_or_before(rebalance_date)
    if score_date is None:
        return {}

    if strategy == "custom" and custom_weights:
        df_ranked = load_ranked_stocks(score_date, selected_symbols=selected_symbols, data_cache=data_cache)
        if df_ranked.empty:
            return {}
        alloc_list = []
        for _, r in df_ranked.iterrows():
            sym = r["symbol"]
            if sym in custom_weights:
                alloc_list.append({"stock_id": r["stock_id"], "weight": float(custom_weights[sym])})
        df = pd.DataFrame(alloc_list)
        if not df.empty:
            tot = df["weight"].sum()
            if tot > 0:
                df["weight"] = df["weight"] / tot
    elif is_equal_weight_strategy(strategy):
        df = construct_equal_weight_portfolio(score_date, top_n=top_n, selected_symbols=selected_symbols, data_cache=data_cache)
    elif strategy == STRATEGY_INVERSE_VOL:
        df = construct_inverse_vol_portfolio(score_date, top_n=top_n, selected_symbols=selected_symbols, data_cache=data_cache)
    else:
        return {}

    if df.empty:
        return {}

    return dict(zip(df["stock_id"].astype(int), df["weight"].astype(float)))


def _load_prices_for_stocks(
    stock_ids: list[int],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load (date, stock_id, close) for given stocks and date range. Pivot to date index, columns = stock_id."""
    if not stock_ids:
        return pd.DataFrame()

    with get_db_context() as db:
        stmt = (
            select(Price.date, Price.stock_id, Price.close)
            .where(
                Price.stock_id.in_(stock_ids),
                Price.date >= start_date,
                Price.date <= end_date,
            )
            .order_by(Price.date)
        )
        rows = db.execute(stmt).all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "stock_id", "close"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Pivot: index=date, columns=stock_id, values=close
    wide = df.pivot(index="date", columns="stock_id", values="close")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def compute_period_returns(
    start_date: date,
    end_date: date,
    weights: dict[int, float],
    price_cache: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Compute daily portfolio returns for [start_date, end_date] using given weights.
    Daily return = sum(weight_s * daily_return_s). Stocks with missing price are skipped
    (weight effectively 0 for that day). Returns Series with date index.
    """
    if not weights:
        return pd.Series(dtype=float)

    stock_ids = list(weights.keys())
    
    if price_cache is not None:
        # Filter cache for requested stocks and dates
        mask = (price_cache.index.date >= start_date) & (price_cache.index.date <= end_date)
        prices = price_cache.loc[mask, price_cache.columns.intersection(stock_ids)]
    else:
        prices = _load_prices_for_stocks(stock_ids, start_date, end_date)
        
    if prices.empty or prices.shape[0] < 2:
        return pd.Series(dtype=float)

    # Daily returns per stock
    returns = prices.pct_change().dropna(how="all")
    if returns.empty:
        return pd.Series(dtype=float)

    # Align weights: only stocks we have data for
    w = pd.Series(weights)
    common = returns.columns.intersection(w.index)
    if len(common) == 0:
        return pd.Series(dtype=float)

    w_aligned = w.reindex(common).fillna(0)
    w_aligned = w_aligned / w_aligned.sum() if w_aligned.sum() > 0 else w_aligned

    # Weighted daily return
    portfolio_return = (returns[common] * w_aligned).sum(axis=1)
    portfolio_return.name = "daily_return"
    return portfolio_return


def _apply_volatility_targeting(
    daily_returns: pd.Series,
    target_vol: float,
    window: int = ROLLING_VOL_WINDOW,
    leverage_min: float = LEVERAGE_MIN,
    leverage_max: float = LEVERAGE_MAX,
) -> pd.Series:
    """
    Scale daily returns to target annual volatility using lagged rolling vol.
    leverage_t = f(vol_{t-1}): no look-ahead. Returns scaled daily return series.
    """
    if daily_returns.empty or len(daily_returns) < 2:
        return daily_returns.copy()
    # Rolling std of returns (window ending at each t)
    rolling_std = daily_returns.rolling(window, min_periods=1).std()
    # Lag by 1: at t use vol estimated through t-1 (no look-ahead)
    rolling_std_lagged = rolling_std.shift(1)
    # Annualize
    rolling_vol = rolling_std_lagged * math.sqrt(252)
    # Leverage: target_vol / est_vol; where vol is 0 or NaN use 1.0 (no scaling)
    leverage = target_vol / rolling_vol
    leverage = leverage.where(rolling_vol.gt(0)).fillna(1.0)
    leverage = leverage.clip(lower=leverage_min, upper=leverage_max)
    return (leverage * daily_returns).reindex(daily_returns.index).fillna(0.0)


def calculate_transaction_cost(
    old_weights: dict[int, float],
    new_weights: dict[int, float],
) -> float:
    """
    Cost = TRANSACTION_COST_RATE * sum(|new_weight - old_weight|).
    All stock_ids from both dicts are considered; missing = 0.
    """
    all_ids = set(old_weights) | set(new_weights)
    if not all_ids:
        return 0.0

    total_turnover = 0.0
    for sid in all_ids:
        ow = old_weights.get(sid, 0.0)
        nw = new_weights.get(sid, 0.0)
        total_turnover += abs(nw - ow)

    return TRANSACTION_COST_RATE * total_turnover


def run_backtest(
    strategy: str = STRATEGY_EQUAL_WEIGHT,
    top_n: int = 20,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    target_vol: Optional[float] = None,
    rolling_vol_window: int = ROLLING_VOL_WINDOW,
    leverage_min: float = LEVERAGE_MIN,
    leverage_max: float = LEVERAGE_MAX,
    selected_symbols: list[str] | None = None,
    custom_weights: dict[str, float] | None = None,
    global_price_cache: pd.DataFrame | None = None,
    global_ranked_cache: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run full backtest: monthly rebalance, weighted daily returns, transaction costs.
    Returns (equity_curve DataFrame, summary dict).
    No look-ahead: at each rebalance we use only scores on or before that date.

    start_date: only rebalance on or after this date (e.g. date(2020, 1, 1)).
    end_date: only rebalance on or before this date.
    target_vol: if set (e.g. 0.15), scale returns to target annual vol using lagged
        20d rolling vol; leverage clipped to [leverage_min, leverage_max].
    """
    rebalance_dates = get_rebalance_dates(start_date=start_date, end_date=end_date)
    if len(rebalance_dates) < 2:
        return (
            pd.DataFrame(columns=["date", "daily_return", "cumulative_return", "drawdown"]),
            {},
        )

    # Optimization: pre-calculate score dates for all rebalance dates
    with get_db_context() as db:
        score_stmt = select(Score.date).distinct().order_by(Score.date)
        all_score_dates = [r[0] for r in db.execute(score_stmt).all()]
    
    score_date_cache = {}
    if all_score_dates:
        for rb in rebalance_dates:
            idx = bisect.bisect_right(all_score_dates, rb)
            score_date_cache[rb] = all_score_dates[idx-1] if idx > 0 else None

    # Optimization: Batch load all rankings for the involved dates
    if global_ranked_cache is not None:
        ranked_data_cache = global_ranked_cache
    else:
        unique_score_dates = list(set(d for d in score_date_cache.values() if d is not None))
        ranked_data_cache = load_ranked_stocks_batch(unique_score_dates, selected_symbols=selected_symbols)

    # First pass: find all stocks involved to batch load prices
    involved_stock_ids = set()
    for reb_date in rebalance_dates:
        new_weights = construct_portfolio_for_date(
            reb_date, strategy, top_n=top_n, 
            selected_symbols=selected_symbols, 
            custom_weights=custom_weights,
            score_date_cache=score_date_cache,
            data_cache=ranked_data_cache
        )
        involved_stock_ids.update(new_weights.keys())

    # Batch load ALL prices for involved stocks
    price_cache_to_use = None
    if global_price_cache is not None:
        price_cache_to_use = global_price_cache
    elif involved_stock_ids:
        # Find start/end range for prices
        p_start = min(rebalance_dates)
        # Last period: extend to last available price date
        with get_db_context() as db:
            p_end = db.execute(select(Price.date).order_by(Price.date.desc()).limit(1)).scalars().first()
        if p_end:
            price_cache_to_use = _load_prices_for_stocks(list(involved_stock_ids), p_start, p_end)
    else:
        # Need p_end for the loop anyway
        with get_db_context() as db:
            p_end = db.execute(select(Price.date).order_by(Price.date.desc()).limit(1)).scalars().first()

    # Build daily return series over all periods
    daily_returns_list: list[pd.Series] = []
    old_weights: dict[int, float] = {}

    for i, reb_date in enumerate(rebalance_dates):
        # Responsiveness check for Streamlit/Manual interruption
        import time
        time.sleep(0.0) # Yield slightly to signal handler
        
        new_weights = construct_portfolio_for_date(
            reb_date, strategy, top_n=top_n, 
            selected_symbols=selected_symbols, 
            custom_weights=custom_weights,
            score_date_cache=score_date_cache,
            data_cache=ranked_data_cache
        )
        if not new_weights:
            continue

        start_d = reb_date
        if i + 1 < len(rebalance_dates):
            end_d = rebalance_dates[i + 1]
        else:
            # use globally found p_end if possible, otherwise re-fetch
            if 'p_end' not in locals() or p_end is None:
                with get_db_context() as db:
                     p_end = db.execute(select(Price.date).order_by(Price.date.desc()).limit(1)).scalars().first()
            end_d = p_end if p_end else reb_date

        period_returns = compute_period_returns(start_d, end_d, new_weights, price_cache=price_cache_to_use)
        if period_returns.empty:
            continue

        # Transaction cost on first day of period
        cost = calculate_transaction_cost(old_weights, new_weights)
        if cost > 0 and len(period_returns) > 0:
            period_returns = period_returns.copy()
            period_returns.iloc[0] = period_returns.iloc[0] - cost

        daily_returns_list.append(period_returns)
        old_weights = new_weights

    if not daily_returns_list:
        return (
            pd.DataFrame(columns=["date", "daily_return", "cumulative_return", "drawdown"]),
            {},
        )

    # Concatenate and drop duplicate indices (overlap at rebalance: keep last = new weights)
    full_returns = pd.concat(daily_returns_list, axis=0)
    full_returns = full_returns[~full_returns.index.duplicated(keep="last")]
    full_returns = full_returns.sort_index()

    # Optional volatility targeting: scale returns using lagged rolling vol (no look-ahead)
    if target_vol is not None and target_vol > 0:
        full_returns = _apply_volatility_targeting(
            full_returns,
            target_vol=target_vol,
            window=rolling_vol_window,
            leverage_min=leverage_min,
            leverage_max=leverage_max,
        )

    # Equity curve
    cum = (1 + full_returns).cumprod()
    cumulative_return = cum - 1.0
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    dates = full_returns.index
    if hasattr(dates, "date"):
        date_vals = dates.date
    else:
        date_vals = pd.to_datetime(dates).date
    equity_curve = pd.DataFrame({
        "date": date_vals,
        "daily_return": full_returns.values,
        "cumulative_return": cumulative_return.values,
        "drawdown": drawdown.values,
    })

    # Metrics
    total_return = float(cumulative_return.iloc[-1]) if len(cumulative_return) > 0 else 0.0
    n_days = len(full_returns)
    n_years = n_days / 252.0 if n_days else 0
    cagr = (1 + total_return) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0
    ann_vol = full_returns.std() * (252 ** 0.5) if len(full_returns) > 1 else 0.0
    sharpe = (full_returns.mean() * 252) / ann_vol if ann_vol and ann_vol > 0 else 0.0
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Win rate: % of months with positive return
    try:
        monthly = (1 + full_returns).resample("ME").prod() - 1
    except TypeError:
        monthly = (1 + full_returns).resample("M").prod() - 1
    win_rate = float((monthly > 0).mean()) if len(monthly) > 0 else 0.0

    # Calculate Average Pairwise Correlation of holdings (Diversification metric)
    avg_corr = 0.5
    try:
        # Use existing cache if available
        if price_cache_to_use is not None and not price_cache_to_use.empty:
            prices = price_cache_to_use.loc[full_returns.index.min():full_returns.index.max()]
            if not prices.empty and prices.shape[1] > 1:
                corr_matrix = prices.pct_change().corr()
                upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                avg_corr = float(corr_matrix.values[upper_tri_indices].mean())
        else:
            all_stock_ids = set()
            for i, reb_date in enumerate(rebalance_dates):
                weights = construct_portfolio_for_date(
                    reb_date, strategy, top_n=top_n, 
                    selected_symbols=selected_symbols, 
                    custom_weights=custom_weights,
                    score_date_cache=score_date_cache
                )
                all_stock_ids.update(weights.keys())
            
            if all_stock_ids:
                prices = _load_prices_for_stocks(list(all_stock_ids), full_returns.index.min().date(), full_returns.index.max().date())
                if not prices.empty and prices.shape[1] > 1:
                    corr_matrix = prices.pct_change().corr()
                    # Average of off-diagonal elements
                    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                    avg_corr = float(corr_matrix.values[upper_tri_indices].mean())
    except Exception:
        pass

    summary = {
        "CAGR": cagr,
        "Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Total Return": total_return,
        "Win Rate": win_rate,
        "avg_pairwise_corr": avg_corr,
    }

    # Print validation
    start_date_display = full_returns.index.min()
    end_date_display = full_returns.index.max()
    if hasattr(start_date_display, "date"):
        start_date_display = start_date_display.date()
    if hasattr(end_date_display, "date"):
        end_date_display = end_date_display.date()

    print()
    print("--- Backtest results ---")
    print(f"  Strategy:   {strategy}")
    if target_vol is not None:
        print(f"  Vol targeting: ON (target={target_vol:.0%}, leverage [{leverage_min}, {leverage_max}])")
    print(f"  Start date: {start_date_display}")
    print(f"  End date:   {end_date_display}")
    print(f"  CAGR:       {cagr:.4f}")
    print(f"  Volatility: {ann_vol:.4f}")
    print(f"  Sharpe:     {sharpe:.4f}")
    print(f"  Max Drawdown: {max_dd:.4f}")
    print(f"  Total Return: {total_return:.4f}")
    print("---")
    print("First 5 rows of equity curve:")
    print(equity_curve.head().to_string())
    print()

    return equity_curve, summary


def run_both_strategies(
    top_n: int = 20,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> None:
    """Run backtest for equal_weight and inverse_volatility; print comparison."""
    for strategy in [STRATEGY_EQUAL_WEIGHT, STRATEGY_INVERSE_VOL]:
        run_backtest(
            strategy=strategy,
            top_n=top_n,
            start_date=start_date,
            end_date=end_date,
        )


if __name__ == "__main__":
    # Optional: set backtest window (None = use all data)
    BACKTEST_START_DATE = date(2022, 1, 1)  # e.g. from 2022-01-01; set to None for no start filter
    BACKTEST_END_DATE = None                 # e.g. date(2024, 12, 31); set to None for no end filter
    # Optional: target 15% annual vol (None = no vol targeting)
    TARGET_VOL = 0.15  # set to None to disable
    run_backtest(
        strategy=STRATEGY_EQUAL_WEIGHT,
        top_n=20,
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE,
        target_vol=TARGET_VOL,
    )
