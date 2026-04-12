"""
Phase 7: Research validation & overfitting control.

Walk-forward testing, weight sensitivity, single-factor and regime analysis.
Uses stored scores from DB for walk-forward and regime (no recalculation).
Custom composite weights used only in research paths for sensitivity/factor tests.
No modification to production scoring logic. No look-ahead: train/test split by time.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from datetime import date, datetime
import os
import bisect
from typing import Any, Optional

import pandas as pd
from sqlalchemy import select
import concurrent.futures

from app.database import get_db_context
from app.models import Price, Score

from services.backtest import (
    STRATEGY_EQUAL_WEIGHT,
    calculate_transaction_cost,
    compute_period_returns,
    get_latest_score_date_on_or_before,
    get_rebalance_dates,
    run_backtest,
)
from services.scoring import load_factor_data


# Default train/test split (fraction of timeline)
TRAIN_FRACTION = 0.60
TEST_FRACTION = 0.40

# Rolling window (years)
ROLLING_TRAIN_YEARS = 3
ROLLING_TEST_YEARS = 1

# Momentum weight range for sensitivity
MOMENTUM_WEIGHT_MIN = 0.15
MOMENTUM_WEIGHT_MAX = 0.35
MOMENTUM_WEIGHT_STEPS = 5

# Overfitting red flags
SHARPE_GAP_WARN = 1.0
SINGLE_FACTOR_DOMINANCE = 0.80


def _compute_composite_with_weights(
    factor_df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Research-only: compute composite_score from factor_df with given weights."""
    if factor_df.empty:
        return factor_df.copy()
    out = factor_df.copy()
    w = weights
    out["composite_score"] = (
        w.get("quality", 0) * out["quality_score"].fillna(0)
        + w.get("growth", 0) * out["growth_score"].fillna(0)
        + w.get("momentum", 0) * out["momentum_score"].fillna(0)
        + w.get("value", 0) * out["value_score"].fillna(0)
    )
    return out


def _rank_cross_sectionally(df: pd.DataFrame) -> pd.DataFrame:
    """Research-only: rank by composite_score per date (dense, descending)."""
    if df.empty or "composite_score" not in df.columns:
        return df.copy()
    out = df.copy()
    out["rank"] = (
        out.groupby("date", sort=True)["composite_score"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    return out


def _get_weights_from_ranked(
    ranked_df: pd.DataFrame,
    score_date: date,
    top_n: int,
    strategy: str,
) -> dict[int, float]:
    """Build equal-weight or inverse-vol portfolio from ranked_df for one score_date."""
    sub = ranked_df[ranked_df["date"] == score_date].sort_values("rank").head(top_n)
    if sub.empty:
        return {}
    if strategy == STRATEGY_EQUAL_WEIGHT:
        n = len(sub)
        return dict(zip(sub["stock_id"].astype(int), [1.0 / n] * n))
    # Inverse vol: weight ∝ 1/(volatility_score + 2)
    if "volatility_score" not in sub.columns:
        n = len(sub)
        return dict(zip(sub["stock_id"].astype(int), [1.0 / n] * n))
    vol = sub["volatility_score"].fillna(0) + 2.0
    inv = 1.0 / vol.clip(lower=1e-8)
    w = (inv / inv.sum()).astype(float)
    return dict(zip(sub["stock_id"].astype(int), w.tolist()))


def _backtest_with_custom_weights(
    factor_weights: dict[str, float],
    strategy: str,
    top_n: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    selected_symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Research-only backtest using custom composite weights (no DB scores).
    Loads factors (optionally filtered), computes composite and rank, then same return/cost logic as backtest.
    """
    # allow filtering the factor universe when requested (research-only)
    # `factor_weights` defines the composite used for ranking
    factor_df = load_factor_data(selected_symbols=selected_symbols)
    if factor_df.empty:
        return (
            pd.DataFrame(columns=["date", "daily_return", "cumulative_return", "drawdown"]),
            {},
        )

    comp = _compute_composite_with_weights(factor_df, factor_weights)
    ranked = _rank_cross_sectionally(comp)
    # Keep volatility for inverse_vol
    if "volatility_score" not in ranked.columns and "volatility_score" in factor_df.columns:
        ranked = ranked.merge(
            factor_df[["stock_id", "date", "volatility_score"]],
            on=["stock_id", "date"],
            how="left",
        )

    rebalance_dates = get_rebalance_dates(start_date=start_date, end_date=end_date)
    if len(rebalance_dates) < 2:
        return (
            pd.DataFrame(columns=["date", "daily_return", "cumulative_return", "drawdown"]),
            {},
        )

    # Optimization: pre-calculate score dates for rebalance mapping
    with get_db_context() as db:
        all_score_dates = [r[0] for r in db.execute(select(Score.date).distinct().order_by(Score.date)).all()]
    
    score_date_cache = {}
    if all_score_dates:
        for rb in rebalance_dates:
            idx = bisect.bisect_right(all_score_dates, rb)
            score_date_cache[rb] = all_score_dates[idx-1] if idx > 0 else None

    # First pass: find all stocks involved to batch load prices
    involved_stock_ids = set()
    for reb_date in rebalance_dates:
        score_date = score_date_cache.get(reb_date)
        if score_date:
            nw = _get_weights_from_ranked(ranked, score_date, top_n, strategy)
            if nw:
                involved_stock_ids.update(nw.keys())

    # Batch load ALL prices for involved stocks
    price_cache = None
    if involved_stock_ids:
        p_start = min(rebalance_dates) if rebalance_dates else start_date
        with get_db_context() as db:
            p_end = db.execute(select(Price.date).order_by(Price.date.desc()).limit(1)).scalars().first()
        if p_end:
            from services.backtest import _load_prices_for_stocks
            price_cache = _load_prices_for_stocks(list(involved_stock_ids), p_start, p_end)

    daily_returns_list: list[pd.Series] = []
    old_weights: dict[int, float] = {}

    for i, reb_date in enumerate(rebalance_dates):
        score_date = score_date_cache.get(reb_date)
        if score_date is None:
            continue
        new_weights = _get_weights_from_ranked(ranked, score_date, top_n, strategy)
        if not new_weights:
            continue

        start_d = reb_date
        if i + 1 < len(rebalance_dates):
            end_d = rebalance_dates[i + 1]
        else:
            end_d = p_end if 'p_end' in locals() and p_end else reb_date

        period_returns = compute_period_returns(start_d, end_d, new_weights, price_cache=price_cache)
        if period_returns.empty:
            continue

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

    full_returns = pd.concat(daily_returns_list, axis=0)
    full_returns = full_returns[~full_returns.index.duplicated(keep="last")]
    full_returns = full_returns.sort_index()

    cum = (1 + full_returns).cumprod()
    cumulative_return = cum - 1.0
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    dates = full_returns.index
    date_vals = dates.date if hasattr(dates, "date") else pd.to_datetime(dates).date
    equity_curve = pd.DataFrame({
        "date": date_vals,
        "daily_return": full_returns.values,
        "cumulative_return": cumulative_return.values,
        "drawdown": drawdown.values,
    })

    total_return = float(cumulative_return.iloc[-1]) if len(cumulative_return) > 0 else 0.0
    n_days = len(full_returns)
    n_years = n_days / 252.0 if n_days else 0
    cagr = (1 + total_return) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0
    ann_vol = full_returns.std() * (252 ** 0.5) if len(full_returns) > 1 else 0.0
    sharpe = (full_returns.mean() * 252) / ann_vol if ann_vol and ann_vol > 0 else 0.0
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    summary = {
        "CAGR": cagr,
        "Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Total Return": total_return,
    }
    return equity_curve, summary


def run_walk_forward(
    strategy: str = STRATEGY_EQUAL_WEIGHT,
    top_n: int = 20,
    use_rolling: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    selected_symbols: list[str] | None = None,
    use_multiprocessing: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward validation. use_rolling=False: single 60/40 train/test split.
    use_rolling=True: rolling 3-year train / 1-year test windows.
    Uses stored scores (no recalculation). Train and test windows do not overlap.
    """
    all_dates = get_rebalance_dates(start_date=start_date, end_date=end_date)
    if len(all_dates) < 4:
        return pd.DataFrame(
            columns=[
                "window_start", "window_end",
                "train_CAGR", "train_Sharpe", "train_MaxDD",
                "test_CAGR", "test_Sharpe", "test_MaxDD",
            ]
        )

    rows: list[dict[str, Any]] = []

    if use_rolling:
        # Rolling: N months train, M months test, slide by 12 months (rebalance dates = month-ends)
        n_train = max(12, int(ROLLING_TRAIN_YEARS * 12))
        n_test = max(6, int(ROLLING_TEST_YEARS * 12))
        step = 12
        i = 0
        if use_multiprocessing:
            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                while i + n_train + n_test <= len(all_dates):
                    train_start = all_dates[i]
                    train_end = all_dates[i + n_train - 1]
                    test_start = all_dates[i + n_train]
                    test_end = all_dates[i + n_train + n_test - 1]

                    f_train = executor.submit(run_backtest, strategy=strategy, top_n=top_n, start_date=train_start, end_date=train_end, selected_symbols=selected_symbols)
                    f_test = executor.submit(run_backtest, strategy=strategy, top_n=top_n, start_date=test_start, end_date=test_end, selected_symbols=selected_symbols)
                    _, train_sum = f_train.result()
                    _, test_sum = f_test.result()

                    if train_sum and test_sum:
                        rows.append({
                            "window_start": train_start,
                            "window_end": test_end,
                            "train_CAGR": train_sum.get("CAGR"),
                            "train_Sharpe": train_sum.get("Sharpe"),
                            "train_MaxDD": train_sum.get("Max Drawdown"),
                            "test_CAGR": test_sum.get("CAGR"),
                            "test_Sharpe": test_sum.get("Sharpe"),
                            "test_MaxDD": test_sum.get("Max Drawdown"),
                        })
                    i += step
        else:
            while i + n_train + n_test <= len(all_dates):
                train_start = all_dates[i]
                train_end = all_dates[i + n_train - 1]
                test_start = all_dates[i + n_train]
                test_end = all_dates[i + n_train + n_test - 1]

                _, train_sum = run_backtest(strategy=strategy, top_n=top_n, start_date=train_start, end_date=train_end, selected_symbols=selected_symbols)
                _, test_sum = run_backtest(strategy=strategy, top_n=top_n, start_date=test_start, end_date=test_end, selected_symbols=selected_symbols)

                if train_sum and test_sum:
                    rows.append({
                        "window_start": train_start,
                        "window_end": test_end,
                        "train_CAGR": train_sum.get("CAGR"),
                        "train_Sharpe": train_sum.get("Sharpe"),
                        "train_MaxDD": train_sum.get("Max Drawdown"),
                        "test_CAGR": test_sum.get("CAGR"),
                        "test_Sharpe": test_sum.get("Sharpe"),
                        "test_MaxDD": test_sum.get("Max Drawdown"),
                    })
                i += step
    else:
        # Single split: first 60% train, last 40% test
        split_idx = max(1, int(len(all_dates) * TRAIN_FRACTION))
        train_start, train_end = all_dates[0], all_dates[split_idx - 1]
        test_start, test_end = all_dates[split_idx], all_dates[-1]

        _, train_sum = run_backtest(strategy=strategy, top_n=top_n, start_date=train_start, end_date=train_end, selected_symbols=selected_symbols)
        _, test_sum = run_backtest(strategy=strategy, top_n=top_n, start_date=test_start, end_date=test_end, selected_symbols=selected_symbols)
        if train_sum and test_sum:
            rows.append({
                "window_start": train_start,
                "window_end": test_end,
                "train_CAGR": train_sum.get("CAGR"),
                "train_Sharpe": train_sum.get("Sharpe"),
                "train_MaxDD": train_sum.get("Max Drawdown"),
                "test_CAGR": test_sum.get("CAGR"),
                "test_Sharpe": test_sum.get("Sharpe"),
                "test_MaxDD": test_sum.get("Max Drawdown"),
            })

    return pd.DataFrame(rows)


def _worker_run_weight_step(step: int, strategy: str, top_n: int, selected_symbols: list[str] | None) -> dict:
    base = {"quality": 0.30, "growth": 0.25, "momentum": 0.25, "value": 0.20}
    other_sum = base["quality"] + base["growth"] + base["value"]
    mom = MOMENTUM_WEIGHT_MIN + (MOMENTUM_WEIGHT_MAX - MOMENTUM_WEIGHT_MIN) * step / MOMENTUM_WEIGHT_STEPS
    scale = (1.0 - mom) / other_sum
    weights = {
        "quality": base["quality"] * scale,
        "growth": base["growth"] * scale,
        "momentum": mom,
        "value": base["value"] * scale,
    }
    _, summary = _backtest_with_custom_weights(weights, strategy, top_n, selected_symbols=selected_symbols)
    return {
        "momentum_weight": round(mom, 3),
        "CAGR": summary.get("CAGR"),
        "Sharpe": summary.get("Sharpe"),
        "Max_Drawdown": summary.get("Max Drawdown"),
    }

def run_weight_sensitivity(
    strategy: str = STRATEGY_EQUAL_WEIGHT,
    top_n: int = 20,
    selected_symbols: list[str] | None = None,
    use_multiprocessing: bool = True,
) -> pd.DataFrame:
    """
    Vary momentum weight from MOMENTUM_WEIGHT_MIN to MOMENTUM_WEIGHT_MAX;
    scale quality, growth, value so total = 1. Run backtest for each. Research-only weights.
    """
    rows = []
    
    if use_multiprocessing:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(_worker_run_weight_step, s, strategy, top_n, selected_symbols): s for s in range(MOMENTUM_WEIGHT_STEPS + 1)}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    rows.append(res)
    else:
        for s in range(MOMENTUM_WEIGHT_STEPS + 1):
            res = _worker_run_weight_step(s, strategy, top_n, selected_symbols)
            if res:
                rows.append(res)
                
    rows.sort(key=lambda x: x["momentum_weight"])
    return pd.DataFrame(rows)


def _worker_run_single_factor(idx: int, name: str, w: dict, top_n: int, selected_symbols: list[str] | None) -> dict:
    _, summary = _backtest_with_custom_weights(w, STRATEGY_EQUAL_WEIGHT, top_n, selected_symbols=selected_symbols)
    return {
        "factor": name,
        "CAGR": summary.get("CAGR"),
        "Sharpe": summary.get("Sharpe"),
        "Max_Drawdown": summary.get("Max Drawdown"),
        "_idx": idx
    }

def run_single_factor_tests(top_n: int = 20, selected_symbols: list[str] | None = None, use_multiprocessing: bool = True) -> pd.DataFrame:
    """
    Backtest using one factor at a time (weight=1, others=0). Research-only.
    Returns summary: factor name, CAGR, Sharpe, Max Drawdown.
    """
    factors = [
        ("momentum", {"quality": 0.0, "growth": 0.0, "momentum": 1.0, "value": 0.0}),
        ("quality", {"quality": 1.0, "growth": 0.0, "momentum": 0.0, "value": 0.0}),
        ("value", {"quality": 0.0, "growth": 0.0, "momentum": 0.0, "value": 1.0}),
        ("growth", {"quality": 0.0, "growth": 1.0, "momentum": 0.0, "value": 0.0}),
    ]
    rows = []
    
    if use_multiprocessing:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(_worker_run_single_factor, i, f[0], f[1], top_n, selected_symbols): i for i, f in enumerate(factors)}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    rows.append(res)
    else:
        for i, f in enumerate(factors):
            res = _worker_run_single_factor(i, f[0], f[1], top_n, selected_symbols)
            if res:
                rows.append(res)
                
    rows.sort(key=lambda x: x["_idx"])
    for r in rows:
        r.pop("_idx", None)
        
    return pd.DataFrame(rows)


def run_regime_analysis(
    strategy: str = STRATEGY_EQUAL_WEIGHT,
    top_n: int = 20,
    selected_symbols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run backtest once; label days by 30d rolling vol (high if > median).
    Return CAGR during high-vol and low-vol regimes.
    """
    curve, _ = run_backtest(strategy=strategy, top_n=top_n, selected_symbols=selected_symbols)
    if curve.empty or "daily_return" not in curve.columns:
        return {"cagr_high_vol": None, "cagr_low_vol": None, "regime_difference": None}

    ret = curve.set_index("date")["daily_return"]
    ret.index = pd.to_datetime(ret.index)
    roll_vol = ret.rolling(30, min_periods=10).std()
    median_vol = roll_vol.median()
    if pd.isna(median_vol) or median_vol <= 0:
        return {"cagr_high_vol": None, "cagr_low_vol": None, "regime_difference": None}

    high_vol = roll_vol >= median_vol
    low_vol = ~high_vol
    r_high = ret[high_vol].dropna()
    r_low = ret[low_vol].dropna()

    def cagr_from_series(s: pd.Series) -> float:
        if len(s) < 2:
            return 0.0
        total = (1 + s).prod() - 1
        n_years = len(s) / 252.0
        return (1 + total) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0

    cagr_high = cagr_from_series(r_high)
    cagr_low = cagr_from_series(r_low)
    diff = abs(cagr_high - cagr_low) if (cagr_high is not None and cagr_low is not None) else None
    return {
        "cagr_high_vol": cagr_high,
        "cagr_low_vol": cagr_low,
        "regime_difference": diff,
    }


def generate_validation_report(
    strategy: str = STRATEGY_EQUAL_WEIGHT,
    top_n: int = 20,
    selected_symbols: list[str] | None = None,
    save_csv: bool = False,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run all validations; print summary and overfitting red flags; return structured results.
    Assumption: walk-forward and regime use stored scores (no recalculation). No look-ahead.
    """
    print("\n" + "=" * 60)
    print("RESEARCH VALIDATION REPORT")
    if selected_symbols:
        print(f"Selected symbols: {len(selected_symbols)} -> {selected_symbols[:10]}")
    print("=" * 60)

    # Walk-forward (single split)
    wf_df = run_walk_forward(strategy=strategy, top_n=top_n, use_rolling=False, selected_symbols=selected_symbols)
    if not wf_df.empty:
        train_sharpe = wf_df["train_Sharpe"].mean()
        test_sharpe = wf_df["test_Sharpe"].mean()
        gap = train_sharpe - test_sharpe
        worst_test = wf_df["test_Sharpe"].min()
        print("\n--- WALK-FORWARD SUMMARY ---")
        print(f"  TRAIN SHARPE:  {train_sharpe:.4f}")
        print(f"  TEST SHARPE:   {test_sharpe:.4f}")
        print(f"  OVERFITTING GAP: {gap:.4f}")
        if gap > SHARPE_GAP_WARN:
            print(f"  [WARNING] Train >> Test Sharpe (gap > {SHARPE_GAP_WARN}). Possible overfitting.")
        print(f"  Worst test window Sharpe: {worst_test:.4f}")
    else:
        train_sharpe = test_sharpe = gap = worst_test = None
        print("\n--- WALK-FORWARD SUMMARY --- (no data)")

    # Weight sensitivity
    sens_df = run_weight_sensitivity(strategy=strategy, top_n=top_n, selected_symbols=selected_symbols)
    if not sens_df.empty and "Sharpe" in sens_df.columns:
        sharpe_std = sens_df["Sharpe"].std()
        sharpe_range = sens_df["Sharpe"].max() - sens_df["Sharpe"].min()
        print("\n--- WEIGHT SENSITIVITY RANGE ---")
        print(sens_df.to_string(index=False))
        print(f"  Sharpe range: {sharpe_range:.4f}, std: {sharpe_std:.4f}")
        if pd.notna(sharpe_std) and sharpe_std > 0.5:
            print("  [WARNING] High sensitivity to weight change. Model may be unstable.")
    else:
        print("\n--- WEIGHT SENSITIVITY --- (no data)")
        sharpe_std = sharpe_range = None

    # Single factor
    single_df = run_single_factor_tests(top_n=top_n, selected_symbols=selected_symbols)
    _, full_sum = run_backtest(strategy=strategy, top_n=top_n, selected_symbols=selected_symbols)
    full_sharpe = full_sum.get("Sharpe") or 0
    print("\n--- SINGLE FACTOR PERFORMANCE ---")
    print(single_df.to_string(index=False))
    if not single_df.empty and "Sharpe" in single_df.columns:
        best_factor = single_df.loc[single_df["Sharpe"].idxmax(), "factor"]
        best_factor_sharpe = single_df["Sharpe"].max()
        if full_sharpe > 0 and best_factor_sharpe > 0 and best_factor_sharpe / full_sharpe >= SINGLE_FACTOR_DOMINANCE:
            print(f"  [WARNING] Single factor ({best_factor}) explains most of performance. Composite may add little.")
    else:
        best_factor = best_factor_sharpe = None

    # Regime
    regime = run_regime_analysis(strategy=strategy, top_n=top_n, selected_symbols=selected_symbols)
    print("\n--- REGIME PERFORMANCE DIFFERENCE ---")
    print(f"  CAGR (high vol): {regime.get('cagr_high_vol')}")
    print(f"  CAGR (low vol):  {regime.get('cagr_low_vol')}")
    print(f"  Regime difference: {regime.get('regime_difference')}")
    if regime.get("regime_difference") is not None and regime["regime_difference"] > 0.2:
        print("  [WARNING] Strategy performance differs strongly by regime. May fail in unseen regimes.")

    print("\n" + "=" * 60)
    print("Validation complete. No production scoring logic was modified.")
    print("=" * 60 + "\n")

    # Optionally save outputs
    if save_csv:
        out_dir = out_dir or os.path.join(os.getcwd(), "reports")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        try:
            if not wf_df.empty:
                wf_path = os.path.join(out_dir, f"walk_forward_{strategy}_top{top_n}_{ts}.csv")
                wf_df.to_csv(wf_path, index=False)
            else:
                wf_path = None
            if not sens_df.empty:
                sens_path = os.path.join(out_dir, f"sensitivity_{strategy}_top{top_n}_{ts}.csv")
                sens_df.to_csv(sens_path, index=False)
            else:
                sens_path = None
            if not single_df.empty:
                single_path = os.path.join(out_dir, f"single_factor_{strategy}_top{top_n}_{ts}.csv")
                single_df.to_csv(single_path, index=False)
            else:
                single_path = None
            regime_path = os.path.join(out_dir, f"regime_{strategy}_top{top_n}_{ts}.json")
            import json

            with open(regime_path, "w", encoding="utf8") as fh:
                json.dump(regime, fh, default=str, indent=2)

            print("Saved validation outputs:")
            if wf_path:
                print(f"  Walk-forward: {wf_path}")
            if sens_path:
                print(f"  Sensitivity:  {sens_path}")
            if single_path:
                print(f"  Single-factor:{single_path}")
            print(f"  Regime:       {regime_path}")
        except Exception as e:
            print(f"Failed to save CSVs: {e}")

        saved_paths = {
            "walk_forward": wf_path,
            "sensitivity": sens_path,
            "single_factor": single_path,
            "regime": regime_path,
        }
    else:
        saved_paths = {}

    return {
        "walk_forward": wf_df,
        "sensitivity": sens_df,
        "single_factor": single_df,
        "regime_analysis": regime,
        "saved_paths": saved_paths,
    }


if __name__ == "__main__":
    generate_validation_report(strategy=STRATEGY_EQUAL_WEIGHT, top_n=20)
