"""services/error_model.py

Rolling 5-Year Prediction Error Coefficient Estimator
======================================================

PURPOSE
-------
Quantifies how well the model's predicted returns align with realised
1-month forward returns, and diagnoses *which factors* drive systematic
over- or under-prediction.

THEORY
------
At every monthly rebalance date t we have two quantities for each stock i:

  μ̂_i,t  — predicted 1-month return (from Bayesian shrinkage or
             composite-score-to-return mapping)

  r_i,t   — actual 1-month forward return (price at t+21 / price at t − 1)
             This is calculated ONLY for t values that are already in the
             past relative to the estimation window end — NO look-ahead.

The forecast error is:

  ε_i,t  = r_i,t − μ̂_i,t

We then regress the full (i, t) panel of errors against the four factor
scores that were available at t (again using only data on or before t):

  ε_i,t = β0 + β1·MOM_i,t + β2·QUAL_i,t + β3·VAL_i,t + β4·VOL_i,t + η_i,t

A non-zero βk means factor k *systematically* biases the prediction in a
way that future model versions could correct. R² measures how much of the
error variance is explained by the four factors.

NO LOOK-AHEAD BIAS
------------------
• Scores and factors used at rebalance date t are fetched as a snapshot on
  or before t  (via get_latest_score_date_on_or_before).
• Forward returns for date t use prices at t and t+FORWARD_DAYS — both of
  which must be ≤ window_end.  Any (stock, date) pair where the forward
  price is not yet available is silently dropped.
• The rolling window [start_date, end_date] is fixed before any query;
  individual observation pairs (μ̂, r) can only be included once t AND
  t+FORWARD_DAYS are both ≤ end_date.

USAGE
-----
    from services.error_model import compute_error_coefficients

    result = compute_error_coefficients(
        end_date=date.today(),
        window_years=5,
    )
    print(result["coefficients"])    # {"momentum": β1, ...}
    print(result["r_squared"])       # 0.0 – 1.0
    print(result["n_observations"])  # number of (stock, month) pairs used

NOTES ON OLS FALLBACK
---------------------
If statsmodels is installed, we use statsmodels.OLS for richer diagnostics
(t-stats, p-values). If it is absent, we fall back to numpy.linalg.lstsq
which still gives correct coefficient estimates and R².
"""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database import get_db_context
from app.models import Factor, Price, Score, Stock
from services.backtest import get_latest_score_date_on_or_before, get_rebalance_dates
from services.return_estimator import bayesian_shrinkage_returns

_logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FORWARD_DAYS: int = 21           # 1 calendar month ≈ 21 trading days
LOOKBACK_PRICE_DAYS: int = 90    # history for shrinkage estimator per stock
ANNUALIZE: int = 252
FACTOR_COLS: List[str] = ["momentum_score", "quality_score", "value_score", "volatility_score"]
FACTOR_LABELS: List[str] = ["momentum", "quality", "value", "volatility"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_scores_for_window(start_date: date, end_date: date) -> pd.DataFrame:
    """Load (stock_id, date, composite_score) for all scores in [start, end].

    NO LOOK-AHEAD: only dates ≤ end_date are fetched.
    """
    with get_db_context() as db:
        stmt = (
            select(Score.stock_id, Score.date, Score.composite_score)
            .where(Score.date >= start_date, Score.date <= end_date)
            .order_by(Score.date, Score.stock_id)
        )
        rows = db.execute(stmt).all()

    if not rows:
        return pd.DataFrame(columns=["stock_id", "score_date", "composite_score"])

    df = pd.DataFrame(rows, columns=["stock_id", "score_date", "composite_score"])
    df["score_date"] = pd.to_datetime(df["score_date"]).dt.date
    return df


def _load_factors_for_window(start_date: date, end_date: date) -> pd.DataFrame:
    """Load (stock_id, date, momentum_score, quality_score, value_score, volatility_score)
    for all factor rows in [start, end].

    NO LOOK-AHEAD: only dates ≤ end_date are fetched.
    """
    with get_db_context() as db:
        stmt = (
            select(
                Factor.stock_id, Factor.date,
                Factor.momentum_score, Factor.quality_score,
                Factor.value_score,   Factor.volatility_score,
            )
            .where(Factor.date >= start_date, Factor.date <= end_date)
            .order_by(Factor.date, Factor.stock_id)
        )
        rows = db.execute(stmt).all()

    if not rows:
        return pd.DataFrame(
            columns=["stock_id", "factor_date"] + FACTOR_COLS
        )

    df = pd.DataFrame(
        rows,
        columns=["stock_id", "factor_date"] + FACTOR_COLS,
    )
    df["factor_date"] = pd.to_datetime(df["factor_date"]).dt.date
    return df


def _load_monthly_prices_for_stocks(
    stock_ids: List[int],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load close prices in [start_date, end_date] for given stocks.

    Returns wide DataFrame: index=date (DatetimeIndex), columns=stock_id.
    NO LOOK-AHEAD: end_date is always the window boundary passed in by caller.
    """
    if not stock_ids:
        return pd.DataFrame()

    # psycopg2 cannot bind numpy integer types — normalise to plain Python int
    stock_ids = [int(x) for x in stock_ids]

    with get_db_context() as db:
        stmt = (
            select(Price.date, Price.stock_id, Price.close)
            .where(
                Price.stock_id.in_(stock_ids),
                Price.date >= start_date,
                Price.date <= end_date,
                Price.close.is_not(None),
            )
            .order_by(Price.date)
        )
        rows = db.execute(stmt).all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "stock_id", "close"])
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="stock_id", values="close")
    return wide.sort_index()


def _forward_return(
    prices: pd.DataFrame,
    as_of_date: date,
    forward_days: int = FORWARD_DAYS,
) -> pd.Series:
    """Compute the 1-month forward return for each stock as of `as_of_date`.

    Forward return for stock i = price[t + forward_days] / price[t] − 1
    where t = as_of_date.

    Any stock without a price at t or an available price within forward_days
    trading days ahead is returned as NaN (will be dropped downstream).

    NO LOOK-AHEAD: we only look at prices already in the `prices` DataFrame,
    which the caller must have sliced to ≤ window_end before calling this.
    """
    ts_ref = pd.Timestamp(as_of_date)

    # Closest price on or just after as_of_date
    future_idx = prices.index[prices.index >= ts_ref]
    if future_idx.empty:
        return pd.Series(dtype=float)

    # Price at t (entry price)
    t0 = future_idx[0]
    p0 = prices.loc[t0]

    # Price at t + forward_days (exit price) — use the first date at or after
    # the target, but only up to forward_days + 5 calendar days slack
    target_exit = t0 + pd.Timedelta(days=int(forward_days * 1.5))
    fwd_candidates = prices.index[(prices.index > t0) & (prices.index <= target_exit)]
    if len(fwd_candidates) < forward_days:
        # Not enough future data available — do not compute (no look-ahead)
        return pd.Series(dtype=float)

    t1 = fwd_candidates[forward_days - 1]  # exactly forward_days steps ahead
    p1 = prices.loc[t1]

    fwd_ret = (p1 / p0) - 1.0
    fwd_ret.name = "forward_return"
    return fwd_ret


def _predicted_return_from_shrinkage(
    stock_ids: List[int],
    as_of_date: date,
    lookback_price_days: int = LOOKBACK_PRICE_DAYS,
) -> pd.Series:
    """Predict 1-month expected return for each stock using Bayesian shrinkage.

    Uses up to `lookback_price_days` of daily returns ending on `as_of_date`.
    Shrunk daily mean × 21 gives 1-month expected return.
    NO LOOK-AHEAD: prices strictly ≤ as_of_date.
    """
    cal_start = as_of_date - timedelta(days=int(lookback_price_days * 1.6))
    prices_hist = _load_monthly_prices_for_stocks(stock_ids, cal_start, as_of_date)

    if prices_hist.empty or len(prices_hist) < 5:
        return pd.Series(dtype=float)

    daily_ret = prices_hist.pct_change(fill_method=None).dropna(how="all")
    if daily_ret.empty:
        return pd.Series(dtype=float)

    # Shrunk daily expected return
    mu_daily = bayesian_shrinkage_returns(daily_ret)
    # Convert to 1-month (21-day) expected return
    mu_monthly = mu_daily * FORWARD_DAYS
    mu_monthly.name = "predicted_return"
    return mu_monthly


def _predicted_return_from_score(
    scores_row: pd.Series,
    score_col: str = "composite_score",
    scale_factor: float = 0.002,
) -> pd.Series:
    """Convert composite score (0–100) to an expected monthly return proxy.

    Simple linear mapping: μ̂ = (score / 100 − 0.5) × scale_factor × 21
    Centred at 0.5 so that a 50-score stock has zero expected alpha.
    Default scale_factor ≈ 0.002 daily → 0.042 (4.2%) monthly maximum.

    Parameters
    ----------
    scores_row   : Series indexed by stock_id with composite_score values.
    scale_factor : Daily alpha per unit of normalised score deviation.
    """
    normalised = scores_row / 100.0 - 0.5  # ∈ [-0.5, +0.5]
    return normalised * scale_factor * FORWARD_DAYS


# ─────────────────────────────────────────────────────────────────────────────
# OLS regression helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_ols(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Fit OLS y ~ X (X already includes intercept column).

    Returns dict with intercept, coefficients per feature, R², n,
    t-statistics and p-values (when statsmodels is available).
    """
    n, k = X.shape
    result: Dict[str, Any] = {
        "n_observations": n,
        "n_features": k - 1,  # exclude intercept
        "intercept": float("nan"),
        "coefficients": {f: float("nan") for f in feature_names},
        "r_squared": float("nan"),
        "adj_r_squared": float("nan"),
        "t_statistics": {f: float("nan") for f in feature_names},
        "p_values": {f: float("nan") for f in feature_names},
        "residual_std": float("nan"),
        "method": "numpy",
    }

    if n < k + 2:
        _logger.warning("OLS: too few observations (%d) for %d parameters; returning NaN", n, k)
        return result

    try:
        import statsmodels.api as sm
        ols = sm.OLS(y, X).fit()
        params = ols.params  # [intercept, β1…βk]
        result["method"] = "statsmodels"
        result["intercept"] = float(params[0])
        for i, name in enumerate(feature_names):
            result["coefficients"][name] = float(params[i + 1])
        result["r_squared"]     = float(ols.rsquared)
        result["adj_r_squared"] = float(ols.rsquared_adj)
        result["residual_std"]  = float(ols.mse_resid ** 0.5)
        for i, name in enumerate(feature_names):
            result["t_statistics"][name] = float(ols.tvalues[i + 1])
            result["p_values"][name]     = float(ols.pvalues[i + 1])
    except ImportError:
        # Fallback: numpy lstsq
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        result["intercept"] = float(beta[0])
        for i, name in enumerate(feature_names):
            result["coefficients"][name] = float(beta[i + 1])
        # R² manually
        y_hat = X @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        result["r_squared"] = r2
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)
        result["adj_r_squared"] = float(adj_r2)
        residual_std = float(np.sqrt(ss_res / max(n - k, 1)))
        result["residual_std"] = residual_std
    except Exception as exc:
        _logger.error("OLS failed: %s", exc, exc_info=True)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Coefficient stability logger
# ─────────────────────────────────────────────────────────────────────────────

def _log_stability(history: List[Dict[str, Any]]) -> None:
    """Log rolling coefficient stability across sub-windows."""
    if len(history) < 2:
        return

    coef_series: Dict[str, List[float]] = {f: [] for f in FACTOR_LABELS}
    for h in history:
        for f in FACTOR_LABELS:
            val = h.get("result", {}).get("coefficients", {}).get(f, float("nan"))
            coef_series[f].append(float(val) if not np.isnan(val) else float("nan"))

    _logger.info("── Coefficient stability across %d sub-windows ──", len(history))
    for f in FACTOR_LABELS:
        vals = [v for v in coef_series[f] if not np.isnan(v)]
        if not vals:
            _logger.info("  %-12s  no data", f)
            continue
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        _logger.info(
            "  %-12s  mean=% .5f  std=%.5f  cv=%.2f  range=[%.5f, %.5f]",
            f, mean_v, std_v,
            std_v / abs(mean_v) if mean_v != 0 else float("inf"),
            float(np.min(vals)), float(np.max(vals)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core observation builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_observation_panel(
    rebalance_dates: List[date],
    scores_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    prices_wide: pd.DataFrame,
    window_end: date,
    prediction_method: str = "shrinkage",
) -> pd.DataFrame:
    """Build the (stock, month) panel of (error, factor_scores).

    For each rebalance date t:
      1. Get scores / factors on or before t (no look-ahead).
      2. Compute predicted return μ̂ using shrinkage or score mapping.
      3. Compute actual forward return r for t → t+21.
      4. error = r − μ̂
      5. Attach factor scores as regressors.

    Returns DataFrame with columns:
      [rebalance_date, stock_id, predicted_return, actual_return,
       error, momentum_score, quality_score, value_score, volatility_score]
    """
    records = []

    for t in rebalance_dates:
        # Forward window must be fully within our observation window
        # — if t+21 > window_end we have no realised return yet (look-ahead)
        fwd_end = t + timedelta(days=int(FORWARD_DAYS * 1.5))
        if fwd_end > window_end:
            _logger.debug("Skipping rebalance %s: forward window extends beyond windowend", t)
            continue

        # ── Score snapshot (no look-ahead) ────────────────────────────────
        score_date = get_latest_score_date_on_or_before(t)
        if score_date is None:
            continue

        scores_snap = (
            scores_df[scores_df["score_date"] == score_date]
            .set_index("stock_id")["composite_score"]
            .dropna()
        )
        if scores_snap.empty:
            continue

        # ── Factor snapshot (no look-ahead — use same score_date) ─────────
        factors_snap = (
            factors_df[factors_df["factor_date"] == score_date]
            .set_index("stock_id")[FACTOR_COLS]
            .dropna(how="all")
        )

        # ── Common stocks ─────────────────────────────────────────────────
        common_stocks = [int(x) for x in scores_snap.index.intersection(factors_snap.index)]
        if not common_stocks:
            continue

        # ── Predicted returns ──────────────────────────────────────────────
        if prediction_method == "shrinkage":
            mu_hat = _predicted_return_from_shrinkage(
                stock_ids=common_stocks,
                as_of_date=t,
            )
        else:
            mu_hat = _predicted_return_from_score(
                scores_row=scores_snap.loc[common_stocks],
            )
        mu_hat = mu_hat.reindex(common_stocks).dropna()
        if mu_hat.empty:
            continue

        # ── Actual forward returns (prices already sliced to ≤ window_end) -
        fwd_ret = _forward_return(prices_wide, as_of_date=t)
        if fwd_ret.empty:
            continue
        fwd_ret = fwd_ret.reindex(mu_hat.index).dropna()

        # ── Build observations ─────────────────────────────────────────────
        for sid in fwd_ret.index:
            mu     = float(mu_hat.get(sid, float("nan")))
            actual = float(fwd_ret[sid])
            if np.isnan(mu) or np.isnan(actual):
                continue

            error = actual - mu

            frow = factors_snap.loc[sid] if sid in factors_snap.index else pd.Series(dtype=float)
            records.append({
                "rebalance_date":   t,
                "stock_id":         int(sid),
                "predicted_return": mu,
                "actual_return":    actual,
                "error":            error,
                "momentum_score":   float(frow.get("momentum_score", float("nan"))),
                "quality_score":    float(frow.get("quality_score",  float("nan"))),
                "value_score":      float(frow.get("value_score",    float("nan"))),
                "volatility_score": float(frow.get("volatility_score", float("nan"))),
            })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_error_coefficients(
    end_date: Optional[date] = None,
    start_date: Optional[date] = None,
    window_years: int = 5,
    prediction_method: str = "shrinkage",
    min_observations: int = 30,
    log_stability: bool = True,
) -> Dict[str, Any]:
    """Estimate model prediction error coefficients over a rolling 5-year window.

    Parameters
    ----------
    end_date          : Last date in the estimation window. Defaults to today.
    start_date        : First date in the estimation window. If None, computed
                        as end_date − window_years years.
    window_years      : Length of rolling window in years (default 5).
    prediction_method : 'shrinkage' (Bayesian shrinkage predicted return) or
                        'score' (composite score → return proxy).
    min_observations  : Minimum number of (stock, month) pairs required to
                        fit the regression.  Returns NaN result if fewer.
    log_stability     : If True, log coefficient stability diagnostics.

    Returns
    -------
    dict
        {
            "intercept": float,                 # β0
            "coefficients": {                   # β1 ... β4
                "momentum":   float,
                "quality":    float,
                "value":      float,
                "volatility": float,
            },
            "r_squared":      float,            # OLS R²
            "adj_r_squared":  float,            # Adjusted R²
            "t_statistics":   dict[str, float], # per-coefficient t-stat
            "p_values":       dict[str, float], # two-sided p-values
            "residual_std":   float,            # std of residuals
            "n_observations": int,              # number of (stock, month) pairs
            "window_start":   date,
            "window_end":     date,
            "prediction_method": str,
            "mean_absolute_error": float,       # MAE of predictions
            "mean_error":          float,       # bias: mean(ε)
            "error_std":           float,       # std(ε)
            "panel_df":            pd.DataFrame,  # full observation panel
        }

    NO LOOK-AHEAD GUARANTEE
    -----------------------
    For every rebalance date t in the window:
      * Scores/factors: date ≤ t   (get_latest_score_date_on_or_before)
      * Forward price:  date ≤ end_date  (prices already bounded)
      * t itself must satisfy t + 21 ≤ end_date (any pair beyond this is dropped)
    """
    # ── Window setup ──────────────────────────────────────────────────────────
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = date(end_date.year - window_years, end_date.month, end_date.day)

    _logger.info(
        "compute_error_coefficients: window=[%s, %s]  method=%s",
        start_date, end_date, prediction_method,
    )

    # ── Empty result template ─────────────────────────────────────────────────
    empty_result: Dict[str, Any] = {
        "intercept":          float("nan"),
        "coefficients":       {f: float("nan") for f in FACTOR_LABELS},
        "r_squared":          float("nan"),
        "adj_r_squared":      float("nan"),
        "t_statistics":       {f: float("nan") for f in FACTOR_LABELS},
        "p_values":           {f: float("nan") for f in FACTOR_LABELS},
        "residual_std":       float("nan"),
        "n_observations":     0,
        "window_start":       start_date,
        "window_end":         end_date,
        "prediction_method":  prediction_method,
        "mean_absolute_error": float("nan"),
        "mean_error":          float("nan"),
        "error_std":           float("nan"),
        "panel_df":           pd.DataFrame(),
    }

    # ── Fetch rebalance dates within window ───────────────────────────────────
    rebalance_dates = get_rebalance_dates(start_date=start_date, end_date=end_date)
    if len(rebalance_dates) < 2:
        _logger.warning("No rebalance dates in window [%s, %s]", start_date, end_date)
        return empty_result

    # ── Fetch scores and factors (bounded by window — no look-ahead) ──────────
    scores_df  = _load_scores_for_window(start_date, end_date)
    factors_df = _load_factors_for_window(start_date, end_date)

    if scores_df.empty:
        _logger.warning("No score data in window [%s, %s]", start_date, end_date)
        return empty_result

    # ── Fetch prices ──────────────────────────────────────────────────────────
    # Extend start by LOOKBACK_PRICE_DAYS to give shrinkage estimator enough history
    price_start = start_date - timedelta(days=int(LOOKBACK_PRICE_DAYS * 1.6))
    all_stock_ids = [int(x) for x in scores_df["stock_id"].unique()]
    prices_wide = _load_monthly_prices_for_stocks(
        all_stock_ids, price_start, end_date   # ← strictly ≤ end_date
    )

    if prices_wide.empty:
        _logger.warning("No price data found — cannot compute forward returns")
        return empty_result

    # ── Build observation panel ───────────────────────────────────────────────
    panel = _build_observation_panel(
        rebalance_dates=rebalance_dates,
        scores_df=scores_df,
        factors_df=factors_df,
        prices_wide=prices_wide,
        window_end=end_date,
        prediction_method=prediction_method,
    )

    if panel.empty:
        _logger.warning(
            "Observation panel is empty after forward-return matching "
            "(window too short or DB has insufficient data)."
        )
        return empty_result

    # ── Drop rows with missing factor scores ──────────────────────────────────
    factor_cols_in_panel = [c for c in FACTOR_COLS if c in panel.columns]
    panel_clean = panel.dropna(subset=["error"] + factor_cols_in_panel)

    n_dropped = len(panel) - len(panel_clean)
    if n_dropped > 0:
        _logger.info("Dropped %d rows with missing error or factor values", n_dropped)

    if len(panel_clean) < min_observations:
        _logger.warning(
            "Too few observations (%d < %d) for stable OLS — returning NaN result",
            len(panel_clean), min_observations,
        )
        out = dict(empty_result)
        out["n_observations"] = len(panel_clean)
        out["panel_df"] = panel_clean
        return out

    # ── OLS: ε = β0 + β1·MOM + β2·QUAL + β3·VAL + β4·VOL ────────────────────
    y = panel_clean["error"].values.astype(float)
    factor_matrix = panel_clean[factor_cols_in_panel].values.astype(float)

    # Design matrix with intercept column
    X = np.column_stack([np.ones(len(y)), factor_matrix])

    ols_result = _run_ols(X, y, feature_names=FACTOR_LABELS[:len(factor_cols_in_panel)])

    # ── Error diagnostics ──────────────────────────────────────────────────────
    mae   = float(np.mean(np.abs(y)))
    me    = float(np.mean(y))          # bias
    e_std = float(np.std(y))

    _logger.info(
        "OLS complete: n=%d  R²=%.4f  intercept=%.6f  "
        "bias(mean_ε)=%.6f  MAE=%.6f  std(ε)=%.6f",
        len(panel_clean),
        ols_result.get("r_squared", float("nan")),
        ols_result.get("intercept", float("nan")),
        me, mae, e_std,
    )
    for f in FACTOR_LABELS:
        β  = ols_result["coefficients"].get(f, float("nan"))
        t  = ols_result.get("t_statistics", {}).get(f, float("nan"))
        pv = ols_result.get("p_values",     {}).get(f, float("nan"))
        _logger.info("  %-12s  β=% .6f  t=% .3f  p=%.4f", f, β, t, pv)

    # ── Stability check across annual sub-windows ──────────────────────────────
    stability_history: List[Dict[str, Any]] = []
    if log_stability and len(rebalance_dates) >= 12:
        sub_years = max(1, window_years - 1)
        for yr in range(sub_years):
            sub_start = date(start_date.year + yr, start_date.month, start_date.day)
            sub_end   = date(min(sub_start.year + 1, end_date.year),
                             sub_start.month, sub_start.day)
            sub_end   = min(sub_end, end_date)
            sub_panel = panel_clean[
                (panel_clean["rebalance_date"] >= sub_start) &
                (panel_clean["rebalance_date"] <= sub_end)
            ]
            if len(sub_panel) < min_observations:
                continue
            sub_y = sub_panel["error"].values.astype(float)
            sub_X = np.column_stack([
                np.ones(len(sub_y)),
                sub_panel[factor_cols_in_panel].values.astype(float),
            ])
            sub_ols = _run_ols(sub_X, sub_y, FACTOR_LABELS[:len(factor_cols_in_panel)])
            stability_history.append({
                "sub_window": f"{sub_start}→{sub_end}",
                "n": len(sub_panel),
                "result": sub_ols,
            })

        _log_stability(stability_history)

    return {
        "intercept":           ols_result.get("intercept", float("nan")),
        "coefficients":        ols_result.get("coefficients", {f: float("nan") for f in FACTOR_LABELS}),
        "r_squared":           ols_result.get("r_squared", float("nan")),
        "adj_r_squared":       ols_result.get("adj_r_squared", float("nan")),
        "t_statistics":        ols_result.get("t_statistics", {}),
        "p_values":            ols_result.get("p_values", {}),
        "residual_std":        ols_result.get("residual_std", float("nan")),
        "n_observations":      ols_result.get("n_observations", len(panel_clean)),
        "window_start":        start_date,
        "window_end":          end_date,
        "prediction_method":   prediction_method,
        "mean_absolute_error": mae,
        "mean_error":          me,
        "error_std":           e_std,
        "stability_history":   stability_history,
        "panel_df":            panel_clean,
        "method":              ols_result.get("method", "numpy"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: formatted summary string
# ─────────────────────────────────────────────────────────────────────────────

def format_error_model_summary(result: Dict[str, Any]) -> str:
    """Return a human-readable summary of the error model result."""
    lines = [
        "═" * 62,
        "  Prediction Error Model — OLS Coefficient Summary",
        "═" * 62,
        f"  Window:         {result.get('window_start')} → {result.get('window_end')}",
        f"  Method:         {result.get('prediction_method')}  "
        f"(OLS via {result.get('method', 'numpy')})",
        f"  N observations: {result.get('n_observations', 0)}",
        f"  R²:             {result.get('r_squared', float('nan')):.4f}",
        f"  Adj-R²:         {result.get('adj_r_squared', float('nan')):.4f}",
        f"  Residual std:   {result.get('residual_std', float('nan')):.6f}",
        "",
        "  Error statistics",
        f"    Mean error (bias):  {result.get('mean_error', float('nan')):+.6f}",
        f"    Mean abs error:     {result.get('mean_absolute_error', float('nan')):.6f}",
        f"    Error std dev:      {result.get('error_std', float('nan')):.6f}",
        "",
        "  Regression: ε = β0 + β1·MOM + β2·QUAL + β3·VAL + β4·VOL",
        f"    β0 (intercept): {result.get('intercept', float('nan')):+.6f}",
    ]

    coefs   = result.get("coefficients",  {})
    t_stats = result.get("t_statistics",  {})
    p_vals  = result.get("p_values",      {})

    for f in FACTOR_LABELS:
        β  = coefs.get(f,   float("nan"))
        t  = t_stats.get(f, float("nan"))
        pv = p_vals.get(f,  float("nan"))
        sig = "**" if (not np.isnan(pv) and pv < 0.05) else "  "
        lines.append(
            f"    {sig}β_{f[:3].upper():<4}: {β:+.6f}  (t={t:+.2f}, p={pv:.4f}){sig}"
        )

    lines.append("═" * 62)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Calibration Persistence
# ─────────────────────────────────────────────────────────────────────────────

# How many months between scheduled re-calibrations
CALIBRATION_INTERVAL_MONTHS: int = 6

# Default Bayesian blend weight:  posterior = w*new + (1-w)*old
DEFAULT_BLEND_WEIGHT: float = 0.6


def _months_between(d1: date, d2: date) -> float:
    """Approximate number of calendar months from d1 to d2 (signed)."""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + (d2.day - d1.day) / 30.0


def _load_latest_calibration() -> Optional[Dict[str, Any]]:
    """Fetch the most recent ModelCalibration row from the DB.

    Returns a plain dict with the stored posterior coefficients, or None if
    the table is empty (first-ever calibration).
    """
    try:
        from app.models import ModelCalibration
        with get_db_context() as db:
            row = (
                db.execute(
                    select(ModelCalibration)
                    .order_by(ModelCalibration.calibration_end_date.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if row is None:
                return None
            return {
                "id":                     row.id,
                "calibration_start_date": row.calibration_start_date,
                "calibration_end_date":   row.calibration_end_date,
                "intercept":              row.intercept,
                "coefficients": {
                    "momentum":   row.beta_momentum,
                    "quality":    row.beta_quality,
                    "value":      row.beta_value,
                    "volatility": row.beta_volatility,
                },
                "r_squared":       row.r_squared,
                "adj_r_squared":   row.adj_r_squared,
                "n_observations":  row.n_observations,
                "residual_std":    row.residual_std,
                "mean_error":      row.mean_error,
                "blend_weight":    row.blend_weight,
                "prediction_method": row.prediction_method,
                "triggered_by":    row.triggered_by,
                "created_at":      row.created_at,
            }
    except Exception as exc:
        _logger.warning("Could not load latest calibration from DB: %s", exc)
        return None


def _calibration_is_due(
    current_date: date,
    interval_months: int = CALIBRATION_INTERVAL_MONTHS,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check whether a new calibration is due.

    Returns
    -------
    (is_due, latest_row)
        is_due     : True when current_date >= last_end + interval_months
        latest_row : The most recent DB calibration dict, or None if none exists.
    """
    latest = _load_latest_calibration()
    if latest is None:
        _logger.info("No prior calibration found — first-ever calibration is due.")
        return True, None

    last_end  = latest["calibration_end_date"]
    months_elapsed = _months_between(last_end, current_date)
    is_due = months_elapsed >= interval_months

    _logger.info(
        "Last calibration end: %s  |  Current date: %s  |  "
        "Elapsed: %.1f months  |  Interval: %d months  |  Due: %s",
        last_end, current_date, months_elapsed, interval_months, is_due,
    )
    return is_due, latest


def _bayesian_blend(
    new_result: Dict[str, Any],
    old_result: Optional[Dict[str, Any]],
    blend_weight: float = DEFAULT_BLEND_WEIGHT,
) -> Dict[str, Any]:
    """Apply Bayesian blending: posterior = w * new + (1−w) * old.

    Parameters
    ----------
    new_result   : Raw OLS result dict from compute_error_coefficients.
    old_result   : Previous calibration dict from _load_latest_calibration,
                   or None (first calibration — no blending, w treated as 1.0).
    blend_weight : w ∈ (0, 1].  Default 0.6 gives 60% weight to the new
                   estimate and 40% to the prior (stabilises volatile coefs).

    Returns
    -------
    dict combining posterior coefficients with OLS diagnostics from new_result.
    The raw (pre-blend) coefficients are preserved under raw_* keys.
    """
    blend_weight = float(np.clip(blend_weight, 0.0, 1.0))

    raw_intercept = new_result.get("intercept", float("nan"))
    raw_coefs     = new_result.get("coefficients", {})

    if old_result is None:
        # First-ever calibration — no prior, use raw OLS directly
        _logger.info(
            "First calibration: no prior available — using OLS output directly "
            "(effective w=1.0)"
        )
        posterior_intercept = raw_intercept
        posterior_coefs     = dict(raw_coefs)
        effective_w         = 1.0
    else:
        effective_w = blend_weight
        old_intercept = old_result.get("intercept") or 0.0
        old_coefs     = old_result.get("coefficients", {})

        _nan_safe = lambda new, old: (
            (new * effective_w + old * (1.0 - effective_w))
            if not (np.isnan(new) or np.isnan(old))
            else (new if not np.isnan(new) else old)
        )

        posterior_intercept = _nan_safe(raw_intercept, float(old_intercept) if old_intercept is not None else 0.0)
        posterior_coefs = {}
        for f in FACTOR_LABELS:
            new_b = raw_coefs.get(f, float("nan"))
            old_b = float(old_coefs.get(f) or 0.0)
            posterior_coefs[f] = _nan_safe(new_b, old_b)

        _logger.info(
            "Bayesian blend applied (w=%.2f):", effective_w,
        )
        for f in FACTOR_LABELS:
            _logger.info(
                "  %-12s  raw=% .6f  prior=% .6f  posterior=% .6f",
                f,
                raw_coefs.get(f, float("nan")),
                float(old_coefs.get(f) or 0.0),
                posterior_coefs[f],
            )

    blended = dict(new_result)
    blended["intercept"]    = posterior_intercept
    blended["coefficients"] = posterior_coefs
    blended["blend_weight"] = effective_w
    # Preserve raw pre-blend values for audit
    blended["raw_intercept"] = raw_intercept
    blended["raw_coefficients"] = dict(raw_coefs)
    return blended


def _save_calibration(
    blended: Dict[str, Any],
    triggered_by: str = "scheduled",
) -> None:
    """Persist a blended calibration result to the model_calibration table.

    Silently skips if the unique constraint on calibration_end_date would
    be violated (idempotent re-run protection).
    """
    try:
        from app.models import ModelCalibration
        from sqlalchemy.exc import IntegrityError

        coefs     = blended.get("coefficients", {})
        raw_coefs = blended.get("raw_coefficients", {})

        row = ModelCalibration(
            calibration_start_date = blended["window_start"],
            calibration_end_date   = blended["window_end"],
            # Posterior blended coefficients
            intercept       = _safe_float(blended.get("intercept")),
            beta_momentum   = _safe_float(coefs.get("momentum")),
            beta_quality    = _safe_float(coefs.get("quality")),
            beta_value      = _safe_float(coefs.get("value")),
            beta_volatility = _safe_float(coefs.get("volatility")),
            # OLS diagnostics (from new window before blending)
            r_squared       = _safe_float(blended.get("r_squared")),
            adj_r_squared   = _safe_float(blended.get("adj_r_squared")),
            n_observations  = blended.get("n_observations"),
            residual_std    = _safe_float(blended.get("residual_std")),
            mean_error      = _safe_float(blended.get("mean_error")),
            # Raw pre-blend coefficients
            raw_intercept       = _safe_float(blended.get("raw_intercept")),
            raw_beta_momentum   = _safe_float(raw_coefs.get("momentum")),
            raw_beta_quality    = _safe_float(raw_coefs.get("quality")),
            raw_beta_value      = _safe_float(raw_coefs.get("value")),
            raw_beta_volatility = _safe_float(raw_coefs.get("volatility")),
            # Metadata
            blend_weight      = _safe_float(blended.get("blend_weight")),
            prediction_method = blended.get("prediction_method", "shrinkage"),
            triggered_by      = triggered_by,
        )

        with get_db_context() as db:
            db.add(row)
            try:
                db.commit()
                _logger.info(
                    "Saved calibration: window=[%s, %s]  β_MOM=%.6f  R²=%.4f",
                    row.calibration_start_date, row.calibration_end_date,
                    row.beta_momentum or float("nan"),
                    row.r_squared    or float("nan"),
                )
            except IntegrityError:
                db.rollback()
                _logger.warning(
                    "Calibration for end_date=%s already exists — skipping insert "
                    "(idempotent protection).",
                    blended["window_end"],
                )

    except Exception as exc:
        _logger.error("Failed to save calibration to DB: %s", exc, exc_info=True)


def _safe_float(val: Any) -> Optional[float]:
    """Convert a value to float, returning None for NaN or non-numeric types."""
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API — Bayesian Update Entry-Point
# ─────────────────────────────────────────────────────────────────────────────

def update_error_model_if_due(
    current_date: Optional[date] = None,
    blend_weight: float = DEFAULT_BLEND_WEIGHT,
    window_years: int = 5,
    prediction_method: str = "shrinkage",
    interval_months: int = CALIBRATION_INTERVAL_MONTHS,
    force: bool = False,
    triggered_by: str = "scheduled",
) -> Dict[str, Any]:
    """Check whether a Bayesian coefficient update is due and run it if so.

    Parameters
    ----------
    current_date      : Reference date (defaults to today).
    blend_weight      : w ∈ (0, 1].  Posterior = w*new + (1-w)*old.
                        Default 0.6 applies 60% rolling update weight.
    window_years      : Rolling window length for OLS (default 5 years).
    prediction_method : 'shrinkage' or 'score'.
    interval_months   : How many months between scheduled updates (default 6).
    force             : If True, run calibration unconditionally, ignoring the
                        6-month check.  Useful for manual re-runs.
    triggered_by      : Audit label stored in DB ('scheduled' or 'manual').

    Returns
    -------
    dict
        {
            "updated":      bool,        # True if a new calibration was computed
            "reason":       str,         # why updated/skipped
            "coefficients": dict,        # posterior coefficients in use NOW
            "intercept":    float,
            "r_squared":    float,       # NaN if skipped
            "n_observations": int,       # 0 if skipped
            "window_start": date | None,
            "window_end":   date | None,
            "blend_weight": float,
            "calibration_source": str,   # 'new' | 'cached' | 'none'
        }

    Behaviour
    ---------
    1. Check DB for most recent calibration_end_date.
    2. If current_date >= last_end + 6 months (or force=True OR first run):
       a. Compute new 5-year OLS error coefficients.
       b. Blend with old posterior:  β_post = w*β_new + (1-w)*β_old.
       c. Persist blended result to model_calibration table.
       d. Return the new posterior.
    3. Else:
       Return the cached DB coefficients (no computation).

    NO LOOK-AHEAD GUARANTEE
    -----------------------
    The estimation window is [current_date - window_years, current_date].
    current_date must be today or a historical date — never a future date.
    compute_error_coefficients enforces temporal causality throughout.
    """
    if current_date is None:
        current_date = date.today()

    _logger.info(
        "update_error_model_if_due: current_date=%s  force=%s  interval=%d months",
        current_date, force, interval_months,
    )

    # Check if update is due
    is_due, latest_db = _calibration_is_due(current_date, interval_months)

    # Build an empty-coef response template
    _empty_coefs = {f: float("nan") for f in FACTOR_LABELS}

    if not force and not is_due:
        # ── Return cached coefficients from DB ─────────────────────────────
        _logger.info(
            "Calibration not due — using cached coefficients "
            "(last calibration: %s).",
            latest_db["calibration_end_date"] if latest_db else "none",
        )
        if latest_db:
            return {
                "updated":      False,
                "reason":       f"Not due (last end={latest_db['calibration_end_date']}, "
                                f"elapsed<{interval_months} months)",
                "coefficients": latest_db.get("coefficients", _empty_coefs),
                "intercept":    latest_db.get("intercept", float("nan")),
                "r_squared":    latest_db.get("r_squared", float("nan")),
                "n_observations": latest_db.get("n_observations", 0),
                "window_start": latest_db.get("calibration_start_date"),
                "window_end":   latest_db.get("calibration_end_date"),
                "blend_weight": latest_db.get("blend_weight", blend_weight),
                "calibration_source": "cached",
            }
        else:
            # No prior AND not forced — calibrate anyway (first run)
            _logger.info("No prior calibration exists — running first-time calibration.")

    # ── Compute new error coefficients (5-year rolling window) ────────────────
    trigger_label = "manual" if force else triggered_by
    _logger.info(
        "Running calibration: window=[%s-5yr, %s]  method=%s  w=%.2f  trigger=%s",
        current_date, current_date, prediction_method, blend_weight, trigger_label,
    )

    new_result = compute_error_coefficients(
        end_date=current_date,
        window_years=window_years,
        prediction_method=prediction_method,
        log_stability=True,
    )

    if new_result.get("n_observations", 0) == 0:
        _logger.warning(
            "New calibration produced 0 observations — "
            "falling back to cached coefficients (DB not updated)."
        )
        if latest_db:
            return {
                "updated":      False,
                "reason":       "Calibration due but produced 0 observations; kept cached",
                "coefficients": latest_db.get("coefficients", _empty_coefs),
                "intercept":    latest_db.get("intercept", float("nan")),
                "r_squared":    float("nan"),
                "n_observations": 0,
                "window_start": latest_db.get("calibration_start_date"),
                "window_end":   latest_db.get("calibration_end_date"),
                "blend_weight": latest_db.get("blend_weight", blend_weight),
                "calibration_source": "cached",
            }
        else:
            return {
                "updated":      False,
                "reason":       "First calibration attempted but produced 0 observations",
                "coefficients": _empty_coefs,
                "intercept":    float("nan"),
                "r_squared":    float("nan"),
                "n_observations": 0,
                "window_start": None,
                "window_end":   None,
                "blend_weight": blend_weight,
                "calibration_source": "none",
            }

    # ── Apply Bayesian blending ───────────────────────────────────────────────
    blended = _bayesian_blend(new_result, latest_db, blend_weight=blend_weight)

    # ── Persist to DB ─────────────────────────────────────────────────────────
    _save_calibration(blended, triggered_by=trigger_label)

    _logger.info(
        "Calibration COMPLETE: window=[%s, %s]  n=%d  R²=%.4f  "
        "β_MOM(posterior)=%.6f  w=%.2f",
        blended["window_start"],
        blended["window_end"],
        blended.get("n_observations", 0),
        blended.get("r_squared", float("nan")),
        blended["coefficients"].get("momentum", float("nan")),
        blended.get("blend_weight", blend_weight),
    )

    return {
        "updated":        True,
        "reason":         ("Forced re-calibration" if force else
                           f"Due: >= {interval_months} months since last calibration"),
        "coefficients":   blended["coefficients"],
        "intercept":      blended["intercept"],
        "r_squared":      blended.get("r_squared", float("nan")),
        "adj_r_squared":  blended.get("adj_r_squared", float("nan")),
        "n_observations": blended.get("n_observations", 0),
        "window_start":   blended["window_start"],
        "window_end":     blended["window_end"],
        "blend_weight":   blended.get("blend_weight", blend_weight),
        "raw_coefficients": blended.get("raw_coefficients", {}),
        "calibration_source": "new",
        "stability_history": blended.get("stability_history", []),
    }


def get_current_coefficients() -> Dict[str, Any]:
    """Return the most recently persisted posterior coefficients without triggering a recalibration.

    Useful for using the error model correction at prediction time without
    the overhead of a full OLS run.

    Returns the same shape as update_error_model_if_due, with calibration_source='cached'.
    Returns NaN coefficients if no calibration has been run yet.
    """
    latest = _load_latest_calibration()
    _empty = {f: float("nan") for f in FACTOR_LABELS}

    if latest is None:
        _logger.warning("get_current_coefficients: no calibration in DB — returning NaN")
        return {
            "updated":            False,
            "reason":             "No calibration in DB",
            "coefficients":       _empty,
            "intercept":          float("nan"),
            "r_squared":          float("nan"),
            "n_observations":     0,
            "window_start":       None,
            "window_end":         None,
            "blend_weight":       float("nan"),
            "calibration_source": "none",
        }

    return {
        "updated":            False,
        "reason":             "Read from DB (no update triggered)",
        "coefficients":       latest.get("coefficients", _empty),
        "intercept":          latest.get("intercept", float("nan")),
        "r_squared":          latest.get("r_squared", float("nan")),
        "n_observations":     latest.get("n_observations", 0),
        "window_start":       latest.get("calibration_start_date"),
        "window_end":         latest.get("calibration_end_date"),
        "blend_weight":       latest.get("blend_weight", float("nan")),
        "calibration_source": "cached",
    }


__all__ = [
    # Core OLS estimator
    "compute_error_coefficients",
    "format_error_model_summary",
    # Bayesian update
    "update_error_model_if_due",
    "get_current_coefficients",
    # Constants
    "FORWARD_DAYS",
    "FACTOR_LABELS",
    "FACTOR_COLS",
    "DEFAULT_BLEND_WEIGHT",
    "CALIBRATION_INTERVAL_MONTHS",
]

