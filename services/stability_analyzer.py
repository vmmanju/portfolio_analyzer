"""services/stability_analyzer.py

Rolling Stability Score for Portfolio Analysis.

PURPOSE
-------
A strong backtest CAGR tells you *how much* a portfolio returned, but not
*how reliably* it behaved over time.  A portfolio that alternated between
spectacular and terrible years is far harder to own psychologically, and far
more sensitive to entry-point timing, than one with steadier performance.

The Stability Score (0-100) quantifies time-series consistency using five
orthogonal dimensions:

  25%  Sharpe Consistency   — how stable the risk-adjusted return was month-to-month
  25%  Drawdown Stability   — how predictable the max-loss environment was
  20%  Turnover Penalty     — how much was spent on transaction costs (rebalancing)
  20%  Correlation Stability — how stable the inter-asset diversification was
  10%  Regime Robustness    — how similar performance was across high/low-vol periods

INPUTS
------
equity_curve    : DataFrame with columns: date, daily_return, (optionally)
                  cumulative_return, drawdown.
weights_history : Optional list of dicts:
                  [{"date": date, "weights": {asset: weight},
                    "avg_pairwise_corr": float (optional)}, ...]
                  If None, turnover and correlation drift scores are estimated
                  from defaults (50/100 — neutral).
"""

import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ── Component weights (must sum to 1.0) ───────────────────────────────────────
W_SHARPE_CONSISTENCY  = 0.25
W_DRAWDOWN_STABILITY  = 0.25
W_TURNOVER_PENALTY    = 0.20
W_CORR_STABILITY      = 0.20
W_REGIME_ROBUSTNESS   = 0.10

ANNUALIZE = 252
ROLLING_WINDOW_DAYS = 252        # ≈ 12 months of trading days


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score(raw: float, bad_threshold: float, good_threshold: float = 0.0) -> float:
    """Linear normalise raw ∈ [good_threshold, bad_threshold] → score ∈ [0,100].

    raw ≤ good_threshold → 100
    raw ≥ bad_threshold  → 0
    """
    if bad_threshold <= good_threshold:
        return 50.0
    frac = (raw - good_threshold) / (bad_threshold - good_threshold)
    return float(max(0.0, min(100.0, 100.0 * (1.0 - frac))))


def _rolling_sharpe(returns: pd.Series, window: int = ROLLING_WINDOW_DAYS) -> pd.Series:
    roll_mean  = returns.rolling(window, min_periods=window // 2).mean()
    roll_std   = returns.rolling(window, min_periods=window // 2).std()
    sharpe = roll_mean / roll_std.replace(0, np.nan) * math.sqrt(ANNUALIZE)
    return sharpe.dropna()


def _rolling_cagr(returns: pd.Series, window: int = ROLLING_WINDOW_DAYS) -> pd.Series:
    """Trailing-window CAGR (annualised compound return)."""
    cum = (1 + returns).rolling(window, min_periods=window // 2).apply(
        lambda x: np.prod(x), raw=True
    )
    n_years = window / ANNUALIZE
    cagr = cum ** (1.0 / n_years) - 1.0
    return cagr.dropna()


def _rolling_maxdd(returns: pd.Series, window: int = ROLLING_WINDOW_DAYS) -> pd.Series:
    """Trailing-window maximum drawdown (negative value)."""
    def _mdd(x: np.ndarray) -> float:
        cum = np.cumprod(1 + x)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        return float(dd.min())

    mdd = returns.rolling(window, min_periods=window // 2).apply(_mdd, raw=True)
    return mdd.dropna()


def _regime_split(returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Split returns into high-vol / low-vol regimes using 20-day rolling vol."""
    vol20 = returns.rolling(20, min_periods=10).std()
    median_vol = vol20.median()
    high_vol = returns[vol20 >= median_vol].dropna()
    low_vol  = returns[vol20 < median_vol].dropna()
    return high_vol, low_vol


def _annualised_sharpe(returns: pd.Series) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * math.sqrt(ANNUALIZE))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_stability(
    equity_curve: pd.DataFrame,
    weights_history: Optional[List[Dict[str, Any]]] = None,
    window_months: int = 12,
) -> Dict[str, Any]:
    """Compute Rolling Stability Score (0-100) for a portfolio.

    Parameters
    ----------
    equity_curve     : DataFrame with at minimum a 'daily_return' column and
                       a 'date' column (or DatetimeIndex).
    weights_history  : Optional list of monthly rebalance records:
                       [{"date": date, "weights": {asset: w},
                         "avg_pairwise_corr": float}, ...]
                       Used for turnover and correlation-drift components.
                       If None, those components default to 50 (neutral).
    window_months    : Rolling window in months (default 12).

    Returns
    -------
    dict with keys:
        stability_score   : float ∈ [0, 100]   — overall composite score
        grade             : str                 — A/B/C/D/F letter grade
        components        : dict                — named component scores (0-100)
        rolling_metrics   : pd.DataFrame        — date-indexed rolling metrics
        summary           : dict                — key stats (min/max/std per metric)
        data_sufficiency  : str                 — 'sufficient'|'limited'|'insufficient'
    """
    _empty = {
        "stability_score": 0.0,
        "grade": "N/A",
        "components": {},
        "rolling_metrics": pd.DataFrame(),
        "summary": {},
        "data_sufficiency": "insufficient",
    }

    if equity_curve is None or equity_curve.empty:
        return _empty
    if "daily_return" not in equity_curve.columns:
        return _empty

    # ── Build DatetimeIndex return series ────────────────────────────────────
    eq = equity_curve.copy()
    try:
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.set_index("date").sort_index()
    except Exception:
        pass
    returns = eq["daily_return"].dropna().astype(float)

    window = min(window_months * 21, len(returns))   # 21 trading days ≈ 1 month
    n_days = len(returns)

    # Data-sufficiency flag
    if n_days < 63:        # < 3 months
        return {**_empty, "data_sufficiency": "insufficient"}
    elif n_days < 252:     # 3–12 months
        data_suff = "limited"
        window = n_days // 2
    else:
        data_suff = "sufficient"

    # ── 1. Rolling metrics ────────────────────────────────────────────────────
    roll_sharpe = _rolling_sharpe(returns, window=window)
    roll_cagr   = _rolling_cagr(returns,   window=window)
    roll_mdd    = _rolling_maxdd(returns,   window=window)

    rolling_df = pd.DataFrame({
        "rolling_sharpe": roll_sharpe,
        "rolling_cagr":   roll_cagr,
        "rolling_maxdd":  roll_mdd,
    }).dropna(how="all")

    # ── 2. Component A: Sharpe Consistency (25%) ──────────────────────────────
    # Low std of rolling Sharpe → high consistency
    if roll_sharpe.empty or len(roll_sharpe) < 3:
        score_sharpe = 50.0
    else:
        sharpe_std = float(roll_sharpe.std())
        # bad = std > 2.0  (swings from -2 to +2 Sharpe repeatedly)
        # good = std <= 0.0
        score_sharpe = _score(sharpe_std, bad_threshold=2.0, good_threshold=0.0)

    # ── 3. Component B: Drawdown Stability (25%) ──────────────────────────────
    # Low std of rolling MDD → predictable drawdown environment
    if roll_mdd.empty or len(roll_mdd) < 3:
        score_dd = 50.0
    else:
        mdd_std = float(roll_mdd.std())
        # bad = std > 0.20 (swings between -5% and -45% drawdown)
        # good = std <= 0.0
        score_dd = _score(mdd_std, bad_threshold=0.20, good_threshold=0.0)

    # ── 4. Component C: Turnover Penalty (20%) ───────────────────────────────
    # Lower turnover = less slippage / lower hidden cost
    if weights_history and len(weights_history) >= 2:
        turnovers: List[float] = []
        for i in range(1, len(weights_history)):
            old_w = weights_history[i - 1].get("weights", {})
            new_w = weights_history[i].get("weights", {})
            all_assets = set(old_w) | set(new_w)
            tv = sum(abs(new_w.get(a, 0.0) - old_w.get(a, 0.0)) for a in all_assets) / 2.0
            turnovers.append(tv)
        avg_turnover = float(np.mean(turnovers)) if turnovers else 0.5
        rolling_df["turnover"] = np.nan
        # Annotate rolling_df at rebalance dates
        for i, wh in enumerate(weights_history[1:], 1):
            try:
                d = pd.Timestamp(wh["date"])
                rolling_df.loc[d, "turnover"] = turnovers[i - 1]
            except Exception:
                pass
    else:
        avg_turnover = 0.5   # neutral if no data
        score_turnover = 50.0
        avg_turnover = None  # signal neutral

    if avg_turnover is not None:
        # bad = full turnover every month (1.0+)
        # good = no turnover (0.0)
        score_turnover = _score(avg_turnover, bad_threshold=1.0, good_threshold=0.0)
    # else already set to 50.0 above

    # ── 5. Component D: Correlation Stability (20%) ───────────────────────────
    if weights_history and len(weights_history) >= 3:
        corrs = [
            h.get("avg_pairwise_corr")
            for h in weights_history
            if h.get("avg_pairwise_corr") is not None
        ]
        if len(corrs) >= 3:
            corr_series = pd.Series(corrs, dtype=float)
            drift = corr_series.diff().dropna().abs()
            avg_corr_drift = float(drift.mean())
            rolling_df["corr_drift"] = np.nan
            # bad = avg drift > 0.30 per month
            # good = 0.0
            score_corr = _score(avg_corr_drift, bad_threshold=0.30, good_threshold=0.0)
        else:
            avg_corr_drift = None
            score_corr = 50.0
    else:
        avg_corr_drift = None
        score_corr = 50.0

    # ── 6. Component E: Regime Robustness (10%) ───────────────────────────────
    if n_days >= 120:
        r_high, r_low = _regime_split(returns)
        sh_high = _annualised_sharpe(r_high)
        sh_low  = _annualised_sharpe(r_low)
        raw_gap = abs(sh_high - sh_low)
        # Adjust gap for positive asymmetry 
        # (if worst regime didn't blow up relative to flatline, it's a good asymmetric payoff)
        min_sh = min(sh_high, sh_low)
        max_sh = max(sh_high, sh_low)
        
        if min_sh >= 0.0:
            # Both regimes were profitable or flat. The gap is entirely driven by massive upside.
            asymmetry_buffer = max_sh * 0.75  # Nullify 75% of upside-driven gap
            sharpe_gap_adjusted = max(0.0, raw_gap - asymmetry_buffer)
        elif min_sh > -0.5:
            # Barely lost money in the worst regime
            asymmetry_buffer = max_sh * 0.25
            sharpe_gap_adjusted = max(0.0, raw_gap - asymmetry_buffer)
        else:
            sharpe_gap_adjusted = raw_gap
            
        sharpe_gap = sharpe_gap_adjusted
        
        # Dynamic bad threshold: prevents massive Sharpe strategies getting penalized for nominally large gaps
        overall_sh = _annualised_sharpe(returns)
        bad_threshold = max(2.0, abs(overall_sh) * 1.5)
        score_regime = _score(sharpe_gap, bad_threshold=bad_threshold, good_threshold=0.0)
        rolling_df["regime_vol_flag"] = (
            returns.rolling(20, min_periods=10).std()
            >= returns.rolling(20, min_periods=10).std().median()
        ).astype(int)
    else:
        sh_high = sh_low = 0.0
        sharpe_gap = 0.0
        score_regime = 50.0

    # ── 7. Composite Stability Score ─────────────────────────────────────────
    stability_score = (
        W_SHARPE_CONSISTENCY * score_sharpe
        + W_DRAWDOWN_STABILITY * score_dd
        + W_TURNOVER_PENALTY  * score_turnover
        + W_CORR_STABILITY    * score_corr
        + W_REGIME_ROBUSTNESS * score_regime
    )
    stability_score = float(max(0.0, min(100.0, stability_score)))

    # Letter grade
    if stability_score >= 80:   grade = "A"
    elif stability_score >= 65: grade = "B"
    elif stability_score >= 50: grade = "C"
    elif stability_score >= 35: grade = "D"
    else:                       grade = "F"

    components = {
        "Sharpe Consistency (25%)":  round(score_sharpe,  1),
        "Drawdown Stability (25%)":  round(score_dd,      1),
        "Turnover Penalty (20%)":    round(score_turnover,1),
        "Correlation Stability (20%)": round(score_corr, 1),
        "Regime Robustness (10%)":   round(score_regime,  1),
    }

    summary = {
        "rolling_sharpe_mean": float(roll_sharpe.mean()) if not roll_sharpe.empty else None,
        "rolling_sharpe_std":  float(roll_sharpe.std())  if not roll_sharpe.empty else None,
        "rolling_sharpe_min":  float(roll_sharpe.min())  if not roll_sharpe.empty else None,
        "rolling_sharpe_max":  float(roll_sharpe.max())  if not roll_sharpe.empty else None,
        "rolling_cagr_mean":   float(roll_cagr.mean())   if not roll_cagr.empty else None,
        "rolling_maxdd_mean":  float(roll_mdd.mean())    if not roll_mdd.empty else None,
        "rolling_maxdd_worst": float(roll_mdd.min())     if not roll_mdd.empty else None,
        "avg_turnover":        float(avg_turnover) if avg_turnover is not None else None,
        "avg_corr_drift":      float(avg_corr_drift) if avg_corr_drift is not None else None,
        "regime_sharpe_high":  round(sh_high, 3),
        "regime_sharpe_low":   round(sh_low, 3),
        "regime_sharpe_gap":   round(sharpe_gap, 3),
        "n_days":              n_days,
    }

    return {
        "stability_score":  stability_score,
        "grade":            grade,
        "components":       components,
        "rolling_metrics":  rolling_df,
        "summary":          summary,
        "data_sufficiency": data_suff,
    }


def grade_to_colour(grade: str) -> str:
    """Map letter grade to a CSS hex colour for UI rendering."""
    return {"A": "#22c55e", "B": "#84cc16", "C": "#f59e0b",
            "D": "#f97316", "F": "#ef4444"}.get(grade, "#6b7280")


__all__ = [
    "compute_rolling_stability",
    "grade_to_colour",
    "W_SHARPE_CONSISTENCY",
    "W_DRAWDOWN_STABILITY",
    "W_TURNOVER_PENALTY",
    "W_CORR_STABILITY",
    "W_REGIME_ROBUSTNESS",
]
