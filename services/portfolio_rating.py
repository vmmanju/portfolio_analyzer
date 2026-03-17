"""services/portfolio_rating.py

Composite Portfolio Rating System (0–100).

FORMULA
-------
Composite Rating = weighted sum of 5 cross-sectionally normalised components:

  30%  Risk-adjusted return     — normalised Sharpe ratio
  20%  Drawdown control         — normalised (1 − |Max Drawdown|)
  20%  Stability Score          — from stability_analyzer (already 0-100)
  15%  Regime consistency       — normalised (1 − regime_sharpe_gap / 2)
  15%  Diversification quality  — normalised (1 − avg_pairwise_correlation)

Cross-sectional normalisation: Replaced by absolute robust scoring bounded to [0, 1]
so each component produces a universal score instead of an arbitrary peer-dependent one.

GRADE THRESHOLDS
----------------
  90–100 → A+        Exceptional
  80–89  → A         Excellent
  70–79  → B         Good
  60–69  → C         Average
  < 60   → D         Below average

WHY THIS FORMULA?
-----------------
The standard Sharpe/CAGR-only ranking ignores:
  • HOW RELIABLY that Sharpe was achieved over time (Stability)
  • WHETHER the portfolio behaved differently in high vs. low vol markets (Regime)
  • HOW MUCH diversification benefit the investor actually received (Correlation)
  • HOW PAINFUL the worst drawdown was (Max Drawdown)

Including all five dimensions produces a more holistic "quality score" that
penalises lucky one-period outliers and rewards robust, diversified strategies.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ── Component weights (must sum to 1.0) ───────────────────────────────────────
W_SHARPE       = 0.30
W_DRAWDOWN     = 0.20
W_STABILITY    = 0.15
W_REGIME       = 0.10
W_DIVERSIF     = 0.15
W_RRC          = 0.10

# Grade thresholds (inclusive lower bounds)
GRADE_THRESHOLDS = [
    (90.0, "A+"),
    (80.0, "A"),
    (70.0, "B"),
    (60.0, "C"),
    (0.0,  "D"),
]

# Regime gap "bad" reference: if gap > this, regime score → 0
REGIME_BAD_GAP = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _grade(score: float) -> str:
    for threshold, letter in GRADE_THRESHOLDS:
        if score >= threshold:
            return letter
    return "D"


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite_portfolio_rating(
    portfolio_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute composite rating for all portfolios in a peer group.

    Parameters
    ----------
    portfolio_metrics_df : DataFrame with one row per portfolio.
        Required columns:
            name              — portfolio name
            sharpe            — annualised Sharpe ratio
            max_drawdown      — max drawdown (negative value, e.g. -0.25)
            stability_score   — 0-100 composite stability score
        Optional columns (default to neutral if missing):
            regime_sharpe_gap — |Sharpe_high_vol - Sharpe_low_vol|  (lower = better)
            avg_pairwise_corr — avg pairwise correlation of holdings  (lower = better)

    Returns
    -------
    pd.DataFrame : Same rows, additional columns:
            composite_score     float 0-100
            grade               str   A+/A/B/C/D
            rank                int   1 = best
            sharpe_component    float 0-100 component score
            drawdown_component  float 0-100 component score
            stability_component float 0-100 component score
            regime_component    float 0-100 component score
            diversif_component  float 0-100 component score
    """
    if portfolio_metrics_df is None or portfolio_metrics_df.empty:
        return pd.DataFrame()

    df = portfolio_metrics_df.copy()
    n = len(df)

    # ── Raw ingredients ───────────────────────────────────────────────────────

    # Sharpe — higher is better
    sharpe_raw = df["sharpe"].apply(_safe_float)

    # Drawdown control — Max Drawdown is negative
    # 0 score for <= -40% (-0.40), 100 score for >= -5% (-0.05)
    maxdd_raw = df["max_drawdown"].apply(lambda x: _safe_float(x, -0.5))
    dd_control = 1.0 - (maxdd_raw.abs() - 0.05) / (0.40 - 0.05)
    dd_control = dd_control.clip(0.0, 1.0)

    # Stability score — already 0-100, convert to [0,1]
    stab_raw = df["stability_score"].apply(lambda x: _safe_float(x, 50.0)) / 100.0

    # Regime gap — lower gap is better → convert to "consistency" score [0,1]
    if "regime_sharpe_gap" in df.columns:
        gap_raw = df["regime_sharpe_gap"].apply(lambda x: _safe_float(x, 1.0))
    else:
        gap_raw = pd.Series(1.0, index=df.index)   # neutral
        
    # Scale bad_gap dynamically so a high-Sharpe portfolio (>2.6) isn't unfairly penalized
    # for having a gap of 2.0 between regimes (e.g. Sharpe 3 vs 5).
    dynamic_bad_gap = np.maximum(REGIME_BAD_GAP, sharpe_raw.abs() * 0.75)
    regime_consistency = 1.0 - (gap_raw / dynamic_bad_gap).clip(0.0, 1.0)

    # Diversification — avg pairwise correlation
    # 0 score for corr >= 0.8, 100 score for corr <= 0.2
    if "avg_pairwise_corr" in df.columns:
        corr_raw = df["avg_pairwise_corr"].apply(lambda x: _safe_float(x, 0.5))
    else:
        corr_raw = pd.Series(0.5, index=df.index)   # neutral 50% correlation
    diversif_raw = 1.0 - (corr_raw - 0.2) / (0.8 - 0.2)
    diversif_raw = diversif_raw.clip(0.0, 1.0)
    
    # Stability score — input 0-100, normalize [0, 1]
    stab_n = df["stability_score"].apply(lambda x: _safe_float(x, 50.0)) / 100.0

    # RRC Score — input 0-100, normalize [0, 1]
    if "rrc_score" in df.columns:
        rrc_n = df["rrc_score"].apply(lambda x: _safe_float(x, 50.0)) / 100.0
    else:
        rrc_n = 0.5

    # ── Absolute robust scoring [0, 1] ─────────────
    # Sharpe capped at 2.5 for a 100 score
    sharpe_n  = (sharpe_raw / 2.5).clip(0.0, 1.0)
    dd_n      = dd_control
    regime_n  = regime_consistency
    diversif_n = diversif_raw

    # ── Composite score 0–100 ─────────────────────────────────────────────────
    composite = (
        W_SHARPE    * sharpe_n   * 100.0 +
        W_DRAWDOWN  * dd_n       * 100.0 +
        W_STABILITY * stab_n     * 100.0 +
        W_REGIME    * regime_n   * 100.0 +
        W_DIVERSIF  * diversif_n * 100.0 +
        W_RRC       * rrc_n      * 100.0
    ).clip(0.0, 100.0)

    # ── Assemble output ───────────────────────────────────────────────────────
    df["composite_score"]     = composite.round(2)
    df["grade"]               = composite.apply(_grade)
    df["rank"]                = composite.rank(ascending=False, method="min").astype(int)
    df["sharpe_component"]    = (sharpe_n   * 100.0).round(1)
    df["drawdown_component"]  = (dd_n       * 100.0).round(1)
    df["stability_component"] = (stab_n     * 100.0).round(1)
    df["regime_component"]    = (regime_n   * 100.0).round(1)
    df["diversif_component"]  = (diversif_n * 100.0).round(1)
    df["rrc_component"]       = (rrc_n      * 100.0).round(1)

    # Sort by composite score descending
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


def build_rating_input_from_results(
    results: Dict[str, Dict[str, Any]],
    extra_rows: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Build the input DataFrame for compute_composite_portfolio_rating.

    Parameters
    ----------
    results       : Output from backtest_user_portfolios / construct_meta_portfolio.
                    Each value dict should have keys: metrics, stability.
    extra_rows    : Optional list of additional row dicts (e.g. for the meta-portfolio).

    Returns
    -------
    pd.DataFrame with columns: name, sharpe, max_drawdown, stability_score,
                                regime_sharpe_gap, avg_pairwise_corr.
    """
    rows = []
    for name, v in results.items():
        m    = v.get("metrics", {})
        stab = v.get("stability", {})
        summ = stab.get("summary", {})
        rows.append({
            "name":               name,
            "sharpe":             _safe_float(m.get("Sharpe")),
            "cagr":               _safe_float(m.get("CAGR")),
            "max_drawdown":       _safe_float(m.get("Max Drawdown"), -0.5),
            "volatility":         _safe_float(m.get("Volatility")),
            "calmar":             _safe_float(m.get("Calmar")),
            "stability_score":    _safe_float(stab.get("stability_score"), 50.0),
            "stability_grade":    stab.get("grade", "N/A"),
            "regime_sharpe_gap":  _safe_float(summ.get("regime_sharpe_gap"), 1.0),
            "avg_pairwise_corr":  _safe_float(v.get("avg_pairwise_corr"), 0.5),
            "rrc_score":          _safe_float(v.get("rrc_score"), 50.0),
            "warnings":           v.get("warnings", []),
        })

    if extra_rows:
        rows.extend(extra_rows)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def grade_colour(grade: str) -> str:
    """CSS hex colour for a composite grade (for UI rendering)."""
    return {
        "A+": "#059669",   # emerald-600
        "A":  "#22c55e",   # green-500
        "B":  "#84cc16",   # lime-500
        "C":  "#f59e0b",   # amber-500
        "D":  "#ef4444",   # red-500
    }.get(grade, "#6b7280")


def top_portfolio(rated_df: pd.DataFrame) -> Optional[str]:
    """Return the name of the #1-ranked portfolio, or None if df is empty."""
    if rated_df.empty or "rank" not in rated_df.columns:
        return None
    row = rated_df[rated_df["rank"] == 1]
    if row.empty:
        return None
    return str(row.iloc[0]["name"])


__all__ = [
    "compute_composite_portfolio_rating",
    "build_rating_input_from_results",
    "grade_colour",
    "top_portfolio",
    "W_SHARPE", "W_DRAWDOWN", "W_STABILITY", "W_REGIME", "W_DIVERSIF",
    "GRADE_THRESHOLDS",
]
