"""services/return_estimator.py

Bayesian Shrinkage Estimator for Expected Returns.

WHY SHRINKAGE?
--------------
Historical sample means are *extremely* noisy estimates of true expected returns.
For a stock with daily vol σ, the standard error of the sample mean over T days is
σ / √T.  With 90 trading days (≈ 4 months), a stock with 2 % daily vol has a
mean standard error of ≈ 0.2 % per day — the same order of magnitude as the mean
itself.  Plugging noisy means directly into a mean-variance optimizer causes it to
"over-fit" to estimation error, concentrating weight on high-noise high-mean assets
that happen to have been lucky over the sample window.

SHRINKAGE FORMULA (James-Stein / Stein estimator)
--------------------------------------------------
μ_shrunk_i = λ · μ_sample_i + (1 − λ) · μ_prior

  μ_sample_i  = sample mean return for asset i  (in-sample average)
  μ_prior     = cross-sectional mean of all μ_sample_i   (the "prior belief" that
                all assets share a common expected return — equivalent to using the
                equal-weight portfolio as a Bayesian prior)
  λ           = shrinkage intensity ∈ [0, 1]

PRACTICAL λ FORMULA
--------------------
The theoretically optimal λ requires estimating the signal-to-noise ratio, which is
itself uncertain.  The following closed-form approximation is standard in practice
(see Ledoit & Wolf 2004, Black & Litterman 1992):

  λ = T / (T + k)

  T = number of return observations used to estimate μ_sample
  k = shrinkage constant (default 60, roughly 3 months of trading days)

Intuition:
  • k ≪ T  →  λ → 1  →  trust the sample mean (short-window shrinkage)
  • k ≫ T  →  λ → 0  →  trust the prior (heavy shrinkage for short samples)
  • With T = 90 days and k = 60: λ = 90/150 = 0.60  (40 % toward prior)
  • With T = 252 days and k = 60: λ = 252/312 = 0.81 (19 % toward prior)

EFFECT ON OPTIMIZATION
-----------------------
Shrinkage pulls extreme sample means toward the grand cross-sectional mean,
reducing the chance that the optimizer over-weights a stock that happened to
have an anomalously high return by luck.  This is analogous to ridge-regression
regularisation, where the prior acts as an L2 penalty on deviations from the
common mean.

References
----------
• Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional
  covariance matrices." JMVA 88(2):365-411.
• James, W. & Stein, C. (1961). "Estimation with quadratic loss." Proceedings of
  the 4th Berkeley Symposium 1:361-379.
• Black, F. & Litterman, R. (1992). "Global portfolio optimization." Financial
  Analysts Journal 48(5):28-43.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ── Default hyperparameter ────────────────────────────────────────────────────
DEFAULT_SHRINKAGE_K: float = 60.0   # ≈ 3 months of trading days


# ─────────────────────────────────────────────────────────────────────────────
# Core public function
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_shrinkage_returns(
    returns_df: pd.DataFrame,
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    mu_prior: Optional[float] = None,
) -> pd.Series:
    """Bayesian (James-Stein) shrinkage estimator for expected daily returns.

    Parameters
    ----------
    returns_df   : DataFrame of daily (or periodic) returns.
                   Rows = time, columns = assets.
    shrinkage_k  : Shrinkage intensity constant k (default 60).
                   Higher k → more shrinkage toward the prior.
                   Lower k → closer to the raw sample mean.
    mu_prior     : Optional scalar prior for all assets.
                   If None (default), uses the cross-sectional mean of
                   sample means (equal-weight portfolio prior).

    Returns
    -------
    pd.Series : Shrunk expected daily return per asset (same index as
                returns_df.columns).  Returns empty Series if input is empty.

    Statistical guarantee
    ---------------------
    For any asset i:
        min(μ_sample_i, μ_prior) ≤ μ_shrunk_i ≤ max(μ_sample_i, μ_prior)
    The shrunk mean always lies strictly between the sample mean and the prior,
    so it can never be more extreme than either end-point.
    """
    if returns_df is None or returns_df.empty:
        return pd.Series(dtype=float)

    # Drop columns that are entirely NaN
    clean = returns_df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if clean.empty:
        return pd.Series(dtype=float)

    # ── Sample statistics ─────────────────────────────────────────────────────
    mu_sample: pd.Series = clean.mean(axis=0)          # E[r_i] per asset
    T: int = len(clean)                                  # number of observations

    # ── Prior return ──────────────────────────────────────────────────────────
    # Default: cross-sectional average — equivalent to equal-weight portfolio
    if mu_prior is None:
        mu_prior_val: float = float(mu_sample.mean())
    else:
        mu_prior_val = float(mu_prior)

    # ── Shrinkage intensity ───────────────────────────────────────────────────
    # λ = T / (T + k)  ∈ (0, 1)
    # Clamped to [0, 1] as a defensive guard; analytically always in (0, 1)
    lam: float = float(T) / (float(T) + float(shrinkage_k))
    lam = float(np.clip(lam, 0.0, 1.0))

    # ── Shrinkage formula ─────────────────────────────────────────────────────
    # μ_shrunk_i = λ · μ_sample_i + (1 − λ) · μ_prior
    mu_shrunk: pd.Series = lam * mu_sample + (1.0 - lam) * mu_prior_val

    return mu_shrunk


def shrunk_annualised_returns(
    returns_df: pd.DataFrame,
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    annualize: int = 252,
    mu_prior: Optional[float] = None,
) -> pd.Series:
    """Convenience wrapper: shrunk expected returns, annualised.

    Equivalent to bayesian_shrinkage_returns(...) * annualize.
    """
    daily = bayesian_shrinkage_returns(returns_df, shrinkage_k=shrinkage_k, mu_prior=mu_prior)
    return daily * annualize


def shrinkage_lambda(T: int, k: float = DEFAULT_SHRINKAGE_K) -> float:
    """Return the shrinkage intensity λ = T / (T + k).

    Useful for logging / diagnostics without running the full estimator.

    >>> round(shrinkage_lambda(90, 60), 4)
    0.6
    >>> round(shrinkage_lambda(252, 60), 4)
    0.8077
    """
    if T <= 0 or k < 0:
        return 0.0
    return float(T) / (float(T) + float(k))


__all__ = [
    "bayesian_shrinkage_returns",
    "shrunk_annualised_returns",
    "shrinkage_lambda",
    "DEFAULT_SHRINKAGE_K",
]


# ─────────────────────────────────────────────────────────────────────────────
# Error-Model Correction Layer
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Any, Dict, Optional as _Opt

_ec_logger = logging.getLogger(__name__)


def apply_error_correction(
    predicted_returns: pd.Series,
    factor_scores: pd.DataFrame,
    calibration: _Opt[Dict[str, Any]] = None,
    use_correction: _Opt[bool] = None,
) -> pd.Series:
    """Subtract the calibrated systematic prediction bias from predicted returns.

    The error model estimates the *expected systematic bias* at the current
    factor-score level:

        bias_i = β0 + β1·MOM_i + β2·QUAL_i + β3·VAL_i + β4·VOL_i

    Corrected return:

        μ_corrected_i = μ_predicted_i − bias_i

    Parameters
    ----------
    predicted_returns : pd.Series
        Bayesian-shrunk expected returns indexed by stock_id.
    factor_scores : pd.DataFrame
        DataFrame with columns [momentum_score, quality_score, value_score,
        volatility_score] and index = stock_id.  Missing columns default to 0.
    calibration : dict | None
        Output of get_current_coefficients() or update_error_model_if_due().
        If None, fetched automatically from DB.
        If the dict's calibration_source == 'none', correction is skipped.
    use_correction : bool | None
        Override the config flag USE_ERROR_CORRECTION.
        None → read from settings.USE_ERROR_CORRECTION.

    Returns
    -------
    pd.Series — corrected expected returns (same index as predicted_returns).
    If correction is disabled or no calibration exists, returns
    predicted_returns unchanged.

    No Look-Ahead
    -------------
    The calibration coefficients come from a rolling 5-year window ending at
    most at the current date.  The function never loads future price or factor
    data — it uses only the passed-in factor_scores and the stored coefficients.
    """
    # ── Config guard ──────────────────────────────────────────────────────────
    if use_correction is None:
        try:
            from app.config import settings
            use_correction = settings.USE_ERROR_CORRECTION
        except Exception:
            use_correction = True   # default safe-fail: apply correction

    if not use_correction:
        _ec_logger.debug("Error correction disabled via USE_ERROR_CORRECTION=False — skipping.")
        return predicted_returns

    # ── Edge-case: empty input ─────────────────────────────────────────────────
    if predicted_returns is None or predicted_returns.empty:
        return predicted_returns

    # ── Load calibration if not supplied ──────────────────────────────────────
    if calibration is None:
        try:
            from services.error_model import get_current_coefficients
            calibration = get_current_coefficients()
        except Exception as exc:
            _ec_logger.warning(
                "apply_error_correction: could not fetch calibration — skipping. (%s)", exc
            )
            return predicted_returns

    source = calibration.get("calibration_source", "none")
    if source == "none":
        _ec_logger.info(
            "apply_error_correction: no calibration in DB yet — correction skipped."
        )
        return predicted_returns

    calib_date = calibration.get("window_end")
    _ec_logger.info(
        "Error correction applied using calibration dated %s  (source=%s)",
        calib_date, source,
    )

    # ── Extract coefficients ───────────────────────────────────────────────────
    intercept = float(calibration.get("intercept") or 0.0)
    coefs      = calibration.get("coefficients") or {}
    beta_mom   = float(coefs.get("momentum")   or 0.0)
    beta_qual  = float(coefs.get("quality")    or 0.0)
    beta_val   = float(coefs.get("value")      or 0.0)
    beta_vol   = float(coefs.get("volatility") or 0.0)

    import math
    if all(math.isnan(v) for v in [intercept, beta_mom, beta_qual, beta_val, beta_vol]
           if not isinstance(v, float) or True):
        _ec_logger.warning("apply_error_correction: all coefficients are NaN — skipping.")
        return predicted_returns

    # ── Align factor scores to predicted-return index ─────────────────────────
    common_idx = predicted_returns.index.intersection(
        factor_scores.index if factor_scores is not None and not factor_scores.empty
        else predicted_returns.index
    )

    bias = pd.Series(intercept, index=predicted_returns.index, dtype=float)

    if (
        factor_scores is not None
        and not factor_scores.empty
        and len(common_idx) > 0
    ):
        fs_aligned = factor_scores.reindex(common_idx)

        def _col(name: str) -> pd.Series:
            if name in fs_aligned.columns:
                return fs_aligned[name].fillna(0.0)
            return pd.Series(0.0, index=common_idx)

        factor_bias = (
            beta_mom  * _col("momentum_score")
            + beta_qual * _col("quality_score")
            + beta_val  * _col("value_score")
            + beta_vol  * _col("volatility_score")
        )
        bias.loc[common_idx] += factor_bias

    corrected = predicted_returns - bias

    # ── Diagnostics ───────────────────────────────────────────────────────────
    n = len(corrected)
    mean_bias = float(bias.mean())
    _ec_logger.info(
        "Error correction: n=%d stocks  mean_bias=%+.6f  "
        "β0=%.4f  β_MOM=%.4f  β_QUAL=%.4f  β_VAL=%.4f  β_VOL=%.4f",
        n, mean_bias, intercept, beta_mom, beta_qual, beta_val, beta_vol,
    )

    return corrected


def corrected_returns(
    returns_df: pd.DataFrame,
    factor_scores: pd.DataFrame,
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    annualize: int = 252,
    mu_prior: _Opt[float] = None,
    calibration: _Opt[Dict[str, Any]] = None,
    use_correction: _Opt[bool] = None,
) -> pd.Series:
    """Bayesian-shrunk + error-corrected expected annualised returns.

    This is the primary function to use at optimization / ranking time.

    Pipeline
    --------
    1. Compute shrunk daily expected returns via bayesian_shrinkage_returns().
    2. Annualise by × annualize.
    3. Call apply_error_correction() to subtract the systematic factor bias.

    Parameters
    ----------
    returns_df    : Daily return matrix (rows=time, cols=stock_id).
    factor_scores : DataFrame with factor score columns, index=stock_id.
                    If empty or None, the factor terms default to 0 (only
                    the intercept bias β0 is removed).
    shrinkage_k   : Shrinkage constant (default 60).
    annualize     : Annualisation multiplier (default 252 trading days).
    mu_prior      : Optional scalar prior for all assets.
    calibration   : Pre-loaded calibration dict; fetched from DB if None.
    use_correction: Override config USE_ERROR_CORRECTION flag.

    Returns
    -------
    pd.Series — corrected annualised expected returns per stock (index=stock_id).
    """
    # Step 1 + 2: shrunk annualised returns
    ann = shrunk_annualised_returns(
        returns_df,
        shrinkage_k=shrinkage_k,
        annualize=annualize,
        mu_prior=mu_prior,
    )

    if ann.empty:
        return ann

    # Step 3: error correction
    return apply_error_correction(
        predicted_returns=ann,
        factor_scores=factor_scores if factor_scores is not None else pd.DataFrame(),
        calibration=calibration,
        use_correction=use_correction,
    )


__all__ = [
    # Core shrinkage estimators
    "bayesian_shrinkage_returns",
    "shrunk_annualised_returns",
    "shrinkage_lambda",
    "DEFAULT_SHRINKAGE_K",
    # Error-correction layer
    "apply_error_correction",
    "corrected_returns",
]

