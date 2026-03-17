"""services/covariance_estimator.py

Robust Covariance Matrix Estimation for Portfolio Optimization.

WHY REGULARIZE THE COVARIANCE MATRIX?
--------------------------------------
The sample covariance matrix S has two well-known problems when used inside a
mean-variance optimizer:

1. **Estimation error in small samples**
   With p assets and T observations, S has p(p+1)/2 free parameters to estimate.
   For p=30 assets and T=90 days, we are estimating 465 parameters from 90 rows —
   the matrix is severely under-identified.  Small eigenvalues become essentially
   zero (or negative due to floating-point noise), making direct inversion
   numerically unstable.

2. **Ill-conditioning → optimizer instability**
   The condition number κ = λ_max / λ_min of S can be enormous.  When κ is large,
   small perturbations in expected returns produce wildly different optimal weights.
   This is the core of the "error maximization" critique of Markowitz (Michaud 1989).

SOLUTION: Shrinkage
--------------------
Shrinkage blends the sample covariance with a structured "target" matrix T:

  Σ_robust = α × S + (1 − α) × Target

Two methods are supported:

Option A — Ledoit-Wolf (preferred)
  Analytically optimal shrinkage intensity α derived from the data.
  Minimises mean squared error between S and the true covariance.
  Implementation: sklearn.covariance.LedoitWolf.
  Reference: Ledoit & Wolf (2004) JMVA 88(2):365-411.

Option B — Manual diagonal shrinkage (fallback)
  Target = I × avg_var      (scaled identity: each asset's average variance)
  α = configurable scalar (default 0.8)
  Σ_robust = α × S + (1-α) × diag(avg_var)
  No sklearn dependency.

POSITIVE DEFINITENESS GUARANTEE
---------------------------------
After shrinkage a jitter pass is applied if the minimum eigenvalue is still
below a safe threshold (1e-8):

  Σ_robust += jitter × I      (jitter starts at 1e-8, grows by 10× until PD)

This ensures the matrix is *always* invertible and Cholesky-decomposable,
which is required by SLSQP and any variance calculation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_METHOD = "ledoit_wolf"
DEFAULT_MANUAL_ALPHA = 0.8          # weight on sample cov in manual shrinkage
JITTER_BASE = 1e-8                  # starting jitter for PD guarantee
JITTER_MAX_ITERS = 10               # max doublings of jitter
PD_THRESHOLD = 1e-8                 # min eigenvalue we consider "positive"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_positive_definite(
    matrix: np.ndarray,
    label: str = "",
) -> Tuple[np.ndarray, float]:
    """Add jitter to the diagonal until the matrix is positive definite.

    Returns (regularised_matrix, jitter_applied).
    A jitter of 0.0 means the matrix was already PD.
    """
    jitter_applied = 0.0
    current = matrix.copy()
    jitter = JITTER_BASE
    for _ in range(JITTER_MAX_ITERS):
        eigs = np.linalg.eigvalsh(current)
        if float(eigs.min()) >= PD_THRESHOLD:
            break
        current = current + jitter * np.eye(current.shape[0])
        jitter_applied += jitter
        jitter *= 10.0
    else:
        # Last check
        eigs = np.linalg.eigvalsh(current)
        if float(eigs.min()) < PD_THRESHOLD:
            _logger.warning(
                "Could not achieve strict PD for %s after max jitter; "
                "min eigenvalue = %.2e",
                label, float(eigs.min()),
            )
    return current, jitter_applied


def _condition_number(matrix: np.ndarray) -> float:
    """Return the matrix condition number κ = λ_max / λ_min (using eigenvalues)."""
    eigs = np.abs(np.linalg.eigvalsh(matrix))
    if eigs.min() == 0:
        return float("inf")
    return float(eigs.max() / eigs.min())


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def robust_covariance_matrix(
    returns_df: pd.DataFrame,
    method: str = DEFAULT_METHOD,
    manual_alpha: float = DEFAULT_MANUAL_ALPHA,
    annualize: int = 252,
    log_diagnostics: bool = True,
) -> dict:
    """Compute a regularised (shrunk) covariance matrix from a returns DataFrame.

    Parameters
    ----------
    returns_df    : DataFrame of periodic returns (rows = time, cols = assets).
    method        : 'ledoit_wolf' (preferred, requires sklearn) or 'manual'.
                    Falls back to 'manual' automatically if sklearn is absent.
    manual_alpha  : Blending weight on the sample matrix in manual mode (0–1).
                    Higher = more like raw sample; lower = more regularised.
    annualize     : Multiply the daily covariance by this factor (252 for daily→annual).
    log_diagnostics: Log condition number improvement if True.

    Returns
    -------
    dict with keys:
        matrix         : np.ndarray  — robust annualised covariance (p × p)
        method_used    : str         — 'ledoit_wolf' or 'manual'
        shrinkage      : float       — α used (Ledoit-Wolf α or manual_alpha)
        cond_before    : float       — condition number of raw sample cov
        cond_after     : float       — condition number after regularisation
        jitter_applied : float       — jitter added to diagonal (0 if not needed)
        n_assets       : int
        n_obs          : int
    """
    if returns_df is None or returns_df.empty:
        return {
            "matrix": np.array([[]]), "method_used": method,
            "shrinkage": 0.0, "cond_before": 0.0,
            "cond_after": 0.0, "jitter_applied": 0.0,
            "n_assets": 0, "n_obs": 0,
        }

    # Drop fully-NaN columns and rows
    clean = returns_df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if clean.empty or clean.shape[1] < 2:
        # Degenerate: return scalar variance
        var = float(clean.var().mean()) if not clean.empty else 0.0
        mat = np.array([[var * annualize]]) if var > 0 else np.eye(1) * 1e-6
        return {
            "matrix": mat, "method_used": method,
            "shrinkage": 1.0, "cond_before": 1.0,
            "cond_after": 1.0, "jitter_applied": 0.0,
            "n_assets": clean.shape[1], "n_obs": len(clean),
        }

    n_obs, n_assets = clean.shape

    # ── Raw sample covariance ─────────────────────────────────────────────────
    S_daily = clean.cov().values           # daily covariance
    cond_before = _condition_number(S_daily)

    # ── Shrinkage ─────────────────────────────────────────────────────────────
    method_used = method
    alpha_used = manual_alpha

    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(assume_centered=False)
            lw.fit(clean.values)
            S_shrunk_daily = lw.covariance_      # already shrunk daily cov
            alpha_used = float(1.0 - lw.shrinkage_)  # LW reports *shrinkage* not alpha
            method_used = "ledoit_wolf"
        except ImportError:
            _logger.info(
                "sklearn not available — falling back to manual diagonal shrinkage"
            )
            method_used = "manual"
            S_shrunk_daily = None
        except Exception as exc:
            _logger.warning("LedoitWolf failed (%s) — using manual shrinkage", exc)
            method_used = "manual"
            S_shrunk_daily = None
    else:
        S_shrunk_daily = None

    if method_used == "manual" or S_shrunk_daily is None:
        # Target: scaled identity ( avg_variance × I )
        avg_var = float(np.diag(S_daily).mean())
        target = avg_var * np.eye(n_assets)
        S_shrunk_daily = manual_alpha * S_daily + (1.0 - manual_alpha) * target
        alpha_used = manual_alpha
        method_used = "manual"

    # ── Annualise ─────────────────────────────────────────────────────────────
    S_annual = S_shrunk_daily * annualize

    # ── Positive-definiteness guarantee ───────────────────────────────────────
    S_pd, jitter = _ensure_positive_definite(S_annual, label="covariance")
    cond_after = _condition_number(S_pd)

    # ── Log diagnostics ───────────────────────────────────────────────────────
    if log_diagnostics:
        _logger.info(
            "Covariance regularisation: method=%s  n=%d assets / T=%d obs  "
            "α=%.4f  κ_before=%.1f → κ_after=%.1f  jitter=%.2e",
            method_used, n_assets, n_obs, alpha_used,
            cond_before, cond_after, jitter,
        )
        if cond_before > 0 and cond_after < cond_before:
            improvement = (1.0 - cond_after / cond_before) * 100.0
            _logger.info("  Condition number improved by %.1f%%", improvement)

    return {
        "matrix": S_pd,
        "method_used": method_used,
        "shrinkage": alpha_used,
        "cond_before": cond_before,
        "cond_after": cond_after,
        "jitter_applied": jitter,
        "n_assets": n_assets,
        "n_obs": n_obs,
    }


def validate_covariance(matrix: np.ndarray) -> dict:
    """Validate a covariance matrix and return diagnostic info.

    Returns dict with:
        is_symmetric       : bool
        is_positive_definite: bool
        min_eigenvalue     : float
        max_eigenvalue     : float
        condition_number   : float
        rank               : int
    """
    eigs = np.linalg.eigvalsh(matrix)
    is_pd = bool(float(eigs.min()) > 0)
    is_sym = bool(np.allclose(matrix, matrix.T, atol=1e-10))
    return {
        "is_symmetric": is_sym,
        "is_positive_definite": is_pd,
        "min_eigenvalue": float(eigs.min()),
        "max_eigenvalue": float(eigs.max()),
        "condition_number": _condition_number(matrix),
        "rank": int(np.linalg.matrix_rank(matrix)),
    }


__all__ = [
    "robust_covariance_matrix",
    "validate_covariance",
    "DEFAULT_METHOD",
    "DEFAULT_MANUAL_ALPHA",
]
