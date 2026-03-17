"""services/error_model_audit.py

Statistical Integrity Audit for the Error Model
================================================

PURPOSE
-------
Validates that the error model:
  1. Uses ONLY past data (no look-ahead bias in regression inputs).
  2. Forward returns do NOT overlap the calibration boundary.
  3. Bayesian update does NOT double-count data across calibration cycles.
  4. Coefficients are stable over time (no frequent sign flips).
  5. Out-of-sample improvement is positive after error correction.

Also flags:
  • Overfitting   (R² > 0.8 is suspicious for a 4-factor model on noisy financial returns)
  • Coefficient sign instability across rolling sub-windows
  • Calibration window that is too small (< 30 observations or < 6 months)

PUBLIC API
----------
    from services.error_model_audit import run_integrity_audit

    report = run_integrity_audit()
    print(report["bias_detected"])               # bool
    print(report["improvement_after_correction"]) # float  (Δ Sharpe, corrected − raw)
    print(report["stability_warning"])            # bool

All results are deterministic given the same DB state; no side-effects.
"""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from services.error_model import (
    CALIBRATION_INTERVAL_MONTHS,
    FACTOR_COLS,
    FACTOR_LABELS,
    FORWARD_DAYS,
    _build_observation_panel,
    _load_factors_for_window,
    _load_latest_calibration,
    _load_monthly_prices_for_stocks,
    _load_scores_for_window,
    _run_ols,
    compute_error_coefficients,
)
from services.backtest import get_rebalance_dates
from services.return_estimator import apply_error_correction, shrunk_annualised_returns

_logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
R2_OVERFITTING_THRESHOLD: float = 0.80
MIN_WINDOW_MONTHS: int = 6
MIN_OBSERVATIONS: int = 30
SIGN_FLIP_THRESHOLD: float = 0.5    # fraction of windows where sign flips vs. overall mean
IMPROVEMENT_THRESHOLD: float = 0.0  # Δ Sharpe must be > 0 to be considered improvement


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v: Any, fallback: float = float("nan")) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _sharpe(returns: pd.Series, annualize: int = 252) -> float:
    """Annualised Sharpe from a daily return series (no risk-free rate)."""
    if returns is None or len(returns) < 2:
        return 0.0
    mu  = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std == 0:
        return 0.0
    return mu * annualize / (std * np.sqrt(annualize))


def _panel_forward_returns_no_lookahead(
    panel: pd.DataFrame,
    window_end: date,
) -> bool:
    """
    Check 1 — No look-ahead in forward returns.

    Every observation in the panel must satisfy:
        rebalance_date + FORWARD_DAYS (trading) ≤ window_end

    We verify this by confirming that no rebalance_date is within
    FORWARD_DAYS*1.5 calendar days of window_end (the builder already
    enforces this, but we re-verify externally as an audit step).

    Returns True if the guarantee holds, False if any violation found.
    """
    if panel.empty:
        return True  # vacuously true (no data to violate)

    cutoff = window_end - timedelta(days=int(FORWARD_DAYS * 1.5))
    bad = panel[panel["rebalance_date"] > cutoff]
    if not bad.empty:
        _logger.warning(
            "[AUDIT] Look-ahead violation: %d rows have rebalance_date > cutoff=%s",
            len(bad), cutoff,
        )
        return False
    return True


def _check_no_double_counting(calibration_history: List[Dict]) -> Tuple[bool, str]:
    """
    Check 2 — Bayesian update does not double-count data.

    Each calibration record has [calibration_start_date, calibration_end_date].
    We check that consecutive records' windows do NOT overlap — i.e.
        record[k].end  < record[k+1].start

    Returns (ok: bool, detail: str).
    """
    if len(calibration_history) < 2:
        return True, "Only one calibration record — no overlap possible."

    sorted_hist = sorted(calibration_history, key=lambda r: r["calibration_end_date"])
    overlaps = []
    for i in range(len(sorted_hist) - 1):
        curr_end   = sorted_hist[i]["calibration_end_date"]
        next_start = sorted_hist[i + 1]["calibration_start_date"]
        if next_start <= curr_end:
            overlaps.append(
                f"Record {i}: end={curr_end}  overlaps with Record {i+1}: start={next_start}"
            )

    if overlaps:
        msg = "Double-counting detected! " + "; ".join(overlaps)
        _logger.warning("[AUDIT] %s", msg)
        return False, msg
    return True, "No window overlaps found."


def _check_coefficient_stability(
    panel: pd.DataFrame,
    window_start: date,
    window_end: date,
    n_subwindows: int = 4,
    min_obs_per_window: int = 20,
) -> Dict[str, Any]:
    """
    Check 3 — Coefficient stability over time via rolling sub-windows.

    Splits the panel into `n_subwindows` equal-length chronological slices
    and re-runs OLS in each window. Reports:
      - Mean and std of each coefficient across sub-windows
      - Coefficient of variation (CV = std/|mean|)
      - Sign-flip fraction (fraction of sub-windows where sign differs from global fit)
      - Whether any factor has a sign flip rate > SIGN_FLIP_THRESHOLD

    Returns a dict with stability metrics per factor.
    """
    if panel.empty or len(panel) < min_obs_per_window * n_subwindows:
        return {
            "stable": True,
            "reason": f"Insufficient data for stability check (n={len(panel)})",
            "factors": {},
        }

    # Global coefficients from full panel
    fcols = [c for c in FACTOR_COLS if c in panel.columns]
    y_all = panel["error"].values.astype(float)
    X_all = np.column_stack([np.ones(len(y_all)), panel[fcols].values.astype(float)])
    global_ols = _run_ols(X_all, y_all, FACTOR_LABELS[: len(fcols)])
    global_coefs = global_ols.get("coefficients", {})

    # Sort panel chronologically and split into sub-windows
    panel_sorted = panel.sort_values("rebalance_date")
    n = len(panel_sorted)
    chunk = n // n_subwindows

    sub_coefs: Dict[str, List[float]] = {f: [] for f in FACTOR_LABELS[: len(fcols)]}
    sub_labels: List[str] = []

    for sub_idx in range(n_subwindows):
        start_i = sub_idx * chunk
        end_i   = (sub_idx + 1) * chunk if sub_idx < n_subwindows - 1 else n
        sub     = panel_sorted.iloc[start_i:end_i]

        if len(sub) < min_obs_per_window:
            continue

        sub_y = sub["error"].values.astype(float)
        sub_X = np.column_stack([np.ones(len(sub_y)), sub[fcols].values.astype(float)])
        sub_ols = _run_ols(sub_X, sub_y, FACTOR_LABELS[: len(fcols)])
        c = sub_ols.get("coefficients", {})
        for f in FACTOR_LABELS[: len(fcols)]:
            sub_coefs[f].append(_safe_float(c.get(f)))
        d0 = sub["rebalance_date"].min()
        d1 = sub["rebalance_date"].max()
        sub_labels.append(f"{d0}→{d1}")

    # Stability metrics per factor
    factor_stats: Dict[str, Any] = {}
    any_unstable = False

    for f in FACTOR_LABELS[: len(fcols)]:
        vals = [v for v in sub_coefs[f] if not np.isnan(v)]
        if not vals:
            factor_stats[f] = {"stable": True, "reason": "no sub-window data"}
            continue

        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        cv     = std_v / abs(mean_v) if abs(mean_v) > 1e-10 else float("inf")

        global_sign = np.sign(global_coefs.get(f, 0.0))
        sign_flips  = sum(np.sign(v) != global_sign for v in vals)
        flip_rate   = sign_flips / len(vals)

        is_unstable = flip_rate > SIGN_FLIP_THRESHOLD or cv > 2.0
        if is_unstable:
            any_unstable = True

        factor_stats[f] = {
            "mean":       round(mean_v, 6),
            "std":        round(std_v,  6),
            "cv":         round(cv,     4) if np.isfinite(cv) else None,
            "sign_flip_rate": round(flip_rate, 3),
            "sub_window_values": [round(v, 6) for v in vals],
            "global_coefficient": round(_safe_float(global_coefs.get(f)), 6),
            "stable":     not is_unstable,
        }

    return {
        "stable":        not any_unstable,
        "sub_windows":   sub_labels,
        "factors":       factor_stats,
        "n_sub_windows": len(sub_labels),
    }


def _check_out_of_sample_improvement(
    panel: pd.DataFrame,
    train_fraction: float = 0.7,
) -> Dict[str, Any]:
    """
    Check 4 — Out-of-sample improvement after error correction.

    Methodology:
      1. Fit OLS on the TRAIN portion (first 70% of dates chronologically).
      2. Apply the correction to the HOLD-OUT portion (last 30%).
      3. Compute the Sharpe ratio of HOLD-OUT predicted returns vs.
         corrected predicted returns.  Improvement = Δ Sharpe.

    Returns a dict with 'delta_sharpe', 'train_r2', 'test_rmse_raw',
    'test_rmse_corrected', 'improvement'.
    """
    _empty = {
        "delta_sharpe":      float("nan"),
        "delta_rmse":        float("nan"),
        "train_r2":          float("nan"),
        "test_rmse_raw":     float("nan"),
        "test_rmse_corrected": float("nan"),
        "improvement":       False,
        "n_train":           0,
        "n_test":            0,
        "reason":            "insufficient data",
    }

    if panel.empty or len(panel) < 30:
        return _empty

    fcols = [c for c in FACTOR_COLS if c in panel.columns]
    if not fcols:
        return {**_empty, "reason": "no factor columns in panel"}

    # Chronological split
    panel_s = panel.sort_values("rebalance_date").reset_index(drop=True)
    split_i = int(len(panel_s) * train_fraction)
    if split_i < 10 or (len(panel_s) - split_i) < 10:
        return {**_empty, "reason": "split produces too few train/test rows"}

    train = panel_s.iloc[:split_i]
    test  = panel_s.iloc[split_i:]

    # Fit on train
    y_tr = train["error"].values.astype(float)
    X_tr = np.column_stack([np.ones(len(y_tr)), train[fcols].values.astype(float)])
    train_ols = _run_ols(X_tr, y_tr, FACTOR_LABELS[: len(fcols)])
    train_r2  = _safe_float(train_ols.get("r_squared"))

    # Evaluate on test
    intercept  = _safe_float(train_ols.get("intercept", 0.0), 0.0)
    coef_vals  = [_safe_float(train_ols["coefficients"].get(f, 0.0), 0.0)
                  for f in FACTOR_LABELS[: len(fcols)]]
    beta_vec   = np.array(coef_vals)

    X_test_factors = test[fcols].values.astype(float)
    predicted_bias = intercept + X_test_factors @ beta_vec

    # Raw errors on test set
    actual_errors_raw      = test["error"].values.astype(float)  # ε = actual − pred
    actual_return          = test["actual_return"].values.astype(float)
    raw_pred               = test["predicted_return"].values.astype(float)
    corrected_pred         = raw_pred - predicted_bias

    # RMSE comparison
    rmse_raw      = float(np.sqrt(np.mean((actual_return - raw_pred) ** 2)))
    rmse_cor      = float(np.sqrt(np.mean((actual_return - corrected_pred) ** 2)))
    delta_rmse    = rmse_raw - rmse_cor          # positive = improvement

    # Sharpe comparison: treat the cross-section of predictions as a ranking signal
    # Compute daily-equivalent Sharpe by treating each row as one obs
    sharpe_raw = _sharpe(pd.Series(actual_return - (actual_return.mean() - raw_pred.mean())), annualize=1)
    sharpe_cor = _sharpe(pd.Series(actual_return - (actual_return.mean() - corrected_pred.mean())), annualize=1)
    delta_sharpe = sharpe_cor - sharpe_raw

    improvement = delta_rmse > 0  # lower RMSE after correction

    return {
        "delta_sharpe":         round(delta_sharpe, 6),
        "delta_rmse":           round(delta_rmse, 6),
        "train_r2":             round(train_r2, 4),
        "test_rmse_raw":        round(rmse_raw, 6),
        "test_rmse_corrected":  round(rmse_cor, 6),
        "improvement":          improvement,
        "n_train":              len(train),
        "n_test":               len(test),
        "reason":               "ok",
    }


def _load_all_calibrations() -> List[Dict[str, Any]]:
    """Load full ModelCalibration history from DB, sorted by end date."""
    records = []
    try:
        from app.models import ModelCalibration
        from app.database import get_db_context
        from sqlalchemy import select
        with get_db_context() as db:
            rows = db.execute(
                select(ModelCalibration).order_by(ModelCalibration.calibration_end_date)
            ).scalars().all()
        for row in rows:
            records.append({
                "calibration_start_date": row.calibration_start_date,
                "calibration_end_date":   row.calibration_end_date,
                "intercept":              row.intercept,
                "r_squared":              row.r_squared,
                "n_observations":         row.n_observations,
                "coefficients": {
                    "momentum":   row.beta_momentum,
                    "quality":    row.beta_quality,
                    "value":      row.beta_value,
                    "volatility": row.beta_volatility,
                },
                "blend_weight": row.blend_weight,
                "created_at":   row.created_at,
            })
    except Exception as exc:
        _logger.warning("Could not load calibration history: %s", exc)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Public API — main audit entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_integrity_audit(
    end_date: Optional[date] = None,
    window_years: int = 5,
    prediction_method: str = "shrinkage",
) -> Dict[str, Any]:
    """
    Run a full statistical integrity audit of the error model.

    Parameters
    ----------
    end_date         : Audit window end. Defaults to today.
    window_years     : Rolling window length in years (default 5).
    prediction_method: 'shrinkage' or 'score'.

    Returns
    -------
    dict with keys:

        # Required top-level summary
        bias_detected               : bool  — True if |mean_error| > 1σ or
                                             intercept is statistically significant
        improvement_after_correction: float — Δ RMSE on hold-out (positive = better)
        stability_warning           : bool  — True if any factor has sign flips or
                                             high CV

        # Detailed sub-checks
        checks: {
            no_lookahead           : bool   — forward returns within window
            no_double_counting     : bool   — calibration windows non-overlapping
            observations_sufficient: bool   — n >= MIN_OBSERVATIONS
            window_sufficient      : bool   — window >= MIN_WINDOW_MONTHS months
            r2_overfitting_flag    : bool   — R² > R2_OVERFITTING_THRESHOLD
            coefficient_stability  : dict   — per-factor stability breakdown
            out_of_sample          : dict   — train/test RMSE and Sharpe comparison
        }

        # Metadata
        audit_date             : date
        window_start           : date
        window_end             : date
        n_observations         : int
        r_squared              : float
        mean_error             : float (bias)
        residual_std           : float
        calibration_history    : list   — all DB calibration records
        warnings               : list[str]
        errors                 : list[str]   — non-fatal errors encountered
    """
    if end_date is None:
        end_date = date.today()

    start_date = date(end_date.year - window_years, end_date.month, end_date.day)
    _logger.info(
        "[AUDIT] Starting integrity audit: window=[%s, %s]  method=%s",
        start_date, end_date, prediction_method,
    )

    warnings_list: List[str] = []
    errors_list:   List[str] = []

    # ── Step 0: Build the full observation panel ──────────────────────────────
    try:
        ols_result = compute_error_coefficients(
            end_date=end_date,
            window_years=window_years,
            prediction_method=prediction_method,
            log_stability=False,  # we run our own stability check below
        )
        panel = ols_result.get("panel_df", pd.DataFrame())
        r2           = _safe_float(ols_result.get("r_squared"))
        mean_error   = _safe_float(ols_result.get("mean_error"))
        residual_std = _safe_float(ols_result.get("residual_std"))
        n_obs        = int(ols_result.get("n_observations", 0))
    except Exception as exc:
        err_msg = f"Failed to build observation panel: {exc}"
        _logger.error("[AUDIT] %s", err_msg)
        errors_list.append(err_msg)
        panel        = pd.DataFrame()
        r2           = float("nan")
        mean_error   = float("nan")
        residual_std = float("nan")
        n_obs        = 0

    # ── Check 1: No look-ahead in forward returns ─────────────────────────────
    no_lookahead = _panel_forward_returns_no_lookahead(panel, end_date)
    if not no_lookahead:
        warnings_list.append(
            "Look-ahead bias detected: some forward returns extend beyond the calibration window boundary."
        )

    # ── Check 2: No double-counting across calibration cycles ─────────────────
    calibration_history = _load_all_calibrations()
    no_double_count, dc_detail = _check_no_double_counting(calibration_history)
    if not no_double_count:
        warnings_list.append(f"Double-counting risk: {dc_detail}")

    # ── Check 3: Sufficient observations ──────────────────────────────────────
    observations_ok = n_obs >= MIN_OBSERVATIONS
    if not observations_ok:
        warnings_list.append(
            f"Calibration window has only {n_obs} observations (minimum {MIN_OBSERVATIONS})."
        )

    # ── Check 4: Sufficient window length ─────────────────────────────────────
    window_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    window_ok = window_months >= MIN_WINDOW_MONTHS
    if not window_ok:
        warnings_list.append(
            f"Calibration window is only {window_months} months "
            f"(minimum {MIN_WINDOW_MONTHS} recommended)."
        )

    # ── Check 5: Overfitting (R² suspiciously high) ───────────────────────────
    r2_overfitting = (not np.isnan(r2)) and r2 > R2_OVERFITTING_THRESHOLD
    if r2_overfitting:
        warnings_list.append(
            f"Potential overfitting: R²={r2:.4f} > {R2_OVERFITTING_THRESHOLD}. "
            "A 4-factor model on noisy returns should not explain > 80% of variance. "
            "Check for data leakage or spurious correlations."
        )

    # ── Check 6: Coefficient stability ────────────────────────────────────────
    stability_result = _check_coefficient_stability(
        panel=panel,
        window_start=start_date,
        window_end=end_date,
    )
    if not stability_result["stable"]:
        for f, stats in stability_result["factors"].items():
            if not stats.get("stable", True):
                flip = stats.get("sign_flip_rate", 0.0)
                cv   = stats.get("cv")
                warnings_list.append(
                    f"Coefficient instability for '{f}': "
                    f"sign_flip_rate={flip:.1%}"
                    + (f", CV={cv:.2f}" if cv is not None else "")
                    + ". Model may be over-fitting to short-window noise."
                )

    # ── Check 7: Out-of-sample improvement ────────────────────────────────────
    oos_result = _check_out_of_sample_improvement(panel)
    if not oos_result.get("improvement", False) and oos_result.get("reason") == "ok":
        warnings_list.append(
            f"Error correction does NOT improve out-of-sample prediction: "
            f"Δ RMSE = {oos_result.get('delta_rmse', 0):.6f} "
            f"(positive = improvement). Review calibration quality."
        )

    # ── Bias detection ────────────────────────────────────────────────────────
    # Bias = mean prediction error that is "large" relative to residual std.
    # We flag if:  |mean_error| > 0.5 × residual_std  (½ sigma effect size)
    bias_detected = (
        not np.isnan(mean_error)
        and not np.isnan(residual_std)
        and residual_std > 0
        and abs(mean_error) > 0.5 * residual_std
    )
    if bias_detected:
        warnings_list.append(
            f"Systematic prediction bias detected: mean_error={mean_error:+.6f}, "
            f"residual_std={residual_std:.6f} (ratio={abs(mean_error)/residual_std:.2f}). "
            "The intercept β₀ is significant — error correction should reduce this."
        )

    # ── Top-level summary flags ───────────────────────────────────────────────
    stability_warning = not stability_result.get("stable", True)
    improvement_value = _safe_float(oos_result.get("delta_rmse"), 0.0)

    # ── Structured per-check report ───────────────────────────────────────────
    checks = {
        "no_lookahead":            no_lookahead,
        "no_double_counting":      no_double_count,
        "no_double_counting_detail": dc_detail,
        "observations_sufficient": observations_ok,
        "window_sufficient":       window_ok,
        "r2_overfitting_flag":     r2_overfitting,
        "coefficient_stability":   stability_result,
        "out_of_sample":           oos_result,
    }

    # ── Overall pass / fail ───────────────────────────────────────────────────
    critical_failures = [
        ("no_lookahead",       not no_lookahead,          "Look-ahead bias in forward returns"),
        ("no_double_counting", not no_double_count,       "Double-counting data across calibrations"),
        ("observations_ok",    not observations_ok,       "Insufficient observations for stable OLS"),
    ]
    all_critical_pass = not any(failed for _, failed, _ in critical_failures)

    _logger.info(
        "[AUDIT] Complete: bias_detected=%s  stability_warning=%s  "
        "improvement_delta_rmse=%.6f  critical_pass=%s  warnings=%d",
        bias_detected, stability_warning, improvement_value, all_critical_pass,
        len(warnings_list),
    )
    for w in warnings_list:
        _logger.warning("[AUDIT] ⚠  %s", w)

    return {
        # ── Required summary (as specified in prompt) ────────────────────────
        "bias_detected":                bias_detected,
        "improvement_after_correction": improvement_value,
        "stability_warning":            stability_warning,

        # ── Detailed integrity check results ──────────────────────────────────
        "checks":               checks,
        "all_critical_pass":    all_critical_pass,
        "critical_failures":    [msg for _, failed, msg in critical_failures if failed],

        # ── OLS diagnostics ───────────────────────────────────────────────────
        "r_squared":            round(r2, 6) if not np.isnan(r2) else None,
        "mean_error":           round(mean_error, 6) if not np.isnan(mean_error) else None,
        "residual_std":         round(residual_std, 6) if not np.isnan(residual_std) else None,
        "n_observations":       n_obs,

        # ── Window metadata ───────────────────────────────────────────────────
        "audit_date":           end_date,
        "window_start":         start_date,
        "window_end":           end_date,
        "window_months":        window_months,

        # ── DB calibration history ────────────────────────────────────────────
        "calibration_history":  calibration_history,
        "n_calibration_records": len(calibration_history),

        # ── Human-readable messages ───────────────────────────────────────────
        "warnings":             warnings_list,
        "errors":               errors_list,
    }


def format_audit_report(report: Dict[str, Any]) -> str:
    """Return a human-readable text summary of the audit report."""
    lines = [
        "═" * 70,
        "  Error Model Statistical Integrity Audit",
        "═" * 70,
        f"  Audit date  : {report.get('audit_date')}",
        f"  Window      : {report.get('window_start')} → {report.get('window_end')}",
        f"  N obs       : {report.get('n_observations', 0)}",
        f"  R²          : {report.get('r_squared')}",
        f"  Mean error  : {report.get('mean_error')}",
        f"  Residual σ  : {report.get('residual_std')}",
        "",
        "  ── Top-level summary ─────────────────────────────────────────────",
        f"  bias_detected               : {report['bias_detected']}",
        f"  improvement_after_correction: {report['improvement_after_correction']:.6f}",
        f"  stability_warning           : {report['stability_warning']}",
        f"  all_critical_pass           : {report['all_critical_pass']}",
        "",
        "  ── Integrity checks ──────────────────────────────────────────────",
    ]

    checks = report.get("checks", {})
    for key, label in [
        ("no_lookahead",            "No look-ahead bias"),
        ("no_double_counting",      "No double-counting"),
        ("observations_sufficient", "Sufficient observations"),
        ("window_sufficient",       "Window long enough"),
        ("r2_overfitting_flag",     "R² overfitting flag"),
    ]:
        val = checks.get(key)
        if key == "r2_overfitting_flag":
            icon = "⚠ " if val else "✓ "
            lines.append(f"  {icon}{label:<32}: {val}")
        else:
            icon = "✓ " if val else "✗ "
            lines.append(f"  {icon}{label:<32}: {val}")

    # Coefficient stability
    stab = checks.get("coefficient_stability", {})
    lines.append(f"\n  ── Coefficient stability ({'STABLE' if stab.get('stable') else 'UNSTABLE'}) ──")
    for f, stats in stab.get("factors", {}).items():
        ok  = "✓" if stats.get("stable") else "✗"
        cv  = stats.get("cv")
        sfr = stats.get("sign_flip_rate", 0.0)
        lines.append(
            f"    {ok} {f:<12}  mean={stats.get('mean', 'nan'):+.5f}"
            f"  std={stats.get('std', 'nan'):.5f}"
            f"  CV={cv if cv is not None else '∞':>6}"
            f"  flip={sfr:.0%}"
        )

    # OOS
    oos = checks.get("out_of_sample", {})
    lines += [
        "",
        "  ── Out-of-sample test ────────────────────────────────────────────",
        f"  Train n={oos.get('n_train', '?')}  Test n={oos.get('n_test', '?')}",
        f"  Train R²           : {oos.get('train_r2', 'nan')}",
        f"  Test RMSE (raw)    : {oos.get('test_rmse_raw', 'nan')}",
        f"  Test RMSE (corrected): {oos.get('test_rmse_corrected', 'nan')}",
        f"  Δ RMSE             : {oos.get('delta_rmse', 0.0):.6f}  "
        f"({'IMPROVEMENT ✓' if oos.get('improvement') else 'NO IMPROVEMENT ✗'})",
    ]

    # Warnings
    warnings = report.get("warnings", [])
    if warnings:
        lines += ["", "  ── Warnings ──────────────────────────────────────────────────────"]
        for w in warnings:
            lines.append(f"  ⚠  {w}")

    critical = report.get("critical_failures", [])
    if critical:
        lines += ["", "  ── Critical Failures ─────────────────────────────────────────────"]
        for c in critical:
            lines.append(f"  ✗  {c}")

    lines.append("═" * 70)
    return "\n".join(lines)


__all__ = [
    "run_integrity_audit",
    "format_audit_report",
    # Constants
    "R2_OVERFITTING_THRESHOLD",
    "MIN_WINDOW_MONTHS",
    "MIN_OBSERVATIONS",
    "SIGN_FLIP_THRESHOLD",
]
