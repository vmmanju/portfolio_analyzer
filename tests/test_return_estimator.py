"""tests/test_return_estimator.py

Unit tests for services/return_estimator.py — Bayesian shrinkage estimator.

Run with:
    python -m pytest tests/test_return_estimator.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from services.return_estimator import (
    DEFAULT_SHRINKAGE_K,
    bayesian_shrinkage_returns,
    shrinkage_lambda,
    shrunk_annualised_returns,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_returns(n_assets: int = 3, n_obs: int = 90, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0005, scale=0.02, size=(n_obs, n_assets))
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n_assets)])


# ─────────────────────────────────────────────────────────────────────────────
# Core property: shrunk mean must lie BETWEEN prior and sample mean
# ─────────────────────────────────────────────────────────────────────────────

class TestShrinkageBoundedness:
    """Statistical guarantee: shrunk mean is between prior and sample mean."""

    def test_shrunk_between_prior_and_sample__default_prior(self):
        """Each shrunk return lies between its sample mean and the prior."""
        returns = _make_returns(n_assets=5, n_obs=90)
        mu_sample = returns.mean()
        mu_prior = float(mu_sample.mean())   # cross-sectional mean
        mu_shrunk = bayesian_shrinkage_returns(returns)

        for col in returns.columns:
            lo = min(mu_sample[col], mu_prior)
            hi = max(mu_sample[col], mu_prior)
            assert lo <= mu_shrunk[col] <= hi, (
                f"{col}: shrunk={mu_shrunk[col]:.6f} not in "
                f"[{lo:.6f}, {hi:.6f}] (sample={mu_sample[col]:.6f}, prior={mu_prior:.6f})"
            )

    def test_shrunk_between_prior_and_sample__custom_prior(self):
        """Works with an explicit scalar prior (e.g., risk-free rate = 0)."""
        returns = _make_returns(n_assets=4, n_obs=120)
        mu_sample = returns.mean()
        mu_prior_val = 0.0   # zero prior
        mu_shrunk = bayesian_shrinkage_returns(returns, mu_prior=mu_prior_val)

        for col in returns.columns:
            lo = min(mu_sample[col], mu_prior_val)
            hi = max(mu_sample[col], mu_prior_val)
            assert lo <= mu_shrunk[col] <= hi, (
                f"{col}: shrunk={mu_shrunk[col]:.6f} not in [{lo:.6f}, {hi:.6f}]"
            )

    def test_shrunk_between_prior_and_sample__high_k(self):
        """High k (strong shrinkage) pulls values tightly toward prior."""
        returns = _make_returns(n_assets=3, n_obs=60)
        mu_sample = returns.mean()
        mu_prior = float(mu_sample.mean())
        mu_shrunk = bayesian_shrinkage_returns(returns, shrinkage_k=500)

        for col in returns.columns:
            lo = min(mu_sample[col], mu_prior)
            hi = max(mu_sample[col], mu_prior)
            assert lo <= mu_shrunk[col] <= hi

    def test_shrunk_between_prior_and_sample__low_k(self):
        """Low k (weak shrinkage) keeps values close to sample mean."""
        returns = _make_returns(n_assets=3, n_obs=90)
        mu_sample = returns.mean()
        mu_shrunk = bayesian_shrinkage_returns(returns, shrinkage_k=1)
        # With k=1 and T=90, λ = 90/91 ≈ 0.989 → very close to sample
        tol = 1e-4
        for col in returns.columns:
            assert abs(mu_shrunk[col] - mu_sample[col]) < tol, (
                f"{col}: shrunk={mu_shrunk[col]} sample={mu_sample[col]}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Lambda formula
# ─────────────────────────────────────────────────────────────────────────────

class TestShrinkageLambda:
    def test_lambda_formula_exact(self):
        assert shrinkage_lambda(90, 60) == pytest.approx(90 / 150, rel=1e-9)

    def test_lambda_formula_252(self):
        assert shrinkage_lambda(252, 60) == pytest.approx(252 / 312, rel=1e-9)

    def test_lambda_zero_obs(self):
        assert shrinkage_lambda(0, 60) == 0.0

    def test_lambda_very_large_T(self):
        # As T → ∞, λ → 1
        assert shrinkage_lambda(1_000_000, 60) == pytest.approx(1.0, abs=1e-4)

    def test_lambda_zero_k(self):
        # k=0 → λ = 1 (no shrinkage)
        assert shrinkage_lambda(50, 0) == 1.0

    @pytest.mark.parametrize("T,k", [(30, 60), (90, 60), (252, 60), (500, 60)])
    def test_lambda_in_unit_interval(self, T, k):
        lam = shrinkage_lambda(T, k)
        assert 0.0 <= lam <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Monotonicity w.r.t. shrinkage_k
# ─────────────────────────────────────────────────────────────────────────────

class TestMonotonicity:
    def test_more_shrinkage_as_k_increases(self):
        """Larger k → shrunk returns closer to the prior (less sample trust)."""
        returns = _make_returns(n_assets=4, n_obs=80, seed=7)
        mu_sample = returns.mean()
        mu_prior = float(mu_sample.mean())

        shrunk_low_k = bayesian_shrinkage_returns(returns, shrinkage_k=10)
        shrunk_high_k = bayesian_shrinkage_returns(returns, shrinkage_k=200)

        for col in returns.columns:
            dist_low = abs(shrunk_low_k[col] - mu_prior)
            dist_high = abs(shrunk_high_k[col] - mu_prior)
            # High-k should be closer to prior
            assert dist_high <= dist_low + 1e-12, (
                f"{col}: dist_low={dist_low:.6f}, dist_high={dist_high:.6f}"
            )

    def test_more_data_reduces_shrinkage(self):
        """Larger T (more observations) → λ closer to 1 → shrunk mean closer to sample mean.

        We verify this directly via the lambda formula rather than comparing
        across samples with different means (which is an apples-to-oranges comparison).
        With fixed k=60:
          T=30  → λ = 30/90  ≈ 0.333  → |shrunk - sample| = (1-λ)|sample - prior|
          T=300 → λ = 300/360 ≈ 0.833 → distance is smaller
        """
        from services.return_estimator import shrinkage_lambda

        k = DEFAULT_SHRINKAGE_K
        lam_short = shrinkage_lambda(30, k)    # ≈ 0.333
        lam_long  = shrinkage_lambda(300, k)   # ≈ 0.833

        # Larger T → larger λ → closer to 1 → smaller shrinkage distance
        assert lam_long > lam_short

        # For any fixed (sample_mean, prior), the distance |shrunk - sample| = (1-λ)|sample - prior|
        # So larger λ means smaller distance.
        sample = 0.010
        prior  = 0.002
        dist_short = (1.0 - lam_short) * abs(sample - prior)
        dist_long  = (1.0 - lam_long)  * abs(sample - prior)
        assert dist_long < dist_short



# ─────────────────────────────────────────────────────────────────────────────
# Limit cases
# ─────────────────────────────────────────────────────────────────────────────

class TestLimitCases:
    def test_k_zero_equals_sample_mean(self):
        """k=0 → λ=1 → shrunk = sample mean exactly."""
        returns = _make_returns(n_assets=3, n_obs=60)
        mu_sample = returns.mean()
        mu_shrunk = bayesian_shrinkage_returns(returns, shrinkage_k=0)
        pd.testing.assert_series_equal(mu_shrunk, mu_sample, check_names=False)

    def test_identical_stocks_no_change(self):
        """If all stocks have the same mean return, shrinkage has no effect."""
        rng = np.random.default_rng(1)
        # All stocks drawn from same distribution → same expected μ
        data = pd.DataFrame({"X": rng.normal(0, 0.01, 100), "Y": rng.normal(0, 0.01, 100)})
        mu_sample = data.mean()
        mu_prior = float(mu_sample.mean())
        mu_shrunk = bayesian_shrinkage_returns(data)
        # shrunk = λ·sample + (1-λ)·prior; with prior ≈ mean(sample), approx equal
        for col in data.columns:
            assert abs(mu_shrunk[col] - (shrinkage_lambda(100) * mu_sample[col]
                                         + (1 - shrinkage_lambda(100)) * mu_prior)) < 1e-15

    def test_single_asset(self):
        """Single-asset case: prior = sample mean → shrunk = sample mean."""
        rng = np.random.default_rng(2)
        data = pd.DataFrame({"Z": rng.normal(0.001, 0.02, 90)})
        mu_sample = float(data["Z"].mean())
        mu_shrunk = bayesian_shrinkage_returns(data)
        # With 1 asset: prior = sample mean → no change regardless of λ
        assert mu_shrunk["Z"] == pytest.approx(mu_sample, rel=1e-9)

    def test_empty_dataframe(self):
        """Empty input returns empty Series."""
        result = bayesian_shrinkage_returns(pd.DataFrame())
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_all_nan_column_dropped(self):
        """Columns that are entirely NaN should be gracefully excluded."""
        data = pd.DataFrame({
            "A": [0.01, 0.02, -0.01],
            "B": [float("nan"), float("nan"), float("nan")],
        })
        mu_shrunk = bayesian_shrinkage_returns(data)
        assert "A" in mu_shrunk.index
        assert "B" not in mu_shrunk.index


# ─────────────────────────────────────────────────────────────────────────────
# Annualised helper
# ─────────────────────────────────────────────────────────────────────────────

class TestShrunkAnnualisedReturns:
    def test_annualised_is_252x_daily(self):
        returns = _make_returns(n_assets=2, n_obs=90)
        daily = bayesian_shrinkage_returns(returns)
        annual = shrunk_annualised_returns(returns)
        pd.testing.assert_series_equal(daily * 252, annual, check_names=False)

    def test_custom_annualise_factor(self):
        returns = _make_returns()
        daily = bayesian_shrinkage_returns(returns)
        annual12 = shrunk_annualised_returns(returns, annualize=12)
        pd.testing.assert_series_equal(daily * 12, annual12, check_names=False)


# ─────────────────────────────────────────────────────────────────────────────
# Known numerical example
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalExample:
    def test_known_values(self):
        """Hand-computed example: verify formula exactly."""
        # Two assets, 4 observations each
        data = pd.DataFrame({
            "A": [0.01, 0.02, 0.03, 0.04],   # mean = 0.025
            "B": [-0.01, 0.00, -0.01, 0.00],  # mean = -0.005
        })
        # T=4, k=60 → λ = 4/64 = 0.0625
        # μ_prior = (0.025 + (-0.005)) / 2 = 0.01
        # μ_shrunk_A = 0.0625 * 0.025 + 0.9375 * 0.01 = 0.0015625 + 0.009375 = 0.010938
        # μ_shrunk_B = 0.0625 * (-0.005) + 0.9375 * 0.01 = -0.0003125 + 0.009375 = 0.009063
        mu_shrunk = bayesian_shrinkage_returns(data, shrinkage_k=60)
        assert mu_shrunk["A"] == pytest.approx(0.0109375, rel=1e-6)
        assert mu_shrunk["B"] == pytest.approx(0.0090625, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Error-Correction Layer Tests
# (apply_error_correction, corrected_returns)
# ─────────────────────────────────────────────────────────────────────────────

from datetime import date as _date
from services.return_estimator import apply_error_correction, corrected_returns


# ── shared fixtures ───────────────────────────────────────────────────────────

def _make_returns_ec(n_stocks=5, n_days=60, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0005, 0.015, (n_days, n_stocks))
    dates = pd.bdate_range(start="2023-01-02", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=list(range(1, n_stocks + 1)))


def _factor_scores_ec() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "momentum_score":   [60.0, 40.0, 55.0, 45.0, 70.0],
            "quality_score":    [55.0, 50.0, 60.0, 45.0, 65.0],
            "value_score":      [50.0, 55.0, 48.0, 52.0, 58.0],
            "volatility_score": [45.0, 55.0, 50.0, 60.0, 40.0],
        },
        index=[1, 2, 3, 4, 5],
    )


def _calib_cached(intercept=0.005, beta_mom=0.001) -> dict:
    return {
        "calibration_source": "cached",
        "window_end": _date(2024, 1, 31),
        "intercept": intercept,
        "coefficients": {
            "momentum":   beta_mom,
            "quality":    0.002,
            "value":     -0.001,
            "volatility": 0.000,
        },
    }


def _calib_none() -> dict:
    return {
        "calibration_source": "none",
        "intercept": float("nan"),
        "coefficients": {f: float("nan") for f in ["momentum", "quality", "value", "volatility"]},
    }


# ── TestApplyErrorCorrection ──────────────────────────────────────────────────

class TestApplyErrorCorrection:

    def _pred(self) -> pd.Series:
        return pd.Series([0.10, 0.08, 0.12, 0.09, 0.11], index=[1, 2, 3, 4, 5])

    def test_intercept_only_correction(self):
        """With zero factor betas, correction = subtract intercept."""
        calib = {
            "calibration_source": "cached",
            "window_end": _date(2024, 1, 1),
            "intercept": 0.01,
            "coefficients": {f: 0.0 for f in ["momentum", "quality", "value", "volatility"]},
        }
        pred = self._pred()
        corrected = apply_error_correction(pred, pd.DataFrame(), calibration=calib, use_correction=True)
        pd.testing.assert_series_equal(corrected, pred - 0.01, atol=1e-10)

    def test_full_bias_formula(self):
        """bias_i = β0 + β_MOM·MOM_i  (other betas=0)."""
        calib = {
            "calibration_source": "cached",
            "window_end": _date(2024, 1, 1),
            "intercept": 0.002,
            "coefficients": {"momentum": 0.010, "quality": 0.0, "value": 0.0, "volatility": 0.0},
        }
        pred = self._pred()
        fs   = _factor_scores_ec()
        corrected = apply_error_correction(pred, fs, calibration=calib, use_correction=True)
        # bias for stock 1: 0.002 + 0.010 * 60 = 0.602
        expected_1 = pred.loc[1] - (0.002 + 0.010 * 60.0)
        assert abs(float(corrected.loc[1]) - expected_1) < 1e-10

    def test_correction_disabled_returns_unchanged(self):
        pred = self._pred()
        result = apply_error_correction(
            pred, _factor_scores_ec(), calibration=_calib_cached(), use_correction=False
        )
        pd.testing.assert_series_equal(result, pred)

    def test_no_calibration_returns_unchanged(self):
        pred = self._pred()
        result = apply_error_correction(pred, pd.DataFrame(), calibration=_calib_none(), use_correction=True)
        pd.testing.assert_series_equal(result, pred)

    def test_empty_predicted_returns_unchanged(self):
        empty = pd.Series(dtype=float)
        result = apply_error_correction(empty, pd.DataFrame(), calibration=_calib_cached(), use_correction=True)
        assert result.empty

    def test_missing_factor_cols_uses_zero(self):
        """Factor columns absent from factor_scores → their beta·score = 0."""
        calib = {
            "calibration_source": "cached",
            "window_end": _date(2024, 1, 1),
            "intercept": 0.0,
            "coefficients": {"momentum": 100.0, "quality": 0.0, "value": 0.0, "volatility": 0.0},
        }
        pred = self._pred()
        result = apply_error_correction(pred, pd.DataFrame(), calibration=calib, use_correction=True)
        # β0=0, no factor data → bias = 0, result == pred
        pd.testing.assert_series_equal(result, pred)

    def test_partial_factor_index_overlap(self):
        """Stock not in factor_scores gets only intercept bias."""
        calib = {
            "calibration_source": "cached",
            "window_end": _date(2024, 1, 1),
            "intercept": 0.005,
            "coefficients": {f: 0.0 for f in ["momentum", "quality", "value", "volatility"]},
        }
        pred     = pd.Series([0.10, 0.09, 0.11], index=[1, 2, 99])
        fs       = _factor_scores_ec()   # has 1,2,3,4,5 — no 99
        corrected = apply_error_correction(pred, fs, calibration=calib, use_correction=True)
        assert float(corrected.loc[99]) == pytest.approx(pred.loc[99] - 0.005, abs=1e-9)
        assert float(corrected.loc[1])  == pytest.approx(pred.loc[1]  - 0.005, abs=1e-9)

    def test_correction_logged(self, caplog):
        """apply_error_correction logs 'Error correction applied'."""
        import logging
        pred = self._pred()
        with caplog.at_level(logging.INFO, logger="services.return_estimator"):
            apply_error_correction(pred, _factor_scores_ec(), calibration=_calib_cached(), use_correction=True)
        assert any("Error correction applied" in r.message for r in caplog.records), \
            f"Expected log message not found. Got: {[r.message for r in caplog.records]}"

    def test_negative_intercept_raises_returns(self):
        """Negative intercept subtracted → corrected > predicted."""
        calib = {
            "calibration_source": "cached",
            "window_end": _date(2024, 1, 1),
            "intercept": -0.02,   # model under-predicted by 2 %
            "coefficients": {f: 0.0 for f in ["momentum", "quality", "value", "volatility"]},
        }
        pred = pd.Series([0.08, 0.10], index=[1, 2])
        corrected = apply_error_correction(pred, pd.DataFrame(), calibration=calib, use_correction=True)
        assert float(corrected.loc[1]) == pytest.approx(0.10, abs=1e-9)
        assert float(corrected.loc[2]) == pytest.approx(0.12, abs=1e-9)


# ── TestCorrectedReturns ──────────────────────────────────────────────────────

class TestCorrectedReturns:

    def test_output_is_series_indexed_by_stock(self):
        ret_df = _make_returns_ec()
        result = corrected_returns(ret_df, _factor_scores_ec(), calibration=_calib_cached(), use_correction=True)
        assert isinstance(result, pd.Series)
        assert set(result.index).issubset(set(ret_df.columns))

    def test_correction_off_equals_plain_shrunk(self):
        ret_df = _make_returns_ec()
        uncorr = corrected_returns(ret_df, _factor_scores_ec(), use_correction=False)
        plain  = shrunk_annualised_returns(ret_df)
        pd.testing.assert_series_equal(uncorr, plain, atol=1e-10)

    def test_positive_bias_lowers_output(self):
        ret_df = _make_returns_ec()
        plain      = shrunk_annualised_returns(ret_df)
        corr       = corrected_returns(ret_df, _factor_scores_ec(), calibration=_calib_cached(), use_correction=True)
        mean_diff  = float((corr - plain.reindex(corr.index)).mean())
        assert mean_diff < 0, "Positive-bias calibration must lower returns after correction"

    def test_no_calibration_equals_plain_shrunk(self):
        ret_df = _make_returns_ec()
        result = corrected_returns(ret_df, _factor_scores_ec(), calibration=_calib_none(), use_correction=True)
        plain  = shrunk_annualised_returns(ret_df)
        pd.testing.assert_series_equal(result, plain, atol=1e-10)

    def test_empty_returns_gives_empty(self):
        result = corrected_returns(pd.DataFrame(), _factor_scores_ec(), calibration=_calib_cached())
        assert result.empty

    def test_returns_are_finite(self):
        ret_df = _make_returns_ec()
        result = corrected_returns(ret_df, _factor_scores_ec(), calibration=_calib_cached(), use_correction=True)
        assert result.notna().all()
        assert np.isfinite(result.values).all()


# ── TestBacktestComparison ────────────────────────────────────────────────────

class TestSyntheticBacktestComparison:
    """Verify that ranking quality improves when the error model correctly
    identifies a spurious momentum bias in Bayesian-shrunk returns."""

    def _build(self, n_stocks=10, n_days=120, seed=7):
        rng = np.random.default_rng(seed)
        quality  = rng.uniform(20, 80, n_stocks)
        momentum = rng.uniform(20, 80, n_stocks)
        true_daily_mu = (quality - 50.0) / 100.0 * 0.001
        daily = np.zeros((n_days, n_stocks))
        for j in range(n_stocks):
            mom_noise = (momentum[j] - 50.0) / 100.0 * 0.0005
            daily[:, j] = rng.normal(true_daily_mu[j] + mom_noise, 0.015, n_days)
        stock_ids = list(range(1, n_stocks + 1))
        ret_df   = pd.DataFrame(daily, index=pd.bdate_range("2023-01-02", periods=n_days), columns=stock_ids)
        factors  = pd.DataFrame(
            {"momentum_score": momentum, "quality_score": quality,
             "value_score": np.full(n_stocks, 50.0), "volatility_score": np.full(n_stocks, 50.0)},
            index=stock_ids,
        )
        true_rank = pd.Series(quality, index=stock_ids).rank(ascending=False).astype(int)
        return ret_df, factors, true_rank

    def test_corrected_rank_correlates_better_with_quality(self):
        """After momentum-bias correction, Spearman ρ with quality-rank improves."""
        from scipy.stats import spearmanr
        ret_df, factors, true_rank = self._build()
        # Calibration identifies the known momentum bias
        beta_ann = 0.0005 / 100.0 * 252
        calib = {
            "calibration_source": "cached",
            "window_end": _date(2024, 1, 1),
            "intercept": 0.0,
            "coefficients": {"momentum": beta_ann, "quality": 0.0, "value": 0.0, "volatility": 0.0},
        }
        uncorr = shrunk_annualised_returns(ret_df)
        corr   = corrected_returns(ret_df, factors, calibration=calib, use_correction=True)

        rank_unc = uncorr.rank(ascending=False).astype(int)
        rank_cor = corr.rank(ascending=False).astype(int)

        rho_unc, _ = spearmanr(true_rank, rank_unc.reindex(true_rank.index))
        rho_cor, _ = spearmanr(true_rank, rank_cor.reindex(true_rank.index))

        assert rho_cor > rho_unc, (
            f"Corrected Spearman ρ ({rho_cor:.3f}) should exceed uncorrected ({rho_unc:.3f})"
        )

