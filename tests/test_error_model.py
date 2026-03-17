"""tests/test_error_model.py

Unit tests for services/error_model.py
=======================================

These tests run entirely with SYNTHETIC data — no database connection needed.
They patch the DB-touching helpers so the pure statistical logic can be
tested in isolation.

Coverage:
  1. _run_ols           — correctness, R², t-stats via both backends
  2. _predicted_return_from_score   — deterministic mapping
  3. _predicted_return_from_shrinkage (mocked prices)
  4. _forward_return    — look-ahead guard, edge cases
  5. _build_observation_panel (fully mocked)
  6. compute_error_coefficients — full integration with mocked DB helpers
  7. format_error_model_summary — output sanity
"""

import math
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# --- project path ----------------------------------------------------------
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from services.error_model import (
    FACTOR_LABELS,
    FORWARD_DAYS,
    _build_observation_panel,
    _forward_return,
    _predicted_return_from_score,
    _run_ols,
    compute_error_coefficients,
    format_error_model_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_prices(rng) -> pd.DataFrame:
    """100-day price series for 5 stocks indexed as DatetimeIndex."""
    n_days, n_stocks = 100, 5
    daily_ret = rng.normal(0.0005, 0.02, (n_days, n_stocks))
    prices = 100.0 * np.exp(np.cumsum(daily_ret, axis=0))
    dates = pd.bdate_range(start="2023-01-02", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=[1, 2, 3, 4, 5])


@pytest.fixture
def simple_panel() -> pd.DataFrame:
    """Controlled panel where β1=0.5, β2=-0.2, intercept=0.01."""
    rng = np.random.default_rng(7)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    noise = rng.normal(0, 0.01, n)
    y = 0.01 + 0.5 * x1 + (-0.2) * x2 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


# ─────────────────────────────────────────────────────────────────────────────
# 1. _run_ols
# ─────────────────────────────────────────────────────────────────────────────

class TestRunOLS:
    def test_intercept_and_coefficients(self, simple_panel):
        X = np.column_stack([
            np.ones(len(simple_panel)),
            simple_panel[["x1", "x2"]].values,
        ])
        y = simple_panel["y"].values
        result = _run_ols(X, y, feature_names=["x1", "x2"])

        assert abs(result["intercept"] - 0.01) < 0.01
        assert abs(result["coefficients"]["x1"] - 0.5) < 0.05
        assert abs(result["coefficients"]["x2"] - (-0.2)) < 0.05

    def test_r_squared_high_for_known_dgp(self, simple_panel):
        X = np.column_stack([
            np.ones(len(simple_panel)),
            simple_panel[["x1", "x2"]].values,
        ])
        y = simple_panel["y"].values
        result = _run_ols(X, y, feature_names=["x1", "x2"])
        # With small noise the R² should be very high
        assert result["r_squared"] > 0.95

    def test_too_few_obs_returns_nan(self):
        X = np.column_stack([np.ones(3), np.arange(3.0)])
        y = np.array([1.0, 2.0, 3.0])
        result = _run_ols(X, y, feature_names=["f1"])
        # n=3, k=2 → ok, but n < k+2=4
        assert math.isnan(result["intercept"])

    def test_perfect_fit_r2_is_one(self):
        n = 50
        rng = np.random.default_rng(1)
        x = rng.standard_normal(n)
        y = 3.0 + 2.5 * x          # perfect linear relationship
        X = np.column_stack([np.ones(n), x])
        result = _run_ols(X, y, feature_names=["x"])
        assert abs(result["r_squared"] - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 2. _predicted_return_from_score
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictedReturnFromScore:
    def test_score_50_is_zero(self):
        scores = pd.Series({1: 50.0, 2: 50.0})
        result = _predicted_return_from_score(scores)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_score_100_is_positive(self):
        scores = pd.Series({1: 100.0})
        result = _predicted_return_from_score(scores)
        assert float(result.iloc[0]) > 0

    def test_score_0_is_negative(self):
        scores = pd.Series({1: 0.0})
        result = _predicted_return_from_score(scores)
        assert float(result.iloc[0]) < 0

    def test_monotone_in_score(self):
        scores = pd.Series({i: float(i * 10) for i in range(11)})
        result = _predicted_return_from_score(scores)
        assert (result.diff().dropna() >= 0).all(), "Predicted return must be monotone in score"

    def test_symmetric_around_50(self):
        s_hi = pd.Series({1: 75.0})
        s_lo = pd.Series({1: 25.0})
        r_hi = float(_predicted_return_from_score(s_hi).iloc[0])
        r_lo = float(_predicted_return_from_score(s_lo).iloc[0])
        assert abs(r_hi + r_lo) < 1e-10, "Score mapping must be antisymmetric around 50"


# ─────────────────────────────────────────────────────────────────────────────
# 3. _forward_return
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardReturn:
    def test_correct_return_value(self, synthetic_prices):
        t = date(2023, 1, 2)
        fwd = _forward_return(synthetic_prices, as_of_date=t)
        assert not fwd.empty

        ts_ref = pd.Timestamp(t)
        future_idx = synthetic_prices.index[synthetic_prices.index >= ts_ref]
        t0 = future_idx[0]
        p0 = synthetic_prices.loc[t0]
        # _forward_return uses candidates STRICTLY > t0 then picks index FORWARD_DAYS-1
        fwd_candidates = synthetic_prices.index[synthetic_prices.index > t0]
        t1_idx = fwd_candidates[FORWARD_DAYS - 1]
        p1 = synthetic_prices.loc[t1_idx]
        expected = (p1 / p0) - 1.0
        pd.testing.assert_series_equal(
            fwd.reindex(expected.index).fillna(0),
            expected.fillna(0),
            check_names=False,
            atol=1e-10,
        )

    def test_no_future_data_returns_empty(self, synthetic_prices):
        # Ask for forward return at the very last date — no data ahead
        last_date = synthetic_prices.index[-1].date()
        fwd = _forward_return(synthetic_prices, as_of_date=last_date)
        assert fwd.empty, "Must return empty when insufficient future data"

    def test_no_prices_at_or_after_date(self, synthetic_prices):
        far_future = date(2099, 1, 1)
        fwd = _forward_return(synthetic_prices, as_of_date=far_future)
        assert fwd.empty


# ─────────────────────────────────────────────────────────────────────────────
# 4. _build_observation_panel  (mocked DB calls)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildObservationPanel:
    """Tests that panel builder correctly handles temporal constraints."""

    def _make_scores_df(self, dates_stocks):
        rows = []
        for d, sid in dates_stocks:
            rows.append({"stock_id": sid, "score_date": d, "composite_score": float(50 + sid)})
        return pd.DataFrame(rows)

    def _make_factors_df(self, dates_stocks):
        rows = []
        for d, sid in dates_stocks:
            rows.append({
                "stock_id": sid, "factor_date": d,
                "momentum_score": 0.5, "quality_score": 0.4,
                "value_score": 0.3, "volatility_score": 0.2,
            })
        return pd.DataFrame(rows)

    def test_drops_pairs_beyond_window(self, synthetic_prices):
        """Rebalance dates close to window_end should be dropped (look-ahead)."""
        score_date = date(2023, 1, 2)
        window_end = date(2023, 2, 1)      # only 21 trading days of future data

        late_reb = date(2023, 2, 1)        # forward window goes beyond window_end

        scores_df  = self._make_scores_df([(score_date, 1)])
        factors_df = self._make_factors_df([(score_date, 1)])

        with patch("services.error_model.get_latest_score_date_on_or_before",
                   return_value=score_date), \
             patch("services.error_model._predicted_return_from_shrinkage",
                   return_value=pd.Series({1: 0.02})):
            panel = _build_observation_panel(
                rebalance_dates=[late_reb],
                scores_df=scores_df,
                factors_df=factors_df,
                prices_wide=synthetic_prices,
                window_end=window_end,
                prediction_method="shrinkage",
            )
        # The late rebalance date should be skipped
        assert panel.empty or len(panel) == 0

    def test_error_column_computed_correctly(self, synthetic_prices):
        """error = actual − predicted; verify arithmetic."""
        score_date = date(2023, 1, 2)
        reb_date   = date(2023, 1, 2)
        window_end = date(2023, 6, 1)

        scores_df  = self._make_scores_df([(score_date, 1)])
        factors_df = self._make_factors_df([(score_date, 1)])

        mu_hat = 0.01  # fixed predicted return

        with patch("services.error_model.get_latest_score_date_on_or_before",
                   return_value=score_date), \
             patch("services.error_model._predicted_return_from_shrinkage",
                   return_value=pd.Series({1: mu_hat})):
            panel = _build_observation_panel(
                rebalance_dates=[reb_date],
                scores_df=scores_df,
                factors_df=factors_df,
                prices_wide=synthetic_prices,
                window_end=window_end,
                prediction_method="shrinkage",
            )

        if not panel.empty:
            row = panel.iloc[0]
            assert abs(row["error"] - (row["actual_return"] - row["predicted_return"])) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 5. compute_error_coefficients — integration (fully mocked DB layer)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeErrorCoefficients:
    """End-to-end tests with all DB calls mocked."""

    def _make_large_panel(self, n=120):
        """Generate a panel with known β1=0.3 (momentum) — no NaN values."""
        rng_gen = np.random.default_rng(99)
        start = date(2020, 1, 31)
        records = []
        for i in range(n):
            reb = start + timedelta(days=30 * i)
            mom  = float(rng_gen.standard_normal())
            qual = float(rng_gen.standard_normal())
            val  = float(rng_gen.standard_normal())
            vol  = float(rng_gen.standard_normal())
            actual = 0.005 + 0.3 * mom + float(rng_gen.normal(0, 0.01))
            records.append({
                "rebalance_date":   reb,
                "stock_id":         i % 10 + 1,
                "predicted_return": 0.005,
                "actual_return":    actual,
                "error":            actual - 0.005,
                "momentum_score":   mom,
                "quality_score":    qual,
                "value_score":      val,
                "volatility_score": vol,
            })
        df = pd.DataFrame(records)
        # Guarantee no NaN so dropna keeps all rows
        assert df.isna().sum().sum() == 0, "Panel has unexpected NaN values"
        return df

    def test_returns_dict_with_required_keys(self):
        panel = self._make_large_panel()
        required = {"intercept", "coefficients", "r_squared", "n_observations",
                    "window_start", "window_end", "mean_absolute_error",
                    "mean_error", "error_std", "panel_df"}
        two_dates = [date(2020, 1, 31), date(2020, 2, 28)]
        scores_df = pd.DataFrame({"stock_id": [1, 2], "score_date": two_dates, "composite_score": [60.0, 55.0]})
        # Non-empty prices so the prices-guard doesn't short-circuit before _build_observation_panel
        dummy_prices = pd.DataFrame({1: [100.0]}, index=pd.to_datetime(["2020-01-31"]))
        with patch("services.error_model.get_rebalance_dates", return_value=two_dates), \
             patch("services.error_model._load_scores_for_window", return_value=scores_df), \
             patch("services.error_model._load_factors_for_window", return_value=pd.DataFrame()), \
             patch("services.error_model._load_monthly_prices_for_stocks", return_value=dummy_prices), \
             patch("services.error_model._build_observation_panel", return_value=panel):
            result = compute_error_coefficients(
                end_date=date(2025, 1, 31),
                start_date=date(2020, 1, 31),
                log_stability=False,
            )
        assert required.issubset(result.keys())

    def test_n_observations_matches_panel(self):
        panel = self._make_large_panel(80)
        two_dates = [date(2020, 1, 31), date(2020, 2, 28)]
        scores_df = pd.DataFrame({"stock_id": [1, 2], "score_date": two_dates, "composite_score": [60.0, 55.0]})
        dummy_prices = pd.DataFrame({1: [100.0]}, index=pd.to_datetime(["2020-01-31"]))
        with patch("services.error_model.get_rebalance_dates", return_value=two_dates), \
             patch("services.error_model._load_scores_for_window", return_value=scores_df), \
             patch("services.error_model._load_factors_for_window", return_value=pd.DataFrame()), \
             patch("services.error_model._load_monthly_prices_for_stocks", return_value=dummy_prices), \
             patch("services.error_model._build_observation_panel", return_value=panel):
            result = compute_error_coefficients(
                end_date=date(2025, 1, 31),
                start_date=date(2020, 1, 31),
                log_stability=False,
            )
        assert result["n_observations"] == len(panel)

    def test_known_momentum_coefficient(self):
        """Recovery test: β_MOM should ≈ 0.3 for the synthetic panel."""
        panel = self._make_large_panel(200)
        two_dates = [date(2020, 1, 31), date(2020, 2, 28)]
        scores_df = pd.DataFrame({"stock_id": [1, 2], "score_date": two_dates, "composite_score": [60.0, 55.0]})
        dummy_prices = pd.DataFrame({1: [100.0]}, index=pd.to_datetime(["2020-01-31"]))
        with patch("services.error_model.get_rebalance_dates", return_value=two_dates), \
             patch("services.error_model._load_scores_for_window", return_value=scores_df), \
             patch("services.error_model._load_factors_for_window", return_value=pd.DataFrame()), \
             patch("services.error_model._load_monthly_prices_for_stocks", return_value=dummy_prices), \
             patch("services.error_model._build_observation_panel", return_value=panel):
            result = compute_error_coefficients(
                end_date=date(2025, 1, 31),
                start_date=date(2020, 1, 31),
                log_stability=False,
            )
        beta_mom = result["coefficients"].get("momentum", float("nan"))
        assert not math.isnan(beta_mom), "momentum coefficient should not be NaN"
        assert abs(beta_mom - 0.3) < 0.05, f"Expected β_MOM ≈ 0.3, got {beta_mom:.4f}"

    def test_empty_db_returns_nan_result(self):
        with patch("services.error_model.get_rebalance_dates", return_value=[]), \
             patch("services.error_model._load_scores_for_window",
                   return_value=pd.DataFrame()), \
             patch("services.error_model._load_factors_for_window",
                   return_value=pd.DataFrame()), \
             patch("services.error_model._load_monthly_prices_for_stocks",
                   return_value=pd.DataFrame()), \
             patch("services.error_model._build_observation_panel",
                   return_value=pd.DataFrame()):
            result = compute_error_coefficients(log_stability=False)
        assert result["n_observations"] == 0
        assert math.isnan(result["r_squared"])

    def test_window_defaults_to_five_years(self):
        """If no start_date given, start = end − 5 years."""
        end = date(2025, 6, 15)
        with patch("services.error_model.get_rebalance_dates", return_value=[]), \
             patch("services.error_model._load_scores_for_window", return_value=pd.DataFrame()), \
             patch("services.error_model._load_factors_for_window", return_value=pd.DataFrame()), \
             patch("services.error_model._load_monthly_prices_for_stocks", return_value=pd.DataFrame()), \
             patch("services.error_model._build_observation_panel", return_value=pd.DataFrame()):
            result = compute_error_coefficients(end_date=end, window_years=5, log_stability=False)

        assert result["window_start"] == date(2020, 6, 15)
        assert result["window_end"]   == end


# ─────────────────────────────────────────────────────────────────────────────
# 6. format_error_model_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatSummary:
    def _make_result(self):
        return {
            "window_start": date(2020, 1, 1),
            "window_end":   date(2025, 1, 1),
            "prediction_method": "shrinkage",
            "method": "numpy",
            "n_observations": 150,
            "r_squared":  0.12,
            "adj_r_squared": 0.10,
            "residual_std": 0.023,
            "mean_error":  0.001,
            "mean_absolute_error": 0.018,
            "error_std": 0.025,
            "intercept": 0.0012,
            "coefficients": {"momentum": 0.15, "quality": -0.08, "value": 0.05, "volatility": -0.12},
            "t_statistics": {"momentum": 2.1, "quality": -1.4, "value": 0.9, "volatility": -2.3},
            "p_values":     {"momentum": 0.037, "quality": 0.16, "value": 0.37, "volatility": 0.022},
        }

    def test_summary_contains_all_factors(self):
        summary = format_error_model_summary(self._make_result())
        for f in FACTOR_LABELS:
            assert f[:3].upper() in summary

    def test_significant_factors_marked(self):
        summary = format_error_model_summary(self._make_result())
        # momentum (p=0.037) and volatility (p=0.022) are significant
        assert "**" in summary

    def test_summary_is_string(self):
        result = self._make_result()
        assert isinstance(format_error_model_summary(result), str)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Bayesian calibration update — _months_between, _bayesian_blend,
#    update_error_model_if_due, get_current_coefficients
# ─────────────────────────────────────────────────────────────────────────────

from services.error_model import (
    DEFAULT_BLEND_WEIGHT,
    CALIBRATION_INTERVAL_MONTHS,
    _months_between,
    _bayesian_blend,
    _safe_float,
    update_error_model_if_due,
    get_current_coefficients,
)


class TestMonthsBetween:
    def test_exactly_six_months(self):
        d1 = date(2024, 1, 1)
        d2 = date(2024, 7, 1)
        assert abs(_months_between(d1, d2) - 6.0) < 0.1

    def test_zero_for_same_date(self):
        d = date(2024, 3, 15)
        assert abs(_months_between(d, d)) < 0.1

    def test_negative_for_past(self):
        d1 = date(2024, 6, 1)
        d2 = date(2024, 1, 1)
        assert _months_between(d1, d2) < 0

    def test_five_years(self):
        d1 = date(2019, 2, 26)
        d2 = date(2024, 2, 26)
        assert abs(_months_between(d1, d2) - 60.0) < 0.1


class TestBayesianBlend:
    def _make_ols_result(self, beta_mom=0.4, r2=0.2):
        return {
            "intercept": 0.002,
            "coefficients": {"momentum": beta_mom, "quality": 0.1,
                             "value": -0.05, "volatility": 0.02},
            "r_squared": r2,
            "adj_r_squared": r2 - 0.02,
            "n_observations": 100,
            "residual_std": 0.015,
            "mean_error": 0.001,
            "window_start": date(2019, 1, 1),
            "window_end": date(2024, 1, 1),
            "prediction_method": "shrinkage",
        }

    def _make_old_calibration(self, beta_mom=0.2):
        return {
            "intercept": 0.001,
            "coefficients": {"momentum": beta_mom, "quality": 0.05,
                             "value": -0.03, "volatility": 0.01},
            "r_squared": 0.15,
            "calibration_start_date": date(2018, 1, 1),
            "calibration_end_date": date(2023, 1, 1),
        }

    def test_blend_formula_momentum(self):
        """posterior_MOM = 0.6 * 0.4 + 0.4 * 0.2 = 0.24 + 0.08 = 0.32"""
        new = self._make_ols_result(beta_mom=0.4)
        old = self._make_old_calibration(beta_mom=0.2)
        blended = _bayesian_blend(new, old, blend_weight=0.6)
        expected = 0.6 * 0.4 + 0.4 * 0.2
        assert abs(blended["coefficients"]["momentum"] - expected) < 1e-10

    def test_first_calibration_uses_raw(self):
        """With no prior (old=None), posterior = raw OLS."""
        new = self._make_ols_result(beta_mom=0.35)
        blended = _bayesian_blend(new, None, blend_weight=0.6)
        assert abs(blended["coefficients"]["momentum"] - 0.35) < 1e-10
        assert blended["blend_weight"] == 1.0

    def test_weight_of_one_replaces_prior(self):
        """w=1.0 means completely overwrite the prior."""
        new = self._make_ols_result(beta_mom=0.5)
        old = self._make_old_calibration(beta_mom=0.1)
        blended = _bayesian_blend(new, old, blend_weight=1.0)
        assert abs(blended["coefficients"]["momentum"] - 0.5) < 1e-10

    def test_weight_of_zero_keeps_prior(self):
        """w=0.0 means completely keep the prior."""
        new = self._make_ols_result(beta_mom=0.9)
        old = self._make_old_calibration(beta_mom=0.1)
        blended = _bayesian_blend(new, old, blend_weight=0.0)
        assert abs(blended["coefficients"]["momentum"] - 0.1) < 1e-10

    def test_raw_coefficients_preserved(self):
        """raw_coefficients in blended dict = pre-blend new OLS values."""
        new = self._make_ols_result(beta_mom=0.4)
        old = self._make_old_calibration(beta_mom=0.2)
        blended = _bayesian_blend(new, old, blend_weight=0.6)
        assert abs(blended["raw_coefficients"]["momentum"] - 0.4) < 1e-10

    def test_nan_safe_blending_uses_non_nan(self):
        """If old coefficient is 0 but new is normal, result is non-NaN."""
        new = self._make_ols_result(beta_mom=0.3)
        old = self._make_old_calibration(beta_mom=0.0)
        blended = _bayesian_blend(new, old, blend_weight=0.6)
        assert not math.isnan(blended["coefficients"]["momentum"])

    def test_all_factors_blended(self):
        """All four factor labels should be present in blended output."""
        new = self._make_ols_result()
        old = self._make_old_calibration()
        blended = _bayesian_blend(new, old, blend_weight=0.6)
        for f in FACTOR_LABELS:
            assert f in blended["coefficients"]
            assert not math.isnan(blended["coefficients"][f])


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(1.5) == 1.5

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_string_returns_none(self):
        assert _safe_float("not_a_number") is None

    def test_zero(self):
        assert _safe_float(0.0) == 0.0


class TestUpdateErrorModelIfDue:
    """Tests for update_error_model_if_due — fully mocked DB and OLS."""

    def _make_new_result(self, n_obs=100):
        return {
            "intercept": 0.001,
            "coefficients": {f: 0.1 for f in FACTOR_LABELS},
            "r_squared": 0.18,
            "adj_r_squared": 0.16,
            "n_observations": n_obs,
            "residual_std": 0.02,
            "mean_error": 0.0005,
            "error_std": 0.025,
            "mean_absolute_error": 0.018,
            "window_start": date(2019, 2, 26),
            "window_end": date(2024, 2, 26),
            "prediction_method": "shrinkage",
            "stability_history": [],
            "panel_df": pd.DataFrame(),
            "method": "numpy",
            "t_statistics": {f: 1.5 for f in FACTOR_LABELS},
            "p_values": {f: 0.13 for f in FACTOR_LABELS},
        }

    def test_returns_cached_when_not_due(self):
        """When < 6 months elapsed, should return cached without recomputing."""
        last_end  = date(2024, 8, 1)
        current   = date(2024, 10, 1)   # only 2 months elapsed
        cached_db = {
            "calibration_end_date":   last_end,
            "calibration_start_date": date(2019, 8, 1),
            "intercept": 0.001,
            "coefficients": {f: 0.05 for f in FACTOR_LABELS},
            "r_squared": 0.10,
            "n_observations": 50,
            "blend_weight": 0.6,
        }
        with patch("services.error_model._load_latest_calibration", return_value=cached_db), \
             patch("services.error_model.compute_error_coefficients") as mock_compute:
            result = update_error_model_if_due(current_date=current)

        mock_compute.assert_not_called()
        assert result["updated"] is False
        assert result["calibration_source"] == "cached"

    def test_runs_calibration_when_due(self):
        """When >= 6 months elapsed, must call compute_error_coefficients."""
        last_end  = date(2024, 1, 1)
        current   = date(2024, 8, 1)    # 7 months elapsed
        cached_db = {
            "calibration_end_date":   last_end,
            "calibration_start_date": date(2019, 1, 1),
            "intercept": 0.001,
            "coefficients": {f: 0.05 for f in FACTOR_LABELS},
            "r_squared": 0.10,
            "n_observations": 50,
            "blend_weight": 0.6,
        }
        new_result = self._make_new_result()
        with patch("services.error_model._load_latest_calibration", return_value=cached_db), \
             patch("services.error_model.compute_error_coefficients", return_value=new_result), \
             patch("services.error_model._save_calibration") as mock_save:
            result = update_error_model_if_due(current_date=current)

        assert result["updated"] is True
        assert result["calibration_source"] == "new"
        mock_save.assert_called_once()

    def test_force_runs_even_when_not_due(self):
        """force=True must trigger calibration regardless of elapsed months."""
        last_end  = date(2024, 8, 1)
        current   = date(2024, 9, 1)   # only 1 month elapsed
        cached_db = {
            "calibration_end_date":   last_end,
            "calibration_start_date": date(2019, 8, 1),
            "intercept": 0.001,
            "coefficients": {f: 0.05 for f in FACTOR_LABELS},
            "r_squared": 0.10,
            "n_observations": 50,
            "blend_weight": 0.6,
        }
        new_result = self._make_new_result()
        with patch("services.error_model._load_latest_calibration", return_value=cached_db), \
             patch("services.error_model.compute_error_coefficients", return_value=new_result), \
             patch("services.error_model._save_calibration") as mock_save:
            result = update_error_model_if_due(current_date=current, force=True)

        assert result["updated"] is True
        assert "Forced" in result["reason"]
        mock_save.assert_called_once()

    def test_first_run_no_prior(self):
        """First-ever calibration with no DB history runs and returns new coefs."""
        new_result = self._make_new_result()
        with patch("services.error_model._load_latest_calibration", return_value=None), \
             patch("services.error_model.compute_error_coefficients", return_value=new_result), \
             patch("services.error_model._save_calibration"):
            result = update_error_model_if_due(current_date=date(2024, 2, 26))

        assert result["updated"] is True
        # First calibration: blend_weight should be 1.0 (no prior)
        assert result["blend_weight"] == 1.0

    def test_zero_observations_keeps_cached(self):
        """If new window yields 0 obs, return cached without updating DB."""
        last_end  = date(2024, 1, 1)
        current   = date(2024, 8, 1)
        cached_db = {
            "calibration_end_date":   last_end,
            "calibration_start_date": date(2019, 1, 1),
            "intercept": 0.001,
            "coefficients": {f: 0.05 for f in FACTOR_LABELS},
            "r_squared": 0.10,
            "n_observations": 50,
            "blend_weight": 0.6,
        }
        empty_result = self._make_new_result(n_obs=0)
        with patch("services.error_model._load_latest_calibration", return_value=cached_db), \
             patch("services.error_model.compute_error_coefficients", return_value=empty_result), \
             patch("services.error_model._save_calibration") as mock_save:
            result = update_error_model_if_due(current_date=current)

        assert result["updated"] is False
        assert result["calibration_source"] == "cached"
        mock_save.assert_not_called()

    def test_blend_applied_to_output(self):
        """When blending, the returned coefficients must be the posterior, not raw."""
        last_end  = date(2024, 1, 1)
        current   = date(2024, 8, 1)
        cached_db = {
            "calibration_end_date":   last_end,
            "calibration_start_date": date(2019, 1, 1),
            "intercept": 0.0,
            "coefficients": {"momentum": 0.2, "quality": 0.0, "value": 0.0, "volatility": 0.0},
            "r_squared": 0.10,
            "n_observations": 50,
            "blend_weight": 0.6,
        }
        new_result = self._make_new_result()
        new_result["coefficients"]["momentum"] = 0.4  # raw new
        with patch("services.error_model._load_latest_calibration", return_value=cached_db), \
             patch("services.error_model.compute_error_coefficients", return_value=new_result), \
             patch("services.error_model._save_calibration"):
            result = update_error_model_if_due(current_date=current, blend_weight=0.6)

        expected_posterior = 0.6 * 0.4 + 0.4 * 0.2   # = 0.32
        assert abs(result["coefficients"]["momentum"] - expected_posterior) < 1e-10
        # Raw coefficients preserved
        assert abs(result["raw_coefficients"]["momentum"] - 0.4) < 1e-10

    def test_returns_required_keys(self):
        """Return dict must always include these keys."""
        new_result = self._make_new_result()
        required = {"updated", "reason", "coefficients", "intercept",
                    "r_squared", "n_observations", "window_start", "window_end",
                    "blend_weight", "calibration_source"}
        with patch("services.error_model._load_latest_calibration", return_value=None), \
             patch("services.error_model.compute_error_coefficients", return_value=new_result), \
             patch("services.error_model._save_calibration"):
            result = update_error_model_if_due(current_date=date(2024, 2, 26))
        assert required.issubset(result.keys())


class TestGetCurrentCoefficients:
    def test_returns_nan_when_no_calibration(self):
        with patch("services.error_model._load_latest_calibration", return_value=None):
            result = get_current_coefficients()
        assert result["calibration_source"] == "none"
        for f in FACTOR_LABELS:
            assert math.isnan(result["coefficients"][f])

    def test_returns_cached_coefficients(self):
        cached = {
            "calibration_start_date": date(2019, 1, 1),
            "calibration_end_date":   date(2024, 1, 1),
            "intercept": 0.002,
            "coefficients": {f: 0.07 for f in FACTOR_LABELS},
            "r_squared": 0.12,
            "n_observations": 80,
            "blend_weight": 0.6,
        }
        with patch("services.error_model._load_latest_calibration", return_value=cached):
            result = get_current_coefficients()
        assert result["calibration_source"] == "cached"
        assert result["updated"] is False
        for f in FACTOR_LABELS:
            assert abs(result["coefficients"][f] - 0.07) < 1e-10

