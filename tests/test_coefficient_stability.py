"""tests/test_coefficient_stability.py

Tests for:
1. Monte Carlo regression drift tests
2. Coefficient stability monitoring tests

These tests focus on the dynamic behavior of the error model's Bayesian blending
and moving windows under controlled synthetic (Monte Carlo) environments.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from unittest.mock import patch
import math

from services.error_model import (
    _bayesian_blend,
    FACTOR_LABELS,
)

class TestMonteCarloRegressionDrift:
    """Monte Carlo regression drift tests.
    
    Verifies that as the underlying true relationship (DGP) between factors and 
    forward returns changes (drifts) over time, the Bayesian blending mechanism 
    adapts smoothly and tracks the new coefficients cleanly without overreacting 
    to single-period noise.
    """
    
    def _simulate_ols_result(self, true_betas: dict, noise_std=0.01):
        """Simulate an OLS result from a true DGP."""
        # We don't need to run actual OLS, we just add Gaussian noise to the true betas
        return {
            "intercept": 0.0,
            "coefficients": {
                f: true_betas.get(f, 0.0) + np.random.normal(0, noise_std)
                for f in FACTOR_LABELS
            },
            "r_squared": 0.15,
            "n_observations": 100,
        }
        
    def test_smooth_drift_tracking(self):
        np.random.seed(42)
        
        # True beta for momentum starts at 0.5, drifts down to 0.1
        true_betas_seq = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        # Track the blended (posterior) beta
        current_prior = None
        tracked_betas = []
        
        for true_b in true_betas_seq:
            # Simulate a noisy observation of the new regime
            new_ols = self._simulate_ols_result({"momentum": true_b}, noise_std=0.05)
            
            # Blend
            if current_prior is None:
                posterior = _bayesian_blend(new_ols, None, blend_weight=1.0)
            else:
                posterior = _bayesian_blend(new_ols, current_prior, blend_weight=0.5)
                
            tracked_betas.append(posterior["coefficients"]["momentum"])
            
            # Setup for next iteration
            current_prior = posterior
            
        # The tracked betas should generally decrease, following the true betas
        # but smoother than the noisy observations
        assert tracked_betas[0] > 0.4  # Starts high
        assert tracked_betas[-1] < 0.25 # Ends much lower due to drift
        
        # Check monotonicity of the tracked drift
        # Given seed 42, we expect a strictly decreasing or mostly decreasing path
        diffs = np.diff(tracked_betas)
        assert (diffs < 0).sum() >= len(diffs) - 1, "The blended tracking should smoothly follow the downward drift"


class TestCoefficientStabilityMonitoring:
    """Coefficient stability monitoring tests.
    
    Ensures that the blending mechanism correctly rejects or dampens extreme 
    spikes (instability) in the latest window, preserving the historical established 
    relationship (prior).
    """

    def _make_prior(self):
        return {
            "intercept": 0.0,
            "coefficients": {f: 0.1 for f in FACTOR_LABELS},
            "n_observations": 500,
            "r_squared": 0.1,
            "calibration_start_date": date(2018, 1, 1),
            "calibration_end_date": date(2023, 1, 1),
        }

    def _make_new_noisy_spike(self):
        return {
            "intercept": 0.0,
            # Momentum suddenly spikes to 2.0 (unrealistic)
            "coefficients": {"momentum": 2.0, "quality": -0.5, "value": 0.1, "volatility": 0.1},
            "r_squared": 0.05,
            "n_observations": 100,
            "window_start": date(2023, 1, 1),
            "window_end": date(2023, 7, 1),
            "prediction_method": "shrinkage"
        }

    def test_bayesian_damping_of_spikes(self):
        """Test that a suddenly completely divergent OLS run is stabilized."""
        prior = self._make_prior()
        noisy_new = self._make_new_noisy_spike()
        
        # Normal blend
        blended = _bayesian_blend(noisy_new, prior, blend_weight=0.3)
        
        # The raw new coefficient was 2.0
        # The prior was 0.1
        # The posterior should be 0.3 * 2.0 + 0.7 * 0.1 = 0.6 + 0.07 = 0.67
        
        posterior_mom = blended["coefficients"]["momentum"]
        assert abs(posterior_mom - 0.67) < 1e-6
        assert posterior_mom < 1.0, "The extreme 2.0 spike must be dampened strongly by the prior"
        
        # Verify raw coefficients are still tracked for monitoring purposes
        assert abs(blended["raw_coefficients"]["momentum"] - 2.0) < 1e-6

    def test_stability_preservation_under_high_noise(self):
        """Monte carlo test: inject huge noise across multiple factors and verify bounded movement."""
        prior = self._make_prior()
        
        np.random.seed(99)
        new_result = {
            "intercept": 0.0,
            "coefficients": {
                # Massive variance
                f: 0.1 + np.random.normal(0, 1.5) for f in FACTOR_LABELS
            },
            "r_squared": 0.0,
            "n_observations": 50,
            "window_start": date(2023, 1, 1),
            "window_end": date(2023, 7, 1),
            "prediction_method": "shrinkage"
        }
        
        # If we have low confidence in the new window, we might set blend_weight low (e.g. 0.1)
        blended = _bayesian_blend(new_result, prior, blend_weight=0.1)
        
        for f in FACTOR_LABELS:
            prior_val = prior["coefficients"][f]
            post_val = blended["coefficients"][f]
            raw_val = new_result["coefficients"][f]
            
            # The posterior should be strictly much closer to the prior than the raw noisy val
            dist_to_prior = abs(post_val - prior_val)
            raw_dist_to_prior = abs(raw_val - prior_val)
            
            # Distance should be EXACTLY 10% of the raw distance
            assert abs(dist_to_prior - (0.1 * raw_dist_to_prior)) < 1e-5
            assert dist_to_prior < raw_dist_to_prior, "Blend must shrink the noisy estimate toward the prior"
