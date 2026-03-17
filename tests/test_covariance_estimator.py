import pytest
import numpy as np
import pandas as pd

from services.covariance_estimator import robust_covariance_matrix, validate_covariance, _ensure_positive_definite

def test_covariance_estimator_empty_df():
    df = pd.DataFrame()
    res = robust_covariance_matrix(df, log_diagnostics=False)
    assert res["matrix"].shape == (1, 0)
    assert res["shrinkage"] == 0.0

def test_covariance_estimator_single_asset():
    df = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.03, 0.01]})
    res = robust_covariance_matrix(df, log_diagnostics=False)
    assert res["matrix"].shape == (1, 1)
    
def test_covariance_estimator_all_nans():
    df = pd.DataFrame({"A": [np.nan, np.nan], "B": [np.nan, np.nan]})
    res = robust_covariance_matrix(df, log_diagnostics=False)
    assert res["matrix"].shape == (1, 1) # Fallback because < 2 assets valid
    assert res["matrix"][0, 0] == 1e-6

def test_covariance_estimator_perfect_multicollinearity_needs_jitter():
    # Construct an ill-conditioned dataset where collinearity breaks PD
    df = pd.DataFrame({
        "A": [0.01, -0.02, 0.03, -0.01, 0.00],
        "B": [0.01, -0.02, 0.03, -0.01, 0.00], # exact duplicate
        "C": [0.02, -0.04, 0.06, -0.02, 0.00]  # exact scaled
    })
    res = robust_covariance_matrix(df, method="manual", manual_alpha=1.0, log_diagnostics=False)
    # The matrix should be forcefully jittered to become PD
    assert res["jitter_applied"] > 0.0
    val = validate_covariance(res["matrix"])
    assert val["is_positive_definite"] is True

def test_covariance_estimator_manual_method():
    df = pd.DataFrame({
        "A": [0.01, -0.02, 0.03, -0.01, 0.02],
        "B": [-0.01, 0.02, -0.01, 0.01, 0.00]
    })
    res = robust_covariance_matrix(df, method="manual", manual_alpha=0.5, log_diagnostics=False)
    assert res["method_used"] == "manual"
    val = validate_covariance(res["matrix"])
    assert val["is_positive_definite"] is True

def test_ensure_positive_definite():
    # Create non-PD matrix manually
    mat = np.array([[1.0, 1.0], [1.0, 1.0]])
    mat_pd, jitter = _ensure_positive_definite(mat)
    assert jitter > 0.0
    assert np.linalg.eigvalsh(mat_pd).min() > 0.0
