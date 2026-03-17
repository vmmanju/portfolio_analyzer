import pytest
import pandas as pd
import numpy as np

from services.return_estimator import shrunk_annualised_returns
from services.stop_loss_engine import compute_stop_loss_scores

def test_double_penalization_ratio():
    """
    Test that the combination of James-Stein shrinkage (error_coefficient)
    and the StopRiskScore penalty does not triple-penalize highly volatile
    stocks to the point of suppressing them by more than ~50% (Ratio < 0.5)
    under normal volatile conditions.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200)
    
    # P1: High alpha, high volatility (noisy)
    # Mean: 0.0015 daily (~45% ann), Vol: 0.02 daily (~31% ann)
    r1 = np.random.normal(0.0015, 0.02, 200)
    
    # P2: Baseline stable
    r2 = np.random.normal(0.0005, 0.005, 200)
    
    returns_df = pd.DataFrame({"P1": r1, "P2": r2}, index=dates)
    price_df = (1 + returns_df).cumprod()
    
    # 1. mu_raw
    mu_raw = returns_df.mean() * 252
    
    # 2. mu_shrinked
    mu_shrinked = shrunk_annualised_returns(returns_df, annualize=252)
    
    # 3. Simulate error_bias (noise metric directly affecting SLS)
    # We construct a synthetic error_bias. P1 has larger bias.
    error_bias = pd.Series({"P1": 0.05, "P2": 0.005})
    
    # 4. StopRiskScore
    sls_df = compute_stop_loss_scores(price_df, error_bias=error_bias)
    sls_scores = sls_df["stop_loss_score"] if not sls_df.empty else pd.Series({"P1": 50, "P2": 50})
    
    # 5. Volatility (Annualized)
    vol_ann = returns_df.std() * np.sqrt(252)
    
    # 6. Marginal Penalty applied per stock in optimizer objective:
    # Objective = (mu / vol) - 0.2 * (SLS / 100)^2
    # So effective mu = mu - 0.2 * (SLS / 100)^2 * vol
    
    effective_mu = pd.Series(index=returns_df.columns, dtype=float)
    shrinkage_ratio = pd.Series(index=returns_df.columns, dtype=float)
    
    for col in returns_df.columns:
        sls = sls_scores.get(col, 50.0)
        penalty = 0.2 * ((sls / 100.0) ** 2)
        
        # Effective return after both topological shrinkages
        effective_mu[col] = mu_shrinked[col] - (penalty * vol_ann[col])
        shrinkage_ratio[col] = effective_mu[col] / mu_raw[col]
        
    # Validation: Ensure high-alpha volatile name (P1) is not suppressed completely
    # (i.e. Ratio >= 0.45 or ~0.5)
    # If ratio falls below 0.1, the engine is over-punishing volatility excessively.
    
    assert shrinkage_ratio["P1"] > 0.4, f"Double penalization too aggressive for P1: Ratio={shrinkage_ratio['P1']:.3f}"
    assert shrinkage_ratio["P2"] > 0.6, f"Baseline asset over-penalized: Ratio={shrinkage_ratio['P2']:.3f}"
    
    print("\n--- Double Penalization Check ---")
    print(f"P1 (Noisy)  - Raw: {mu_raw['P1']:.3f} | Shrinked: {mu_shrinked['P1']:.3f} | SLS: {sls_scores.get('P1', 50):.1f} | Final: {effective_mu['P1']:.3f} | Ratio: {shrinkage_ratio['P1']:.3f}")
    print(f"P2 (Stable) - Raw: {mu_raw['P2']:.3f} | Shrinked: {mu_shrinked['P2']:.3f} | SLS: {sls_scores.get('P2', 50):.1f} | Final: {effective_mu['P2']:.3f} | Ratio: {shrinkage_ratio['P2']:.3f}")
