import pytest
import pandas as pd
import numpy as np
from datetime import date
from services.risk_responsiveness import compute_rrc_scores, compute_portfolio_rrc

def test_compute_rrc_scores_sufficient_data():
    """Test RRC computation with synthetically generated data that contains a volatility spike."""
    np.random.seed(42)
    # Generate 500 days of data
    dates = pd.date_range(end=date.today(), periods=500, freq='B')
    
    # 2 assets
    # Asset 1: Low vol, responsive (less drawdown during stress)
    # Asset 2: High vol, unresponsive (high drawdown during stress)
    
    market_returns = np.random.normal(0.0005, 0.01, 500)
    
    # Introduce a volatility spike at index 400 (30 days of high vol followed by 10 days of drawdown)
    market_returns[400:430] = np.random.normal(-0.005, 0.03, 30)
    
    returns_asset1 = market_returns * 0.8 + np.random.normal(0, 0.005, 500)
    returns_asset2 = market_returns * 1.5 + np.random.normal(0, 0.01, 500)
    
    # Asset 1 adapts well -> flat during stress
    returns_asset1[400:430] = np.random.normal(0, 0.005, 30)
    
    df1 = pd.DataFrame({
        1: returns_asset1,
        2: returns_asset2
    }, index=dates)
    
    # Prices
    price_df = (1 + df1).cumprod() * 100.0
    
    rrc = compute_rrc_scores(price_df)
    
    assert len(rrc) == 2
    # Asset 1 should have a higher RRC score because it avoided the drawdown
    assert rrc[1] > rrc[2]
    assert 0 <= rrc[1] <= 100
    assert 0 <= rrc[2] <= 100

def test_compute_rrc_scores_insufficient_data():
    """Test RRC computation gracefully handles insufficient data."""
    dates = pd.date_range(end=date.today(), periods=10, freq='B')
    price_df = pd.DataFrame({
        1: np.random.rand(10) * 100,
        2: np.random.rand(10) * 100
    }, index=dates)
    
    rrc = compute_rrc_scores(price_df)
    
    # Should fall back to 50.0
    assert rrc[1] == 50.0
    assert rrc[2] == 50.0

def test_compute_portfolio_rrc():
    """Test RRC computation for a mock portfolio equity curve."""
    np.random.seed(42)
    dates = pd.date_range(end=date.today(), periods=500, freq='B')
    
    market_returns = np.random.normal(0.0005, 0.01, 500)
    market_returns[400:430] = np.random.normal(-0.005, 0.03, 30)
    
    port_returns = market_returns * 0.5 + np.random.normal(0, 0.002, 500)
    
    returns_df = pd.DataFrame({'market': market_returns}, index=dates)
    equity_curve = (1 + pd.Series(port_returns, index=dates)).cumprod() * 100.0
    
    rrc_data = compute_portfolio_rrc(equity_curve, returns_df)
    
    assert "rrc_score" in rrc_data
    assert "components" in rrc_data
    assert 0 <= rrc_data["rrc_score"] <= 100
    assert "containment" in rrc_data["components"]
    assert "beta_compression" in rrc_data["components"]
    assert "sharpe_stability" in rrc_data["components"]
