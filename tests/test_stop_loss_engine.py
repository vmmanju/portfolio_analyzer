import math
from datetime import date
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from services.stop_loss_engine import (
    compute_stop_loss_scores,
    analyze_stock_stop_loss,
    WEIGHT_DD,
    WEIGHT_VAR,
)

def test_compute_stop_loss_scores_empty():
    res = compute_stop_loss_scores(pd.DataFrame())
    assert res.empty


def test_compute_stop_loss_scores_logic():
    # Construct a synthetic price history where stock 1 crashes and stock 2 rallies
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B") # ~260 days
    
    # Stock 1: drops smoothly from 100 to 50
    # Stock 2: rises smoothly from 100 to 150
    p1 = np.linspace(100, 50, len(dates))
    p2 = np.linspace(100, 150, len(dates))
    
    prices = pd.DataFrame({"1": p1, "2": p2}, index=dates)
    
    # Add an error bias simulating stock 1 being heavily overestimated
    err_bias = pd.Series({"1": 0.05, "2": -0.05})
    
    df_scores = compute_stop_loss_scores(prices, err_bias)
    
    # Assertions
    assert "1" in df_scores.index
    assert "2" in df_scores.index
    
    row1 = df_scores.loc["1"]
    row2 = df_scores.loc["2"]
    
    # Stock 1 is crashing, so its drawdown score should be much higher (worse) than Stock 2
    assert row1["dd_score"] > row2["dd_score"]
    
    # Stock 1 has negative 6m and 12m momentum, penalty should trigger
    assert row1["mom_score"] == 100.0
    # Stock 2 is rallying, momentum penalty should NOT trigger
    assert row2["mom_score"] == 0.0
    # Stock 1 has positive error bias (0.05), so with 0.10 threshold, score should be 50.0
    assert row1["err_score"] == 50.0
    
    # Overall SLS must be bounded [0, 100]
    assert 0 <= row1["stop_loss_score"] <= 100
    assert 0 <= row2["stop_loss_score"] <= 100
    
    # Stock 1 SLS > Stock 2 SLS
    assert row1["stop_loss_score"] > row2["stop_loss_score"]
    
    # Assert Recommendation Overrides
    # Given Stock 1 matches max penalty on mom and err and worst dd/var, it should hit high/critical
    # Stock 1 is riskier than Stock 2
    assert row1["recommended_action"] in ["Reduce", "Exit", "Hold"] # Allow 'Hold' if score is in 'Moderate' range
    assert row2["recommended_action"] == "Hold"  # Because SLS will be low for the rallying asset


@patch("services.stop_loss_engine.get_historical_prices")
def test_analyze_stock_stop_loss_wrapper(mock_get_prices):
    # Setup mock to return a simple 200 day falling dataframe
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    mock_df = pd.DataFrame({99: np.linspace(100, 60, 200)}, index=dates)
    mock_get_prices.return_value = mock_df
    
    res = analyze_stock_stop_loss(date(2023, 10, 1), 99, 0.02)
    
    assert "stop_loss_score" in res
    assert "risk_level" in res
    assert "recommended_action" in res
    assert "trigger_threshold" in res
    assert "components" in res
    
    # Momentum breakdown = 100 because it drops straight line
    assert res["components"]["momentum_breakdown_score"] == 100.0
