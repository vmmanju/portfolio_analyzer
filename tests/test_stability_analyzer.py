import pytest
import pandas as pd
import numpy as np

from services.stability_analyzer import compute_rolling_stability

def test_stability_analyzer_empty_df():
    df = pd.DataFrame()
    res = compute_rolling_stability(df)
    assert res["stability_score"] == 0.0
    assert res["data_sufficiency"] == "insufficient"

def test_stability_analyzer_missing_column():
    df = pd.DataFrame({"random_col": [1.0, 2.0]})
    res = compute_rolling_stability(df)
    assert res["stability_score"] == 0.0
    assert res["data_sufficiency"] == "insufficient"

def test_stability_analyzer_low_data():
    # Provide only 50 rows of data, which is less than the 126 minimum required for "sufficient"
    # It should return without crashing.
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=50),
        "daily_return": np.random.normal(0.001, 0.01, 50)
    })
    res = compute_rolling_stability(df)
    assert res["data_sufficiency"] in ["limited", "insufficient"]
    # Ensure there's a score (even if it's default/imperfect) and it doesn't crash
    assert "stability_score" in res

def test_stability_analyzer_zero_variance_returns():
    # If a portfolio has exactly 0.0 return every day
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=200),
        "daily_return": [0.0] * 200
    })
    res = compute_rolling_stability(df)
    # The score should compute cleanly without divide-by-zero errors
    assert isinstance(res["stability_score"], float)
