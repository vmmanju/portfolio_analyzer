from datetime import date

import pytest

from services.research_validation import run_walk_forward


def test_run_walk_forward_respects_date_range():
    """
    Ensure run_walk_forward uses the provided start_date and end_date when
    computing walk-forward windows.
    """
    start = date(2017, 3, 1)
    end = date(2026, 2, 23)

    df = run_walk_forward(strategy="equal_weight", top_n=20, use_rolling=False, start_date=start, end_date=end)

    if df.empty:
        pytest.skip("No walk-forward windows returned for the provided date range")

    # Each returned window should be within the requested bounds
    for _, row in df.iterrows():
        assert row["window_start"] >= start
        assert row["window_end"] <= end
