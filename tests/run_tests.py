from datetime import date
import sys
from pathlib import Path

# Ensure project root is on sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from services.research_validation import run_walk_forward
from services.research_validation import generate_validation_report
from services.portfolio import (
    get_latest_scoring_date,
    construct_equal_weight_portfolio,
    construct_inverse_vol_portfolio,
)
from services.backtest import get_rebalance_dates, compute_period_returns
from services.recommendation import get_stock_signal, generate_recommendation
from services.portfolio import store_portfolio
from services.scoring import compute_composite_scores, rank_cross_sectionally
import pandas as pd
import importlib.util
from pathlib import Path as _Path

# Attempt to load supplementary tests (not a package import to avoid requiring __init__)
_extra_tests_mod = None
_pt = _Path(__file__).parent / "test_portfolio_comparison.py"
if _pt.exists():
    try:
        spec = importlib.util.spec_from_file_location("test_portfolio_comparison", str(_pt))
        _extra_tests_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_extra_tests_mod)
    except Exception:
        _extra_tests_mod = None


def test_get_rebalance_dates_filter():
    start = date(2010, 1, 1)
    end = date(2026, 2, 23)
    dates = get_rebalance_dates(start_date=start, end_date=end)
    if not dates:
        print("SKIP get_rebalance_dates_filter: no dates")
        return True
    if not all(start <= d <= end for d in dates):
        print("FAIL get_rebalance_dates_filter: date out of bounds")
        return False
    print("PASS get_rebalance_dates_filter")
    return True


def test_rank_ties_behavior():
    d = pd.to_datetime(["2026-01-01"] * 3).date
    df = pd.DataFrame({
        "stock_id": [1, 2, 3],
        "date": d,
        "quality_score": [1.0, 1.0, 1.0],
        "growth_score": [0.0, 0.0, 0.0],
        "momentum_score": [0.0, 0.0, 0.0],
        "value_score": [0.0, 0.0, 0.0],
    })
    comp = compute_composite_scores(df)
    ranked = rank_cross_sectionally(comp)
    if ranked.empty or "rank" not in ranked.columns:
        print("SKIP rank_ties_behavior: no ranking")
        return True
    ranks = set(int(x) for x in ranked["rank"].dropna().tolist())
    if ranks != {1}:
        print(f"FAIL rank_ties_behavior: unexpected ranks {ranks}")
        return False
    print("PASS rank_ties_behavior")
    return True


def test_generate_recommendation_rules():
    # Construct a signal that should produce a High conviction Buy
    signal = {
        "rank": 1,
        "composite_score": 1.0,
        "momentum_score": 1.0,
        "volatility_score": -0.5,
        "signal_strength": 5,
        "trend_direction": "improving",
        "regime": "low_vol",
    }
    rec = generate_recommendation(signal, risk_profile="moderate")
    if not isinstance(rec, dict):
        print("FAIL generate_recommendation_rules: non-dict returned")
        return False
    if "recommendation" not in rec or "conviction" not in rec:
        print("FAIL generate_recommendation_rules: missing keys")
        return False
    print("PASS generate_recommendation_rules")
    return True


def test_generate_validation_report_csv():
    import tempfile
    import os

    # write reports to a temp dir
    td = tempfile.mkdtemp(prefix="rv_reports_")
    try:
        res = generate_validation_report(strategy="equal_weight", top_n=5, selected_symbols=None, save_csv=True, out_dir=td)
        if not isinstance(res, dict):
            print("FAIL generate_validation_report_csv: did not return dict")
            return False
        # expect at least one saved file in out_dir (walk_forward or sensitivity may be empty)
        files = os.listdir(td)
        if not files:
            print("FAIL generate_validation_report_csv: no files written")
            return False
        print("PASS generate_validation_report_csv")
        return True
    finally:
        # cleanup
        for f in os.listdir(td):
            try:
                os.remove(os.path.join(td, f))
            except Exception:
                pass
        try:
            os.rmdir(td)
        except Exception:
            pass


def assert_close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_walk_forward_date_range():
    start = date(2017, 3, 1)
    end = date(2026, 2, 23)
    df = run_walk_forward(strategy="equal_weight", top_n=20, use_rolling=False, start_date=start, end_date=end)
    if df.empty:
        print("SKIP walk_forward: no windows")
        return True
    for _, row in df.iterrows():
        if not (row["window_start"] >= start and row["window_end"] <= end):
            print(f"FAIL walk_forward: window out of bounds: {row}")
            return False
    print("PASS walk_forward date range")
    return True


def test_latest_scoring_date():
    d = get_latest_scoring_date()
    if d is None:
        print("SKIP latest_scoring_date: no scores")
        return True
    assert hasattr(d, "isoformat")
    print(f"PASS latest_scoring_date: {d}")
    return True


def test_construct_portfolios():
    latest = get_latest_scoring_date()
    if latest is None:
        print("SKIP construct_portfolios: no scoring date")
        return True
    eq = construct_equal_weight_portfolio(latest, top_n=3)
    if eq.empty:
        print("SKIP equal_weight: no allocations")
        return True
    total = float(eq['weight'].sum())
    if not assert_close(total, 1.0, tol=1e-6):
        print(f"FAIL equal_weight: weights sum to {total}")
        return False
    iv = construct_inverse_vol_portfolio(latest, top_n=3)
    if iv.empty:
        print("SKIP inverse_vol: no allocations")
        return True
    total_iv = float(iv['weight'].sum())
    if not assert_close(total_iv, 1.0, tol=1e-6):
        print(f"FAIL inverse_vol: weights sum to {total_iv}")
        return False
    print("PASS construct_portfolios")
    return True


def test_rebalance_and_period_returns():
    dates = get_rebalance_dates()
    if len(dates) < 2:
        print("SKIP rebalance: not enough dates")
        return True
    # Use first two rebalance dates to form a period
    start = dates[0]
    end = dates[1]
    latest = get_latest_scoring_date()
    if latest is None:
        print("SKIP period_returns: no scoring date")
        return True
    weights_df = construct_equal_weight_portfolio(latest, top_n=5)
    if weights_df.empty:
        print("SKIP period_returns: no weights")
        return True
    weights = dict(zip(weights_df['stock_id'].astype(int), weights_df['weight'].astype(float)))
    series = compute_period_returns(start, end, weights)
    if series.empty:
        print("SKIP period_returns: no returns in period")
        return True
    print("PASS period_returns")
    return True


def test_get_stock_signal():
    symbol = "RELIANCE.NS"
    sig = get_stock_signal(symbol)
    if sig is None:
        print("SKIP get_stock_signal: symbol not present or no data")
        return True
    if "recommendation" not in sig:
        print("FAIL get_stock_signal: missing recommendation in signal")
        return False
    print(f"PASS get_stock_signal: {symbol}")
    return True


def test_store_portfolio_replacement():
    latest = get_latest_scoring_date()
    if latest is None:
        print("SKIP store_portfolio: no scoring date")
        return True
    alloc_df = construct_equal_weight_portfolio(latest, top_n=3)
    if alloc_df.empty:
        print("SKIP store_portfolio: no allocation")
        return True
    strategy = "test_strategy_run_tests"
    ins1, del1 = store_portfolio(latest, alloc_df, strategy)
    ins2, del2 = store_portfolio(latest, alloc_df, strategy)
    if ins1 != len(alloc_df):
        print(f"FAIL store_portfolio insert mismatch: {ins1} vs {len(alloc_df)}")
        return False
    if del2 != ins1:
        print(f"FAIL store_portfolio replace mismatch: deleted {del2} expected {ins1}")
        return False
    print("PASS store_portfolio_replacement")
    return True


def test_compute_and_rank():
    # Build a small factor DataFrame
    d = pd.to_datetime(["2026-01-01"] * 3).date
    df = pd.DataFrame({
        "stock_id": [1, 2, 3],
        "date": d,
        "quality_score": [1.0, 0.0, 0.0],
        "growth_score": [0.0, 1.0, 0.0],
        "momentum_score": [0.0, 0.0, 1.0],
        "value_score": [0.0, 0.0, 0.0],
    })
    comp = compute_composite_scores(df)
    if "composite_score" not in comp.columns:
        print("FAIL compute_composite_scores: missing column")
        return False
    ranked = rank_cross_sectionally(comp)
    if "rank" not in ranked.columns:
        print("FAIL rank_cross_sectionally: missing rank")
        return False
    print("PASS compute_and_rank")
    return True


def main():
    tests = [
        test_walk_forward_date_range,
        test_latest_scoring_date,
        test_construct_portfolios,
        test_rebalance_and_period_returns,
        test_get_stock_signal,
        test_store_portfolio_replacement,
        test_compute_and_rank,
        test_get_rebalance_dates_filter,
        test_rank_ties_behavior,
        test_generate_recommendation_rules,
    ]

    # Include supplementary tests if available
    for mod_name, file_name in [
        ("portfolio_comparison", "test_portfolio_comparison.py"),
        ("model_governance", "test_model_governance.py"),
    ]:
        _pt = _Path(__file__).parent / file_name
        if _pt.exists():
            try:
                spec = importlib.util.spec_from_file_location(f"test_{mod_name}", str(_pt))
                test_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(test_mod)
                for name in dir(test_mod):
                    if name.startswith("test_"):
                        fn = getattr(test_mod, name)
                        if callable(fn):
                            tests.append(fn)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    all_ok = True
    for t in tests:
        try:
            ok = t()
        except Exception as e:
            print(f"ERROR running {t.__name__}: {e}")
            ok = False
        if not ok:
            print(f"FAIL identified in: {t.__name__}")
            all_ok = False
    if all_ok:
        print("\nALL TESTS PASSED")
        return 0
    else:
        print("\nSOME TESTS FAILED")
        return 2


if __name__ == "__main__":
    sys.exit(main())
