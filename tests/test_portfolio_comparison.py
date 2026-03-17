from datetime import date

import pandas as pd
import numpy as np

import services.portfolio_comparison as pc


def test_rate_portfolios_basic():
    # Three portfolios with distinct metrics
    results = {
        "P1": {"metrics": {"Sharpe": 1.2, "CAGR": 0.15, "Max Drawdown": -0.10, "Calmar": 1.5}},
        "P2": {"metrics": {"Sharpe": 0.8, "CAGR": 0.08, "Max Drawdown": -0.20, "Calmar": 0.4}},
        "P3": {"metrics": {"Sharpe": 1.5, "CAGR": 0.20, "Max Drawdown": -0.05, "Calmar": 4.0}},
    }
    rated = pc.rate_portfolios(results)
    if not isinstance(rated, dict) or set(rated.keys()) != set(results.keys()):
        print("FAIL rate_portfolios_basic: invalid output")
        return False

    # Scores between 0 and 1 (min-max normalized combination)
    for v in rated.values():
        sc = v.get("rating_score", None)
        if sc is None or not (0.0 <= sc <= 1.0):
            print(f"FAIL rate_portfolios_basic: score out of range: {sc}")
            return False

    # Ensure grades are one of A/B/C/D
    for v in rated.values():
        if v.get("grade") not in ("A", "B", "C", "D"):
            print(f"FAIL rate_portfolios_basic: invalid grade {v.get('grade')}")
            return False

    print("PASS rate_portfolios_basic")
    return True


def _make_eq_df(dts, returns):
    return pd.DataFrame({"date": pd.to_datetime(dts), "daily_return": returns, "cumulative_return": (1 + pd.Series(returns)).cumprod() - 1, "drawdown": (1 + pd.Series(returns)).cumprod().cummax() - (1 + pd.Series(returns)).cumprod()})


def test_compute_portfolio_correlation():
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    r1 = [0.01, -0.005, 0.002]
    r2 = [0.009, -0.004, 0.003]

    results = {
        "A": {"equity_curve": _make_eq_df(dates, r1)},
        "B": {"equity_curve": _make_eq_df(dates, r2)},
    }

    corr, cov = pc.compute_portfolio_correlation(results)
    if corr is None or cov is None:
        print("FAIL compute_portfolio_correlation: returned None")
        return False
    if corr.shape != (2, 2):
        print(f"FAIL compute_portfolio_correlation: unexpected shape {corr.shape}")
        return False
    # diagonal should be 1.0
    if not np.isclose(corr.loc["A", "A"], 1.0):
        print("FAIL compute_portfolio_correlation: diag not 1")
        return False

    print("PASS compute_portfolio_correlation")
    return True


def test_construct_meta_portfolio_basic():
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]
    r1 = [0.01, -0.005, 0.002, 0.003]
    r2 = [0.008, -0.002, 0.001, 0.002]
    r3 = [0.0, 0.01, -0.005, 0.004]

    results = {
        "P1": {"equity_curve": _make_eq_df(dates, r1)},
        "P2": {"equity_curve": _make_eq_df(dates, r2)},
        "P3": {"equity_curve": _make_eq_df(dates, r3)},
    }

    meta = pc.construct_meta_portfolio(results)
    weights = meta.get("weights", {})
    if not weights:
        print("FAIL construct_meta_portfolio_basic: no weights returned")
        return False
    # Sum to 1
    s = sum(weights.values())
    if not np.isclose(s, 1.0):
        print(f"FAIL construct_meta_portfolio_basic: weights not sum to 1 ({s})")
        return False
    # Each weight within [0, 0.6]
    for w in weights.values():
        if w < -1e-8 or w > 0.6 + 1e-6:
            print(f"FAIL construct_meta_portfolio_basic: weight out of bounds {w}")
            return False

    meq = meta.get("meta_equity_curve")
    if meq is None or meq.empty:
        print("FAIL construct_meta_portfolio_basic: meta equity empty")
        return False

    mm = meta.get("meta_metrics", {})
    if not all(k in mm for k in ("CAGR", "Sharpe", "Max Drawdown")):
        print("FAIL construct_meta_portfolio_basic: missing meta metrics")
        return False

    print("PASS construct_meta_portfolio_basic")
    return True


def test_backtest_user_portfolios_uses_cache():
    start = date(2020, 1, 1)
    end = date(2020, 12, 31)
    p = pc.UserPortfolio(name="Cached", symbols=["A"], strategy="equal_weight", regime_mode="static", top_n=5)

    key = pc._cache_key_for_portfolio(p, start, end)
    # create a synthetic equity curve
    eq = _make_eq_df(["2020-01-01", "2020-01-02"], [0.01, -0.005])
    metrics = {"CAGR": 0.01, "Volatility": 0.1, "Sharpe": 0.1, "Max Drawdown": -0.005, "Total Return": 0.005, "Win Rate": 0.5, "n_days": 2}
    pc._BACKTEST_CACHE[key] = {"equity_curve": eq, "metrics": metrics, "warnings": []}

    res = pc.backtest_user_portfolios([p], start, end)
    if "Cached" not in res:
        print("FAIL backtest_user_portfolios_uses_cache: missing key in results")
        return False
    out = res["Cached"]
    if out.get("metrics", {}).get("CAGR") != metrics["CAGR"]:
        print("FAIL backtest_user_portfolios_uses_cache: metrics mismatch")
        return False

    print("PASS backtest_user_portfolios_uses_cache")
    return True


def main():
    tests = [
        test_rate_portfolios_basic,
        test_compute_portfolio_correlation,
        test_construct_meta_portfolio_basic,
        test_backtest_user_portfolios_uses_cache,
        test_construct_meta_portfolio_constraints,
    ]
    all_ok = True
    for t in tests:
        try:
            ok = t()
        except Exception as e:
            print(f"ERROR running {t.__name__}: {e}")
            ok = False
        if not ok:
            all_ok = False
    if all_ok:
        print("\nALL PORTFOLIO COMPARISON TESTS PASSED")
        return 0
    else:
        print("\nSOME PORTFOLIO COMPARISON TESTS FAILED")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


def test_backtest_user_portfolios_empty_list_returns_empty():
    res = pc.backtest_user_portfolios([], date(2020, 1, 1), date(2020, 12, 31))
    if not isinstance(res, dict) or res:
        print("FAIL backtest_user_portfolios_empty_list_returns_empty: expected empty dict")
        return False
    print("PASS backtest_user_portfolios_empty_list_returns_empty")
    return True


def test_cache_key_truncation_symbols_limit():
    # Create > MAX_STOCKS_PER_PORTFOLIO symbols and ensure the cache key truncates
    many = [f"SYM{i}" for i in range(100)]
    p = pc.UserPortfolio(name="LongList", symbols=many, strategy="equal_weight", regime_mode="static", top_n=5)
    key = pc._cache_key_for_portfolio(p, date(2020, 1, 1), date(2020, 12, 31))
    try:
        import json
        payload = json.loads(key)
    except Exception:
        print("FAIL cache_key_truncation_symbols_limit: key not JSON")
        return False
    if len(payload.get("symbols", [])) != pc.MAX_STOCKS_PER_PORTFOLIO:
        print(f"FAIL cache_key_truncation_symbols_limit: expected {pc.MAX_STOCKS_PER_PORTFOLIO} symbols got {len(payload.get('symbols', []))}")
        return False
    print("PASS cache_key_truncation_symbols_limit")
    return True


def test_rate_portfolio_requires_normalized_keys_raises():
    try:
        pc.rate_portfolio({"Sharpe": 0.5})
        print("FAIL rate_portfolio_requires_normalized_keys_raises: did not raise")
        return False
    except ValueError:
        print("PASS rate_portfolio_requires_normalized_keys_raises")
        return True


def test_construct_meta_portfolio_optimizer_fallback():
    # Force optimizer to report failure by monkeypatching scipy.optimize.minimize if available
    try:
        import scipy.optimize as _sco
    except Exception:
        print("SKIP construct_meta_portfolio_optimizer_fallback: scipy not available")
        return True

    from types import SimpleNamespace
    orig_minimize = getattr(_sco, "minimize", None)
    try:
        _sco.minimize = lambda *a, **k: SimpleNamespace(success=False, x=None)
        dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
        r1 = [0.01, -0.005, 0.002]
        r2 = [0.008, -0.002, 0.001]
        results = {
            "A": {"equity_curve": _make_eq_df(dates, r1)},
            "B": {"equity_curve": _make_eq_df(dates, r2)},
        }
        meta = pc.construct_meta_portfolio(results)
        weights = meta.get("weights", {})
        if not weights:
            print("FAIL construct_meta_portfolio_optimizer_fallback: no weights returned")
            return False
        s = sum(weights.values())
        if abs(s - 1.0) > 1e-8:
            print(f"FAIL construct_meta_portfolio_optimizer_fallback: weights sum {s}")
            return False
        print("PASS construct_meta_portfolio_optimizer_fallback")
        return True
    finally:
        if orig_minimize is not None:
            _sco.minimize = orig_minimize


def test_compute_portfolio_correlation_with_all_nan_returns():
    dates = ["2020-01-01", "2020-01-02"]
    r1 = [float("nan"), float("nan")]
    r2 = [float("nan"), float("nan")]
    results = {
        "A": {"equity_curve": _make_eq_df(dates, r1)},
        "B": {"equity_curve": _make_eq_df(dates, r2)},
    }
    corr, cov = pc.compute_portfolio_correlation(results)
    if corr is None or cov is None:
        print("FAIL compute_portfolio_correlation_with_all_nan_returns: returned None")
        return False
    if not corr.isna().all().all():
        print("FAIL compute_portfolio_correlation_with_all_nan_returns: expected all-NaN correlation")
        return False
    print("PASS compute_portfolio_correlation_with_all_nan_returns")
    return True


def test_construct_meta_portfolio_stock_level_allocation_basic():
    # Build simple return series (re-using helper)
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]
    r1 = [0.01, -0.005, 0.002, 0.003]
    r2 = [0.008, -0.002, 0.001, 0.002]
    r3 = [0.0, 0.01, -0.005, 0.004]

    results = {
        "P1": {"equity_curve": _make_eq_df(dates, r1)},
        "P2": {"equity_curve": _make_eq_df(dates, r2)},
        "P3": {"equity_curve": _make_eq_df(dates, r3)},
    }

    # Prepare UserPortfolio defs (names must match results keys)
    ups = [
        pc.UserPortfolio(name="P1", symbols=["A"], strategy="equal_weight", regime_mode="static", top_n=2),
        pc.UserPortfolio(name="P2", symbols=["B"], strategy="equal_weight", regime_mode="static", top_n=2),
        pc.UserPortfolio(name="P3", symbols=["C"], strategy="equal_weight", regime_mode="static", top_n=3),
    ]

    # Patch the allocation builders inside the portfolio_comparison module so we don't rely on DB scoring
    orig_eq = getattr(pc, "construct_equal_weight_portfolio", None)
    orig_iv = getattr(pc, "construct_inverse_vol_portfolio", None)
    orig_db = getattr(pc, "get_db_context", None)

    def mk_df(ids, weights):
        return pd.DataFrame({"stock_id": ids, "weight": weights})

    outs = [
        mk_df([101, 102], [0.5, 0.5]),
        mk_df([102, 103], [0.6, 0.4]),
        mk_df([103, 104, 105], [0.4, 0.3, 0.3]),
    ]

    def mock_eq(as_of_date, top_n=20, selected_symbols=None):
        if outs:
            return outs.pop(0).copy()
        return pd.DataFrame(columns=["stock_id", "weight"]) 

    try:
        pc.construct_equal_weight_portfolio = mock_eq
        pc.construct_inverse_vol_portfolio = mock_eq

        meta = pc.construct_meta_portfolio(results, portfolios=ups, start_date=date(2020, 1, 1), end_date=date(2020, 1, 6))

        # Validate stock-level results present
        if "stock_weights" not in meta:
            print("FAIL construct_meta_portfolio_stock_level_allocation_basic: missing stock_weights")
            return False

        sw = meta.get("stock_weights", {})
        if not isinstance(sw, dict) or not sw:
            print("FAIL construct_meta_portfolio_stock_level_allocation_basic: empty stock_weights")
            return False

        # Keys should represent the union of stock ids (either as id-strings or mapped symbols)
        keys = set(sw.keys())
        expected_ids = {101, 102, 103, 104, 105}
        if len(keys) != len(expected_ids):
            print(f"FAIL construct_meta_portfolio_stock_level_allocation_basic: expected {len(expected_ids)} stock entries but got {len(keys)}: {keys}")
            return False
        # Numeric keys (if any) must correspond to expected ids
        numeric_keys = {int(k) for k in keys if isinstance(k, str) and k.isdigit()}
        if not numeric_keys.issubset(expected_ids):
            print(f"FAIL construct_meta_portfolio_stock_level_allocation_basic: numeric keys {numeric_keys} unexpected")
            return False

        # weights should sum to ~1
        total = sum(float(v) for v in sw.values())
        if not np.isclose(total, 1.0):
            print(f"FAIL construct_meta_portfolio_stock_level_allocation_basic: weights sum to {total}")
            return False

        # Now simulate DB mapping to symbols by patching get_db_context used inside module
        class DummyDB:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def execute(self, stmt):
                class Res:
                    def all(self):
                        return [(101, "AAA"), (102, "BBB"), (103, "CCC"), (104, "DDD"), (105, "EEE")]
                return Res()

        # reset outs for second run
        outs2 = [
            mk_df([101, 102], [0.5, 0.5]),
            mk_df([102, 103], [0.6, 0.4]),
            mk_df([103, 104, 105], [0.4, 0.3, 0.3]),
        ]

        def mock_eq2(as_of_date, top_n=20, selected_symbols=None):
            if outs2:
                return outs2.pop(0).copy()
            return pd.DataFrame(columns=["stock_id", "weight"]) 

        pc.construct_equal_weight_portfolio = mock_eq2
        pc.construct_inverse_vol_portfolio = mock_eq2
        pc.get_db_context = lambda: DummyDB()

        meta2 = pc.construct_meta_portfolio(results, portfolios=ups, start_date=date(2020, 1, 1), end_date=date(2020, 1, 6))
        sw2 = meta2.get("stock_weights", {})
        if not sw2:
            print("FAIL construct_meta_portfolio_stock_level_allocation_basic: second-run stock_weights empty")
            return False

        # Expect symbol keys from mapping
        expected_syms = {"AAA", "BBB", "CCC", "DDD", "EEE"}
        if not expected_syms.issubset(set(sw2.keys())):
            print(f"FAIL construct_meta_portfolio_stock_level_allocation_basic: expected symbols {expected_syms} got {set(sw2.keys())}")
            return False

        print("PASS construct_meta_portfolio_stock_level_allocation_basic")
        return True
    finally:
        # restore
        if orig_eq is not None:
            pc.construct_equal_weight_portfolio = orig_eq
        if orig_iv is not None:
            pc.construct_inverse_vol_portfolio = orig_iv
        if orig_db is not None:
            pc.get_db_context = orig_db

def test_construct_meta_portfolio_constraints():
    # Provide 3 portfolios. P1 is a clear winner, P2 and P3 are clear losers.
    # Unconstrained optimization would put 100% in P1 and 0% in others.
    # The constraints dictate max 60% per portfolio, min 5% per portfolio.
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]
    r1 = [0.05, 0.05, 0.05, 0.05]   # clear winner
    r2 = [-0.05, -0.05, -0.05, -0.05] # clear loser
    r3 = [-0.04, -0.04, -0.04, -0.04] # clear loser

    results = {
        "P1": {"equity_curve": _make_eq_df(dates, r1)},
        "P2": {"equity_curve": _make_eq_df(dates, r2)},
        "P3": {"equity_curve": _make_eq_df(dates, r3)},
    }

    meta = pc.construct_meta_portfolio(results)
    weights = meta.get("weights", {})
    
    # Check bounds. We expect max_weight_cap to be 0.6 with 3 assets.
    # P1 should hit the cap, and P2/P3 should share the remaining 0.4.
    assert weights["P1"] <= 0.601, f"P1 weight too high: {weights['P1']}"
    assert weights["P2"] >= -0.001, f"P2 weight too low: {weights['P2']}"
    assert weights["P3"] >= -0.001, f"P3 weight too low: {weights['P3']}"
    assert abs(sum(weights.values()) - 1.0) < 1e-5, f"Weights don't sum to 1: {sum(weights.values())}"
    
    print("PASS test_construct_meta_portfolio_constraints")
    return True
