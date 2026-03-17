"""Portfolio comparison, rating and meta-portfolio optimizer.

Provides:
- `UserPortfolio` dataclass
- `backtest_user_portfolios` to run cached backtests for up to 5 user portfolios
- `rate_portfolios` / `rate_portfolio` for normalized rating and grading
- `compute_portfolio_correlation` to obtain correlation & covariance matrices
- `construct_meta_portfolio` to perform mean-variance optimization across portfolios

Notes:
- Keeps a simple in-memory cache keyed by portfolio definition + date range to avoid
  re-running expensive backtests during an interactive session.
- Uses scipy.optimize when available; falls back to a simple heuristic if not.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from datetime import date

import math
import json

import pandas as pd
import numpy as np
import concurrent.futures

from services.backtest import run_backtest, construct_portfolio_for_date, get_rebalance_dates, compute_period_returns, calculate_transaction_cost
from services.portfolio import construct_equal_weight_portfolio, construct_inverse_vol_portfolio, get_latest_scoring_date
from services.covariance_estimator import robust_covariance_matrix
from services.return_estimator import shrunk_annualised_returns
from services.stability_analyzer import compute_rolling_stability
from services.portfolio_rating import (
    compute_composite_portfolio_rating,
    build_rating_input_from_results,
    grade_colour,
    top_portfolio as _top_portfolio,
)
from app.models import Stock, Price
from app.database import get_db_context
from sqlalchemy import select

# Configuration
MAX_PORTFOLIOS = 6           # 5 user + 1 auto-hybrid slot
MAX_STOCKS_PER_PORTFOLIO = 50
ANNUALIZE = 252
RISK_FREE = 0.0

# Auto-hybrid strategy identifier (matches auto_diversified_portfolio.py)
STRATEGY_AUTO_HYBRID = "auto_diversified_hybrid"

# Simple in-memory cache to avoid recomputation in the same process/session
_BACKTEST_CACHE: Dict[str, Dict[str, Any]] = {}


@dataclass
class UserPortfolio:
    name: str
    symbols: List[str]
    strategy: str
    regime_mode: str
    top_n: int


def _cache_key_for_portfolio(p: UserPortfolio, start_date: date, end_date: date) -> str:
    payload = {
        "name": p.name,
        "symbols": sorted(p.symbols or [])[:MAX_STOCKS_PER_PORTFOLIO],
        "strategy": p.strategy,
        "regime_mode": p.regime_mode,
        "top_n": p.top_n,
        "start": start_date.isoformat() if start_date else None,
        "end": end_date.isoformat() if end_date else None,
    }
    return json.dumps(payload, sort_keys=True)


def _compute_metrics_from_returns(daily_returns: pd.Series) -> Dict[str, float]:
    """Calculate standard performance metrics from a daily returns series."""
    out: Dict[str, float] = {}
    if daily_returns is None or daily_returns.empty:
        return {
            "CAGR": 0.0,
            "Volatility": 0.0,
            "Sharpe": 0.0,
            "Max Drawdown": 0.0,
            "Sortino": 0.0,
            "Calmar": 0.0,
            "Total Return": 0.0,
        }

    dr = daily_returns.dropna().astype(float)
    if dr.empty:
        return {
            "CAGR": 0.0,
            "Volatility": 0.0,
            "Sharpe": 0.0,
            "Max Drawdown": 0.0,
            "Sortino": 0.0,
            "Calmar": 0.0,
            "Total Return": 0.0,
        }

    total_return = float((1 + dr).cumprod().iloc[-1] - 1)
    n_days = dr.shape[0]
    n_years = n_days / ANNUALIZE if n_days > 0 else 0.0
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol = float(dr.std(ddof=0) * math.sqrt(ANNUALIZE)) if n_days > 1 else 0.0
    sharpe = float(((dr.mean() * ANNUALIZE) - RISK_FREE) / ann_vol) if ann_vol and ann_vol > 0 else 0.0

    cum = (1 + dr).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    # Sortino: downside deviation
    downside = dr.copy()
    downside[downside > 0] = 0
    downside_std = math.sqrt((downside ** 2).mean()) * math.sqrt(ANNUALIZE) if not downside.empty else 0.0
    sortino = float(((dr.mean() * ANNUALIZE) - RISK_FREE) / downside_std) if downside_std and downside_std > 0 else 0.0

    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("inf")

    out["CAGR"] = cagr
    out["Volatility"] = ann_vol
    out["Sharpe"] = sharpe
    out["Max Drawdown"] = max_dd
    out["Sortino"] = sortino
    out["Calmar"] = calmar
    out["Total Return"] = total_return
    out["n_days"] = n_days
    return out


def _minmax_scalar(val: float, arr: np.ndarray) -> float:
    if np.nanmax(arr) == np.nanmin(arr):
        return 0.5
    return float((val - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr)))


def _worker_run_portfolio(p: UserPortfolio, start_date: date, end_date: date, key: str) -> tuple[str, str, dict]:
    # Discard any inherited DB connections from parent process to avoid psycopg2 OperationalErrors during multiprocessing
    from app.database import engine
    engine.dispose(close=False)

    # enforce stock limit
    symbols = (p.symbols or [])[:MAX_STOCKS_PER_PORTFOLIO]
    target_vol = 0.15 if p.regime_mode == "volatility_targeting" else None

    # ── Route hybrid portfolios to the dedicated builder ─────────────────
    _hybrid_extra_warnings: List[str] = []
    if p.strategy == STRATEGY_AUTO_HYBRID:
        from services.auto_diversified_portfolio import backtest_hybrid_portfolio
        equity_curve, summary = backtest_hybrid_portfolio(
            start_date=start_date,
            end_date=end_date,
            top_n=p.top_n,
            regime_mode=p.regime_mode,
        )
        # Pop hybrid-specific warnings out of the summary dict
        _hybrid_extra_warnings = summary.pop("warnings", [])
    else:
        equity_curve, summary = run_backtest(
            strategy=p.strategy,
            top_n=p.top_n,
            start_date=start_date,
            end_date=end_date,
            target_vol=target_vol,
            selected_symbols=symbols,
        )

    # Ensure equity_curve has a daily_return column
    daily = pd.Series(dtype=float)
    if equity_curve is not None and not equity_curve.empty and "daily_return" in equity_curve.columns:
        eq_df = equity_curve.copy()
        try:
            eq_df["date"] = pd.to_datetime(eq_df["date"], errors="coerce")
        except Exception:
            pass
        try:
            idx = pd.DatetimeIndex(eq_df["date"].values)
        except Exception:
            idx = pd.to_datetime(eq_df["date"], errors="coerce")
            if isinstance(idx, pd.Series):
                idx = pd.DatetimeIndex(idx.values)
        if getattr(idx, "tz", None) is not None:
            try:
                idx = idx.tz_localize(None)
            except Exception:
                try:
                    idx = idx.tz_convert(None)
                except Exception:
                    pass
        daily = pd.Series(eq_df["daily_return"].values, index=idx)
    else:
        eq_df = pd.DataFrame(columns=["date", "daily_return", "cumulative_return", "drawdown"])

    derived = _compute_metrics_from_returns(daily)

    metrics = {
        "CAGR": summary.get("CAGR", derived.get("CAGR")),
        "Volatility": summary.get("Volatility", derived.get("Volatility")),
        "Sharpe": summary.get("Sharpe", derived.get("Sharpe")),
        "Max Drawdown": summary.get("Max Drawdown", derived.get("Max Drawdown")),
        "Sortino": derived.get("Sortino"),
        "Calmar": derived.get("Calmar"),
        "Total Return": summary.get("Total Return", derived.get("Total Return")),
        "Win Rate": summary.get("Win Rate"),
        "n_days": derived.get("n_days"),
    }

    # Validation flags + hybrid-specific warnings
    warnings: List[str] = list(_hybrid_extra_warnings)
    n_years = (metrics.get("n_days", 0) / ANNUALIZE) if metrics.get("n_days") else 0
    if n_years < 1:
        warnings.append("duration_lt_1y")
    if metrics.get("Sharpe", 0) > 3:
        warnings.append("sharpe_unrealistic")
    if abs(metrics.get("Max Drawdown", 0)) < 0.05:
        warnings.append("maxdd_unrealistic")

    # Compute stability score from equity curve
    try:
        stab = compute_rolling_stability(eq_df)
    except Exception:
        stab = {"stability_score": 0.0, "grade": "N/A", "components": {}, "summary": {}}

    out = {
        "equity_curve": eq_df,
        "metrics": metrics,
        "warnings": warnings,
        "stability": stab,
        "avg_pairwise_corr": summary.get("avg_pairwise_corr", 0.5),
    }
    return p.name, key, out


def backtest_user_portfolios(portfolios: List[UserPortfolio], start_date: date, end_date: date, use_multiprocessing: bool = True) -> Dict[str, Dict[str, Any]]:
    """Backtest up to `MAX_PORTFOLIOS` user portfolios and return their equity curves and metrics.

    Returns a mapping:
        { portfolio_name: {"equity_curve": DataFrame, "metrics": {...}} }

    Caches results in-process to avoid re-running identical backtests.
    """
    if not portfolios:
        return {}
    if len(portfolios) > MAX_PORTFOLIOS:
        raise ValueError(f"At most {MAX_PORTFOLIOS} portfolios allowed")

    results: Dict[str, Dict[str, Any]] = {}
    to_compute = []
    
    # We use a module-level lock for cache mutations
    if not hasattr(backtest_user_portfolios, '_cache_lock'):
        import threading
        backtest_user_portfolios._cache_lock = threading.Lock()
        
    with backtest_user_portfolios._cache_lock:
        for p in portfolios:
            key = _cache_key_for_portfolio(p, start_date, end_date)
            if key in _BACKTEST_CACHE:
                results[p.name] = _BACKTEST_CACHE[key]
            else:
                to_compute.append((p, key))

    if to_compute:
        if use_multiprocessing:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PORTFOLIOS)
            try:
                futures = {executor.submit(_worker_run_portfolio, p, start_date, end_date, key): p for p, key in to_compute}
                for future in concurrent.futures.as_completed(futures):
                    p_name, key, out = future.result()
                    results[p_name] = out
                    with backtest_user_portfolios._cache_lock:
                        _BACKTEST_CACHE[key] = out
            except (KeyboardInterrupt, SystemExit, Exception) as e:
                executor.shutdown(wait=False, cancel_futures=True)
                raise e
            finally:
                executor.shutdown(wait=False)
        else:
            for p, key in to_compute:
                p_name, key, out = _worker_run_portfolio(p, start_date, end_date, key)
                results[p_name] = out
                with backtest_user_portfolios._cache_lock:
                    _BACKTEST_CACHE[key] = out

    # Compute Risk Responsiveness Component (RRC) contextually
    series_map = {}
    for name, out in results.items():
        eq = out.get("equity_curve")
        if eq is not None and not eq.empty and "daily_return" in eq.columns:
            try:
                idx = pd.DatetimeIndex(pd.to_datetime(eq["date"], errors="coerce"))
                daily = pd.Series(eq["daily_return"].values, index=idx).fillna(0.0)
                series_map[name] = daily
            except Exception:
                pass
                
    if series_map:
        returns_df = pd.concat(series_map.values(), axis=1)
        returns_df.columns = list(series_map.keys())
        price_df_portfolios = (1 + returns_df).cumprod()
        
        from services.risk_responsiveness import compute_portfolio_rrc
        for name, out in results.items():
            if name in price_df_portfolios.columns:
                rrc_data = compute_portfolio_rrc(price_df_portfolios[name], returns_df)
                out["rrc_score"] = rrc_data.get("rrc_score", 50.0)
                out["rrc_components"] = rrc_data.get("components", {})
            else:
                out["rrc_score"] = 50.0
                out["rrc_components"] = {}

    return results


def rate_portfolio(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Rate a single portfolio given normalized metric values.

    This function expects metrics already normalized (0-1) under keys
    `Sharpe_norm`, `CAGR_norm`, `MaxDrawdown_norm`, `Calmar_norm`.

    If those keys are missing, it will raise ValueError. Returns a dict with
    `rating_score` and placeholder `grade` (grading across peers requires
    `rate_portfolios`).
    """
    if not all(k in metrics for k in ("Sharpe_norm", "CAGR_norm", "MaxDrawdown_norm", "Calmar_norm")):
        raise ValueError("rate_portfolio expects normalized keys: Sharpe_norm, CAGR_norm, MaxDrawdown_norm, Calmar_norm")

    score = (
        0.35 * float(metrics["Sharpe_norm"]) +
        0.25 * float(metrics["CAGR_norm"]) +
        0.20 * (1.0 - float(metrics["MaxDrawdown_norm"])) +
        0.20 * float(metrics["Calmar_norm"]).__float__()
    )
    return {"rating_score": float(score), "grade": ""}


def rate_portfolios(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Rate all portfolios using the composite 5-component rating system.

    Now powered by portfolio_rating.compute_composite_portfolio_rating which
    incorporates Sharpe (30%), Drawdown (20%), Stability (20%), Regime (15%)
    and Diversification (15%) — all cross-sectionally normalised.

    Returns mapping:
        { portfolio_name: {
            "rating_score": float (0-1, legacy compatibility),
            "composite_score": float (0-100),
            "grade": str (A+/A/B/C/D),
            "rank": int,
            "stability_score": float,
            "stability_grade": str,
            "stability_components": dict,
            "normalized": dict,
        } }
    """
    names = list(results.keys())
    if not names:
        return {}

    rated_df = _compute_composite_from_results(results)
    if rated_df is None or rated_df.empty:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in rated_df.iterrows():
        name = row["name"]
        stab = results.get(name, {}).get("stability", {})
        out[name] = {
            # Legacy key (0-1 scale for backward compat)
            "rating_score": float(row["composite_score"]) / 100.0,
            # New composite system
            "composite_score":  float(row["composite_score"]),
            "grade":            row["grade"],
            "rank":             int(row["rank"]),
            "grade_colour":     grade_colour(row["grade"]),
            # Stability passthrough
            "stability_score":       float(stab.get("stability_score", 50.0)),
            "stability_grade":       stab.get("grade", "N/A"),
            "stability_components":  stab.get("components", {}),
            # Component breakdown
            "components": {
                "sharpe_component":    float(row.get("sharpe_component", 0)),
                "drawdown_component":  float(row.get("drawdown_component", 0)),
                "stability_component": float(row.get("stability_component", 0)),
                "regime_component":    float(row.get("regime_component", 0)),
                "diversif_component":  float(row.get("diversif_component", 0)),
                "rrc_component":       float(row.get("rrc_component", 0)),
            },
            "normalized": {
                "Sharpe_norm":      float(row.get("sharpe_component", 0)) / 100.0,
                "MaxDrawdown_norm": 1.0 - float(row.get("drawdown_component", 0)) / 100.0,
                "Stability_norm":   float(row.get("stability_component", 0)) / 100.0,
                "Regime_norm":      float(row.get("regime_component", 0)) / 100.0,
                "Diversif_norm":    float(row.get("diversif_component", 0)) / 100.0,
            },
        }
    return out


def _compute_composite_from_results(
    results: Dict[str, Dict[str, Any]],
    extra_rows: Optional[List[Dict[str, Any]]] = None,
) -> Optional[pd.DataFrame]:
    """Internal: build rating input DF and run composite rating."""
    try:
        input_df = build_rating_input_from_results(results, extra_rows=extra_rows)
        if input_df.empty:
            return None
        return compute_composite_portfolio_rating(input_df)
    except Exception as _exc:
        import logging
        logging.getLogger(__name__).warning("Composite rating failed: %s", _exc)
        return None


def compute_full_ratings(
    results: Dict[str, Dict[str, Any]],
    meta_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Unified rating across all portfolios including the meta-portfolio.

    Parameters
    ----------
    results     : Output from backtest_user_portfolios.
    meta_result : Optional dict from construct_meta_portfolio
                  (requires 'meta_metrics' and 'meta_equity_curve' keys).

    Returns
    -------
    dict with:
        ratings:      { name: rating_dict }  — same shape as rate_portfolios()
        top_name:     str — name of #1-ranked portfolio
        rated_df:     pd.DataFrame — full sorted rating table
        meta_rating:  dict | None — rating for the meta-portfolio (if provided)
    """
    # Build extra row for meta-portfolio if provided
    extra_rows: List[Dict[str, Any]] = []
    _meta_name = "🔗 Meta (Blended)"
    if meta_result:
        meta_m = meta_result.get("meta_metrics", {})
        meta_eq = meta_result.get("meta_equity_curve", pd.DataFrame())
        try:
            meta_stab = compute_rolling_stability(meta_eq) if not meta_eq.empty else {}
        except Exception:
            meta_stab = {}
        meta_stab_summ = meta_stab.get("summary", {})
        extra_rows.append({
            "name":              _meta_name,
            "sharpe":            float(meta_m.get("Sharpe", 0.0)),
            "cagr":              float(meta_m.get("CAGR", 0.0)),
            "max_drawdown":      float(meta_m.get("Max Drawdown", -0.5)),
            "volatility":        float(meta_m.get("Volatility", 0.0)),
            "calmar":            float(meta_m.get("Calmar", 0.0)),
            "stability_score":   float(meta_stab.get("stability_score", 50.0)),
            "stability_grade":   meta_stab.get("grade", "N/A"),
            "regime_sharpe_gap": float(meta_stab_summ.get("regime_sharpe_gap", 1.0)),
            "avg_pairwise_corr": 0.5,   # meta portfolio correlation not trivially known
            "rrc_score":         float(meta_result.get("meta_rrc", 50.0)),
            "warnings":          [],
        })
        # Store meta stability back for use in result
        if meta_result is not None:
            meta_result["stability"] = meta_stab

    rated_df = _compute_composite_from_results(results, extra_rows=extra_rows or None)
    if rated_df is None or rated_df.empty:
        # Fall back to legacy rating
        return {"ratings": rate_portfolios(results), "top_name": None,
                "rated_df": pd.DataFrame(), "meta_rating": None}

    ratings = rate_portfolios(results)   # already computed but keep for compat

    # Pull meta rating out of the unified rated_df
    meta_row = rated_df[rated_df["name"] == _meta_name]
    meta_rating_dict: Optional[Dict[str, Any]] = None
    if not meta_row.empty:
        r = meta_row.iloc[0]
        meta_stab_ref = {}
        if meta_result:
            meta_stab_ref = meta_result.get("stability", {})
        meta_rating_dict = {
            "composite_score": float(r["composite_score"]),
            "grade":           r["grade"],
            "rank":            int(r["rank"]),
            "grade_colour":    grade_colour(r["grade"]),
            "stability_score": float(meta_stab_ref.get("stability_score", 50.0)),
            "stability_grade": meta_stab_ref.get("grade", "N/A"),
        }

    return {
        "ratings":     ratings,
        "top_name":    _top_portfolio(rated_df[rated_df["name"] != _meta_name]),
        "rated_df":    rated_df,
        "meta_rating": meta_rating_dict,
    }


def compute_portfolio_correlation(results: Dict[str, Dict[str, Any]]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Compute pairwise correlation and covariance matrices of daily returns for provided portfolios.

    Returns (correlation_df, covariance_df) or (None, None) if insufficient data.
    """
    frames = {}
    for name, v in results.items():
        df = v.get("equity_curve")
        if df is None or df.empty or "daily_return" not in df.columns:
            continue
        # Robust datetime index creation
        idx = pd.to_datetime(df["date"], errors="coerce")
        if isinstance(idx, pd.Series):
            idx = pd.DatetimeIndex(idx.values)
        if getattr(idx, "tz", None) is not None:
            try:
                idx = idx.tz_localize(None)
            except Exception:
                try:
                    idx = idx.tz_convert(None)
                except Exception:
                    pass
        ser = pd.Series(df["daily_return"].values, index=idx)
        ser.name = name
        frames[name] = ser

    if not frames:
        return None, None

    returns_df = pd.concat(frames.values(), axis=1)
    returns_df.columns = list(frames.keys())

    # pandas corr/cov uses pairwise complete observations
    corr = returns_df.corr()
    cov = returns_df.cov()
    return corr, cov


def construct_meta_portfolio(
    results: Dict[str, Dict[str, Any]],
    portfolios: Optional[List[UserPortfolio]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    user_ratings: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Construct a meta-portfolio by treating each user portfolio as an asset.

    Uses mean-variance optimization to maximize portfolio-level Sharpe subject to:
      - sum(weights) == 1
      - weights >= 0
      - weights_i <= 0.6 (unless infeasible for small N)

    Returns:
        {"weights": {name: weight}, "meta_equity_curve": df, "meta_metrics": {...}}
    """
    corr, cov = compute_portfolio_correlation(results)
    # Build returns DataFrame
    series_map: Dict[str, pd.Series] = {}
    for name, v in results.items():
        df = v.get("equity_curve")
        if df is None or df.empty or "daily_return" not in df.columns:
            continue
        idx = pd.to_datetime(df["date"], errors="coerce")
        if isinstance(idx, pd.Series):
            idx = pd.DatetimeIndex(idx.values)
        if getattr(idx, "tz", None) is not None:
            try:
                idx = idx.tz_localize(None)
            except Exception:
                try:
                    idx = idx.tz_convert(None)
                except Exception:
                    pass
        ser = pd.Series(df["daily_return"].values, index=idx)
        ser.name = name
        series_map[name] = ser

    if not series_map:
        return {"weights": {}, "meta_equity_curve": pd.DataFrame(), "meta_metrics": {}}

    returns_df = pd.concat(series_map.values(), axis=1)
    returns_df.columns = list(series_map.keys())

    # Robust covariance (Ledoit-Wolf shrinkage) and Bayesian-shrunk expected returns
    # for stable mean-variance optimisation. Raw sample cov is ill-conditioned
    # when T (trading days) is small relative to the number of portfolios.
    mu = shrunk_annualised_returns(returns_df, annualize=ANNUALIZE)
    cov_result = robust_covariance_matrix(
        returns_df, method="ledoit_wolf", annualize=ANNUALIZE, log_diagnostics=True
    )
    Sigma_arr = cov_result["matrix"]   # np.ndarray, PD-guaranteed

    names = list(returns_df.columns)
    n = len(names)

    # ── Stop-Loss Scoring & RRC Layer Integration ────────────────────────────
    try:
        from services.stop_loss_engine import compute_stop_loss_scores
        from services.risk_responsiveness import compute_portfolio_rrc
        
        # price_df format requires 1 + returns cumprod
        price_df_portfolios = (1 + returns_df).cumprod()
        sls_df = compute_stop_loss_scores(price_df_portfolios)
        
        sls_scores = np.array([
            sls_df.loc[name, "stop_loss_score"] if (not sls_df.empty and name in sls_df.index) else 50.0
            for name in names
        ], dtype=float)
        
        rrc_list = []
        for name in names:
            rrc_data = compute_portfolio_rrc(price_df_portfolios[name], returns_df)
            rrc_list.append(rrc_data["rrc_score"])
        rrc_portfolios = np.array(rrc_list, dtype=float)
    except Exception:
        sls_scores = np.array([50.0] * n)
        rrc_portfolios = np.array([50.0] * n)

    # Feasibility check for per-asset cap
    max_weight_cap = 0.6 if n * 0.6 >= 1.0 else 1.0

    try:
        import scipy.optimize as sco

        def neg_sharpe(w: np.ndarray) -> float:
            w = np.array(w)
            ret = float(np.dot(w, mu.reindex(names).fillna(0.0).values))
            vol = float(np.sqrt(w.dot(Sigma_arr).dot(w)))
            if vol <= 0:
                return 1e6
            
            sharpe = ret / vol
            
            # Penalty layer: adjusted sharpe = sharpe - 0.2 * sum(w_i * (SLS_i / 100)^2)
            # Convex amplification punishes high-risk names disproportionately
            convex_sls = float(np.dot(w, (sls_scores / 100.0) ** 2))
            penalty = 0.2 * convex_sls
            
            # Risk Responsiveness Component bonus
            rrc_bonus = 0.1 * float(np.dot(w, rrc_portfolios) / 100.0)
            
            return -(sharpe - penalty + rrc_bonus)

        bounds = tuple((0.0, float(max_weight_cap)) for _ in range(n))
        cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
        w0 = np.array([1.0 / n] * n)

        res = sco.minimize(neg_sharpe, w0, bounds=bounds, constraints=cons, method="SLSQP", options={"maxiter": 500})
        if res.success:
            w_opt = np.array(res.x)
        else:
            # fallback to heuristic
            raise RuntimeError("optimizer failed")
    except Exception:
        # Heuristic: weight proportional to positive Sharpe (or mean return if Sharpe not informative)
        sharpes = np.array([results.get(nm, {}).get("metrics", {}).get("Sharpe", 0.0) for nm in names], dtype=float)
        positive = np.clip(sharpes, a_min=0.0, a_max=None)
        if positive.sum() > 0:
            w_raw = positive / positive.sum()
        else:
            # fallback equal weight
            w_raw = np.array([1.0 / n] * n)

        # Clip to cap and renormalize
        w_clipped = np.minimum(w_raw, max_weight_cap)
        if w_clipped.sum() == 0:
            w_opt = np.array([1.0 / n] * n)
        else:
            w_opt = w_clipped / w_clipped.sum()

    weights = {name: float(w_opt[i]) for i, name in enumerate(names)}

    # Meta daily returns (pairwise available dates -> pandas will align)
    meta_daily = returns_df.dot(pd.Series(w_opt, index=names))

    # Build equity curve df similar to run_backtest
    cum = (1 + meta_daily).cumprod()
    cumulative_return = cum - 1.0
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max

    eq_df = pd.DataFrame({
        "date": pd.to_datetime(meta_daily.index).date,
        "daily_return": meta_daily.values,
        "cumulative_return": cumulative_return.values,
        "drawdown": drawdown.values,
    })

    meta_metrics = _compute_metrics_from_returns(meta_daily)
    
    meta_sls = float(np.dot(w_opt, sls_scores)) if 'sls_scores' in locals() else 0.0
    
    # Compute RRC for Meta-Portfolio
    try:
        from services.risk_responsiveness import compute_portfolio_rrc
        meta_rrc_result = compute_portfolio_rrc(cum, returns_df)
        meta_rrc = float(meta_rrc_result.get("rrc_score", 50.0))
        meta_rrc_components = meta_rrc_result.get("components", {})
    except Exception:
        meta_rrc = 50.0
        meta_rrc_components = {}

    out = {
        "weights": weights, 
        "meta_equity_curve": eq_df, 
        "meta_metrics": meta_metrics,
        "meta_stop_loss_score": meta_sls,
        "meta_rrc": meta_rrc,
        "meta_rrc_components": meta_rrc_components
    }

    # Compute best user rating (if needed) using provided ratings or by computing from results
    try:
        if user_ratings is None:
            user_ratings = rate_portfolios(results)
        # Find best user-defined portfolio
        best_user_name = max(user_ratings.items(), key=lambda kv: float(kv[1].get("rating_score", -1)))[0]
        best_user_score = float(user_ratings[best_user_name].get("rating_score", 0.0))
    except Exception:
        best_user_name = None
        best_user_score = 0.0

    # Compute a meta rating for the portfolio-level meta (normalized across users)
    try:
        # Build user metric arrays consistent with `rate_portfolios`
        sharpe_u = np.array([results[n]["metrics"].get("Sharpe", 0.0) for n in names], dtype=float)
        cagr_u = np.array([results[n]["metrics"].get("CAGR", 0.0) for n in names], dtype=float)
        maxdd_u = np.array([abs(results[n]["metrics"].get("Max Drawdown", 0.0)) for n in names], dtype=float)
        calmar_u = np.array([results[n]["metrics"].get("Calmar", 0.0) for n in names], dtype=float)

        def _minmax_scalar(val: float, arr: np.ndarray) -> float:
            if np.nanmax(arr) == np.nanmin(arr):
                return 0.5
            return float((val - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr)))

        meta_sharpe_n = _minmax_scalar(meta_metrics.get("Sharpe", 0.0), sharpe_u)
        meta_cagr_n = _minmax_scalar(meta_metrics.get("CAGR", 0.0), cagr_u)
        meta_maxdd_n = _minmax_scalar(abs(meta_metrics.get("Max Drawdown", 0.0)), maxdd_u)
        meta_calmar_n = _minmax_scalar(meta_metrics.get("Calmar", 0.0), calmar_u)

        meta_rating_score = 0.35 * meta_sharpe_n + 0.25 * meta_cagr_n + 0.20 * (1.0 - meta_maxdd_n) + 0.20 * meta_calmar_n
    except Exception:
        meta_rating_score = float(out.get("meta_metrics", {}).get("Sharpe", 0.0))

    out["meta_portfolio_rating"] = float(meta_rating_score)

    # If portfolios + date range provided, build a stock-level blended portfolio and backtest it
    if portfolios and start_date is not None and end_date is not None:
        # Map portfolio name -> UserPortfolio
        pmap: Dict[str, UserPortfolio] = {p.name: p for p in portfolios}

        # Latest scoring date to build per-portfolio allocations (static snapshot)
        as_of = get_latest_scoring_date()

        stock_w_raw: Dict[int, float] = {}
        if as_of is not None:
            for pname in names:
                pdef = pmap.get(pname)
                if pdef is None:
                    continue

                try:
                    if pdef.strategy in ("equal_weight", "equal_weight_top_n"):
                        alloc_df = construct_equal_weight_portfolio(as_of, top_n=pdef.top_n, selected_symbols=pdef.symbols)
                    else:
                        alloc_df = construct_inverse_vol_portfolio(as_of, top_n=pdef.top_n, selected_symbols=pdef.symbols)
                except Exception:
                    alloc_df = pd.DataFrame(columns=["stock_id", "weight"])

                if alloc_df is None or alloc_df.empty:
                    continue

                alloc_map = dict(zip(alloc_df["stock_id"].astype(int), alloc_df["weight"].astype(float)))
                port_w = float(weights.get(pname, 0.0))
                for sid, sw in alloc_map.items():
                    stock_w_raw[sid] = stock_w_raw.get(sid, 0.0) + port_w * float(sw)

        if not stock_w_raw:
            out["stock_weights"] = {}
            out["stock_equity_curve"] = pd.DataFrame()
            out["stock_metrics"] = {}
            out["winner"] = {
                "winner_type": "meta" if meta_rating_score > best_user_score else "user",
                "best_user_name": best_user_name,
                "best_user_score": best_user_score,
                "meta_score": meta_rating_score,
            }
            return out

        # Clip per-stock cap and renormalize
        sids = list(stock_w_raw.keys())
        w_arr = np.array([stock_w_raw[sid] for sid in sids], dtype=float)
        max_stock_cap = 0.6
        w_clipped = np.minimum(w_arr, max_stock_cap)
        if w_clipped.sum() <= 0:
            w_norm = np.array([1.0 / len(w_clipped)] * len(w_clipped), dtype=float)
        else:
            w_norm = w_clipped / float(w_clipped.sum())

        final_stock_weights: Dict[int, float] = {int(sid): float(w_norm[i]) for i, sid in enumerate(sids)}

        # Compute portfolio daily returns for the blended stock-level allocation
        try:
            stock_daily = compute_period_returns(start_date, end_date, final_stock_weights)
        except Exception:
            stock_daily = pd.Series(dtype=float)

        if stock_daily is None:
            stock_daily = pd.Series(dtype=float)

        # Build equity curve for the stock-level blended portfolio
        if not stock_daily.empty:
            cum_s = (1 + stock_daily).cumprod()
            cumulative_return_s = cum_s - 1.0
            running_max_s = cum_s.cummax()
            drawdown_s = (cum_s - running_max_s) / running_max_s
            try:
                date_vals = pd.to_datetime(stock_daily.index).date
            except Exception:
                date_vals = pd.to_datetime(stock_daily.index)
            stock_eq_df = pd.DataFrame({
                "date": date_vals,
                "daily_return": stock_daily.values,
                "cumulative_return": cumulative_return_s.values,
                "drawdown": drawdown_s.values,
            })
        else:
            stock_eq_df = pd.DataFrame(columns=["date", "daily_return", "cumulative_return", "drawdown"])

        stock_metrics = _compute_metrics_from_returns(stock_daily)

        # Compute stock-level meta rating normalized against user portfolios
        try:
            meta_stock_sharpe_n = _minmax_scalar(stock_metrics.get("Sharpe", 0.0), sharpe_u)
            meta_stock_cagr_n = _minmax_scalar(stock_metrics.get("CAGR", 0.0), cagr_u)
            meta_stock_maxdd_n = _minmax_scalar(abs(stock_metrics.get("Max Drawdown", 0.0)), maxdd_u)
            meta_stock_calmar_n = _minmax_scalar(stock_metrics.get("Calmar", 0.0), calmar_u)
            meta_stock_score = 0.35 * meta_stock_sharpe_n + 0.25 * meta_stock_cagr_n + 0.20 * (1.0 - meta_stock_maxdd_n) + 0.20 * meta_stock_calmar_n
        except Exception:
            meta_stock_score = float(stock_metrics.get("Sharpe", 0.0))

        # Resolve stock ids -> symbols for display
        with get_db_context() as db:
            rows = db.execute(select(Stock.id, Stock.symbol).where(Stock.id.in_(list(final_stock_weights.keys())))).all()
        id_to_symbol = {r[0]: r[1] for r in rows} if rows else {}
        stock_symbol_weights = {id_to_symbol.get(sid, str(sid)): w for sid, w in final_stock_weights.items()}

        out["stock_weights"] = stock_symbol_weights
        out["stock_equity_curve"] = stock_eq_df
        out["stock_metrics"] = stock_metrics
        out["meta_stock_rating"] = float(meta_stock_score)

        out["winner"] = {
            "winner_type": "meta" if meta_stock_score > best_user_score else "user",
            "best_user_name": best_user_name,
            "best_user_score": best_user_score,
            "meta_score": meta_stock_score,
        }

        return out

    # If no stock-level blending requested, include winner info comparing portfolio-level meta
    out["winner"] = {
        "winner_type": "meta" if meta_rating_score > best_user_score else "user",
        "best_user_name": best_user_name,
        "best_user_score": best_user_score,
        "meta_score": meta_rating_score,
    }

    return out


__all__ = [
    "UserPortfolio",
    "backtest_user_portfolios",
    "rate_portfolios",
    "rate_portfolio",
    "compute_portfolio_correlation",
    "construct_meta_portfolio",
]
