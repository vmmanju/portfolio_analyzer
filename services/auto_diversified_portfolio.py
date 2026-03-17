"""services/auto_diversified_portfolio.py

Diversification-Aware Hybrid Auto Portfolio Builder.

Builds a monthly-rebalanced, correlation-filtered, volatility-aware portfolio
from the scored universe. Integrates as a plug-in UserPortfolio type
into the Portfolio Comparison module.

NO LOOK-AHEAD BIAS — at every rebalance date only data on or before that date
is consumed:
  - Scores   : latest scoring_date <= rebalance_date
  - Prices   : window ending on rebalance_date (NO future close prices)
  - Vol-target: lagged (t-1) rolling vol — standard look-behind pattern
"""

import json
import logging
import math
import sys
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sqlalchemy import select

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database import get_db_context
from app.models import Price
from services.backtest import (
    LEVERAGE_MAX,
    LEVERAGE_MIN,
    ROLLING_VOL_WINDOW,
    _apply_volatility_targeting,
    calculate_transaction_cost,
    compute_period_returns,
    get_latest_score_date_on_or_before,
    get_rebalance_dates,
)
from services.portfolio import load_ranked_stocks
from services.return_estimator import shrunk_annualised_returns, corrected_returns
from services.covariance_estimator import robust_covariance_matrix

# ── Public constant (used as strategy identifier in UserPortfolio.strategy) ──
STRATEGY_AUTO_HYBRID = "auto_diversified_hybrid"

# ── Tuning knobs ─────────────────────────────────────────────────────────────
CANDIDATE_POOL_SIZE: int = 50        # candidates fetched before corr-filter (was 30)
CORR_LOOKBACK_DAYS: int = 90         # trading days for correlation/vol window
TARGET_ANNUAL_VOL: float = 0.15      # vol-targeting target
ANNUALIZE: int = 252
SCORE_TILT_FACTOR: float = 0.25      # how much composite score tilts weights

# ── Persistence paths ─────────────────────────────────────────────────────────
_STATE_DIR = _project_root / "data"

def _get_rebalance_state_file(user_id: Optional[int] = None) -> Path:
    if user_id is not None:
        return _STATE_DIR / f"hybrid_rebalance_state_user_{user_id}.json"
    return _STATE_DIR / "hybrid_rebalance_state.json"

_LOG_DIR = _project_root / "logs"
_LOG_FILE = _LOG_DIR / "hybrid_rebalance.log"
_BUILD_CACHE_FILE = _STATE_DIR / "hybrid_build_cache.json"

# ── Logger (writes to file + propagates to root) ──────────────────────────────
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_logger = logging.getLogger("hybrid_scheduler")
if not _logger.handlers:
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    _logger.addHandler(_fh)
    _logger.setLevel(logging.INFO)

# ── In-memory caches (process-lifetime; keyed by immutable tuples) ─────────
_RETURNS_CACHE: Dict[tuple, pd.DataFrame] = {}
_BUILD_CACHE: Dict[str, Dict[str, Any]] = {}
_RRC_SCORE_CACHE: Dict[tuple, pd.Series] = {} # Key: (date, tuple(stock_ids))
_build_cache_lock = threading.RLock()

def _load_build_cache() -> None:
    global _BUILD_CACHE
    try:
        if _BUILD_CACHE_FILE.exists():
            with _build_cache_lock:
                data = json.loads(_BUILD_CACHE_FILE.read_text(encoding="utf-8"))
                # Restore integer keys for weights (JSON converts keys to strings)
                restored_cache = {}
                for k, v in data.items():
                    if isinstance(v, dict) and "weights" in v:
                        v["weights"] = {int(sid): weight for sid, weight in v["weights"].items()}
                    restored_cache[k] = v
                _BUILD_CACHE = restored_cache
    except Exception as e:
        _logger.debug("Could not load build cache: %s", e)

def _save_build_cache() -> None:
    try:
        with _build_cache_lock:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            # Filter out heavy internal matrices to keep JSON disk-cache clean and small
            # These are still kept in memory (_BUILD_CACHE) for the current process
            serializable_cache = {}
            for k, v in _BUILD_CACHE.items():
                if isinstance(v, dict):
                    clean_v = {ck: cv for ck, cv in v.items() if not ck.startswith("_internal_")}
                    serializable_cache[k] = clean_v
                else:
                    serializable_cache[k] = v
                    
            _BUILD_CACHE_FILE.write_text(json.dumps(serializable_cache, default=str, indent=2), encoding="utf-8")
    except Exception as e:
        _logger.debug("Could not save build cache: %s", e)

# Load cache on module import
_load_build_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_returns_up_to(
    stock_ids: List[int],
    as_of_date: date,
    lookback_days: int = CORR_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Daily return matrix ending on (and including) as_of_date.

    NO LOOK-AHEAD: Price.date <= as_of_date enforced in the query.
    Returns DataFrame[date_index, stock_id_columns].  May be empty.
    """
    if not stock_ids:
        return pd.DataFrame()

    ids_set = set(stock_ids)
    
    # ── 1. Fast path: exact match cache ──────────────────────────────────────
    cache_key = (tuple(sorted(stock_ids)), as_of_date, lookback_days)
    if cache_key in _RETURNS_CACHE:
        return _RETURNS_CACHE[cache_key]

    # ── 2. Smart path: check for superset in cache ───────────────────────────
    # If we have a cached result for the same date with more stocks/history, just slice it.
    for (c_ids, c_date, c_lookback), c_df in _RETURNS_CACHE.items():
        if c_date == as_of_date and c_lookback >= lookback_days:
            c_id_set = set(c_ids)
            if ids_set.issubset(c_id_set):
                # Slice by stocks and then by time
                try:
                    sliced = c_df[stock_ids].iloc[-lookback_days:]
                    # Also save this sub-key for even faster lookup next time
                    _RETURNS_CACHE[cache_key] = sliced
                    return sliced
                except KeyError:
                    continue # Some stocks missing despite set logic? shouldn't happen but safe-fail

    # ── 3. DB fetch path ─────────────────────────────────────────────────────
    # Widen to calendar days to account for weekends + holidays
    calendar_start = as_of_date - timedelta(days=int(lookback_days * 1.6))

    with get_db_context() as db:
        rows = db.execute(
            select(Price.date, Price.stock_id, Price.close)
            .where(
                Price.stock_id.in_(stock_ids),
                Price.date >= calendar_start,
                Price.date <= as_of_date,          # ← no look-ahead
            )
            .order_by(Price.date)
        ).all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "stock_id", "close"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    wide = df.pivot(index="date", columns="stock_id", values="close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    returns = wide.pct_change().dropna(how="all")
    # Keep only the trailing lookback_days trading rows
    if len(returns) > lookback_days:
        returns = returns.iloc[-lookback_days:]

    # Limit cache size to avoid runaway memory
    if len(_RETURNS_CACHE) > 200:
        _RETURNS_CACHE.clear()
    _RETURNS_CACHE[cache_key] = returns
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_monthly_rebalance_dates(start_date: date, end_date: date) -> List[date]:
    """Month-end trading dates in [start_date, end_date] that have scoring data.

    Wrapper around the backtest engine's `get_rebalance_dates`.
    Portfolio rebuilds happen ONLY on these dates — not daily.
    """
    return get_rebalance_dates(start_date=start_date, end_date=end_date)


def get_month_end_trading_days(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[date]:
    """Return all month-end trading days available in the price table.

    start_date / end_date are inclusive filters (None = no bound).
    Each returned date is the last price-table entry for its calendar month,
    provided at least one score exists on or before that date.

    This is the canonical source for "when should the hybrid portfolio rebalance?"
    """
    return get_rebalance_dates(start_date=start_date, end_date=end_date)


# ── Rebalance-state persistence helpers ──────────────────────────────────────

import threading
_state_file_lock = threading.RLock()

def _load_rebalance_state(user_id: Optional[int] = None) -> Dict[str, Any]:
    """Load the on-disk rebalance state (JSON).  Returns empty dict on any error."""
    try:
        with _state_file_lock:
            state_file = _get_rebalance_state_file(user_id)
            if state_file.exists():
                return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_rebalance_state(state: Dict[str, Any], user_id: Optional[int] = None) -> None:
    """Persist current rebalance state to JSON.  Silently ignores write errors."""
    try:
        with _state_file_lock:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            state_file = _get_rebalance_state_file(user_id)
            state_file.write_text(json.dumps(state, default=str, indent=2), encoding="utf-8")
    except Exception as exc:
        _logger.warning("Could not save rebalance state: %s", exc)


def build_diversified_hybrid_portfolio(
    as_of_date: date,
    top_n: Union[int, str] = 20,
    min_score: float = 0.0,
    max_corr: float = 0.80,
    regime_mode: str = "volatility_targeting",
    risk_profile: str = "medium",
    strategy: str = "composite",
    use_sls: bool = True,
    _skip_logging: bool = False,
    precomputed_scores: Optional[Dict[str, Any]] = None,
    force_n_recalc: bool = False,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a diversification-aware hybrid portfolio as of `as_of_date`.

    Steps
    -----
    1. Candidate pre-selection — top CANDIDATE_POOL_SIZE scored stocks
       (score_date <= as_of_date; no look-ahead).
    2. Greedy correlation filtering — add stocks only if pairwise corr
       with already-selected stocks is below max_corr.
    3. Hybrid weighting — inverse-volatility base weights tilted by
       normalised composite score (SCORE_TILT_FACTOR).
    4. Portfolio statistics from the lookback window.

    Returns dict with keys:
        selected_stocks, weights, expected_volatility, expected_sharpe,
        rebalance_date, warnings, avg_pairwise_corr.
    """
    import time
    t_start = time.time()
    
    # ── 0. Result Cache Lookups ──────────────────────────────────────────────
    # Key must be hashable and JSON-serializable (as string)
    params = (as_of_date.isoformat(), top_n, min_score, max_corr, regime_mode, risk_profile, strategy, use_sls)
    cache_key = ":".join(map(str, params))
    
    if cache_key in _BUILD_CACHE:
        return _BUILD_CACHE[cache_key]
        
    result: Dict[str, Any] = {
        "selected_stocks": [],
        "weights": {},
        "expected_volatility": 0.0,
        "expected_sharpe": 0.0,
        "rebalance_date": as_of_date,
        "warnings": [],
        "avg_pairwise_corr": 0.0,
        "optimal_n": None,
    }
    warnings: List[str] = []

    if isinstance(top_n, str) and top_n.lower() == "auto":
        from services.auto_n_selector import update_optimal_n_if_due
        n_res = update_optimal_n_if_due(
            as_of_date=as_of_date,
            min_score=min_score,
            max_corr=max_corr,
            regime_mode=regime_mode,
            risk_profile=risk_profile,
            force=force_n_recalc,
            user_id=user_id,
        )
        actual_n = n_res.get("optimal_n", 15)
        result["auto_n_evaluation"] = n_res
        if not _skip_logging:
            _logger.info("Automatic N Selected: %d for %s (source: %s)", actual_n, as_of_date, n_res.get('calibration_source'))
    else:
        actual_n = int(top_n)
        
    result["optimal_n"] = actual_n

    # ── Step 1: candidates (scores on or before as_of_date) ──────────────────
    score_date = get_latest_score_date_on_or_before(as_of_date)
    if score_date is None:
        result["warnings"] = ["no_scoring_data"]
        return result

    ranked = load_ranked_stocks(score_date)
    if ranked.empty:
        result["warnings"] = ["no_ranked_stocks"]
        return result

    candidates = ranked.copy()
    if min_score > 0.0:
        candidates = candidates[candidates["composite_score"] >= min_score]
        
    # ── RRC Integration (Candidate Filtering) ──
    from services.risk_responsiveness import compute_rrc_scores, get_historical_prices_rrc
    
    if precomputed_scores and "rrc" in precomputed_scores:
        rrc_scores = precomputed_scores["rrc"]
    else:
        # Check module-level RRC cache
        stock_ids_for_rrc = candidates["stock_id"].astype(int).tolist()
        rrc_cache_key = (as_of_date, tuple(sorted(stock_ids_for_rrc)))
        if rrc_cache_key in _RRC_SCORE_CACHE:
            rrc_scores = _RRC_SCORE_CACHE[rrc_cache_key]
        else:
            t0 = time.time()
            rrc_prices = get_historical_prices_rrc(as_of_date, stock_ids_for_rrc, lookback_years=5)
            rrc_scores = compute_rrc_scores(rrc_prices)
            _RRC_SCORE_CACHE[rrc_cache_key] = rrc_scores
            _logger.info("RRC compute time for %s: %.2f sec", as_of_date, time.time() - t0)
    
    adj_count = 0
    for idx, row in candidates.iterrows():
        sid = int(row["stock_id"])
        if sid in rrc_scores.index:
            rrc = rrc_scores[sid]
            score = row["composite_score"]
            if rrc < 30:
                candidates.at[idx, "composite_score"] = score * 0.90
                adj_count += 1
            elif rrc > 70:
                candidates.at[idx, "composite_score"] = score * 1.05
                adj_count += 1
                
    if adj_count > 0:
        warnings.append(f"Responsiveness adjustment applied to {adj_count} candidates.")
        _logger.info("Responsiveness adjustment applied to %d candidates on %s", adj_count, as_of_date)
        
    # Re-rank after adjustment to ensure pool sizes fetch correctly sorted entries
    candidates = candidates.sort_values("composite_score", ascending=False)
    candidates["rank"] = range(1, len(candidates) + 1)
    
    candidates = candidates.head(CANDIDATE_POOL_SIZE)
    if candidates.empty:
        result["warnings"] = ["no_candidates_after_filter"]
        return result

    stock_ids: List[int] = candidates["stock_id"].astype(int).tolist()

    # ── Step 2: greedy correlation filtering ─────────────────────────────────
    returns_df = _load_returns_up_to(stock_ids, as_of_date)

    if returns_df.empty or returns_df.shape[0] < 5:
        warnings.append("insufficient_price_history_for_correlation")
        selected_ids = stock_ids[:actual_n]
        corr_matrix = pd.DataFrame()
    else:
        valid_cols = [c for c in returns_df.columns if returns_df[c].notna().sum() >= 5]
        returns_df = returns_df[valid_cols]
        ids_with_data = [s for s in stock_ids if s in returns_df.columns]
        ids_no_data = [s for s in stock_ids if s not in returns_df.columns]

        corr_matrix = (
            returns_df[ids_with_data].corr()
            if len(ids_with_data) > 1
            else pd.DataFrame()
        )

        selected_ids: List[int] = []
        for sid in ids_with_data + ids_no_data:
            if len(selected_ids) >= actual_n:
                break
            if not selected_ids:
                selected_ids.append(sid)
                continue
            if not corr_matrix.empty and sid in corr_matrix.columns:
                already = [s for s in selected_ids if s in corr_matrix.columns]
                if already:
                    max_pair = corr_matrix.loc[sid, already].abs().max()
                    if max_pair >= max_corr:
                        continue   # too correlated — skip
            selected_ids.append(sid)

        if not selected_ids:
            selected_ids = stock_ids[:actual_n]

    # ── Step 3: hybrid weighting ──────────────────────────────────────────────
    sel_df = candidates[candidates["stock_id"].isin(selected_ids)].copy()
    sel_ids = [s for s in selected_ids if s in sel_df["stock_id"].values]
    if not sel_ids:
        result["warnings"] = warnings + ["no_stocks_survive_filter"]
        return result

    # Actual annualised volatility from price history (preferred)
    vol_map: Dict[int, float] = {}
    if not returns_df.empty:
        for sid in sel_ids:
            if sid in returns_df.columns:
                dv = float(returns_df[sid].dropna().std())
                vol_map[sid] = dv * math.sqrt(ANNUALIZE) if dv > 0 else None

    # Fallback: z-scored volatility_score proxy (offset to stay positive)
    for sid in sel_ids:
        if sid not in vol_map or not vol_map[sid]:
            row = sel_df[sel_df["stock_id"] == sid]
            if not row.empty and pd.notna(row["volatility_score"].iloc[0]):
                v = abs(float(row["volatility_score"].iloc[0]) + 3.0)
                vol_map[sid] = max(v, 1e-6)
            else:
                vol_map[sid] = 1.0

    # Inverse-vol base weights
    inv_vols = {sid: 1.0 / max(vol_map[sid], 1e-8) for sid in sel_ids}
    tot = sum(inv_vols.values())
    base_w = {sid: v / tot for sid, v in inv_vols.items()} if tot > 0 else {sid: 1.0 / len(sel_ids) for sid in sel_ids}

    # Score tilt
    scores_dict = sel_df.set_index("stock_id")["composite_score"].to_dict()
    sv = [scores_dict.get(sid, 0.0) for sid in sel_ids]
    s_min, s_max = min(sv), max(sv)
    norm_sc = (
        {sid: (scores_dict.get(sid, 0.0) - s_min) / (s_max - s_min) for sid in sel_ids}
        if s_max > s_min else {sid: 0.5 for sid in sel_ids}
    )
    tilted = {sid: base_w[sid] * (1.0 + SCORE_TILT_FACTOR * norm_sc[sid]) for sid in sel_ids}
    tot2 = sum(tilted.values())
    final_w = {sid: w / tot2 for sid, w in tilted.items()} if tot2 > 0 else base_w

    # ── Step 3b: Stop-Loss Scoring Layer ─────────────────────────────────────
    if use_sls:
        from services.stop_loss_engine import get_historical_prices, compute_stop_loss_scores
        
        # Local variables to help linter with Optional typing
        cached_sls = None
        if precomputed_scores is not None:
            cached_sls = precomputed_scores.get("sls")
            
        if cached_sls is not None:
            sls_scores = cached_sls
        else:
            t0 = time.time()
            price_df_sls = get_historical_prices(as_of_date, list(final_w.keys()))
            if not price_df_sls.empty:
                # Factor scores for bias
                _fcols = ["momentum_score", "quality_score", "value_score", "volatility_score"]
                _afcols = [c for c in _fcols if c in sel_df.columns]
                f_scores_df = sel_df.set_index("stock_id")[_afcols] if _afcols else pd.DataFrame()
                
                sls_scores = compute_stop_loss_scores(price_df_sls)
                _logger.info("SLS compute time for %s: %.2f sec", as_of_date, time.time() - t0)
            else:
                sls_scores = pd.DataFrame()
        
        if not sls_scores.empty:
            _sls_adj_cnt = 0
            for sid in list(final_w.keys()):
                if sid in sls_scores.index:
                    _sc = sls_scores.loc[sid, "stop_loss_score"]
                    if _sc > 75:
                        final_w[sid] = 0.0
                        _sls_adj_cnt += 1
                    elif _sc > 60:
                        final_w[sid] *= 0.5
                        _sls_adj_cnt += 1
    
            final_w = {sid: w for sid, w in final_w.items() if w > 0}
            tot_sls = sum(final_w.values())
            if tot_sls > 0:
                final_w = {sid: w / tot_sls for sid, w in final_w.items()}
                
            if _sls_adj_cnt > 0:
                warnings.append(f"Stop-loss adjustment applied to {_sls_adj_cnt} stocks.")
                _logger.info("Stop-loss adjustment applied to %d stocks on %s", _sls_adj_cnt, as_of_date)
                sel_ids = list(final_w.keys())
    # ── Step 4: portfolio statistics from lookback window ────────────────────
    exp_vol = exp_sharpe = avg_corr = 0.0
    if not returns_df.empty:
        sel_with_data = [s for s in sel_ids if s in returns_df.columns]
        if len(sel_with_data) >= 2:
            ret_sel = returns_df[sel_with_data].dropna(how="any")
            if len(ret_sel) >= 5:
                w_arr = np.array([final_w.get(s, 0.0) for s in sel_with_data])
                w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr
                # Robust (shrunk) covariance — more stable than raw sample cov
                # for short lookback windows (T ≈ 90 << p*(p+1)/2 parameters).
                cov_result = robust_covariance_matrix(
                    ret_sel, method="ledoit_wolf", annualize=ANNUALIZE, log_diagnostics=False
                )
                cov = cov_result["matrix"]
                pv = float(w_arr @ cov @ w_arr)
                exp_vol = math.sqrt(max(pv, 0.0))
                # Build factor-score DataFrame for error correction
                # Uses the same sel_df that already exists at this scope.
                _factor_cols = ["momentum_score", "quality_score", "value_score", "volatility_score"]
                _available_fcols = [c for c in _factor_cols if c in sel_df.columns]
                factor_scores_df = (
                    sel_df.set_index("stock_id")[_available_fcols]
                    if _available_fcols else pd.DataFrame()
                )

                # Corrected returns: Bayesian shrinkage → subtract systematic bias
                # Falls back to plain shrinkage if no calibration exists or
                # USE_ERROR_CORRECTION=False in config.
                shrunk_ann = corrected_returns(
                    ret_sel, factor_scores=factor_scores_df, annualize=ANNUALIZE
                )
                exp_ret = float(w_arr @ shrunk_ann.reindex([sel_with_data[j] for j in range(len(sel_with_data))]).fillna(0.0).values)
                exp_sharpe = exp_ret / exp_vol if exp_vol > 0 else 0.0
                c = ret_sel.corr().values
                n = len(sel_with_data)
                if n > 1:
                    avg_corr = float(np.nanmean(np.abs(c[np.triu_indices(n, k=1)])))

    # ── Safeguard warnings ────────────────────────────────────────────────────
    if len(sel_ids) < 8:
        warnings.append(f"low_diversification: {len(sel_ids)} stocks selected (< 8)")
    if avg_corr > 0.6:
        warnings.append(f"high_concentration: avg pairwise corr = {avg_corr:.2f}")

    sym_map = sel_df.set_index("stock_id")["symbol"].to_dict()
    result.update({
        "selected_stocks": [(sid, sym_map.get(sid, str(sid))) for sid in sel_ids],
        "weights": final_w,
        "expected_volatility": exp_vol,
        "expected_sharpe": exp_sharpe,
        "rebalance_date": as_of_date,
        "warnings": warnings,
        "avg_pairwise_corr": avg_corr,
        "_internal_scores": scores_dict if 'scores_dict' in locals() else {},
        "_internal_vols": vol_map if 'vol_map' in locals() else {},
        "_internal_cov": cov if 'cov' in locals() else None,
        "_internal_shrunk_ann": shrunk_ann if 'shrunk_ann' in locals() else None,
        "_internal_corr_matrix": corr_matrix if 'corr_matrix' in locals() else None,
    })
    
    # Update cache
    _BUILD_CACHE[cache_key] = result
    if len(_BUILD_CACHE) % 10 == 0:
        _save_build_cache()
    
    _logger.info("Total hybrid build time for %s: %.2f sec", as_of_date, time.time() - t_start)
    return result


def backtest_hybrid_portfolio(
    start_date: date,
    end_date: date,
    top_n: Union[int, str] = 15,
    min_score: float = 0.0,
    max_corr: float = 0.80,
    regime_mode: str = "volatility_targeting",
    risk_profile: str = "medium",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Backtest the hybrid portfolio over [start_date, end_date].

    Monthly rebalance only — portfolio is NOT rebuilt daily.
    NO LOOK-AHEAD BIAS:
      - Each rebalance calls build_diversified_hybrid_portfolio(reb_date) which
        uses only data on or before reb_date.
      - Vol targeting uses lagged (t-1) rolling vol via _apply_volatility_targeting.

    Returns (equity_curve_df, summary_dict) matching run_backtest's format.
    """
    _empty = (
        pd.DataFrame(columns=["date", "daily_return", "cumulative_return", "drawdown"]),
        {},
    )

    rebalance_dates = get_monthly_rebalance_dates(start_date, end_date)
    if len(rebalance_dates) < 2:
        return _empty

    daily_returns_list: List[pd.Series] = []
    old_weights: Dict[int, float] = {}
    period_sharpes: List[float] = []
    all_warnings: List[str] = []
    period_corrs: List[float] = []

    for i, reb_date in enumerate(rebalance_dates):
        # Build portfolio with NO future data
        hybrid = build_diversified_hybrid_portfolio(
            as_of_date=reb_date,
            top_n=top_n, min_score=min_score,
            max_corr=max_corr, regime_mode=regime_mode, risk_profile=risk_profile,
        )
        if "avg_pairwise_corr" in hybrid:
            period_corrs.append(hybrid["avg_pairwise_corr"])
        all_warnings.extend(hybrid.get("warnings", []))
        new_weights = hybrid.get("weights", {})
        if not new_weights:
            continue

        # Period end: next rebalance date or last price date
        if i + 1 < len(rebalance_dates):
            end_d = rebalance_dates[i + 1]
        else:
            with get_db_context() as db:
                last_price_date = db.execute(
                    select(Price.date).order_by(Price.date.desc()).limit(1)
                ).scalars().first()
            end_d = last_price_date if last_price_date else reb_date

        period_ret = compute_period_returns(reb_date, end_d, new_weights)
        if period_ret.empty:
            old_weights = new_weights
            continue

        # Transaction costs on first day
        cost = calculate_transaction_cost(old_weights, new_weights)
        if cost > 0 and len(period_ret) > 0:
            period_ret = period_ret.copy()
            period_ret.iloc[0] -= cost

        # Track per-period Sharpe for stability check
        if len(period_ret) > 1:
            pv = float(period_ret.std(ddof=0))
            ps = float(period_ret.mean() * ANNUALIZE) / (pv * math.sqrt(ANNUALIZE)) if pv > 0 else 0.0
            period_sharpes.append(ps)

        daily_returns_list.append(period_ret)
        old_weights = new_weights

    if not daily_returns_list:
        return _empty

    full_ret = pd.concat(daily_returns_list)
    full_ret = full_ret[~full_ret.index.duplicated(keep="last")].sort_index()

    # Regime overlay — lagged vol (no look-ahead by construction)
    if regime_mode == "volatility_targeting":
        full_ret = _apply_volatility_targeting(
            full_ret, TARGET_ANNUAL_VOL,
            window=ROLLING_VOL_WINDOW, leverage_min=LEVERAGE_MIN, leverage_max=LEVERAGE_MAX,
        )

    cum = (1 + full_ret).cumprod()
    cum_ret = cum - 1.0
    run_max = cum.cummax()
    dd = (cum - run_max) / run_max

    date_vals = full_ret.index.date if hasattr(full_ret.index, "date") else pd.to_datetime(full_ret.index).date
    equity_curve = pd.DataFrame({
        "date": date_vals,
        "daily_return": full_ret.values,
        "cumulative_return": cum_ret.values,
        "drawdown": dd.values,
    })

    # Metrics
    n_days = len(full_ret)
    n_years = n_days / ANNUALIZE
    total_ret = float(cum_ret.iloc[-1]) if n_days > 0 else 0.0
    cagr = (1 + total_ret) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0
    ann_vol = float(full_ret.std(ddof=0) * math.sqrt(ANNUALIZE)) if n_days > 1 else 0.0
    sharpe = float(full_ret.mean() * ANNUALIZE) / ann_vol if ann_vol > 0 else 0.0
    max_dd = float(dd.min()) if n_days > 0 else 0.0
    try:
        monthly = (1 + full_ret).resample("ME").prod() - 1
    except TypeError:
        monthly = (1 + full_ret).resample("M").prod() - 1
    win_rate = float((monthly > 0).mean()) if len(monthly) > 0 else 0.0

    # Sharpe stability warning
    if len(period_sharpes) >= 3 and float(np.std(period_sharpes)) > 2.0:
        all_warnings.append(f"sharpe_unstable: std = {float(np.std(period_sharpes)):.2f}")

    avg_corr_overall = float(np.mean(period_corrs)) if period_corrs else 0.5

    summary: Dict[str, Any] = {
        "CAGR": cagr, "Volatility": ann_vol, "Sharpe": sharpe,
        "Max Drawdown": max_dd, "Total Return": total_ret, "Win Rate": win_rate,
        "warnings": list(set(all_warnings)),   # deduplicate
        "avg_pairwise_corr": avg_corr_overall,
        "optimal_n": hybrid.get("optimal_n", top_n) if "hybrid" in locals() else top_n,
    }
    return equity_curve, summary


def rebalance_only_if_new_month(
    top_n: Union[int, str] = 15,
    min_score: float = 0.0,
    max_corr: float = 0.80,
    regime_mode: str = "volatility_targeting",
    risk_profile: str = "medium",
    force: bool = False,
    as_of_date: Optional[date] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Rebalance the hybrid portfolio at most once per calendar month.

    Guards
    ------
    - If the state file records that a rebalance already occurred in the
      current calendar month, this call is a NO-OP (returns status="skipped").
    - Pass force=True to trigger a rebalance regardless of the guard.
    - If no daemon is running, call this function on application boot or
      from the manual "Trigger Rebalance" UI button; the month-change
      guard ensures idempotency.

    Parameters
    ----------
    top_n        : number of stocks to select.
    min_score    : minimum composite score for a stock to be considered.
    max_corr     : maximum pairwise correlation allowed between selected stocks.
    regime_mode  : 'volatility_targeting' or 'static'.
    risk_profile : passed through to build_diversified_hybrid_portfolio.
    force        : if True, always rebalance regardless of the month guard.
    as_of_date   : override "today" for testing.  None = date.today().

    Returns dict with keys
    ----------------------
    status          : 'rebalanced' | 'skipped' | 'error'
    rebalanced_on   : date of the rebalance (or None if skipped/error)
    last_rebalance  : date of the most-recent successful rebalance from state
    portfolio       : result of build_diversified_hybrid_portfolio (or None)
    message         : human-readable summary
    """
    today = as_of_date or date.today()
    
    with _state_file_lock:
        state = _load_rebalance_state(user_id)

        # ── Month-change guard ────────────────────────────────────────────────────
        last_str: Optional[str] = state.get("last_rebalance_date")
        last_date: Optional[date] = None
        if last_str:
            try:
                last_date = date.fromisoformat(last_str)
            except ValueError:
                last_date = None

        already_this_month = (
            last_date is not None
            and last_date.year == today.year
            and last_date.month == today.month
        )

        if already_this_month and not force:
            msg = (
                f"Rebalance skipped — already ran on {last_date} "
                f"(current month {today.year}-{today.month:02d}). "
                f"Use force=True to override."
            )
            _logger.info(msg)
            return {
                "status": "skipped",
                "rebalanced_on": None,
                "last_rebalance": last_date,
                "portfolio": None,
                "message": msg,
            }

    # ── Determine the as-of date: prefer latest score date <= today ───────────
    score_date = get_latest_score_date_on_or_before(today)
    build_date = score_date if score_date else today

    # ── Build the portfolio ───────────────────────────────────────────────────
    _logger.info(
        "Hybrid portfolio rebalance starting (as_of=%s, top_n=%d, max_corr=%.2f, force=%s)",
        build_date, top_n, max_corr, force,
    )
    try:
        portfolio = build_diversified_hybrid_portfolio(
            as_of_date=build_date,
            top_n=top_n,
            min_score=min_score,
            max_corr=max_corr,
            regime_mode=regime_mode,
            risk_profile=risk_profile,
            force_n_recalc=force,
            user_id=user_id,
        )
    except Exception as exc:
        err_msg = f"Hybrid portfolio rebalance FAILED: {exc}"
        _logger.error(err_msg)
        return {
            "status": "error",
            "rebalanced_on": None,
            "last_rebalance": last_date,
            "portfolio": None,
            "message": err_msg,
        }

    # ── Persist state ─────────────────────────────────────────────────────────
    n_stocks = len(portfolio.get("selected_stocks", []))
    port_warnings = portfolio.get("warnings", [])
    history_entry: Dict[str, Any] = {
        "date": str(today),
        "build_date": str(build_date),
        "n_stocks": n_stocks,
        "top_n": top_n,
        "max_corr": max_corr,
        "forced": force,
        "warnings": port_warnings,
        "expected_volatility": portfolio.get("expected_volatility", 0.0),
        "expected_sharpe": portfolio.get("expected_sharpe", 0.0),
    }
    history: List[Dict] = state.get("history", [])
    history.append(history_entry)
    # Keep only last 36 months of history
    if len(history) > 36:
        history = history[-36:]

    new_state: Dict[str, Any] = {
        "last_rebalance_date": str(today),
        "last_build_date": str(build_date),
        "history": history,
    }
    _save_rebalance_state(new_state, user_id)

    # ── Persist to DB (INSERT-only, non-blocking on failure) ─────────────────
    try:
        from services.allocation_persistence import (
            save_monthly_allocation,
            save_portfolio_metrics,
        )
        weights_dict: Dict[int, float] = portfolio.get("weights", {})
        if weights_dict:
            save_monthly_allocation(
                portfolio_type=STRATEGY_AUTO_HYBRID,
                rebalance_date=today,
                weights=weights_dict,
                portfolio_name=None,    # system portfolio — no user name
                forced=force,
                optimal_n_used=portfolio.get("optimal_n"),
                user_id=user_id,
            )
        save_portfolio_metrics(
            portfolio_type=STRATEGY_AUTO_HYBRID,
            rebalance_date=today,
            metrics={
                "CAGR":         portfolio.get("expected_sharpe", 0.0),   # forward-looking proxy
                "Sharpe":       portfolio.get("expected_sharpe", 0.0),
                "Volatility":   portfolio.get("expected_volatility", 0.0),
                "Max Drawdown": 0.0,
                "Calmar":       0.0,
            },
            context={
                "n_stocks":             n_stocks,
                "expected_volatility":  portfolio.get("expected_volatility"),
                "expected_sharpe":      portfolio.get("expected_sharpe"),
                "avg_pairwise_corr":    portfolio.get("avg_pairwise_corr"),
            },
            forced=force,
            user_id=user_id,
        )
    except Exception as _db_exc:
        _logger.warning("DB persistence after rebalance failed (non-fatal): %s", _db_exc)

    # ── Log success ───────────────────────────────────────────────────────────
    success_msg = (
        f"Hybrid portfolio updated on {today}  "
        f"| build_date={build_date}  "
        f"| selected={n_stocks} stocks  "
        f"| exp_vol={portfolio.get('expected_volatility', 0):.1%}  "
        f"| exp_sharpe={portfolio.get('expected_sharpe', 0):.2f}"
        + (f"  | warnings={port_warnings}" if port_warnings else "")
    )
    _logger.info(success_msg)

    return {
        "status": "rebalanced",
        "rebalanced_on": today,
        "last_rebalance": today,
        "portfolio": portfolio,
        "message": success_msg,
    }


__all__ = [
    "STRATEGY_AUTO_HYBRID",
    "get_monthly_rebalance_dates",
    "get_month_end_trading_days",
    "build_diversified_hybrid_portfolio",
    "backtest_hybrid_portfolio",
    "rebalance_only_if_new_month",
]
