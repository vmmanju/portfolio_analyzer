import numpy as np
import pandas as pd
from datetime import date
from typing import Dict, Any, List, Optional
import logging
import json
from pathlib import Path
import math

_logger = logging.getLogger(__name__)

# Minimum N floor — prevents the optimizer from selecting too few stocks
# for adequate diversification. Can be overridden if the universe is small.
MINIMUM_N = 10

def _get_state_file(user_id: Optional[int] = None) -> Path:
    if user_id is not None:
        return Path(f"data/auto_n_state_user_{user_id}.json")
    return Path("data/auto_n_state.json")

def _load_cached_n_state(user_id: Optional[int] = None) -> Dict[str, Any]:
    state_file = _get_state_file(user_id)
    if not state_file.exists():
        return {}
    try:
        with open(state_file, "r") as f:
            return json.load(f)
    except Exception as e:
        _logger.warning("Could not load auto N state: %s", e)
        return {}

def _save_cached_n_state(state: Dict[str, Any], user_id: Optional[int] = None) -> None:
    try:
        state_file = _get_state_file(user_id)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_copy = dict(state)
        if "evaluation_table" in state_copy and isinstance(state_copy["evaluation_table"], pd.DataFrame):
            state_copy["evaluation_table"] = state_copy["evaluation_table"].to_dict('records')
            
        with open(state_file, "w") as f:
            json.dump(state_copy, f, indent=2, default=str)
    except Exception as e:
        _logger.warning("Could not save auto N state: %s", e)

def get_current_optimal_n(user_id: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve the current cached N-selection status without triggering a simulation."""
    state = _load_cached_n_state(user_id)
    if not state:
        return {
            "optimal_n": 15,
            "benefit_curve": [],
            "diagnostics": {"error": "No cached state available."},
            "evaluation_table": pd.DataFrame(),
            "calibration_source": "none"
        }
    if "evaluation_table" in state and isinstance(state["evaluation_table"], list):
        state["evaluation_table"] = pd.DataFrame(state["evaluation_table"])
    state["calibration_source"] = "cached"
    return state

def update_optimal_n_if_due(
    as_of_date: date,
    min_score: float = 0.0,
    max_corr: float = 0.80,
    regime_mode: str = "volatility_targeting",
    risk_profile: str = "medium",
    force: bool = False,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Check whether a new Auto N-selection is due (every 6 months) and compute if so."""
    state = _load_cached_n_state(user_id)
    
    if not force and state and "last_run_date" in state:
        try:
            last_run = date.fromisoformat(state["last_run_date"])
            # If we are looking for the same or older date, apply 6-month cache
            if as_of_date <= last_run:
                months_elapsed = (as_of_date.year - last_run.year) * 12 + (as_of_date.month - last_run.month)
                if months_elapsed < 6:
                    _logger.info("Auto N selection not due. Using cached N=%d (last run: %s)", state.get("optimal_n", 15), last_run)
                    if "evaluation_table" in state and isinstance(state["evaluation_table"], list):
                        state["evaluation_table"] = pd.DataFrame(state["evaluation_table"])
                    state["calibration_source"] = "cached"
                    state["updated"] = False
                    return state
            else:
                _logger.info("Newer data detected (%s > %s). Triggering Auto N re-selection.", as_of_date, last_run)
        except Exception as e:
            _logger.debug("Failed parsing last_run_date: %s", e)
            
    _logger.info("Auto N selection due (or forced). Running optimal N selection for %s...", as_of_date)
    new_state = select_optimal_n(as_of_date, min_score, max_corr, regime_mode, risk_profile)
    
    new_state["last_run_date"] = as_of_date.isoformat()
    new_state["calibration_source"] = "new"
    new_state["updated"] = True
    
    _save_cached_n_state(new_state, user_id)
    return new_state

def select_optimal_n(
    as_of_date: date,
    min_score: float = 0.0,
    max_corr: float = 0.80,
    regime_mode: str = "volatility_targeting",
    risk_profile: str = "medium"
) -> Dict[str, Any]:
    """Dynamically determine the optimal number of stocks (N*) to include in a portfolio."""
    from services.auto_diversified_portfolio import build_diversified_hybrid_portfolio, _load_returns_up_to
    from services.stop_loss_engine import get_historical_prices, compute_stop_loss_scores
    from services.risk_responsiveness import get_historical_prices_rrc, compute_rrc_scores
    from services.stability_analyzer import compute_rolling_stability

    # 1. Pre-calculate RRC and SLS scores for a candidate pool to avoid redundant overhead in the loop
    from services.portfolio import load_ranked_stocks
    from services.backtest import get_latest_score_date_on_or_before
    
    precomputed = {"rrc": None, "sls": None}
    pool_ids = []
    
    score_date = get_latest_score_date_on_or_before(as_of_date)
    if score_date:
        ranked = load_ranked_stocks(score_date)
        if not ranked.empty:
            # We fetch top 60 to have plenty of headroom for correlation and scoring filters
            pool_ids = ranked.head(60)["stock_id"].astype(int).tolist()
            
    if pool_ids:
        try:
            rrc_prices = get_historical_prices_rrc(as_of_date, pool_ids, lookback_years=5)
            precomputed["rrc"] = compute_rrc_scores(rrc_prices)
            
            sls_prices = get_historical_prices(as_of_date, pool_ids)
            precomputed["sls"] = compute_stop_loss_scores(sls_prices)
        except Exception as e:
            _logger.warning("Precomputation failed in Auto N-selector: %s", e)

    # 1a. Broad Returns Pre-fetch (90 & 180 days) to prime the cache for all loop iterations
    # This avoids 13x database hits during the evaluation loop below.
    if pool_ids:
        _load_returns_up_to(pool_ids, as_of_date, lookback_days=180) # primes 180-day stability cache
        _load_returns_up_to(pool_ids, as_of_date, lookback_days=90)  # primes 90-day correlation/vol cache

    # 2. Performance-Optimized Candidate Selection
    # Instead of calling build_diversified_hybrid_portfolio 13 times, we run the logic once for max(N)
    # and then subset/reweight. This is possible because the greedy filter is sequential.
    max_n = 30
    try:
        max_port = build_diversified_hybrid_portfolio(
            as_of_date=as_of_date,
            top_n=max_n,
            min_score=min_score,
            max_corr=max_corr,
            regime_mode=regime_mode,
            risk_profile=risk_profile,
            use_sls=True, # Enforce SLS during N-selection
            _skip_logging=True,
            precomputed_scores=precomputed
        )
    except Exception as e:
        _logger.error("Failed to build baseline portfolio for N-selection: %s", e)
        return {"optimal_n": 15, "evaluation_table": pd.DataFrame(), "benefit_curve": [], "diagnostics": {"error": str(e)}}

    # The subset of stocks selected for any n <= max_n will (by greedy definition) be the 
    # first n surviving stocks from the max_port.
    total_selected_ordered: List[int] = []
    if max_port.get("selected_stocks"):
        total_selected_ordered = [int(pair[0]) for pair in max_port["selected_stocks"]]
    
    if not total_selected_ordered:
         return {"optimal_n": 15, "evaluation_table": pd.DataFrame(), "benefit_curve": [], "diagnostics": {"error": "No stocks selected for evaluation."}}

    # 3. Define Candidate N Range & Evaluate
    # Enforce MINIMUM_N floor — only evaluate N >= MINIMUM_N (unless universe is too small)
    effective_min_n = min(MINIMUM_N, len(total_selected_ordered))
    n_candidates = [n for n in range(effective_min_n, 31, 2) if n <= len(total_selected_ordered)]
    if not n_candidates:
        n_candidates = [len(total_selected_ordered)]
    eval_results = []
    
    # Extract data needed for weighting from the max_port and precomputed_scores
    # This avoids calling the full builder 13 times.
    scores_dict = max_port.get("_internal_scores", {})
    vol_map = max_port.get("_internal_vols", {})
    from services.auto_diversified_portfolio import SCORE_TILT_FACTOR
    
    for n in n_candidates:
        sub_ids = total_selected_ordered[:n]
        
        # Fast Weighting (matches build_diversified_hybrid_portfolio Step 3)
        # Apply inverse-vol + tilt
        inv_vols = {sid: 1.0 / max(vol_map.get(sid, 1.0), 1e-8) for sid in sub_ids}
        tot_inv = sum(inv_vols.values())
        base_w = {sid: v / tot_inv for sid, v in inv_vols.items()} if tot_inv > 0 else {sid: 1.0 / n for sid in sub_ids}
        
        sv = [scores_dict.get(sid, 0.0) for sid in sub_ids]
        s_min, s_max = (min(sv), max(sv)) if sv else (0,0)
        norm_sc = {sid: (scores_dict.get(sid, 0.0) - s_min) / (s_max - s_min) if s_max > s_min else 0.5 for sid in sub_ids}
        
        tilted = {sid: base_w[sid] * (1.0 + SCORE_TILT_FACTOR * norm_sc[sid]) for sid in sub_ids}
        tot_tilted = sum(tilted.values())
        weights = {sid: w / tot_tilted for sid, w in tilted.items()} if tot_tilted > 0 else base_w
        
        stock_ids = list(weights.keys())
        w_array = np.array(list(weights.values()))
        
        # ── Fast Stats ────────────────────────────────────────────────────────
        # We need expected_vol and expected_sharpe for the benefit scoring.
        # We use the covariance and shrunk returns already computed in build_diversified_hybrid_portfolio
        # (conveniently accessible via the pre-build 'max_port' stats if we add them to the dict)
        exp_vol = 0.15
        exp_sharpe = 1.0
        avg_corr = 0.5
        
        # If max_port provided internal covariance data, use it for N-specific stats
        cov = max_port.get("_internal_cov")
        shrunk_ann = max_port.get("_internal_shrunk_ann")
        
        if cov is not None and shrunk_ann is not None:
            # Re-index to current sub_ids
            try:
                # w_array matches stock_ids
                pv = float(w_array @ cov.loc[stock_ids, stock_ids].values @ w_array)
                exp_vol = math.sqrt(max(pv, 0.0))
                exp_ret = float(w_array @ shrunk_ann.reindex(stock_ids).fillna(0.0).values)
                exp_sharpe = exp_ret / exp_vol if exp_vol > 0 else 0.0
                
                # Internal corr slice
                corr_sub = max_port.get("_internal_corr_matrix")
                if corr_sub is not None:
                    n_stocks = len(stock_ids)
                    if n_stocks > 1:
                        c_vals = corr_sub.loc[stock_ids, stock_ids].values
                        avg_corr = float(np.nanmean(np.abs(c_vals[np.triu_indices(n_stocks, k=1)])))
            except Exception:
                pass
        
        div_score = max(0.0, 1.0 - avg_corr)
        
        # Concentration Penalty — scaling formula instead of binary
        # Starts penalizing when top-3 stocks hold > 40% weight or avg_corr > 0.50
        w_sorted = np.sort(w_array)[::-1]
        top_3_weight = w_sorted[:3].sum() if len(w_sorted) >= 3 else 1.0
        weight_penalty = max(0.0, top_3_weight - 0.40) * 2.0   # 0.0 at 40%, 0.2 at 50%, 0.4 at 60%
        corr_penalty = max(0.0, avg_corr - 0.50) * 2.0          # 0.0 at 0.50, 0.2 at 0.60
        conc_penalty = min(1.0, weight_penalty + corr_penalty)   # cap at 1.0
            
        # Stop-loss & RRC aggregate (use precomputed if available)
        sls_agg = 50.0
        _sls_df = precomputed.get("sls")
        if _sls_df is not None:
            v_sls = [_sls_df.loc[s, "stop_loss_score"] * weights[s] for s in stock_ids if s in _sls_df.index]
            if v_sls: sls_agg = float(sum(v_sls))
                
        rrc_agg = 50.0
        _rrc_scores = precomputed.get("rrc")
        if _rrc_scores is not None:
            v_rrc = [_rrc_scores[s] * weights[s] for s in stock_ids if s in _rrc_scores.index]
            if v_rrc: rrc_agg = float(sum(v_rrc))
            
        # Stability estimate via pseudo-equity curve over last 180 days
        # This now hits the cache primed at step 1a.
        stab_score = 50.0
        try:
            hist_ret = _load_returns_up_to(stock_ids, as_of_date, lookback_days=180)
            if not hist_ret.empty:
                valid_cols = [c for c in stock_ids if c in hist_ret.columns]
                if valid_cols:
                    _w_series = pd.Series(weights)
                    port_ret = (hist_ret[valid_cols].fillna(0.0).values * _w_series.reindex(valid_cols).fillna(0.0).values).sum(axis=1)
                    port_ret_series = pd.Series(port_ret, index=hist_ret.index)
                    cum = (1.0 + port_ret_series).cumprod()
                    rmax = cum.cummax()
                    eq_df = pd.DataFrame({
                        "date": pd.to_datetime(cum.index).date,
                        "daily_return": port_ret.values,
                        "cumulative_return": cum.values - 1.0,
                        "drawdown": ((cum - rmax) / rmax).values
                    })
                    stab_res = compute_rolling_stability(eq_df)
                    stab_score = float(stab_res.get("stability_score", 50.0))
        except Exception as e:
            _logger.debug("Stability estimate failed for N=%d: %s", n, e)

        eval_results.append({
            "n": n,
            "sharpe_raw": exp_sharpe,
            "diversification_score": div_score,
            "stability_score": stab_score,
            "rrc_agg": rrc_agg,
            "sls_agg": sls_agg,
            "concentration_penalty": conc_penalty,
            "top_3_weight": top_3_weight,
            "avg_corr": avg_corr,
            "exp_vol": exp_vol
        })
        
    if not eval_results:
        return {
            "optimal_n": 15,
            "evaluation_table": pd.DataFrame(),
            "benefit_curve": [],
            "diagnostics": {"error": "No valid candidates produced."}
        }
        
    df = pd.DataFrame(eval_results)
    
    # Normalize Sharpe across the candidates
    sharpe_min = df["sharpe_raw"].min()
    sharpe_max = df["sharpe_raw"].max()
    if sharpe_max > sharpe_min:
        df["sharpe_normalized"] = (df["sharpe_raw"] - sharpe_min) / (sharpe_max - sharpe_min)
    else:
        df["sharpe_normalized"] = 0.5
        
    # Overfitting Risk: Detect if a smaller N has vastly higher sharpe than a larger N
    df = df.sort_values("n").reset_index(drop=True)
    df["overfitting_risk"] = 0.0
    for i in range(len(df)):
        current_n = df.loc[i, "n"]
        current_sh = df.loc[i, "sharpe_raw"]
        higher_n_sh = df[df["n"] > current_n]["sharpe_raw"]
        if not higher_n_sh.empty:
            max_higher_sh = higher_n_sh.max()
            if max_higher_sh > 0 and current_sh > max_higher_sh * 1.5:  # 50% higher than any larger N
                df.loc[i, "overfitting_risk"] = 1.0
                
    # Normalize scores for 0-1 range
    # Governance is (1 - overfitting_risk) scaled 0-1
    df["governance_score"] = 1.0 - df["overfitting_risk"]

    # Reweighted benefit formula — reduced Sharpe dominance for better diversification
    df["PortfolioBenefit"] = (
        0.30 * df["sharpe_normalized"] +      # was 0.40 — smaller N advantage reduced
        0.25 * df["diversification_score"] +  # was 0.20 — more weight on low correlation
        0.25 * (df["stability_score"] / 100.0) +  # was 0.20 — more weight on consistency
        0.10 * (df["rrc_agg"] / 100.0) +
        0.10 * df["governance_score"] -
        0.10 * df["concentration_penalty"]    # was 0.05 — scaling penalty has more impact
    )
    
    # Regime-Aware Adjustment
    avg_vol = df["exp_vol"].mean()
    high_vol_regime = bool(avg_vol > 0.25)
    low_vol_regime = bool(avg_vol < 0.12)
    
    diagnostics: Dict[str, Any] = {
        "concentration_flag": bool(df["concentration_penalty"].sum() > 0),
        "overfit_flag": bool(df["overfitting_risk"].sum() > 0),
        "regime_adjustment_applied": high_vol_regime or low_vol_regime,
        "high_vol_regime": high_vol_regime,
        "low_vol_regime": low_vol_regime,
        "stability_guardrail_triggered": False
    }
    
    min_allowed_n = 5
    if high_vol_regime:
        min_allowed_n += 2
        diagnostics["regime_note"] = "High volatility detected. Minimum N increased to 7."
    elif low_vol_regime:
        diagnostics["regime_note"] = "Low volatility detected. Smaller N allowed."
    else:
        diagnostics["regime_note"] = "Normal volatility regime."
        
    df_allowed = df[df["n"] >= min_allowed_n].copy()
    if df_allowed.empty:
        df_allowed = df
        
    # Choose flat benefit curve bias towards larger N
    max_benefit = df_allowed["PortfolioBenefit"].max()
    threshold = max_benefit - 0.02 # 2% tolerance for flat curve
    
    top_candidates = df_allowed[df_allowed["PortfolioBenefit"] >= threshold]
    
    # If flat, select slightly larger N (stability bias)
    optimal_row = top_candidates.sort_values("n", ascending=False).iloc[0]
    n_star = int(optimal_row["n"])
    
    # Stability Guardrail: If stability < 50, try to find a larger N with better stability
    if optimal_row["stability_score"] < 50:
        higher_n_cands = df_allowed[df_allowed["n"] > n_star]
        improves_stab = higher_n_cands[higher_n_cands["stability_score"] > optimal_row["stability_score"]]
        if not improves_stab.empty:
            n_star = int(improves_stab.iloc[0]["n"])
            diagnostics["stability_guardrail_triggered"] = True
            _logger.info("Stability guardrail triggered: N increased to %d", n_star)

    benefit_curve = df[["n", "PortfolioBenefit"]].to_dict('records')
    
    _logger.info("Automatic N Selected: %d (max benefit = %.3f)", n_star, max_benefit)
    
    return {
        "optimal_n": n_star,
        "evaluation_table": df,
        "benefit_curve": benefit_curve,
        "diagnostics": diagnostics
    }
