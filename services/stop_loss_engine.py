"""services/stop_loss_engine.py

Vectorized Stop-Loss Scoring Layer (SLS).
Computes a dynamic, statistics-driven risk score predicting structural trend breakdown.

Features:
- No look-ahead bias.
- No naive fixed-percentage stops.
- Uses 6-month and 12-month rolling data.
- Generates 0-100 score where higher score means higher danger.
"""

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.database import get_db_context
from app.models import Price

# ── Hyperparameters ──────────────────────────────────────────────────────────
WEIGHT_DD         = 0.25
WEIGHT_VAR        = 0.20
WEIGHT_MOM        = 0.20
WEIGHT_ERR_BIAS   = 0.20
WEIGHT_REGIME     = 0.15

LOOK_BACK_12M_DAYS = 252
LOOK_BACK_6M_DAYS  = 126
LOOK_BACK_1M_DAYS  = 21

ANNUALIZE_FACTOR = 252


# Smart Cache for individual stock histories to avoid redundant large-range fetches
_STOCK_HISTORY_CACHE: Dict[int, pd.Series] = {}
import threading
_price_cache_lock = threading.RLock()

def get_historical_prices(as_of_date: date, stock_ids: List[int], lookback_days: int = LOOK_BACK_12M_DAYS) -> pd.DataFrame:
    """Fetch trailing close prices for a universe of stocks ending on as_of_date with smart global caching."""
    if not stock_ids:
        return pd.DataFrame()

    calendar_start = as_of_date - timedelta(days=int(lookback_days * 1.6))
    
    with _price_cache_lock:
        to_fetch_sid = []
        fetch_ranges = {} # sid -> (missing_start, missing_end)

        for sid in stock_ids:
            if sid not in _STOCK_HISTORY_CACHE:
                to_fetch_sid.append(sid)
                fetch_ranges[sid] = (calendar_start, as_of_date)
            else:
                hist = _STOCK_HISTORY_CACHE[sid]
                h_min = hist.index.min().date()
                h_max = hist.index.max().date()
                
                # Case 1: Gap before
                if h_min > calendar_start:
                    fetch_ranges[sid] = (calendar_start, h_min)
                    if sid not in to_fetch_sid: to_fetch_sid.append(sid)
                
                # Case 2: Gap after
                if h_max < as_of_date:
                    cur = fetch_ranges.get(sid, (h_max, h_max))
                    fetch_ranges[sid] = (min(cur[0], h_max), as_of_date)
                    if sid not in to_fetch_sid: to_fetch_sid.append(sid)

    if to_fetch_sid:
        # Union of all missing ranges
        global_start = min(r[0] for r in fetch_ranges.values())
        global_end = max(r[1] for r in fetch_ranges.values())

        with get_db_context() as db:
            rows = db.execute(
                select(Price.date, Price.stock_id, Price.close)
                .where(
                    Price.stock_id.in_(to_fetch_sid),
                    Price.date >= global_start,
                    Price.date <= global_end
                )
                .order_by(Price.date)
            ).all()

        if rows:
            df = pd.DataFrame(rows, columns=["date", "stock_id", "close"])
            df["date"] = pd.to_datetime(df["date"])
            for sid in to_fetch_sid:
                s_subset = df[df["stock_id"] == sid].set_index("date")["close"]
                if not s_subset.empty:
                    with _price_cache_lock:
                        if sid in _STOCK_HISTORY_CACHE:
                            old_hist = _STOCK_HISTORY_CACHE[sid]
                            new_hist = pd.concat([old_hist, s_subset])
                            _STOCK_HISTORY_CACHE[sid] = new_hist[~new_hist.index.duplicated(keep="last")].sort_index()
                        else:
                            _STOCK_HISTORY_CACHE[sid] = s_subset.sort_index()

    # Build the final pivot DataFrame from cache
    with _price_cache_lock:
        data_to_pivot = {}
        for sid in stock_ids:
            if sid in _STOCK_HISTORY_CACHE:
                hist = _STOCK_HISTORY_CACHE[sid]
                mask = (hist.index.date >= calendar_start) & (hist.index.date <= as_of_date)
                data_to_pivot[sid] = hist[mask]

    if not data_to_pivot:
        return pd.DataFrame()
        
    pivot_df = pd.DataFrame(data_to_pivot).ffill().sort_index()
    return pivot_df


def compute_stop_loss_scores(
    price_df: pd.DataFrame,
    error_bias: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Computes the continuous Stop-Loss Score (0-100) for all assets in price_df.
    Higher score means higher risk of structural breakdown.

    Parameters
    ----------
    price_df   : pd.DataFrame of daily close prices ending on the evaluation date.
                 Expected length up to 252 rows.
    error_bias : Optional pd.Series of expected systematic error adjustment.
                 Positive values mean the estimator historically overshoots (downside bias).

    Returns
    -------
    pd.DataFrame with index = stock_id and columns:
        [stop_loss_score, risk_level, recommended_action, trigger_threshold, 
         dd_score, var_score, mom_score, err_score, regime_score]
    """
    if price_df.empty:
        return pd.DataFrame()

    symbols = price_df.columns
    total_days = len(price_df)
    
    w_6m = min(LOOK_BACK_6M_DAYS, total_days)
    w_12m = min(LOOK_BACK_12M_DAYS, total_days)
    w_1m = min(LOOK_BACK_1M_DAYS, total_days)

    current_price = price_df.iloc[-1]
    daily_returns = price_df.pct_change().dropna(how="all")

    # 1️⃣ Peak Distance Component (Drawdown) 
    rolling_6m_high = price_df.iloc[-w_6m:].max()
    dd = (current_price - rolling_6m_high) / rolling_6m_high
    
    if len(symbols) > 5:
        # Normalize across universe: worst (most negative) DD gets score near 100
        dd_score = dd.rank(pct=True, ascending=False) * 100
    else:
        # ABSOLUTE MAPPING: 0% drop -> 0, 15% drop -> 100
        dd_score = pd.Series(np.clip(abs(dd) / 0.15 * 100, 0, 100), index=symbols)
    dd_score.fillna(0.0, inplace=True)

    # 2️⃣ Volatility-Adjusted Risk (Z-Score)
    current_return = daily_returns.iloc[-1] if not daily_returns.empty else pd.Series(0.0, index=symbols)
    mean_return = daily_returns.iloc[-w_6m:].mean()
    volatility = daily_returns.iloc[-w_6m:].std()
    
    z_score = (current_return - mean_return) / volatility.replace(0, np.nan)
    
    if len(symbols) > 5:
        # Normalize: Most negative Z-Score (worst drop) gets score near 100
        var_score = z_score.rank(pct=True, ascending=False) * 100
    else:
        # ABSOLUTE MAPPING: 0 std -> 0, -3 std -> 100
        var_score = pd.Series(0.0, index=symbols)
        # Apply only to negative z-scores
        mask = z_score < 0
        var_score[mask] = np.clip(abs(z_score[mask]) / 3.0 * 100, 0, 100)
    var_score.fillna(0.0, inplace=True)

    # 3️⃣ Momentum Breakdown
    mom_6m = (current_price / price_df.iloc[-w_6m]) - 1.0 if total_days >= w_6m else pd.Series(0.0, index=symbols)
    mom_12m = (current_price / price_df.iloc[-w_12m]) - 1.0 if total_days >= w_12m else mom_6m
    
    mom_penalty = pd.Series(0.0, index=symbols)
    # Give max penalty if structural momentum is breaking down across long timeframe
    mom_penalty[(mom_12m < 0) & (mom_6m < 0)] = 100.0

    # 4️⃣ Error Model Downside Bias
    if error_bias is not None and not error_bias.empty:
        # Align with symbols
        aligned_bias = error_bias.reindex(symbols).fillna(0.0)
        
        if len(symbols) > 5:
            # Larger positive bias = larger subtraction from predicted return = higher risk
            err_score = aligned_bias.rank(pct=True, ascending=True) * 100
        else:
            # ABSOLUTE MAPPING: 0 bias -> 0, 10% bias -> 100
            err_score = pd.Series(np.clip(aligned_bias / 0.10 * 100, 0, 100), index=symbols)
    else:
        # For unknown stocks, don't penalize. Neutral is 0 risk.
        err_score = pd.Series(0.0, index=symbols)

    # 5️⃣ Regime Stress Multiplier (Derived from the cross-section of returns)
    if not daily_returns.empty and len(daily_returns) >= w_1m:
        market_returns = daily_returns.mean(axis=1)
        market_vol_20 = market_returns.rolling(20).std().iloc[-1]
        market_vol_126 = market_returns.rolling(w_6m).std().median()
        
        if len(symbols) > 5:
            is_high_vol = bool(market_vol_20 > market_vol_126)
        else:
            # In single-stock mode, only trigger high regime if volatility spike is extreme (>2x median)
            # This prevents normal stock fluctuations from being flagged as "Regime Stress"
            is_high_vol = bool(market_vol_20 > 2.0 * market_vol_126)
    else:
        is_high_vol = False
        
    regime_score = 100.0 if is_high_vol else 0.0
    regime_series = pd.Series(regime_score, index=symbols)

    # ── 6️⃣ Risk Responsiveness Component (RRC) ──────────────────────────────
    from services.risk_responsiveness import compute_rrc_scores
    rrc_scores = compute_rrc_scores(price_df)
    rrc_aligned = rrc_scores.reindex(symbols).fillna(50.0)

    # ── Combine & Normalize ──────────────────────────────────────────────────
    sls_raw_base = (
        WEIGHT_DD * dd_score +
        WEIGHT_VAR * var_score +
        WEIGHT_MOM * mom_penalty +
        WEIGHT_ERR_BIAS * err_score +
        WEIGHT_REGIME * regime_series
    )
    
    # Mix in RRC: high responsiveness (high RRC) reduces danger
    sls_raw = 0.90 * sls_raw_base + 0.10 * (100.0 - rrc_aligned)
    
    sls_raw_series = pd.Series(sls_raw, index=symbols)
    sls = np.clip(sls_raw_series, 0, 100)

    # ── Map to Risk Level & Recommendation ───────────────────────────────────
    risk_level = pd.Series("Low", index=symbols)
    risk_level[sls >= 40] = "Moderate"
    risk_level[sls >= 70] = "High"
    risk_level[sls >= 80] = "Critical"

    action = pd.Series("Hold", index=symbols)
    # Above 80: Critical Danger
    action[sls >= 80] = "Exit"
    # 70-80: High Warning
    action[(sls >= 70) & (sls < 80)] = "Reduce"

    # ── Dynamic Trigger Threshold ────────────────────────────────────────────
    # Rather than a naive 10%, dynamic threshold = peak - 2.5 * annual_vol
    annual_vol = volatility * math.sqrt(ANNUALIZE_FACTOR)
    annual_vol.fillna(0.15, inplace=True)
    
    # Floor the drop tolerance at 5% and cap at 30% to prevent extreme values
    drop_pct = np.clip(2.5 * annual_vol, 0.05, 0.30)
    trigger_threshold = rolling_6m_high * (1 - drop_pct)

    results = pd.DataFrame({
        "stop_loss_score": sls.round(2),
        "risk_level": risk_level,
        "recommended_action": action,
        "trigger_threshold": trigger_threshold.round(2),
        "raw_drawdown": dd.round(4),
        "dd_score": dd_score.round(2),
        "var_score": var_score.round(2),
        "mom_score": mom_penalty.round(2),
        "err_score": err_score.round(2),
        "regime_score": regime_series.round(2)
    })
    
    return results


def analyze_stock_stop_loss(
    as_of_date: date,
    stock_id: int,
    error_bias: Optional[float] = None
) -> Dict[str, Any]:
    """
    Convenience wrapper to get the Stop-Loss Profile for a single stock.
    Returns the exact dictionary format requested in the spec.
    """
    err_series = pd.Series({stock_id: error_bias}) if error_bias is not None else None
    price_df = get_historical_prices(as_of_date, [stock_id])
    
    if price_df.empty:
        return {
            "stop_loss_score": 0.0,
            "risk_level": "Unknown",
            "recommended_action": "Hold",
            "trigger_threshold": 0.0,
            "raw_drawdown": 0.0,
            "components": {}
        }
        
    df_scores = compute_stop_loss_scores(price_df, err_series)
    row = df_scores.loc[stock_id]
    
    return {
        "stop_loss_score": float(row["stop_loss_score"]),
        "risk_level": str(row["risk_level"]),
        "recommended_action": str(row["recommended_action"]),
        "trigger_threshold": float(row["trigger_threshold"]),
        "raw_drawdown": float(row["raw_drawdown"]),
        "components": {
            "peak_distance_score": float(row["dd_score"]),
            "volatility_adjusted_score": float(row["var_score"]),
            "momentum_breakdown_score": float(row["mom_score"]),
            "error_bias_score": float(row["err_score"]),
            "regime_stress_score": float(row["regime_score"]),
        }
    }
def analyze_universe_stop_loss(
    as_of_date: date,
    stock_ids: List[int],
    error_biases: Optional[Dict[int, float]] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Batch version of analyze_stock_stop_loss for a whole universe/portfolio.
    Returns a mapping: stock_id -> profile_dict
    """
    if not stock_ids:
        return {}

    err_series = pd.Series(error_biases) if error_biases else None
    price_df = get_historical_prices(as_of_date, stock_ids)
    
    if price_df.empty:
        return {sid: {
            "stop_loss_score": 0.0,
            "risk_level": "Unknown",
            "recommended_action": "Hold",
            "trigger_threshold": 0.0,
            "raw_drawdown": 0.0,
            "components": {}
        } for sid in stock_ids}
        
    df_scores = compute_stop_loss_scores(price_df, err_series)
    
    out = {}
    for sid in stock_ids:
        if sid not in df_scores.index:
            out[sid] = {
                "stop_loss_score": 0.0,
                "risk_level": "Unknown",
                "recommended_action": "Hold",
                "trigger_threshold": 0.0,
                "raw_drawdown": 0.0,
                "components": {}
            }
            continue
            
        row = df_scores.loc[sid]
        out[sid] = {
            "stop_loss_score": float(row["stop_loss_score"]),
            "risk_level": str(row["risk_level"]),
            "recommended_action": str(row["recommended_action"]),
            "trigger_threshold": float(row["trigger_threshold"]),
            "raw_drawdown": float(row["raw_drawdown"]),
            "components": {
                "peak_distance_score": float(row["dd_score"]),
                "volatility_adjusted_score": float(row["var_score"]),
                "momentum_breakdown_score": float(row["mom_score"]),
                "error_bias_score": float(row["err_score"]),
                "regime_stress_score": float(row["regime_score"]),
            }
        }
    return out
