import math
import numpy as np
import pandas as pd
import logging
from datetime import date, timedelta
from typing import Dict, Any, List, Union

from sqlalchemy import select
from app.database import get_db_context
from app.models import Price

_logger = logging.getLogger(__name__)

# Smart Cache for individual stock histories to avoid redundant large-range fetches
_STOCK_HISTORY_CACHE: Dict[int, pd.Series] = {}
import threading
_price_cache_lock = threading.RLock()

def get_historical_prices_rrc(as_of_date: date, stock_ids: List[int], lookback_years: int = 5) -> pd.DataFrame:
    """Fetch trailing close prices for RRC evaluation with smart global caching."""
    if not stock_ids:
        return pd.DataFrame()

    calendar_start = as_of_date - timedelta(days=int(lookback_years * 365 + 10))
    
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
                    # If already has range, expand it
                    cur = fetch_ranges.get(sid, (h_max, h_max))
                    fetch_ranges[sid] = (min(cur[0], h_max), as_of_date)
                    if sid not in to_fetch_sid: to_fetch_sid.append(sid)

    if to_fetch_sid:
        # Optimization: Union of all missing ranges
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

def compute_rrc_scores(price_df: pd.DataFrame) -> pd.Series:
    """
    Vectorized calculation of Risk Responsiveness Component for multiple stocks.
    Returns a pandas Series with RRC scores (0-100) indexed by stock_id.
    """
    if price_df.empty or len(price_df) < 126:
        # Not enough data, return neutral 50s
        return pd.Series(50.0, index=price_df.columns)
        
    returns = price_df.pct_change().fillna(0.0)
    market_returns = returns.mean(axis=1)
    
    # 1. Volatility Spike Reaction (30%)
    vol_30 = market_returns.rolling(30).std()
    vol_75th = vol_30.quantile(0.75) if vol_30.notna().any() else 0.0
    is_high_vol = vol_30 > vol_75th
    
    # Find distinct volatility spikes
    spike_days = []
    for i in range(1, len(is_high_vol)):
        if is_high_vol.iloc[i] and not is_high_vol.iloc[i-1]:
            if not spike_days or (i - spike_days[-1] > 10):
                spike_days.append(i)
                
    last_3_spikes = spike_days[-3:] if len(spike_days) >= 3 else spike_days
    
    spike_drawdowns = pd.DataFrame(index=price_df.columns)
    for idx, s_idx in enumerate(last_3_spikes):
        end_idx = min(len(price_df), s_idx + 10)
        # Check max drawdown in the next 10 days
        window_prices = price_df.iloc[s_idx:end_idx]
        if not window_prices.empty:
            peak = window_prices.cummax()
            dds = (window_prices - peak) / peak
            max_dd = dds.min()
            spike_drawdowns[f"dd_{idx}"] = max_dd
            
    if not spike_drawdowns.empty:
        avg_spike_dd = spike_drawdowns.mean(axis=1).fillna(0)
    else:
        avg_spike_dd = pd.Series(0.0, index=price_df.columns)
        
    # Rank Spike DD (Lower negative is worse -> higher is better responsiveness)
    # Higher value = less drawdown
    if len(price_df.columns) > 5:
        score_spike = avg_spike_dd.rank(pct=True, ascending=True).fillna(0.5)
    else:
        # Absolute: 0 spike-drawdown is excellent (1.0), -10% is poor (0.0)
        score_spike = pd.Series(np.clip(1.0 + avg_spike_dd / 0.10, 0, 1), index=price_df.columns).fillna(0.5)
    
    # 2. Beta Compression (25%) & 4. Correlation Drift (20%)
    # Rolling 60-day stats
    cov_60 = returns.rolling(60).cov(market_returns)
    var_60 = market_returns.rolling(60).var()
    beta_60 = cov_60.div(var_60, axis=0).fillna(1.0)
    corr_60 = returns.rolling(60).corr(market_returns).fillna(1.0)
    
    high_vol_mask = is_high_vol
    normal_vol_mask = ~is_high_vol
    
    # Needs sufficiently large masks to compute means
    if high_vol_mask.sum() > 5 and normal_vol_mask.sum() > 5:
        beta_normal = beta_60[normal_vol_mask].mean()
        beta_high = beta_60[high_vol_mask].mean()
        beta_compression = beta_normal - beta_high
        
        corr_normal = corr_60[normal_vol_mask].mean()
        corr_high = corr_60[high_vol_mask].mean()
        corr_drift = corr_normal - corr_high
    else:
        beta_compression = pd.Series(0.0, index=price_df.columns)
        corr_drift = pd.Series(0.0, index=price_df.columns)
        
    if len(price_df.columns) > 5:
        score_beta = beta_compression.rank(pct=True, ascending=True).fillna(0.5)
        score_corr = corr_drift.rank(pct=True, ascending=True).fillna(0.5)
    else:
        # Absolute: Positive compression is good. 0.5 drop in beta -> excellent (1.0)
        score_beta = pd.Series(np.clip(0.5 + beta_compression, 0, 1), index=price_df.columns).fillna(0.5)
        # Absolute: Positive drift is good. 0.2 drop in corr -> excellent (1.0)
        score_corr = pd.Series(np.clip(0.5 + corr_drift / 0.4, 0, 1), index=price_df.columns).fillna(0.5)
    
    # 3. Drawdown Recovery Speed (25%)
    # Measure in the last 252 days
    recent_prices = price_df.iloc[-252:] if len(price_df) >= 252 else price_df
    peaks = recent_prices.cummax()
    dds = (recent_prices - peaks) / peaks
    
    recovery_days = pd.Series(252.0, index=price_df.columns)
    for col in price_df.columns:
        stock_dds = dds[col]
        min_dd_idx = stock_dds.idxmin()
        min_dd = stock_dds.loc[min_dd_idx]
        
        if min_dd < -0.05: # Only meaningful drawdowns
            post_trough = stock_dds.loc[min_dd_idx:]
            # Recover 50% means DD becomes > min_dd / 2
            target_dd = min_dd / 2.0
            recovered = post_trough[post_trough >= target_dd]
            if not recovered.empty:
                days_to_recover = len(post_trough.loc[:recovered.index[0]])
                recovery_days[col] = float(days_to_recover)
        else:
            recovery_days[col] = 0.0 # No real drawdown, instant recovery
            
    # Rank recovery (Less days is better)
    if len(price_df.columns) > 5:
        score_recovery = recovery_days.rank(pct=True, ascending=False).fillna(0.5)
    else:
        # Absolute: 0 days recovery -> 1.0, 60+ days -> 0.0
        score_recovery = pd.Series(np.clip(1.0 - recovery_days / 60.0, 0, 1), index=price_df.columns).fillna(0.5)

    rrc_raw = (
        0.30 * score_spike +
        0.25 * score_beta +
        0.25 * score_recovery +
        0.20 * score_corr
    )
    rrc = rrc_raw * 100.0
    return pd.Series(np.clip(rrc, 0, 100), index=price_df.columns)

def compute_stock_rrc(symbol: Union[int, str], as_of_date: date) -> float:
    """
    Compute RRC for a single stock.
    Typically symbol is a stock_id (int). 
    If a string is passed, it returns default 50.0 (as DB needs int stock_id).
    """
    if isinstance(symbol, str):
        # Graceful fallback if called with string ticker but DB expects int IDs.
        return 50.0
    
    # We must fetch a broader universe to get a valid 'market' baseline.
    # To keep it lightweight if possible, we just fetch this symbol.
    # Note: Market baseline will be poor with 1 stock. Using global RRC batch is preferred.
    df = get_historical_prices_rrc(as_of_date, [symbol], lookback_years=5)
    if df.empty or symbol not in df.columns:
        return 50.0
        
    # Since market is just the mean of the dataframe, if dataframe has 1 stock,
    # beta=1.0, corr=1.0. This makes beta compression & corr drift 0.
    # A true system should pass the batch DataFrame.
    res = compute_rrc_scores(df)
    return float(res.get(symbol, 50.0))

def compute_portfolio_rrc(equity_curve: pd.Series, returns_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute Risk Responsiveness for a Portfolio equity curve.
    returns_df acts as the market context matrix.
    """
    if equity_curve.empty or len(equity_curve) < 126 or returns_df.empty:
        return {"rrc_score": 50.0, "components": {}}
        
    port_returns = equity_curve.pct_change().fillna(0.0)
    market_returns = returns_df.mean(axis=1).reindex(port_returns.index).fillna(0.0)
    
    # 1. Drawdown containment speed
    vol_30 = market_returns.rolling(30).std()
    vol_75th = vol_30.quantile(0.75) if vol_30.notna().any() else 0.0
    is_high_vol = vol_30 > vol_75th
    
    spike_days = []
    for i in range(1, len(is_high_vol)):
        if is_high_vol.iloc[i] and not is_high_vol.iloc[i-1]:
            if not spike_days or (i - spike_days[-1] > 10):
                spike_days.append(i)
                
    spikes = spike_days[-3:] if len(spike_days) >= 3 else spike_days
    avg_spike_dd = 0.0
    if spikes:
        dds = []
        for s_idx in spikes:
            end_idx = min(len(equity_curve), s_idx + 10)
            window = equity_curve.iloc[s_idx:end_idx]
            if not window.empty:
                peak = window.cummax()
                dd = ((window - peak) / peak).min()
                dds.append(dd)
        avg_spike_dd = np.mean(dds)
    
    # Score 0-1, arbitrary scaling where 0% drop is 1.0, -10% drop is 0.0
    containment_score = max(0.0, min(1.0, 1.0 + (avg_spike_dd / 0.10)))
    
    # 2. Volatility regime shift adaptation (Rolling Beta Compression)
    cov_60 = port_returns.rolling(60).cov(market_returns)
    var_60 = market_returns.rolling(60).var()
    beta_60 = cov_60 / var_60.replace(0, np.nan)
    beta_60 = beta_60.fillna(1.0)
    
    high_vol_mask = is_high_vol
    normal_vol_mask = ~is_high_vol
    if high_vol_mask.sum() > 5 and normal_vol_mask.sum() > 5:
        beta_normal = beta_60[normal_vol_mask].mean()
        beta_high = beta_60[high_vol_mask].mean()
        beta_comp = beta_normal - beta_high
    else:
        beta_comp = 0.0
        
    beta_score = max(0.0, min(1.0, 0.5 + beta_comp))
    
    # 3. Sharpe stability during stress
    normal_ret = port_returns[normal_vol_mask]
    high_ret = port_returns[high_vol_mask]
    
    sharpe_normal = (normal_ret.mean() / normal_ret.std()) if normal_ret.std() > 0 else 0
    sharpe_high = (high_ret.mean() / high_ret.std()) if high_ret.std() > 0 else 0
    
    sharpe_stab = sharpe_high / sharpe_normal if sharpe_normal > 0 else 0
    sharpe_score = max(0.0, min(1.0, sharpe_stab))
    
    rrc_raw = (0.4 * containment_score) + (0.3 * beta_score) + (0.3 * sharpe_score)
    rrc = float(rrc_raw * 100.0)
    
    return {
        "rrc_score": rrc,
        "components": {
            "containment": containment_score,
            "beta_compression": beta_score,
            "sharpe_stability": sharpe_score
        }
    }
