"""
Phase 8: Recommendation engine.

Rule-based stock signals and allocation suggestions from latest composite score,
3-month score trend, volatility_score, and regime. No ML/LLM. Fully transparent.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from datetime import date, timedelta
import os
from typing import Any, Optional

from sqlalchemy import select

from app.database import get_db_context
from app.models import Factor, Score, Stock


# Configurable thresholds
TREND_IMPROVING_THRESHOLD = 0.15   # composite_score change vs 3mo ago
TREND_DECLINING_THRESHOLD = -0.15
REGIME_VOL_CUTOFF = 0.0            # volatility_score > this => high_vol (z-scored)
SCORE_TREND_LOOKBACK_DAYS = 90     # ~3 months

# Validation output limit (can be overridden with env var RECOMMENDATION_VALIDATION_LIMIT)
DEFAULT_VALIDATION_LIMIT = int(os.getenv("RECOMMENDATION_VALIDATION_LIMIT", "100"))


def _get_stock_id(symbol: str) -> Optional[int]:
    """Return stock_id for symbol, or None if not found."""
    with get_db_context() as db:
        row = db.execute(select(Stock.id).where(Stock.symbol == symbol)).scalars().first()
    return row


def _get_latest_score_and_factor(stock_id: int) -> Optional[dict[str, Any]]:
    """Load latest score and factor for stock; return single dict or None."""
    with get_db_context() as db:
        # Latest score date for this stock
        score_date_stmt = (
            select(Score.date)
            .where(Score.stock_id == stock_id)
            .order_by(Score.date.desc())
            .limit(1)
        )
        score_date = db.execute(score_date_stmt).scalars().first()
        if score_date is None:
            return None

        score_stmt = (
            select(Score.composite_score, Score.rank)
            .where(Score.stock_id == stock_id, Score.date == score_date)
        )
        score_row = db.execute(score_stmt).first()
        factor_stmt = (
            select(
                Factor.momentum_score,
                Factor.volatility_score,
            )
            .where(Factor.stock_id == stock_id, Factor.date == score_date)
        )
        factor_row = db.execute(factor_stmt).first()

    if score_row is None:
        return None

    composite = float(score_row[0]) if score_row[0] is not None else None
    rank = int(score_row[1]) if score_row[1] is not None else None
    momentum = float(factor_row[0]) if factor_row and factor_row[0] is not None else None
    volatility = float(factor_row[1]) if factor_row and factor_row[1] is not None else None

    return {
        "date": score_date,
        "composite_score": composite,
        "rank": rank,
        "momentum_score": momentum,
        "volatility_score": volatility,
    }


def _get_score_trend(stock_id: int, latest_date: date) -> Optional[str]:
    """
    Compare latest composite to ~3 months ago. Return improving/declining/stable.
    Uses scores in [latest - 90d, latest]; compares latest to oldest in window.
    """
    cutoff = latest_date - timedelta(days=SCORE_TREND_LOOKBACK_DAYS)
    with get_db_context() as db:
        stmt = (
            select(Score.date, Score.composite_score)
            .where(
                Score.stock_id == stock_id,
                Score.date >= cutoff,
                Score.date <= latest_date,
            )
            .order_by(Score.date)
        )
        rows = db.execute(stmt).all()

    if not rows or len(rows) < 2:
        return "stable"

    dates_sorted = sorted(rows, key=lambda r: r[0])
    old_composite = float(dates_sorted[0][1]) if dates_sorted[0][1] is not None else 0.0
    new_composite = float(dates_sorted[-1][1]) if dates_sorted[-1][1] is not None else 0.0
    change = new_composite - old_composite

    if change >= TREND_IMPROVING_THRESHOLD:
        return "improving"
    if change <= TREND_DECLINING_THRESHOLD:
        return "declining"
    return "stable"


def _get_total_stocks_on_date(as_of_date: date) -> int:
    """Count distinct stocks with a score on the given date."""
    with get_db_context() as db:
        stmt = select(Score.stock_id).where(Score.date == as_of_date).distinct()
        n = len(db.execute(stmt).all())
    return n


def _signal_strength_from_rank(rank: Optional[int], n_stocks: int) -> int:
    """Map rank to 1-5 signal strength (1=weak, 5=strong). Best rank -> 5."""
    if rank is None or n_stocks <= 0:
        return 3
    # Quintile: top 20% -> 5, next 20% -> 4, ...
    percentile = 1.0 - (rank - 1) / max(1, n_stocks)
    return min(5, max(1, 1 + int(percentile * 5)))


def _regime_from_volatility(volatility_score: Optional[float]) -> str:
    """Classify regime from z-scored volatility_score."""
    if volatility_score is None:
        return "low_vol"
    return "high_vol" if volatility_score > REGIME_VOL_CUTOFF else "low_vol"


def _recommendation_text(signal_strength: int, trend: str, regime: str) -> str:
    """Rule-based short recommendation."""
    if signal_strength >= 4 and trend == "improving":
        return "Buy"
    if signal_strength >= 4 and trend in ("stable", "declining"):
        return "Hold"
    if signal_strength == 3:
        return "Hold"
    if signal_strength <= 2 and trend == "declining":
        return "Reduce"
    if signal_strength <= 2:
        return "Hold"
    return "Hold"


def _get_dynamic_stock_signal(symbol: str) -> Optional[dict[str, Any]]:
    """Fallback signal generator for stocks not in the local database.
    Fetches daily price data directly and computes approximate z-scores.
    """
    from services.data_fetcher import fetch_price_data
    df = fetch_price_data(symbol)
    if df.empty or len(df) < 126:
        return None
        
    df = df.set_index("date")
    close = df["close"].astype(float)
    daily_ret = close.pct_change().dropna(how="all")
    
    # Raw factors
    m6 = (close.iloc[-1] / close.iloc[-126]) - 1.0 if len(close) >= 126 else 0.0
    m12 = (close.iloc[-1] / close.iloc[-252]) - 1.0 if len(close) >= 252 else m6
    import math
    import pandas as pd
    try:
        volatility = daily_ret.rolling(252, min_periods=126).std().iloc[-1] * math.sqrt(252)
    except Exception:
        volatility = 0.25
        
    # Crude approximation of z-scores
    z_6m = (m6 - 0.075) / 0.15
    z_12m = (m12 - 0.15) / 0.20
    z_vol = (volatility - 0.25) / 0.15
    
    momentum_score = (z_6m + z_12m) / 2.0
    vol_score = -1.0 * z_vol
    
    # Calculate composite score from FACTOR_WEIGHTS (quality, growth, value assumed 0 for simplicity)
    composite = 0.25 * momentum_score  # momentum weight is 0.25
    
    # Trend over 3 months
    m6_past = (close.iloc[-63] / close.iloc[-189]) - 1.0 if len(close) >= 189 else 0.0
    m12_past = (close.iloc[-63] / close.iloc[-315]) - 1.0 if len(close) >= 315 else m6_past
    z_6m_past = (m6_past - 0.075) / 0.15
    z_12m_past = (m12_past - 0.15) / 0.20
    composite_past = 0.25 * ((z_6m_past + z_12m_past) / 2.0)
    
    if composite > composite_past + 0.1:
        trend = "improving"
    elif composite < composite_past - 0.1:
        trend = "declining"
    else:
        trend = "stable"
        
    # Approximate rank (using rough percentile derived from composite score)
    pctile = max(0.01, min(0.99, 0.5 + (composite / 3.0)))
    rank = int((1.0 - pctile) * 500)
    
    signal_strength = _signal_strength_from_rank(rank, 500)
    regime = _regime_from_volatility(vol_score)
    recommendation = _recommendation_text(signal_strength, trend, regime)
    
    # Stop loss Engine compatibility
    from services.stop_loss_engine import compute_stop_loss_scores
    price_df = pd.DataFrame({'DYNAMIC': close.tail(252)})
    try:
        sls_df = compute_stop_loss_scores(price_df)
        if not sls_df.empty and 'stop_loss_score' in sls_df.columns:
            sls_score = sls_df.loc['DYNAMIC', 'stop_loss_score']
            sls_action = sls_df.loc['DYNAMIC', 'recommended_action']
        else:
            sls_score = 50.0
            sls_action = "Hold"
    except Exception:
        sls_score = 50.0
        sls_action = "Hold"
        
    # Hardcode RRC for dynamic stocks, or use crude approximation
    rrc_score = 50.0

    return {
        "symbol": symbol + " (On-the-fly)",
        "composite_score": composite,
        "momentum_score": momentum_score,
        "volatility_score": vol_score,
        "rank": rank,
        "trend_direction": trend,
        "regime": regime,
        "signal_strength": signal_strength,
        "recommendation": recommendation,
        "stop_loss_score": sls_score,
        "stop_loss_action": sls_action,
        "rrc_score": rrc_score,
    }

def get_stock_signal(symbol: str) -> Optional[dict[str, Any]]:
    """
    Load latest composite, momentum, volatility, rank; 3-month trend; regime.
    Return signal dict or None if symbol/data missing.
    """
    stock_id = _get_stock_id(symbol)
    if stock_id is None:
        # Stock is not in the database! Run an on-the-fly dynamic estimation.
        return _get_dynamic_stock_signal(symbol)

    latest = _get_latest_score_and_factor(stock_id)
    if latest is None:
        return None

    trend = _get_score_trend(stock_id, latest["date"])
    n_stocks = _get_total_stocks_on_date(latest["date"])
    signal_strength = _signal_strength_from_rank(latest["rank"], n_stocks)
    regime = _regime_from_volatility(latest.get("volatility_score"))
    recommendation = _recommendation_text(signal_strength, trend, regime)

    # ── Stop-Loss Score Integration ───────────────────────────────────────────
    from services.stop_loss_engine import analyze_stock_stop_loss
    try:
        sls_result = analyze_stock_stop_loss(latest["date"], stock_id)
        sls_score = sls_result["stop_loss_score"]
        sls_action = sls_result["recommended_action"]
    except Exception:
        sls_score = 50.0
        sls_action = "Hold"
        
    # ── Risk Responsiveness Component ─────────────────────────────────────────
    from services.risk_responsiveness import compute_stock_rrc
    try:
        rrc_score = compute_stock_rrc(stock_id, latest["date"])
    except Exception:
        rrc_score = 50.0

    return {
        "symbol": symbol,
        "composite_score": latest["composite_score"],
        "momentum_score": latest.get("momentum_score"),
        "volatility_score": latest.get("volatility_score"),
        "rank": latest["rank"],
        "trend_direction": trend,
        "regime": regime,
        "signal_strength": signal_strength,
        "recommendation": recommendation,
        "stop_loss_score": sls_score,
        "stop_loss_action": sls_action,
    }


# Risk profiles: (min_pct, max_pct) suggested allocation per name
ALLOCATION_BY_RISK = {
    "conservative": (0.0, 0.05),   # 0–5% per name
    "moderate": (0.0, 0.10),       # 0–10%
    "aggressive": (0.0, 0.15),     # 0–15%
}
DEFAULT_RISK_PROFILE = "moderate"


def generate_recommendation(
    signal_dict: dict[str, Any],
    risk_profile: str = DEFAULT_RISK_PROFILE,
) -> dict[str, Any]:
    """
    From signal dict and risk profile, return recommendation, conviction,
    suggested_allocation_range, risk_commentary, explanation. Rule-based.
    """
    strength = signal_dict.get("signal_strength", 3)
    trend = signal_dict.get("trend_direction", "stable")
    regime = signal_dict.get("regime", "low_vol")
    rec = signal_dict.get("recommendation", "Hold")

    # ── Stop-Loss & RRC Overrides & Final Scoring ─────────────────────────────
    sls_score = signal_dict.get("stop_loss_score", 50.0)
    sls_action = signal_dict.get("stop_loss_action", "Hold")
    comp_score = signal_dict.get("composite_score", 0.0)
    rrc_score = signal_dict.get("rrc_score", 50.0)
    
    # Simple normalizer for the 1-100 typical range (clip 0-1)
    comp_norm = min(max((comp_score if comp_score is not None else 50.0) / 100.0, 0.0), 1.0)
    sls_norm = min(max(sls_score / 100.0, 0.0), 1.0)
    rrc_norm = min(max(rrc_score / 100.0, 0.0), 1.0)
    
    final_stock_score = 0.6 * comp_norm - 0.3 * sls_norm + 0.1 * rrc_norm

    # Override base Recommendation based on Risk Responsiveness
    if rrc_score < 20:
        if rec in ["Buy", "Hold"]:
            rec = "Reduce"
            strength = min(strength, 2)
    elif rrc_score > 80:
        if rec == "Exit": rec = "Reduce"
        elif rec == "Reduce": rec = "Hold"
        elif rec == "Hold": rec = "Buy"

    # Override base Recommendation based on Stop-Loss Risk
    if sls_score > 80:
        rec = "Exit"
        strength = 1 # Force lowest strength
    elif sls_score > 60:
        if rec == "Buy":
            rec = "Hold"
        elif rec == "Hold":
            rec = "Reduce"
        strength = min(strength, 2)
        
    # Conviction from signal_strength, trend and sls
    if strength >= 4 and trend == "improving" and sls_score < 40:
        conviction = "High"
    elif strength >= 3 and trend != "declining" and sls_score < 60:
        conviction = "Medium"
    else:
        conviction = "Low"

    # Allocation range from risk profile
    alloc = ALLOCATION_BY_RISK.get(
        risk_profile.lower() if isinstance(risk_profile, str) else "",
        ALLOCATION_BY_RISK[DEFAULT_RISK_PROFILE],
    )
    # Scale by signal_strength: stronger signal -> use upper part of range
    min_pct, max_pct = alloc
    if strength <= 2:
        suggested_max = min_pct + (max_pct - min_pct) * 0.4
    elif strength == 3:
        suggested_max = min_pct + (max_pct - min_pct) * 0.7
    else:
        suggested_max = max_pct
        
    if sls_score > 80:
        suggested_allocation_range = (0.0, 0.0)
    else:
        suggested_allocation_range = (min_pct, suggested_max)

    # Risk commentary
    risk_comments = []
    if regime == "high_vol":
        risk_comments.append("Stock is in a high-volatility regime.")
    if sls_score > 80:
        risk_comments.append("CRITICAL DOWNSIDE RISK (SLS > 80). Immediate exit suggested.")
    elif sls_score > 60:
        risk_comments.append(f"Elevated downside risk detected (SLS: {sls_score:.1f}). Consider tightening stops or reducing size.")
    
    if not risk_comments:
        risk_commentary = "Volatility regime is low; standard position sizing applies."
    else:
        risk_commentary = " ".join(risk_comments)

    # Explanation
    parts = [
        f"Composite score rank {signal_dict.get('rank', 'N/A')}, trend {trend}.",
        f"Signal strength {strength}/5, conviction {conviction}.",
        f"Stop-Loss Engine Score: {sls_score:.1f}/100."
    ]
    if regime == "high_vol":
        parts.append("High-vol regime suggests caution on size.")
    explanation = " ".join(parts)

    return {
        "recommendation": rec,
        "conviction": conviction,
        "suggested_allocation_range": suggested_allocation_range,
        "risk_commentary": risk_commentary,
        "explanation": explanation,
        "final_stock_score": final_stock_score,
        "stop_loss_score": sls_score,
    }


def _run_validation() -> None:
    """Print validation output when run standalone.

    Uses `RECOMMENDATION_VALIDATION_LIMIT` env var (default defined by
    `DEFAULT_VALIDATION_LIMIT`) to limit the number of symbols fetched from DB.
    """
    print("\n" + "=" * 60)
    print("RECOMMENDATION ENGINE – validation")
    print("=" * 60)

    # Get symbols from DB (limited by env-configured default)
    with get_db_context() as db:
        symbols = [r[0] for r in db.execute(select(Stock.symbol).limit(DEFAULT_VALIDATION_LIMIT)).all()]

    if not symbols:
        print("No stocks in DB. Run data_fetcher and scoring first.")
        print("=" * 60 + "\n")
        return

    for symbol in symbols[:3]:
        signal = get_stock_signal(symbol)
        if signal is None:
            print(f"\n{symbol}: no signal (missing data)")
            continue
        print(f"\n--- Signal: {symbol} ---")
        for k, v in signal.items():
            print(f"  {k}: {v}")

        rec = generate_recommendation(signal, risk_profile="moderate")
        print(f"  --- Recommendation (moderate risk) ---")
        for k, v in rec.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    _run_validation()
