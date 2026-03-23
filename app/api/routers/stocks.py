from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import Optional, List
import pandas as pd
from datetime import date

from app.api.dependencies import get_db
from app.api.schemas.stock import StockResponse, StockResearchResponse, StockListResponse
from services.recommendation import get_stock_signal, generate_recommendation
from services.portfolio import load_ranked_stocks, get_latest_scoring_date
from app.models import Stock, Factor, Score

router = APIRouter(prefix="/stocks", tags=["stocks"])

@router.get("/{symbol}", response_model=StockResponse)
def get_stock(symbol: str):
    signal = get_stock_signal(symbol)
    if not signal:
        raise HTTPException(status_code=404, detail="Stock not found or data missing")
    
    rec = generate_recommendation(signal, risk_profile="moderate")
    
    return StockResponse(
        symbol=signal["symbol"],
        composite_score=signal["composite_score"],
        rank=signal["rank"],
        momentum=signal.get("momentum_score", 0.0) or 0.0,
        volatility=signal.get("volatility_score", 0.0) or 0.0,
        stop_loss_score=signal.get("stop_loss_score", 50.0),
        rrc=signal.get("rrc_score", 50.0),
        recommendation=rec["recommendation"],
        suggested_allocation=rec["suggested_allocation_range"],
        current_regime=signal["regime"]
    )

@router.get("/{symbol}/research", response_model=StockResearchResponse)
def get_stock_research(symbol: str, db: Session = Depends(get_db)):
    stock = db.execute(select(Stock).where(Stock.symbol == symbol)).scalars().first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")
        
    latest_date = get_latest_scoring_date()
    if not latest_date:
        raise HTTPException(status_code=404, detail="No scoring data available")
        
    factor_row = db.execute(select(Factor).where(Factor.stock_id == stock.id, Factor.date == latest_date)).scalars().first()
    factor_breakdown = {}
    if factor_row:
        factor_breakdown = {
            "quality": factor_row.quality_score or 0.0,
            "value": factor_row.value_score or 0.0,
            "momentum": factor_row.momentum_score or 0.0,
            "growth": factor_row.growth_score or 0.0,
            "volatility": factor_row.volatility_score or 0.0
        }
        
    scores = db.execute(
        select(Score)
        .where(Score.stock_id == stock.id)
        .order_by(Score.date.desc())
        .limit(126)
    ).scalars().all()
    
    score_trend = [{"date": s.date.isoformat(), "composite": s.composite_score} for s in reversed(scores)]
    
    return StockResearchResponse(
        symbol=symbol,
        factor_breakdown=factor_breakdown,
        score_trend=score_trend,
        error_coefficients={}, 
        governance_diagnostics={},
        calibration_date=latest_date.isoformat()
    )

@router.get("", response_model=StockListResponse)
def list_stocks(
    top_n: Optional[int] = Query(50, ge=1),
    min_score: Optional[float] = Query(None, ge=0.0, le=100.0),
    sector: Optional[str] = None,
    regime: Optional[str] = None
):
    # Handle FastAPI Query objects if called internally without dependency injection
    top_n_val = top_n.default if hasattr(top_n, "default") else top_n
    min_score_val = min_score.default if hasattr(min_score, "default") else min_score
    
    latest_date = get_latest_scoring_date()
    if not latest_date:
        return StockListResponse(stocks=[], total_count=0)
        
    df = load_ranked_stocks(latest_date)
    if df.empty:
        return StockListResponse(stocks=[], total_count=0)
        
    if sector:
        df = df[df["sector"].str.lower() == sector.lower()]
    if min_score_val is not None:
        df = df[df["composite_score"] >= min_score_val]
        
    df = df.head(top_n_val)
    
    stocks = []
    for _, row in df.iterrows():
        # Get simplified signal for lists
        stocks.append(
            StockResponse(
                symbol=row["symbol"],
                composite_score=row["composite_score"],
                rank=row["rank"],
                momentum=0.0, 
                volatility=row.get("volatility_score", 0.0) or 0.0,
                stop_loss_score=50.0,
                rrc=50.0,
                recommendation="Hold",
                suggested_allocation=(0.0, 0.0),
                current_regime="low_vol"
            )
        )
        
    return StockListResponse(stocks=stocks, total_count=len(stocks))
