from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List
from datetime import date
import pandas as pd

from app.api.schemas.portfolio import (
    HybridPortfolioResponse, MetaPortfolioResponse, 
    PortfolioCompareRequest, PortfolioCompareResponse,
    PortfolioHistoryResponse, PortfolioResponse
)
from services.auto_diversified_portfolio import STRATEGY_AUTO_HYBRID
from services import portfolio_comparison as pc
from services.portfolio import get_latest_scoring_date

router = APIRouter(prefix="/portfolio", tags=["portfolios"])

@router.get("/hybrid", response_model=HybridPortfolioResponse)
def get_hybrid_portfolio(
    start_date: str,
    end_date: str,
    risk_profile: str = "moderate",
    top_n: str = "auto"
):
    try:
        s_date = date.fromisoformat(start_date)
        e_date = date.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")

    p = pc.UserPortfolio(
        name="Auto Hybrid",
        symbols=[],
        strategy=STRATEGY_AUTO_HYBRID,
        regime_mode="volatility_targeting",
        top_n="auto" if top_n.lower() == "auto" else int(top_n)
    )
    
    results = pc.backtest_user_portfolios([p], s_date, e_date, use_multiprocessing=False)
    if not results or "Auto Hybrid" not in results:
        raise HTTPException(status_code=500, detail="Backtest failed for hybrid portfolio")
        
    res = results["Auto Hybrid"]
    metrics = res.get("metrics", {})
    weights_series = res.get("final_weights", pd.Series())
    
    return HybridPortfolioResponse(
        selected_stocks=list(weights_series.index),
        weights=weights_series.to_dict(),
        optimal_n_used=metrics.get("optimal_n", p.top_n),
        expected_metrics=metrics,
        stop_loss_summary={}
    )

@router.get("/meta", response_model=MetaPortfolioResponse)
def get_meta_portfolio():
    # Stub: Meta portfolio blending service should be called here
    return MetaPortfolioResponse(
        portfolio_weights={"Current Model": 0.5, "Auto Hybrid": 0.5},
        component_portfolios=["Current Model", "Auto Hybrid"],
        composite_rating=85.5,
        stability_score=90.0
    )

@router.post("/compare", response_model=PortfolioCompareResponse)
def compare_portfolios(req: PortfolioCompareRequest):
    try:
        s_date = date.fromisoformat(req.start_date)
        e_date = date.fromisoformat(req.end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")
        
    ups = []
    for p in req.portfolios:
        ups.append(
            pc.UserPortfolio(
                name=p.name,
                symbols=p.symbols,
                strategy=p.strategy,
                regime_mode=p.regime_mode,
                top_n=p.top_n
            )
        )
        
    results = pc.backtest_user_portfolios(ups, s_date, e_date, use_multiprocessing=req.use_multiprocessing)
    full_r = pc.compute_full_ratings(results, meta_result=None)
    ratings = full_r["ratings"]
    
    comparison_table = []
    for name, v in results.items():
        m = v.get("metrics", {})
        r = ratings.get(name, {})
        comparison_table.append({
            "name": name,
            "cagr": m.get("CAGR", 0.0),
            "sharpe": m.get("Sharpe", 0.0),
            "max_drawdown": m.get("Max Drawdown", 0.0),
            "composite_score": r.get("composite_score", 0.0),
            "grade": r.get("grade", "N/A"),
            "rank": r.get("rank", 0)
        })
        
    # Placeholder for correlation matrix
    corr_matrix = {}
    if len(ups) > 1:
        corr_matrix = {ups[0].name: {ups[1].name: 0.85}, ups[1].name: {ups[0].name: 0.85}}
        
    return PortfolioCompareResponse(
        comparison_table=comparison_table,
        ratings=ratings,
        correlation_matrix=corr_matrix,
        meta_portfolio_suggestion=None
    )

@router.get("/{portfolio_name}/history", response_model=PortfolioHistoryResponse)
def get_portfolio_history(portfolio_name: str):
    return PortfolioHistoryResponse(
        monthly_allocations=[],
        metrics_history={"CAGR": 0.15, "Sharpe": 1.2},
        stability_trend=[]
    )
