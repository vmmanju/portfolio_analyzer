from fastapi import APIRouter, Depends, HTTPException
from datetime import date
import pandas as pd

from app.api.schemas.research import GovernanceResponse, AutoNResponse, StabilityResponse
from services.model_governance import run_overfitting_diagnostics, run_conservative_bias_check, compute_model_governance_score
from services.auto_n_selector import select_optimal_n

router = APIRouter(prefix="/research", tags=["research"])

@router.get("/governance", response_model=GovernanceResponse)
def get_governance():
    # Using placeholder inputs from previous known good backtest runs
    overfit = run_overfitting_diagnostics({"metrics": {"Sharpe": 1.5, "monthly_turnover": 0.2}})
    bias = run_conservative_bias_check({"metrics": {"Volatility": 0.12, "Beta": 0.8, "Sharpe": 1.5, "CAGR": 0.15}})
    score = compute_model_governance_score(overfit, bias, 80.0)
    
    return GovernanceResponse(
        governance_score=score,
        overfitting_diagnostics=overfit.get("diagnostics", {}),
        conservative_bias=bias.get("impact_score", 0.0)
    )

@router.get("/auto-n", response_model=AutoNResponse)
def get_auto_n(as_of_date: str = "2024-01-01"):
    try:
        s_date = date.fromisoformat(as_of_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
        
    res = select_optimal_n(s_date)
    if not res:
        raise HTTPException(status_code=500, detail="Auto N optimization failed")
        
    eval_df_records = res.get("eval_df", pd.DataFrame()).to_dict('records') if hasattr(res.get("eval_df"), 'to_dict') else []
    
    return AutoNResponse(
        optimal_n=res.get("optimal_n", 15),
        benefit_curve=eval_df_records,
        diagnostics=res.get("diagnostics", {})
    )

@router.get("/stability", response_model=StabilityResponse)
def get_stability():
    return StabilityResponse(
        rolling_sharpe=[],
        rolling_drawdown=[],
        turnover=0.15,
        correlation_drift=0.05
    )
