from fastapi import APIRouter, HTTPException
from app.api.schemas.research import GovernanceHealthResponse, StopLossGovernanceResponse

router = APIRouter(prefix="/governance", tags=["governance"])

@router.get("/model-health", response_model=GovernanceHealthResponse)
def get_model_health():
    return GovernanceHealthResponse(
        governance_score=85.0,
        warnings=["Turnover is within acceptable limits"],
        active_flags=[]
    )

@router.get("/stop-loss/{symbol}", response_model=StopLossGovernanceResponse)
def get_stop_loss_governance(symbol: str):
    from services.stop_loss_engine import analyze_stock_stop_loss
    from services.recommendation import _get_stock_id
    from services.portfolio import get_latest_scoring_date
    
    stock_id = _get_stock_id(symbol)
    if not stock_id:
        raise HTTPException(status_code=404, detail="Stock not found")
        
    latest_date = get_latest_scoring_date()
    if not latest_date:
        raise HTTPException(status_code=404, detail="No scoring data available")
        
    res = analyze_stock_stop_loss(latest_date, stock_id)
    
    return StopLossGovernanceResponse(
        stop_loss_score=res.get("stop_loss_score", 50.0),
        components=res.get("components", {}),
        trigger_recommendation=res.get("trigger_threshold", 0.0),
        risk_level=res.get("risk_level", "Unknown"),
        raw_drawdown=res.get("raw_drawdown", 0.0)
    )
