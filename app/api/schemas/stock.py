from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Tuple

class StockResponse(BaseModel):
    symbol: str
    composite_score: float
    rank: int
    momentum: float
    volatility: float
    stop_loss_score: float
    rrc: float
    recommendation: str
    suggested_allocation: Tuple[float, float]
    current_regime: str

class StockResearchResponse(BaseModel):
    symbol: str
    factor_breakdown: Dict[str, float]
    score_trend: List[Dict[str, Any]]
    error_coefficients: Dict[str, float]
    governance_diagnostics: Dict[str, Any]
    calibration_date: Optional[str]

class StockListResponse(BaseModel):
    stocks: List[StockResponse]
    total_count: int
