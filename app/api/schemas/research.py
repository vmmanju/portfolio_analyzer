from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class GovernanceResponse(BaseModel):
    governance_score: float
    overfitting_diagnostics: Dict[str, float]
    conservative_bias: float

class AutoNResponse(BaseModel):
    optimal_n: Any
    benefit_curve: List[Dict[str, float]]
    diagnostics: Dict[str, Any]

class StabilityResponse(BaseModel):
    rolling_sharpe: List[Dict[str, Any]]
    rolling_drawdown: List[Dict[str, Any]]
    turnover: float
    correlation_drift: float

class CalibrationLatestResponse(BaseModel):
    current_coefficients: Dict[str, float]
    calibration_window_start: Optional[str]
    calibration_window_end: Optional[str]
    next_update_date: Optional[str]
    r_squared: float

class CalibrationHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]

class GovernanceHealthResponse(BaseModel):
    governance_score: float
    warnings: List[str]
    active_flags: List[str]

class StopLossGovernanceResponse(BaseModel):
    stop_loss_score: float
    components: Dict[str, float]
    trigger_recommendation: float
    risk_level: str
    raw_drawdown: float
