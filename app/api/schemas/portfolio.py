from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class PortfolioResponse(BaseModel):
    portfolio_type: str
    weights: Dict[str, float]
    metrics: Dict[str, float]
    composite_rating: float

class HybridPortfolioResponse(BaseModel):
    selected_stocks: List[str]
    weights: Dict[str, float]
    optimal_n_used: Any
    expected_metrics: Dict[str, float]
    stop_loss_summary: Dict[str, Any]

class MetaPortfolioResponse(BaseModel):
    portfolio_weights: Dict[str, float]
    component_portfolios: List[str]
    composite_rating: float
    stability_score: float

class UserPortfolioDef(BaseModel):
    name: str
    symbols: List[str]
    strategy: str = "equal_weight"
    regime_mode: str = "static"
    top_n: int = 5

class PortfolioCompareRequest(BaseModel):
    portfolios: List[UserPortfolioDef]
    start_date: str
    end_date: str
    use_multiprocessing: bool = False

class PortfolioHistoryResponse(BaseModel):
    monthly_allocations: List[Dict[str, Any]]
    metrics_history: Dict[str, float]
    stability_trend: List[Dict[str, Any]]

class PortfolioCompareResponse(BaseModel):
    comparison_table: List[Dict[str, Any]]
    ratings: Dict[str, Any]
    correlation_matrix: Dict[str, Dict[str, float]]
    meta_portfolio_suggestion: Optional[MetaPortfolioResponse]
