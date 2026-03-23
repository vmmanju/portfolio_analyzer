STOCK_EXPLANATION_TEMPLATE = """Stock: {symbol}
Composite Score: {composite_score} (Rank: {rank})
Stop-Loss Score: {stop_loss_score}
Risk Responsiveness (RRC): {rrc}
Regime: {current_regime}
Recommendation: {recommendation}
"""

PORTFOLIO_EXPLANATION_TEMPLATE = """Portfolio Type: {portfolio_type}
CAGR: {cagr}
Sharpe: {sharpe}
Stability: {stability}
Composite Rating: {composite_rating}
Governance Score: {governance_score}
"""

RESEARCH_EXPLANATION_TEMPLATE = """Overfitting Flags: {overfitting_flags}
Conservative Bias: {conservative_bias}
Calibration Drift: {calibration_drift}
"""

RISK_EXPLANATION_TEMPLATE = """Risk / Stop-Loss Diagnostics:
Stop-Loss Score: {stop_loss_score} (Level: {risk_level})
Raw Drawdown: {raw_drawdown}
Trigger Price: {trigger}

Component Stress:
- Peak Distance     : {peak_distance_score}
- Momentum Breakdown: {momentum_breakdown_score}
- Volatility Adj    : {volatility_adjusted_score}
- Regime Stress     : {regime_stress_score}
- Error Bias        : {error_bias_score}
"""

# Generic fallback template
GENERIC_EXPLANATION_TEMPLATE = """Context: 
{context}
"""
