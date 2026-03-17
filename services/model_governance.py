"""
Phase 10: Model Governance & Integrity Checking
Implements conservative modeling and overfitting detection.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

def run_overfitting_diagnostics(portfolio_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run diagnostic checks to detect overfitting in quantitative models.
    
    Tests:
    - In-Sample vs Out-of-Sample Gap
    - Sensitivity Instability 
    - High R² in Error Model
    - Coefficient Sign Instability
    - Excessive Turnover
    - Extreme Sharpe
    """
    flags = []
    diagnostics = {}
    overfit_score = 0
    total_checks = 0

    metrics = portfolio_results.get("metrics", {})
    
    # 1. Extreme Sharpe
    sharpe = metrics.get("Sharpe", 0.0)
    diagnostics["sharpe"] = sharpe
    if sharpe > 3.0:
        flags.append("Extreme Sharpe (>3) – potential curve fitting")
        overfit_score += 1
    total_checks += 1

    # 2. Excessive Turnover
    turnover = metrics.get("monthly_turnover", 0.0)
    diagnostics["turnover"] = turnover
    if turnover > 0.60:
        flags.append(f"Excessive turnover ({turnover*100:.1f}%)")
        overfit_score += 1
    total_checks += 1

    # 3. In-Sample vs Out-of-Sample Gap (if walk-forward data provided)
    wf_data = portfolio_results.get("walk_forward")
    if wf_data is not None and not wf_data.empty and "train_Sharpe" in wf_data.columns and "test_Sharpe" in wf_data.columns:
        gap = (wf_data["train_Sharpe"] - wf_data["test_Sharpe"]).mean()
        diagnostics["wf_sharpe_gap"] = gap
        if gap > 1.0:
            flags.append(f"Train/Test Sharpe gap > 1.0 ({gap:.2f})")
            overfit_score += 1
        total_checks += 1

    # 4. Sensitivity Instability (if sensitivity data provided)
    sens_data = portfolio_results.get("sensitivity")
    if sens_data is not None and not sens_data.empty and "Sharpe" in sens_data.columns:
        sharpes = sens_data["Sharpe"].values
        max_s = sharpes.max()
        min_s = sharpes.min()
        if max_s > 0:
            change_pct = (max_s - min_s) / abs(max_s)
            diagnostics["sensitivity_sharpe_change"] = change_pct
            if change_pct > 0.25:
                flags.append(f"High sensitivity to weight changes ({change_pct*100:.1f}%)")
                overfit_score += 1
        total_checks += 1

    # 5. Error Model R-squared
    error_r2 = portfolio_results.get("error_r2", 0.0)
    diagnostics["error_r2"] = error_r2
    if error_r2 > 0.8:
        flags.append(f"Suspiciously high Error Model R² ({error_r2:.2f})")
        overfit_score += 1
    total_checks += 1

    # 6. Coefficient Sign Instability
    sign_flips = portfolio_results.get("coefficient_flips", 0)
    diagnostics["coefficient_flips"] = sign_flips
    if sign_flips > 3: # Arbitrary threshold for "frequent"
        flags.append(f"Frequent coefficient sign flips ({sign_flips} flips)")
        overfit_score += 1
    total_checks += 1

    if total_checks == 0:
        risk_level = "Unknown"
    else:
        ratio = overfit_score / total_checks
        if ratio >= 0.5:
            risk_level = "High"
        elif ratio >= 0.2:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

    return {
        "overfitting_risk": risk_level,
        "flags": flags,
        "diagnostics": diagnostics,
        "penalty_points": overfit_score * 5
    }

def run_conservative_bias_check(portfolio_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run diagnostics to detect overly conservative modeling.
    
    Tests:
    - Low Beta (<0.3)
    - Volatility too low (<8%)
    - Underperformance in bull regime
    - Excessive cash equivalence
    - Sharpe moderate but CAGR suppressed
    """
    flags = []
    diagnostics = {}
    conservative_score = 0
    total_checks = 0

    metrics = portfolio_results.get("metrics", {})
    
    # 1. Volatility too low
    vol = metrics.get("Volatility", 0.0)
    diagnostics["volatility"] = vol
    if vol > 0 and vol < 0.08:
        flags.append(f"Suspiciously low volatility ({vol*100:.1f}%)")
        conservative_score += 1
    total_checks += 1

    # 2. Low Beta
    beta = metrics.get("Beta", 1.0)
    diagnostics["beta"] = beta
    if beta < 0.3:
        flags.append(f"Sustained low beta ({beta:.2f})")
        conservative_score += 1
    total_checks += 1

    # 3. Suppressed CAGR vs Sharpe (moderate sharpe, very low CAGR)
    sharpe = metrics.get("Sharpe", 0.0)
    cagr = metrics.get("CAGR", 0.0)
    diagnostics["sharpe"] = sharpe
    diagnostics["cagr"] = cagr
    if sharpe > 0.8 and cagr < 0.05:
        flags.append(f"Sharpe is good ({sharpe:.2f}) but CAGR is heavily suppressed ({cagr*100:.1f}%)")
        conservative_score += 1
    total_checks += 1

    # 4. Underperformance in bull regime
    regime = portfolio_results.get("regime", {})
    bull_cagr = regime.get("cagr_low_vol", cagr)  # Using low vol (stable) as proxy for bull/calm
    diagnostics["bull_cagr"] = bull_cagr
    if bull_cagr < 0.08:
        flags.append(f"Underperformance in stable regimes ({bull_cagr*100:.1f}%)")
        conservative_score += 1
    total_checks += 1

    impact_score = (conservative_score / max(1, total_checks)) * 100
    bias_bool = conservative_score >= 3

    if bias_bool:
        classification = "Overly Conservative Model"
    elif conservative_score >= 1:
        classification = "Slightly Conservative"
    else:
        classification = "Balanced"

    return {
        "conservative_bias": bias_bool,
        "classification": classification,
        "impact_score": impact_score,
        "flags": flags,
        "diagnostics": diagnostics,
        "penalty_points": conservative_score * 3
    }


def compute_model_governance_score(overfit_dict: Dict[str, Any], conservative_dict: Dict[str, Any], stability_score: float) -> float:
    """
    Calculate an overall model governance score (0-100).
    100 - (OverfitPenalty + ConservativePenalty + InstabilityPenalty)
    """
    o_penalty = overfit_dict.get("penalty_points", 0)
    c_penalty = conservative_dict.get("penalty_points", 0)
    
    # Stability score is from 0-100. Lower is worse. Max penalty for stability could be 20.
    s_penalty = max(0.0, 20.0 - (stability_score * 0.2)) 
    
    score = 100.0 - (o_penalty + c_penalty + s_penalty)
    return max(0.0, min(100.0, score))


LIGHTWEIGHT_METRICS = {
    "sharpe", "volatility", "drawdown", "composite score", "stability", "stop-loss score", "rrc"
}

def determine_analysis_type(metric_name: str) -> str:
    """
    Determine whether analysis is lightweight (auto-run) or heavy (user controlled).
    """
    name = str(metric_name).lower().strip()
    if name in LIGHTWEIGHT_METRICS:
        return "auto"
    return "user_controlled"
