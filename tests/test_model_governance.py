from services.model_governance import run_overfitting_diagnostics, run_conservative_bias_check, compute_model_governance_score

def test_overfitting_diagnostics_high_risk():
    results = {
        "metrics": {
            "Sharpe": 4.5,
            "monthly_turnover": 0.75
        },
        "error_r2": 0.9,
        "coefficient_flips": 5
    }
    diag = run_overfitting_diagnostics(results)
    assert diag["overfitting_risk"] == "High"
    assert len(diag["flags"]) >= 4
    return True

def test_overfitting_diagnostics_low_risk():
    results = {
        "metrics": {
            "Sharpe": 1.2,
            "monthly_turnover": 0.15
        }
    }
    diag = run_overfitting_diagnostics(results)
    assert diag["overfitting_risk"] == "Low"
    assert len(diag["flags"]) == 0
    return True

def test_conservative_bias_high():
    results = {
        "metrics": {
            "Volatility": 0.05,
            "Beta": 0.2,
            "Sharpe": 1.1,
            "CAGR": 0.03
        }
    }
    diag = run_conservative_bias_check(results)
    assert diag["conservative_bias"] is True
    assert diag["classification"] == "Overly Conservative Model"
    return True

def test_conservative_bias_balanced():
    results = {
        "metrics": {
            "Volatility": 0.15,
            "Beta": 0.8,
            "Sharpe": 1.1,
            "CAGR": 0.12
        }
    }
    diag = run_conservative_bias_check(results)
    assert diag["conservative_bias"] is False
    assert diag["classification"] == "Balanced"
    return True

def test_governance_score():
    overfit = {"penalty_points": 5}
    conservative = {"penalty_points": 3}
    stability = 80.0
    score = compute_model_governance_score(overfit, conservative, stability)
    # 100 - (5 + 3 + (20 - 16)) = 100 - 12 = 88
    assert score == 88.0
    return True
