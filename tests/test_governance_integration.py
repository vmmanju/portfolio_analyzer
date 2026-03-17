import unittest
from unittest.mock import patch, MagicMock
from datetime import date
import pandas as pd
from services.auto_diversified_portfolio import STRATEGY_AUTO_HYBRID
from services.model_governance import run_overfitting_diagnostics

class TestGovernanceIntegration(unittest.TestCase):
    
    def test_governance_with_hybrid_metrics(self):
        # Semi-realistic portfolio results from a hybrid backtest
        portfolio_results = {
            "metrics": {
                "Sharpe": 1.5,
                "Volatility": 0.12,
                "monthly_turnover": 0.20,
                "CAGR": 0.18
            },
            "ratings": {
                "stability_score": 75.0,
                "composite_score": 82.0
            },
            "avg_pairwise_corr": 0.25,
            "rrc_score": 65.0
        }
        
        # Test overfitting diagnostics
        overfit = run_overfitting_diagnostics(portfolio_results)
        self.assertEqual(overfit["overfitting_risk"], "Low")
        self.assertEqual(len(overfit["flags"]), 0)
        
        # Simulate realistic "risky" results
        risky_results = {
            "metrics": {
                "Sharpe": 3.5, # Too high
                "monthly_turnover": 0.70 # Too high
            }
        }
        overfit_risky = run_overfitting_diagnostics(risky_results)
        self.assertEqual(overfit_risky["overfitting_risk"], "High")
        self.assertIn("Extreme Sharpe (>3)", overfit_risky["flags"][0])

if __name__ == "__main__":
    unittest.main()
