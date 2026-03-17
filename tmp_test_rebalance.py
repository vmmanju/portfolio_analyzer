import logging
from datetime import date
from services.auto_diversified_portfolio import build_diversified_hybrid_portfolio
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
_logger = logging.getLogger("test_rebalance")

def run_test():
    test_date = date(2026, 1, 30)
    _logger.info("Running manual rebalance test for %s...", test_date)
    try:
        result = build_diversified_hybrid_portfolio(
            as_of_date=test_date,
            top_n="auto", # Test the full N-selection path
            risk_profile="medium",
            regime_mode="volatility_targeting"
        )
        _logger.info("Rebalance result: %s", result.get("selected_stocks", []))
        _logger.info("Optimal N: %s", result.get("optimal_n"))
        _logger.info("Warnings: %s", result.get("warnings", []))
    except Exception as e:
        _logger.exception("Rebalance failed with error: %s", e)

if __name__ == "__main__":
    run_test()
