import sys
from pathlib import Path
sys.path.insert(0, str(Path("c:/Users/mmanj/OneDrive/Documents/manju/CursorPractice/portfolio_analyzer").resolve()))

from datetime import date
from services.portfolio_comparison import UserPortfolio, backtest_user_portfolios, compute_full_ratings

if __name__ == '__main__':
    p = UserPortfolio(name="Test", symbols=[], strategy="equal_weight", regime_mode="static", top_n=5)
    p2 = UserPortfolio(name="Test2", symbols=[], strategy="inverse_volatility", regime_mode="static", top_n=5)
    start = date(2023, 1, 1)
    end = date(2026, 2, 23)
    res = backtest_user_portfolios([p, p2], start, end, use_multiprocessing=False)
    
    rat = compute_full_ratings(res)
    print("RRC inside Test1 metrics:", res["Test"].get("rrc_score", "not found"))
    print("RRC inside Test2 metrics:", res["Test2"].get("rrc_score", "not found"))
    print("RRC inside rating output (Test):", rat["rated_df"][rat["rated_df"]["name"] == "Test"]["rrc_score"].values[0] if "rrc_score" in rat["rated_df"].columns else "not found column")
