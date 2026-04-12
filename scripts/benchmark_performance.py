import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import time
from datetime import date
from services.backtest import run_backtest, STRATEGY_EQUAL_WEIGHT

def benchmark_backtest():
    print("Starting benchmark for run_backtest...")
    start_time = time.time()
    
    # Run a standard backtest over 2 years
    start_date = date(2022, 1, 1)
    end_date = date(2024, 1, 1)
    
    equity_curve, summary = run_backtest(
        strategy=STRATEGY_EQUAL_WEIGHT,
        top_n=20,
        start_date=start_date,
        end_date=end_date
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Benchmark completed in {duration:.2f} seconds")
    if not equity_curve.empty:
        print(f"Equity curve rows: {len(equity_curve)}")
        print(f"CAGR: {summary.get('CAGR', 0):.4f}")
    else:
        print("No results returned. Check if DB has data for 2022-2024.")

if __name__ == "__main__":
    benchmark_backtest()
