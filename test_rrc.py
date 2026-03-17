import sys
from pathlib import Path
sys.path.insert(0, str(Path("c:/Users/mmanj/OneDrive/Documents/manju/CursorPractice/portfolio_analyzer").resolve()))

from datetime import date
from services.risk_responsiveness import compute_stock_rrc

print(compute_stock_rrc(1, date(2026, 2, 23)))
