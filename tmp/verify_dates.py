import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database import get_db_context
from app.models import Factor, Score, Price
from sqlalchemy import func, select

with get_db_context() as db:
    max_price_date = db.execute(select(func.max(Price.date))).scalar()
    max_score_date = db.execute(select(func.max(Score.date))).scalar()
    
print(f"Max Price date: {max_price_date}")
print(f"Max Score date: {max_score_date}")
