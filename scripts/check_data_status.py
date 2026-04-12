import os
import sys
from pathlib import Path

# Add project root to sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database import get_db_context
from app.models import Price, Score
from sqlalchemy import select, func
from services.portfolio import get_latest_scoring_date

def check_db():
    try:
        with get_db_context() as db:
            p_count = db.execute(select(func.count(Price.id))).scalar()
            s_count = db.execute(select(func.count(Score.id))).scalar()
            p_latest = db.execute(select(func.max(Price.date))).scalar()
            s_latest = db.execute(select(func.max(Score.date))).scalar()
            print(f"Price count: {p_count}, Latest Price: {p_latest}")
            print(f"Score count: {s_count}, Latest Score: {s_latest}")
            
            latest_scoring_date = get_latest_scoring_date()
            print(f"get_latest_scoring_date() returns: {latest_scoring_date}")
            
    except Exception as e:
        print(f"Error checking DB: {e}")

if __name__ == "__main__":
    check_db()
