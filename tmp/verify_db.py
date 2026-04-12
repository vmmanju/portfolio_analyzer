import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database import get_db_context
from app.models import Factor, Score
from sqlalchemy import func, select

with get_db_context() as db:
    factor_count = db.execute(select(func.count(Factor.id))).scalar()
    score_count = db.execute(select(func.count(Score.id))).scalar()
    
print(f"Factor count: {factor_count}")
print(f"Score count: {score_count}")
