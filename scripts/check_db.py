import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database import get_db_context
from app.models import Stock
from sqlalchemy import select

with get_db_context() as db:
    rows = db.execute(select(Stock.symbol)).scalars().all()

print('db_count', len(rows))
print(rows[:50])
