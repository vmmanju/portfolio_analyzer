import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.config import settings
from services.user_portfolios import load_user_portfolios
from app.database import get_db_context
from sqlalchemy import text

print(f"ENVIRONMENT: {settings.ENVIRONMENT}")
print(f"DATABASE_URL: {settings.DATABASE_URL}")

try:
    with get_db_context() as db:
        res = db.execute(text("SELECT 1")).scalar()
        print(f"Database Connection: SUCCESS (1={res})")
except Exception as e:
    print(f"Database Connection: FAILED - {e}")

try:
    uid = 1
    loaded = load_user_portfolios(user_id=uid)
    print(f"Portfolios for UserID {uid}: {len(loaded)}")
    for p in loaded:
        print(f" - {p.get('name')}")
except Exception as e:
    print(f"Portfolio Loading: FAILED - {e}")
