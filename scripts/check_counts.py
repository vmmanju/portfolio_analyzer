import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database import get_db_context
from sqlalchemy import text

with get_db_context() as db:
    tables = ["user_portfolios", "prices", "stocks", "monthly_allocations"]
    for table in tables:
        try:
            count = db.execute(text(f"SELECT count(*) FROM {table}")).scalar()
            print(f"{table}: {count}")
        except Exception as e:
            print(f"Error checking {table}: {e}")

    try:
        latest_date = db.execute(text("SELECT max(date) FROM prices")).scalar()
        print(f"Latest Price Date: {latest_date}")
    except Exception as e:
        print(f"Error checking latest date: {e}")
