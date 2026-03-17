import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database import get_db_context
from sqlalchemy import text

with get_db_context() as db:
    print("--- User Portfolio Counts ---")
    counts = db.execute(text("SELECT user_id, COUNT(*) FROM user_portfolios GROUP BY user_id")).all()
    for row in counts:
        print(f"UserID: {row[0]}, Portfolios: {row[1]}")

    print("\n--- User Emails ---")
    users = db.execute(text("SELECT id, email FROM users")).all()
    for u in users:
        print(f"ID: {u.id}, Email: {u.email}")
