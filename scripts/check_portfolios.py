import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database import get_db_context
from sqlalchemy import text

with get_db_context() as db:
    print("--- User Portfolios ---")
    portfolios = db.execute(text("SELECT id, name, user_id FROM user_portfolios ORDER BY user_id")).all()
    for p in portfolios:
        print(f"P_ID: {p.id} | Name: {p.name} | User_ID: {p.user_id}")
        stocks = db.execute(text("SELECT symbol FROM user_portfolio_stocks WHERE portfolio_id = :pid"), {"pid": p.id}).all()
        print(f"  Stocks: {[s.symbol for s in stocks]}")

    print("\n--- Users ---")
    users = db.execute(text("SELECT id, email FROM users")).all()
    for u in users:
        print(f"U_ID: {u.id} | Email: {u.email}")
