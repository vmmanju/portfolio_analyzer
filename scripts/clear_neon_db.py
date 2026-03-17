import sys
from pathlib import Path
import os

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database import get_db_context
from sqlalchemy import text

def reset_data():
    print("Clearing scores, factors, and prices to free up Neon DB space...")
    with get_db_context() as db:
        # Neon might block inserts, but deletes should work. Just in case, try one by one.
        print("Truncating scores...")
        db.execute(text("TRUNCATE TABLE scores RESTART IDENTITY CASCADE"))
        db.commit()
        print("Truncating factors...")
        db.execute(text("TRUNCATE TABLE factors RESTART IDENTITY CASCADE"))
        db.commit()
        print("Truncating prices...")
        db.execute(text("TRUNCATE TABLE prices RESTART IDENTITY CASCADE"))
        db.commit()
    print("Database cleared successfully! Ready for 2-year history sync.")

if __name__ == "__main__":
    reset_data()
