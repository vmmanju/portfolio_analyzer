import json
from app.database import get_db_context
from app.models import Stock, Fundamental
from sqlalchemy import select

def audit_tcs_fundamentals():
    with get_db_context() as db:
        stock = db.execute(select(Stock).where(Stock.symbol == "TCS.NS")).scalars().first()
        if not stock:
            print("ERROR: Stock TCS.NS not found in 'stocks' table.")
            return

        print(f"DEBUG: Found Stock ID={stock.id} for TCS.NS")
        
        # Check all fundamental records for this stock
        funds = db.execute(select(Fundamental).where(Fundamental.stock_id == stock.id)).scalars().all()
        if not funds:
            print("ERROR: No records found for TCS.NS in 'fundamentals' table.")
        else:
            print(f"DEBUG: Found {len(funds)} records in 'fundamentals' table.")
            for f in funds:
                print(f"  - Quarter={f.quarter}, EPS={f.eps}, ROE={f.roe}, D/E={f.debt_equity}")

if __name__ == "__main__":
    audit_tcs_fundamentals()
