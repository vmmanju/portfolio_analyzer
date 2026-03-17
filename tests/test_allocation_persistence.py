import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from app.models import Base, MonthlyAllocation
import services.allocation_persistence as ap

# We need an in-memory SQLite engine to test exactly how idempotency
# works with real SQLAlchemy calls.
engine = create_engine("sqlite:///:memory:")
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class MockDBContext:
    def __enter__(self):
        self.db = TestingSessionLocal()
        return self.db
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.db.commit()
        else:
            self.db.rollback()
        self.db.close()
        return False

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_save_monthly_allocation_idempotency(monkeypatch):
    monkeypatch.setattr(ap, "get_db_context", MockDBContext)
    
    rebalance_date = date(2023, 1, 31)
    weights = {101: 0.6, 102: 0.4}
    
    # Run 1: Should insert 2 rows successfully (pass user_id=1 for unique constraint reliability)
    inserted = ap.save_monthly_allocation("test_port", rebalance_date, weights, portfolio_name="my_port", user_id=1)
    assert inserted == 2
    
    # Run 2: Exact same run should trigger IntegrityError per row
    inserted = ap.save_monthly_allocation("test_port", rebalance_date, weights, portfolio_name="my_port", user_id=1)
    assert inserted == 0
    
    # Verify count is 2 in db
    with MockDBContext() as db:
        count = db.query(MonthlyAllocation).filter_by(portfolio_type="test_port").count()
        assert count == 2

def test_save_portfolio_metrics_idempotency(monkeypatch):
    monkeypatch.setattr(ap, "get_db_context", MockDBContext)
    
    rebalance_date = date(2023, 2, 28)
    metrics = {"CAGR": 0.1, "Volatility": 0.2, "Sharpe": 0.5}
    
    # Run 1
    success = ap.save_portfolio_metrics("test_port", rebalance_date, metrics, portfolio_name="my_port", user_id=1)
    assert success is True
    
    # Run 2
    success = ap.save_portfolio_metrics("test_port", rebalance_date, metrics, portfolio_name="my_port", user_id=1)
    assert success is False
