from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Price, Stock, User
import services.portfolio_tracker as pt


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


def _seed_prices():
    with MockDBContext() as db:
        db.add_all(
            [
                Stock(id=1, symbol="TCS"),
                Stock(id=2, symbol="INFY"),
                Price(stock_id=1, date=date(2025, 1, 1), close=3500),
                Price(stock_id=1, date=date(2025, 1, 2), close=3600),
                Price(stock_id=2, date=date(2025, 1, 2), close=1500),
            ]
        )


def test_save_tracked_positions_round_trip(monkeypatch):
    monkeypatch.setattr(pt, "get_db_context", MockDBContext)
    monkeypatch.setattr(pt, "engine", engine)
    Base.metadata.create_all(bind=engine)
    try:
        _seed_prices()
        saved = pt.save_tracked_positions(
            [
                {"symbol": "TCS", "invested_amount": 10000, "quantity": 3},
                {"symbol": "INFY", "invested_amount": 5000, "quantity": 2},
            ],
            user_id=1,
        )
        assert len(saved) == 2

        loaded = pt.load_tracked_positions(user_id=1)
        assert loaded == [
            {"symbol": "INFY", "invested_amount": 5000.0, "quantity": 2.0},
            {"symbol": "TCS", "invested_amount": 10000.0, "quantity": 3.0},
        ]
    finally:
        Base.metadata.drop_all(bind=engine)


def test_build_tracker_snapshot_computes_current_amount(monkeypatch):
    monkeypatch.setattr(pt, "get_db_context", MockDBContext)
    monkeypatch.setattr(pt, "engine", engine)
    Base.metadata.create_all(bind=engine)
    try:
        _seed_prices()
        snapshot = pt.build_tracker_snapshot(
            [
                {"symbol": "TCS", "invested_amount": 10000, "quantity": 3},
                {"symbol": "INFY", "invested_amount": 5000, "quantity": 2},
            ]
        )

        positions_df = snapshot["positions_df"]
        tcs_row = positions_df.loc[positions_df["Symbol"] == "TCS"].iloc[0]
        infy_row = positions_df.loc[positions_df["Symbol"] == "INFY"].iloc[0]

        assert tcs_row["Current Price"] == 3600.0
        assert tcs_row["Current Amount"] == 10800.0
        assert infy_row["Current Amount"] == 3000.0

        summary = snapshot["summary"]
        assert summary["total_invested"] == 15000.0
        assert summary["total_current"] == 13800.0
        assert summary["total_pnl"] == -1200.0
        assert summary["priced_positions"] == 2
    finally:
        Base.metadata.drop_all(bind=engine)


def test_save_tracked_positions_creates_missing_tracker_table(monkeypatch):
    monkeypatch.setattr(pt, "get_db_context", MockDBContext)
    monkeypatch.setattr(pt, "engine", engine)
    Base.metadata.create_all(bind=engine, tables=[User.__table__, Stock.__table__, Price.__table__])
    try:
        _seed_prices()
        saved = pt.save_tracked_positions(
            [{"symbol": "TCS", "invested_amount": 10000, "quantity": 3}],
            user_id=1,
        )
        assert saved == [{"symbol": "TCS", "invested_amount": 10000.0, "quantity": 3.0}]

        loaded = pt.load_tracked_positions(user_id=1)
        assert len(loaded) == 1
        assert loaded[0]["symbol"] == "TCS"
    finally:
        Base.metadata.drop_all(bind=engine)


def test_build_tracker_snapshot_prefers_live_prices(monkeypatch):
    monkeypatch.setattr(pt, "get_db_context", MockDBContext)
    monkeypatch.setattr(pt, "engine", engine)
    Base.metadata.create_all(bind=engine)
    try:
        _seed_prices()

        def _fake_fetch(symbol, start_date=None):
            if symbol == "TCS":
                return pt.pd.DataFrame(
                    [
                        {"date": date(2025, 1, 3), "open": 0, "high": 0, "low": 0, "close": 3700, "volume": 0}
                    ]
                )
            return pt.pd.DataFrame()

        monkeypatch.setattr(pt, "fetch_price_data", _fake_fetch)

        snapshot = pt.build_tracker_snapshot(
            [{"symbol": "TCS", "invested_amount": 10000, "quantity": 3}],
            prefer_live=True,
        )

        tcs_row = snapshot["positions_df"].loc[snapshot["positions_df"]["Symbol"] == "TCS"].iloc[0]
        assert tcs_row["Current Price"] == 3700.0
        assert tcs_row["Price Source"] == "live"
        assert snapshot["summary"]["total_current"] == 11100.0
    finally:
        Base.metadata.drop_all(bind=engine)
