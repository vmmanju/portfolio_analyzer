"""Create tables and verify database connectivity."""

import sys

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.database import Base, engine, get_db_context
from app.models import (  # noqa: F401 - ensure models are registered with Base
    Factor,
    Fundamental,
    PortfolioAllocation,
    Price,
    Score,
    Stock,
)


def setup_database() -> None:
    """Create all tables and test connection."""
    try:
        # Test connection via pool_pre_ping
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        Base.metadata.create_all(bind=engine)

        # Verify session works
        with get_db_context() as session:
            session.execute(text("SELECT 1"))

        print("Database setup complete")
    except OperationalError as e:
        print("Database connection failed.", file=sys.stderr)
        print(file=sys.stderr)
        print("PostgreSQL is not running or not reachable.", file=sys.stderr)
        print(f"  URL: {settings.DATABASE_URL}", file=sys.stderr)
        print(file=sys.stderr)
        print("Steps:", file=sys.stderr)
        print("  1. Install PostgreSQL and start the server (e.g. start the Windows service).", file=sys.stderr)
        print("  2. Create the database:  psql -U postgres -c \"CREATE DATABASE aistocks;\"", file=sys.stderr)
        print("  3. Or set DATABASE_URL in .env to your running PostgreSQL instance.", file=sys.stderr)
        print(file=sys.stderr)
        print(f"Original error: {e.orig}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    setup_database()
