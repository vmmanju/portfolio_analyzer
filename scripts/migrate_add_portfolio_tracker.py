"""Migration script: add the portfolio_tracker_positions table."""

import argparse
import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sqlalchemy import inspect, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_logger = logging.getLogger("migrate_portfolio_tracker")

NEW_TABLE = "portfolio_tracker_positions"


def _get_engine_and_base():
    from app.database import engine, Base  # noqa: F401
    import app.models  # noqa: F401
    return engine, Base


def run_migration(dry_run: bool = False, drop_existing: bool = False) -> None:
    engine, Base = _get_engine_and_base()
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    _logger.info("Connected to: %s", str(engine.url).split("@")[-1])
    _logger.info("Existing tables: %s", existing_tables)

    if NEW_TABLE in existing_tables and not drop_existing:
        _logger.info("Table '%s' already exists; nothing to do.", NEW_TABLE)
        return

    if dry_run:
        from sqlalchemy.schema import CreateTable

        if NEW_TABLE in Base.metadata.tables:
            ddl = str(CreateTable(Base.metadata.tables[NEW_TABLE]).compile(engine))
            _logger.info("DDL for %s:\n%s", NEW_TABLE, ddl)
        return

    with engine.begin() as conn:
        if NEW_TABLE in existing_tables and drop_existing:
            _logger.warning("Dropping table: %s", NEW_TABLE)
            conn.execute(text(f"DROP TABLE IF EXISTS {NEW_TABLE}"))

        Base.metadata.create_all(engine, tables=[Base.metadata.tables[NEW_TABLE]])
        _logger.info("Created table: %s", NEW_TABLE)

    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {NEW_TABLE}"))
        _logger.info("Self-test passed: %s has %d rows", NEW_TABLE, result.scalar())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add the portfolio_tracker_positions table.")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL but do not execute.")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="DROP and recreate the table if it already exists (DESTRUCTIVE).",
    )
    args = parser.parse_args()
    run_migration(dry_run=args.dry_run, drop_existing=args.drop_existing)
