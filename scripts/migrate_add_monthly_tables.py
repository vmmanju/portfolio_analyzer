"""scripts/migrate_add_monthly_tables.py

Migration script: add monthly_allocations and portfolio_metrics tables.

Usage
-----
    python scripts/migrate_add_monthly_tables.py [--dry-run] [--drop-existing]

Arguments
---------
--dry-run        Print the SQL that would run but do NOT execute it.
--drop-existing  Drop and recreate the two new tables (DESTRUCTIVE — loses data).
                 Default: safe incremental create (skips tables that already exist).

Algorithm
---------
1. Connect to the database using the existing DATABASE_URL from app/config.
2. Inspect the existing schema.
3. For each new table (monthly_allocations, portfolio_metrics):
   a. If it already exists and --drop-existing is NOT set  → skip (idempotent).
   b. If --drop-existing IS set                            → drop then recreate.
   c. Otherwise                                            → CREATE TABLE.
4. Log every action taken.
5. Run a self-test SELECT to verify the tables are accessible.

This script is intentionally dependency-light (no Alembic required) so it can
be run in any environment that has the project installed.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sqlalchemy import inspect, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_logger = logging.getLogger("migrate")

NEW_TABLES = ["monthly_allocations", "portfolio_metrics"]


def _get_engine_and_base():
    """Import engine and Base after path is configured."""
    from app.database import engine, Base  # noqa: F401
    # Import models so that ORM has registered all tables on Base.metadata
    import app.models  # noqa: F401
    return engine, Base


def run_migration(dry_run: bool = False, drop_existing: bool = False) -> None:
    engine, Base = _get_engine_and_base()
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    _logger.info("Connected to: %s", str(engine.url).split("@")[-1])
    _logger.info("Existing tables: %s", existing_tables)
    _logger.info("New tables to migrate: %s", NEW_TABLES)

    if dry_run:
        _logger.info("DRY-RUN mode — no changes will be made.")

    tables_to_create = []
    tables_to_drop_first = []

    for tname in NEW_TABLES:
        if tname in existing_tables:
            if drop_existing:
                _logger.warning("Table '%s' exists and --drop-existing is set — will DROP.", tname)
                tables_to_drop_first.append(tname)
                tables_to_create.append(tname)
            else:
                _logger.info("Table '%s' already exists — skipping (safe incremental mode).", tname)
        else:
            _logger.info("Table '%s' does not exist — will CREATE.", tname)
            tables_to_create.append(tname)

    if not tables_to_create and not tables_to_drop_first:
        _logger.info("Nothing to do. All tables already present.")
        return

    if dry_run:
        _logger.info("Would drop:   %s", tables_to_drop_first)
        _logger.info("Would create: %s", tables_to_create)
        # Print DDL for each table
        from sqlalchemy.schema import CreateTable
        for tname in tables_to_create:
            if tname in Base.metadata.tables:
                ddl = str(CreateTable(Base.metadata.tables[tname]).compile(engine))
                _logger.info("DDL for %s:\n%s", tname, ddl)
        return

    with engine.begin() as conn:
        # Drop in reverse dependency order if requested
        if tables_to_drop_first:
            # monthly_allocations → portfolio_metrics (no FK between them, order doesn't matter)
            for tname in reversed(tables_to_drop_first):
                _logger.warning("Dropping table: %s", tname)
                conn.execute(text(f"DROP TABLE IF EXISTS {tname}"))

        # Create only the requested tables using SQLAlchemy DDL
        target_metadata_tables = {
            tname: Base.metadata.tables[tname]
            for tname in tables_to_create
            if tname in Base.metadata.tables
        }
        if target_metadata_tables:
            Base.metadata.create_all(engine, tables=list(target_metadata_tables.values()))
            for tname in tables_to_create:
                _logger.info("✓ Created table: %s", tname)

    # Self-test: verify tables are selectable
    _logger.info("Running self-test queries…")
    with engine.connect() as conn:
        for tname in NEW_TABLES:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {tname}"))
                count = result.scalar()
                _logger.info("  ✓ %s  →  %d rows", tname, count)
            except Exception as exc:
                _logger.error("  ✗ %s  →  %s", tname, exc)
                sys.exit(1)

    _logger.info("Migration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add monthly_allocations and portfolio_metrics tables.")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL but do not execute.")
    parser.add_argument("--drop-existing", action="store_true",
                        help="DROP and recreate if tables already exist (DESTRUCTIVE).")
    args = parser.parse_args()
    run_migration(dry_run=args.dry_run, drop_existing=args.drop_existing)
