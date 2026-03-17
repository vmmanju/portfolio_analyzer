"""scripts/migrate_add_model_calibration.py

Migration: add model_calibration table (Bayesian error-model coefficient history).

Usage
-----
    python scripts/migrate_add_model_calibration.py [--dry-run] [--drop-existing]

Arguments
---------
--dry-run        Print the SQL that would run but do NOT execute it.
--drop-existing  Drop and recreate the table (DESTRUCTIVE — loses data).
                 Default: safe incremental create (skips if already exists).

Algorithm
---------
1. Connect to the database using DATABASE_URL from app/config.
2. Inspect existing schema.
3. For model_calibration:
   a. Exists + NOT --drop-existing   → skip (idempotent).
   b. --drop-existing is set         → drop then recreate.
   c. Does not exist                 → CREATE TABLE.
4. Run self-test SELECT.
"""

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
_logger = logging.getLogger("migrate_calibration")

NEW_TABLE = "model_calibration"


def _get_engine_and_base():
    from app.database import engine, Base  # noqa: F401
    import app.models  # noqa: F401  — registers ModelCalibration on Base.metadata
    return engine, Base


def run_migration(dry_run: bool = False, drop_existing: bool = False) -> None:
    engine, Base = _get_engine_and_base()
    inspector     = inspect(engine)
    existing      = inspector.get_table_names()

    _logger.info("Connected to: %s", str(engine.url).split("@")[-1])
    _logger.info("Existing tables: %s", existing)
    _logger.info("Target table:    %s", NEW_TABLE)

    if dry_run:
        _logger.info("DRY-RUN mode — no changes will be made.")

    already_exists = NEW_TABLE in existing

    if already_exists and not drop_existing:
        _logger.info("Table '%s' already exists — skipping (safe incremental mode).", NEW_TABLE)
        _self_test(engine)
        return

    if dry_run:
        action = "DROP then CREATE" if (already_exists and drop_existing) else "CREATE"
        _logger.info("Would %s table: %s", action, NEW_TABLE)
        if NEW_TABLE in Base.metadata.tables:
            from sqlalchemy.schema import CreateTable
            ddl = str(CreateTable(Base.metadata.tables[NEW_TABLE]).compile(engine))
            _logger.info("DDL:\n%s", ddl)
        return

    with engine.begin() as conn:
        if already_exists and drop_existing:
            _logger.warning("Dropping existing table: %s", NEW_TABLE)
            conn.execute(text(f"DROP TABLE IF EXISTS {NEW_TABLE}"))

        if NEW_TABLE in Base.metadata.tables:
            Base.metadata.create_all(engine, tables=[Base.metadata.tables[NEW_TABLE]])
            _logger.info("✓ Created table: %s", NEW_TABLE)
        else:
            _logger.error("Table %r not found in ORM metadata — check app/models.py", NEW_TABLE)
            sys.exit(1)

    _self_test(engine)
    _logger.info("Migration complete.")


def _self_test(engine) -> None:
    """Verify the table is accessible with a COUNT(*) query."""
    _logger.info("Running self-test…")
    with engine.connect() as conn:
        try:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {NEW_TABLE}"))
            count  = result.scalar()
            _logger.info("  ✓ %s  →  %d rows", NEW_TABLE, count)
        except Exception as exc:
            _logger.error("  ✗ %s  →  %s", NEW_TABLE, exc)
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add model_calibration table for Bayesian error-model history."
    )
    parser.add_argument("--dry-run",       action="store_true", help="Print SQL, do not execute.")
    parser.add_argument("--drop-existing", action="store_true",
                        help="DROP and recreate if table already exists (DESTRUCTIVE).")
    args = parser.parse_args()
    run_migration(dry_run=args.dry_run, drop_existing=args.drop_existing)
