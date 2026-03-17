"""Database migration: add multi-user support.

Run this ONCE to upgrade an existing single-user database to multi-user.
It will:
  1. Create the 'users' table if it doesn't exist.
  2. Create a default user for all existing data.
  3. Add 'user_id' columns to per-user tables.
  4. Backfill existing rows with the default user's ID.
  5. Update constraints.

Usage:
  cd portfolio_analyzer
  python scripts/migrate_multiuser.py
"""

import sys
import os

# Ensure package imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text, inspect
from app.database import engine, Base
from app.models import User  # noqa: F401 — ensures table is registered


DEFAULT_USER_EMAIL = "admin@localhost"
DEFAULT_USER_NAME = "Default Admin"


def run_migration():
    """Run the multi-user migration."""
    conn = engine.connect()
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    print("=" * 60)
    print("Multi-User Migration")
    print("=" * 60)

    # ── Step 1: Create users table ─────────────────────────────────────────
    if "users" not in existing_tables:
        print("\n[1/5] Creating 'users' table...")
        conn.execute(text("""
            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(256) NOT NULL UNIQUE,
                name VARCHAR(256),
                picture_url VARCHAR(512),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
        print("  ✅ Created 'users' table")
    else:
        print("\n[1/5] 'users' table already exists — skipping")

    # ── Step 2: Create default user ────────────────────────────────────────
    print("\n[2/5] Creating default user...")
    result = conn.execute(
        text("SELECT id FROM users WHERE email = :email"),
        {"email": DEFAULT_USER_EMAIL},
    ).fetchone()

    if result:
        default_user_id = result[0]
        print(f"  Default user already exists (id={default_user_id})")
    else:
        conn.execute(
            text("INSERT INTO users (email, name) VALUES (:email, :name)"),
            {"email": DEFAULT_USER_EMAIL, "name": DEFAULT_USER_NAME},
        )
        conn.commit()
        result = conn.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": DEFAULT_USER_EMAIL},
        ).fetchone()
        default_user_id = result[0]
        print(f"  ✅ Created default user (id={default_user_id})")

    # ── Step 3: Add user_id columns ────────────────────────────────────────
    tables_to_update = [
        "portfolio_allocations",
        "user_portfolios",
        "monthly_allocations",
        "portfolio_metrics",
    ]

    print("\n[3/5] Adding user_id columns...")
    for table_name in tables_to_update:
        if table_name not in existing_tables:
            print(f"  ⏭️  '{table_name}' doesn't exist — skipping")
            continue

        columns = [col["name"] for col in inspector.get_columns(table_name)]
        if "user_id" in columns:
            print(f"  ⏭️  '{table_name}.user_id' already exists — skipping")
            continue

        conn.execute(text(f"""
            ALTER TABLE {table_name}
            ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
        """))
        conn.commit()
        print(f"  ✅ Added user_id to '{table_name}'")

    # ── Step 4: Backfill existing rows ─────────────────────────────────────
    print("\n[4/5] Backfilling existing rows with default user...")
    for table_name in tables_to_update:
        if table_name not in existing_tables:
            continue

        result = conn.execute(text(f"""
            UPDATE {table_name}
            SET user_id = :uid
            WHERE user_id IS NULL
        """), {"uid": default_user_id})
        conn.commit()
        print(f"  ✅ '{table_name}': {result.rowcount} rows updated")

    # ── Step 5: Update constraints ─────────────────────────────────────────
    print("\n[5/5] Updating unique constraints...")

    # Drop old constraints and add new ones
    constraint_updates = [
        # (table, old_constraint_name, new_constraint_sql)
        (
            "user_portfolios",
            "user_portfolios_name_key",  # old unique on name only
            "ALTER TABLE user_portfolios ADD CONSTRAINT uq_user_portfolio_user_name UNIQUE (user_id, name)",
        ),
        (
            "monthly_allocations",
            "uq_monthly_alloc_type_name_date_stock",
            "ALTER TABLE monthly_allocations ADD CONSTRAINT uq_monthly_alloc_user_type_name_date_stock UNIQUE (user_id, portfolio_type, portfolio_name, rebalance_date, stock_id)",
        ),
        (
            "portfolio_metrics",
            "uq_portfolio_metrics_type_name_date",
            "ALTER TABLE portfolio_metrics ADD CONSTRAINT uq_portfolio_metrics_user_type_name_date UNIQUE (user_id, portfolio_type, portfolio_name, rebalance_date)",
        ),
    ]

    for table_name, old_name, new_sql in constraint_updates:
        if table_name not in existing_tables:
            continue

        # Try to drop old constraint (may not exist)
        try:
            conn.execute(text(f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {old_name}"))
            conn.commit()
        except Exception:
            conn.rollback()

        # Add new constraint
        try:
            conn.execute(text(new_sql))
            conn.commit()
            print(f"  ✅ Updated constraint on '{table_name}'")
        except Exception as e:
            conn.rollback()
            if "already exists" in str(e).lower():
                print(f"  ⏭️  Constraint on '{table_name}' already exists")
            else:
                print(f"  ⚠️  Constraint on '{table_name}' failed: {e}")

    # ── Add user_id indexes ────────────────────────────────────────────────
    for table_name in tables_to_update:
        if table_name not in existing_tables:
            continue
        idx_name = f"ix_{table_name}_user_id"
        try:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} (user_id)"))
            conn.commit()
        except Exception:
            conn.rollback()

    conn.close()

    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"Default user: {DEFAULT_USER_EMAIL} (id={default_user_id})")
    print("=" * 60)


if __name__ == "__main__":
    run_migration()
