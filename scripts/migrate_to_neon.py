"""Export schema + small tables for Neon free tier migration.
Drops all tables in Neon first, then imports fresh.
"""
import subprocess, sys, os

NEON_URL = "postgresql://neondb_owner:npg_EQTd1DNPCY9m@ep-steep-snow-a1a4wgoe.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
LOCAL_URL = "postgresql://postgres:ailearner1@localhost:5432/aistocks"

# Step 1: Drop all tables in Neon (clean slate)
print("=== Step 1: Cleaning Neon database ===")
drop_sql = """
DO $$ DECLARE
    r RECORD;
BEGIN
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
        EXECUTE 'DROP TABLE IF EXISTS public.' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
END $$;
"""

from sqlalchemy import create_engine, text
neon_engine = create_engine(NEON_URL)
with neon_engine.connect() as conn:
    conn.execute(text(drop_sql))
    conn.commit()
    # Verify
    result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
    tables = [r[0] for r in result]
    print(f"  Tables remaining: {tables}")
print("  ✅ Neon cleaned\n")

# Step 2: Export schema only from local
print("=== Step 2: Exporting schema from local ===")
schema_file = "neon_schema.sql"
os.environ["PGPASSWORD"] = "ailearner1"
subprocess.run([
    "pg_dump", "-h", "localhost", "-U", "postgres", "-d", "aistocks",
    "--schema-only", "--no-owner", "--no-acl",
    "-f", schema_file
], check=True)
print(f"  ✅ Schema exported to {schema_file}\n")

# Step 3: Export small tables data
print("=== Step 3: Exporting small tables data ===")
small_tables = ["stocks", "users", "user_portfolios", "user_portfolio_stocks",
                "model_calibration", "portfolio_allocations", "monthly_allocations",
                "portfolio_metrics", "fundamentals"]
data_file = "neon_data.sql"
subprocess.run([
    "pg_dump", "-h", "localhost", "-U", "postgres", "-d", "aistocks",
    "--data-only", "--no-owner", "--no-acl",
    *[arg for t in small_tables for arg in ["-t", t]],
    "-f", data_file
], check=True)
print(f"  ✅ Small tables data exported to {data_file}\n")

# Step 4: Import schema into Neon
print("=== Step 4: Importing schema into Neon ===")
os.environ["PGPASSWORD"] = "npg_EQTd1DNPCY9m"
result = subprocess.run([
    "psql", NEON_URL, "-f", schema_file
], capture_output=True, text=True)
errors = [l for l in result.stderr.split('\n') if 'ERROR' in l]
if errors:
    print(f"  ⚠️ {len(errors)} schema errors (may be expected for extensions)")
    for e in errors[:5]:
        print(f"    {e}")
else:
    print("  ✅ Schema imported clean")
print()

# Step 5: Import data into Neon
print("=== Step 5: Importing small tables data into Neon ===")
result = subprocess.run([
    "psql", NEON_URL, "-f", data_file
], capture_output=True, text=True)
errors = [l for l in result.stderr.split('\n') if 'ERROR' in l]
if errors:
    print(f"  ⚠️ {len(errors)} data errors")
    for e in errors[:5]:
        print(f"    {e}")
else:
    print("  ✅ Data imported clean")
print()

# Step 6: Verify
print("=== Step 6: Verification ===")
with neon_engine.connect() as conn:
    result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"))
    tables = [r[0] for r in result]
    print(f"  Tables: {tables}")
    for t in ["stocks", "users", "user_portfolios"]:
        if t in tables:
            count = conn.execute(text(f"SELECT count(*) FROM {t}")).scalar()
            print(f"  {t}: {count} rows")

print("\n✅ Neon migration complete!")
print("Market data (prices, factors, scores) needs to be re-fetched via the data pipeline.")
