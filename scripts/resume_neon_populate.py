import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from services.factor_engine import run_factor_engine
from services.scoring import run_scoring_engine

def main():
    print("=== Resuming Neon Data Populate ===")
    print("\n--- Calculating Matrix Factors ---")
    run_factor_engine()
    print("\n--- Computing Final Stock Scores ---")
    run_scoring_engine()
    print("\n✅ Neon Populate Complete.")

if __name__ == "__main__":
    main()
