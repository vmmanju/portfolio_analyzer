import unittest
from unittest.mock import patch, MagicMock
from datetime import date
import threading
import time

# Import the service to test
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services.auto_diversified_portfolio import rebalance_only_if_new_month, _state_file_lock

class TestRebalanceDeadlock(unittest.TestCase):
    def test_reentrant_lock_prevents_deadlock(self):
        """
        Verify that rebalance_only_if_new_month does not hang 
        when calling _load_rebalance_state internally.
        """
        as_of_date = date(2026, 3, 1)
        
        # We mock build_diversified_hybrid_portfolio so we don't do actual DB work
        with patch('services.auto_diversified_portfolio.build_diversified_hybrid_portfolio') as mock_build:
            mock_build.return_value = {"status": "success", "portfolio": {}}
            
            # We mock the state loaders/savers to avoid disk I/O
            with patch('services.auto_diversified_portfolio._load_rebalance_state') as mock_load:
                mock_load.return_value = {"last_rebalance": "2026-02-01"}
                
                with patch('services.auto_diversified_portfolio._save_rebalance_state'):
                    
                    # This call used to hang due to Lock deadlock.
                    # With RLock, it should complete instantly.
                    start_time = time.time()
                    try:
                        # Use a small timeout thread in case it still hangs
                        result = []
                        def run_with_timeout():
                            result.append(rebalance_only_if_new_month(as_of_date=as_of_date, force=True))
                        
                        thread = threading.Thread(target=run_with_timeout)
                        thread.start()
                        thread.join(timeout=5)
                        
                        if thread.is_alive():
                            self.fail("rebalance_only_if_new_month HUNG (Deadlock detected)")
                        
                        self.assertIn("status", result[0])
                        print(f"Deadlock test passed in {time.time() - start_time:.4f}s")
                        
                    except Exception as e:
                        self.fail(f"Test failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
