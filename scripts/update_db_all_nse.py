"""
Script to expand the universe to the entire NSE 500.
Fetches NIFTY 500 list, syncs standard Stock definitions, updates prices, and 
triggers full factor/scoring computation for user recommendation capability.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import urllib.request as r
import json
import ssl
from typing import List, Dict

from services.universe_loader import sync_universe_with_db
from services.data_fetcher import update_all_stocks
from services.factor_engine import run_factor_engine
from services.scoring import run_scoring_engine

def fetch_nifty_500() -> List[Dict]:
    """Fetch NIFTY 500 list from Wikipedia using requests and pandas."""
    import pandas as pd
    import requests
    import io
    try:
        url = "https://en.wikipedia.org/wiki/NIFTY_500"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        
        tables = pd.read_html(io.StringIO(res.text))
        
        # Find the constituents table
        df = None
        for t in tables:
            if len(t) > 400:
                df = t
                break
                
        if df is None:
            print("Could not find the NIFTY 500 table on the Wikipedia page.")
            return []
            
        # In some pandas versions, the header is read as row 0
        if 0 in df.columns:
            df.columns = df.iloc[0]
            df = df.drop(0)
            
        records = []
        for _, row in df.iterrows():
            sym = str(row.get('Symbol', ''))
            if not sym or sym.lower() == 'nan':
                continue
                
            name = str(row.get('Company Name', sym))
            sector = str(row.get('Industry', 'General'))
            
            # Format to Yahoo Finance format
            yf_sym = f"{sym.strip()}.NS"
            records.append({
                "symbol": yf_sym,
                "name": name.strip(),
                "sector": sector.strip()
            })
        return records
    except Exception as e:
        print(f"Failed to fetch NIFTY 500 from Wikipedia: {e}")
        return []

def main():
    print("=== Expanding DB Universe to NIFTY 500 ===")
    records = fetch_nifty_500()
    if not records:
        print("Could not fetch NIFTY 500 symbols. Aborting.")
        return
        
    print(f"Successfully fetched {len(records)} symbols from NSE.")
    
    print("\n--- Synchronizing Universe to Database ---")
    sync_universe_with_db(records)
    
    print("\n--- Updating Prices for All DB Stocks ---")
    processed, inserted, skipped = update_all_stocks()
    print(f"Processed: {processed}, Inserted rows: {inserted}, Skipped rows: {skipped}")
    
    print("\n--- Calculating Matrix Factors ---")
    run_factor_engine()
    
    print("\n--- Computing Final Stock Scores ---")
    run_scoring_engine()
    
    print("\n✅ NSE 500 Update Complete. The Recommendation engine is now populated with the full universe.")

if __name__ == "__main__":
    main()
