import yfinance as yf
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def fetch_live_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Fetch live quarterly fundamental metrics from yfinance.
    Used as a fallback when the database table is empty.
    """
    # Normalize symbol
    symbol = symbol.strip().upper()
    if not symbol.endswith(".NS") and "." not in symbol:
        symbol += ".NS"
        
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract desired metrics from .info (more reliable than dragging full financials)
        eps = info.get("trailingEps")
        roe = info.get("returnOnEquity")
        debt_equity = info.get("debtToEquity")
        
        # Scaling ROE if it's a decimal (yf usually returns 0.39 for 39%)
        if roe is not None:
            roe = round(roe * 100, 2)
            
        return {
            "eps": eps,
            "roe": roe,
            "debt_equity": debt_equity,
            "period": "LTM (Live)",
            "source": "yfinance_live"
        }
    except Exception as e:
        logger.error(f"Failed to fetch live fundamentals for {symbol}: {e}")
        return {}
