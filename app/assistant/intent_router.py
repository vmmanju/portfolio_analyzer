import re
from typing import Dict, Any, List, Tuple

class IntentRouter:
    """
    Rule-based deterministic intent router. Easily swappable with an LLM-based 
    router in the future for semantic understanding.
    """
    
    INTENT_MAPPINGS = {
        "stock_recommendation": [r"(?i)\b(buy|sell|hold|reduce|reduce)\b\s+([A-Z]+)"],
        "stock_research": [r"(?i)\b(research|analyze|factor)\b\s+([A-Z]+)"],
        "portfolio_comparison": [r"(?i)\b(compare)\b.*\b(portfolio|strategy|strategies)\b", r"(?i)\b(portfolio|strategy)\b.*\b(compare|strongest|best|highest)\b"],
        "hybrid_portfolio": [r"(?i)\b(hybrid|auto diversified)\b"],
        "meta_portfolio": [r"(?i)\b(meta|blended)\b"],
        "risk_diagnostics": [r"(?i)\b(stop-loss|risk)\b\s+([A-Z]+)"],
        "automatic_n": [r"(?i)\b(automatic\s*n|how many stocks|12|13|choose)\b"],
        "governance_check": [r"(?i)\b(overfitting|governance|bias)\b"],
        "calibration": [r"(?i)\b(calibration|drift|coefficients)\b"],
        "stability": [r"(?i)\b(stability|rolling|turnover)\b"],
        "sector_ranking": [
            r"(?i)\b(top|best|rank|ranking)\b.*\b(sectors?)\b",
            r"(?i)\bsectors?\b.*\b(top|best|rank|ranking|performing)\b",
            r"(?i)\bsector\b.*\bperformance\b",
        ]
    }

    @classmethod
    def extract_symbol(cls, query: str) -> str:
        # 1. Look for explicit .NS symbols
        match = re.search(r'\b([A-Z0-9]+\.NS)\b', query, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        # 2. Look for all-caps strings >= 2 chars, ignoring common words
        stop_words = {"SHOULD", "WHAT", "HOW", "WHY", "IS", "ARE", "CAN", "WILL", "DO", "DOES", "BUY", "SELL", "HOLD", "REDUCE", "THE", "A", "AN", "THIS", "THAT", "NS", "STOCK", "STOCKS", "INFO", "RESEARCH", "RECOMMEND", "TO", "GIVE", "SHOW", "LIST"}
        matches = re.findall(r'\b([A-Z]{2,6})\b', query)
        for m in matches:
            if m not in stop_words:
                return m + ".NS"
                
        # 3. Look for Capitalized words (Title Case)
        matches = re.findall(r'\b([A-Z][a-z]+)\b', query)
        for m in matches:
            if m.upper() not in stop_words:
                return m.upper() + ".NS"
                
        return "UNKNOWN"

    @classmethod
    def route_query(cls, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parses the query and maps to an Intent and Functions.
        Tries LLM routing first, falling back to deterministic Regex.
        Returns: Tuple[IntentName, List of Function Calls [{func_name: args}]]
        """
        from app.assistant.llm_engine import llm_route_query
        
        llm_result = llm_route_query(query)
        if isinstance(llm_result, tuple) and len(llm_result) == 5:
            llm_intent, llm_symbol, llm_portfolio, llm_symbols, llm_sector = llm_result
        elif isinstance(llm_result, tuple) and len(llm_result) == 4:
            llm_intent, llm_symbol, llm_portfolio, llm_symbols = llm_result
            llm_sector = ""
        else:
            llm_intent, llm_symbol, llm_portfolio, llm_symbols, llm_sector = "", "", "", [], ""
        intent = ""
        symbol = "UNKNOWN"
        portfolio_name = ""
        symbols_list = []
        sector = ""
        
        if llm_intent:
            intent = llm_intent
            symbol = llm_symbol if llm_symbol else cls.extract_symbol(query)
            portfolio_name = llm_portfolio
            symbols_list = llm_symbols
            sector = llm_sector
            
            # Deterministic override: if user asks for top N stocks, force Market Overview
            if re.search(r'(?:list\s+|show\s+)?top\s+(\d+)', query, re.IGNORECASE) and not re.search(r'\bsectors?\b', query, re.IGNORECASE):
                intent = "Market Overview"
        else:
            # Fallback to Regex
            if any(re.search(p, query, re.IGNORECASE) for p in cls.INTENT_MAPPINGS["sector_ranking"]): intent = "Sector Ranking"
            elif any(re.search(p, query, re.IGNORECASE) for p in [r"recommend.*stocks", r"top\s+\d+", r"best.*stocks", r"list.*top"]): intent = "Market Overview"
            elif any(re.search(p, query) for p in cls.INTENT_MAPPINGS["automatic_n"]): intent = "Automatic N Explanation"
            elif any(re.search(p, query) for p in cls.INTENT_MAPPINGS["hybrid_portfolio"]): intent = "Hybrid Portfolio Request"
            elif any(re.search(p, query) for p in cls.INTENT_MAPPINGS["portfolio_comparison"]): intent = "Portfolio Comparison"
            elif any(re.search(p, query) for p in cls.INTENT_MAPPINGS["governance_check"]): intent = "Governance Check"
            elif any(re.search(p, query) for p in cls.INTENT_MAPPINGS["calibration"]): intent = "Calibration Explanation"
            elif any(re.search(p, query) for p in cls.INTENT_MAPPINGS["stability"]): intent = "Stability Diagnostic"
            elif re.search(r"(?i)\bstop[- ]?loss\b", query) or any(re.search(p, query) for p in cls.INTENT_MAPPINGS["risk_diagnostics"]): intent = "Risk Diagnostics"
            elif any(re.search(p, query) for p in cls.INTENT_MAPPINGS["stock_research"]): intent = "Stock Research"
            elif cls.extract_symbol(query) != "UNKNOWN": intent = "Stock Recommendation"
            else: intent = "Unknown Intent"
            
            symbol = cls.extract_symbol(query)

        # Map mapped 'intent' to expected backend functions
        if intent == "Automatic N Explanation": 
            return intent, [{"get_auto_n_analysis": {}}, {"get_hybrid_portfolio": {}}]
        elif intent == "Hybrid Portfolio Request": 
            return intent, [{"get_hybrid_portfolio": {}}]
        elif intent == "Governance Check": 
            return intent, [{"get_governance": {}}]
        elif intent == "Calibration Explanation": 
            return intent, [{"get_latest_calibration": {}}]
        elif intent == "Stability Diagnostic": 
            return intent, [{"get_stability_analysis": {}}]
        elif intent == "Meta Portfolio Request": 
            return intent, [{"get_meta_portfolio": {}}]
        elif intent == "Market Overview":
            n_match = re.search(r'top\s+(\d+)', query.lower())
            top_n = int(n_match.group(1)) if n_match else 10
            return intent, [{"list_stocks": {"top_n": top_n, "sector": sector}}]
        elif intent == "Portfolio Comparison":
            return intent, [{"compare_portfolios": {}}]
        elif intent == "Sector Ranking":
            n_match = re.search(r'(?:top|best)\s+(\d+)\s+sectors?', query.lower())
            if not n_match:
                n_match = re.search(r'(?:top|best)\s+(\d+)', query.lower())
            top_n = int(n_match.group(1)) if n_match else 10
            
            s_match = re.search(r'(\d+)\s+stocks?', query.lower())
            stocks_per_sector = int(s_match.group(1)) if s_match else 0
            
            return intent, [{"get_sector_rankings": {"top_n": top_n, "stocks_per_sector": stocks_per_sector}}]
        elif intent == "Portfolio Review":
            return intent, [{"get_portfolio_review": {"portfolio_name": portfolio_name}}]
        elif intent == "Multi-Stock Review":
            return intent, [{"get_multi_stock_review": {"symbols": symbols_list}}]
        elif intent == "Price Movement Analysis":
            days = 365
            if "year" in query.lower():
                y_match = re.search(r'(\d+)\s+year', query.lower())
                days = int(y_match.group(1)) * 365 if y_match else 365
            elif "month" in query.lower():
                m_match = re.search(r'(\d+)\s+month', query.lower())
                days = int(m_match.group(1)) * 30 if m_match else 180
            if symbol == "UNKNOWN": symbol = cls.extract_symbol(query)
            return intent, [{"get_price_movement": {"symbol": symbol, "days": days}}]
        elif intent == "Risk Diagnostics": 
            if symbol == "UNKNOWN": symbol = cls.extract_symbol(query)
            return intent, [{"get_stop_loss": {"symbol": symbol}}]
        elif intent == "Stock Research": 
            if symbol == "UNKNOWN": symbol = cls.extract_symbol(query)
            return intent, [{"get_stock_research": {"symbol": symbol}}]
        elif intent == "Stock Recommendation": 
            if symbol == "UNKNOWN": symbol = cls.extract_symbol(query)
            if symbol != "UNKNOWN":
                return intent, [{"get_stock_summary": {"symbol": symbol}}, {"get_stop_loss": {"symbol": symbol}}]
            
        return "Unknown Intent", []
