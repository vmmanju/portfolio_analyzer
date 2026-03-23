import json
import logging
from typing import Dict, Any, Tuple, List
import urllib.request
import urllib.error

# Keep track of fallback routing and functions
logger = logging.getLogger(__name__)

import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = "llama3.2:latest"

INTENTS_LIST = [
    "Stock Recommendation",
    "Stock Research",
    "Market Overview",
    "Price Movement Analysis",
    "Multi-Stock Review",
    "Portfolio Review",
    "Portfolio Comparison",
    "Hybrid Portfolio Request",
    "Meta Portfolio Request",
    "Risk Diagnostics",
    "Automatic N Explanation",
    "Governance Check",
    "Calibration Explanation",
    "Stability Diagnostic"
]

def _ensure_model():
    """Check if model exists, if not pull it."""
    try:
        check_url = OLLAMA_URL.replace("/api/generate", "/api/tags")
        with urllib.request.urlopen(check_url, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = [m.get("name") for m in data.get("models", [])]
            if any(MODEL_NAME in m for m in models):
                return True
    except Exception:
        pass
        
    logger.info(f"Pulling model {MODEL_NAME}...")
    try:
        pull_url = OLLAMA_URL.replace("/api/generate", "/api/pull")
        payload = {"name": MODEL_NAME, "stream": False}
        req = urllib.request.Request(pull_url, data=json.dumps(payload).encode("utf-8"))
        with urllib.request.urlopen(req, timeout=300) as response:
            return True
    except Exception as e:
        logger.error(f"Failed to pull model: {e}")
        return False

def _call_ollama(prompt: str, json_format: bool = False, system_prompt: str = "") -> str:
    """Wrapper to make an HTTP request to the local Ollama instance."""
    _ensure_model()
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1 # Keep it deterministic
        }
    }
    if system_prompt:
        payload["system"] = system_prompt
    if json_format:
        payload["format"] = "json"
        
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, 
        data=data, 
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode())
            return result.get("response", "")
    except urllib.error.URLError as e:
        logger.warning(f"Ollama connection failed: {e}")
        return ""
    except Exception as e:
        logger.warning(f"Ollama processing failed: {e}")
        return ""

def llm_route_query(user_query: str) -> Tuple[str, str, str, List[str]]:
    """
    Uses Llama 3.2 to classify the intent and extract any stock symbols or portfolio names.
    Returns (intent_name, symbol_string, portfolio_name, symbols_list).
    """
    system_prompt = f"""
    You are an intent routing AI for a financial engine.
    You must classify the user's query into exactly ONE of the following intents:
    {json.dumps(INTENTS_LIST)}
    
    GUIDELINES:
    1. If the user asks for a general list of stocks (e.g. 'Recommend 10 stocks', 'Top 5 performers'), use 'Market Overview' intent.
    2. If the user mentions a specific stock/company:
       - Extract the PRIMARY ticker symbol (e.g. 'Reliance' -> 'RELIANCE.NS', 'Larsen and Toubro' -> 'LT.NS', 'TCS' -> 'TCS.NS').
       - If you are unsure of the ticker, provide the Most Likely NSE ticker followed by '.NS'.
       - Put the result in the "symbol" field.
    3. If the user provides MULTIPLE stocks or asks to COMPARE multiple stocks (e.g. 'Compare TCS and INFY', 'Review TCS, INFY, SBIN'), use 'Multi-Stock Review' and put them ALL in "symbols".
    4. If the intent is 'Portfolio Review' and they mention a named portfolio, extract it into "portfolio_name".
    5. ONLY use 'Portfolio Comparison' if the user explicitly asks to compare "portfolios" or "strategies" (e.g., 'compare my saved portfolios').
    
    Output strictly valid JSON with keys: "intent", "symbol", "portfolio_name", "symbols".
    "symbols" should be an array of strings.
    """
    
    prompt = f"User Query: {user_query}\n\nReturn the JSON object."
    
    response = _call_ollama(prompt, json_format=True, system_prompt=system_prompt)
    if not response:
        return "", "", "", []
        
    try:
        data = json.loads(response)
        intent = data.get("intent", "")
        symbol = data.get("symbol", "")
        portfolio_name = data.get("portfolio_name", "")
        symbols = data.get("symbols", [])
        
        # Validation mapping
        if intent not in INTENTS_LIST:
            intent = "" # force fallback
            
        return intent, symbol, portfolio_name, symbols
    except Exception:
        return "", "", "", []

def llm_generate_explanation(intent: str, user_query: str, engine_data: Dict[str, Any]) -> str:
    """
    Uses Llama 3.2 to generate a human-readable explanation based entirely on structured JSON.
    """
    system_prompt = """
    You are an elite quantitative financial AI assistant.
    You will be provided with a user's question, and a JSON payload representing the exact computed metrics from the backend engine.
    
    CRITICAL RULES:
    1. DO NOT invent metrics. Use ONLY the data provided in the JSON.
    2. DO NOT predict guaranteed returns or offer direct financial advice.
    3. Format the response as a clean, concise markdown report using bullet points.
    4. If explaining "why" a score is high or low, explicitly reference the sub-components found in the JSON data.
    5. Be extremely crisp and professional.
    6. IMPORTANT FOR RANKS: A mathematically smaller rank (e.g. 1) is a BETTER ("higher") structural rank than a larger number (e.g. 200). So Rank 106 is fundamentally better than Rank 217.
    """
    
    prompt = f"User Query: {user_query}\n\nEngine Data JSON:\n```json\n{json.dumps(engine_data, indent=2)}\n```\n\nGenerate the response:"
    
    # Allow slightly higher creativity for explanation formatting, but keep it grounded
    response = _call_ollama(prompt, system_prompt=system_prompt)
    return response.strip()
