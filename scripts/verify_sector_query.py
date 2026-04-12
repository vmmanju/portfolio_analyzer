import os
import sys
from pathlib import Path

# Add project root to sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.assistant.assistant import process_query
from app.assistant.schemas import AssistantRequest

def verify_sector_query():
    query = "Show me top 10 sectors"
    print(f"Query: {query}")
    print("-" * 40)
    
    req = AssistantRequest(user_query=query, user_id=1)
    # We use a mock or ensure Ollama is not called if we just want to see the function execution and python formatting
    # In a real environment, it would call Ollama. Here we can check if it at least executes and returns something.
    
    # To test WITH Ollama, we remove the patches
    res = process_query(req)
            
    print(f"Intent detected: {res.intent}")
    print(f"Functions called: {res.functions_called}")
    
    with open("assistant_result.md", "w", encoding="utf-8") as f:
        f.write(res.explanation)
    print("-" * 40)
    print("Assistant Explanation written to assistant_result.md")

if __name__ == "__main__":
    verify_sector_query()
