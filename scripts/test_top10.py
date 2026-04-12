"""
Quick end-to-end test of the 'List top 10 stocks' query.
Bypasses the LLM explain step (which is now skipped for Market Overview anyway).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve()))

from app.assistant.intent_router import IntentRouter
from app.assistant.function_registry import REGISTRY
from app.assistant.assistant import format_explanation

QUERY = "List top 10 stocks"

print("=== Step 1: Route query ===")
intent, function_calls = IntentRouter.route_query(QUERY)
print(f"  Intent:         {intent}")
print(f"  Function calls: {function_calls}")

print("\n=== Step 2: Execute functions ===")
raw_outputs = {}
for call in function_calls:
    for func_name, args in call.items():
        if func_name in REGISTRY:
            print(f"  Calling {func_name}({args}) ...")
            result = REGISTRY[func_name](args)
            raw_outputs[func_name] = result
            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  OK — {result.get('count', '?')} items returned")

print("\n=== Step 3: Format explanation ===")
explanation = format_explanation(intent, raw_outputs, QUERY)
print(explanation)
