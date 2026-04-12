import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve()))

from app.assistant.intent_router import IntentRouter
try:
    intent, funcs = IntentRouter.route_query("List top 10 stocks")
    print("Intent:", intent)
    print("Funcs:", funcs)
except Exception as e:
    print("Error:", str(e))
