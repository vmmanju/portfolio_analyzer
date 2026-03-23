from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class AssistantRequest(BaseModel):
    user_query: str
    context: Optional[Dict[str, Any]] = None

class AssistantResponse(BaseModel):
    intent: str
    functions_called: List[str]
    explanation: str
    raw_outputs: Dict[str, Any]
