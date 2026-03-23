from fastapi import APIRouter
from app.assistant.schemas import AssistantRequest, AssistantResponse
from app.assistant.assistant import process_query

router = APIRouter(prefix="/assistant", tags=["assistant"])

@router.post("/query", response_model=AssistantResponse)
def query_assistant(req: AssistantRequest):
    return process_query(req)
