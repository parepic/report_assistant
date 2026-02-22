from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dep import get_config, get_openai_client_chatbot, get_qdrant_client
from app.data_classes import GlobalConfig
from app.clients.QdrantClientWrapper import QdrantClientWrapper
from app.clients.OpenAiClientWrapper import OpenAIClientWrapper
from app.services.chatbot import main as chatbot_main

router = APIRouter()


class ChatRequest(BaseModel):
    doc_id: str
    prompt: str


@router.post("/chatbot")
async def chatbot(
    request: ChatRequest,
    config: GlobalConfig = Depends(get_config),
    qdrant_client: QdrantClientWrapper = Depends(get_qdrant_client),
    openai_client: OpenAIClientWrapper = Depends(get_openai_client_chatbot),
) -> dict:
    return chatbot_main(
        config=config,
        prompt=request.prompt,
        doc_id=request.doc_id,
        qdrant_client=qdrant_client,
        openai_client=openai_client,
    )
