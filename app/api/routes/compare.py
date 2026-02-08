from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dep import get_config, get_openai_client_comparison, get_qdrant_client, get_db
from app.data_classes import GlobalConfig
from app.clients.QdrantClientWrapper import QdrantClientWrapper
from app.clients.OpenAiClientWrapper import OpenAIClientWrapper
from app.services.comparison import main as comparison_main
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


class CompareRequest(BaseModel):
    doc_id: str


@router.post("/comparison")
async def comparison(
    request: CompareRequest,
    config: GlobalConfig = Depends(get_config),
    qdrant_client: QdrantClientWrapper = Depends(get_qdrant_client),
    openai_client: OpenAIClientWrapper = Depends(get_openai_client_comparison),
    session: AsyncSession = Depends(get_db),
) -> dict:
    return await comparison_main(
        config=config,
        doc_id=request.doc_id,
        qdrant_client=qdrant_client,
        openai_client=openai_client,
        session=session
    )
