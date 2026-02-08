from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dep import get_db
from app.services.db_service import list_documents

router = APIRouter()


class DocumentListItem(BaseModel):
    doc_id: str
    company: str
    fiscal_year: int


@router.get("/documents", response_model=List[DocumentListItem])
async def get_documents(
    session: AsyncSession = Depends(get_db),
) -> List[DocumentListItem]:
    return await list_documents(session=session)
