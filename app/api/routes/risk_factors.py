from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dep import get_db
from app.api.routes.documents import DocumentListItem
from app.services.db_service import list_factors

router = APIRouter()


class RiskFactorListItem(BaseModel):
    id: int
    risk_factor: str
    text: str
    idx: int


@router.get("/risk_factors", response_model=List[RiskFactorListItem])
async def get_risk_factors(
    document_id: str,
    session: AsyncSession = Depends(get_db),
) -> List[RiskFactorListItem]:
    return await list_factors(session=session, document_id=document_id)
