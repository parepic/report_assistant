from __future__ import annotations

from typing import List, Dict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, RiskFactor


async def list_documents(session: AsyncSession) -> List[Dict[str, str | int]]:
	result = await session.execute(
		select(Document.id, Document.company, Document.fiscal_year)
	)
	rows = result.all()
	return [
		{
			"doc_id": row.id,
			"company": row.company,
			"fiscal_year": row.fiscal_year,
		}
		for row in rows
	]


async def list_factors(session: AsyncSession, document_id: str) -> List[Dict[str, str | int]]:
	result = await session.execute(
		select(RiskFactor.id, RiskFactor.risk_factor, RiskFactor.text, RiskFactor.idx)
		.filter(RiskFactor.document_id == document_id)
		.order_by(RiskFactor.idx)
	)
	rows = result.all()
	return [
		{
			"id": row.id,
			"risk_factor": row.risk_factor,
			"text": row.text,
			"idx": row.idx,
		}
		for row in rows
	]