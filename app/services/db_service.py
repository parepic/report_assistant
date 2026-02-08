from __future__ import annotations

from typing import List, Dict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document


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
