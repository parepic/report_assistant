from pydantic import BaseModel
from typing import List, Literal

class ChunkStrategyFixedSize(BaseModel):
    embed_model: str
    method: Literal["fixed_size"] = "fixed_size"
    chunk_size: int
    overlap: int

    def create_chunks(self, text: str) -> List[str]:
        """
        Simple sequential overlapping chunking based on character counts.
        """
        chunk_size = self.chunk_size
        overlap = self.overlap

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(chunk_text)
            start += chunk_size - overlap
        return chunks
