
# ---------------------------------
# Chunking algorithms
# ---------------------------------

from typing import List
from report_assistant.data_classes import ChunkStrategy


def chunk_sequential(text: str, strategy: ChunkStrategy) -> List[str]:
    """
    Simple sequential overlapping chunking based on character counts.
    """
    chunk_size = strategy.chunk_size
    overlap = strategy.overlap

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks