
# ---------------------------------
# Chunking algorithms
# ---------------------------------

from typing import List
from report_assistant.data_classes import ChunkStrategy
import spacy


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



def chunk_sentences(text: str, strategy: ChunkStrategy) -> List[str]:
    """
    Chunking based on sentence boundaries using Spacy.
    """
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    doc = nlp(text)
    sents = [s.text for s in doc.sents]

    chunk_size = strategy.chunk_size
    overlap = strategy.overlap

    chunks: List[str] = []
    step = chunk_size - overlap

    for start in range(0, len(sents), step):
        window = sents[start : start + chunk_size]
        chunks.append(" ".join(window))
        if len(window) < chunk_size:
            break

    return chunks