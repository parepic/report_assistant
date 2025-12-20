from pydantic import BaseModel
from typing import List, Literal
import spacy

class ChunkStrategySentence(BaseModel):
    embed_model: str
    method: Literal["sentence"] = "sentence"
    chunk_size: int
    overlap: int

    def create_chunks(self, text: str) -> List[str]:
        """
        Chunking based on sentence boundaries using Spacy.
        """
        
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

        doc = nlp(text)
        sents = [s.text for s in doc.sents]

        chunk_size = self.chunk_size
        overlap = self.overlap

        chunks: List[str] = []
        step = chunk_size - overlap

        for start in range(0, len(sents), step):
            window = sents[start : start + chunk_size]
            chunks.append(" ".join(window))
            if len(window) < chunk_size:
                break

        return chunks