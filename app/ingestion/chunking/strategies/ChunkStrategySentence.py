from pydantic import BaseModel
from typing import List, Literal, Optional
import spacy


class ChunkStrategySentence(BaseModel):
    embed_model: str
    method: Literal["sentence"] = "sentence"
    chunk_size: int
    overlap: int
    max_chunk_chars: Optional[int] = 2500  # hard cap to avoid oversized table chunks

    def create_chunks(self, text: str) -> List[str]:
        """
        Chunking based on sentence boundaries using Spacy with optional character
        cap to prevent extremely long table-only chunks from being embedded.
        """

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

        doc = nlp(text)
        sents = [s.text for s in doc.sents]

        def split_long_chunk(chunk: str) -> List[str]:
            max_len = self.max_chunk_chars
            if not max_len or len(chunk) <= max_len:
                return [chunk]

            parts: List[str] = []
            words = chunk.split()
            current: List[str] = []
            current_len = 0

            for word in words:
                if current and current_len + 1 + len(word) > max_len:
                    parts.append(" ".join(current))
                    current = [word]
                    current_len = len(word)
                else:
                    if current:
                        current_len += 1 + len(word)
                    else:
                        current_len = len(word)
                    current.append(word)

            if current:
                parts.append(" ".join(current))

            return parts

        chunk_size = self.chunk_size
        overlap = self.overlap

        chunks: List[str] = []
        step = chunk_size - overlap

        for start in range(0, len(sents), step):
            window = sents[start : start + chunk_size]
            combined = " ".join(window)
            for part in split_long_chunk(combined):
                chunks.append(part)
            if len(window) < chunk_size:
                break

        return chunks
