from pydantic import BaseModel
from typing import List, Literal, Optional, Tuple
import re
import spacy
from report_assistant.utils.utils import get_heading_level_pattern


class ChunkStrategySentenceMetadata(BaseModel):
    embed_model: str
    method: Literal["sentence_metadata"] = "sentence_metadata"
    chunk_size: int
    overlap: int
    max_chunk_size: Optional[int] = 2000

    def create_chunks(self, text: str) -> List[str]:
        """
        Sentence-based chunking (same windowing as ChunkStrategySentence)
        while tracking last seen section and subsection and
        prepending them into each chunk as plain-text metadata.
        """

        if not text:
            return []

        # Determine heading patterns based on first heading encountered
        section_re, subsection_re = get_heading_level_pattern(text)

        # Lightweight sentence tokenizer
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

        current_section: Optional[str] = None
        current_subsection: Optional[str] = None

        # Collect sentences with associated section/subsection metadata
        sentences_with_meta: List[Tuple[str, Optional[str], Optional[str]]] = []

        # Split on two or more newlines to detect headings cleanly
        raw_blocks = re.split(r"(?:\r?\n){2,}", text.strip())

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            # Skip Pandoc grid tables that become one gigantic "sentence"
            # if looks_like_grid_table(block):
            #     continue

            # Detect section
            m_sec = section_re.match(block)
            if m_sec:
                current_section = m_sec.group(1).strip()
                current_subsection = None
                continue

            # Detect subsection
            m_sub = subsection_re.match(block)
            if m_sub:
                current_subsection = m_sub.group(1).strip()
                continue

            # Non-heading text: split into sentences and normalize
            doc = nlp(block)
            for sent in doc.sents:
                sent_text = " ".join(sent.text.split())
                if len(sent_text) > 2500:
                    # Skip pathological sentences (usually tables without punctuation)
                    continue
                if not sent_text:
                    continue
                sentences_with_meta.append((sent_text, current_section, current_subsection))

        if not sentences_with_meta:
            return []

        chunk_size = self.chunk_size
        overlap = self.overlap
        step = chunk_size - overlap

        chunks: List[str] = []

        for start in range(0, len(sentences_with_meta), step):
            window = sentences_with_meta[start : start + chunk_size]
            if not window:
                break

            # Build body from sentences in the window
            body = " ".join(sent for sent, _, _ in window)

            # Derive metadata from the first sentence in the window
            _, sec, subsec = window[0]
            prefix_parts: List[str] = []
            if sec:
                prefix_parts.append(f"Section: {sec}")
            if subsec:
                prefix_parts.append(f"Subsection: {subsec}")

            if prefix_parts:
                chunk_text = " | ".join(prefix_parts) + "\n" + body
            else:
                chunk_text = body
    
            # Split chunk if it exceeds max_chunk_size
            if self.max_chunk_size and len(chunk_text) > self.max_chunk_size:
                # Split the chunk into smaller parts
                for i in range(0, len(chunk_text), self.max_chunk_size):
                    chunks.append(chunk_text[i:i + self.max_chunk_size])
            else:
                chunks.append(chunk_text)

            if len(window) < chunk_size:
                break

        return chunks
