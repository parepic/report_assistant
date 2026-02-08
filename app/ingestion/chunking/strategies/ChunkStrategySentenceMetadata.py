from pydantic import BaseModel
from typing import List, Literal, Optional, Tuple
import re
import spacy


class ChunkStrategySentenceMetadata(BaseModel):
    embed_model: str
    method: Literal["sentence_metadata"] = "sentence_metadata"
    chunk_size: int
    overlap: int
    max_chunk_size: Optional[int] = 2000

    def create_chunks(self, text: str) -> List[dict]:
        """
        Sentence-based chunking (same windowing as ChunkStrategySentence)
        while tracking last seen section and subsection and
        returning per-chunk metadata alongside the text.
        """

        if not text:
            return []

        # Lightweight sentence tokenizer
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

        current_section: Optional[str] = None
        current_subsection: Optional[str] = None

        def _is_heading(block_text: str) -> bool:
            stripped = block_text.lstrip()
            return stripped.startswith("*") or stripped.startswith("#")

        def _clean_heading(block_text: str) -> str:
            return block_text.lstrip("*# ").strip()

        # Collect sentences with associated section/subsection metadata
        sentences_with_meta: List[Tuple[str, Optional[str], Optional[str]]] = []

        # Split on two or more newlines to detect headings cleanly
        raw_blocks = re.split(r"(?:\r?\n){2,}", text.strip())
        _BOLD_HEADING_RE = re.compile(
            r"(?m)(^|\n)\s*(\*{2,})\s*([^\n]+?)\s*\2"
        )
        _HASH_HEADING_RE = re.compile(
            r"(?m)(^|\n)\s*(#{1,6})\s*([^\n]+)"
        )

        def _split_blocks_on_headings(blocks: List[str]) -> List[str]:
            refined: List[str] = []

            for block in blocks:
                remaining = block.strip()

                while remaining:
                    star_match = _BOLD_HEADING_RE.search(remaining)
                    hash_match = _HASH_HEADING_RE.search(remaining)

                    if star_match and hash_match:
                        next_match = star_match if star_match.start() <= hash_match.start() else hash_match
                    else:
                        next_match = star_match or hash_match

                    if not next_match:
                        refined.append(remaining.strip())
                        break

                    prefix = remaining[:next_match.start()].strip()
                    if prefix:
                        refined.append(prefix)

                    heading_block = remaining[next_match.start():next_match.end()].strip()
                    if heading_block:
                        refined.append(heading_block)
                    remaining = remaining[next_match.end():].lstrip()
            return [b for b in refined if b]

        raw_blocks = _split_blocks_on_headings(raw_blocks)

        for idx, block in enumerate(raw_blocks):
            block = block.strip()
            if not block:
                continue

            # Skip Pandoc grid tables that become one gigantic "sentence"
            # if looks_like_grid_table(block):
            #     continue

            # Detect headings based on prefix and lookahead
            if _is_heading(block):
                next_block = ""
                for next_candidate in raw_blocks[idx + 1:]:
                    next_candidate = next_candidate.strip()
                    if next_candidate:
                        next_block = next_candidate
                        break
                if next_block and _is_heading(next_block):
                    current_section = _clean_heading(block)
                    current_subsection = None
                else:
                    current_subsection = _clean_heading(block)
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

        chunks: List[dict] = []

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
                prefix_parts.append(f"risk category: {sec}")
            if subsec:
                prefix_parts.append(f"risk factor: {subsec}")

            prefix_line = " | ".join(prefix_parts)
            if prefix_line:
                chunk_text = prefix_line + "\n" + body
            else:
                chunk_text = body
    
            # Split chunk if it exceeds max_chunk_size
            if self.max_chunk_size and len(chunk_text) > self.max_chunk_size:
                # Split the chunk into smaller parts
                if prefix_line:
                    max_body_len = self.max_chunk_size - len(prefix_line) - 1
                    if max_body_len <= 0:
                        max_body_len = self.max_chunk_size
                    for i in range(0, len(body), max_body_len):
                        chunks.append({
                            "text": prefix_line + "\n" + body[i:i + max_body_len],
                            "risk_category": sec,
                            "risk_factor": subsec,
                        })
                else:
                    for i in range(0, len(chunk_text), self.max_chunk_size):
                        chunks.append({
                            "text": chunk_text[i:i + self.max_chunk_size],
                            "risk_category": sec,
                            "risk_factor": subsec,
                        })
            else:
                chunks.append({
                    "text": chunk_text,
                    "risk_category": sec,
                    "risk_factor": subsec,
                })

            if len(window) < chunk_size:
                break

        return chunks
