from pydantic import BaseModel
from typing import List, Literal, Optional, Tuple, Dict, Any
import re



class ChunkStrategyParagraph(BaseModel):
    embed_model: str
    method: Literal["paragraph"] = "paragraph"
    max_chunk_chars: Optional[int] = None 

    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Paragraph-based chunking while tracking last seen section and subsection.
        Paragraphs are separated by blank lines.
        Headings do NOT create new chunks (they update metadata only).
        Text does NOT include risk factor/category prefix (metadata only stored in dict).
        
        Returns list of dicts with keys:
        - 'text': the paragraph text (without prefix)
        - 'risk_category': section heading (if any)
        - 'risk_factor': subsection heading (if any)
        """

        if not text:
            return []

        current_section: Optional[str] = None
        current_subsection: Optional[str] = None

        def _is_heading(block_text: str) -> bool:
            stripped = block_text.lstrip()
            return stripped.startswith("*") or stripped.startswith("#")

        def _clean_heading(block_text: str) -> str:
            return block_text.lstrip("*# ").strip()

        # Collect paragraphs with associated section/subsection metadata
        paragraphs_with_meta: List[Tuple[str, Optional[str], Optional[str]]] = []

        # Split on two or more newlines to detect paragraphs and headings
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

            # Non-heading text: treat block as a paragraph
            # Normalize whitespace within the paragraph
            para_text = " ".join(block.split())
            if not para_text:
                continue

            # If paragraph exceeds max_chunk_chars, split it
            if self.max_chunk_chars and len(para_text) > self.max_chunk_chars:
                # Split paragraph into chunks of max_chunk_chars
                for i in range(0, len(para_text), self.max_chunk_chars):
                    chunk_part = para_text[i : i + self.max_chunk_chars]
                    paragraphs_with_meta.append((chunk_part, current_section, current_subsection))
            else:
                paragraphs_with_meta.append((para_text, current_section, current_subsection))

        if not paragraphs_with_meta:
            return []

        # Create dict for each paragraph
        chunks: List[Dict[str, Any]] = []
        for para_text, sec, subsec in paragraphs_with_meta:
            # No prefix - just use the paragraph text as-is
            if self.max_chunk_chars and len(para_text) > self.max_chunk_chars:
                for i in range(0, len(para_text), self.max_chunk_chars):
                    chunks.append({
                        "text": para_text[i:i + self.max_chunk_chars],
                        "risk_category": sec,
                        "risk_factor": subsec
                    })
            else:
                chunk_dict = {
                    "text": para_text,
                    "risk_category": sec,
                    "risk_factor": subsec
                }
                chunks.append(chunk_dict)

        return chunks
