from pydantic import BaseModel
from typing import List, Literal, Optional, Tuple, Dict, Any
import re
from report_assistant.utils.utils import get_heading_level_pattern


class ChunkStrategyParagraphMetadata(BaseModel):
    embed_model: str
    method: Literal["paragraph_metadata"] = "paragraph_metadata"
    max_chunk_chars: Optional[int] = None 

    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Paragraph-based chunking while tracking last seen section and subsection.
        Paragraphs are separated by blank lines.
        Headings do NOT create new chunks (they update metadata only).
        
        Returns list of dicts with keys:
        - 'text': the paragraph text
        - 'risk_category': section heading (if any)
        - 'risk_factor': subsection heading (if any)
        """

        if not text:
            return []

        # Determine heading patterns based on first heading encountered
        section_re, subsection_re = get_heading_level_pattern(text)

        current_section: Optional[str] = None
        current_subsection: Optional[str] = None

        # Collect paragraphs with associated section/subsection metadata
        paragraphs_with_meta: List[Tuple[str, Optional[str], Optional[str]]] = []

        # Split on two or more newlines to detect paragraphs and headings
        raw_blocks = re.split(r"(?:\r?\n){2,}", text.strip())

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

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
            prefix_parts: List[str] = []
            if sec:
                prefix_parts.append(f"risk category: {sec}")
            if subsec:
                prefix_parts.append(f"risk factor: {subsec}")
            prefix_line = " | ".join(prefix_parts)

            if prefix_line:
                chunk_text = prefix_line + "\n" + para_text
            else:
                chunk_text = para_text

            if self.max_chunk_chars and len(chunk_text) > self.max_chunk_chars:
                if prefix_line:
                    max_body_len = self.max_chunk_chars - len(prefix_line) - 1
                    if max_body_len <= 0:
                        max_body_len = self.max_chunk_chars
                    for i in range(0, len(para_text), max_body_len):
                        chunks.append({
                            "text": prefix_line + "\n" + para_text[i:i + max_body_len],
                            "risk_category": sec,
                            "risk_factor": subsec
                        })
                else:
                    for i in range(0, len(para_text), self.max_chunk_chars):
                        chunks.append({
                            "text": para_text[i:i + self.max_chunk_chars],
                            "risk_category": sec,
                            "risk_factor": subsec
                        })
            else:
                chunk_dict = {
                    "text": chunk_text,
                    "risk_category": sec,
                    "risk_factor": subsec
                }
                chunks.append(chunk_dict)

        return chunks