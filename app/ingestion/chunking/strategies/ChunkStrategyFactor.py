"""
Factor-level chunking strategy.

This strategy groups report content by detected risk-factor sections.
Each produced chunk contains the full text body that appears under
one risk-factor heading and continues until the next risk-factor heading.

Heading detection mirrors the existing paragraph-based strategy so behavior
stays consistent across chunking modes.
"""

from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any
import re


class ChunkStrategyFactor(BaseModel):
    """
    Chunk strategy that emits one chunk per detected risk factor.

    `method` is used by Pydantic's discriminator in the strategy union.
    """

    method: Literal["factor"] = "factor"

    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Build factor-level chunks from the input report text.

        Behavior:
        - Detects headings using the same markdown-style logic as paragraph strategy.
        - Treats a heading as a risk factor only when the next non-empty block is not
          another heading.
        - Collects all subsequent non-heading content under that risk factor until a
          new risk factor is encountered.
                - Emits one chunk per risk factor, where chunk text is the full section
                    content for that factor while preserving paragraph boundaries.

        Args:
            text: Full source text of the report section to chunk.

        Returns:
            List of chunk dictionaries. Each dictionary contains:
            - `text`: full body text for one risk factor section.
            - `risk_factor`: cleaned heading text identifying that factor.
        """

        if not text:
            return []

        def _is_heading(block_text: str) -> bool:
            """Return True when a block starts with markdown heading markers."""
            stripped = block_text.lstrip()
            return stripped.startswith("*") or stripped.startswith("#")

        def _clean_heading(block_text: str) -> str:
            """Normalize heading text by stripping markdown markers and spacing."""
            return block_text.lstrip("*# ").strip()

        # Split on blank lines first, then refine blocks so embedded headings
        # inside larger blocks are isolated as separate entries.
        raw_blocks = re.split(r"(?:\r?\n){2,}", text.strip())
        bold_heading_re = re.compile(r"(?m)(^|\n)\s*(\*{2,})\s*([^\n]+?)\s*\2")
        hash_heading_re = re.compile(r"(?m)(^|\n)\s*(#{1,6})\s*([^\n]+)")

        def _split_blocks_on_headings(blocks: List[str]) -> List[str]:
            """
            Refine initial blocks by extracting markdown headings into standalone blocks.

            This ensures heading detection works even when converted text contains
            heading and body content in the same initial block.
            """

            refined: List[str] = []

            for block in blocks:
                remaining = block.strip()

                while remaining:
                    star_match = bold_heading_re.search(remaining)
                    hash_match = hash_heading_re.search(remaining)

                    if star_match and hash_match:
                        next_match = (
                            star_match
                            if star_match.start() <= hash_match.start()
                            else hash_match
                        )
                    else:
                        next_match = star_match or hash_match

                    if not next_match:
                        refined.append(remaining.strip())
                        break

                    prefix = remaining[: next_match.start()].strip()
                    if prefix:
                        refined.append(prefix)

                    heading_block = remaining[next_match.start() : next_match.end()].strip()
                    if heading_block:
                        refined.append(heading_block)

                    remaining = remaining[next_match.end() :].lstrip()

            return [block for block in refined if block]

        raw_blocks = _split_blocks_on_headings(raw_blocks)

        chunks: List[Dict[str, Any]] = []
        current_risk_factor: Optional[str] = None
        current_lines: List[str] = []

        def _flush_current_factor() -> None:
            """Append the currently buffered factor section as one chunk if valid."""
            if not current_risk_factor:
                return

            section_text = "\n\n".join(
                part.strip()
                for part in current_lines
                if part and part.strip()
            ).strip()
            if not section_text:
                return

            chunks.append({
                "text": section_text,
                "risk_factor": current_risk_factor,
            })

        for idx, block in enumerate(raw_blocks):
            block = block.strip()
            if not block:
                continue

            if _is_heading(block):
                next_block = ""
                for next_candidate in raw_blocks[idx + 1 :]:
                    next_candidate = next_candidate.strip()
                    if next_candidate:
                        next_block = next_candidate
                        break

                # Heading followed by heading means structural heading, not factor.
                if next_block and _is_heading(next_block):
                    continue

                # New risk factor begins; persist previous section first.
                _flush_current_factor()
                current_risk_factor = _clean_heading(block)
                current_lines = []
                continue

            # Append body text only when currently inside a risk-factor section.
            if current_risk_factor:
                if block:
                    current_lines.append(block)

        _flush_current_factor()
        return chunks
