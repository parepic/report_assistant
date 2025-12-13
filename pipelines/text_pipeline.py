"""
Utilities for extracting text from filings (DOCX) and producing chunked text
with configurable chunk size and overlap.

Intended flow:
1) extract_docx_to_blocks(...) -> list of blocks with page/section metadata.
2) chunk_blocks(...) -> chunked records ready for embedding and storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence


@dataclass
class TextBlock:
    page: int
    section: Optional[str]
    text: str


@dataclass
class Chunk:
    chunk_id: str
    page_span: Sequence[int]
    section: Optional[str]
    text: str


def extract_docx_to_blocks(doc_path: Path) -> List[TextBlock]:
    """
    Extract text blocks from a DOCX file, keeping page/section hints where possible.

    Note: Requires `python-docx` if you want richer parsing. This stub uses a
    simple paragraphs-only approach; swap in a more robust extractor as needed.
    """
    try:
        import docx  # type: ignore
    except ImportError as exc:
        raise ImportError("python-docx not installed; pip install python-docx") from exc

    doc = docx.Document(doc_path)
    blocks: List[TextBlock] = []
    for idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue
        # DOCX does not retain page numbers; default to sequential order (1-based).
        blocks.append(TextBlock(page=idx + 1, section=None, text=text))
    return blocks


def chunk_blocks(
    blocks: Iterable[TextBlock],
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 100,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> List[Chunk]:
    """
    Turn text blocks into overlapping chunks.

    - `chunk_size_tokens`: max tokens per chunk.
    - `overlap_tokens`: token overlap between successive chunks.
    - `tokenizer`: optional callable; defaults to `.split()` on whitespace.
    """
    if tokenizer is None:
        tokenizer = lambda s: s.split()

    chunks: List[Chunk] = []
    buffer_tokens: List[str] = []
    buffer_meta: List[TextBlock] = []
    chunk_counter = 0

    def flush_chunk(tokens: List[str], meta: List[TextBlock]) -> None:
        nonlocal chunk_counter
        if not tokens:
            return
        chunk_counter += 1
        page_span = sorted({m.page for m in meta}) or [0]
        text = " ".join(tokens)
        section = next((m.section for m in meta if m.section), None)
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{chunk_counter:05d}",
                page_span=page_span,
                section=section,
                text=text,
            )
        )

    for block in blocks:
        tokens = tokenizer(block.text)
        if not tokens:
            continue
        buffer_tokens.extend(tokens)
        buffer_meta.append(block)

        while len(buffer_tokens) >= chunk_size_tokens:
            chunk_tokens = buffer_tokens[:chunk_size_tokens]
            flush_chunk(chunk_tokens, buffer_meta)

            # Slide the window with the desired overlap.
            buffer_tokens = buffer_tokens[chunk_size_tokens - overlap_tokens :]
            # Keep metadata for overlapping span; conservative approach keeps all seen pages.
            buffer_meta = buffer_meta[-1:]

    # Flush remainder.
    flush_chunk(buffer_tokens, buffer_meta)
    return chunks
