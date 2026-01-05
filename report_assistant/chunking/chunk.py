"""
Chunking entrypoint.

This module separates loading/saving concerns from the chunking algorithm so new
chunking strategies can reuse the same I/O pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from docx import Document

from report_assistant.chunking.convert_to_markdown import clean_markdown_text, docx_to_markdown, convert_to_markdown_pypandoc
from report_assistant.data_classes import ChunkFile, GlobalConfig
from report_assistant.utils.load_utils import get_index_path, load_document_entry


# ---------------------------------
# Loading / saving helpers
# ---------------------------------

def load_text(format: str, path: Path) -> str:
    """
    Read all text from a file into a single string.
    """
    if format == "docx":
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    raise ValueError(f"Unsupported document format: {format}")



# ---------------------------------
# Orchestration
# ---------------------------------

def main(config: GlobalConfig) -> None:
    strategy = config.chunk_strategy

    index_path = get_index_path(config)
    entry = load_document_entry(config.report_id, index_path, config)
    file_path = entry.source_file_path
    
    # Removed this for now as something was causing error when embedding

    markdown_text = convert_to_markdown_pypandoc(file_path)
    markdown_text = clean_markdown_text(markdown_text)
    markdown_path = entry.text_dir / f"{entry.doc_id}.md"
    markdown_path.write_text(markdown_text, encoding="utf-8")

    # text = load_text(entry.source_format, file_path)
    # text_path = entry.text_dir / f"{entry.doc_id}.txt"
    # text_path.write_text(text, encoding="utf-8")

    chunks = strategy.create_chunks(markdown_text)
    chunk_file = ChunkFile(strategy=strategy, chunks=chunks)

    output_file = entry.chunks_dir / f"{entry.doc_id}.json"
    output_file.write_text(chunk_file.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Chunked {len(chunks)} chunks for {config.report_id} using {strategy.method} strategy. "
        f"Saved chunks to {output_file} and text to {markdown_path}"
    )



if __name__ == "__main__":
    main()
