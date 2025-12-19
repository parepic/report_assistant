"""
Chunking entrypoint.

This module separates loading/saving concerns from the chunking algorithm so new
chunking strategies can reuse the same I/O pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from docx import Document

from report_assistant.chunking.algorithms import chunk_sequential, chunk_sentences
from report_assistant.chunking.convert_to_markdown import convert_to_markdown_pypandoc
from report_assistant.data_classes import ChunkFile, ChunkStrategy, DocumentEntry, GlobalConfig
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


def run_chunking(text: str, strategy: ChunkStrategy) -> List[str]:
    """
    Dispatch chunking based on strategy.method.
    """
    if strategy.method == "sequential":
        return chunk_sequential(text, strategy)
    if strategy.method == "sentences":
        return chunk_sentences(text, strategy)
    
    raise ValueError(f"Unknown chunking method: {strategy.method}")



# ---------------------------------
# Orchestration
# ---------------------------------

def main(config: GlobalConfig) -> None:
    strategy = config.chunk_strategy

    index_path = get_index_path(config)
    entry = load_document_entry(config.report_id, index_path, config)
    file_path = entry.source_file_path

    # Not necessary if we can convert to an .md file and use its contents directly
    # text = load_text(entry.source_format, file_path)
    # text_file = save_plain_text(entry, text)
    markdown_text = convert_to_markdown_pypandoc(file_path)

    markdown_path = entry.text_dir / f"{entry.doc_id}.md"
    markdown_path.write_text(markdown_text, encoding="utf-8")


    chunks = run_chunking(markdown_text, strategy)
    chunk_file = ChunkFile(strategy=strategy, chunks=chunks)

    output_file = entry.chunks_dir / f"{entry.doc_id}.json"
    output_file.write_text(chunk_file.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Chunked {len(chunks)} chunks for {config.report_id} using {strategy.method} strategy. "
        f"Saved chunks to {output_file} and markdown to {markdown_path}"
    )


if __name__ == "__main__":
    main()
