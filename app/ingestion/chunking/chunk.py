"""
Chunking entrypoint.

This module separates loading/saving concerns from the chunking algorithm so new
chunking strategies can reuse the same I/O pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from docx import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ingestion.chunking.convert_to_markdown import clean_markdown_text, docx_to_markdown, convert_to_markdown_pypandoc
from app.data_classes import ChunkFile, GlobalConfig
from app.ingestion.chunking.strategies.ChunkStrategyFactor import ChunkStrategyFactor
from app.utils.load_utils import get_index_path, load_document_entry, load_global_config
from app.models import Document as DBDocument


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


def load_text_from_db(database_url: str, document_id: str) -> str:
    """
    Load stored text from the database for a given document_id.

    Args:
        document_id: The document ID to query

    Returns:
        The text content of the document

    Raises:
        FileNotFoundError: If document not found in database or text is missing
    """
    engine = create_engine(database_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        doc = session.query(DBDocument).filter(DBDocument.id == document_id).first()
        
        if not doc:
            raise FileNotFoundError(f"Document with id '{document_id}' not found in database")
        
        if not doc.text:
            raise FileNotFoundError(f"Document text is empty for id '{document_id}'")

        return doc.text
        
    finally:
        session.close()
        engine.dispose()


# ---------------------------------
# Orchestration
# ---------------------------------

def main(config: GlobalConfig, mode: str = "chatbot") -> None:
    if mode == "chatbot":
        strategy = config.chunk_strategy_chatbot
    elif mode == "yoy":
        strategy = config.chunk_strategy_yoy
    elif mode == "factor":
        strategy = ChunkStrategyFactor(method="factor")
    else:
        raise ValueError(f"Unknown strategy key: {mode}")
    
    index_path = get_index_path(config)
    entry = load_document_entry(config.report_id, index_path, config)
    print("entry loaded:", entry.doc_id)
    file_path = entry.source_file_path
    
    # Load markdown text from database
    markdown_text = load_text_from_db(config.POSTGRESQL_URL, entry.doc_id)

    chunks = strategy.create_chunks(markdown_text)
    chunk_file = ChunkFile(strategy=strategy, chunks=chunks)

    output_file = entry.chunks_dir / f"{entry.doc_id}.json"
    output_file.write_text(chunk_file.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Chunked {len(chunks)} chunks for {config.report_id} using {strategy.method} strategy. "
        f"Saved chunks to {output_file}"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk documents using specified strategy.")
    parser.add_argument('--yoy', action='store_true', help='Use YOY strategy')
    parser.add_argument('--chatbot', action='store_true', help='Use chatbot strategy (default)')
    
    args = parser.parse_args()
    
    # Determine strategy key
    if args.yoy:
        mode = "yoy"
    elif args.chatbot:
        mode = "chatbot"
    else:
        mode = "chatbot"  # default
    
    config = load_global_config()
    main(config, mode=mode)