"""Shared helper layer for test modules.

This module centralizes lightweight factories used across multiple test files.
The goal is to keep individual test modules focused on behavior assertions,
while reusing consistent object construction for common runtime inputs.

Context:
- Production code paths under test accept objects with a small set of
  attributes (config, document entry, chunk file), not always full ORM/Pydantic
  entities.
- Recreating these objects inline in every test adds noise and makes behavior
  diffs harder to read.

Helper responsibilities:
- `make_qdrant_config`: Minimal config object for wrapper constructor tests.
- `make_real_document_entry`: Builds a real `DocumentEntry` for ingestion flows.
- `make_real_chunk_file_fixed_size`: Builds a real `ChunkFile` so strategy hash
  generation is exercised exactly as production does.
- `make_runtime_config`: Simulates runtime config with embedding profile.

These helpers are intentionally deterministic so test assertions remain stable.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.data_classes import ChunkFile, DocumentEntry, EmbeddingProfileConfig
from app.ingestion.chunking.strategies.ChunkStrategyFixedSize import ChunkStrategyFixedSize


class FakeSentenceMetadataStrategy:
    """Minimal strategy double exposing the subset used by save_qdrant tests."""

    def model_dump(self) -> dict:
        """Return a strategy payload shape equivalent to sentence-metadata mode."""
        return {"method": "sentence_metadata"}


def make_qdrant_config(url: str = "http://qdrant:6333") -> SimpleNamespace:
    """Build a minimal config accepted by `QdrantClientWrapper` constructor."""
    return SimpleNamespace(QDRANT_URL=url)


def make_real_document_entry() -> DocumentEntry:
    """Build a real `DocumentEntry` object used by ingestion orchestration tests."""
    return DocumentEntry(
        doc_id="doc-1",
        company="Microsoft",
        fiscal_year=2024,
        source_file_path=Path(__file__),
        chunks_dir=Path("app/output/company__microsoft/chunks"),
    )


def make_real_chunk_file_fixed_size(
    chunk_size: int = 200,
    overlap: int = 20,
) -> ChunkFile:
    """Build a real `ChunkFile` so strategy hashes are computed as in production."""
    strategy = ChunkStrategyFixedSize(chunk_size=chunk_size, overlap=overlap)
    return ChunkFile(
        strategy=strategy,
        chunks=[
            {"text": "Risk paragraph one", "risk_factor": "Operations"},
            {"text": "Risk paragraph two", "risk_factor": "Regulation"},
        ],
    )


def make_runtime_config(
    provider: str = "ollama",
    embed_model: str = "nomic-embed-text",
) -> SimpleNamespace:
    """Build a minimal runtime config used across `save_qdrant.main` tests."""
    return SimpleNamespace(
        report_id="doc-1",
        OLLAMA_URL="http://ollama:11434",
        QDRANT_DB_NAME_CHATBOT="report_assistant_chatbot",
        QDRANT_DB_NAME_YOY="report_assistant_yoy",
        EMBEDDING_PROFILE=EmbeddingProfileConfig(
            provider=provider,
            embed_model=embed_model,
        ),
    )
