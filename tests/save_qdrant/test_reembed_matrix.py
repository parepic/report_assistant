"""`save_qdrant.main` tests: re-embedding decision matrix.

Scope of this file:
- Encodes the core idempotency matrix for vector ingestion:
  - same strategy + same profile
  - changed strategy + same profile
  - same strategy + changed profile
- Validates that collection routing and strategy-hash checks combine correctly.

Why these tests exist:
- The re-embed decision contract is the most important safety behavior after
  introducing embedding profiles and strategy-hash based filtering.
- Regressions here can cause unnecessary recompute or accidental skips.

Scenario map:
- Skip when same profile and same strategy already exist.
- Re-embed when strategy hash changes under same profile.
- Re-embed when embedding profile changes under same strategy.
- Assert profile-derived collection names are used for existence checks.
- Assert existence checks can use a real computed strategy hash from a real
  strategy object.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from app.ingestion.save_qdrant import main
from tests.helpers import (
    make_real_document_entry,
    make_real_chunk_file_fixed_size,
    make_runtime_config,
)


@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
@patch("builtins.input", return_value="no")
def test_skips_reembedding_when_same_strategy_and_same_profile_exist(
    mock_input: MagicMock,
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert_to_company_collection: MagicMock,
) -> None:
    """
    Validate idempotent "no-op" behavior for an already-embedded strategy/profile pair.

    Given:
    - We are ingesting `doc-1` into the chatbot collection.
    - The active embedding profile is `ollama + nomic-embed-text`.
    - The loaded chunk file is a real `ChunkFile`; strategy hash is computed
      by production code from `ChunkStrategyFixedSize(chunk_size=200, overlap=20)`.
    - Qdrant reports existing_count > 0 for exactly:
      (collection=`report_assistant_chatbot__ollama_nomic-embed-text`,
       strategy_hash=<computed from real strategy>,
       doc_id=`doc-1`).

    When:
    - `save_qdrant.main(...)` runs and detects existing vectors.
    - User chooses "no" at overwrite prompt.

    Then:
    - The ingestion should stop early (idempotent skip path).
    - No re-embedding should happen (`embed_chunks` is not called).
    - No upsert should happen (`upsert_to_company_collection` is not called).

    Why this matters:
    - Prevents duplicate embeddings and unnecessary compute/cost when rerunning
      ingestion for an unchanged document+strategy under the same embedding profile.
    """
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = make_real_document_entry()
    real_chunk_file = make_real_chunk_file_fixed_size(chunk_size=200, overlap=20)
    mock_load_chunks.return_value = real_chunk_file
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2
    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 5
    mock_qdrant.get_collection_vector_dim.return_value = 2
    mock_qdrant_cls.return_value = mock_qdrant
    config = make_runtime_config(provider="ollama", embed_model="nomic-embed-text")

    main(config=config, mode="chatbot")

    mock_qdrant.count_existing_points.assert_called_once_with(
        "report_assistant_chatbot__ollama_nomic-embed-text",
        real_chunk_file.strategy_hash,
        "doc-1",
    )
    mock_input.assert_called_once()
    mock_embed_chunks.assert_not_called()
    mock_upsert_to_company_collection.assert_not_called()


@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
def test_reembeds_when_strategy_changes_with_same_embedding_profile(
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert_to_company_collection: MagicMock,
) -> None:
    """
    Validate re-embed behavior when chunk strategy changes under the same profile.

    Given:
    - Same document (`doc-1`) and same embedding profile
      (`ollama + nomic-embed-text`).
    - Current run uses real strategy A (`chunk_size=220`, `overlap=20`) and
      therefore real computed hash A.
    - We simulate that old strategy data may exist for real strategy B
      (`chunk_size=200`, `overlap=20`) and real computed hash B, but
      current strategy hash A does not exist yet.

    When:
    - `save_qdrant.main(...)` checks Qdrant for existing vectors using
      current strategy hash A.

    Then:
    - Existence check must use hash A, not old hash B.
    - Pipeline should proceed with embedding + upsert.

    Why this matters:
    - Confirms strategy changes are treated as a new index variant, even if the
      same document/profile was previously processed with a different strategy.
    """
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = make_real_document_entry()
    current_chunk_file = make_real_chunk_file_fixed_size(chunk_size=220, overlap=20)
    old_chunk_file = make_real_chunk_file_fixed_size(chunk_size=200, overlap=20)
    mock_load_chunks.return_value = current_chunk_file
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection_vector_dim.return_value = 2

    def _count_side_effect(collection_name: str, strategy_hash: str, doc_id: str) -> int:
        if strategy_hash == old_chunk_file.strategy_hash:
            return 10
        return 0

    mock_qdrant.count_existing_points.side_effect = _count_side_effect
    mock_qdrant_cls.return_value = mock_qdrant
    config = make_runtime_config(provider="ollama", embed_model="nomic-embed-text")

    main(config=config, mode="chatbot")

    mock_qdrant.count_existing_points.assert_called_once_with(
        "report_assistant_chatbot__ollama_nomic-embed-text",
        current_chunk_file.strategy_hash,
        "doc-1",
    )
    mock_embed_chunks.assert_called_once()
    mock_upsert_to_company_collection.assert_called_once()


@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
def test_reembeds_when_embedding_profile_changes_with_same_strategy(
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert_to_company_collection: MagicMock,
) -> None:
    """
    Validate re-embed behavior when embedding profile changes but strategy is same.

    Given:
    - Same document (`doc-1`) and same real computed strategy hash from
      `ChunkStrategyFixedSize(chunk_size=200, overlap=20)`.
    - Active profile for this run is `openai + text-embedding-3-small`.
    - Old embeddings might already exist under another profile collection
      (`ollama + nomic-embed-text`).

    When:
    - `save_qdrant.main(...)` performs existence check.

    Then:
    - Check must target only the openai-derived collection
      (`report_assistant_chatbot__openai_text-embedding-3-small`).
    - Existing vectors in old-profile collections must not block ingest.
    - Embedding + upsert should run for the new profile.

    Why this matters:
    - Guarantees profile isolation: each embedding model/provider keeps its own
      index space, enabling side-by-side experiments and migrations.
    """
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = make_real_document_entry()
    real_chunk_file = make_real_chunk_file_fixed_size(chunk_size=200, overlap=20)
    mock_load_chunks.return_value = real_chunk_file
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection_vector_dim.return_value = 2

    def _count_side_effect(collection_name: str, strategy_hash: str, doc_id: str) -> int:
        if collection_name == "report_assistant_chatbot__ollama_nomic-embed-text":
            return 10
        return 0

    mock_qdrant.count_existing_points.side_effect = _count_side_effect
    mock_qdrant_cls.return_value = mock_qdrant
    config = make_runtime_config(provider="openai", embed_model="text-embedding-3-small")

    main(config=config, mode="chatbot")

    mock_qdrant.count_existing_points.assert_called_once_with(
        "report_assistant_chatbot__openai_text-embedding-3-small",
        real_chunk_file.strategy_hash,
        "doc-1",
    )
    mock_embed_chunks.assert_called_once()
    mock_upsert_to_company_collection.assert_called_once()


@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
def test_main_uses_profile_specific_collection_for_existence_check(
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert_to_company_collection: MagicMock,
) -> None:
    """
    Validate collection routing contract used by existence checks.

    Given:
    - Active embedding profile is `openai + text-embedding-3-small`.
    - Strategy hash is production-computed from
      `ChunkStrategyFixedSize(chunk_size=200, overlap=20)` and doc id is `doc-1`.

    When:
    - `save_qdrant.main(...)` asks Qdrant whether vectors already exist.

    Then:
    - Collection name must be derived from profile:
      `report_assistant_chatbot__openai_text-embedding-3-small`.
    - Count call should include the same strategy/doc identifiers.
    - Upsert path should execute if count returns zero.

    Why this matters:
    - This assertion is the core guard against cross-profile false positives in
      duplicate detection logic.
    """
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = make_real_document_entry()
    real_chunk_file = make_real_chunk_file_fixed_size(chunk_size=200, overlap=20)
    mock_load_chunks.return_value = real_chunk_file
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2
    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 0
    mock_qdrant.get_collection_vector_dim.return_value = 2
    mock_qdrant_cls.return_value = mock_qdrant
    config = make_runtime_config(provider="openai", embed_model="text-embedding-3-small")

    main(config=config, mode="chatbot")

    mock_qdrant.count_existing_points.assert_called_once_with(
        "report_assistant_chatbot__openai_text-embedding-3-small",
        real_chunk_file.strategy_hash,
        "doc-1",
    )
    mock_upsert_to_company_collection.assert_called_once()


@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
def test_main_uses_real_computed_strategy_hash_in_existence_check(
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert_to_company_collection: MagicMock,
) -> None:
    """
    Validate that existence checks use production-computed strategy hashes.

    Given:
    - A real `ChunkFile` built with a real `ChunkStrategyFixedSize` object.
    - Strategy hash is computed by production code (not manually injected).
    - Active profile is `ollama + nomic-embed-text`.

    When:
    - `save_qdrant.main(...)` runs and evaluates whether current vectors exist.

    Then:
    - Qdrant count call must use `real_chunk_file.strategy_hash` exactly.
    - If count is zero, ingestion continues to upsert.

    Why this matters:
    - Prevents tests from only passing with mocked hash literals; confirms
      integration between strategy object serialization and ingest dedupe logic.
    """
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = make_real_document_entry()
    real_chunk_file = make_real_chunk_file_fixed_size(chunk_size=180, overlap=30)
    mock_load_chunks.return_value = real_chunk_file
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2
    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 0
    mock_qdrant.get_collection_vector_dim.return_value = 2
    mock_qdrant_cls.return_value = mock_qdrant
    config = make_runtime_config(provider="ollama", embed_model="nomic-embed-text")

    main(config=config, mode="chatbot")

    mock_qdrant.count_existing_points.assert_called_once_with(
        "report_assistant_chatbot__ollama_nomic-embed-text",
        real_chunk_file.strategy_hash,
        "doc-1",
    )
    mock_upsert_to_company_collection.assert_called_once()
