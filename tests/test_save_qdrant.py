"""Unit tests for save_qdrant orchestration and payload writing.

These tests verify local control-flow and payload construction while all
external IO is mocked (Qdrant network, embedding calls, file loading).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.data_classes import EmbeddingProfileConfig, GlobalConfig
from app.ingestion.save_qdrant import main, upsert_to_company_collection


class _FakeStrategy:
    """Minimal strategy test double exposing fields used by save_qdrant."""

    def __init__(self, embed_model: str = "nomic-embed-text") -> None:
        self.embed_model = embed_model

    def model_dump(self) -> dict:
        """Return strategy metadata payload expected by ingestion code."""
        return {"method": "sentence_metadata", "embed_model": self.embed_model}


def _fake_entry() -> SimpleNamespace:
    """Build a minimal document entry used by save_qdrant tests."""
    return SimpleNamespace(
        doc_id="doc-1",
        company="Microsoft",
        fiscal_year=2024,
        chunks_dir=Path("app/output/company__microsoft/chunks"),
    )


def _fake_chunks_file() -> SimpleNamespace:
    """Build a minimal chunk file object used by save_qdrant tests."""
    return SimpleNamespace(
        strategy=_FakeStrategy(embed_model="nomic-embed-text"),
        strategy_hash="strategy-hash-1",
        chunks=[
            {"text": "Risk paragraph one", "risk_factor": "Operations"},
            {"text": "Risk paragraph two", "risk_factor": "Regulation"},
        ],
    )


def _fake_config() -> SimpleNamespace:
    """Build a minimal config object for save_qdrant main-flow tests."""
    return SimpleNamespace(
        report_id="doc-1",
        OLLAMA_URL="http://ollama:11434",
        QDRANT_DB_NAME_CHATBOT="report_assistant_chatbot",
        QDRANT_DB_NAME_YOY="report_assistant_yoy",
        EMBEDDING_PROFILE=EmbeddingProfileConfig(
            provider="ollama",
            embed_model="nomic-embed-text",
        ),
    )


def test_upsert_to_company_collection_batches_and_includes_payload_fields() -> None:
    """Writes expected payload metadata and performs 128-sized batched upserts."""
    class _RecordingClient:
        """Capture snapshots of upserted points to avoid mutable-list assertion issues."""

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def upsert(self, collection_name: str, points: list) -> None:
            """Store immutable snapshots of every upsert call."""
            self.calls.append(
                {
                    "collection_name": collection_name,
                    "points": list(points),
                }
            )

    recording_client = _RecordingClient()
    strategy = _FakeStrategy(embed_model="nomic-embed-text")
    chunk_file = SimpleNamespace(strategy=strategy, strategy_hash="hash-123")
    entry = SimpleNamespace(doc_id="doc-x", company="Apple", fiscal_year=2025)

    chunks = [
        {"text": f"text-{index}", "risk_factor": "Liquidity"}
        for index in range(130)
    ]
    vectors = [np.array([0.1, 0.2], dtype="float32") for _ in range(130)]

    upsert_to_company_collection(
        client=recording_client,
        collection_name="chatbot_idx",
        chunks=chunks,
        vectors=vectors,
        chunk_file=chunk_file,
        entry_file=entry,
    )

    assert len(recording_client.calls) == 2
    first_batch_points = recording_client.calls[0]["points"]
    second_batch_points = recording_client.calls[1]["points"]
    assert len(first_batch_points) == 128
    assert len(second_batch_points) == 2
    first_payload = first_batch_points[0].payload
    assert first_payload["doc_id"] == "doc-x"
    assert first_payload["company"] == "apple"
    assert first_payload["fiscal_year"] == 2025
    assert first_payload["strategy_hash"] == "hash-123"
    assert first_payload["chunk_idx"] == 0
    assert first_payload["risk_factor"] == "Liquidity"


def test_upsert_to_company_collection_rejects_mismatched_lengths() -> None:
    """Raises a clear error when chunk and vector lengths do not match."""
    strategy = _FakeStrategy(embed_model="nomic-embed-text")
    chunk_file = SimpleNamespace(strategy=strategy, strategy_hash="hash-123")
    entry = SimpleNamespace(doc_id="doc-x", company="Apple", fiscal_year=2025)

    with pytest.raises(ValueError, match="Chunks count does not match vectors count"):
        upsert_to_company_collection(
            client=MagicMock(),
            collection_name="chatbot_idx",
            chunks=[{"text": "a", "risk_factor": "Liquidity"}],
            vectors=[],
            chunk_file=chunk_file,
            entry_file=entry,
        )


@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
def test_main_chatbot_mode_runs_full_noninteractive_flow(
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert_to_company_collection: MagicMock,
) -> None:
    """Creates collection, embeds chunks, indexes payload fields, and upserts."""
    mock_get_index_path.return_value = "app/data/index.json"
    fake_entry = _fake_entry()
    fake_chunks_file = _fake_chunks_file()
    mock_load_document_entry.return_value = fake_entry
    mock_load_chunks.return_value = fake_chunks_file
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2

    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 0
    mock_qdrant_cls.return_value = mock_qdrant

    config = _fake_config()

    main(config=config, mode="chatbot")

    mock_qdrant.create_collection_if_missing.assert_called_once_with(
        "report_assistant_chatbot__ollama_nomic-embed-text", 768
    )
    mock_qdrant.count_existing_points.assert_called_once_with(
        "report_assistant_chatbot__ollama_nomic-embed-text", "strategy-hash-1", "doc-1"
    )
    mock_qdrant.delete_existing_points.assert_not_called()
    mock_embed_chunks.assert_called_once_with(
        fake_chunks_file.chunks,
        "http://ollama:11434",
        "nomic-embed-text",
    )
    payload_example = mock_qdrant.create_payload_indexes_if_missing.call_args.args[1]
    assert payload_example["doc_id"] == "doc-1"
    assert payload_example["company"] == "Microsoft"
    assert payload_example["fiscal_year"] == 2024
    assert payload_example["strategy_hash"] == "strategy-hash-1"
    mock_upsert_to_company_collection.assert_called_once()


@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
@patch("builtins.input", return_value="yes")
def test_main_deletes_existing_vectors_when_user_confirms(
    mock_input: MagicMock,
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert_to_company_collection: MagicMock,
) -> None:
    """Deletes existing strategy/doc vectors when overwrite prompt is accepted."""
    mock_get_index_path.return_value = "app/data/index.json"
    fake_entry = _fake_entry()
    fake_chunks_file = _fake_chunks_file()
    mock_load_document_entry.return_value = fake_entry
    mock_load_chunks.return_value = fake_chunks_file
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2

    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 3
    mock_qdrant_cls.return_value = mock_qdrant
    config = _fake_config()

    main(config=config, mode="chatbot")

    mock_input.assert_called_once()
    mock_qdrant.delete_existing_points.assert_called_once_with(
        "report_assistant_chatbot__ollama_nomic-embed-text", "strategy-hash-1", "doc-1"
    )
    mock_upsert_to_company_collection.assert_called_once()


@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
def test_main_uses_fixed_vector_dimension_of_768(
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert_to_company_collection: MagicMock,
) -> None:
    """Captures current behavior where collection dimension is hardcoded to 768."""
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = _fake_entry()
    mock_load_chunks.return_value = _fake_chunks_file()
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2

    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 0
    mock_qdrant_cls.return_value = mock_qdrant
    config = _fake_config()

    main(config=config, mode="chatbot")

    mock_qdrant.create_collection_if_missing.assert_called_once_with(
        "report_assistant_chatbot__ollama_nomic-embed-text", 768
    )
    mock_upsert_to_company_collection.assert_called_once()


@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
def test_main_uses_yoy_collection_in_yoy_mode(
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert_to_company_collection: MagicMock,
) -> None:
    """Routes indexing to the YOY collection when mode is yoy."""
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = _fake_entry()
    mock_load_chunks.return_value = _fake_chunks_file()
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2

    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 0
    mock_qdrant_cls.return_value = mock_qdrant
    config = _fake_config()

    main(config=config, mode="yoy")

    mock_qdrant.create_collection_if_missing.assert_called_once_with(
        "report_assistant_yoy__ollama_nomic-embed-text", 768
    )
    mock_qdrant.count_existing_points.assert_called_once_with(
        "report_assistant_yoy__ollama_nomic-embed-text", "strategy-hash-1", "doc-1"
    )
    mock_upsert_to_company_collection.assert_called_once()


@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
@patch("builtins.input", return_value="no")
def test_main_stops_when_existing_vectors_and_user_declines_overwrite(
    mock_input: MagicMock,
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert_to_company_collection: MagicMock,
) -> None:
    """Stops safely before embedding/upsert when overwrite is rejected."""
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = _fake_entry()
    mock_load_chunks.return_value = _fake_chunks_file()
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2
    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 2
    mock_qdrant_cls.return_value = mock_qdrant
    config = _fake_config()

    main(config=config, mode="chatbot")

    mock_input.assert_called_once()
    mock_qdrant.delete_existing_points.assert_not_called()
    mock_embed_chunks.assert_not_called()
    mock_upsert_to_company_collection.assert_not_called()


@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
def test_main_rejects_missing_embed_model(
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
) -> None:
    """Fails fast when chunk strategy does not define an embedding model."""
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = _fake_entry()
    chunks_file = _fake_chunks_file()
    chunks_file.strategy.embed_model = None
    mock_load_chunks.return_value = chunks_file
    config = _fake_config()

    with pytest.raises(ValueError, match="Missing embed model in chunk strategy"):
        main(config=config, mode="chatbot")


def test_global_config_requires_embedding_profile() -> None:
    """GlobalConfig validation fails when EMBEDDING_PROFILE is missing."""
    config_dict = {
        "data_path": "app/data/",
        "output_path": "app/output/",
        "report_id": "doc-1",
        "OLLAMA_URL": "http://localhost:11434",
        "QDRANT_URL": "http://localhost:6333",
        "POSTGRESQL_URL": "postgresql+psycopg://postgres:postgres@localhost:5432/report_assistant",
        "LLM_MODEL_CHATBOT": "gpt-4.1-mini",
        "LLM_MODEL_SUMMARIZER": "gpt-4.1-mini",
        "LLM_MODEL_EVAL": "gpt-4.1-mini",
        "QDRANT_DB_NAME_CHATBOT": "report_assistant_chatbot",
        "QDRANT_DB_NAME_YOY": "report_assistant_yoy",
        "chunk_strategy_chatbot": {
            "embed_model": "nomic-embed-text",
            "method": "sentence_metadata",
            "chunk_size": 3,
            "overlap": 1,
        },
        "chunk_strategy_yoy": {
            "embed_model": "nomic-embed-text",
            "method": "paragraph",
        },
        "top_k": 4,
    }

    with pytest.raises(Exception, match="EMBEDDING_PROFILE"):
        GlobalConfig.model_validate(config_dict)


def test_global_config_rejects_unknown_embedding_provider() -> None:
    """GlobalConfig validation fails when provider is not one of the allowed literals."""
    config_dict = {
        "data_path": "app/data/",
        "output_path": "app/output/",
        "report_id": "doc-1",
        "OLLAMA_URL": "http://localhost:11434",
        "QDRANT_URL": "http://localhost:6333",
        "POSTGRESQL_URL": "postgresql+psycopg://postgres:postgres@localhost:5432/report_assistant",
        "LLM_MODEL_CHATBOT": "gpt-4.1-mini",
        "LLM_MODEL_SUMMARIZER": "gpt-4.1-mini",
        "LLM_MODEL_EVAL": "gpt-4.1-mini",
        "QDRANT_DB_NAME_CHATBOT": "report_assistant_chatbot",
        "QDRANT_DB_NAME_YOY": "report_assistant_yoy",
        "EMBEDDING_PROFILE": {
            "provider": "azure_openai",
            "embed_model": "text-embedding-3-small",
        },
        "chunk_strategy_chatbot": {
            "embed_model": "nomic-embed-text",
            "method": "sentence_metadata",
            "chunk_size": 3,
            "overlap": 1,
        },
        "chunk_strategy_yoy": {
            "embed_model": "nomic-embed-text",
            "method": "paragraph",
        },
        "top_k": 4,
    }

    with pytest.raises(Exception, match="provider"):
        GlobalConfig.model_validate(config_dict)


def test_global_config_rejects_invalid_embedding_model() -> None:
    """GlobalConfig validation fails when model is empty or lacks alphanumeric content."""
    config_dict = {
        "data_path": "app/data/",
        "output_path": "app/output/",
        "report_id": "doc-1",
        "OLLAMA_URL": "http://localhost:11434",
        "QDRANT_URL": "http://localhost:6333",
        "POSTGRESQL_URL": "postgresql+psycopg://postgres:postgres@localhost:5432/report_assistant",
        "LLM_MODEL_CHATBOT": "gpt-4.1-mini",
        "LLM_MODEL_SUMMARIZER": "gpt-4.1-mini",
        "LLM_MODEL_EVAL": "gpt-4.1-mini",
        "QDRANT_DB_NAME_CHATBOT": "report_assistant_chatbot",
        "QDRANT_DB_NAME_YOY": "report_assistant_yoy",
        "EMBEDDING_PROFILE": {
            "provider": "openai",
            "embed_model": "----",
        },
        "chunk_strategy_chatbot": {
            "embed_model": "nomic-embed-text",
            "method": "sentence_metadata",
            "chunk_size": 3,
            "overlap": 1,
        },
        "chunk_strategy_yoy": {
            "embed_model": "nomic-embed-text",
            "method": "paragraph",
        },
        "top_k": 4,
    }

    with pytest.raises(Exception, match="model"):
        GlobalConfig.model_validate(config_dict)
