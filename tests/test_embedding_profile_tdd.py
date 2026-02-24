"""TDD tests for multi-embedding profile support.

These tests intentionally describe target behavior for the upcoming refactor.
They are marked xfail until profile-aware routing and metadata are implemented.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.data_classes import EmbeddingProfileConfig
from app.ingestion import save_qdrant


def _profile_config() -> SimpleNamespace:
    """Build a minimal config with a future embedding-profile section."""
    return SimpleNamespace(
        report_id="doc-1",
        OLLAMA_URL="http://ollama:11434",
        QDRANT_DB_NAME_CHATBOT="report_assistant_chatbot",
        QDRANT_DB_NAME_YOY="report_assistant_yoy",
        EMBEDDING_PROFILE=EmbeddingProfileConfig(
            provider="openai",
            embed_model="text-embedding-3-small",
        ),
    )


def _fake_entry() -> SimpleNamespace:
    """Create a minimal entry object for save_qdrant main flow tests."""
    return SimpleNamespace(
        doc_id="doc-1",
        company="Microsoft",
        fiscal_year=2024,
        chunks_dir=Path("app/output/company__microsoft/chunks"),
    )


def _fake_chunks_file() -> SimpleNamespace:
    """Create a minimal chunk file test double with strategy metadata."""
    strategy = SimpleNamespace(embed_model="nomic-embed-text", model_dump=lambda: {"method": "sentence_metadata"})
    return SimpleNamespace(
        strategy=strategy,
        strategy_hash="strategy-hash-1",
        chunks=[{"text": "risk text", "risk_factor": "Operations"}],
    )


def test_collection_name_derivation_is_deterministic_and_normalized() -> None:
    """Collection naming should be deterministic from provider+model and normalized safely."""
    profile = EmbeddingProfileConfig(provider="openai", embed_model="text-embedding-3-small")
    collection_name_a = save_qdrant.derive_collection_name("report_assistant_chatbot", profile)
    collection_name_b = save_qdrant.derive_collection_name("report_assistant_chatbot", profile)

    assert collection_name_a == collection_name_b
    assert collection_name_a.endswith("openai_text-embedding-3-small")


@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
def test_main_routes_to_profile_specific_collection(
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert: MagicMock,
) -> None:
    """Main indexing flow should route into profile-specific collection names."""
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = _fake_entry()
    mock_load_chunks.return_value = _fake_chunks_file()
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")]
    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 0
    mock_qdrant_cls.return_value = mock_qdrant

    save_qdrant.main(config=_profile_config(), mode="chatbot")

    create_call = mock_qdrant.create_collection_if_missing.call_args.args
    assert create_call[0] == "report_assistant_chatbot__openai_text-embedding-3-small"
    assert mock_upsert.called


@pytest.mark.xfail(reason="Embedding-profile payload metadata not implemented yet.")
def test_upsert_payload_contains_embedding_profile_metadata() -> None:
    """Payload written to Qdrant should include provider/model/dimension metadata."""
    class _RecordingClient:
        """Collect all upserted points for payload assertions."""

        def __init__(self) -> None:
            self.points = []

        def upsert(self, collection_name: str, points: list) -> None:
            """Record points passed during upsert."""
            self.points.extend(list(points))

    strategy = SimpleNamespace(embed_model="text-embedding-3-small", model_dump=lambda: {"method": "sentence"})
    chunk_file = SimpleNamespace(strategy=strategy, strategy_hash="strategy-hash")
    entry = SimpleNamespace(doc_id="doc-1", company="Microsoft", fiscal_year=2024)
    client = _RecordingClient()

    save_qdrant.upsert_to_company_collection(
        client=client,
        collection_name="chatbot",
        chunks=[{"text": "risk", "risk_factor": "Operations"}],
        vectors=[np.array([0.1, 0.2, 0.3], dtype="float32")],
        chunk_file=chunk_file,
        entry_file=entry,
    )

    payload = client.points[0].payload
    assert payload["embed_provider"] == "openai"
    assert payload["embed_model"] == "text-embedding-3-small"
    assert payload["embed_dim"] == 3


@pytest.mark.xfail(reason="Vector-dimension validation is not implemented yet.")
@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
def test_main_fails_fast_on_collection_dimension_mismatch(
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert: MagicMock,
) -> None:
    """Main flow should fail before upsert when collection dimension mismatches embedding dimension."""
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = _fake_entry()
    mock_load_chunks.return_value = _fake_chunks_file()
    mock_embed_chunks.return_value = [np.array([0.1, 0.2, 0.3], dtype="float32")]
    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 0
    mock_qdrant.get_collection_vector_dim.return_value = 1536
    mock_qdrant_cls.return_value = mock_qdrant

    with pytest.raises(ValueError, match="dimension mismatch"):
        save_qdrant.main(config=_profile_config(), mode="chatbot")

    mock_upsert.assert_not_called()


@pytest.mark.xfail(reason="Delete scope does not include embedding profile yet.")
@patch("app.ingestion.save_qdrant.upsert_to_company_collection")
@patch("app.ingestion.save_qdrant.embed_chunks")
@patch("app.ingestion.save_qdrant.load_chunks")
@patch("app.ingestion.save_qdrant.load_document_entry")
@patch("app.ingestion.save_qdrant.get_index_path")
@patch("app.ingestion.save_qdrant.QdrantClientWrapper")
@patch("builtins.input", return_value="yes")
def test_delete_scope_includes_embedding_profile(
    mock_input: MagicMock,
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert: MagicMock,
) -> None:
    """Overwrite deletion should be scoped by document, strategy, and active embedding profile."""
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = _fake_entry()
    mock_load_chunks.return_value = _fake_chunks_file()
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")]
    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 2
    mock_qdrant_cls.return_value = mock_qdrant

    save_qdrant.main(config=_profile_config(), mode="chatbot")

    mock_input.assert_called_once()
    kwargs = mock_qdrant.delete_existing_points.call_args.kwargs
    assert kwargs["embedding_profile"] == "openai:text-embedding-3-small"
    assert mock_upsert.called
