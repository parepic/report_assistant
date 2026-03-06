"""`save_qdrant.main` tests: baseline ingestion flow and overwrite control.

Scope of this file:
- Covers the default ingestion orchestration path for chatbot and YOY modes.
- Covers overwrite prompt behavior for existing vectors.
- Keeps focus on control-flow sequencing, not strategy/profile matrix behavior
  (that is isolated in `test_reembed_matrix.py`).

Why these tests exist:
- `main` coordinates many side effects (load, embed, index, upsert). These
  tests ensure the expected calls happen in the right order and with the right
  arguments.

Scenario map:
- Happy-path chatbot ingestion.
- Delete+rebuild when user confirms overwrite.
- Fixed vector dimension behavior (current baseline, pre-dynamic-dimension).
- YOY mode collection routing.
- Early exit when user declines overwrite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from app.ingestion.save_qdrant import main
from tests.helpers import (
    make_real_chunk_file_fixed_size,
    make_real_document_entry,
    make_runtime_config,
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
    real_entry = make_real_document_entry()
    real_chunk_file = make_real_chunk_file_fixed_size(chunk_size=200, overlap=20)
    mock_load_document_entry.return_value = real_entry
    mock_load_chunks.return_value = real_chunk_file
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2

    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 0
    mock_qdrant.get_collection_vector_dim.return_value = 2
    mock_qdrant_cls.return_value = mock_qdrant

    config = make_runtime_config()

    main(config=config, mode="chatbot")

    mock_qdrant.create_collection_if_missing.assert_called_once_with(
        "report_assistant_chatbot__ollama_nomic-embed-text", 768
    )
    mock_qdrant.count_existing_points.assert_called_once_with(
        "report_assistant_chatbot__ollama_nomic-embed-text", real_chunk_file.strategy_hash, "doc-1"
    )
    mock_qdrant.delete_existing_points.assert_not_called()
    mock_embed_chunks.assert_called_once_with(real_chunk_file.chunks, config)
    payload_example = mock_qdrant.create_payload_indexes_if_missing.call_args.args[1]
    assert payload_example["doc_id"] == "doc-1"
    assert payload_example["company"] == "Microsoft"
    assert payload_example["fiscal_year"] == 2024
    assert payload_example["strategy_hash"] == real_chunk_file.strategy_hash
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
    real_entry = make_real_document_entry()
    real_chunk_file = make_real_chunk_file_fixed_size(chunk_size=200, overlap=20)
    mock_load_document_entry.return_value = real_entry
    mock_load_chunks.return_value = real_chunk_file
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2

    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 3
    mock_qdrant.get_collection_vector_dim.return_value = 2
    mock_qdrant_cls.return_value = mock_qdrant
    config = make_runtime_config()

    main(config=config, mode="chatbot")

    mock_input.assert_called_once()
    mock_qdrant.delete_existing_points.assert_called_once_with(
        "report_assistant_chatbot__ollama_nomic-embed-text", real_chunk_file.strategy_hash, "doc-1"
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
    mock_load_document_entry.return_value = make_real_document_entry()
    mock_load_chunks.return_value = make_real_chunk_file_fixed_size(chunk_size=200, overlap=20)
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2

    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 0
    mock_qdrant.get_collection_vector_dim.return_value = 2
    mock_qdrant_cls.return_value = mock_qdrant
    config = make_runtime_config()

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
def test_main_uses_openai_default_dimension_for_text_embedding_3_small(
    mock_qdrant_cls: MagicMock,
    mock_get_index_path: MagicMock,
    mock_load_document_entry: MagicMock,
    mock_load_chunks: MagicMock,
    mock_embed_chunks: MagicMock,
    mock_upsert_to_company_collection: MagicMock,
) -> None:
    """OpenAI `text-embedding-3-small` should create collections with dimension 1536."""
    mock_get_index_path.return_value = "app/data/index.json"
    mock_load_document_entry.return_value = make_real_document_entry()
    mock_load_chunks.return_value = make_real_chunk_file_fixed_size(chunk_size=200, overlap=20)
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2

    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 0
    mock_qdrant.get_collection_vector_dim.return_value = 2
    mock_qdrant_cls.return_value = mock_qdrant
    config = make_runtime_config(provider="openai", embed_model="text-embedding-3-small")

    main(config=config, mode="chatbot")

    mock_qdrant.create_collection_if_missing.assert_called_once_with(
        "report_assistant_chatbot__openai_text-embedding-3-small", 1536
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
    real_entry = make_real_document_entry()
    real_chunk_file = make_real_chunk_file_fixed_size(chunk_size=200, overlap=20)
    mock_load_document_entry.return_value = real_entry
    mock_load_chunks.return_value = real_chunk_file
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2

    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 0
    mock_qdrant.get_collection_vector_dim.return_value = 2
    mock_qdrant_cls.return_value = mock_qdrant
    config = make_runtime_config()

    main(config=config, mode="yoy")

    mock_qdrant.create_collection_if_missing.assert_called_once_with(
        "report_assistant_yoy__ollama_nomic-embed-text", 768
    )
    mock_qdrant.count_existing_points.assert_called_once_with(
        "report_assistant_yoy__ollama_nomic-embed-text", real_chunk_file.strategy_hash, "doc-1"
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
    real_entry = make_real_document_entry()
    real_chunk_file = make_real_chunk_file_fixed_size(chunk_size=200, overlap=20)
    mock_load_document_entry.return_value = real_entry
    mock_load_chunks.return_value = real_chunk_file
    mock_embed_chunks.return_value = [np.array([0.1, 0.2], dtype="float32")] * 2
    mock_qdrant = MagicMock()
    mock_qdrant.count_existing_points.return_value = 2
    mock_qdrant.get_collection_vector_dim.return_value = 2
    mock_qdrant_cls.return_value = mock_qdrant
    config = make_runtime_config()

    main(config=config, mode="chatbot")

    mock_input.assert_called_once()
    mock_qdrant.count_existing_points.assert_called_once_with(
        "report_assistant_chatbot__ollama_nomic-embed-text", real_chunk_file.strategy_hash, "doc-1"
    )
    mock_qdrant.delete_existing_points.assert_not_called()
    mock_embed_chunks.assert_not_called()
    mock_upsert_to_company_collection.assert_not_called()
