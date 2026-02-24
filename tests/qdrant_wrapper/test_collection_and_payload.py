"""Qdrant wrapper tests: collection lifecycle and payload schema behavior.

Scope of this file:
- Verifies wrapper behavior that uses the Qdrant Python SDK directly
  (collection creation, existence checks, payload index typing).
- Excludes REST query payload behavior, which is covered in
  `test_query_requests.py`.

Why these tests exist:
- Collection and payload-index setup errors usually fail at runtime only after
  data ingest starts. These tests catch contract mismatches early.

Scenario map:
- Create collection when absent.
- Skip create when already present.
- Read collection vector dimension for strict unnamed vector config.
- Raise when collection uses named vectors or omits vector size.
- Build strict count/delete filters with `strategy_hash + doc_id`.
- Map Python primitive values to Qdrant payload schema types.
- Create payload indexes only for missing non-text fields.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.clients.QdrantClientWrapper import QdrantClientWrapper
from qdrant_client.models import Distance, PayloadSchemaType
from tests.helpers import make_qdrant_config


@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_create_collection_if_missing_creates_when_absent(mock_qdrant_client_cls: MagicMock) -> None:
    """Creates collection with cosine distance when collection does not exist."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    mock_qdrant_client_cls.return_value = mock_client
    wrapper = QdrantClientWrapper(make_qdrant_config())

    wrapper.create_collection_if_missing("collection_a", vector_dim=1536)

    mock_client.create_collection.assert_called_once()
    kwargs = mock_client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "collection_a"
    assert kwargs["vectors_config"].size == 1536
    assert kwargs["vectors_config"].distance == Distance.COSINE


@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_create_collection_if_missing_skips_when_present(mock_qdrant_client_cls: MagicMock) -> None:
    """Avoids create calls when collection already exists."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_qdrant_client_cls.return_value = mock_client
    wrapper = QdrantClientWrapper(make_qdrant_config())

    wrapper.create_collection_if_missing("collection_b", vector_dim=768)

    mock_client.create_collection.assert_not_called()


@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_get_collection_vector_dim_reads_unnamed_vector_size(mock_qdrant_client_cls: MagicMock) -> None:
    """Returns vector size from standard (unnamed) vectors configuration."""
    mock_client = MagicMock()
    mock_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=1536)
            )
        )
    )
    mock_qdrant_client_cls.return_value = mock_client
    wrapper = QdrantClientWrapper(make_qdrant_config())

    result = wrapper.get_collection_vector_dim("collection_x")

    assert result == 1536


@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_get_collection_vector_dim_raises_for_named_vectors(mock_qdrant_client_cls: MagicMock) -> None:
    """Named-vector collections are unsupported by this app's strict contract."""
    mock_client = MagicMock()
    mock_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={
                    "text": SimpleNamespace(size=3072),
                    "image": SimpleNamespace(size=1024),
                }
            )
        )
    )
    mock_qdrant_client_cls.return_value = mock_client
    wrapper = QdrantClientWrapper(make_qdrant_config())

    with pytest.raises(ValueError, match="named vectors"):
        wrapper.get_collection_vector_dim("collection_x")


@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_get_collection_vector_dim_raises_when_size_attribute_missing(mock_qdrant_client_cls: MagicMock) -> None:
    """Missing `vectors.size` should fail naturally via attribute access."""
    mock_client = MagicMock()
    mock_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace()
            )
        )
    )
    mock_qdrant_client_cls.return_value = mock_client
    wrapper = QdrantClientWrapper(make_qdrant_config())

    with pytest.raises(AttributeError, match="size"):
        wrapper.get_collection_vector_dim("collection_x")


@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_count_and_delete_filter_include_strategy_and_doc(mock_qdrant_client_cls: MagicMock) -> None:
    """Builds count/delete filters using both strategy hash and document id."""
    mock_client = MagicMock()
    mock_client.count.return_value = SimpleNamespace(count=7)
    mock_qdrant_client_cls.return_value = mock_client
    wrapper = QdrantClientWrapper(make_qdrant_config())

    count = wrapper.count_existing_points("chatbot_idx", "abc123", "doc-2024")
    wrapper.delete_existing_points("chatbot_idx", "abc123", "doc-2024")

    assert count == 7
    count_filter = mock_client.count.call_args.kwargs["count_filter"]
    delete_filter = mock_client.delete.call_args.kwargs["points_selector"]
    assert [condition.key for condition in count_filter.must] == ["strategy_hash", "doc_id"]
    assert count_filter.must[0].match.value == "abc123"
    assert count_filter.must[1].match.value == "doc-2024"
    assert [condition.key for condition in delete_filter.must] == ["strategy_hash", "doc_id"]


@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_python_value_to_payload_type_maps_expected_types(mock_qdrant_client_cls: MagicMock) -> None:
    """Maps Python scalar types to expected Qdrant payload schema types."""
    mock_qdrant_client_cls.return_value = MagicMock()
    wrapper = QdrantClientWrapper(make_qdrant_config())

    assert wrapper.python_value_to_payload_type(True) == PayloadSchemaType.BOOL
    assert wrapper.python_value_to_payload_type(7) == PayloadSchemaType.INTEGER
    assert wrapper.python_value_to_payload_type(7.3) == PayloadSchemaType.FLOAT
    assert wrapper.python_value_to_payload_type("abc") == PayloadSchemaType.KEYWORD


@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_create_payload_indexes_skips_text_and_existing_fields(mock_qdrant_client_cls: MagicMock) -> None:
    """Creates indexes only for missing payload fields and never for raw text."""
    mock_client = MagicMock()
    mock_client.get_collection.return_value = SimpleNamespace(
        payload_schema={"doc_id": {"type": "keyword"}}
    )
    mock_qdrant_client_cls.return_value = mock_client
    wrapper = QdrantClientWrapper(make_qdrant_config())

    wrapper.create_payload_indexes_if_missing(
        collection_name="chatbot_idx",
        payload_example={
            "text": "chunk text",
            "doc_id": "doc-1",
            "chunk_idx": 1,
            "risk_factor": "Liquidity",
        },
    )

    assert mock_client.create_payload_index.call_count == 2
    created_fields = [
        call.kwargs["field_name"] for call in mock_client.create_payload_index.call_args_list
    ]
    assert "text" not in created_fields
    assert "doc_id" not in created_fields
    assert set(created_fields) == {"chunk_idx", "risk_factor"}
