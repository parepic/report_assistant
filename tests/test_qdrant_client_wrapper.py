"""Unit tests for Qdrant client wrapper behavior.

These tests isolate network and Qdrant SDK calls so we can validate payload
construction and filtering logic without external services.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.clients.QdrantClientWrapper import QdrantClientWrapper
from qdrant_client.models import Distance, PayloadSchemaType


def _config(url: str = "http://qdrant:6333") -> SimpleNamespace:
    """Return a minimal config object accepted by the wrapper constructor."""
    return SimpleNamespace(QDRANT_URL=url)


@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_create_collection_if_missing_creates_when_absent(mock_qdrant_client_cls: MagicMock) -> None:
    """Creates collection with cosine distance when collection does not exist."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    mock_qdrant_client_cls.return_value = mock_client
    wrapper = QdrantClientWrapper(_config())

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
    wrapper = QdrantClientWrapper(_config())

    wrapper.create_collection_if_missing("collection_b", vector_dim=768)

    mock_client.create_collection.assert_not_called()


@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_count_and_delete_filter_include_strategy_and_doc(mock_qdrant_client_cls: MagicMock) -> None:
    """Builds count/delete filters using both strategy hash and document id."""
    mock_client = MagicMock()
    mock_client.count.return_value = SimpleNamespace(count=7)
    mock_qdrant_client_cls.return_value = mock_client
    wrapper = QdrantClientWrapper(_config())

    count = wrapper.count_existing_points("chatbot_idx", "abc123", "doc-2024")
    wrapper.delete_existing_points("chatbot_idx", "abc123", "doc-2024")

    assert count == 7
    count_filter = mock_client.count.call_args.kwargs["count_filter"]
    delete_filter = mock_client.delete.call_args.kwargs["points_selector"]
    assert [condition.key for condition in count_filter.must] == ["strategy_hash", "doc_id"]
    assert count_filter.must[0].match.value == "abc123"
    assert count_filter.must[1].match.value == "doc-2024"
    assert [condition.key for condition in delete_filter.must] == ["strategy_hash", "doc_id"]


@patch("app.clients.QdrantClientWrapper.requests.post")
@patch("app.clients.QdrantClientWrapper.get_embedding")
@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_fetch_top_k_query_builds_expected_request(
    mock_qdrant_client_cls: MagicMock,
    mock_get_embedding: MagicMock,
    mock_post: MagicMock,
) -> None:
    """Builds REST search payload with filters and returns ranked payloads."""
    mock_qdrant_client_cls.return_value = MagicMock()
    mock_get_embedding.return_value = [0.1, 0.2, 0.3]
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "result": [
            {"id": "p1", "payload": {"text": "chunk-1"}},
            {"id": "p2", "payload": {"text": "chunk-2"}},
        ]
    }
    mock_post.return_value = fake_response

    wrapper = QdrantClientWrapper(_config("http://qdrant-service:6333"))
    result = wrapper.fetch_top_k_query(
        query="risk factors",
        collection_name="chatbot_idx",
        ollama_url="http://ollama:11434",
        embed_model="nomic-embed-text",
        strategy_hash="strategy-v1",
        doc_id="doc-1",
        k=4,
    )

    assert [item["rank"] for item in result] == [1, 2]
    assert result[0]["id"] == "p1"
    assert result[1]["payload"]["text"] == "chunk-2"
    called_url = mock_post.call_args.args[0]
    called_payload = mock_post.call_args.kwargs["json"]
    assert called_url == "http://qdrant-service:6333/collections/chatbot_idx/points/search"
    assert called_payload["limit"] == 4
    assert called_payload["vector"] == [0.1, 0.2, 0.3]
    assert called_payload["filter"]["must"][0]["key"] == "doc_id"
    assert called_payload["filter"]["must"][1]["key"] == "strategy_hash"


@patch("app.clients.QdrantClientWrapper.requests.post")
@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_fetch_top_k_vector_builds_expected_request(
    mock_qdrant_client_cls: MagicMock,
    mock_post: MagicMock,
) -> None:
    """Uses given vector to query Qdrant and returns id plus payload entries."""
    mock_qdrant_client_cls.return_value = MagicMock()
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "result": [
            {"id": "a", "payload": {"text": "alpha"}},
            {"id": "b", "payload": {"text": "beta"}},
        ]
    }
    mock_post.return_value = fake_response

    wrapper = QdrantClientWrapper(_config())
    result = wrapper.fetch_top_k_vector(
        collection_name="yoy_idx",
        vector=[1.0, 2.0, 3.0],
        strategy_hash="hash-yoy",
        doc_id="doc-2",
        k=10,
    )

    assert result == [
        {"id": "a", "payload": {"text": "alpha"}},
        {"id": "b", "payload": {"text": "beta"}},
    ]
    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["limit"] == 10
    assert called_payload["vector"] == [1.0, 2.0, 3.0]
    assert called_payload["filter"]["must"][0]["match"]["value"] == "doc-2"
    assert called_payload["filter"]["must"][1]["match"]["value"] == "hash-yoy"


@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_python_value_to_payload_type_maps_expected_types(mock_qdrant_client_cls: MagicMock) -> None:
    """Maps Python scalar types to expected Qdrant payload schema types."""
    mock_qdrant_client_cls.return_value = MagicMock()
    wrapper = QdrantClientWrapper(_config())

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
    wrapper = QdrantClientWrapper(_config())

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


@patch("app.clients.QdrantClientWrapper.requests.post")
@patch("app.clients.QdrantClientWrapper.get_embedding")
@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_fetch_top_k_query_raises_on_http_error(
    mock_qdrant_client_cls: MagicMock,
    mock_get_embedding: MagicMock,
    mock_post: MagicMock,
) -> None:
    """Propagates HTTP errors from the Qdrant REST request."""
    mock_qdrant_client_cls.return_value = MagicMock()
    mock_get_embedding.return_value = [0.1, 0.2]
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = RuntimeError("boom")
    mock_post.return_value = fake_response
    wrapper = QdrantClientWrapper(_config())

    with pytest.raises(RuntimeError, match="boom"):
        wrapper.fetch_top_k_query(
            query="risk",
            collection_name="chatbot_idx",
            ollama_url="http://ollama:11434",
            embed_model="nomic-embed-text",
            doc_id="doc-1",
        )


@patch("app.clients.QdrantClientWrapper.requests.post")
@patch("app.clients.QdrantClientWrapper.get_embedding")
@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_fetch_top_k_query_allows_empty_filter_when_no_doc_or_strategy(
    mock_qdrant_client_cls: MagicMock,
    mock_get_embedding: MagicMock,
    mock_post: MagicMock,
) -> None:
    """Sends an empty must-filter list when no doc_id and no strategy_hash are supplied."""
    mock_qdrant_client_cls.return_value = MagicMock()
    mock_get_embedding.return_value = [0.1, 0.2]
    fake_response = MagicMock()
    fake_response.json.return_value = {"result": []}
    mock_post.return_value = fake_response
    wrapper = QdrantClientWrapper(_config())

    wrapper.fetch_top_k_query(
        query="risk",
        collection_name="chatbot_idx",
        ollama_url="http://ollama:11434",
        embed_model="nomic-embed-text",
    )

    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["filter"] == {"must": []}
