"""Qdrant wrapper tests: REST query payload construction and error handling.

Scope of this file:
- Verifies request payloads and response mapping for
  `fetch_top_k_query` and `fetch_top_k_vector`.
- Verifies propagation of HTTP-layer errors.

Why these tests exist:
- Small payload-shape regressions can silently degrade retrieval quality.
- These tests lock down filter wiring (`doc_id`, `strategy_hash`) and request
  body semantics (`vector`, `limit`, endpoint path).

Scenario map:
- Query embedding + search request with full filters.
- Vector-only search request shape.
- HTTP error bubbling from Qdrant REST endpoint.
- Empty filter behavior when optional filter args are omitted.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.clients.QdrantClientWrapper import QdrantClientWrapper
from tests.helpers import make_qdrant_config


@patch("app.clients.QdrantClientWrapper.requests.post")
@patch("app.clients.QdrantClientWrapper.build_embedding_client")
@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_fetch_top_k_query_builds_expected_request(
    mock_qdrant_client_cls: MagicMock,
    mock_build_embedding_client: MagicMock,
    mock_post: MagicMock,
) -> None:
    """Builds REST search payload with filters and returns ranked payloads."""
    mock_qdrant_client_cls.return_value = MagicMock()
    mock_embedding_client = MagicMock()
    mock_embedding_client.embed_text.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_build_embedding_client.return_value = mock_embedding_client
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "result": [
            {"id": "p1", "payload": {"text": "chunk-1"}},
            {"id": "p2", "payload": {"text": "chunk-2"}},
        ]
    }
    mock_post.return_value = fake_response

    wrapper = QdrantClientWrapper(make_qdrant_config("http://qdrant-service:6333"))
    result = wrapper.fetch_top_k_query(
        query="risk factors",
        collection_name="chatbot_idx",
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

    wrapper = QdrantClientWrapper(make_qdrant_config())
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


@patch("app.clients.QdrantClientWrapper.requests.post")
@patch("app.clients.QdrantClientWrapper.build_embedding_client")
@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_fetch_top_k_query_raises_on_http_error(
    mock_qdrant_client_cls: MagicMock,
    mock_build_embedding_client: MagicMock,
    mock_post: MagicMock,
) -> None:
    """Propagates HTTP errors from the Qdrant REST request."""
    mock_qdrant_client_cls.return_value = MagicMock()
    mock_embedding_client = MagicMock()
    mock_embedding_client.embed_text.return_value.tolist.return_value = [0.1, 0.2]
    mock_build_embedding_client.return_value = mock_embedding_client
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = RuntimeError("boom")
    mock_post.return_value = fake_response
    wrapper = QdrantClientWrapper(make_qdrant_config())

    with pytest.raises(RuntimeError, match="boom"):
        wrapper.fetch_top_k_query(
            query="risk",
            collection_name="chatbot_idx",
            doc_id="doc-1",
        )


@patch("app.clients.QdrantClientWrapper.requests.post")
@patch("app.clients.QdrantClientWrapper.build_embedding_client")
@patch("app.clients.QdrantClientWrapper.QdrantClient")
def test_fetch_top_k_query_allows_empty_filter_when_no_doc_or_strategy(
    mock_qdrant_client_cls: MagicMock,
    mock_build_embedding_client: MagicMock,
    mock_post: MagicMock,
) -> None:
    """Sends an empty must-filter list when no doc_id and no strategy_hash are supplied."""
    mock_qdrant_client_cls.return_value = MagicMock()
    mock_embedding_client = MagicMock()
    mock_embedding_client.embed_text.return_value.tolist.return_value = [0.1, 0.2]
    mock_build_embedding_client.return_value = mock_embedding_client
    fake_response = MagicMock()
    fake_response.json.return_value = {"result": []}
    mock_post.return_value = fake_response
    wrapper = QdrantClientWrapper(make_qdrant_config())

    wrapper.fetch_top_k_query(
        query="risk",
        collection_name="chatbot_idx",
    )

    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["filter"] == {"must": []}
