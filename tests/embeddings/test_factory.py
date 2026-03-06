"""Tests for embedding provider abstraction factory and OpenAI key resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.data_classes import EmbeddingProfileConfig
from app.embeddings.factory import build_embedding_client
from app.embeddings.ollama_client import OllamaEmbeddingClient
from app.embeddings.openai_client import OpenAIEmbeddingClient


def _runtime_config(provider: str, model: str, ollama_url: str = "http://ollama:11434") -> SimpleNamespace:
    """Create a minimal runtime config shape accepted by the embedding factory."""
    return SimpleNamespace(
        OLLAMA_URL=ollama_url,
        EMBEDDING_PROFILE=EmbeddingProfileConfig(provider=provider, embed_model=model),
    )


def test_factory_returns_ollama_client_for_ollama_profile() -> None:
    """Factory should build the Ollama client when provider is set to `ollama`."""
    config = _runtime_config(provider="ollama", model="nomic-embed-text")

    client = build_embedding_client(config)

    assert isinstance(client, OllamaEmbeddingClient)


@patch("app.embeddings.openai_client.OpenAI")
def test_factory_returns_openai_client_for_openai_profile(mock_openai: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory should build the OpenAI client when provider is set to `openai`."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = _runtime_config(provider="openai", model="text-embedding-3-small")

    client = build_embedding_client(config)

    assert isinstance(client, OpenAIEmbeddingClient)
    mock_openai.assert_called_once_with(api_key="test-key")


@patch("app.embeddings.openai_client.OpenAI")
def test_openai_client_raises_clear_error_when_api_key_missing(
    mock_openai: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI embedding client must fail fast when no API key is available."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingClient(embed_model="text-embedding-3-small")

    mock_openai.assert_not_called()


@patch("app.embeddings.openai_client.OpenAI")
def test_openai_client_prefers_explicit_key_over_environment(
    mock_openai: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit api_key argument should take precedence over environment variable values."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    OpenAIEmbeddingClient(embed_model="text-embedding-3-small", api_key="explicit-key")

    mock_openai.assert_called_once_with(api_key="explicit-key")


@patch("app.embeddings.openai_client.load_dotenv")
@patch("app.embeddings.openai_client.OpenAI")
def test_openai_client_loads_dotenv_before_key_resolution(
    mock_openai: MagicMock,
    mock_load_dotenv: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client should always call `load_dotenv` so local .env values can be used."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    OpenAIEmbeddingClient(embed_model="text-embedding-3-small")

    mock_load_dotenv.assert_called_once()
    mock_openai.assert_called_once_with(api_key="env-key")
