"""Embedding client factory.

The factory centralizes provider selection so application layers can ask for one
embedding client from runtime config without importing provider-specific code.
"""

from __future__ import annotations

from app.data_classes import GlobalConfig
from app.embeddings.base import EmbeddingClient
from app.embeddings.ollama_client import OllamaEmbeddingClient
from app.embeddings.openai_client import OpenAIEmbeddingClient


def build_embedding_client(config: GlobalConfig) -> EmbeddingClient:
    """Build the concrete embedding client based on `EMBEDDING_PROFILE.provider`.

    Args:
        config: Runtime global configuration containing embedding profile values.

    Returns:
        A provider-specific embedding client implementing the shared contract.

    Raises:
        ValueError: If provider value is unsupported.
    """
    profile = config.EMBEDDING_PROFILE
    if profile.provider == "ollama":
        return OllamaEmbeddingClient(
            base_url=str(config.OLLAMA_URL),
            embed_model=profile.embed_model,
        )

    if profile.provider == "openai":
        return OpenAIEmbeddingClient(embed_model=profile.embed_model)

    raise ValueError(f"Unsupported embedding provider: {profile.provider}")
