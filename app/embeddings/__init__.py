"""Provider-agnostic embedding module exports."""

from app.embeddings.base import EmbeddingClient
from app.embeddings.factory import build_embedding_client
from app.embeddings.ollama_client import OllamaEmbeddingClient
from app.embeddings.openai_client import OpenAIEmbeddingClient

__all__ = [
    "EmbeddingClient",
    "OllamaEmbeddingClient",
    "OpenAIEmbeddingClient",
    "build_embedding_client",
]
