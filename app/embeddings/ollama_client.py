"""Ollama embedding client implementation.

This adapter isolates Ollama-specific REST endpoints and payload shape
normalization so higher layers can use a provider-neutral embedding contract.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import requests

from app.embeddings.base import EmbeddingClient


class OllamaEmbeddingClient(EmbeddingClient):
    """Embedding client backed by an Ollama server.

    The client supports both Ollama embedding endpoints for compatibility:
    - `/api/embed` (newer)
    - `/api/embeddings` (older)
    """

    def __init__(self, base_url: str, embed_model: str) -> None:
        """Initialize the Ollama embedding client.

        Args:
            base_url: Ollama server base URL.
            embed_model: Ollama embedding model identifier.
        """
        self.base_url = base_url
        self.embed_model = embed_model

    def embed_text(self, text: str) -> np.ndarray:
        """Embed one text string using Ollama and return a float32 vector.

        Args:
            text: Source text to embed.

        Returns:
            A NumPy float32 embedding vector.

        Raises:
            ValueError: If text is empty.
            RuntimeError: If Ollama request fails on both supported endpoints.
        """
        if not text:
            raise ValueError("Text must be non-empty for embedding.")

        try:
            payload = {"model": self.embed_model, "input": text}
            resp = requests.post(f"{self.base_url}/api/embed", json=payload, timeout=60)
            if resp.status_code != 404:
                resp.raise_for_status()
                data = resp.json()
                return np.array(data["embeddings"][0], dtype="float32")
        except requests.RequestException as exc:
            # Fall back to the legacy endpoint before failing hard.
            fallback_error = exc
        else:
            fallback_error = None

        try:
            payload = {"model": self.embed_model, "prompt": text}
            resp = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return np.array(data["embedding"], dtype="float32")
        except requests.RequestException as exc:
            message = "Ollama embedding request failed for both /api/embed and /api/embeddings endpoints."
            if fallback_error is not None:
                raise RuntimeError(message) from exc
            raise RuntimeError(message) from exc

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Embed a list of chunk dictionaries via repeated `embed_text` calls.

        Args:
            chunks: Chunk objects containing the key `text`.

        Returns:
            A list of embedding vectors aligned with chunk order.

        Raises:
            ValueError: If any chunk has missing or empty text.
            RuntimeError: If provider requests fail.
        """
        vectors: List[np.ndarray] = []
        print(f"Creating embeddings for {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                raise ValueError(f"Chunk {i} has no 'text' field or it is empty")
            vectors.append(self.embed_text(chunk_text))
            if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                print(f"  Embedded {i + 1}/{len(chunks)} chunks")
        return vectors
