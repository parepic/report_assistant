"""Embedding abstraction contracts.

This module defines the provider-agnostic contract used by ingestion and retrieval
code paths that need vector embeddings. The goal is to isolate vendor-specific
transport and response parsing logic behind a stable Python interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

import numpy as np


class EmbeddingClient(Protocol):
    """Provider-agnostic contract for turning text into vector embeddings.

    Implementations must return vectors as `float32` NumPy arrays to keep a
    consistent numeric type across providers and avoid downstream dtype drift
    when writing points to Qdrant.
    """

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string and return one `float32` vector.

        Args:
            text: Input text to convert into a semantic vector.

        Returns:
            A NumPy array containing the embedding values in `float32` format.

        Raises:
            ValueError: If input text is empty or missing.
            RuntimeError: If the provider call fails or returns an invalid payload.
        """

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Embed a list of chunk dictionaries that contain a `text` field.

        Args:
            chunks: Chunk payloads produced by ingestion chunking strategies.

        Returns:
            A list of embedding vectors aligned with the order of `chunks`.

        Raises:
            ValueError: If any chunk has missing or empty text.
            RuntimeError: If provider calls fail for any chunk.
        """
