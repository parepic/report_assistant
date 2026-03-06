"""OpenAI embedding client implementation.

This adapter encapsulates OpenAI SDK calls for embeddings and keeps key loading
consistent across shell environment and local `.env` files.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from app.embeddings.base import EmbeddingClient

class OpenAIEmbeddingClient(EmbeddingClient):
    """Embedding client backed by OpenAI's embeddings API."""

    def __init__(self, embed_model: str, api_key: Optional[str] = None) -> None:
        """Initialize an OpenAI embedding client.

        Key-resolution order:
        1. Explicit `api_key` argument.
        2. `OPENAI_API_KEY` from the shell environment.
        3. `OPENAI_API_KEY` loaded from a `.env` file via `python-dotenv`.

        Args:
            embed_model: OpenAI embedding model identifier.
            api_key: Optional explicit API key.

        Raises:
            ValueError: If no API key is available after environment resolution.
        """
        load_dotenv()
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY is required for openai embeddings. "
                "Set it in your shell environment or .env file."
            )

        self.embed_model = embed_model
        self.client = OpenAI(api_key=resolved_key)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed one text string using OpenAI and return a float32 vector.

        Args:
            text: Source text to embed.

        Returns:
            A NumPy float32 embedding vector.

        Raises:
            ValueError: If text is empty.
            RuntimeError: If the OpenAI API response is malformed.
        """
        if not text:
            raise ValueError("Text must be non-empty for embedding.")

        response = self.client.embeddings.create(model=self.embed_model, input=text)
        if not response.data:
            raise RuntimeError("OpenAI embeddings API returned an empty data payload.")

        return np.array(response.data[0].embedding, dtype="float32")

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
