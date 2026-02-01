"""
Utility functions for Qdrant vector database operations.
"""

import re
import requests
from typing import List, Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)
from qdrant_client.http.exceptions import UnexpectedResponse


def slugify_name(company: str) -> str:
    """
    Qdrant collection names should be simple. This keeps letters, digits, _ and -.
    """
    s = company.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    if not s:
        raise ValueError("Company name became empty after sanitization.")
    return f"company__{s}"


def get_embedding_dimension(ollama_url: str, embed_model: str) -> int:
    """
    Get the embedding dimension by querying Ollama model info or making a dummy embedding.
    """
    # First, try to get dimension from /api/show (if available in Modelfile)
    try:
        payload = {"name": embed_model}
        resp = requests.post(f"{ollama_url}/api/show", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        modelfile = data.get("modelfile", "")
        
        # Parse for common embedding dimension parameters (adjust regex as needed for your models)
        match = re.search(r'PARAMETER\s+embedding_length\s+(\d+)', modelfile, re.IGNORECASE)
        if match:
            return int(match.group(1))
    except (requests.RequestException, KeyError, ValueError):
        pass  # Fall back to dummy embedding
    
    # Fallback: Make a dummy embedding and get its length
    dummy_emb = get_embedding("test", ollama_url, embed_model)
    return len(dummy_emb)


def get_qdrant_client(config) -> QdrantClient:
    """Get Qdrant client from config."""
    url = config.QDRANT_URL or "http://localhost:6333"
    return QdrantClient(url=url)


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a Qdrant collection exists."""
    try:
        client.get_collection(collection_name)
        return True
    except UnexpectedResponse:
        return False


def create_collection_if_missing(client: QdrantClient, collection_name: str, vector_dim: int) -> None:
    """Create a Qdrant collection if it doesn't already exist."""
    if collection_exists(client, collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )


def check_and_handle_existing_points(
    client: QdrantClient,
    collection_name: str,
    strategy_hash: str,
    doc_id: str,
) -> bool:
    """
    Check if points with the same chunk_strategy and doc_id metadata already exist.
    If yes, prompt user to delete them or stop.
    """
    # Build filter for exact match on strategy_hash and doc_id
    scroll_filter = Filter(must=[
        FieldCondition(key="strategy_hash", match=MatchValue(value=strategy_hash)),
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
    ])

    # Count points matching the filter
    count_result = client.count(collection_name=collection_name, count_filter=scroll_filter)
    existing_count = count_result.count

    if existing_count == 0:
        return True

    print(f"Found {existing_count} existing points with the same strategy hash and doc_id in collection '{collection_name}'.")
    response = input("Do you want to delete them and recreate embeddings? (yes/no): ").strip().lower()

    if response == "yes":
        client.delete(collection_name=collection_name, points_selector=scroll_filter)
        print(f"Deleted {existing_count} existing points.")
    else:
        print("Process stopped by user.")
        return False
    return True


def get_embedding(text: str, ollama_url: str, embed_model: str) -> List[float]:
    """
    Get a single embedding vector from Ollama.
    Tries /api/embed (new) then /api/embeddings (older).
    """
    try:
        payload = {"model": embed_model, "input": text}
        resp = requests.post(f"{ollama_url}/api/embed", json=payload, timeout=60)
        if resp.status_code != 404:
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"][0]
    except requests.RequestException:
        pass

    payload = {"model": embed_model, "prompt": text}
    resp = requests.post(f"{ollama_url}/api/embeddings", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


def embed_chunks(chunks: List[Dict[str, Any]], ollama_url: str, embed_model: str) -> List[np.ndarray]:
    """Create embeddings for a list of chunk dicts (extracting 'text' field from each)."""
    vectors: List[np.ndarray] = []
    print(f"Creating embeddings for {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            raise ValueError(f"Chunk {i} has no 'text' field or it is empty")
        emb = np.array(get_embedding(chunk_text, ollama_url, embed_model), dtype="float32")
        vectors.append(emb)
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks")
    return vectors


def python_value_to_payload_type(value: Any) -> PayloadSchemaType:
    """
    Map Python values to Qdrant payload index schema types.
    """
    if isinstance(value, bool):
        return PayloadSchemaType.BOOL
    if isinstance(value, int) and not isinstance(value, bool):
        return PayloadSchemaType.INTEGER
    if isinstance(value, float):
        return PayloadSchemaType.FLOAT
    # default for strings and anything else
    return PayloadSchemaType.KEYWORD


def create_payload_indexes_if_missing(
    client: QdrantClient,
    collection_name: str,
    payload_example: Dict[str, Any],
) -> None:
    """
    Creates a payload index for each key in payload_example if the field is not indexed yet.
    Qdrant stores current payload schema in collection info.
    """
    info = client.get_collection(collection_name)
    existing_schema = info.payload_schema or {}

    for key, value in payload_example.items():
        # We only index simple fields. Skip big text fields by default.
        if key in {"text"}:
            continue

        if key in existing_schema:
            continue

        field_schema = python_value_to_payload_type(value)
        client.create_payload_index(
            collection_name=collection_name,
            field_name=key,
            field_schema=field_schema,
        )

