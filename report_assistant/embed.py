import json
import re
from qdrant_client import QdrantClient
import requests
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import yaml
import uuid
import hashlib


from qdrant_client.models import Filter, FieldCondition, MatchValue

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

from qdrant_client.http.exceptions import UnexpectedResponse

def check_and_handle_existing_points(
    client: QdrantClient,
    collection_name: str,
    chunk_strategy: Dict[str, Any],
) -> bool:
    """
    Check if points with the same chunk_strategy metadata already exist.
    If yes, prompt user to delete them or stop.
    """
    strategy_hash = hashlib.sha256(json.dumps(sorted(chunk_strategy.items()), sort_keys=True).encode()).hexdigest()

    # Build filter for exact match on strategy_hash
    scroll_filter = Filter(must=[FieldCondition(key="strategy_hash", match=MatchValue(value=strategy_hash))])

    # Count points matching the filter
    count_result = client.count(collection_name=collection_name, count_filter=scroll_filter)
    existing_count = count_result.count

    if existing_count == 0:
        return True

    print(f"Found {existing_count} existing points with the same strategy hash in collection '{collection_name}'.")
    response = input("Do you want to delete them and recreate embeddings? (yes/no): ").strip().lower()

    if response == "yes":
        client.delete(collection_name=collection_name, points_selector=scroll_filter)
        print(f"Deleted {existing_count} existing points.")
    else:
        print("Process stopped by user.")
        return False
    return True


def load_global_config() -> dict:
    path = Path("global.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Global config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def slugify_collection_name(company: str) -> str:
    """
    Qdrant collection names should be simple. This keeps letters, digits, _ and -.
    """
    s = company.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    if not s:
        raise ValueError("Company name became empty after sanitization.")
    return f"company__{s}"


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


def get_qdrant_client(config: dict) -> QdrantClient:
    # You can set QDRANT_URL in global.yaml. Fallback is local default.
    url = config.get("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        client.get_collection(collection_name)
        return True
    except UnexpectedResponse:
        return False


def create_collection_if_missing(client: QdrantClient, collection_name: str, vector_dim: int) -> None:
    if collection_exists(client, collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )


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


def load_chunks(output_path: str, company: str) -> Dict[str, Any]:
    chunks_file = Path(output_path) / "chunks" / f"{company}.json"
    if not chunks_file.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    with chunks_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "chunks" not in data or "strategy" not in data:
        raise ValueError(f"Invalid chunks file format in {chunks_file}")
    return data


def embed_chunks(chunks: List[str], ollama_url: str, embed_model: str) -> List[np.ndarray]:
    vectors: List[np.ndarray] = []
    print(f"Creating embeddings for {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        emb = np.array(get_embedding(chunk, ollama_url, embed_model), dtype="float32")
        vectors.append(emb)
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks")
    return vectors


def upsert_to_company_collection(
    client: QdrantClient,
    collection_name: str,
    company: str,
    chunks: List[str],
    vectors: List[np.ndarray],
    chunk_strategy: Dict[str, Any],
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("Chunks count does not match vectors count.")

    # Payload fields shared for all points
    base_payload: Dict[str, Any] = dict(chunk_strategy)
    base_payload["company"] = company
    strategy_hash = hashlib.sha256(json.dumps(sorted(chunk_strategy.items()), sort_keys=True).encode()).hexdigest()
    base_payload["strategy_hash"] = strategy_hash

    # Build and upsert points in batches
    batch_size = 128
    points: List[PointStruct] = []

    for i, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
        payload = dict(base_payload)
        payload["chunk_idx"] = i
        payload["text"] = chunk_text

        # Stable point id. Later you can include doc_id / cfg hash here.
        point_id = str(uuid.uuid4())

        points.append(
            PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload=payload,
            )
        )

        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points.clear()

    if points:
        client.upsert(collection_name=collection_name, points=points)


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
        import re
        match = re.search(r'PARAMETER\s+embedding_length\s+(\d+)', modelfile, re.IGNORECASE)
        if match:
            return int(match.group(1))
    except (requests.RequestException, KeyError, ValueError):
        pass  # Fall back to dummy embedding
    
    # Fallback: Make a dummy embedding and get its length
    dummy_emb = get_embedding("test", ollama_url, embed_model)
    return len(dummy_emb)


def main():
    config = load_global_config()

    output_path = config["output_path"]
    company = config["company"]
    ollama_url = config["OLLAMA_URL"]

    chunks_data = load_chunks(output_path, company)
    chunk_strategy = chunks_data["strategy"]
    chunks = chunks_data["chunks"]
    embed_model = chunk_strategy.get("embed_model")
    if not embed_model:
        raise ValueError("Missing embed model in chunk strategy.")

    collection_name = slugify_collection_name(company)
    client = get_qdrant_client(config)
    vector_dim = get_embedding_dimension(ollama_url, embed_model)
    print(vector_dim)
    create_collection_if_missing(client, collection_name, vector_dim)
    proceed = check_and_handle_existing_points(client, collection_name, chunk_strategy)
    if proceed:
        vectors = embed_chunks(chunks, ollama_url, embed_model)
        payload_example = dict(chunk_strategy)
        payload_example["company"] = company
        payload_example["chunk_idx"] = 0
        payload_example["strategy_hash"] = "dummy_hash"
        create_payload_indexes_if_missing(client, collection_name, payload_example)
        print(len(chunks), " ", len(vectors))
        upsert_to_company_collection(client, collection_name, company, chunks, vectors, chunk_strategy)
        print(f"Upserted {len(vectors)} vectors into Qdrant collection '{collection_name}'.")


if __name__ == "__main__":
    main()
