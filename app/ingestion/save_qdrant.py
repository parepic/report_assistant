import uuid
from typing import List, Dict, Any

import numpy as np
from qdrant_client.models import PointStruct

from app.data_classes import ChunkFile, DocumentEntry, EmbeddingProfileConfig, GlobalConfig
from app.embeddings import build_embedding_client
from app.utils.load_utils import get_index_path, load_chunks, load_document_entry, load_global_config
from app.clients.QdrantClientWrapper import QdrantClientWrapper


def derive_collection_name(
    base_collection_name: str,
    embedding_profile: EmbeddingProfileConfig,
) -> str:
    """
    Derive a deterministic profile-specific collection name.
    """
    provider = embedding_profile.provider.strip()
    embed_model = embedding_profile.embed_model.strip()
    return f"{base_collection_name}__{provider}_{embed_model}"


def resolve_vector_dim(config: GlobalConfig) -> int:
    """
    Resolve embedding vector dimension for collection creation.

    Resolution order:
    1. Explicit `EMBEDDING_PROFILE.dimension` from configuration.
    2. Known provider/model defaults for currently supported embedding models.

    This keeps collection creation deterministic and avoids requiring an API call
    to infer dimensions before duplicate-check logic runs.
    """
    profile = config.EMBEDDING_PROFILE
    if profile.dimension is not None:
        return profile.dimension

    known_dims = {
        ("ollama", "nomic-embed-text"): 768,
        ("openai", "text-embedding-3-small"): 1536,
        ("openai", "text-embedding-3-large"): 3072,
    }
    key = (profile.provider, profile.embed_model)
    if key in known_dims:
        return known_dims[key]

    raise ValueError(
        "Unable to resolve embedding vector dimension for "
        f"provider='{profile.provider}', model='{profile.embed_model}'. "
        "Set EMBEDDING_PROFILE.dimension in app/global.yaml."
    )


def embed_chunks(chunks: List[Dict[str, Any]], config: GlobalConfig) -> List[np.ndarray]:
    """
    Embed chunks through the provider-agnostic embedding client factory.

    This helper is intentionally local to `save_qdrant` so ingestion tests can
    patch one seam while provider-specific details remain isolated.
    """
    embedding_client = build_embedding_client(config)
    return embedding_client.embed_chunks(chunks)


def upsert_to_company_collection(
    client: QdrantClientWrapper,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    vectors: List[np.ndarray],
    chunk_file: ChunkFile,
    entry_file: DocumentEntry
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("Chunks count does not match vectors count.")

    # Payload fields shared for all points
    strategy_dict = chunk_file.strategy.model_dump()
    base_payload: Dict[str, Any] = dict(strategy_dict)
    base_payload["doc_id"] = entry_file.doc_id
    base_payload["company"] = entry_file.company.strip().lower()
    base_payload["fiscal_year"] = entry_file.fiscal_year
    base_payload["strategy_hash"] = chunk_file.strategy_hash

    # Build and upsert points in batches
    batch_size = 128
    points: List[PointStruct] = []

    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        payload = dict(base_payload)
        payload["chunk_idx"] = i
        payload["text"] = chunk["text"]
        payload["risk_factor"] = chunk["risk_factor"]
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


def main(config: GlobalConfig, mode="chatbot") -> None:

    index_path = get_index_path(config)
    entry = load_document_entry(config.report_id, index_path, config)

    chunks_file = load_chunks(entry.chunks_dir / f"{entry.doc_id}.json")

    chunk_strategy = chunks_file.strategy
    vector_dim = resolve_vector_dim(config)

    base_collection_name = config.QDRANT_DB_NAME_CHATBOT if mode == "chatbot" else config.QDRANT_DB_NAME_YOY
    embedding_profile = config.EMBEDDING_PROFILE
    collection_name = derive_collection_name(base_collection_name, embedding_profile)
    qdrant = QdrantClientWrapper(config)
    qdrant.create_collection_if_missing(collection_name, vector_dim)
    existing_count = qdrant.count_existing_points(collection_name, chunks_file.strategy_hash, entry.doc_id)
    if existing_count > 0:
        print(
            f"Found {existing_count} existing points with the same strategy hash "
            f"and doc_id in collection '{collection_name}'."
        )
        response = input("Do you want to delete them and recreate embeddings? (yes/no): ").strip().lower()
        if response == "yes":
            qdrant.delete_existing_points(collection_name, chunks_file.strategy_hash, entry.doc_id)
            print(f"Deleted {existing_count} existing points.")
        else:
            print("Process stopped by user.")
            return

    chunks = chunks_file.chunks
    vectors = embed_chunks(chunks, config)
    payload_example = chunk_strategy.model_dump()
    payload_example["doc_id"] = entry.doc_id
    payload_example["company"] = entry.company
    payload_example["fiscal_year"] = entry.fiscal_year
    payload_example["risk_factor"] = "dummy"
    payload_example["strategy_hash"] = chunks_file.strategy_hash
    qdrant.create_payload_indexes_if_missing(collection_name, payload_example)
    print(len(chunks), " ", len(vectors))
    upsert_to_company_collection(qdrant, collection_name, chunks, vectors, chunks_file, entry)
    print(f"Upserted {len(vectors)} vectors into Qdrant collection '{collection_name}'.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Embed chunks into Qdrant vector database.")
    parser.add_argument('--yoy', action='store_true', help='Use YOY mode')
    parser.add_argument('--chatbot', action='store_true', help='Use chatbot mode (default)')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.yoy:
        mode = "yoy"
    elif args.chatbot:
        mode = "chatbot"
    else:
        mode = "chatbot"  # default
    
    main(load_global_config(), mode=mode)
