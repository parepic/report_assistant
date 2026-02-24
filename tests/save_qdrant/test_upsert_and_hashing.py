"""`save_qdrant` tests: low-level upsert semantics and strategy-hash logic.

Scope of this file:
- Validates behavior of pure/low-level units that do not depend on
  `save_qdrant.main` orchestration.

Why these tests exist:
- Upsert batching mistakes can cause missing or duplicated vectors.
- Strategy-hash correctness is central to idempotent re-embed behavior.

Scenario map:
- Batch splitting and payload fields in `upsert_to_company_collection`.
- Validation error when chunk/vector counts diverge.
- Stable hash for identical strategy parameters.
- Different hash when strategy parameters change.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.data_classes import compute_strategy_hash
from app.ingestion.chunking.strategies import ChunkStrategySentenceMetadata
from app.ingestion.chunking.strategies.ChunkStrategyFixedSize import ChunkStrategyFixedSize
from app.ingestion.chunking.strategies.ChunkStrategySentence import ChunkStrategySentence
from app.ingestion.save_qdrant import upsert_to_company_collection
from tests.helpers import FakeSentenceMetadataStrategy


def test_upsert_to_company_collection_batches_and_includes_payload_fields() -> None:
    """Writes expected payload metadata and performs 128-sized batched upserts."""

    class _RecordingClient:
        """Capture snapshots of upserted points to avoid mutable-list assertion issues."""

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def upsert(self, collection_name: str, points: list) -> None:
            """Store immutable snapshots of every upsert call."""
            self.calls.append(
                {
                    "collection_name": collection_name,
                    "points": list(points),
                }
            )

    recording_client = _RecordingClient()
    strategy = FakeSentenceMetadataStrategy()
    chunk_file = SimpleNamespace(strategy=strategy, strategy_hash="hash-123")
    entry = SimpleNamespace(doc_id="doc-x", company="Apple", fiscal_year=2025)

    chunks = [
        {"text": f"text-{index}", "risk_factor": "Liquidity"}
        for index in range(130)
    ]
    vectors = [np.array([0.1, 0.2], dtype="float32") for _ in range(130)]

    upsert_to_company_collection(
        client=recording_client,
        collection_name="chatbot_idx",
        chunks=chunks,
        vectors=vectors,
        chunk_file=chunk_file,
        entry_file=entry,
    )

    assert len(recording_client.calls) == 2
    first_batch_points = recording_client.calls[0]["points"]
    second_batch_points = recording_client.calls[1]["points"]
    assert len(first_batch_points) == 128
    assert len(second_batch_points) == 2
    first_payload = first_batch_points[0].payload
    assert first_payload["doc_id"] == "doc-x"
    assert first_payload["company"] == "apple"
    assert first_payload["fiscal_year"] == 2025
    assert first_payload["strategy_hash"] == "hash-123"
    assert first_payload["chunk_idx"] == 0
    assert first_payload["risk_factor"] == "Liquidity"


def test_upsert_to_company_collection_rejects_mismatched_lengths() -> None:
    """Raises a clear error when chunk and vector lengths do not match."""
    strategy = FakeSentenceMetadataStrategy()
    chunk_file = SimpleNamespace(strategy=strategy, strategy_hash="hash-123")
    entry = SimpleNamespace(doc_id="doc-x", company="Apple", fiscal_year=2025)

    with pytest.raises(ValueError, match="Chunks count does not match vectors count"):
        upsert_to_company_collection(
            client=MagicMock(),
            collection_name="chatbot_idx",
            chunks=[{"text": "a", "risk_factor": "Liquidity"}],
            vectors=[],
            chunk_file=chunk_file,
            entry_file=entry,
        )


def test_compute_strategy_hash_same_strategy_values_are_stable() -> None:
    """Same strategy config should always produce the same hash."""
    strategy_a = ChunkStrategyFixedSize(chunk_size=200, overlap=20)
    strategy_b = ChunkStrategyFixedSize(chunk_size=200, overlap=20)

    assert compute_strategy_hash(strategy_a) == compute_strategy_hash(strategy_b)


def test_compute_strategy_hash_changes_when_chunking_params_change() -> None:
    """Changing strategy parameters should change strategy hash."""
    strategy_a = ChunkStrategyFixedSize(chunk_size=200, overlap=20)
    strategy_b = ChunkStrategyFixedSize(chunk_size=220, overlap=20)

    assert compute_strategy_hash(strategy_a) != compute_strategy_hash(strategy_b)


def test_compute_strategy_hash_changes_when_strategy_method_changes() -> None:
    """Different strategy methods should never share the same strategy hash."""
    fixed_size_strategy = ChunkStrategySentence(chunk_size=200, overlap=20)
    paragraph_strategy = ChunkStrategySentenceMetadata(chunk_size=200, overlap=20)

    assert compute_strategy_hash(fixed_size_strategy) != compute_strategy_hash(paragraph_strategy)
