"""
Qdrant client wrapper utilities.
"""

from typing import List, Dict, Any, Optional

import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
	Distance,
	VectorParams,
	Filter,
	FieldCondition,
	MatchValue,
	PayloadSchemaType,
	PointStruct
)

from app.data_classes import GlobalConfig
from app.embeddings import build_embedding_client


class QdrantClientWrapper:
	def __init__(self, config: GlobalConfig) -> None:
		self.config = config
		self.url = config.QDRANT_URL or "http://localhost:6333"
		self.client = QdrantClient(url=self.url)

	def collection_exists(self, collection_name: str) -> bool:
		"""Check if a Qdrant collection exists."""
		return self.client.collection_exists(collection_name=collection_name)

	def create_collection_if_missing(self, collection_name: str, vector_dim: int) -> None:
		"""Create a Qdrant collection if it doesn't already exist."""
		if self.client.collection_exists(collection_name):
			return
		self.client.create_collection(
			collection_name=collection_name,
			vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
		)

	def _build_strategy_doc_filter(self, strategy_hash: str, doc_id: str) -> Filter:
		return Filter(must=[
			FieldCondition(key="strategy_hash", match=MatchValue(value=strategy_hash)),
			FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
		])

	def count_existing_points(self, collection_name: str, strategy_hash: str, doc_id: str) -> int:
		"""Return count of points matching strategy_hash and doc_id."""
		scroll_filter = self._build_strategy_doc_filter(strategy_hash, doc_id)
		count_result = self.client.count(collection_name=collection_name, count_filter=scroll_filter)
		return count_result.count

	def delete_existing_points(self, collection_name: str, strategy_hash: str, doc_id: str) -> None:
		"""Delete points matching strategy_hash and doc_id."""
		scroll_filter = self._build_strategy_doc_filter(strategy_hash, doc_id)
		self.client.delete(collection_name=collection_name, points_selector=scroll_filter)

	def python_value_to_payload_type(self, value: Any) -> PayloadSchemaType:
		"""Map Python values to Qdrant payload index schema types."""
		if isinstance(value, bool):
			return PayloadSchemaType.BOOL
		if isinstance(value, int) and not isinstance(value, bool):
			return PayloadSchemaType.INTEGER
		if isinstance(value, float):
			return PayloadSchemaType.FLOAT
		return PayloadSchemaType.KEYWORD

	def create_payload_indexes_if_missing(
		self,
		collection_name: str,
		payload_example: Dict[str, Any],
	) -> None:
		"""
		Creates a payload index for each key in payload_example if the field is not indexed yet.
		Qdrant stores current payload schema in collection info.

		TODO(typed-payload-schema): Accept a typed payload schema/index contract
		instead of a runtime example payload dict so index definitions are explicit
		and centrally validated.
		"""
		info = self.client.get_collection(collection_name)
		existing_schema = info.payload_schema or {}

		for key, value in payload_example.items():
			if key in {"text"}:
				continue

			if key in existing_schema:
				continue

			field_schema = self.python_value_to_payload_type(value)
			self.client.create_payload_index(
				collection_name=collection_name,
				field_name=key,
				field_schema=field_schema,
			)

	def upsert(self, collection_name: str, points: List[PointStruct]) -> None:
		"""Upsert points to Qdrant collection."""
		self.client.upsert(collection_name=collection_name, points=points)

	def fetch_top_k_query(
		self,
		query: str,
		collection_name: str,
		strategy_hash: Optional[str] = None,
		doc_id: Optional[str] = None,
		k: int = 4
	) -> List[Dict[str, Any]]:
		"""
		Embed a query through the active embedding provider and retrieve top-k chunks from Qdrant.

		This method uses the runtime `GlobalConfig` attached to the wrapper to build
		a provider-specific embedding client via the embedding factory. That keeps
		retrieval provider-agnostic: if `EMBEDDING_PROFILE.provider` changes from
		`ollama` to `openai`, this retrieval path automatically follows the same
		profile used during ingestion.

		Args:
			query: End-user query text to embed.
			collection_name: Target Qdrant collection name.
			strategy_hash: Optional chunking strategy hash filter.
			doc_id: Optional document identifier filter.
			k: Maximum number of hits to return.

		Returns:
			A ranked list of hit dictionaries (`id`, `rank`, `payload`).
		"""
		embedding_client = build_embedding_client(self.config)
		query_emb = embedding_client.embed_text(query).tolist()
		payload = {
			"vector": query_emb,
			"limit": k,
			"with_payload": ["text", "risk_factor", "doc_id", "company", "fiscal_year", "strategy_hash", "chunk_idx"]
		}
		# Always filter by company, optionally by strategy_hash
		must_filters = []
		if doc_id:
			must_filters.append({
				"key": "doc_id",
				"match": {
					"value": doc_id
				}
			})
		if strategy_hash:
			must_filters.append({
				"key": "strategy_hash",
				"match": {
					"value": strategy_hash
				}
			})

		payload["filter"] = {"must": must_filters}

		resp = requests.post(f"{self.url}/collections/{collection_name}/points/search", json=payload)
		resp.raise_for_status()
		data = resp.json()
		return [
			{
				"id": hit.get("id"),
				"rank": idx + 1,
				"payload": hit.get("payload", {}),
			}
			for idx, hit in enumerate(data["result"])
		]


	def fetch_top_k_vector(
		self,
		collection_name: str,
		vector: List[float],
		strategy_hash: Optional[str] = None,
		doc_id: Optional[str] = None,
		k: int = 4
	) -> List[Dict[str, Any]]:
		"""
		Embed the query and retrieve top-k chunk texts from Qdrant using REST API.
		Optionally filter by strategy_hash to only retrieve chunks created with a specific chunking strategy.
		"""
		payload = {
			"vector": vector,
			"limit": k,
			"with_payload": ["text", "risk_factor", "doc_id", "company", "fiscal_year", "strategy_hash", "chunk_idx"]
		}
		# Always filter by company, optionally by strategy_hash
		must_filters = []
		if doc_id:
			must_filters.append({
				"key": "doc_id",
				"match": {
					"value": doc_id
				}
			})
		if strategy_hash:
			must_filters.append({
				"key": "strategy_hash",
				"match": {
					"value": strategy_hash
				}
			})

		payload["filter"] = {"must": must_filters}

		resp = requests.post(f"{self.url}/collections/{collection_name}/points/search", json=payload)
		resp.raise_for_status()
		data = resp.json()
		return [
			{
				"id": hit.get("id"),
				"payload": hit.get("payload", {}),
			}
			for hit in data["result"]
		]


	def scroll(self, **kwargs):
		"""Scroll through points in a collection matching a filter. Returns (results, next_offset)."""
		return self.client.scroll(**kwargs)

	def retrieve(self, **kwargs):
		"""Retrieve points from a collection matching a filter. Returns results."""
		return self.client.retrieve(**kwargs)
