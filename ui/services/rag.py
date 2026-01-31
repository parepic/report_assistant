from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from openai import OpenAI

from report_assistant.data_classes import DocumentEntry, GlobalConfig, compute_strategy_hash
from report_assistant.llm import answer_question
from report_assistant.utils.load_utils import load_global_config
from report_assistant.utils.utils import slugify_name


@lru_cache(maxsize=1)
def get_config() -> GlobalConfig:
    return load_global_config()


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def _normalize_entry(entry: DocumentEntry | dict[str, Any]) -> DocumentEntry:
    if isinstance(entry, DocumentEntry):
        return entry
    return DocumentEntry.model_validate(entry)


def answer_for_entry(question: str, entry: DocumentEntry | dict[str, Any]) -> str:
    config = get_config()
    normalized = _normalize_entry(entry)
    collection_name = slugify_name(normalized.company)
    strategy_hash = compute_strategy_hash(config.chunk_strategy)

    return answer_question(
        question=question,
        collection_name=collection_name,
        qdrant_url=str(config.QDRANT_URL),
        ollama_url=str(config.OLLAMA_URL),
        embed_model=config.chunk_strategy.embed_model,
        client=get_openai_client(),
        llm_model=config.LLM_MODEL,
        top_k=config.top_k,
        strategy_hash=strategy_hash,
    )
