"""Configuration-contract tests for embedding profile requirements.

Scope of this file:
- Validates `GlobalConfig` schema behavior for embedding-profile constraints.
- Focuses only on validation contracts, not runtime ingestion behavior.

Why these tests exist:
- Configuration errors should fail fast at validation time, not during ingest.
- These tests protect strict typing decisions made for provider/model inputs.

Scenario map:
- Missing `EMBEDDING_PROFILE` is rejected.
- Unsupported provider values are rejected.
- Invalid `embed_model` format is rejected.
"""

from __future__ import annotations

import pytest

from app.data_classes import GlobalConfig


def _base_config_dict() -> dict:
    """Return a baseline valid config payload for `GlobalConfig` validation tests."""
    return {
        "data_path": "app/data/",
        "output_path": "app/output/",
        "report_id": "doc-1",
        "OLLAMA_URL": "http://localhost:11434",
        "QDRANT_URL": "http://localhost:6333",
        "POSTGRESQL_URL": "postgresql+psycopg://postgres:postgres@localhost:5432/report_assistant",
        "LLM_MODEL_CHATBOT": "gpt-4.1-mini",
        "LLM_MODEL_SUMMARIZER": "gpt-4.1-mini",
        "LLM_MODEL_EVAL": "gpt-4.1-mini",
        "QDRANT_DB_NAME_CHATBOT": "report_assistant_chatbot",
        "QDRANT_DB_NAME_YOY": "report_assistant_yoy",
        "chunk_strategy_chatbot": {
            "method": "sentence_metadata",
            "chunk_size": 3,
            "overlap": 1,
        },
        "chunk_strategy_yoy": {
            "method": "paragraph",
        },
        "top_k": 4,
    }


def test_global_config_requires_embedding_profile() -> None:
    """GlobalConfig validation fails when EMBEDDING_PROFILE is missing."""
    config_dict = _base_config_dict()

    with pytest.raises(Exception, match="EMBEDDING_PROFILE"):
        GlobalConfig.model_validate(config_dict)


def test_global_config_rejects_unknown_embedding_provider() -> None:
    """GlobalConfig validation fails when provider is not one of the allowed literals."""
    config_dict = _base_config_dict()
    config_dict["EMBEDDING_PROFILE"] = {
        "provider": "azure_openai",
        "embed_model": "text-embedding-3-small",
    }

    with pytest.raises(Exception, match="provider"):
        GlobalConfig.model_validate(config_dict)


def test_global_config_rejects_invalid_embedding_model() -> None:
    """GlobalConfig validation fails when model is empty or lacks alphanumeric content."""
    config_dict = _base_config_dict()
    config_dict["EMBEDDING_PROFILE"] = {
        "provider": "openai",
        "embed_model": "----",
    }

    with pytest.raises(Exception, match="model"):
        GlobalConfig.model_validate(config_dict)
