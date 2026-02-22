from __future__ import annotations

import os
from typing import Any, Dict

import requests


def _get_api_base_url() -> str:
    return os.getenv("REPORT_ASSISTANT_API_URL", "http://localhost:8000").rstrip("/")


def answer_for_entry(question: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = entry.get("doc_id")
    if not doc_id:
        raise ValueError("Selected document is missing doc_id.")

    response = requests.post(
        f"{_get_api_base_url()}/chatbot",
        json={"doc_id": doc_id, "prompt": question},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        return payload
    return {"llm_response": str(payload), "citations": []}
