from __future__ import annotations

import os
from typing import Any, Dict

import requests


def _get_api_base_url() -> str:
    return os.getenv("REPORT_ASSISTANT_API_URL", "http://localhost:8000").rstrip("/")


def compare_to_last_year(doc_id: str) -> Dict[str, Any]:
    if not doc_id:
        raise ValueError("doc_id is required for comparison.")
    response = requests.post(
        f"{_get_api_base_url()}/comparison",
        json={"doc_id": doc_id},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unexpected comparison response format.")
