from __future__ import annotations

import os
from typing import Any, Dict, List

import requests


def _get_api_base_url() -> str:
    return os.getenv("REPORT_ASSISTANT_API_URL", "http://localhost:8000").rstrip("/")


def browse_risk_factors(doc_id: str) -> List[Dict[str, Any]]:
    """
    Fetch risk factors for a single document from the backend API.

    Args:
        doc_id: Document identifier used by the API route as `document_id`.

    Returns:
        List of risk factor dictionaries with fields such as:
        `id`, `risk_factor`, `text`, and `idx`.
    """
    if not doc_id:
        raise ValueError("doc_id is required for browsing risk factors.")

    response = requests.get(
        f"{_get_api_base_url()}/risk_factors",
        params={"document_id": doc_id},
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected risk_factors response format.")
    return payload
