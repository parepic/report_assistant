from __future__ import annotations

import os
from typing import Dict, List

import requests


DocumentListItem = Dict[str, str | int]


def _get_api_base_url() -> str:
    return os.getenv("REPORT_ASSISTANT_API_URL", "http://localhost:8000").rstrip("/")


def load_report_entries() -> List[DocumentListItem]:
    response = requests.get(
        f"{_get_api_base_url()}/documents",
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected documents response format.")
    return data


def report_display_name(entry: DocumentListItem) -> str:
    company = str(entry.get("company", "")).strip() or "Unknown"
    year = entry.get("fiscal_year", "?")
    doc_id = entry.get("doc_id", "")
    return f"{company} — {year} · {doc_id}"


def report_metadata(entry: DocumentListItem) -> str:
    company = str(entry.get("company", "")).strip() or "Unknown"
    year = entry.get("fiscal_year", "?")
    return f"{company} · FY{year}"


def to_entry_dict(entry: DocumentListItem) -> dict:
    return dict(entry)


def from_entry_dict(data: dict) -> DocumentListItem:
    return dict(data)
