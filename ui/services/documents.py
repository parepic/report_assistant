from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

from report_assistant.data_classes import DocumentEntry, GlobalConfig
from report_assistant.utils.load_utils import get_index_path, load_document_entries, load_global_config


@lru_cache(maxsize=1)
def get_config() -> GlobalConfig:
    return load_global_config()


def _is_under_reports_root(path: Path, reports_root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(reports_root.resolve())
    except AttributeError:
        try:
            path.resolve().relative_to(reports_root.resolve())
            return True
        except ValueError:
            return False


def load_report_entries() -> List[DocumentEntry]:
    config = get_config()
    index_path = get_index_path(config)
    entries = load_document_entries(index_path)
    reports_root = Path(config.data_path) / "reports"

    filtered: List[DocumentEntry] = []
    for entry in entries:
        source_path = Path(entry.source_file_path)
        if source_path.name.startswith("~$") or source_path.name.startswith("."):
            continue
        if not _is_under_reports_root(source_path, reports_root):
            continue
        filtered.append(entry)
    return filtered


def report_display_name(entry: DocumentEntry) -> str:
    filename = Path(entry.source_file_path).stem.replace("_", " ").replace("-", " ")
    company = entry.company.strip()
    year = entry.fiscal_year
    return f"{company} — {year} · {filename}"


def report_metadata(entry: DocumentEntry) -> str:
    return f"{entry.company} · FY{entry.fiscal_year}"


def to_entry_dict(entry: DocumentEntry) -> dict:
    return entry.model_dump()


def from_entry_dict(data: dict) -> DocumentEntry:
    return DocumentEntry.model_validate(data)
