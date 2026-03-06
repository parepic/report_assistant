#!/usr/bin/env python3
"""
Batch ingestion runner for all documents listed in index.json.

This script reads the document registry, validates that each entry points to a
real source file, and runs the ingestion stages (DB, chatbot, YoY, factors) in
sequence for each eligible document. It is designed for non-interactive
bootstrap runs where existing records should be skipped rather than prompting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from app.data_classes import GlobalConfig
from app.ingestion.chunking.chunk import main as chunk_main
from app.ingestion.preprocess import main as preprocess_main
from app.ingestion.save_postgresql import main as save_sql_main
from app.ingestion.save_postgresql_factors import main as save_sql_factors_main
from app.ingestion.save_qdrant import main as save_qdrant_main
from app.utils.load_utils import get_index_path, load_global_config


def load_index_entries(index_path: Path) -> List[Dict]:
    """
    Load raw index entries from a JSON file.

    Args:
        index_path: Absolute or repo-relative path to the index.json file.

    Returns:
        A list of raw dictionary entries in the order provided by the index file.

    Raises:
        FileNotFoundError: If index_path does not exist.
        ValueError: If index contents are not a list.
    """
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    raw = index_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Index file {index_path} must contain a list of entries.")
    return data


def build_config_for_report(base_config: GlobalConfig, report_id: str) -> GlobalConfig:
    """
    Create a per-report config object without mutating the original.

    Args:
        base_config: The global configuration loaded from app/global.yaml.
        report_id: The document ID to inject into the config for ingestion.

    Returns:
        A new GlobalConfig with report_id overridden.
    """
    return base_config.model_copy(update={"report_id": report_id})


def select_report_ids(
    entries: Iterable[Dict],
    only_report_ids: Optional[List[str]],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Select document IDs from the index, optionally filtering by a provided list.

    Args:
        entries: Raw index entry dictionaries.
        only_report_ids: Optional list of report IDs to include. If None, all
            entries are considered.

    Returns:
        A tuple of (selected_ids, duplicate_ids, missing_in_index_ids).
    """
    selected: List[str] = []
    duplicates: List[str] = []
    missing_in_index: List[str] = []

    by_lower: Dict[str, str] = {}
    for entry in entries:
        doc_id = entry.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            continue
        key = doc_id.strip().lower()
        if key in by_lower:
            duplicates.append(doc_id)
            continue
        by_lower[key] = doc_id

    if only_report_ids:
        for requested in only_report_ids:
            key = requested.strip().lower()
            if key in by_lower:
                selected.append(by_lower[key])
            else:
                missing_in_index.append(requested)
    else:
        selected = list(by_lower.values())

    return selected, duplicates, missing_in_index


def filter_existing_sources(entries: Iterable[Dict], report_ids: Iterable[str]) -> Tuple[List[str], List[str]]:
    """
    Filter report IDs to those whose source_file_path exists on disk.

    Args:
        entries: Raw index entry dictionaries.
        report_ids: Report IDs to validate against index entries.

    Returns:
        A tuple of (eligible_ids, missing_source_ids).
    """
    entry_by_id = {entry.get("doc_id"): entry for entry in entries}
    eligible: List[str] = []
    missing: List[str] = []

    for report_id in report_ids:
        entry = entry_by_id.get(report_id)
        if not entry:
            missing.append(report_id)
            continue
        source_path = Path(entry.get("source_file_path", "")).expanduser()
        if source_path.is_file():
            eligible.append(report_id)
        else:
            missing.append(report_id)

    return eligible, missing


def run_full_ingestion(config: GlobalConfig, on_existing: str) -> None:
    """
    Execute all ingestion stages for a single report.

    Order:
    1) Preprocess and save to PostgreSQL
    2) Chunk by risk factor + save risk factors to PostgreSQL
    3) Chunk + embed for chatbot collection
    4) Chunk + embed for YoY collection

    Args:
        config: Report-specific configuration object.
        on_existing: Duplicate handling policy ("prompt", "skip", "delete").
    """
    print(f"\n=== Ingesting: {config.report_id} ===")
    print("Starting preprocessing...")
    preprocess_main(config)
    print("Preprocessing completed. Saving to PostgreSQL...")
    save_sql_main(config, on_existing=on_existing)
    print("Starting chunking (factor)...")
    chunk_main(config, mode="factor")
    print("Chunking completed. Saving risk factors to PostgreSQL...")
    save_sql_factors_main(config, on_existing=on_existing)
    print("Starting chunking (chatbot)...")
    chunk_main(config, mode="chatbot")
    print("Chunking completed. Starting embedding (chatbot)...")
    save_qdrant_main(config, mode="chatbot", on_existing=on_existing)
    print("Starting chunking (yoy)...")
    chunk_main(config, mode="yoy")
    print("Chunking completed. Starting embedding (yoy)...")
    save_qdrant_main(config, mode="yoy", on_existing=on_existing)
    print(f"Completed ingestion for {config.report_id}.")


def main() -> None:
    """
    CLI entrypoint for batch ingestion from index.json.

    This entrypoint supports optional filtering by report IDs and defaults to
    non-interactive behavior that skips existing records and missing sources.
    """
    parser = argparse.ArgumentParser(
        description="Batch ingestion for all documents in index.json."
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Override index.json path. Defaults to data_path/index.json from global.yaml.",
    )
    parser.add_argument(
        "--report-ids",
        nargs="+",
        default=None,
        help="Optional list of specific report IDs to ingest.",
    )
    parser.add_argument(
        "--on-existing",
        choices=["prompt", "skip", "delete"],
        default="skip",
        help="How to handle existing documents/vectors for a report.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch run on first error instead of continuing.",
    )

    args = parser.parse_args()

    base_config = load_global_config()
    index_path = Path(args.index_path) if args.index_path else get_index_path(base_config)
    entries = load_index_entries(index_path)

    selected_ids, duplicates, missing_in_index = select_report_ids(entries, args.report_ids)
    eligible_ids, missing_sources = filter_existing_sources(entries, selected_ids)

    if duplicates:
        print(f"Skipping {len(duplicates)} duplicate doc_id entries in index.json.")
    if missing_in_index:
        print(f"{len(missing_in_index)} requested report_ids not found in index.json: {missing_in_index}")
    if missing_sources:
        print(f"Skipping {len(missing_sources)} entries with missing source files: {missing_sources}")

    print(f"\nBatch ingestion starting for {len(eligible_ids)} documents.")

    failures: List[str] = []
    for report_id in eligible_ids:
        config = build_config_for_report(base_config, report_id)
        try:
            run_full_ingestion(config, on_existing=args.on_existing)
        except Exception as exc:
            print(f"Error ingesting {report_id}: {exc}")
            failures.append(report_id)
            if args.stop_on_error:
                break

    print("\nBatch ingestion complete.")
    if failures:
        print(f"Failures ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
