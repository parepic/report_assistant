import json
from pathlib import Path
from typing import Optional
import yaml

from report_assistant.data_classes import GlobalConfig, DocumentEntry, ChunkFile



def load_global_config() -> "GlobalConfig":
    """
    Load and validate global configuration from global.yaml.
    """

    path = Path("global.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Global config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return GlobalConfig.model_validate(config_dict)


def load_document_entries(index_path: Path, config: Optional[GlobalConfig] = None) -> list["DocumentEntry"]:
    """
    Load a list of DocumentEntry objects from a JSON index file.
    """

    if not index_path.is_file():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    entries_raw = index_path.read_text(encoding="utf-8")
    entries_data = json.loads(entries_raw)
    if not isinstance(entries_data, list):
        raise ValueError(f"Index file {index_path} must contain a list of document entries.")

    if config is None:
        return [DocumentEntry.model_validate(entry) for entry in entries_data]

    return [DocumentEntry.from_index_entry(entry, config) for entry in entries_data]


def load_document_entry(report_id: str, index_path: Path, config: GlobalConfig) -> DocumentEntry:
    """
    Load the metadata entry for a given company or report from index.json.
    Returns a validated DocumentEntry.
    """
    entries = load_document_entries(index_path, config)
    report_norm = report_id.strip().lower()

    try:
        return next(
            entry
            for entry in entries
            if entry.doc_id.lower() == report_norm
        )
    except StopIteration as exc:
        raise ValueError(f"No document entry found for report '{report_id}' in index at {index_path}") from exc


def get_index_path(config: GlobalConfig) -> Path:
    """
    Resolve the index file location from the global config.
    """
    return Path(config.data_path) / "index.json"




def load_chunks(path: Path) -> "ChunkFile":
    """
    Load chunk data from a JSON file into a ChunkFile object.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    chunks_raw = path.read_text(encoding="utf-8")
    chunks_data = json.loads(chunks_raw)
    return ChunkFile.model_validate(chunks_data)
