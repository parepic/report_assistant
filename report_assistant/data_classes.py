"""
Typed models used across the pipeline.

These wrap the JSON/YAML structures so callers can rely on validation
and attribute access rather than raw dict lookups.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, computed_field, field_validator



class GlobalConfig(BaseModel):
    # Where to find input data and store output
    data_path: str
    output_path: str

    # Currently, we select a report or company to embed the files
    report_id: str

    OLLAMA_URL: HttpUrl | str
    QDRANT_URL: HttpUrl | str
    LLM_MODEL: str

    chunk_strategy: ChunkStrategy




class ChunkStrategy(BaseModel):
    embed_model: str
    method: str
    chunk_size: int
    overlap: int = 0

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v

    @field_validator("overlap")
    @classmethod
    def overlap_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("overlap cannot be negative")
        return v






class ChunkFile(BaseModel):
    version: str = Field(default="1")
    strategy: ChunkStrategy
    chunks: List[str]

    # In case we want to use the hash of the chunkfile so that we can avoid redundant processing
    @computed_field
    @property
    def strategy_hash(self) -> str:
        strategy_payload = self.strategy.model_dump(mode="python")
        serialized = json.dumps(
            {
                "strategy": strategy_payload,
                "chunks": self.chunks,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()







class DocumentEntry(BaseModel):
    doc_id: str
    company: str
    fiscal_year: int
    source_file_path: Path
    questions_file_path: Optional[Path] = None
    text_dir: Optional[Path] = None
    chunks_dir: Optional[Path] = None

    @computed_field
    @property
    def source_format(self) -> str:
        """
        File format inferred from the source filename extension (e.g., 'docx').
        """
        suffix = self.source_file_path.suffix
        if not suffix:
            raise ValueError("source_file has no extension; cannot infer format")
        return suffix.lstrip(".").lower()

    @field_validator("source_file_path")
    @classmethod
    def must_exist(cls, v):  # simple example
        if not Path(v).is_file():
            raise ValueError(f"source_file not found: {v}")
        return v

    @classmethod
    def from_index_entry(cls, data: dict, config: GlobalConfig) -> "DocumentEntry":
        """
        Construct a DocumentEntry from raw index data and attach processed paths.
        """
        entry = cls.model_validate(data)
        attach_processed_paths(config, entry)
        return entry


def attach_processed_paths(config: GlobalConfig, entry: DocumentEntry) -> DocumentEntry:
    """
    Derive processed paths from the global config base + normalized company/doc_id
    and assign them directly to the DocumentEntry.
    """
    from report_assistant.utils.utils import slugify_name
    
    company_slug = slugify_name(entry.company)
    processed_dir = Path(config.output_path) / company_slug
    text_dir = processed_dir / "text"
    chunks_dir = processed_dir / "chunks"

    entry.text_dir = text_dir
    entry.chunks_dir = chunks_dir

    entry.text_dir.mkdir(parents=True, exist_ok=True)
    entry.chunks_dir.mkdir(parents=True, exist_ok=True)
    return entry
