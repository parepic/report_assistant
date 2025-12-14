import argparse
import json
from pathlib import Path
from typing import List

import yaml
from docx import Document


def load_global_config() -> dict:
    """
    Load the global configuration from properties/global.yaml
    """
    path = Path("global.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Global config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_document_entry(company_name: str,
                        index_path: str | Path) -> dict:
    """
    Load the metadata entry for a given company from documents.json.
    """
    index_path = Path(index_path)
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)  # list[dict]

    company_norm = company_name.strip().lower()
    matched = None

    for entry in entries:
        doc_id = entry.get("doc_id", "")
        company_field = entry.get("company", "")

        # Primary: company field match
        if company_field.lower() == company_norm:
            matched = entry
            break

        # Fallback: doc_id starts with company name
        if doc_id.lower().startswith(company_norm):
            matched = entry
            break

    if matched is None:
        raise ValueError(f"No document entry found for company '{company_name}'")

    return matched


def load_docx_text(path: str | Path) -> str:
    """
    Read all text from a .docx file into a single string.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def chunk_sequential(index_path: str | Path, output_path: str | Path, company: str, chunk_config: dict) -> List[str]:
    """
    Load the document for the company and chunk it using sequential strategy.
    """
    entry = load_document_entry(company, index_path)
    source_file = entry["source_file"]
    text = load_docx_text(source_file)
    chunk_size = chunk_config["chunk_size"]
    overlap = chunk_config["overlap"]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(chunk_text)
        start += chunk_size - overlap

    output_dir = Path(output_path) / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{company}.json"
    data = {
        "strategy": chunk_config,
        "chunks": chunks
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return chunks


def main():
    config = load_global_config()
    chunk_strategy_config = config["chunk_strategy"]
    strategy = chunk_strategy_config["method"]
    data_path = config["data_path"]
    output_path = config["output_path"]
    company = config["company"]

    index_path = Path(data_path) / "index" / "documents.json"
    
    # Determine the function by matching the strategy name
    func_name = f"chunk_{strategy}"
    if func_name not in globals():
        raise ValueError(f"Unknown strategy function: {func_name}")
    
    func = globals()[func_name]
    chunks = func(index_path, output_path, company, chunk_strategy_config)

    params_str = "_".join(f"{k}-{v}" for k, v in chunk_strategy_config.items() if k != "method")
    print(f"Chunked {len(chunks)} chunks for {company} using {strategy} strategy. Saved to {Path(output_path) / 'chunks' / f'{company}.json'}")


if __name__ == "__main__":
    main()
