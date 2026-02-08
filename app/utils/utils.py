import re
from typing import Any, Dict, List, Tuple

import numpy as np
import requests


def get_heading_level_pattern(text: str) -> Tuple[re.Pattern, re.Pattern]:
    """
    Determine section and subsection heading patterns based on the first heading level found.
    
    If first heading is single '#' (level 1):
      - Section is '#'
      - Subsection is '##'
    
    If first heading is '##' (level 2):
      - Section is '##'
      - Subsection is '###'
    
    Returns tuple of (section_pattern, subsection_pattern)
    """
    # Find first line with heading markers
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('#'):
            # Count leading hashes
            hash_count = len(stripped) - len(stripped.lstrip('#'))
            
            if hash_count == 1:
                # First heading is single #, so section is #, subsection is ##
                section_pattern = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
                subsection_pattern = re.compile(r"^\s*##\s+(.+?)\s*$", re.MULTILINE)
            else:
                # First heading is ##, so section is ##, subsection is ###
                section_pattern = re.compile(r"^\s*##\s+(.+?)\s*$", re.MULTILINE)
                subsection_pattern = re.compile(r"^\s*###\s+(.+?)\s*$", re.MULTILINE)
            
            return section_pattern, subsection_pattern
    
    # Default to # for section and ## for subsection if no headings found
    section_pattern = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
    subsection_pattern = re.compile(r"^\s*##\s+(.+?)\s*$", re.MULTILINE)
    return section_pattern, subsection_pattern


def slugify_name(company: str) -> str:
    """
    Qdrant collection names should be simple. This keeps letters, digits, _ and -.
    """
    s = company.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    if not s:
        raise ValueError("Company name became empty after sanitization.")
    return f"company__{s}"


def get_embedding(text: str, ollama_url: str, embed_model: str) -> List[float]:
    """
    Get a single embedding vector from Ollama.
    Tries /api/embed (new) then /api/embeddings (older).
    """
    try:
        payload = {"model": embed_model, "input": text}
        resp = requests.post(f"{ollama_url}/api/embed", json=payload, timeout=60)
        if resp.status_code != 404:
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"][0]
    except requests.RequestException:
        pass

    payload = {"model": embed_model, "prompt": text}
    resp = requests.post(f"{ollama_url}/api/embeddings", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]

def embed_chunks(chunks: List[Dict[str, Any]], ollama_url: str, embed_model: str) -> List[np.ndarray]:
  """Create embeddings for a list of chunk dicts (extracting 'text' field from each)."""
  vectors: List[np.ndarray] = []
  print(f"Creating embeddings for {len(chunks)} chunks...")
  for i, chunk in enumerate(chunks):
    chunk_text = chunk.get("text", "")
    if not chunk_text:
      raise ValueError(f"Chunk {i} has no 'text' field or it is empty")
    emb = np.array(get_embedding(chunk_text, ollama_url, embed_model), dtype="float32")
    vectors.append(emb)
    if (i + 1) % 10 == 0 or i == len(chunks) - 1:
      print(f"  Embedded {i + 1}/{len(chunks)} chunks")
  return vectors

