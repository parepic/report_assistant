# run.py
#
# Simple RAG pipeline using:
# - documents.json index
# - .docx source files
# - Ollama for embeddings + LLM
#
# Prereqs:
#   pip install python-docx numpy requests
#   ollama pull phi
#   ollama pull nomic-embed-text
#
# Folder layout (example):
#   data/index/documents.json
#   data/reports/...
#   (see your screenshot)

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
from docx import Document


# ----------------- CONFIG -----------------

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"  # embedding model in Ollama
LLM_MODEL = "phi"                 # small LLM in Ollama (change if you want)

INDEX_FILE = "data/index/documents.json"

# Chunking config
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 200    # characters of overlap

DOC_ID: str | None = None  # global doc_id


# ----------------- INDEX / METADATA -----------------

def load_document_entry(company_name: str,
                        index_path: str = INDEX_FILE) -> dict:
    """
    Load the metadata entry for a given company from documents.json.
    Sets the global DOC_ID to the matched doc's doc_id.
    """
    global DOC_ID

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

    DOC_ID = matched["doc_id"]
    return matched


# ----------------- DOCX LOADING + CHUNKING -----------------

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


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split a long string into overlapping character-based chunks.
    """
    chunks = []
    start = 0
    while start < 3:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ----------------- OLLAMA: EMBEDDINGS + LLM -----------------

def get_embedding(text: str) -> List[float]:
    """
    Get a single embedding vector from Ollama.
    Tries /api/embed (new) then /api/embeddings (older).
    """
    # New API: POST /api/embed  { "model": "...", "input": "..." }
    try:
        payload = {"model": EMBED_MODEL, "input": text}
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json=payload, timeout=60)
        if resp.status_code != 404:
            resp.raise_for_status()
            data = resp.json()
            # "embeddings" is a list of vectors; for a single string take index 0
            return data["embeddings"][0]
    except requests.RequestException:
        pass

    # Older API: POST /api/embeddings  { "model": "...", "prompt": "..." }
    payload = {"model": EMBED_MODEL, "prompt": text}
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]



def llm_generate(prompt: str) -> str:
    """
    Call the LLM via Ollama's generate API.
    """
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["response"]


# ----------------- RAG HELPERS -----------------

def build_corpus_vectors(chunks: List[str]) -> List[np.ndarray]:
    """
    Build an embedding vector for each text chunk.
    """
    vectors: List[np.ndarray] = []
    print(f"Building embeddings for {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        emb = np.array(get_embedding(chunk), dtype="float32")
        vectors.append(emb)
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks")
    return vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_top_k(query: str,
                   chunks: List[str],
                   vectors: List[np.ndarray],
                   k: int = 4) -> List[Tuple[int, str]]:
    """
    Return top-k most relevant chunks (index, text) for the query.
    """
    query_emb = np.array(get_embedding(query), dtype="float32")

    scores: List[Tuple[float, int]] = []
    for idx, vec in enumerate(vectors):
        sim = cosine_similarity(query_emb, vec)
        scores.append((sim, idx))

    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:k]

    return [(idx, chunks[idx]) for _, idx in top]


def answer_question(question: str,
                    chunks: List[str],
                    vectors: List[np.ndarray]) -> str:
    """
    RAG:
      1) Retrieve relevant chunks
      2) Build context
      3) Ask LLM with that context
    """
    top_chunks = retrieve_top_k(question, chunks, vectors, k=4)

    context = ""
    for i, (idx, chunk) in enumerate(top_chunks):
        context += f"Chunk {idx}:\n{chunk}\n\n"

    prompt = f"""
You are a helpful assistant answering questions about a company document.
Use ONLY the information in the context below. If the answer is not there,
say you don't know and do not make things up.

doc_id: {DOC_ID}

Context:
{context}

Question: {question}

Answer:
""".strip()

    return llm_generate(prompt)


# ----------------- MAIN FLOW -----------------

def main():
    # 1. Ask for company name
    company = input("Enter company name (e.g. Microsoft): ").strip()

    # 2. Load index entry & docx
    entry = load_document_entry(company)
    source_file = entry["source_file"]

    print(f"\nUsing doc_id = {DOC_ID}")
    print(f"Source file = {source_file}\n")

    text = load_docx_text(source_file)
    print(f"Loaded document with {len(text)} characters.\n")

    # 3. Chunk + embed
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.\n")

    vectors = build_corpus_vectors(chunks)
    print("\nEmbeddings ready. You can now ask questions! Type 'exit' to quit.\n")

    # 4. Simple QA loop
    while True:
        q = input("You: ")
        if q.lower() in {"exit", "quit"}:
            break
        ans = answer_question(q, chunks, vectors)
        print("\nAssistant:", ans, "\n")


if __name__ == "__main__":
    main()
