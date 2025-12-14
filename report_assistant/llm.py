import json
from pathlib import Path
from typing import List

import numpy as np
import requests
import yaml


def load_global_config() -> dict:
    path = Path("global.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Global config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def llm_generate(prompt: str, ollama_url: str, llm_model: str) -> str:
    """
    Call the LLM via Ollama's generate API.
    """
    payload = {
        "model": llm_model,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(f"{ollama_url}/api/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["response"]


def retrieve_top_k_from_qdrant(query: str,
                               collection_name: str,
                               qdrant_url: str,
                               ollama_url: str,
                               embed_model: str,
                               k: int = 4) -> List[str]:
    """
    Embed the query and retrieve top-k chunk texts from Qdrant using REST API.
    """
    query_emb = get_embedding(query, ollama_url, embed_model)
    payload = {
        "vector": query_emb,
        "limit": k,
        "with_payload": ["text"]
    }
    resp = requests.post(f"{qdrant_url}/collections/{collection_name}/points/search", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return [hit["payload"]["text"] for hit in data["result"]]


def answer_question(question: str,
                    collection_name: str,
                    qdrant_url: str,
                    ollama_url: str,
                    embed_model: str,
                    llm_model: str) -> str:
    """
    RAG:
      1) Retrieve relevant chunks from Qdrant
      2) Build context
      3) Ask LLM with that context
    """
    top_chunks = retrieve_top_k_from_qdrant(question, collection_name, qdrant_url, ollama_url, embed_model, k=4)

    context = ""
    for i, chunk in enumerate(top_chunks):
        context += f"Chunk {i+1}:\n{chunk}\n\n"
    prompt = f"""
You are a helpful assistant answering questions about a company document.
Use ONLY the information in the context below. If the answer is not there,
say you don't know and do not make things up.

Context:
{context}

Question: {question}

Answer:
""".strip()

    return llm_generate(prompt, ollama_url, llm_model)


def main():
    config = load_global_config()
    ollama_url = config["OLLAMA_URL"]
    qdrant_url = config["QDRANT_URL"]
    llm_model = config["LLM_MODEL"]
    embed_model = config["chunk_strategy"]["embed_model"]

    # Ask for company name
    company_input = input("Enter company name (e.g. Microsoft): ").strip().lower()
    collection_name = f"company__{company_input}"

    print(f"\nUsing collection: {collection_name}")
    print("You can now ask questions! Type 'exit' to quit.\n")

    # Simple QA loop
    while True:
        q = input("You: ")
        if q.lower() in {"exit", "quit"}:
            break
        ans = answer_question(q, collection_name, qdrant_url, ollama_url, embed_model, llm_model)
        print("\nAssistant:", ans, "\n")


if __name__ == "__main__":
    main()
