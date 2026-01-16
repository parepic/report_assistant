from typing import List, Optional

import requests

from report_assistant.data_classes import GlobalConfig, compute_strategy_hash


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
                               strategy_hash: Optional[str] = None,
                               k: int = 4) -> List[str]:
    """
    Embed the query and retrieve top-k chunk texts from Qdrant using REST API.
    Optionally filter by strategy_hash to only retrieve chunks created with a specific chunking strategy.
    """
    query_emb = get_embedding(query, ollama_url, embed_model)
    payload = {
        "vector": query_emb,
        "limit": k,
        "with_payload": ["text"]
    }
    
    # Add filter if strategy_hash is provided
    if strategy_hash:
        payload["filter"] = {
            "must": [
                {
                    "key": "strategy_hash",
                    "match": {
                        "value": strategy_hash
                    }
                }
            ]
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
                    llm_model: str,
                    top_k: int,
                    strategy_hash: Optional[str] = None,) -> str:
    """
    RAG:
      1) Retrieve relevant chunks from Qdrant
      2) Build context
      3) Ask LLM with that context
    """
    top_chunks = retrieve_top_k_from_qdrant(
        question, collection_name, qdrant_url, ollama_url, embed_model, 
        strategy_hash=strategy_hash, k=top_k
    )

    context = ""
    for i, chunk in enumerate(top_chunks):
        context += f"Chunk {i+1}:\n{chunk}\n\n"
    print("here is context: ", context)
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


def main(config: GlobalConfig) -> None:

    ollama_url = config.OLLAMA_URL
    qdrant_url = config.QDRANT_URL
    llm_model = config.LLM_MODEL
    embed_model = config.chunk_strategy.embed_model
    top_k = config.top_k

    # Compute strategy hash from global config
    strategy_hash = compute_strategy_hash(config.chunk_strategy)

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
        ans = answer_question(
            q, collection_name, qdrant_url, ollama_url, embed_model, llm_model,
            strategy_hash=strategy_hash, top_k=top_k
        )
        print("\nAssistant:", ans, "\n")


if __name__ == "__main__":
    main()
