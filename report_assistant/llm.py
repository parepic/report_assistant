from typing import List, Optional
import os

import requests
from openai import OpenAI
from dotenv import load_dotenv

from report_assistant.data_classes import GlobalConfig, compute_strategy_hash
from report_assistant.utils.qdrant_utils import slugify_name

# Load environment variables from .env file
load_dotenv()





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


def llm_generate(prompt: str, client: OpenAI, llm_model: str) -> str:
    """
    Call the LLM via OpenAI API.
    """
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def retrieve_top_k_from_qdrant(query: str,
                               collection_name: str,
                               company: str,
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
    
    # Always filter by company, optionally by strategy_hash
    must_filters = [
        {
            "key": "company",
            "match": {
                "value": company
            }
        }
    ]
    
    # Add strategy_hash filter if provided
    if strategy_hash:
        must_filters.append({
            "key": "strategy_hash",
            "match": {
                "value": strategy_hash
            }
        })
    
    payload["filter"] = {"must": must_filters}
    
    resp = requests.post(f"{qdrant_url}/collections/{collection_name}/points/search", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return [hit["payload"]["text"] for hit in data["result"]]


def answer_question(question: str,
                    collection_name: str,
                    company: str,
                    qdrant_url: str,
                    ollama_url: str,
                    embed_model: str,
                    client: OpenAI,
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
        question, collection_name, company, qdrant_url, ollama_url, embed_model, 
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

    return llm_generate(prompt, client, llm_model)


def main(config: GlobalConfig) -> None:

    # Initialize OpenAI client with API key from .env
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    ollama_url = config.OLLAMA_URL
    qdrant_url = config.QDRANT_URL
    llm_model = config.LLM_MODEL
    embed_model = config.chunk_strategy_chatbot.embed_model
    top_k = config.top_k

    # Compute strategy hash from global config
    strategy_hash = compute_strategy_hash(config.chunk_strategy_chatbot)

    # Ask for company name
    company_input = input("Enter company name (e.g. Microsoft): ").strip().lower()
    collection_name = config.QDRANT_DB_NAME_CHATBOT

    print(f"\nUsing collection: {collection_name}")
    print("You can now ask questions! Type 'exit' to quit.\n")

    # Simple QA loop
    while True:
        q = input("You: ")
        if q.lower() in {"exit", "quit"}:
            break
        ans = answer_question(
            q, collection_name, company_input, qdrant_url, ollama_url, embed_model, client, llm_model,
            strategy_hash=strategy_hash, top_k=top_k
        )
        print("\nAssistant:", ans, "\n")
        

if __name__ == "__main__":
    main()
