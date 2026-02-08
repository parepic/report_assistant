from typing import List, Optional
import os

import requests
from dotenv import load_dotenv

from app.data_classes import GlobalConfig, compute_strategy_hash
from app.clients.QdrantClientWrapper import QdrantClientWrapper
from app.clients.OpenAiClientWrapper import OpenAIClientWrapper

# Load environment variables from .env file
load_dotenv()

def answer_question(question: str,
                    qdrant_client: QdrantClientWrapper,
                    collection_name: str,
                    ollama_url: str,
                    embed_model: str,
                    openai_client: OpenAIClientWrapper,
                    llm_model: str,
                    top_k: int,
                    doc_id: Optional[str] = None,
                    strategy_hash: Optional[str] = None,) -> str:
    """
    RAG:
      1) Retrieve relevant chunks from Qdrant
      2) Build context
      3) Ask LLM with that context
    """
    top_chunks = qdrant_client.fetch_top_k_query(
        question, collection_name, ollama_url, embed_model, 
        strategy_hash=strategy_hash, doc_id=doc_id, k=top_k
    )

    context = ""
    for i, chunk in enumerate(top_chunks):
        context += f"Chunk {i+1}:\n{chunk["text"]}\n\n"
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

    return  openai_client.llm_generate(prompt)


def main(config: GlobalConfig) -> None:
    
    qdrant_client = QdrantClientWrapper(config)
    openai_client = OpenAIClientWrapper(api_key=os.getenv("OPENAI_API_KEY"), llm_model=config.LLM_MODEL_CHATBOT)
    
    ollama_url = config.OLLAMA_URL
    llm_model = config.LLM_MODEL_CHATBOT
    embed_model = config.chunk_strategy_chatbot.embed_model
    top_k = config.top_k

    # Compute strategy hash from global config
    strategy_hash = compute_strategy_hash(config.chunk_strategy_chatbot)
    collection_name = config.QDRANT_DB_NAME_CHATBOT

    print(f"\nUsing collection: {collection_name}")
    print("You can now ask questions! Type 'exit' to quit.\n")

    # Simple QA loop
    while True:
        q = input("You: ")
        if q.lower() in {"exit", "quit"}:
            break
        ans = answer_question(
            q, qdrant_client, collection_name, ollama_url, embed_model, openai_client, llm_model,
            strategy_hash=strategy_hash, top_k=top_k, doc_id=config.report_id
        )
        print("\nAssistant:", ans, "\n")
        

if __name__ == "__main__":
    main()
