from __future__ import annotations

from typing import Dict, Any

from app.data_classes import GlobalConfig, compute_strategy_hash
from app.clients.QdrantClientWrapper import QdrantClientWrapper
from app.clients.OpenAiClientWrapper import OpenAIClientWrapper
from app.prompts import prompt_chatbot

def main(
    config: GlobalConfig,
    prompt: str,
    doc_id: str,
    qdrant_client: QdrantClientWrapper,
    openai_client: OpenAIClientWrapper,
) -> Dict[str, Any]:
    strategy_hash = compute_strategy_hash(config.chunk_strategy_chatbot)
    top_chunks = qdrant_client.fetch_top_k_query(
        prompt, config.QDRANT_DB_NAME_CHATBOT, config.OLLAMA_URL, config.chunk_strategy_chatbot.embed_model, 
        strategy_hash=strategy_hash, doc_id=doc_id, k=config.top_k
    )
    context = ""
    for i, chunk in enumerate(top_chunks):
        context += f"Chunk {i+1}:\n{chunk["payload"]["text"]}\n\n"
    print("here is context: ", context)
    prompt_template = prompt_chatbot.format(context=context, question=prompt)
    response = openai_client.llm_generate(prompt_template)

    return {
        "llm_response": response,
    }

