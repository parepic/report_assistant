from __future__ import annotations

from typing import Dict, Any, List
import re
from app.data_classes import GlobalConfig, compute_strategy_hash
from app.clients.QdrantClientWrapper import QdrantClientWrapper
from app.clients.OpenAiClientWrapper import OpenAIClientWrapper
from app.ingestion.save_qdrant import derive_collection_name
from app.prompts import prompt_chatbot


def process_citations(llm_response: str, top_chunks: list) -> tuple[str, List[Dict[str, str]]]:
    """
    Separates the LLM prose from source identifiers and maps them to metadata,
    cleaning the source text by removing everything before the last '|'.
    """
    marker = "Sources Used:"
    
    # 1. Split text and sources
    if marker in llm_response:
        parts = llm_response.split(marker)
        clean_body = parts[0].strip()
        raw_source_text = parts[1]
    else:
        # Fallback if the LLM forgot the marker
        return llm_response.strip(), []

    # 2. Extract indices
    source_indices = [int(d) for d in re.findall(r"\d+", raw_source_text)]
    
    # 3. Build structured citation list
    structured_citations = []
    seen_indices = set()
    
    for idx in source_indices:
        if 0 < idx <= len(top_chunks) and idx not in seen_indices:
            payload = top_chunks[idx - 1].get("payload", {})
            original_text = payload.get("text", "")

            # --- Cleaning Logic: Keep only text after the last '|' ---
            last_pipe_idx = original_text.rfind('|')
            if last_pipe_idx != -1:
                # Slice from after the pipe and strip whitespace
                cleaned_text = original_text[last_pipe_idx + 1:].strip()
            else:
                cleaned_text = original_text.strip()
            
            structured_citations.append({
                "text": cleaned_text, # Using the cleaned version here
                "risk_factor": payload.get("risk_factor", "Detailed Disclosure")
            })
            seen_indices.add(idx)
    
    return clean_body, structured_citations


def main(
    config: GlobalConfig,
    prompt: str,
    doc_id: str,
    qdrant_client: QdrantClientWrapper,
    openai_client: OpenAIClientWrapper,
    return_chunks: bool = False,
) -> Dict[str, Any]:
    
    strategy_hash = compute_strategy_hash(config.chunk_strategy_chatbot)
    
    # 1. Retrieval
    collection_name = derive_collection_name(
        config.QDRANT_DB_NAME_CHATBOT,
        config.EMBEDDING_PROFILE,
    )
    top_chunks = qdrant_client.fetch_top_k_query(
        prompt, 
        collection_name, 
        strategy_hash=strategy_hash, 
        doc_id=doc_id, 
        k=config.top_k
    )
    
    if not top_chunks:
        return {"error": "No relevant chunks found for the selected document."}

    # 2. Context Building
    context = ""
    for i, chunk in enumerate(top_chunks):
        context += f"Source {i+1}:\n{chunk['payload']['text']}\n\n"
    
    company = top_chunks[0]["payload"]["company"]
    
    # 3. LLM Generation
    prompt_template = prompt_chatbot.format(
        context=context, 
        question=prompt, 
        company_name=company
    )
    print(prompt_template)
    raw_response = openai_client.llm_generate(prompt_template)
    print(raw_response)
    # 4. Post-processing into Clean Prose + Structured Citations
    clean_response, citations = process_citations(raw_response, top_chunks)

    # 5. Result Assembly
    result = {
        "llm_response": clean_response,
        "citations": citations,
    }

    if return_chunks:
        result["retrieved_chunks"] = top_chunks
        
    return result
