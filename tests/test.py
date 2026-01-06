import json
from typing import List, Dict, Any

from report_assistant.data_classes import GlobalConfig, compute_strategy_hash
from report_assistant.utils.load_utils import get_index_path, load_document_entry
from report_assistant.utils.utils import slugify_name
from report_assistant.llm import retrieve_top_k_from_qdrant, llm_generate


def load_questions(questions_file_path) -> List[Dict[str, Any]]:
    """Load questions from a JSON file."""
    with open(questions_file_path, 'r') as f:
        return json.load(f)


def filter_questions_by_type(questions: List[Dict[str, Any]], question_types: List[str] = None) -> List[Dict[str, Any]]:
    """Filter questions by type. If question_types is None, return all questions."""
    if question_types is None:
        return questions
    return [q for q in questions if q.get("type") in question_types]


def run_test_questions(config: GlobalConfig) -> None:
    """
    Run test questions from the questions file.
    For each question, retrieve chunks and generate an answer,
    then print the results along with the expected answer and location.
    """
    # Load document entry
    index_path = get_index_path(config)
    entry = load_document_entry(config.report_id, index_path, config)
    # Validate questions file exists
    question_path = entry.questions_file_path 
    if question_path is None or not question_path.is_file():
        raise ValueError(f"No questions_file_path found for report '{config.report_id}' in index at {index_path}")
    
    # Load and filter questions
    questions = load_questions(question_path)
    filtered_questions = filter_questions_by_type(questions, config.question_types)
    
    print(f"Loaded {len(questions)} questions from {question_path}")
    if config.question_types:
        print(f"Filtering by types: {config.question_types}")
        print(f"Processing {len(filtered_questions)} questions after filtering\n")
    else:
        print(f"Processing all {len(filtered_questions)} questions\n")

    # Setup for retrieval
    collection_name = slugify_name(entry.company)
    ollama_url = config.OLLAMA_URL
    qdrant_url = config.QDRANT_URL
    embed_model = config.chunk_strategy.embed_model
    llm_model = config.LLM_MODEL
    top_k = config.top_k
    strategy_hash = compute_strategy_hash(config.chunk_strategy)
    
    print(f"Collection: {collection_name}")
    print(f"Strategy hash: {config.chunk_strategy}")
    print(f"Top-k: {top_k}")
    print("=" * 80)
    print()
    
    # Process each question
    for idx, question_data in enumerate(filtered_questions, 1):
        question_text = question_data.get("question", "")
        expected_answer = question_data.get("answer", "")
        question_type = question_data.get("type", "")
        location = question_data.get("location", "")
        
        print(f"\n{'=' * 80}")
        print(f"Question {idx}/{len(filtered_questions)}")
        print(f"{'=' * 80}")
        print(f"Type: {question_type}")
        print(f"Location: {location}")
        print(f"\nQuestion: {question_text}")
        print(f"\nExpected Answer: {expected_answer}")
        
        # Retrieve chunks
        try:
            retrieved_chunks = retrieve_top_k_from_qdrant(
                question_text,
                collection_name,
                qdrant_url,
                ollama_url,
                embed_model,
                strategy_hash=strategy_hash,
                k=top_k
            )
            
            # Print chunks if configured
            if config.print_chunks:
                print(f"\n--- Retrieved Chunks ({len(retrieved_chunks)}) ---")
                for i, chunk in enumerate(retrieved_chunks, 1):
                    print(f"\nChunk {i}:")
                    print(chunk)
                print()
            
            # Build context for LLM
            context = ""
            for i, chunk in enumerate(retrieved_chunks):
                context += f"Chunk {i+1}:\n{chunk}\n\n"
            
            # Generate answer
            prompt = f"""You are a helpful assistant answering questions about a company document.
Use ONLY the information in the context below. If the answer is not there,
say you don't know and do not make things up.

Context:
{context}

Question: {question_text}

Answer:
""".strip()
            
            llm_answer = llm_generate(prompt, ollama_url, llm_model)
            
            print(f"--- LLM Answer ---")
            print(llm_answer)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process question: {e}")
    
    print(f"\n{'=' * 80}")
    print(f"Completed processing {len(filtered_questions)} questions")
    print(f"{'=' * 80}")


def main(config: GlobalConfig) -> None:
    run_test_questions(config)


if __name__ == "__main__":
    main()