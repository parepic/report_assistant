"""DeepEval RAG evaluation runner.

This script:
1) Loads a small question set; currently hard coded to be Microsoft FY24Q4 10-K Item 1A questions.
2) Runs the same RAG pipeline used in the app (retrieve -> prompt -> answer).
3) Builds DeepEval test cases with retrieval context attached.
4) Evaluates retrieval quality and answer quality with LLM-as-judge metrics.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, FaithfulnessMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from openai import OpenAI
from app.clients import QdrantClientWrapper

# Ensure repo root is on sys.path so sibling packages are importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.data_classes import compute_strategy_hash
from app.utils.load_utils import load_global_config


# Load questions from a JSON file.



# Load global config and compute strategy hash.
config = load_global_config()
strategy_hash = compute_strategy_hash(config.chunk_strategy_chatbot)
QUESTIONS_PATH = Path("app/data/questions/Microsoft/MSFT_FY24Q4_10K-item1a.json")
COLLECTION_NAME = config.QDRANT_DB_NAME_CHATBOT
raw = QUESTIONS_PATH.read_text(encoding="utf-8")
items = json.loads(raw)
items = items[:3]

ollama_url = config.OLLAMA_URL
qdrant_url = config.QDRANT_URL
llm_model = config.LLM_MODEL_CHATBOT
embed_model = config.chunk_strategy_chatbot.embed_model
company = "microsoft"
top_k = config.top_k
threshold = config.threshold

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


goldens = [
    Golden(input=item["question"], expected_output=item["expected_answer"])
    for item in items
]
dataset = EvaluationDataset(goldens=goldens)

print(f"Loaded {len(dataset.goldens)} goldens from {QUESTIONS_PATH}")

print("embed b ", embed_model)

# Generate test cases by retrieving context and getting model responses.
for item, golden in zip(items, dataset.goldens):
    retrieved_chunks = QdrantClientWrapper.topk_k_query(
        golden.input,
        COLLECTION_NAME,
        ollama_url,
        embed_model,
        embed_model,
        strategy_hash=strategy_hash,
        k=top_k,
    )
    # Build the same prompt format used in the app.
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"Chunk {i + 1}:\n{chunk}\n\n"
    prompt = (
        "You are a helpful assistant answering questions about a company document.\n"
        "Use ONLY the information in the context below. If the answer is not there,\n"
        "say you don't know and do not make things up.\n\n"
        f"Context:\n{context}\n"
        f"Question: {golden.input}\n\n"
        "Answer:\n"
    )
    print(f"Generating response for question: {golden.input}")
    response_text = llm_generate(prompt, client, llm_model)
    print("Generated response: ", response_text)




    reference_excerpt = item.get("reference_excerpt")
    expected_context = [reference_excerpt] if reference_excerpt else None

    test_case_kwargs = {
        "input": golden.input,
        "actual_output": response_text,
        "expected_output": golden.expected_output,
        "retrieval_context": retrieved_chunks,
        "context": expected_context,
    }
    test_case = LLMTestCase(
        **test_case_kwargs
    )
    dataset.add_test_case(test_case)






def summarize_metrics(test_cases: list[LLMTestCase], metrics: list) -> list[dict]:
    summaries = []
    for metric in metrics:
        scores = []
        passes = 0
        fails = 0
        for test_case in test_cases:
            metric.measure(test_case)
            score = getattr(metric, "score", None)
            success = getattr(metric, "success", None)
            if success is None and score is not None:
                threshold_value = getattr(metric, "threshold", None)
                if threshold_value is not None:
                    success = score >= threshold_value
            if success:
                passes += 1
            else:
                fails += 1
            if score is not None:
                scores.append(score)

        average_score = sum(scores) / len(scores) if scores else None
        summaries.append(
            {
                "metric": getattr(metric, "name", metric.__class__.__name__),
                "threshold": getattr(metric, "threshold", None),
                "passes": passes,
                "fails": fails,
                "average_score": average_score,
                "scores": scores,
            }
        )
    return summaries

# Evaluate retrieval metrics (context quality).
retrieval_metrics = [
    # Keep the smallest set first: one retrieval metric
    ContextualPrecisionMetric(threshold=threshold, model=llm_model),
    ContextualRecallMetric(threshold=threshold, model=llm_model),
]

# Evaluate response metrics (answer quality).
response_metrics = [
    # Keep the smallest set first: one response metric
    AnswerRelevancyMetric(threshold=threshold, model=llm_model),
    FaithfulnessMetric(threshold=threshold, model=llm_model),
]

print("Evaluating retrieval metrics:")
retrieval_summary = summarize_metrics(dataset.test_cases, retrieval_metrics)
print("Evaluating response metrics:")
response_summary = summarize_metrics(dataset.test_cases, response_metrics)

results = {
    "run_at": datetime.now().isoformat(timespec="seconds"),
    "questions_path": str(QUESTIONS_PATH),
    "collection_name": COLLECTION_NAME,
    "model": llm_model,
    "top_k": top_k,
    "strategy_hash": strategy_hash,
    "retrieval_metrics": retrieval_summary,
    "response_metrics": response_summary,
}

results_path = Path("deepeval_eval") / "runs" / f"deepeval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
results_path.parent.mkdir(parents=True, exist_ok=True)
results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"Wrote results to {results_path}")
