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
from app.clients.QdrantClientWrapper import QdrantClientWrapper
from app.clients.OpenAiClientWrapper import OpenAIClientWrapper
from app.services.chatbot import main as chatbot_main

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
items = items[:10]

ollama_url = config.OLLAMA_URL
qdrant_url = config.QDRANT_URL
llm_model = config.LLM_MODEL_CHATBOT
embed_model = config.chunk_strategy_chatbot.embed_model
company = "microsoft"
top_k = config.top_k
threshold = config.threshold

client = OpenAIClientWrapper(api_key=os.getenv("OPENAI_API_KEY"), llm_model=llm_model)

goldens = [
    Golden(input=item["question"], expected_output=item["expected_answer"])
    for item in items
]


dataset = EvaluationDataset(goldens=goldens)

print(f"Loaded {len(dataset.goldens)} goldens from {QUESTIONS_PATH}")

print("embed b ", embed_model)

# Generate test cases by retrieving context and getting model responses.
for item, golden in zip(items, dataset.goldens):
    response = chatbot_main(
        config=config,
        prompt=golden.input,
        doc_id="microsoft_10-k-item1a-2024",
        qdrant_client=QdrantClientWrapper(config),
        openai_client=client,
        return_chunks=True,
    )
    print("Generated response: ", response)


    reference_excerpt = item.get("reference_excerpt")
    expected_context = [reference_excerpt] if reference_excerpt else None
    chunks = [elem["payload"]["text"] for elem in response["retrieved_chunks"]]
    test_case_kwargs = {
        "input": golden.input,
        "actual_output": response["llm_response"],
        "expected_output": golden.expected_output,
        "retrieval_context": chunks,
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
        reasons = []
        i = 0
        for test_case in test_cases:
            metric.measure(test_case)
            # print("iteration ", i, test_case, "\n\n\n")
            i+=1
            score = getattr(metric, "score", None)
            success = getattr(metric, "success", None)
            reason = getattr(metric, "reason", None)
            if reason is not None:
                reasons.append(reason)
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
                "reasons": reasons,
            }
        )
    return summaries

# # Evaluate retrieval metrics (context quality).
retrieval_metrics = [
    # Keep the smallest set first: one retrieval metric
    ContextualPrecisionMetric(threshold=threshold, model=llm_model, include_reason=True),
    ContextualRecallMetric(threshold=threshold, model=llm_model, include_reason=True),
]

# Evaluate response metrics (answer quality).
response_metrics = [
    # Keep the smallest set first: one response metric
    AnswerRelevancyMetric(threshold=threshold, model=llm_model, verbose_mode=True),
    FaithfulnessMetric(threshold=threshold, model=llm_model, verbose_mode=True),
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

results_path = Path("app/deepeval_eval") / "runs" / f"deepeval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
results_path.parent.mkdir(parents=True, exist_ok=True)
results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"Wrote results to {results_path}")
